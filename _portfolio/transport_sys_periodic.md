---
title: 'Periodic transport system in FEniCSx'
excerpt: "Solving a transport/wave system on a periodic mesh in FEniCSX"

collection: notes
---



<img src="https://markusrenoldner.github.io/files/transport_sys_periodic_animation_u.gif" alt="demo" />



We study the problem to find two time-dependent functions $u,\phi$ on a bounded domain $\Omega=(0,1)^2 \subset \mathbb{R}^2$, that satisfy 

$$\begin{cases}
\displaystyle \partial_{t} u + b\cdot \nabla \phi=f, \\
\partial_{t} \phi + b\cdot \nabla u=g,
\end{cases}
$$

where $b$ is a vectorfield with $\nabla\cdot b = 0$, that dictates the advection direction, and $f,g$ are given functions.

Smooth solutions of the above system also satisfy a decoupled version of the above system, 

$$\partial_{tt} u + b\cdot\nabla (b\cdot\nabla u) = h,$$

for a suitable function $h$. This last problem resembles the linear wave equation, hence the name.

The aim of this tutorial is to show how to use periodic boundary conditions. We will choose the following setting:

$$\begin{align*}
\phi|_{\Gamma_D} &=0 , \\
\phi|_{\Gamma_l} &= \phi|_{\Gamma_r}, \\
u|_{\Gamma_l} &= |_{\Gamma_r},
\end{align*}$$

where we set $\partial\Omega = \Gamma_D\cup \Gamma_l \cup \Gamma_r$, with

$$\begin{align*}
\Gamma_D &:=\{(x,y)\in \bar{\Omega}: y=0 \text{ or }y=1\},\quad&&\text{i.e. the bottom and top wall}\\
\Gamma_l &:=\{(x,y)\in \bar{\Omega}: x=0\},\quad&&\text{i.e. the left wall}\\
\Gamma_r &:=\{(x,y)\in \bar{\Omega}: x=1\},\quad&&\text{i.e. the right wall}.
\end{align*}$$




```
# imports
from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element
from dolfinx.fem import (Constant, Function, dirichletbc,
                         form, functionspace, locate_dofs_topological,
                         locate_dofs_geometrical,assemble_scalar)
from dolfinx.fem.petsc import assemble_matrix_block,assemble_vector_block, create_vector
from dolfinx.mesh import *
from ufl import div, dx, ds, grad, inner, FacetNormal, dot
from petsc4py import PETSc
from dolfinx import plot
import pyvista
import matplotlib as mpl
import dolfinx.fem as fem
from dolfinx_mpc import (LinearProblem, MultiPointConstraint, 
                         assemble_matrix_nest, assemble_vector_nest, 
                         create_matrix_nest, create_vector_nest)
from dolfinx import default_scalar_type
```



Here we prepare three convenience functions, that will help us produce an animation of the evolution of the solutions $u,\phi$ afterwards.

```

def create_gif(plotfunc, functionspace, filename, fps):
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(functionspace))
    plotter = pyvista.Plotter(off_screen=True) 

    plotter.open_gif(filename, fps=fps)
    plotter.show_axes()
    grid.point_data["uh1"] = plotfunc.x.array

    # optional:
    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    return grid, plotter, plotfunc
    

def update_gif(grid, plotter, plotfunc, warp_gif, clip_gif, clip_normal="y"): 
    maXual = max(plotfunc.x.array)
    maXual=1
    grid.point_data["uh"] = plotfunc.x.array

    if warp_gif and clip_gif:
        print("warp and clip not possible at the same time")
        plotter.clear()
        plotter.add_mesh(grid, clim=[-maXual,maXual],show_edges=True)
    elif warp_gif:
        grid_warped = grid.warp_by_scalar("uh", factor=0.2*1/maXual)
        plotter.clear()
        plotter.add_mesh(grid_warped, clim=[-maXual,maXual],show_edges=True)
    elif clip_gif:
        grad_clipped = grid.clip(clip_normal, invert=True)
        plotter.clear()
        plotter.add_mesh(grad_clipped, clim=[-maXual,maXual],show_edges=True)
        plotter.add_mesh(grid, style='wireframe', clim=[-maXual,maXual],show_edges=True)
    else:
        plotter.clear()
        plotter.add_mesh(grid, clim=[-maXual,maXual],show_edges=True)

    plotter.write_frame()
    plotter.show_axes()
    return plotter


def finalize_gif(plotter):
    plotter.close()

    
```

The goal is to solve the transport system using Lagrange Finite Elements. For this we propose the following weak formulation:

$$\begin{cases}
\displaystyle \int_\Omega  \partial_{t}u v + \int_\Omega  b\cdot\nabla\phi v=0 \quad  \forall v \in X^r(\Omega)\\
\displaystyle\int_\Omega  \partial_{t} \phi \psi  - \int_\Omega   u b\cdot\nabla\psi = 0 \quad  \forall \psi\in X^r_0(\Omega)
\end{cases}$$

Here $X^r$ denotes the Lagrange order $r$, global FEM space of piecewise polynomial functions that are globally continuous, defined as 
$$
X^r(\Omega) := \{u_h \in C (\bar{\Omega} ) : u_h |_K \in \mathbb{P}^r\ \forall K\} .
$$
The notation $X^r_0$ denotes the restriction of the above space to a space with zero boundary values.

For the derivative in time, we use the Crank-Nicolson scheme. We define some matrices:
$$\begin{aligned}
    M_{ij} &= \left \langle v_j, v_i \right \rangle \\
    N_{ij} &= \left \langle \psi_j,\psi_i \right \rangle  \\
    E_{ij} &= \left \langle \nabla\psi_j , b v_i \right \rangle \\
    F_{ij} &= \left \langle  v_j b,\nabla \psi_i\right \rangle 
\end{aligned}$$

The scheme is then:
$$
\begin{aligned}
    \begin{pmatrix}
        M &
        + \frac{\Delta t}{2}E \\
        - \frac{\Delta t}{2}F &
        N
    \end{pmatrix}  
    \begin{pmatrix} 
        \vec u  ^{n+1}\\ 
        \vec\phi^{n+1} 
    \end{pmatrix} = 
    \begin{pmatrix}
        M \vec u^n - \frac{\Delta t}{2} E \vec\phi^n\\
         \frac{\Delta t}{2} F\vec u^n +N\vec \phi^n
    \end{pmatrix}
\end{aligned}
$$


$$
\begin{aligned}
    \begin{pmatrix}
        M &
        + \frac{\Delta t}{2}E \\
        - \frac{\Delta t}{2}F &
        N
    \end{pmatrix}  
    \begin{pmatrix} 
        \vec u  ^{n+1}\\ 
        \vec\phi^{n+1} 
    \end{pmatrix} = 
    \begin{pmatrix}
        M \vec u^n - \frac{\Delta t}{2} E \vec\phi^n\\
         \frac{\Delta t}{2} F\vec u^n +N\vec \phi^n
    \end{pmatrix}
\end{aligned}
$$



It turns out the continuous problem, as well as the fully discrete version is stable, and one can show, that solutions $u,\phi$ admit the following conserved quanitity:
$$\mathcal{E} = \int_\Omega u^2 +\phi^2.$$

We will now define this in fenicsx.

```
# parameters
dt = 0.02
t = 0.
T_end = 2
num_steps = int(T_end/dt)
Nx = 40
h = 1/Nx
bconst=1.
filename_gifu = "gif_u.gif"
filename_gifp = "gif_p.gif"
warp_gif = True

# mesh, spaces, functions
msh = create_unit_square(MPI.COMM_WORLD, Nx, Nx)
P1 = element("Lagrange", "triangle", 1) 
Xu = functionspace(msh, P1) 
Xp = functionspace(msh, P1) 
eps = 0.8
bfield = Constant(msh, (eps,np.sqrt(1-eps**2)))
x = ufl.SpatialCoordinate(msh)
n = FacetNormal(msh)
u = ufl.TrialFunction(Xu)
p = ufl.TrialFunction(Xp)
W = ufl.TestFunction(Xu)
q = ufl.TestFunction(Xp)
u_old = Function(Xu)
p_old = Function(Xp)

# bump function as initial conditions
u_old.interpolate(lambda x: 0.*x[0])
p_old.interpolate(lambda x: np.exp(-1 * (((x[0]-0.5)/0.15)**2 + ((x[1]-0.5)/0.15)**2)))

# gif
gridu, plotteru, plotfuncu = create_gif(u_old, Xu, filename_gifu, fps=0.1/dt)
gridp, plotterp, plotfuncp = create_gif(p_old, Xp, filename_gifp, fps=0.1/dt)

# Dirichlet BC
facets = locate_entities_boundary(msh, dim=1, marker=lambda x: np.logical_or.reduce((
                                           np.isclose(x[1], 1.0),
                                           np.isclose(x[1], 0.0))))

dofs = fem.locate_dofs_topological(V=Xp, entity_dim=1, entities=facets)
bcs = [fem.dirichletbc(0.0, dofs=dofs, V=Xp)]

# Periodic BC
tol = 250 * np.finfo(default_scalar_type).resolution
def periodic_boundary(x):
    return np.isclose(x[0], 1, atol=tol)

def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    return out_x

mpc_p = MultiPointConstraint(Xp)
mpc_p.create_periodic_constraint_geometrical(Xp, periodic_boundary, periodic_relation, bcs)
mpc_p.finalize()

mpc_V = MultiPointConstraint(Xu)
mpc_V.create_periodic_constraint_geometrical(Xu, periodic_boundary, periodic_relation, bcs)
mpc_V.finalize()

# new time step values
u_new = fem.Function(mpc_V.function_space) # this?
p_new = fem.Function(mpc_p.function_space) # this?
# u_new = Function(Xu) # seems to work as well
# p_new = Function(Xp) # seems to work as well
```

Now we assemble the linear system.

```
# weak form implicit euler
M = inner(u,W)*dx 
E = inner(grad(p),bfield*W)*dx
F = inner(grad(u),bfield*q)*dx
N = inner(p,q)*dx
a = form([[M, +dt/2*E],
          [+dt/2*F, N]])

# assemble
A = create_matrix_nest(a, [mpc_V, mpc_p])
assemble_matrix_nest(A, a, [mpc_V, mpc_p], bcs)
A.assemble()
# up_vec = A.createVecLeft()

# solver
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

l = form([inner(u_old,W)*dx - dt/2*inner(grad(p_old),bfield*W)*dx,
            inner(p_old,q)*dx - dt/2*inner(grad(u_old),bfield*q)*dx]) 
L = create_vector_nest(l,[mpc_V, mpc_p])
```



In the time-loop, we have to update the RHS:
1. The Crank-Nicolson scheme requires old function values (i.e from the previous time step) on the RHS, and new function values on the LHS. These values change in each iteration of course.
2. Potentially, one would have to update the boundary conditions, which in our case is not necessary.

```
for i in range(num_steps):
    t += dt

    # update RHS
    assemble_vector_nest(L,l,[mpc_V, mpc_p])

    # Dirichlet BC values in RHS
    fem.petsc.apply_lifting_nest(L, a, bcs)
    for b_sub in L.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  
    bcs0 = fem.bcs_by_block(fem.extract_function_spaces(l), bcs)
    fem.petsc.set_bc_nest(L, bcs0)
    
    # solve
    up_vec = L.copy()
    solver.solve(L, up_vec)

    # extract solution
    u_new.x.petsc_vec.setArray(up_vec.getNestSubVecs()[0].array)
    p_new.x.petsc_vec.setArray(up_vec.getNestSubVecs()[1].array)

    # update MPC slave dofs
    mpc_V.backsubstitution(u_new)
    mpc_p.backsubstitution(p_new)

    # gif
    plotterV = update_gif(gridu, plotteru, plotfuncu, warp_gif, clip_gif=False)
    plotterp = update_gif(gridp, plotterp, plotfuncp, warp_gif, clip_gif=False)

    # print energy to check conservation
    E = assemble_scalar(form(inner(u_new,u_new)*dx + inner(p_new,p_new)*dx))
    print("time:", round(t,5), "\t E:", E)
                    
    # Update solution at previous time step
    u_old.x.array[:] = u_new.x.array
    p_old.x.array[:] = p_new.x.array

# gif and plot
finalize_gif(plotterV)
finalize_gif(plotterp)
print("gifs saved as", filename_gifu, filename_gifp)

```




The runtime should not be much more than a minute.



The `.py` file can be downloaded [here](https://www.markusrenoldner.com/files/transport_sys_periodic.py).






















