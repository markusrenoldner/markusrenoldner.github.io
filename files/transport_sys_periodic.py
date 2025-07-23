
"""
transport/advection system with
- Dirichlet BC in y=0, y=1
- periodic BC in x=0, x=1

see e.g.: https://github.com/jorgensd/dolfinx_mpc/blob/main/python/demos/demo_stokes_nest.py

"""

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


def create_gif(plotfunc, functionspace, filename, fps):
    """
    Create a GIF animation from a given plotting function and function space.
    Parameters:
    plotfunc (function): The plotting function that generates the data to be visualized.
    functionspace (dolfinx.FunctionSpace): The function space containing the mesh and function data.
    filename (str): The name of the output GIF file.
    warp_gif (bool): If True, warp the mesh by scalar values for visualization.
    fps (int): Frames per second for the GIF animation.
    Returns:
    tuple: A tuple containing the grid, plotter, and plotfunc used for creating the GIF.
    Example:
    --------
    # grid, plotter, plotfunc = create_gif(plotfunc, functionspace, "output.gif", True, 10)
    # Refer to the tutorial at https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html for more details.
        
    TUTORIAL:
    grid, plotter, warped, plotfunc = create_gif(...)
    for i in range(N):
        plotter = update_gif(...)
    finalize_gif(...)
    """
    # pyvista.start_xvfb() # why?
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(functionspace))
    plotter = pyvista.Plotter(off_screen=True) 
    # plotter.camera_position = [(0.2, 0.2, 5),  # position of camera
    #                            (0, 0, 0),  # point which cam looks at
    #                            (0, 0, 1),]    # up-axis

    plotter.open_gif(filename, fps=fps)
    plotter.show_axes()
    grid.point_data["uh1"] = plotfunc.x.array

    # optional:
    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    return grid, plotter, plotfunc
    

def update_gif(grid, plotter, plotfunc, warp_gif, clip_gif, clip_normal="y"): 
    """
    Update the plotter with the given grid and plot function, and optionally warp or clip the grid.

    Parameters:
    grid (pyvista.UnstructuredGrid): The grid to be plotted.
    plotter (pyvista.Plotter): The plotter instance used for plotting.
    plotfunc (object): An object containing the data to be plotted, with an attribute `x.array`.
    warp_gif (bool): If True, warp the grid by the scalar values.
    clip_gif (bool): If True, clip the grid along the specified normal.
    clip_normal (str, optional): The normal direction for clipping. Default is "y".

    Returns:
    pyvista.Plotter: The updated plotter instance.
    """
    maxval = max(plotfunc.x.array)
    maxval=1
    grid.point_data["uh"] = plotfunc.x.array

    if warp_gif and clip_gif:
        print("warp and clip not possible at the same time")
        plotter.clear()
        plotter.add_mesh(grid, clim=[-maxval,maxval],show_edges=True)
    elif warp_gif:
        grid_warped = grid.warp_by_scalar("uh", factor=0.2*1/maxval)
        plotter.clear()
        plotter.add_mesh(grid_warped, clim=[-maxval,maxval],show_edges=True)
    elif clip_gif:
        grad_clipped = grid.clip(clip_normal, invert=True)
        plotter.clear()
        plotter.add_mesh(grad_clipped, clim=[-maxval,maxval],show_edges=True)
        plotter.add_mesh(grid, style='wireframe', clim=[-maxval,maxval],show_edges=True)
    else:
        plotter.clear()
        plotter.add_mesh(grid, clim=[-maxval,maxval],show_edges=True)
        # optional:
        # viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
        # plotter.add_mesh(grid, show_edges=True, lighting=False,
        #                  cmap=viridis, scalar_bar_args=sargs,
        #                  clim=[-maxval,maxval])

    plotter.write_frame()
    plotter.show_axes()
    # plotter.camera_position = [(0.2, 0.2, 5),  # position of camera
    #                            (0, 0, 0),  # point which cam looks at
    #                            (0, 0, 1),]    # up-axis
    # plotter.camera_position = 'xy'
    # plotter.camera_position = [(3,3,2),     # position of camera
    #                            (0.5,0.5,0), # point which cam looks at
    #                            (0,0,1)]     # up-axis
    return plotter


def finalize_gif(plotter):
    plotter.close()

    
# parameters
t = 0.
Nt = 100
T_end = 2
dt = T_end/Nt
Nx = 20
h = 1/Nx
bconst=1.
filename_gifV = "testu.gif"
filename_gifp = "testp.gif"
warp_gif = True

# mesh, spaces, functions
msh = create_unit_square(MPI.COMM_WORLD, Nx, Nx)
P1 = element("Lagrange", "triangle", 1) 
# P3 = element("Lagrange", "triangle", 3) 
XV = functionspace(msh, P1) 
Xp = functionspace(msh, P1) 
eps = 0.4
bfield = Constant(msh, (np.sqrt(1-eps**2),eps))
x = ufl.SpatialCoordinate(msh)
n = FacetNormal(msh)
V = ufl.TrialFunction(XV)
p = ufl.TrialFunction(Xp)
W = ufl.TestFunction(XV)
q = ufl.TestFunction(Xp)
V_old = Function(XV)
p_old = Function(Xp)

# initial conditions
p_old.interpolate(lambda x: 0.*x[0])
V_old.interpolate(lambda x: np.exp(-1 * (+ 1*((x[0]-0.5)/0.15)**2 
                                         + 1*((x[1]-0.5)/0.15)**2)))

# gif
gridV, plotterV, plotfuncV = create_gif(V_old, XV, filename_gifV, fps=0.1/dt)
gridp, plotterp, plotfuncp = create_gif(p_old, Xp, filename_gifp, fps=0.1/dt)

# Dirichlet BC
facets = locate_entities_boundary(msh, dim=1, marker=lambda x: np.logical_or.reduce((
                                           np.isclose(x[1], 1.0),
                                           np.isclose(x[1], 0.0))))

dofs = fem.locate_dofs_topological(V=Xp, entity_dim=1, entities=facets)
bcs = [fem.dirichletbc(0.0, dofs=dofs, V=Xp)]

# Periodic BC
# tol = 250 * np.finfo(default_scalar_type).resolution
def periodic_boundary(x):
    return np.isclose(x[0], 1)

def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 0
    out_x[1] = x[1]
    return out_x

mpc_p = MultiPointConstraint(Xp)
mpc_p.create_periodic_constraint_geometrical(Xp, periodic_boundary, periodic_relation, bcs)
mpc_p.finalize()

mpc_V = MultiPointConstraint(XV)
mpc_V.create_periodic_constraint_geometrical(XV, periodic_boundary, periodic_relation, bcs)
mpc_V.finalize()

# new time step values
V_new = fem.Function(mpc_V.function_space) # necessary?
p_new = fem.Function(mpc_p.function_space) # necessary?
# V_new = Function(XV) # seems to work as well
# p_new = Function(Xp) # seems to work as well

# weak form implicit euler
M = inner(V,W)*dx 
E = inner(grad(p),bfield*W)*dx
F = inner(grad(V),bfield*q)*dx
N = inner(p,q)*dx
a = form([[M, +dt/2*E],
          [+dt/2*F, N]])
# a = [[form(M),form(dt/2*E)], # alternative
#     [form(dt/2*F), form(N)]] # alternative

# assemble
A = create_matrix_nest(a, [mpc_V, mpc_p])
assemble_matrix_nest(A, a, [mpc_V, mpc_p], bcs)
A.assemble()
# Vp_vec = A.createVecLeft()

# solver
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

l = form([inner(V_old,W)*dx - dt/2*inner(grad(p_old),bfield*W)*dx,
            inner(p_old,q)*dx - dt/2*inner(grad(V_old),bfield*q)*dx]) 
# l = [form(inner(V_old,W)*dx - dt/2*inner(grad(p_old),bfield*W)*dx), # alternative
#      form(inner(p_old,q)*dx - dt/2*inner(grad(V_old),bfield*q)*dx)] # alternative
L = create_vector_nest(l,[mpc_V, mpc_p])

progress_0 = 0
for i in range(Nt):
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
    Vp_vec = L.copy()
    solver.solve(L, Vp_vec)

    # extract solution
    V_new.x.petsc_vec.setArray(Vp_vec.getNestSubVecs()[0].array)
    p_new.x.petsc_vec.setArray(Vp_vec.getNestSubVecs()[1].array)

    # extract solution: alternative which also works for non-nested arrays
    # offset = XV.dofmap.index_map.size_local * XV.dofmap.index_map_bs
    # V_new.x.array[:offset] = Vp_vec.array_r[:offset]
    # p_new.x.array[:(len(Vp_vec.array_r) - offset)] = Vp_vec.array_r[offset:]

    # update MPC slave dofs
    mpc_V.backsubstitution(V_new)
    mpc_p.backsubstitution(p_new)

    # gif
    plotterV = update_gif(gridV, plotterV, plotfuncV, warp_gif, clip_gif=False)
    plotterp = update_gif(gridp, plotterp, plotfuncp, warp_gif, clip_gif=False)

    # progress
    progress = int(((i+1) / Nt) * 100)
    if progress >= progress_0+20:
        progress_0 = progress
        E = assemble_scalar(form(inner(V_new,V_new)*dx + inner(p_new,p_new)*dx))
        print("|--progress:", progress, "% \t time:", round(t,5), "\t E:", E)
                    
    # Update solution at previous time step
    V_old.x.array[:] = V_new.x.array
    p_old.x.array[:] = p_new.x.array

# gif and plot
finalize_gif(plotterV)
finalize_gif(plotterp)
print("gifs saved as", filename_gifV, filename_gifp)


