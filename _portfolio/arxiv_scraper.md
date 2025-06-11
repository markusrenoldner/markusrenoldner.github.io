---
title: 'Costomised, daily arxiv summary'
excerpt: "A short python script, that outputs a short summary of recent arxiv uploads based on user defined keywords and author names."

collection: portfolio
---

The following python script takes as inputs

- a list of authors
- a list of keywords

and prints into the terminal a list of articles, that either contain the authorname, or whose title contains any of the keywords. The list also contains a clickable link that leads to the arxiv PDF file.

The script can also be used with command-line arguments.


```
import requests
from bs4 import BeautifulSoup
import argparse


def fetch_arxiv_math_na(keywords, authors, subject, max_results=1000):

    """
    keywords and authors are strings, and not case sensitive.

    max_results valid values: 25, 50, 100, 250, 500, 1000, 2000

    subjects (examples):
    - math.NA: Numerical Analysis
    - math.AP: Analysis of PDEs
    - math.DG: Differential Geometry
    - math.OC: Optimization and Control
    - physics.plasm-ph: Plasma Physics

    return: 
    - prints the title and arXiv link of the papers that match the keywords or authors.
    """

    url = "https://arxiv.org/list/"+str(subject)+"/recent?skip=0&show="+str(max_results)

    # fetch page
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    entries = soup.find_all("dt")
    details = soup.find_all("dd")

    # extract human readable info
    for dt, dd in zip(entries, details):

        # arXiv ID and link
        link_tag = dt.find("a", title="Abstract") # search for HTML tag <a> with title "Abstract"
        if not link_tag:
            continue
        arxiv_id = link_tag.text.strip()
        abs_link = f"https://arxiv.org/abs/{arxiv_id}"

        # title
        title_line = dd.find("div", class_="list-title mathjax")
        if title_line:
            title = title_line.text.replace("Title:", "").strip()
        else:
            title = ""

        # authors
        author_line = dd.find("div", class_="list-authors")
        if author_line:
            author_text = author_line.text.replace("Authors:", "").strip()
        else:
            author_text = ""

        # match title or author
        title_lc = title.lower()
        authors_lc = author_text.lower()
        if any(k.lower() in title_lc for k in keywords) or any(a.lower() in authors_lc for a in authors):
            print(f"{title}\
                  \n{author_text}\
                  \n{abs_link}\n")


if __name__ == "__main__":

    # Example usage
    keywords = ["finite element", 
                "fluid", 
                "navier"
                ]
    authors = ["hiptmair",
               "schöberl"]

    fetch_arxiv_math_na(keywords, authors, subject="math.NA")

```


To use the tool with command line arguments, write the following into the main function:

```
    parser = argparse.ArgumentParser(description="Fetch recent arXiv papers by keyword or author")
    parser.add_argument("--keywords", nargs="*", default=[], help="List of keywords to search in titles")
    parser.add_argument("--authors", nargs="*", default=[], help="List of author names to search")
    args = parser.parse_args()
    fetch_arxiv_math_na(args.keywords, args.authors, subject="math.NA")
```

The `.py` file can be downloaded [here](https://www.markusrenoldner.com/files/arxiv_scraper.py).
