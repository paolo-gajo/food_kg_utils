# Food recipe knowledge graph utilities

This repo contains utilities to make knowledge graphs of food recipes.

To make the GialloZafferano dataset:

1. The scripts `get_urls.py` and `scrape.py` download pages from GialloZafferano and store them in a dataset. 

2. `annotate_locs.py` uses an LLM to extract the place of origin of a dish, if present.

3. `make_graph.py` adds other metadata and builds edges between nodes.