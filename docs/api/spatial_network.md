# Spatial Networks

Spatial cell analysis functions: Delaunay triangulation networks, Louvain
community detection, RF embedding UMAP, and morphological feature UMAP.

```python
from xldvp_seg.analysis.spatial_network import run_spatial_network

run_spatial_network(
    detections=detections,
    output_dir="results/spatial/",
    marker_filter=["SMA_class==positive"],
)
```

::: xldvp_seg.analysis.spatial_network
    options:
      show_root_heading: false
      members_order: source
