# SlideAnalysis

Central state object wrapping pipeline output. Provides lazy-loaded properties,
filtering, and export to AnnData/SpatialData.

```python
from xldvp_seg.core import SlideAnalysis

slide = SlideAnalysis.load("/path/to/output/slide_name/run_timestamp/")
print(f"{slide.n_detections} detections, {slide.cell_type}")

df = slide.features_df       # pandas DataFrame of all features
pos = slide.positions_um     # Nx2 array of spatial coordinates

good = slide.filter(score_threshold=0.5)
neurons = slide.filter(marker="NeuN", positive=True)

adata = slide.to_anndata()
```

::: xldvp_seg.core.slide_analysis.SlideAnalysis
    options:
      show_root_heading: true
      members_order: source
