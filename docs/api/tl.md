# Analysis Tools (tl)

Scanpy-style analysis functions that operate on `SlideAnalysis` objects.
Each function mutates detections in-place and returns the slide for chaining.

```python
from xldvp_seg.core import SlideAnalysis
from xldvp_seg.api import tl

slide = SlideAnalysis.load("output/...")
tl.markers(slide, marker_channels=[1, 2], marker_names=["NeuN", "tdTomato"])
tl.score(slide, classifier="classifiers/rf_morph.pkl")
tl.cluster(slide, feature_groups="morph,channel")
```

::: xldvp_seg.api.tl
    options:
      show_root_heading: false
      members_order: source
