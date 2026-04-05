# Training

Feature loading and filtering for classifier training. Provides functions to
build feature matrices from detection JSON files, filter by feature set, and
match against annotation labels.

```python
from xldvp_seg.training.feature_loader import load_features_and_annotations

X, y, feature_names = load_features_and_annotations(
    detections_path="cell_detections.json",
    annotations_path="annotations.json",
    feature_set="morph",
)
```

::: xldvp_seg.training.feature_loader
    options:
      show_root_heading: false
      members_order: source
