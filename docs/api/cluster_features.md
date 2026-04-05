# Feature Clustering

Utility functions for feature-based clustering of cell detections. Includes
channel discovery, feature selection, feature matrix extraction, normalization,
cluster auto-labeling, and spatial smoothing.

Feature groups:

| Group | Description |
|-------|-------------|
| `morph` | All morphological features (shape + color) |
| `shape` | Pure geometry (area, circularity, solidity, aspect_ratio) |
| `color` | Intensity/color (gray_mean, hue_mean, relative_brightness) |
| `sam2` | SAM2 embedding features (256 dimensions) |
| `channel` | Per-channel statistics (mean, std, ratios) |
| `deep` | Deep features (ResNet + DINOv2, 6144 dimensions) |

::: xldvp_seg.analysis.cluster_features
    options:
      show_root_heading: false
      members_order: source
