# Visualization

Reusable HTML visualization components for fluorescence microscopy data.
Provides CZI thumbnail loading, color palette assignment, binary data encoding,
detection data streaming, and composable JavaScript components for Canvas 2D
rendering.

Used by `scripts/generate_multi_slide_spatial_viewer.py` and
`scripts/generate_contour_viewer.py`.

```python
from xldvp_seg.visualization import (
    read_czi_thumbnail_channels,
    encode_channel_b64,
    assign_group_colors,
    build_contour_js_data,
    load_js,
)
```

## Fluorescence

CZI thumbnail loading and base64 encoding for HTML embedding.

::: xldvp_seg.visualization.fluorescence
    options:
      show_root_heading: false
      members_order: source

## Colors

Color palettes and group assignment for visualization.

::: xldvp_seg.visualization.colors
    options:
      show_root_heading: false
      members_order: source

## Encoding

Binary data encoding for efficient HTML embedding.

::: xldvp_seg.visualization.encoding
    options:
      show_root_heading: false
      members_order: source

## Data Loading

Detection JSON streaming, position extraction, and group discovery.

::: xldvp_seg.visualization.data_loading
    options:
      show_root_heading: false
      members_order: source

## Graph Patterns

Spatial graph pattern detection for viewer overlays.

::: xldvp_seg.visualization.graph_patterns
    options:
      show_root_heading: false
      members_order: source

## JS Loader

Composable JavaScript component loading for HTML viewers.

::: xldvp_seg.visualization.js_loader
    options:
      show_root_heading: false
      members_order: source
