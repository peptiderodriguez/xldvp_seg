# Marker Classification

Classify cells as marker-positive or marker-negative using threshold-based
methods on per-channel intensity features.

Methods available:

- **SNR** (default) -- median-based signal-to-noise ratio >= 1.5
- **Otsu** -- automatic threshold maximizing inter-class variance
- **Otsu half** -- permissive Otsu/2 threshold for dim markers
- **GMM** -- 2-component Gaussian mixture model for overlapping distributions

::: xldvp_seg.analysis.marker_classification
    options:
      show_root_heading: false
      members_order: source
