# I/O

## CZI Loader

Unified CZI loading with RAM-first caching. Handles multi-channel loading,
channel resolution by name/wavelength, and lazy channel access.

::: xldvp_seg.io.czi_loader
    options:
      show_root_heading: false
      members_order: source
      members:
        - get_loader
        - CZILoader

## SpatialData Export

Convert pipeline detections to SpatialData format for integration with
the scverse ecosystem (scanpy, squidpy).

::: xldvp_seg.io.spatialdata_export
    options:
      show_root_heading: false
      members_order: source
