"""
Pipeline orchestration modules for the unified segmentation pipeline.

Submodules:
    server: HTTP server + Cloudflare tunnel management
    resume: Resume/checkpoint detection and tile reload
    samples: HTML crop creation, tile grid, islet marker calibration
    finalize: Channel legend, CSV/JSON/HTML export, summary
    detection_setup: Strategy creation, classifier loading, parameter building
    preprocessing: Photobleach, flat-field, Reinhard normalization
    cli: Argument parser construction and postprocessing
    background: Local background correction (KD-tree based)
    post_detection: Post-dedup pipeline (dilate, re-extract features, bg correct)
"""
