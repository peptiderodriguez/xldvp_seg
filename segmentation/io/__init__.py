"""
I/O operations for the segmentation pipeline.

Provides:
- CZI file loading with optional RAM caching
- HTML export for annotation interfaces
- Tile pipeline for async tile loading with producer-consumer pattern
"""

from .czi_loader import (
    CZILoader,
    get_loader,
    clear_cache,
    get_cached_paths,
)

from .html_export import (
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
    generate_annotation_page,
    generate_index_page,
    export_samples_to_html,
)

from .html_generator import HTMLPageGenerator

from .tile_pipeline import (
    TilePipeline,
    TilePipelineWithPreprocessing,
    preprocess_tiles_batch,
    preprocess_tiles_batch_with_coords,
    AsyncTileLoader,
)

__all__ = [
    # CZI Loading
    'CZILoader',
    'get_loader',
    'clear_cache',
    'get_cached_paths',
    # HTML Export (functional API)
    'percentile_normalize',
    'draw_mask_contour',
    'image_to_base64',
    'generate_annotation_page',
    'generate_index_page',
    'export_samples_to_html',
    # HTML Export (class-based API)
    'HTMLPageGenerator',
    # Tile Pipeline (async loading)
    'TilePipeline',
    'TilePipelineWithPreprocessing',
    'preprocess_tiles_batch',
    'preprocess_tiles_batch_with_coords',
    'AsyncTileLoader',
]
