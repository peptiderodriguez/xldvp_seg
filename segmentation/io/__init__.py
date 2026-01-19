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
    generate_dual_index_page,
    export_samples_to_html,
)

from .html_generator import (
    HTMLPageGenerator,
    # MK/HSPC batch export functions
    load_samples_from_ram,
    create_mk_hspc_index,
    generate_mk_hspc_page_html,
    generate_mk_hspc_pages,
    export_mk_hspc_html_from_ram,
    # Backward compatibility aliases
    create_export_index,
    generate_export_page_html,
    generate_export_pages,
    export_html_from_ram,
)

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
    'generate_dual_index_page',
    'export_samples_to_html',
    # HTML Export (class-based API)
    'HTMLPageGenerator',
    # MK/HSPC batch export functions
    'load_samples_from_ram',
    'create_mk_hspc_index',
    'generate_mk_hspc_page_html',
    'generate_mk_hspc_pages',
    'export_mk_hspc_html_from_ram',
    # Backward compatibility aliases
    'create_export_index',
    'generate_export_page_html',
    'generate_export_pages',
    'export_html_from_ram',
    # Tile Pipeline (async loading)
    'TilePipeline',
    'TilePipelineWithPreprocessing',
    'preprocess_tiles_batch',
    'preprocess_tiles_batch_with_coords',
    'AsyncTileLoader',
]
