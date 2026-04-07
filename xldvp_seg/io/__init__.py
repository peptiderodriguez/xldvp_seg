"""
I/O operations for the segmentation pipeline.

Provides:
- CZI file loading with optional RAM caching
- HTML export for annotation interfaces
- Tile pipeline for async tile loading with producer-consumer pattern
"""

from .czi_loader import (
    CZILoader,
    clear_cache,
    get_cached_paths,
    get_loader,
)
from .html_export import (
    export_samples_to_html,
    generate_annotation_page,
    generate_dual_index_page,
    generate_index_page,
)
from .html_generator import (
    HTMLPageGenerator,
    # Backward compatibility aliases
    create_export_index,
    create_mk_hspc_index,
    export_html_from_ram,
    export_mk_hspc_html_from_ram,
    generate_export_page_html,
    generate_export_pages,
    generate_mk_hspc_page_html,
    generate_mk_hspc_pages,
    # MK/HSPC batch export functions
    load_samples_from_ram,
)
from .html_utils import (
    draw_mask_contour,
    image_to_base64,
    percentile_normalize,
)
from .tile_pipeline import (
    AsyncTileLoader,
    TilePipeline,
    TilePipelineWithPreprocessing,
    preprocess_tiles_batch,
    preprocess_tiles_batch_with_coords,
)

__all__ = [
    # CZI Loading
    "CZILoader",
    "get_loader",
    "clear_cache",
    "get_cached_paths",
    # HTML Export (functional API)
    "percentile_normalize",
    "draw_mask_contour",
    "image_to_base64",
    "generate_annotation_page",
    "generate_index_page",
    "generate_dual_index_page",
    "export_samples_to_html",
    # HTML Export (class-based API)
    "HTMLPageGenerator",
    # MK/HSPC batch export functions
    "load_samples_from_ram",
    "create_mk_hspc_index",
    "generate_mk_hspc_page_html",
    "generate_mk_hspc_pages",
    "export_mk_hspc_html_from_ram",
    # Backward compatibility aliases
    "create_export_index",
    "generate_export_page_html",
    "generate_export_pages",
    "export_html_from_ram",
    # Tile Pipeline (async loading)
    "TilePipeline",
    "TilePipelineWithPreprocessing",
    "preprocess_tiles_batch",
    "preprocess_tiles_batch_with_coords",
    "AsyncTileLoader",
]
