"""
Processing pipeline components for the segmentation pipeline.

Provides:
- Detection pipeline for processing CZI files
- Batch processing for multiple slides
- Coordinate conversion utilities
- Tile processing helpers to reduce nesting
"""

from .pipeline import (
    DetectionPipeline,
    create_simple_detector,
)

from .batch import (
    SlideInfo,
    BatchProcessor,
    BatchResult,
    collect_slides,
    create_batch_summary_html,
)

from .coordinates import (
    regionprop_centroid_to_xy,
    xy_to_array_index,
    array_index_to_xy,
    extract_crop_bounds,
    extract_crop,
    global_to_tile_coords,
    tile_to_global_coords,
    generate_uid,
    mask_to_crop_coords,
)

from .tile_processing import (
    build_detection_params,
    load_and_validate_tile,
    enrich_detection_features,
    save_tile_outputs,
    process_tile_complete,
)

from .memory import (
    validate_system_resources,
    get_safe_worker_count,
    get_memory_usage,
    log_memory_status,
)

from .mk_hspc_utils import (
    ensure_rgb_array,
    check_tile_validity,
    prepare_tile_for_detection,
    build_mk_hspc_result,
    extract_tile_from_shared_memory,
)

__all__ = [
    # Pipeline
    'DetectionPipeline',
    'create_simple_detector',
    # Batch
    'SlideInfo',
    'BatchProcessor',
    'BatchResult',
    'collect_slides',
    'create_batch_summary_html',
    # Coordinates
    'regionprop_centroid_to_xy',
    'xy_to_array_index',
    'array_index_to_xy',
    'extract_crop_bounds',
    'extract_crop',
    'global_to_tile_coords',
    'tile_to_global_coords',
    'generate_uid',
    'mask_to_crop_coords',
    # Tile processing
    'build_detection_params',
    'load_and_validate_tile',
    'enrich_detection_features',
    'save_tile_outputs',
    'process_tile_complete',
    # Memory management
    'validate_system_resources',
    'get_safe_worker_count',
    'get_memory_usage',
    'log_memory_status',
    # MK/HSPC utilities
    'ensure_rgb_array',
    'check_tile_validity',
    'prepare_tile_for_detection',
    'build_mk_hspc_result',
    'extract_tile_from_shared_memory',
]
