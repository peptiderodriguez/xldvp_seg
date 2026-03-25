"""
Processing pipeline components for the segmentation pipeline.

Provides:
- Detection pipeline for processing CZI files
- Batch processing for multiple slides
- Coordinate conversion utilities
- Tile processing helpers to reduce nesting
"""

from .batch import (
    BatchProcessor,
    BatchResult,
    SlideInfo,
    collect_slides,
    create_batch_summary_html,
)
from .coordinates import (
    # Coordinate validation
    CoordinateValidationError,
    array_index_to_xy,
    convert_detections_to_spatial_uids,
    # Coordinate labeling helpers
    create_coordinate_dict,
    extract_crop,
    extract_crop_bounds,
    format_coordinates_for_export,
    generate_uid,
    global_to_tile_coords,
    is_spatial_uid,
    mask_to_crop_coords,
    migrate_uid_format,
    # UID parsing and migration
    parse_uid,
    regionprop_centroid_to_xy,
    tile_to_global_coords,
    validate_array_indices,
    validate_bbox_xyxy,
    validate_xy_coordinates,
    xy_to_array_index,
)
from .memory import (
    get_memory_usage,
    get_safe_worker_count,
    log_memory_status,
    validate_system_resources,
)
from .mk_hspc_utils import (
    build_mk_hspc_result,
    check_tile_validity,
    ensure_rgb_array,
    extract_tile_from_shared_memory,
    prepare_tile_for_detection,
)
from .multigpu_shm import (
    SharedSlideManager,
)
from .multigpu_worker import (
    MultiGPUTileProcessor,
)
from .pipeline import (
    DetectionPipeline,
    create_simple_detector,
)
from .tile_processing import (
    enrich_detection_features,
    save_tile_outputs,
)

__all__ = [
    # Pipeline
    "DetectionPipeline",
    "create_simple_detector",
    # Batch
    "SlideInfo",
    "BatchProcessor",
    "BatchResult",
    "collect_slides",
    "create_batch_summary_html",
    # Coordinates
    "regionprop_centroid_to_xy",
    "xy_to_array_index",
    "array_index_to_xy",
    "extract_crop_bounds",
    "extract_crop",
    "global_to_tile_coords",
    "tile_to_global_coords",
    "generate_uid",
    "mask_to_crop_coords",
    # UID parsing and migration
    "parse_uid",
    "migrate_uid_format",
    "is_spatial_uid",
    # Coordinate validation
    "CoordinateValidationError",
    "validate_xy_coordinates",
    "validate_array_indices",
    "validate_bbox_xyxy",
    # Coordinate labeling helpers
    "create_coordinate_dict",
    "format_coordinates_for_export",
    "convert_detections_to_spatial_uids",
    # Tile processing
    "enrich_detection_features",
    "save_tile_outputs",
    # Memory management
    "validate_system_resources",
    "get_safe_worker_count",
    "get_memory_usage",
    "log_memory_status",
    # MK/HSPC utilities
    "ensure_rgb_array",
    "check_tile_validity",
    "prepare_tile_for_detection",
    "build_mk_hspc_result",
    "extract_tile_from_shared_memory",
    # Multi-GPU processing
    "MultiGPUTileProcessor",
    # Multi-GPU shared memory
    "SharedSlideManager",
]
