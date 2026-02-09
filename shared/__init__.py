"""
DEPRECATED: The shared module has been restructured.

Please update your imports to use the new segmentation package:

Old imports:
    from shared.czi_loader import CZILoader
    from shared.logging_config import get_logger
    from shared.tissue_detection import has_tissue
    from shared.html_export import percentile_normalize

New imports:
    from segmentation.io.czi_loader import CZILoader
    from segmentation.utils.logging import get_logger
    from segmentation.detection.tissue import has_tissue
    from segmentation.io.html_export import percentile_normalize

Or use the top-level imports:
    from segmentation import CZILoader, get_logger, has_tissue

This compatibility layer will be removed in a future version.
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "The 'shared' module is deprecated. "
    "Please update imports to use 'segmentation' package instead. "
    "See shared/__init__.py for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new locations for backward compatibility
from segmentation.detection.tissue import (
    calculate_block_variances,
    has_tissue,
    compute_variance_threshold,
    compute_tissue_thresholds,
    calibrate_tissue_threshold,
    filter_tissue_tiles,
)

from segmentation.io.html_export import (
    generate_annotation_page,
    generate_index_page,
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)

from segmentation.utils.config import (
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    load_config,
    save_config,
    create_run_config,
    get_pixel_size,
    get_normalization_percentiles,
    get_default_path,
    get_output_dir,
)

from segmentation.processing.coordinates import (
    regionprop_centroid_to_xy,
    xy_to_array_index,
    array_index_to_xy,
    extract_crop_bounds,
    extract_crop,
    global_to_tile_coords,
    tile_to_global_coords,
    generate_uid,
    mask_to_crop_coords,
    # UID parsing and migration
    parse_uid,
    migrate_uid_format,
    is_spatial_uid,
    # Coordinate validation
    CoordinateValidationError,
    validate_xy_coordinates,
    validate_array_indices,
    validate_bbox_xyxy,
    # Coordinate labeling helpers
    create_coordinate_dict,
    format_coordinates_for_export,
    convert_detections_to_spatial_uids,
)

from segmentation.io.czi_loader import CZILoader, get_loader, clear_cache, get_cached_paths

from segmentation.processing.pipeline import DetectionPipeline, create_simple_detector

from segmentation.utils.logging import (
    get_logger,
    setup_logging,
    log_parameters,
    log_processing_start,
    log_processing_end,
    ProcessingTimer,
)

from segmentation.processing.batch import (
    SlideInfo,
    BatchProcessor,
    BatchResult,
    collect_slides,
    create_batch_summary_html,
)

from segmentation.utils.feature_extraction import (
    extract_resnet_features_batch,
    extract_resnet_features_single,
    preprocess_crop_for_resnet,
    create_resnet_transform,
)

# Schemas require pydantic - try to import, but don't fail if not available
try:
    from segmentation.utils.schemas import (
        Detection,
        DetectionFile,
        Config,
        Annotations,
        validate_detection_file,
        validate_config_file,
        validate_annotations_file,
        infer_and_validate,
    )
    _HAS_SCHEMAS = True
except ImportError:
    _HAS_SCHEMAS = False

__all__ = [
    # Tissue detection
    'calculate_block_variances',
    'has_tissue',
    'compute_variance_threshold',
    'compute_tissue_thresholds',
    'calibrate_tissue_threshold',
    'filter_tissue_tiles',
    # HTML export
    'generate_annotation_page',
    'generate_index_page',
    'export_samples_to_html',
    'percentile_normalize',
    'draw_mask_contour',
    'image_to_base64',
    # Configuration
    'DEFAULT_CONFIG',
    'DEFAULT_PATHS',
    'load_config',
    'save_config',
    'create_run_config',
    'get_pixel_size',
    'get_normalization_percentiles',
    'get_default_path',
    'get_output_dir',
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
    # UID parsing and migration
    'parse_uid',
    'migrate_uid_format',
    'is_spatial_uid',
    # Coordinate validation
    'CoordinateValidationError',
    'validate_xy_coordinates',
    'validate_array_indices',
    'validate_bbox_xyxy',
    # Coordinate labeling helpers
    'create_coordinate_dict',
    'format_coordinates_for_export',
    'convert_detections_to_spatial_uids',
    # CZI loading
    'CZILoader',
    'get_loader',
    'clear_cache',
    'get_cached_paths',
    # Detection pipeline
    'DetectionPipeline',
    'create_simple_detector',
    # Feature extraction
    'extract_resnet_features_batch',
    'extract_resnet_features_single',
    'preprocess_crop_for_resnet',
    'create_resnet_transform',
    # Logging
    'get_logger',
    'setup_logging',
    'log_parameters',
    'log_processing_start',
    'log_processing_end',
    'ProcessingTimer',
    # Batch processing
    'SlideInfo',
    'BatchProcessor',
    'BatchResult',
    'collect_slides',
    'create_batch_summary_html',
]

# Add schemas to __all__ if available
if _HAS_SCHEMAS:
    __all__.extend([
        'Detection',
        'DetectionFile',
        'Config',
        'Annotations',
        'validate_detection_file',
        'validate_config_file',
        'validate_annotations_file',
        'infer_and_validate',
    ])
