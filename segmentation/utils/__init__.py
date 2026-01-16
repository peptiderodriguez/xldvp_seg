"""
Utility modules for the segmentation pipeline.

Provides:
- Configuration management
- Logging utilities
- JSON schema validation (requires pydantic)
- Feature extraction utilities
"""

from .config import (
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    DETECTION_DEFAULTS,
    load_config,
    save_config,
    create_run_config,
    get_pixel_size,
    get_normalization_percentiles,
    get_default_path,
    get_output_dir,
    get_detection_defaults,
)

from .logging import (
    get_logger,
    setup_logging,
    log_parameters,
    log_processing_start,
    log_processing_end,
    ProcessingTimer,
)

from .feature_extraction import (
    extract_resnet_features_batch,
    extract_resnet_features_single,
    preprocess_crop_for_resnet,
    create_resnet_transform,
    extract_morphological_features,
    compute_hsv_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
    MORPHOLOGICAL_FEATURE_COUNT,
)

# Schemas require pydantic - import separately if needed
# from segmentation.utils.schemas import Detection, DetectionFile, Config, Annotations

__all__ = [
    # Config
    'DEFAULT_CONFIG',
    'DEFAULT_PATHS',
    'DETECTION_DEFAULTS',
    'load_config',
    'save_config',
    'create_run_config',
    'get_pixel_size',
    'get_normalization_percentiles',
    'get_default_path',
    'get_output_dir',
    'get_detection_defaults',
    # Logging
    'get_logger',
    'setup_logging',
    'log_parameters',
    'log_processing_start',
    'log_processing_end',
    'ProcessingTimer',
    # Feature Extraction
    'extract_resnet_features_batch',
    'extract_resnet_features_single',
    'preprocess_crop_for_resnet',
    'create_resnet_transform',
    'extract_morphological_features',
    'compute_hsv_features',
    'SAM2_EMBEDDING_DIM',
    'RESNET50_FEATURE_DIM',
    'MORPHOLOGICAL_FEATURE_COUNT',
]
