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
    VESSEL_FEATURE_COUNT,
)

from .vessel_features import (
    extract_vessel_features,
    extract_vessel_features_batch,
    extract_all_vessel_features_multichannel,
    extract_multichannel_intensity_features,
    compute_channel_ratios,
    vessel_features_to_vector,
    get_vessel_feature_importance,
    VESSEL_FEATURE_NAMES,
    VESSEL_FEATURE_COUNT as VESSEL_SPECIFIC_FEATURE_COUNT,
    RING_WALL_FEATURE_NAMES,
    DIAMETER_SIZE_FEATURE_NAMES,
    SHAPE_FEATURE_NAMES,
    INTENSITY_GRADIENT_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    DERIVED_FEATURE_NAMES,
    DEFAULT_CHANNEL_NAMES,
    MULTICHANNEL_WALL_FEATURES,
    MULTICHANNEL_LUMEN_FEATURES,
    CROSS_CHANNEL_RATIO_NAMES,
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
    'VESSEL_FEATURE_COUNT',
    # Vessel-Specific Features
    'extract_vessel_features',
    'extract_vessel_features_batch',
    'extract_all_vessel_features_multichannel',
    'extract_multichannel_intensity_features',
    'compute_channel_ratios',
    'vessel_features_to_vector',
    'get_vessel_feature_importance',
    'VESSEL_FEATURE_NAMES',
    'VESSEL_SPECIFIC_FEATURE_COUNT',
    'RING_WALL_FEATURE_NAMES',
    'DIAMETER_SIZE_FEATURE_NAMES',
    'SHAPE_FEATURE_NAMES',
    'INTENSITY_GRADIENT_FEATURE_NAMES',
    'CONTEXT_FEATURE_NAMES',
    'DERIVED_FEATURE_NAMES',
    'DEFAULT_CHANNEL_NAMES',
    'MULTICHANNEL_WALL_FEATURES',
    'MULTICHANNEL_LUMEN_FEATURES',
    'CROSS_CHANNEL_RATIO_NAMES',
]
