"""
Unified configuration module for cell detection pipelines.

Provides centralized defaults and config file loading/saving for:
- MK (Megakaryocytes)
- HSPC (Hematopoietic Stem/Progenitor Cells)
- NMJ (Neuromuscular Junctions)
- Vessel (Blood Vessels)

Usage:
    from shared.config import load_config, save_config, DEFAULT_CONFIG

    # Load config with defaults
    config = load_config('/path/to/experiment')

    # Override defaults
    config = load_config('/path/to/experiment', cell_type='nmj')

Environment Variables:
    SEGMENTATION_OUTPUT_DIR: Default output directory
    NMJ_OUTPUT_DIR: NMJ-specific output directory
    MK_OUTPUT_DIR: MK/HSPC-specific output directory
    NMJ_MODEL_PATH: Path to NMJ classifier model
    SEGMENTATION_DATA_DIR: Default input data directory
"""

import copy
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypedDict, Tuple

import numpy as np


from segmentation.utils.json_utils import NumpyEncoder as _NumpyEncoder


# =============================================================================
# CONFIGURATION TYPE DEFINITIONS
# =============================================================================

class BatchSizeConfig(TypedDict):
    """
    Configuration for batch processing sizes.

    Attributes:
        resnet_feature_extraction: Number of cells per GPU batch for ResNet features.
            Valid range: 1-128. Larger values use more GPU memory but are faster.
        sam2_embedding: Number of tile images SAM2 processes at once.
            Valid range: 1-4. Usually 1 due to high memory requirements.
        gc_interval_tiles: Run garbage collection every N tiles in sequential mode.
            Valid range: 1-100. Lower values reduce memory but slow processing.
    """
    resnet_feature_extraction: int
    sam2_embedding: int
    gc_interval_tiles: int


class MemoryConfig(TypedDict):
    """
    Configuration for memory thresholds.

    Attributes:
        min_ram_gb: Minimum RAM (GB) required to start processing.
            Valid range: 4.0-64.0. Below this, processing will abort.
        mem_per_worker_small_tile: GB per worker for tiles < 4096 pixels.
            Valid range: 4.0-32.0. Used to calculate safe worker count.
        mem_per_worker_large_tile: GB per worker for tiles >= 4096 pixels.
            Valid range: 8.0-64.0. Large tiles need more memory per worker.
        min_gpu_gb: Minimum GPU memory (GB) for SAM2 + ResNet models.
            Valid range: 4.0-24.0. Required for CUDA-based processing.
    """
    min_ram_gb: float
    mem_per_worker_small_tile: float
    mem_per_worker_large_tile: float
    min_gpu_gb: float


class PixelSizeConfig(TypedDict, total=False):
    """
    Pixel size configuration per cell type (micrometers per pixel).

    All values should be in range 0.05-1.0 µm/px.
    Typical values: 0.1725 (NMJ high-res), 0.22 (standard Axioscan).
    """
    mk: float
    cell: float
    nmj: float
    vessel: float
    mesothelium: float
    islet: float
    default: float


class NormalizationPercentilesConfig(TypedDict, total=False):
    """
    Percentile normalization ranges per cell type.

    Each value is a list of [low_percentile, high_percentile].
    Valid ranges: low 0-50, high 50-100. Low must be less than high.
    """
    mk: List[float]
    cell: List[float]
    nmj: List[float]
    vessel: List[float]
    mesothelium: List[float]
    islet: List[float]
    default: List[float]


class ProcessingConfig(TypedDict, total=False):
    """
    General processing configuration.

    Attributes:
        pixel_size_um: Pixel sizes per cell type (see PixelSizeConfig).
        normalization_percentiles: Percentile ranges per cell type.
        contour_color: RGB color for mask contours [R, G, B], each 0-255.
        contour_thickness: Line thickness for contours. Valid range: 1-10.
        samples_per_page: Samples per HTML page. Valid range: 50-1000.
        html_theme: Theme for HTML export. Valid values: 'dark', 'light'.
        tile_size: Tile dimensions in pixels. Valid range: 1000-8192.
        sample_fraction: Fraction of tiles to process. Valid range: 0.01-1.0.
        default_channel: Default channel index. Valid range: 0-10.
        calibration_samples: Samples for calibration. Valid range: 10-500.
        num_workers: Number of parallel workers. Valid range: 1-32.
        serve_port: Port for HTTP server. Valid range: 1024-65535.
    """
    pixel_size_um: PixelSizeConfig
    normalization_percentiles: NormalizationPercentilesConfig
    contour_color: List[int]
    contour_thickness: int
    samples_per_page: int
    html_theme: str
    tile_size: int
    sample_fraction: float
    default_channel: int
    calibration_samples: int
    num_workers: int
    serve_port: int


# Validation constraints for each config type
_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    "batch_sizes": {
        "resnet_feature_extraction": {"min": 1, "max": 128, "type": int},
        "sam2_embedding": {"min": 1, "max": 4, "type": int},
        "gc_interval_tiles": {"min": 1, "max": 100, "type": int},
    },
    "memory_thresholds": {
        "min_ram_gb": {"min": 4.0, "max": 64.0, "type": float},
        "mem_per_worker_small_tile": {"min": 4.0, "max": 32.0, "type": float},
        "mem_per_worker_large_tile": {"min": 8.0, "max": 64.0, "type": float},
        "min_gpu_gb": {"min": 4.0, "max": 24.0, "type": float},
    },
    "processing": {
        "contour_thickness": {"min": 1, "max": 10, "type": int},
        "samples_per_page": {"min": 50, "max": 1000, "type": int},
        "tile_size": {"min": 1000, "max": 8192, "type": int},
        "sample_fraction": {"min": 0.01, "max": 1.0, "type": float},
        "default_channel": {"min": 0, "max": 10, "type": int},
        "calibration_samples": {"min": 10, "max": 500, "type": int},
        "num_workers": {"min": 1, "max": 32, "type": int},
        "serve_port": {"min": 1024, "max": 65535, "type": int},
    },
    "pixel_size": {
        "_all_keys": {"min": 0.05, "max": 1.0, "type": float},
    },
    "percentiles": {
        "low": {"min": 0, "max": 50, "type": (int, float)},
        "high": {"min": 50, "max": 100, "type": (int, float)},
    },
}


# Environment-based paths (with sensible defaults)
DEFAULT_PATHS = {
    "output_dir": os.getenv("SEGMENTATION_OUTPUT_DIR", str(Path.home() / "segmentation_output")),
    "nmj_output_dir": os.getenv("NMJ_OUTPUT_DIR", str(Path.home() / "nmj_output")),
    "mk_output_dir": os.getenv("MK_OUTPUT_DIR", str(Path.home() / "xldvp_seg_output")),
    "nmj_model_path": os.getenv("NMJ_MODEL_PATH", str(Path.home() / "nmj_output" / "nmj_classifier.pth")),
    "data_dir": os.getenv("SEGMENTATION_DATA_DIR", "/mnt/x/01_Users/EdRo_axioscan"),
}


def get_default_path(key: str) -> str:
    """
    Get a default path from environment or fallback.

    Args:
        key: Path key name (e.g., 'output_dir', 'data_dir')

    Returns:
        Path string, empty string if key not found
    """
    return DEFAULT_PATHS.get(key, "")


def get_output_dir(cell_type: str) -> Path:
    """
    Get the default output directory for a cell type.

    Args:
        cell_type: Type of cell being processed (nmj, mk, cell, vessel, mesothelium)

    Returns:
        Path object to the appropriate output directory
    """
    if cell_type == "nmj":
        return Path(DEFAULT_PATHS["nmj_output_dir"])
    elif cell_type in ("mk", "cell"):
        return Path(DEFAULT_PATHS["mk_output_dir"])
    elif cell_type == "islet":
        return Path(os.getenv("ISLET_OUTPUT_DIR", str(Path.home() / "islet_output")))
    else:
        return Path(DEFAULT_PATHS["output_dir"])


# Default configuration values
DEFAULT_CONFIG = {
    # Pixel sizes (micrometers per pixel) by cell type
    "pixel_size_um": {
        "mk": 0.22,
        "cell": 0.22,
        "nmj": 0.1725,
        "vessel": 0.22,
        "mesothelium": 0.22,
        "islet": 0.22,
        "default": 0.22
    },

    # Percentile normalization ranges
    "normalization_percentiles": {
        "mk": [5, 95],
        "cell": [5, 95],
        "nmj": [1, 99.5],  # Wider range for NMJ images
        "vessel": [5, 95],
        "mesothelium": [5, 95],
        "islet": [1, 99.5],
        "default": [5, 95]
    },

    # Contour visualization settings
    "contour_color": [50, 255, 50],  # Bright green RGB
    "contour_thickness": 4,

    # HTML export settings
    "samples_per_page": 300,
    "html_theme": "dark",

    # Processing settings
    "tile_size": 3000,
    "sample_fraction": 0.10,
    "default_channel": 0,
    "calibration_samples": 50,
    "num_workers": 4,

    # Server settings
    "serve_port": 8081,
}

# Cell-type-specific detection parameters
DETECTION_DEFAULTS = {
    "nmj": {
        "channel": 1,
        "intensity_percentile": 98,
        "min_area_px": 150,
        "min_skeleton_length": 30,
        "max_solidity": 0.85,  # Branched structures have low solidity
        "classifier_threshold": 0.75,
        "min_area_um2": 25.0,  # Minimum NMJ area in µm²
    },
    "mk": {
        "channel": 0,
        "min_area_um2": 200,
        "max_area_um2": 2000,
        "min_area_px": 1000,  # Approximate for 0.22 µm/px
        "max_area_px": 100000,
    },
    "cell": {
        "channel": 0,
        "min_area_px": 50,
        "max_area_px": 500,
    },
    "vessel": {
        "channel": 0,
        "min_diameter_um": 10,
        "max_diameter_um": 1000,
        "min_wall_thickness_um": 2,
        "max_aspect_ratio": 4.0,
        "min_circularity": 0.3,
        "min_ring_completeness": 0.5,
        "cd31_channel": None,  # Optional CD31 validation channel
    },
    "mesothelium": {
        "channel": 0,
        "target_chunk_area_um2": 1500,
        "min_ribbon_width_um": 5,
        "max_ribbon_width_um": 30,
        "min_fragment_area_um2": 1500,
        "add_fiducials": True,
    },
    "islet": {
        "channel": 1,  # AF633 membrane marker
        "membrane_channel": 1,  # AF633
        "nuclear_channel": 4,  # DAPI
        "min_area_um2": 30,
        "max_area_um2": 500,
    },
}


def get_detection_defaults(cell_type: str) -> Dict[str, Any]:
    """
    Get default detection parameters for a cell type.

    Args:
        cell_type: Type of cell being processed (nmj, mk, cell, vessel, mesothelium)

    Returns:
        Dict of detection parameters, empty dict if cell_type not found
    """
    return DETECTION_DEFAULTS.get(cell_type, {}).copy()


def get_pixel_size(config: Dict[str, Any], cell_type: str) -> float:
    """
    Get pixel size for a cell type from config.

    Args:
        config: Configuration dictionary
        cell_type: Type of cell being processed

    Returns:
        Pixel size in micrometers per pixel
    """
    pixel_sizes = config.get("pixel_size_um", DEFAULT_CONFIG["pixel_size_um"])
    if isinstance(pixel_sizes, dict):
        return pixel_sizes.get(cell_type, pixel_sizes.get("default", 0.22))
    return pixel_sizes


def get_normalization_percentiles(config: Dict[str, Any], cell_type: str) -> Tuple[float, float]:
    """
    Get normalization percentiles for a cell type from config.

    Args:
        config: Configuration dictionary
        cell_type: Type of cell being processed

    Returns:
        Tuple of (low_percentile, high_percentile)
    """
    percentiles = config.get("normalization_percentiles",
                            DEFAULT_CONFIG["normalization_percentiles"])
    if isinstance(percentiles, dict):
        p = percentiles.get(cell_type, percentiles.get("default", [5, 95]))
    else:
        p = percentiles
    return tuple(p)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """
    Recursively merge override dict into base dict (in-place).

    For nested dicts, merges keys rather than replacing the entire dict.
    For all other types (including lists), override values are deep-copied
    to prevent shared mutable references between base and override.

    Args:
        base: Base dictionary to merge into (modified in-place)
        override: Dictionary with values to overlay on base
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def load_config(
    experiment_dir: Union[str, Path],
    cell_type: Optional[str] = None,
    config_filename: str = "config.json"
) -> Dict[str, Any]:
    """
    Load configuration from experiment directory.

    Merges with DEFAULT_CONFIG, so missing values use defaults.

    Args:
        experiment_dir: Path to experiment output directory
        cell_type: Optional cell type to include in config
        config_filename: Name of config file (default: config.json)

    Returns:
        Dict with merged configuration
    """
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / config_filename

    # Start with defaults (deep copy so nested dicts are independent)
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            # Deep merge file config over defaults (preserves nested dict keys)
            _deep_merge(config, file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    # Add cell type if specified
    if cell_type:
        config["cell_type"] = cell_type

    return config


def save_config(
    experiment_dir: Union[str, Path],
    config: Dict[str, Any],
    config_filename: str = "config.json"
) -> Path:
    """
    Save configuration to experiment directory.

    Args:
        experiment_dir: Path to experiment output directory
        config: Configuration dict to save
        config_filename: Name of config file (default: config.json)

    Returns:
        Path to saved config file
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / config_filename

    with open(config_path, 'w') as f:
        json.dump(config, f, cls=_NumpyEncoder)

    return config_path


def create_run_config(
    experiment_name: str,
    cell_type: str,
    slide_name: Optional[str] = None,
    channel: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a configuration dict for a processing run.

    Args:
        experiment_name: Name of the experiment (used for localStorage keys)
        cell_type: Type of cell being processed (mk, cell, nmj, vessel)
        slide_name: Optional slide name
        channel: Channel to process
        **kwargs: Additional config overrides

    Returns:
        Configuration dict ready to save
    """
    config = {
        "experiment_name": experiment_name,
        "cell_type": cell_type,
        "channel": channel,
        "pixel_size_um": get_pixel_size(DEFAULT_CONFIG, cell_type),
        "normalization_percentiles": list(get_normalization_percentiles(DEFAULT_CONFIG, cell_type)),
        "contour_color": DEFAULT_CONFIG["contour_color"],
        "contour_thickness": DEFAULT_CONFIG["contour_thickness"],
        "samples_per_page": DEFAULT_CONFIG["samples_per_page"],
        "tile_size": DEFAULT_CONFIG["tile_size"],
    }

    if slide_name:
        config["slide_name"] = slide_name

    # Apply any overrides
    config.update(kwargs)

    # Validate critical keys
    if 'pixel_size_um' in kwargs:
        ps = kwargs['pixel_size_um']
        if ps is not None and (not isinstance(ps, (int, float)) or ps <= 0):
            raise ValueError(f"Invalid pixel_size_um: {ps}")

    return config


# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

BATCH_SIZES = {
    "resnet_feature_extraction": 32,  # Cells per GPU batch for ResNet features
    "sam2_embedding": 1,  # SAM2 processes one tile image at a time
    "gc_interval_tiles": 10,  # Run garbage collection every N tiles in sequential mode
}


def get_batch_size(operation: str) -> int:
    """
    Get batch size for a specific operation.

    Args:
        operation: Name of the operation (e.g., 'resnet_feature_extraction', 'sam2_embedding')

    Returns:
        Batch size for the operation, defaults to 1 if not found
    """
    return BATCH_SIZES.get(operation, 1)


# =============================================================================
# MEMORY MANAGEMENT CONFIGURATION
# =============================================================================

MEMORY_THRESHOLDS = {
    "min_ram_gb": 8.0,  # Minimum RAM to start processing
    "mem_per_worker_small_tile": 10.0,  # GB per worker for tile < 4096
    "mem_per_worker_large_tile": 15.0,  # GB per worker for tile >= 4096
    "min_gpu_gb": 6.0,  # Minimum GPU memory for SAM2 + ResNet
}


def get_memory_threshold(key: str) -> float:
    """
    Get memory threshold value.

    Args:
        key: Threshold name (e.g., 'min_ram_gb', 'min_gpu_gb', 'mem_per_worker_small_tile')

    Returns:
        Threshold value in GB, defaults to 0.0 if not found
    """
    return MEMORY_THRESHOLDS.get(key, 0.0)


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def _validate_range(
    value: Union[int, float],
    key: str,
    min_val: Union[int, float],
    max_val: Union[int, float],
    expected_type: Union[type, Tuple[type, ...]]
) -> List[str]:
    """
    Validate a single value is within expected range and type.

    Args:
        value: The value to validate
        key: Name of the config key (for error messages)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        expected_type: Expected type (int, float, or tuple of types)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Type check (allow int for float fields)
    if expected_type == float:
        if not isinstance(value, (int, float)):
            errors.append(f"{key}: expected numeric type, got {type(value).__name__}")
            return errors
    elif isinstance(expected_type, tuple):
        if not isinstance(value, expected_type):
            type_names = "/".join(t.__name__ for t in expected_type)
            errors.append(f"{key}: expected {type_names}, got {type(value).__name__}")
            return errors
    else:
        if not isinstance(value, expected_type):
            errors.append(f"{key}: expected {expected_type.__name__}, got {type(value).__name__}")
            return errors

    # Range check
    if value < min_val or value > max_val:
        errors.append(f"{key}: value {value} out of range [{min_val}, {max_val}]")

    return errors


def _validate_rgb_color(
    color: Any,
    key: str = "contour_color"
) -> List[str]:
    """
    Validate an RGB color list.

    Args:
        color: List of [R, G, B] values
        key: Name of the config key

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    if not isinstance(color, list):
        errors.append(f"{key}: expected list, got {type(color).__name__}")
        return errors
    if len(color) != 3:
        errors.append(f"{key}: expected 3 values [R, G, B], got {len(color)}")
        return errors
    for i, val in enumerate(color):
        if not isinstance(val, int):
            errors.append(f"{key}[{i}]: expected int, got {type(val).__name__}")
        elif val < 0 or val > 255:
            errors.append(f"{key}[{i}]: value {val} out of range [0, 255]")
    return errors


def _validate_percentiles(
    percentiles: Any,
    key: str = "normalization_percentiles"
) -> List[str]:
    """
    Validate normalization percentiles configuration.

    Args:
        percentiles: Dict mapping cell types to [low, high] percentile pairs
        key: Name of the config key

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    rules = _VALIDATION_RULES["percentiles"]

    for cell_type, values in percentiles.items():
        if not isinstance(values, list) or len(values) != 2:
            errors.append(f"{key}.{cell_type}: expected [low, high] list, got {values}")
            continue

        low, high = values
        low_rule = rules["low"]
        high_rule = rules["high"]

        errors.extend(_validate_range(
            low, f"{key}.{cell_type}[0]",
            low_rule["min"], low_rule["max"], low_rule["type"]
        ))
        errors.extend(_validate_range(
            high, f"{key}.{cell_type}[1]",
            high_rule["min"], high_rule["max"], high_rule["type"]
        ))

        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if low >= high:
                errors.append(f"{key}.{cell_type}: low ({low}) must be less than high ({high})")

    return errors


def _validate_pixel_sizes(
    pixel_sizes: Any,
    key: str = "pixel_size_um"
) -> List[str]:
    """
    Validate pixel size configuration.

    Args:
        pixel_sizes: Dict mapping cell types to pixel sizes in µm/px
        key: Name of the config key

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    rule = _VALIDATION_RULES["pixel_size"]["_all_keys"]

    for cell_type, value in pixel_sizes.items():
        errors.extend(_validate_range(
            value, f"{key}.{cell_type}",
            rule["min"], rule["max"], rule["type"]
        ))

    return errors


def validate_config(
    config: Optional[Dict[str, Any]] = None,
    batch_sizes: Optional[Dict[str, int]] = None,
    memory_thresholds: Optional[Dict[str, float]] = None,
    raise_on_error: bool = False
) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate configuration dictionaries against expected types and ranges.

    This function validates one or more configuration dictionaries:
    - Processing config (DEFAULT_CONFIG structure)
    - Batch size config (BATCH_SIZES structure)
    - Memory threshold config (MEMORY_THRESHOLDS structure)

    Args:
        config: Main processing config dict (like DEFAULT_CONFIG).
            If None, validates the global DEFAULT_CONFIG.
        batch_sizes: Batch processing config dict (like BATCH_SIZES).
            If None, validates the global BATCH_SIZES.
        memory_thresholds: Memory threshold config dict (like MEMORY_THRESHOLDS).
            If None, validates the global MEMORY_THRESHOLDS.
        raise_on_error: If True, raises ConfigValidationError on first error.
            If False (default), returns dict with all errors.

    Returns:
        Dict with validation results:
            - 'valid': bool, True if all validations passed
            - 'errors': List of error message strings
            - 'warnings': List of warning message strings

    Raises:
        ConfigValidationError: If raise_on_error=True and validation fails

    Example:
        >>> result = validate_config()
        >>> if not result['valid']:
        ...     print("Errors:", result['errors'])

        >>> # Validate custom config
        >>> my_config = {"tile_size": 500}  # Too small!
        >>> result = validate_config(config=my_config)
        >>> # Returns: {'valid': False, 'errors': ['tile_size: value 500 out of range [1000, 8192]'], ...}
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Use defaults if not provided
    if config is None:
        config = DEFAULT_CONFIG
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES
    if memory_thresholds is None:
        memory_thresholds = MEMORY_THRESHOLDS

    # Validate batch sizes
    batch_rules = _VALIDATION_RULES["batch_sizes"]
    for key, rule in batch_rules.items():
        if key in batch_sizes:
            errors.extend(_validate_range(
                batch_sizes[key], f"batch_sizes.{key}",
                rule["min"], rule["max"], rule["type"]
            ))

    # Validate memory thresholds
    mem_rules = _VALIDATION_RULES["memory_thresholds"]
    for key, rule in mem_rules.items():
        if key in memory_thresholds:
            errors.extend(_validate_range(
                memory_thresholds[key], f"memory_thresholds.{key}",
                rule["min"], rule["max"], rule["type"]
            ))

    # Validate processing config
    proc_rules = _VALIDATION_RULES["processing"]
    for key, rule in proc_rules.items():
        if key in config:
            errors.extend(_validate_range(
                config[key], key,
                rule["min"], rule["max"], rule["type"]
            ))

    # Validate pixel sizes
    if "pixel_size_um" in config:
        pixel_sizes = config["pixel_size_um"]
        if isinstance(pixel_sizes, dict):
            errors.extend(_validate_pixel_sizes(pixel_sizes))
        elif isinstance(pixel_sizes, (int, float)):
            rule = _VALIDATION_RULES["pixel_size"]["_all_keys"]
            errors.extend(_validate_range(
                pixel_sizes, "pixel_size_um",
                rule["min"], rule["max"], rule["type"]
            ))

    # Validate normalization percentiles
    if "normalization_percentiles" in config:
        percentiles = config["normalization_percentiles"]
        if isinstance(percentiles, dict):
            errors.extend(_validate_percentiles(percentiles))

    # Validate contour color
    if "contour_color" in config:
        errors.extend(_validate_rgb_color(config["contour_color"]))

    # Validate html_theme
    if "html_theme" in config:
        valid_themes = ["dark", "light"]
        if config["html_theme"] not in valid_themes:
            errors.append(f"html_theme: '{config['html_theme']}' not in {valid_themes}")

    # Check for logical consistency warnings
    if "mem_per_worker_small_tile" in memory_thresholds and "mem_per_worker_large_tile" in memory_thresholds:
        if memory_thresholds["mem_per_worker_small_tile"] > memory_thresholds["mem_per_worker_large_tile"]:
            warnings.append(
                "mem_per_worker_small_tile > mem_per_worker_large_tile: "
                "small tiles should require less memory"
            )

    result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    if raise_on_error and errors:
        raise ConfigValidationError(f"Configuration validation failed: {errors[0]}")

    return result


def get_config_summary(
    config: Optional[Dict[str, Any]] = None,
    batch_sizes: Optional[Dict[str, int]] = None,
    memory_thresholds: Optional[Dict[str, float]] = None,
    include_validation: bool = True
) -> str:
    """
    Generate a formatted summary of current configuration values.

    Produces a human-readable summary of all configuration settings,
    organized by category. Optionally includes validation status.

    Args:
        config: Main processing config dict. If None, uses DEFAULT_CONFIG.
        batch_sizes: Batch sizes dict. If None, uses BATCH_SIZES.
        memory_thresholds: Memory thresholds dict. If None, uses MEMORY_THRESHOLDS.
        include_validation: If True, appends validation status to summary.

    Returns:
        Formatted multi-line string with configuration summary.

    Example:
        >>> print(get_config_summary())
        ========================================
        Configuration Summary
        ========================================

        Processing Settings:
          tile_size: 3000 px
          sample_fraction: 0.10 (10.0%)
          num_workers: 4
          ...
    """
    # Use defaults if not provided
    if config is None:
        config = DEFAULT_CONFIG
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES
    if memory_thresholds is None:
        memory_thresholds = MEMORY_THRESHOLDS

    lines = [
        "=" * 50,
        "Configuration Summary",
        "=" * 50,
        "",
    ]

    # Processing settings
    lines.append("Processing Settings:")
    if "tile_size" in config:
        lines.append(f"  tile_size: {config['tile_size']} px")
    if "sample_fraction" in config:
        pct = config['sample_fraction'] * 100
        lines.append(f"  sample_fraction: {config['sample_fraction']:.2f} ({pct:.1f}%)")
    if "num_workers" in config:
        lines.append(f"  num_workers: {config['num_workers']}")
    if "default_channel" in config:
        lines.append(f"  default_channel: {config['default_channel']}")
    if "calibration_samples" in config:
        lines.append(f"  calibration_samples: {config['calibration_samples']}")
    lines.append("")

    # Display settings
    lines.append("Display Settings:")
    if "contour_color" in config:
        rgb = config['contour_color']
        lines.append(f"  contour_color: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
    if "contour_thickness" in config:
        lines.append(f"  contour_thickness: {config['contour_thickness']} px")
    if "samples_per_page" in config:
        lines.append(f"  samples_per_page: {config['samples_per_page']}")
    if "html_theme" in config:
        lines.append(f"  html_theme: {config['html_theme']}")
    if "serve_port" in config:
        lines.append(f"  serve_port: {config['serve_port']}")
    lines.append("")

    # Pixel sizes
    if "pixel_size_um" in config:
        lines.append("Pixel Sizes (um/px):")
        pixel_sizes = config["pixel_size_um"]
        if isinstance(pixel_sizes, dict):
            for cell_type, size in sorted(pixel_sizes.items()):
                lines.append(f"  {cell_type}: {size}")
        else:
            lines.append(f"  all: {pixel_sizes}")
        lines.append("")

    # Normalization percentiles
    if "normalization_percentiles" in config:
        lines.append("Normalization Percentiles [low, high]:")
        percentiles = config["normalization_percentiles"]
        if isinstance(percentiles, dict):
            for cell_type, pct in sorted(percentiles.items()):
                lines.append(f"  {cell_type}: [{pct[0]}, {pct[1]}]")
        else:
            lines.append(f"  all: {percentiles}")
        lines.append("")

    # Batch sizes
    lines.append("Batch Sizes:")
    for key, value in sorted(batch_sizes.items()):
        lines.append(f"  {key}: {value}")
    lines.append("")

    # Memory thresholds
    lines.append("Memory Thresholds:")
    for key, value in sorted(memory_thresholds.items()):
        lines.append(f"  {key}: {value} GB")
    lines.append("")

    # Validation status
    if include_validation:
        result = validate_config(config, batch_sizes, memory_thresholds)
        lines.append("-" * 50)
        if result["valid"]:
            lines.append("Validation: PASSED")
        else:
            lines.append("Validation: FAILED")
            for error in result["errors"]:
                lines.append(f"  ERROR: {error}")
        if result["warnings"]:
            for warning in result["warnings"]:
                lines.append(f"  WARNING: {warning}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# EXTRACTED MAGIC NUMBERS (detection pipeline constants)
# =============================================================================

# Feature extraction dimensions — FULL PIPELINE values.
#
# These represent the total features produced by the complete detection pipeline
# (not just a single extraction function). For the per-function "single-pass"
# constants, see segmentation/utils/feature_extraction.py.
#
# Morphological breakdown (78 total for 3-channel NMJ pipeline):
#   - 22 base features from extract_morphological_features()
#     (area, perimeter, circularity, solidity, aspect_ratio, extent,
#      equiv_diameter, RGB+gray+HSV stats, texture features)
#   - ~5 NMJ-specific features (skeleton_length, eccentricity, mean_intensity,
#     solidity override, bbox)
#   - ~45 multi-channel stats (15 per channel x 3 channels via MultiChannelFeatureMixin)
#   - ~6 inter-channel ratios + specificity metrics
#   (Exact count varies by cell type and number of channels; 78 is the empirical
#    value from the 3-channel NMJ classifier training.)
#
# Full-pipeline feature dimensions.
# These are the TOTAL dimensions produced by the detection pipeline (masked + context
# crops combined). The per-function "single-pass" constants live in
# segmentation/utils/feature_extraction.py:
#   feature_extraction.MORPHOLOGICAL_FEATURE_COUNT = 22  (base only; 78 after NMJ + multi-ch)
#   feature_extraction.SAM2_EMBEDDING_DIM = 256          (same — no doubling for SAM2)
#   feature_extraction.RESNET50_FEATURE_DIM = 2048       (single pass; doubled here for masked+ctx)
#   (DINOv2 ViT-L/14 produces 1024 per pass; doubled here for masked+ctx)
MORPHOLOGICAL_FEATURES_COUNT = 78      # Full-pipeline morphological features (see breakdown above)
SAM2_EMBEDDING_DIMENSION = 256         # SAM2 256D embedding vectors (= feature_extraction.SAM2_EMBEDDING_DIM)
RESNET_EMBEDDING_DIMENSION = 4096      # ResNet50 2x2048D (= 2 * feature_extraction.RESNET50_FEATURE_DIM)
DINOV2_EMBEDDING_DIMENSION = 2048      # DINOv2-L 2x1024D (masked + context)
TOTAL_FEATURES_PER_CELL = 6478         # Total: 78 + 256 + 4096 + 2048

# DEPRECATED: Do not use. Always read pixel_size from CZI metadata via loader.get_pixel_size().
# Retained only so old test assertions don't break. All production code must use CZI metadata.
_LEGACY_PIXEL_SIZE_UM = 0.1725

# Batch processing sizes (lines 1119, 1128, 1138, 1185, 1195)
RESNET_INFERENCE_BATCH_SIZE = 16       # Default batch size for ResNet inference

# Processing parameters (lines 1954, 1999, 3074)
CPU_UTILIZATION_FRACTION = 0.8         # Use 80% of available CPU cores


# =============================================================================
# HELPER FUNCTIONS FOR ACCESSING MAGIC NUMBERS
# =============================================================================

def get_feature_dimensions() -> Dict[str, int]:
    """
    Get all full-pipeline feature dimension constants.

    Returns:
        Dict with keys: morphological (78), sam2_embedding (256),
        resnet_embedding (4096), dinov2_embedding (2048), total (6478)
    """
    return {
        "morphological": MORPHOLOGICAL_FEATURES_COUNT,
        "sam2_embedding": SAM2_EMBEDDING_DIMENSION,
        "resnet_embedding": RESNET_EMBEDDING_DIMENSION,
        "dinov2_embedding": DINOV2_EMBEDDING_DIMENSION,
        "total": TOTAL_FEATURES_PER_CELL,
    }


def get_cpu_worker_count(total_cores: Optional[int] = None) -> int:
    """
    Calculate safe number of CPU workers based on available cores.

    Args:
        total_cores: Total CPU cores available. If None, uses os.cpu_count()

    Returns:
        Number of workers to use (80% of total cores)
    """
    if total_cores is None:
        total_cores = os.cpu_count() or 1
    return max(1, int(total_cores * CPU_UTILIZATION_FRACTION))
