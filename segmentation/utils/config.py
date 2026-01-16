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

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


# Environment-based paths (with sensible defaults)
DEFAULT_PATHS = {
    "output_dir": os.getenv("SEGMENTATION_OUTPUT_DIR", "/home/dude/segmentation_output"),
    "nmj_output_dir": os.getenv("NMJ_OUTPUT_DIR", "/home/dude/nmj_output"),
    "mk_output_dir": os.getenv("MK_OUTPUT_DIR", "/home/dude/xldvp_seg_output"),
    "nmj_model_path": os.getenv("NMJ_MODEL_PATH", "/home/dude/nmj_output/nmj_classifier.pth"),
    "data_dir": os.getenv("SEGMENTATION_DATA_DIR", "/mnt/x/01_Users/EdRo_axioscan"),
}


def get_default_path(key: str) -> str:
    """Get a default path from environment or fallback."""
    return DEFAULT_PATHS.get(key, "")


def get_output_dir(cell_type: str) -> Path:
    """Get the default output directory for a cell type."""
    if cell_type == "nmj":
        return Path(DEFAULT_PATHS["nmj_output_dir"])
    elif cell_type in ("mk", "cell"):
        return Path(DEFAULT_PATHS["mk_output_dir"])
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
        "default": 0.22
    },

    # Percentile normalization ranges
    "normalization_percentiles": {
        "mk": [5, 95],
        "cell": [5, 95],
        "nmj": [1, 99.5],  # Wider range for NMJ images
        "vessel": [5, 95],
        "mesothelium": [5, 95],
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
        "intensity_percentile": 99,
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
}


def get_detection_defaults(cell_type: str) -> Dict[str, Any]:
    """Get default detection parameters for a cell type."""
    return DETECTION_DEFAULTS.get(cell_type, {}).copy()


def get_pixel_size(config: Dict[str, Any], cell_type: str) -> float:
    """Get pixel size for a cell type from config."""
    pixel_sizes = config.get("pixel_size_um", DEFAULT_CONFIG["pixel_size_um"])
    if isinstance(pixel_sizes, dict):
        return pixel_sizes.get(cell_type, pixel_sizes.get("default", 0.22))
    return pixel_sizes


def get_normalization_percentiles(config: Dict[str, Any], cell_type: str) -> tuple:
    """Get normalization percentiles for a cell type from config."""
    percentiles = config.get("normalization_percentiles",
                            DEFAULT_CONFIG["normalization_percentiles"])
    if isinstance(percentiles, dict):
        p = percentiles.get(cell_type, percentiles.get("default", [5, 95]))
    else:
        p = percentiles
    return tuple(p)


def load_config(
    experiment_dir: str | Path,
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

    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            # Merge file config over defaults
            config.update(file_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    # Add cell type if specified
    if cell_type:
        config["cell_type"] = cell_type

    return config


def save_config(
    experiment_dir: str | Path,
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
        json.dump(config, f, indent=2)

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

    return config
