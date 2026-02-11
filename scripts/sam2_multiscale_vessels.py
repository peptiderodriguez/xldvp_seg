#!/usr/bin/env python3
"""
SAM2-based vessel detection at 1/8 scale.

Approach:
- Load channels at 1/2 scale into RAM (4x smaller than full-res)
- Process at 1/8 scale (downsample 4x from 1/2 base)
- No diameter filters - detect all ring structures, filter in post-processing
- 4000px tiles cover 32000px at full res

Pipeline:
1. SAM2 generates mask proposals on SMA channel
2. Filter to lumens (dark inside, bright SMA wall around)
3. Watershed expand from lumen to find outer wall
4. Extract vessel measurements + save crops with contours

Output:
- vessel_detections_multiscale.json - all vessels with coordinates and measurements
- crops/ - vessel images with green (outer) and cyan (inner) contours
- index.html - viewer sorted by diameter
"""

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.transform import resize
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from scipy.ndimage import convolve
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import json
import os
import sys
import random
import gc
import time
import tempfile
import torch
from tqdm import tqdm

# Global counter for tiles skipped by Sobel pre-filter
TILES_SKIPPED_BY_SOBEL = 0

# Global dilation config (set from CLI args in main())
DILATION_MODE = 'adaptive'
SMA_MIN_WALL_INTENSITY = 30

# Pixel size in micrometers (set from CZI metadata in main(), can be overridden via CLI)
PIXEL_SIZE_UM = 0.1725  # Default fallback

# ==============================================================================
# GMM-based Adaptive Thresholding
# ==============================================================================
# More robust to staining intensity variation across the slide than fixed thresholds.
# Instead of `ratio < 1.0`, fits a 2-component GMM to separate populations.

def _make_gmm():
    """Create a consistently configured 2-component GMM."""
    return GaussianMixture(
        n_components=2,
        covariance_type='full',
        max_iter=100,
        n_init=3,
        random_state=42
    )


class GMMClassifier:
    """
    GMM-based probabilistic classifier for separating two populations.

    Instead of fixed thresholds like `ratio < 1.0`, this fits a 2-component
    Gaussian Mixture Model to the data and classifies samples based on
    their positive predictive value (probability of belonging to the
    "positive" class, which is the lower-mean component for lumen detection).
    """

    def __init__(self, min_samples=50):
        """
        Args:
            min_samples: Minimum samples needed to fit GMM. Below this,
                        falls back to fixed threshold.
        """
        self.min_samples = min_samples
        self.gmm = None
        self.positive_component = None  # Index of "positive" (lower mean) component
        self.fitted = False

    def fit(self, values: np.ndarray):
        """
        Fit 2-component GMM to the values.

        Args:
            values: 1D array of ratio values (e.g., inside/outside intensity ratios)
        """
        values = np.asarray(values).flatten()

        if len(values) < self.min_samples:
            self.fitted = False
            return self

        # Fit 2-component GMM
        self.gmm = _make_gmm()
        self.gmm.fit(values.reshape(-1, 1))

        # Identify which component has lower mean (this is "positive" for lumens,
        # since lumens should have lower intensity ratios)
        means = self.gmm.means_.flatten()
        self.positive_component = int(np.argmin(means))
        self.fitted = True

        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        """
        Get probability of each value belonging to the positive (lower) class.

        Args:
            values: 1D array of values to classify

        Returns:
            Array of probabilities (PPV) for each value
        """
        if not self.fitted:
            # Fallback: use fixed threshold, return 1.0 for values < 1.0
            return (np.asarray(values) < 1.0).astype(float)

        values = np.asarray(values).flatten().reshape(-1, 1)
        proba = self.gmm.predict_proba(values)
        return proba[:, self.positive_component]

    def classify(self, values: np.ndarray, threshold_ppv: float = 0.5) -> np.ndarray:
        """
        Classify values as positive (True) or negative (False).

        Args:
            values: 1D array of values to classify
            threshold_ppv: Probability threshold for classification (default 0.5)

        Returns:
            Boolean array where True = positive class (likely lumen)
        """
        proba = self.predict_proba(values)
        return proba >= threshold_ppv


def gmm_classify_ratios(ratios: np.ndarray, threshold_ppv: float = 0.5,
                        min_samples: int = 50, lower_is_positive: bool = True) -> np.ndarray:
    """
    GMM-based classification for intensity ratio arrays.

    Fits a 2-component GMM to separate two populations and returns boolean mask
    of samples with PPV > threshold.

    Args:
        ratios: 1D array of ratio values to classify
        threshold_ppv: Probability threshold for classification (default 0.5 for balanced)
                      Use higher values (e.g., 0.95) for high precision
        min_samples: Minimum samples to fit GMM. Below this, falls back to fixed threshold.
        lower_is_positive: If True, the lower-mean component is "positive" (for lumen detection).
                          If False, the higher-mean component is "positive" (for enrichment).

    Returns:
        Boolean array where True = positive class
    """
    ratios = np.asarray(ratios).flatten()

    if len(ratios) < min_samples:
        # Fallback to fixed threshold
        if lower_is_positive:
            return ratios < 1.0
        else:
            return ratios > 1.1

    # Fit 2-component GMM
    gmm = _make_gmm()
    gmm.fit(ratios.reshape(-1, 1))

    # Select positive component based on mean
    means = gmm.means_.flatten()
    if lower_is_positive:
        positive_component = int(np.argmin(means))
    else:
        positive_component = int(np.argmax(means))

    proba = gmm.predict_proba(ratios.reshape(-1, 1))
    return proba[:, positive_component] >= threshold_ppv


def gmm_classify_single(value: float, all_values: np.ndarray, threshold_ppv: float = 0.5,
                        min_samples: int = 50, lower_is_positive: bool = True) -> bool:
    """
    Classify a single value using GMM fitted on all observed values.

    This is useful when you need to classify individual masks but want the
    threshold to be adaptive based on the distribution of all masks in the tile.

    Args:
        value: Single value to classify
        all_values: Array of all observed values (for fitting GMM)
        threshold_ppv: Probability threshold for classification
        min_samples: Minimum samples to fit GMM
        lower_is_positive: If True, lower values are "positive"

    Returns:
        True if value is classified as positive
    """
    all_values = np.asarray(all_values).flatten()

    if len(all_values) < min_samples:
        # Fallback to fixed threshold
        if lower_is_positive:
            return value < 1.0
        else:
            return value > 1.1

    # Use GMMClassifier for consistent fitting
    classifier = GMMClassifier(min_samples=min_samples)
    classifier.fit(all_values)

    if not classifier.fitted:
        if lower_is_positive:
            return value < 1.0
        else:
            return value > 1.1

    # For lower_is_positive, GMMClassifier already uses argmin(means) for positive
    if lower_is_positive:
        proba = classifier.predict_proba(np.array([value]))
        return proba[0] >= threshold_ppv
    else:
        # Need the higher-mean component -- use 1 - default proba
        means = classifier.gmm.means_.flatten()
        higher_component = int(np.argmax(means))
        proba = classifier.gmm.predict_proba(np.array([[value]]))
        return proba[0, higher_component] >= threshold_ppv


def adaptive_diameter_ratio_filter(vessels):
    """
    Use GMM to identify vessels with outlier outer/inner diameter ratios.
    Most real vessels have ratio close to 1.2-1.8, outliers are likely false positives.

    Args:
        vessels: List of vessel dicts with outer_diameter_um and inner_diameter_um

    Returns:
        List of filtered vessels (outliers removed)
    """
    if len(vessels) == 0:
        return []

    if len(vessels) == 1:
        # Single vessel - use fixed threshold fallback
        v = vessels[0]
        inner = v.get('inner_diameter_um', 0)
        outer = v.get('outer_diameter_um', 0)
        if inner > 0 and (outer / inner) <= 2.0:
            return vessels
        return []

    # Compute ratios for all vessels
    ratios = []
    valid_indices = []
    for i, v in enumerate(vessels):
        inner = v.get('inner_diameter_um', 0)
        outer = v.get('outer_diameter_um', 0)
        if inner > 0:
            ratios.append(outer / inner)
            valid_indices.append(i)

    if len(ratios) == 0:
        return []

    ratios = np.array(ratios)

    # Use GMM to classify ratios - lower ratios are "normal" vessels
    # lower_is_positive=True because we want to keep vessels with lower diameter ratios
    is_normal = gmm_classify_ratios(
        ratios,
        threshold_ppv=0.5,
        min_samples=50,
        lower_is_positive=True
    )

    # Filter vessels
    filtered = []
    for idx, is_valid in zip(valid_indices, is_normal):
        if is_valid:
            filtered.append(vessels[idx])

    # Also include vessels where inner_diameter was 0 (couldn't compute ratio)
    # These are edge cases that should be reviewed manually
    for i, v in enumerate(vessels):
        if i not in valid_indices:
            filtered.append(v)

    print(f"  Adaptive diameter ratio filter: {len(vessels)} -> {len(filtered)} vessels")
    if len(ratios) > 0:
        print(f"    Ratio stats: min={ratios.min():.2f}, max={ratios.max():.2f}, mean={ratios.mean():.2f}")

    return filtered


def compute_adaptive_iou_threshold(iou_values):
    """
    Use Otsu's method to find optimal IoU threshold for merge decisions.

    Args:
        iou_values: Array of IoU values between vessel pairs

    Returns:
        float: Optimal IoU threshold
    """
    iou_values = np.asarray(iou_values).flatten()

    # Handle edge cases
    if len(iou_values) == 0:
        return 0.3  # Default fallback

    if len(iou_values) == 1:
        return 0.3  # Not enough data for Otsu

    # Check if all values are the same (Otsu will fail)
    if np.all(iou_values == iou_values[0]):
        return 0.3  # Default fallback

    # Check for sufficient variance
    if np.std(iou_values) < 0.01:
        return 0.3  # Not enough variance for meaningful Otsu

    try:
        # Otsu's method finds threshold that maximizes between-class variance
        threshold = threshold_otsu(iou_values)
        # Floor at 0.2 to avoid merging unrelated vessels
        return max(threshold, 0.2)
    except Exception:
        return 0.3  # Fallback on any error


# Optional GeoDataFrame support for GIS compatibility
try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Optional Cellpose support for cell segmentation within vessel ROIs
try:
    from cellpose import models as cellpose_models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

# Global Cellpose model (lazy-loaded once)
_CELLPOSE_MODEL = None

def get_cellpose_model():
    """Get or create the global CellposeSAM model (singleton pattern)."""
    global _CELLPOSE_MODEL
    if _CELLPOSE_MODEL is None and CELLPOSE_AVAILABLE:
        try:
            from cellpose.models import CellposeSAM
            print("Loading CellposeSAM model...")
            _CELLPOSE_MODEL = CellposeSAM(gpu=True)
            print("CellposeSAM model loaded.")
        except Exception as e:
            print(f"Warning: Could not load CellposeSAM: {e}")
            _CELLPOSE_MODEL = None
    return _CELLPOSE_MODEL

sys.path.insert(0, '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg')
from segmentation.io.czi_loader import CZILoader

# Configuration
CZI_PATH = "/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT_DIR = "/fs/pool/pool-mann-edwin/vessel_output/sam2_multiscale"
SAMPLE_FRACTION = 1.0  # 100% of tiles
TILE_OVERLAP = 0.5  # 50% overlap between tiles
TISSUE_VARIANCE_THRESHOLD = 50
COVERAGE_THRESHOLD = 0.90  # Min coverage to consider fine-scale complete (for cross-scale merging)

# Base scale for RAM loading (finest scale we need)
# All coarser scales will downsample from this
BASE_SCALE = 2  # 1/2 scale

# Multi-scale configuration
# Tile sizes chosen to balance memory usage and coverage
TILE_SIZE = 4000  # Default tile size
SCALES = [
    {'name': '1/64', 'scale_factor': 64, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1000},
    {'name': '1/32', 'scale_factor': 32, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1200},
    {'name': '1/16', 'scale_factor': 16, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1400},
    {'name': '1/8', 'scale_factor': 8, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1700},
    {'name': '1/4', 'scale_factor': 4, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 2000},
    {'name': '1/2', 'scale_factor': 2, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 2500},
]
# Note: 1/2 scale vessels are only kept if corroborated by coarser scales (see merge_vessels_across_scales)


def _extract_tile_from_cache(channels, base_scale, tile_x, tile_y, tile_size, channel, scale_factor):
    """Extract a tile from channel arrays at the specified scale.

    Shared logic for DownsampledChannelCache and SharedChannelCache.

    Args:
        channels: Dict mapping channel index to numpy arrays at base_scale
        base_scale: Scale factor of stored data (e.g. 2 for 1/2 scale)
        tile_x, tile_y: Tile origin in SCALED coordinates (at target scale_factor)
        tile_size: Output tile size
        channel: Channel index
        scale_factor: Target scale (must be >= base_scale)

    Returns:
        Tile data at target scale, or None if out of bounds
    """
    if scale_factor < base_scale:
        raise ValueError(f"scale_factor {scale_factor} < base_scale {base_scale}")

    if channel not in channels:
        raise ValueError(f"Channel {channel} not loaded")

    relative_scale = scale_factor // base_scale
    base_tile_size = tile_size * relative_scale

    base_x = (tile_x * scale_factor) // base_scale
    base_y = (tile_y * scale_factor) // base_scale

    data = channels[channel]
    y2 = min(base_y + base_tile_size, data.shape[0])
    x2 = min(base_x + base_tile_size, data.shape[1])

    if base_x >= data.shape[1] or base_y >= data.shape[0]:
        return None

    region = data[base_y:y2, base_x:x2]

    if region.size == 0:
        return None

    if relative_scale > 1:
        target_h = region.shape[0] // relative_scale
        target_w = region.shape[1] // relative_scale
        if target_h == 0 or target_w == 0:
            return None
        region = cv2.resize(region, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return region


class DownsampledChannelCache:
    """
    Holds channels loaded at BASE_SCALE (1/2), provides tiles at any coarser scale.

    Memory usage: ~4x smaller than full-res (for BASE_SCALE=2).
    """

    def __init__(self, loader: CZILoader, channels: list, base_scale: int = 2, strip_height: int = 5000):
        """
        Load channels at base_scale into RAM using strip-by-strip loading.

        Args:
            loader: CZILoader instance (used for reading, not for RAM storage)
            channels: List of channel indices to load
            base_scale: Scale factor for base resolution (2 = 1/2 scale)
            strip_height: Height of strips for loading (in full-res pixels)
        """
        self.base_scale = base_scale
        self.channels = {}
        self.full_res_size = loader.mosaic_size  # (width, height) at full res
        self.loader = loader  # Keep reference for strip reading

        # Size at base scale
        self.base_width = self.full_res_size[0] // base_scale
        self.base_height = self.full_res_size[1] // base_scale

        print(f"\nLoading channels at 1/{base_scale} scale ({self.base_width} x {self.base_height})...")
        print(f"  (Full res: {self.full_res_size[0]} x {self.full_res_size[1]})")

        for ch in channels:
            print(f"  Loading channel {ch}...", flush=True)
            self.channels[ch] = self._load_channel_in_strips(ch, strip_height)
            mem_gb = self.channels[ch].nbytes / (1024**3)
            print(f"    Channel {ch}: {self.channels[ch].shape}, {mem_gb:.2f} GB")

        total_gb = sum(arr.nbytes for arr in self.channels.values()) / (1024**3)
        print(f"  Total RAM: {total_gb:.2f} GB")

    def _load_channel_in_strips(self, channel: int, strip_height: int) -> np.ndarray:
        """Load a channel in strips, downsampling each strip manually."""
        # Pre-allocate output array at base scale
        channel_array = np.empty((self.base_height, self.base_width), dtype=np.uint16)

        n_strips = (self.loader.height + strip_height - 1) // strip_height

        for i in tqdm(range(n_strips), desc=f"Loading ch{channel}"):
            # Full-res strip coordinates
            y_off = i * strip_height
            h = min(strip_height, self.loader.height - y_off)

            # Read strip at FULL resolution (scale_factor=1 to avoid aicspylibczi bug)
            strip = self.loader.reader.read_mosaic(
                region=(self.loader.x_start, self.loader.y_start + y_off, self.loader.width, h),
                scale_factor=1,
                C=channel
            )
            strip = np.squeeze(strip)

            # Manually downsample to base_scale using cv2.resize
            target_h = h // self.base_scale
            target_w = self.loader.width // self.base_scale
            if target_h > 0 and target_w > 0:
                strip_downsampled = cv2.resize(strip, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                continue

            # Calculate output position at base scale
            y_out = y_off // self.base_scale

            # Handle edge case where strip might be slightly smaller
            actual_h = min(strip_downsampled.shape[0], self.base_height - y_out)
            actual_w = min(strip_downsampled.shape[1], self.base_width)
            if actual_h > 0:
                channel_array[y_out:y_out + actual_h, :actual_w] = strip_downsampled[:actual_h, :actual_w]

            del strip, strip_downsampled
            gc.collect()

        return channel_array

    def get_tile(self, tile_x: int, tile_y: int, tile_size: int, channel: int, scale_factor: int) -> np.ndarray:
        """
        Get a tile at the specified scale.

        Args:
            tile_x, tile_y: Tile origin in SCALED coordinates (at target scale_factor)
            tile_size: Output tile size (e.g., 5000)
            channel: Channel index
            scale_factor: Target scale (must be >= base_scale)

        Returns:
            Tile data at target scale, shape (tile_size, tile_size) or smaller at edges
        """
        return _extract_tile_from_cache(self.channels, self.base_scale, tile_x, tile_y, tile_size, channel, scale_factor)

    def apply_photobleaching_correction(self, verbose: bool = True):
        """
        Apply slide-wide photobleaching correction to all loaded channels.

        This corrects for:
        - Horizontal/vertical banding from stitched acquisitions
        - Uneven illumination gradients across the slide

        Must be called AFTER loading but BEFORE extracting tiles.
        """
        from segmentation.preprocessing.illumination import (
            normalize_rows_columns,
            morphological_background_subtraction,
            estimate_band_severity
        )

        if verbose:
            print("\nApplying slide-wide photobleaching correction...")

        for ch, data in self.channels.items():
            if verbose:
                # Estimate severity before correction
                severity = estimate_band_severity(data)
                print(f"  Channel {ch}: row_cv={severity['row_cv']:.1f}%, col_cv={severity['col_cv']:.1f}% ({severity['severity']})")

            # Step 1: Normalize row/column means (fixes banding)
            corrected = normalize_rows_columns(data)

            # Step 2: Morphological background subtraction (fixes gradients)
            # Use larger kernel for slide-wide correction
            corrected = morphological_background_subtraction(corrected, kernel_size=201)

            # Clip to valid range and convert back to uint16
            corrected = np.clip(corrected, 0, 65535).astype(np.uint16)

            # Replace channel data in-place
            self.channels[ch] = corrected

            if verbose:
                # Check improvement
                severity_after = estimate_band_severity(corrected)
                print(f"    After: row_cv={severity_after['row_cv']:.1f}%, col_cv={severity_after['col_cv']:.1f}% ({severity_after['severity']})")

        if verbose:
            print("  Photobleaching correction complete.")

    def release(self):
        """Release all channel data."""
        self.channels.clear()
        gc.collect()


class SharedChannelCache:
    """
    Read-only channel cache backed by shared memory arrays.

    Used by multi-GPU workers to access channel data placed into shared memory
    by the main process. Mirrors DownsampledChannelCache.get_tile() exactly.
    """

    def __init__(self, channel_arrays, base_scale, full_res_size):
        """
        Args:
            channel_arrays: Dict {ch_idx: np.ndarray} backed by shared memory
            base_scale: Scale factor of the loaded data (e.g. 2 for 1/2 scale)
            full_res_size: (width, height) at full resolution
        """
        self.channels = channel_arrays
        self.base_scale = base_scale
        self.full_res_size = full_res_size
        self.base_width = full_res_size[0] // base_scale
        self.base_height = full_res_size[1] // base_scale

    def get_tile(self, tile_x, tile_y, tile_size, channel, scale_factor):
        """
        Get a tile at the specified scale. Same logic as DownsampledChannelCache.get_tile().

        Args:
            tile_x, tile_y: Tile origin in SCALED coordinates (at target scale_factor)
            tile_size: Output tile size
            channel: Channel index
            scale_factor: Target scale (must be >= base_scale)

        Returns:
            Tile data at target scale, or None if out of bounds
        """
        return _extract_tile_from_cache(self.channels, self.base_scale, tile_x, tile_y, tile_size, channel, scale_factor)

    def release(self):
        """Clear references (does not unlink shared memory)."""
        self.channels.clear()


# Channel indices
NUCLEAR = 0
CD31 = 1
SMA = 2
PM = 3

# Directory creation moved to main() to avoid side effects on import (critical for multi-GPU workers)

def normalize_channel(arr, p_low=1, p_high=99):
    """Normalize to uint8 using percentile clipping (robust to outliers)."""
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (p_low, p_high))
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
    return arr.astype(np.uint8)


# ==============================================================================
# Cellpose-SAM Cell Segmentation Functions
# ==============================================================================

def segment_cells_in_vessel(pm_roi, nuclear_roi, diameter_hint=30):
    """
    Segment cells within a vessel ROI using Cellpose-SAM.

    Uses CellposeSAM (SAM backbone) with plasma membrane and nuclear
    channels for accurate cell boundary detection.

    Args:
        pm_roi: 2D numpy array of plasma membrane channel ROI (uint8 or uint16)
        nuclear_roi: 2D numpy array of nuclear channel ROI (uint8 or uint16)
        diameter_hint: Estimated cell diameter in pixels (default: 30)

    Returns:
        tuple: (cell_masks, num_cells)
            - cell_masks: Labeled image where each cell has a unique integer ID (0=background)
            - num_cells: Number of cells detected
    """
    if not CELLPOSE_AVAILABLE:
        print("Warning: Cellpose not available. Install with: pip install cellpose")
        return np.zeros_like(pm_roi, dtype=np.int32), 0

    # Handle empty ROIs
    if pm_roi is None or nuclear_roi is None:
        return np.zeros((1, 1), dtype=np.int32), 0

    if pm_roi.size == 0 or nuclear_roi.size == 0:
        return np.zeros_like(pm_roi, dtype=np.int32), 0

    # Ensure ROIs have the same shape
    if pm_roi.shape != nuclear_roi.shape:
        # Resize nuclear_roi to match pm_roi if needed
        nuclear_roi = cv2.resize(nuclear_roi, (pm_roi.shape[1], pm_roi.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

    # Normalize ROIs to float32 [0, 1] range for Cellpose
    pm_norm = pm_roi.astype(np.float32)
    if pm_norm.max() > 0:
        pm_norm = pm_norm / pm_norm.max()

    nuclear_norm = nuclear_roi.astype(np.float32)
    if nuclear_norm.max() > 0:
        nuclear_norm = nuclear_norm / nuclear_norm.max()

    # Stack channels for Cellpose: [membrane, nuclei]
    # Cellpose expects shape (H, W, 2) with channels=[1, 2] meaning
    # channel 1 = cytoplasm/membrane, channel 2 = nuclei
    img_stack = np.stack([pm_norm, nuclear_norm], axis=-1)

    try:
        # Get the global CellposeSAM model (singleton - loaded once)
        model = get_cellpose_model()
        if model is None:
            print("Warning: CellposeSAM model not available")
            return np.zeros_like(pm_roi, dtype=np.int32), 0

        # Run Cellpose-SAM segmentation
        # channels=[1, 2] means: cytoplasm/membrane is channel 0, nuclei is channel 1
        masks, flows, styles = model.eval(
            img_stack,
            diameter=diameter_hint,
            channels=[1, 2],  # [cytoplasm_channel, nucleus_channel]
        )

        # Count cells (unique labels excluding background 0)
        num_cells = len(np.unique(masks)) - 1 if masks.max() > 0 else 0

        return masks.astype(np.int32), num_cells

    except Exception as e:
        print(f"Warning: Cellpose segmentation failed: {e}")
        return np.zeros_like(pm_roi, dtype=np.int32), 0


def measure_cell_intensities(cell_masks, cd31_roi, sma_roi):
    """
    Measure mean intensities in CD31 and SMA channels for each cell.

    For each cell in the labeled cell_masks, computes the mean pixel
    intensity within that cell's region in both the CD31 and SMA channels.

    Args:
        cell_masks: Labeled image from segment_cells_in_vessel() where each
                   cell has a unique integer ID (0=background)
        cd31_roi: 2D numpy array of CD31 channel ROI
        sma_roi: 2D numpy array of SMA channel ROI

    Returns:
        dict: {
            'cd31': numpy array of mean CD31 intensities for each cell,
            'sma': numpy array of mean SMA intensities for each cell,
            'cell_ids': numpy array of cell IDs corresponding to intensities
        }
        Returns empty arrays if no cells are found.
    """
    # Handle empty inputs
    if cell_masks is None or cell_masks.max() == 0:
        return {
            'cd31': np.array([], dtype=np.float32),
            'sma': np.array([], dtype=np.float32),
            'cell_ids': np.array([], dtype=np.int32)
        }

    if cd31_roi is None or sma_roi is None:
        return {
            'cd31': np.array([], dtype=np.float32),
            'sma': np.array([], dtype=np.float32),
            'cell_ids': np.array([], dtype=np.int32)
        }

    if cd31_roi.size == 0 or sma_roi.size == 0:
        return {
            'cd31': np.array([], dtype=np.float32),
            'sma': np.array([], dtype=np.float32),
            'cell_ids': np.array([], dtype=np.int32)
        }

    # Ensure ROIs match cell_masks shape
    if cd31_roi.shape != cell_masks.shape:
        cd31_roi = cv2.resize(cd31_roi, (cell_masks.shape[1], cell_masks.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
    if sma_roi.shape != cell_masks.shape:
        sma_roi = cv2.resize(sma_roi, (cell_masks.shape[1], cell_masks.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

    # Get unique cell IDs (excluding background 0)
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids > 0]

    if len(cell_ids) == 0:
        return {
            'cd31': np.array([], dtype=np.float32),
            'sma': np.array([], dtype=np.float32),
            'cell_ids': np.array([], dtype=np.int32)
        }

    # Convert ROIs to float for accurate mean calculation
    cd31_float = cd31_roi.astype(np.float64)
    sma_float = sma_roi.astype(np.float64)

    # Vectorized intensity measurement using scipy.ndimage.mean
    cd31_means = ndimage.mean(cd31_float, labels=cell_masks, index=cell_ids)
    sma_means = ndimage.mean(sma_float, labels=cell_masks, index=cell_ids)

    # Filter out cells with very few pixels (likely noise)
    cell_sizes = ndimage.sum(np.ones_like(cell_masks, dtype=np.int32), labels=cell_masks, index=cell_ids)
    valid_mask = np.asarray(cell_sizes) >= 5
    valid_cell_ids = cell_ids[valid_mask]

    return {
        'cd31': np.array(cd31_means, dtype=np.float32)[valid_mask],
        'sma': np.array(sma_means, dtype=np.float32)[valid_mask],
        'cell_ids': np.array(valid_cell_ids, dtype=np.int32)
    }


# ==============================================================================
# Slide-wide GMM Classification for Cell Composition
# ==============================================================================

def classify_cells_gmm(all_cd31_intensities, all_sma_intensities, min_samples=50):
    """
    Fit slide-wide GMM to classify cells as CD31+ (endothelial) or SMA+ (smooth muscle).

    Fits 2-component GMM on all CD31 intensities across slide and separately on all
    SMA intensities. Determines threshold at probability > 0.5 for "positive" component
    (the higher intensity component).

    Args:
        all_cd31_intensities: 1D array of mean CD31 intensities for all cells across slide
        all_sma_intensities: 1D array of mean SMA intensities for all cells across slide
        min_samples: Minimum samples needed to fit GMM. Below this, uses median fallback.

    Returns:
        Tuple of (cd31_threshold, sma_threshold, gmm_info dict)
    """
    all_cd31 = np.asarray(all_cd31_intensities).flatten()
    all_sma = np.asarray(all_sma_intensities).flatten()

    gmm_info = {
        'n_cells_total': len(all_cd31),
        'fallback_used': False,
    }

    # Handle empty input
    if len(all_cd31) == 0:
        gmm_info['fallback_used'] = True
        gmm_info['error'] = 'No cells found'
        return 0.0, 0.0, gmm_info

    # Fallback to median-based threshold if insufficient samples
    if len(all_cd31) < min_samples:
        cd31_threshold = float(np.median(all_cd31))
        sma_threshold = float(np.median(all_sma))
        gmm_info['fallback_used'] = True
        gmm_info['cd31_threshold_method'] = 'median'
        gmm_info['sma_threshold_method'] = 'median'
        gmm_info['cd31_threshold'] = cd31_threshold
        gmm_info['sma_threshold'] = sma_threshold
        return cd31_threshold, sma_threshold, gmm_info

    # Fit CD31 GMM
    cd31_gmm = _make_gmm()
    cd31_gmm.fit(all_cd31.reshape(-1, 1))

    cd31_means = cd31_gmm.means_.flatten()
    cd31_positive_idx = int(np.argmax(cd31_means))

    cd31_values = np.linspace(all_cd31.min(), all_cd31.max(), 1000)
    cd31_proba = cd31_gmm.predict_proba(cd31_values.reshape(-1, 1))[:, cd31_positive_idx]
    crossing_idx = np.argmin(np.abs(cd31_proba - 0.5))
    cd31_threshold = float(cd31_values[crossing_idx])

    gmm_info['cd31_gmm_means'] = cd31_means.tolist()
    gmm_info['cd31_gmm_weights'] = cd31_gmm.weights_.tolist()
    gmm_info['cd31_positive_component'] = cd31_positive_idx
    gmm_info['cd31_threshold'] = cd31_threshold
    gmm_info['cd31_threshold_method'] = 'gmm'

    # Fit SMA GMM
    sma_gmm = _make_gmm()
    sma_gmm.fit(all_sma.reshape(-1, 1))

    sma_means = sma_gmm.means_.flatten()
    sma_positive_idx = int(np.argmax(sma_means))

    sma_values = np.linspace(all_sma.min(), all_sma.max(), 1000)
    sma_proba = sma_gmm.predict_proba(sma_values.reshape(-1, 1))[:, sma_positive_idx]
    crossing_idx = np.argmin(np.abs(sma_proba - 0.5))
    sma_threshold = float(sma_values[crossing_idx])

    gmm_info['sma_gmm_means'] = sma_means.tolist()
    gmm_info['sma_gmm_weights'] = sma_gmm.weights_.tolist()
    gmm_info['sma_positive_component'] = sma_positive_idx
    gmm_info['sma_threshold'] = sma_threshold
    gmm_info['sma_threshold_method'] = 'gmm'

    return cd31_threshold, sma_threshold, gmm_info


def compute_cell_composition(cell_intensities, cd31_threshold, sma_threshold):
    """
    Compute cell composition for a vessel given intensity data and thresholds.

    Classifies each cell as:
    - Endothelial (CD31+): CD31 intensity >= cd31_threshold
    - Smooth muscle (SMA+): SMA intensity >= sma_threshold (and not CD31+)
    - Other: Neither CD31+ nor SMA+

    Args:
        cell_intensities: Dict with 'cd31' and 'sma' arrays
        cd31_threshold: Threshold for CD31+ classification
        sma_threshold: Threshold for SMA+ classification

    Returns:
        Dict with cell composition counts and fractions
    """
    cd31_arr = np.asarray(cell_intensities.get('cd31', []))
    sma_arr = np.asarray(cell_intensities.get('sma', []))

    n_total = len(cd31_arr)

    if n_total == 0:
        return {
            'n_total': 0,
            'n_endothelial': 0,
            'n_smooth_muscle': 0,
            'n_other': 0,
            'frac_endothelial': 0.0,
            'frac_smooth_muscle': 0.0,
            'frac_other': 0.0,
        }

    cd31_positive = cd31_arr >= cd31_threshold
    sma_positive = sma_arr >= sma_threshold

    n_endothelial = int(cd31_positive.sum())
    n_smooth_muscle = int((sma_positive & ~cd31_positive).sum())
    n_other = n_total - n_endothelial - n_smooth_muscle

    return {
        'n_total': n_total,
        'n_endothelial': n_endothelial,
        'n_smooth_muscle': n_smooth_muscle,
        'n_other': n_other,
        'frac_endothelial': float(n_endothelial / n_total),
        'frac_smooth_muscle': float(n_smooth_muscle / n_total),
        'frac_other': float(n_other / n_total),
    }


def save_vessel_crop(vessel, display_rgb, outer_contour, inner_contour, scale_factor, output_dir=None, sma_contour=None):
    """Save a crop of the vessel - both raw and with contours drawn.

    Crop is centered on the mask centroid and sized to be 2x the mask bounding box.
    Saves two files: {uid}_raw.jpg (no contours) and {uid}.jpg (with contours).

    Args:
        output_dir: Output directory. Defaults to global OUTPUT_DIR if None.
        sma_contour: Optional SMA (smooth muscle) contour. When present, drawn in
            magenta as the 3rd ring. Color scheme: cyan=lumen, green=CD31, magenta=SMA.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    # Get bounding box of all contours combined
    contour_list = [outer_contour, inner_contour]
    if sma_contour is not None and len(sma_contour) > 0:
        contour_list.append(sma_contour)
    all_points = np.vstack(contour_list)
    x_min, y_min = all_points.min(axis=0).flatten()
    x_max, y_max = all_points.max(axis=0).flatten()

    # Mask dimensions
    mask_w = x_max - x_min
    mask_h = y_max - y_min

    # Crop size = 2x the mask size (100% larger)
    crop_w = int(mask_w * 2)
    crop_h = int(mask_h * 2)

    # Minimum crop size to avoid tiny crops
    crop_w = max(crop_w, 100)
    crop_h = max(crop_h, 100)

    # Center of the mask bounding box
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # Crop region centered on mask
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(display_rgb.shape[1], cx + crop_w // 2)
    y2 = min(display_rgb.shape[0], cy + crop_h // 2)

    # Check for valid crop
    if x2 <= x1 or y2 <= y1:
        return None

    crop_raw = display_rgb[y1:y2, x1:x2].copy()

    if crop_raw.size == 0:
        return None

    # Save raw crop (no contours)
    raw_path = os.path.join(output_dir, 'crops', f"{vessel['uid']}_raw.jpg")
    cv2.imwrite(raw_path, cv2.cvtColor(crop_raw, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Draw contours on a copy for the contoured version
    crop_contoured = crop_raw.copy()
    outer_in_crop = outer_contour - np.array([x1, y1])
    inner_in_crop = inner_contour - np.array([x1, y1])

    cv2.drawContours(crop_contoured, [outer_in_crop], -1, (0, 255, 0), 2)  # Green for CD31 outer wall
    cv2.drawContours(crop_contoured, [inner_in_crop], -1, (0, 255, 255), 2)  # Cyan for inner lumen

    # Draw SMA contour in magenta if present
    if sma_contour is not None and len(sma_contour) > 0:
        sma_in_crop = sma_contour.reshape(-1, 1, 2) - np.array([x1, y1])
        cv2.drawContours(crop_contoured, [sma_in_crop], -1, (255, 0, 255), 2)  # Magenta for SMA

    # Store crop offset for reference
    vessel['crop_offset'] = [int(x1), int(y1)]
    vessel['crop_scale_factor'] = scale_factor

    # Save contoured crop
    crop_path = os.path.join(output_dir, 'crops', f"{vessel['uid']}.jpg")
    cv2.imwrite(crop_path, cv2.cvtColor(crop_contoured, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Store raw path for side-by-side display
    vessel['crop_path_raw'] = raw_path

    return crop_path


def save_vessel_crop_fullres(vessel, channel_cache, output_dir=None, padding_factor=1.0):
    """Save a crop at full CZI resolution with refined contours.

    Reads a small ROI from the CZI at native resolution, normalizes the 3
    display channels (SMA, CD31, nuclear), and draws all contours at
    pixel-perfect resolution.

    Color scheme: cyan=lumen, green=CD31, magenta=SMA.

    Args:
        vessel: Vessel dict with refined contours in full-res tile-local coords.
        channel_cache: DownsampledChannelCache (used for its .loader reference).
        output_dir: Output directory. Defaults to global OUTPUT_DIR.
        padding_factor: Padding around contour bbox as multiple of vessel size.

    Returns:
        crop_path if successful, None otherwise.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    scale_factor = vessel['scale_factor']
    tile_x = vessel['tile_x']
    tile_y = vessel['tile_y']

    # All contours are in full-res tile-local coordinates
    inner_pts = np.array(vessel['inner_contour']).reshape(-1, 2)
    outer_pts = np.array(vessel['outer_contour']).reshape(-1, 2)

    if len(inner_pts) < 3 or len(outer_pts) < 3:
        return None

    # Collect all contour points for bounding box
    all_pts_list = [inner_pts, outer_pts]
    has_sma = vessel.get('has_sma_ring', False) and vessel.get('sma_contour')
    if has_sma:
        sma_pts = np.array(vessel['sma_contour']).reshape(-1, 2)
        all_pts_list.append(sma_pts)
    all_pts = np.vstack(all_pts_list)

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min

    # Crop size = vessel bbox + padding
    pad_x = int(w * padding_factor / 2)
    pad_y = int(h * padding_factor / 2)
    roi_x1 = max(0, int(x_min) - pad_x)
    roi_y1 = max(0, int(y_min) - pad_y)
    roi_x2 = int(x_max) + pad_x
    roi_y2 = int(y_max) + pad_y
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    if roi_w <= 0 or roi_h <= 0:
        return None

    # Convert to global CZI mosaic coords
    tile_origin_x = tile_x * scale_factor
    tile_origin_y = tile_y * scale_factor
    global_x = tile_origin_x + roi_x1
    global_y = tile_origin_y + roi_y1

    loader = channel_cache.loader

    # Read 3 display channels at full resolution
    try:
        sma_raw = loader.reader.read_mosaic(
            region=(loader.x_start + global_x, loader.y_start + global_y, roi_w, roi_h),
            scale_factor=1, C=SMA
        )
        cd31_raw = loader.reader.read_mosaic(
            region=(loader.x_start + global_x, loader.y_start + global_y, roi_w, roi_h),
            scale_factor=1, C=CD31
        )
        nuc_raw = loader.reader.read_mosaic(
            region=(loader.x_start + global_x, loader.y_start + global_y, roi_w, roi_h),
            scale_factor=1, C=NUCLEAR
        )
    except Exception:
        return None

    sma_raw = np.squeeze(sma_raw)
    cd31_raw = np.squeeze(cd31_raw)
    nuc_raw = np.squeeze(nuc_raw)

    if sma_raw.size == 0 or cd31_raw.size == 0 or nuc_raw.size == 0:
        return None

    # Normalize and build display RGB (R=SMA, G=CD31, B=nuclear)
    display_rgb = np.stack([
        normalize_channel(sma_raw),
        normalize_channel(cd31_raw),
        normalize_channel(nuc_raw),
    ], axis=-1)

    # Map contours from tile-local to crop-local coords
    offset = np.array([roi_x1, roi_y1])
    inner_crop = (inner_pts - offset).astype(np.int32).reshape(-1, 1, 2)
    outer_crop = (outer_pts - offset).astype(np.int32).reshape(-1, 1, 2)

    uid = vessel['uid']

    # Save raw crop (no contours)
    raw_path = os.path.join(output_dir, 'crops', f"{uid}_raw.jpg")
    cv2.imwrite(raw_path, cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Draw contours
    crop_contoured = display_rgb.copy()
    cv2.drawContours(crop_contoured, [outer_crop], -1, (0, 255, 0), 2)   # Green: CD31
    cv2.drawContours(crop_contoured, [inner_crop], -1, (0, 255, 255), 2)  # Cyan: lumen

    if has_sma:
        sma_crop = (sma_pts - offset).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(crop_contoured, [sma_crop], -1, (255, 0, 255), 2)  # Magenta: SMA

    crop_path = os.path.join(output_dir, 'crops', f"{uid}.jpg")
    cv2.imwrite(crop_path, cv2.cvtColor(crop_contoured, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Update vessel dict
    vessel['crop_path'] = crop_path
    vessel['crop_path_raw'] = raw_path
    vessel['crop_offset'] = [int(roi_x1), int(roi_y1)]
    vessel['crop_scale_factor'] = 1  # Full resolution

    del display_rgb, crop_contoured, sma_raw, cd31_raw, nuc_raw
    return crop_path


def regenerate_missing_crops(vessels, channel_cache):
    """Regenerate crop images for vessels that are missing crop files on disk.

    This handles the case where crops from prior scales are lost (e.g., after
    a system reboot) but the vessel data was preserved in checkpoints.

    Groups vessels by tile to minimize redundant CZI reads, then calls
    save_vessel_crop() for each vessel with a missing crop.
    """
    # Build scale_factor -> tile_size lookup from SCALES config
    scale_tile_sizes = {s['scale_factor']: s.get('tile_size', TILE_SIZE) for s in SCALES}

    # Find vessels with missing crops
    missing = []
    for i, v in enumerate(vessels):
        crop_path = v.get('crop_path')
        if crop_path is None or not os.path.exists(crop_path):
            missing.append(i)

    if not missing:
        print("All crop files present - no regeneration needed.")
        return

    print(f"Regenerating crops for {len(missing)} vessels with missing crop files...")

    # Group by (tile_x, tile_y, scale_factor) to batch tile reads
    tile_groups = {}
    for idx in missing:
        v = vessels[idx]
        key = (v['tile_x'], v['tile_y'], v['scale_factor'])
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append(idx)

    n_generated = 0
    n_failed = 0

    for (tile_x, tile_y, scale_factor), indices in tqdm(tile_groups.items(), desc="Regenerating crops"):
        tile_size = scale_tile_sizes.get(scale_factor, TILE_SIZE)

        # Read tile channels
        sma = channel_cache.get_tile(tile_x, tile_y, tile_size, SMA, scale_factor)
        cd31 = channel_cache.get_tile(tile_x, tile_y, tile_size, CD31, scale_factor)
        nuclear = channel_cache.get_tile(tile_x, tile_y, tile_size, NUCLEAR, scale_factor)

        if sma is None or cd31 is None or nuclear is None:
            n_failed += len(indices)
            continue

        # Create display RGB (same as process_tile_at_scale)
        display_rgb = np.stack([
            normalize_channel(sma),
            normalize_channel(cd31),
            normalize_channel(nuclear),
        ], axis=-1)

        for idx in indices:
            v = vessels[idx]
            # Contours are tile-local at full-res scale; convert back to tile scale
            outer_contour = np.array(v['outer_contour'], dtype=np.int32) // scale_factor
            inner_contour = np.array(v['inner_contour'], dtype=np.int32) // scale_factor

            if len(outer_contour) == 0 or len(inner_contour) == 0:
                n_failed += 1
                continue

            # Extract SMA contour if available
            sma_contour_regen = None
            sma_raw = v.get('sma_contour')
            if sma_raw and len(sma_raw) >= 3 and v.get('has_sma_ring', False):
                sma_contour_regen = np.array(sma_raw, dtype=np.int32) // scale_factor

            crop_path = save_vessel_crop(
                v, display_rgb, outer_contour, inner_contour,
                scale_factor, sma_contour=sma_contour_regen
            )
            if crop_path:
                v['crop_path'] = crop_path
                n_generated += 1
            else:
                n_failed += 1

        del display_rgb, sma, cd31, nuclear
        gc.collect()

    print(f"  Regenerated {n_generated} crops, {n_failed} failed")


def is_tissue_tile(tile, threshold=TISSUE_VARIANCE_THRESHOLD):
    return np.var(tile) > threshold


def compute_edge_density(tile):
    """
    Compute mean Sobel gradient magnitude for a tile.

    Args:
        tile: 2D numpy array (grayscale tile data)

    Returns:
        float: Mean gradient magnitude (absolute, not relative to percentile)
    """
    # Convert to float32 for processing
    tile_float = tile.astype(np.float32)

    # Apply Gaussian smoothing to reduce noise (sigma=2)
    smoothed = gaussian_filter(tile_float, sigma=2)

    # Compute Sobel gradient magnitude
    sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return float(gradient_magnitude.mean())


def compute_adaptive_sobel_threshold(channel_cache, scale_factor, sample_tiles=100, tile_size=1000):
    """
    Sample tiles across the slide, compute edge densities, fit GMM to separate
    tissue tiles from background tiles.

    Args:
        channel_cache: DownsampledChannelCache with loaded channels
        scale_factor: Scale to sample at
        sample_tiles: Number of random tiles to sample
        tile_size: Size of tiles to sample (in scaled coordinates)

    Returns:
        float: Adaptive edge density threshold
    """
    print(f"\nComputing adaptive Sobel threshold at 1/{scale_factor} scale...")

    # Get mosaic dimensions at this scale
    scaled_width = channel_cache.full_res_size[0] // scale_factor
    scaled_height = channel_cache.full_res_size[1] // scale_factor

    # Generate random tile positions
    max_x = max(0, scaled_width - tile_size)
    max_y = max(0, scaled_height - tile_size)

    if max_x == 0 or max_y == 0:
        print("  Image too small for sampling, using default threshold 0.05")
        return 0.05

    # Sample random positions
    n_samples = min(sample_tiles, (max_x // tile_size) * (max_y // tile_size))
    n_samples = max(n_samples, 20)  # At least 20 samples

    positions = []
    for _ in range(n_samples):
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        positions.append((x, y))

    # Compute edge densities for each tile using SMA channel (most reliable for tissue)
    edge_densities = []
    for x, y in tqdm(positions, desc="Sampling edge densities"):
        tile = channel_cache.get_tile(x, y, tile_size, SMA, scale_factor)
        if tile is None or tile.size == 0:
            continue

        # Skip tiles with very low variance (likely empty/background)
        if np.var(tile) < 10:
            edge_densities.append(0.0)
            continue

        density = compute_edge_density(tile)
        edge_densities.append(density)

    edge_densities = np.array(edge_densities)

    if len(edge_densities) < 20:
        print(f"  Only {len(edge_densities)} valid samples, using default threshold 0.05")
        return 0.05

    print(f"  Sampled {len(edge_densities)} tiles")
    print(f"  Edge density range: {edge_densities.min():.4f} - {edge_densities.max():.4f}")
    print(f"  Mean: {edge_densities.mean():.4f}, Std: {edge_densities.std():.4f}")

    # Handle edge cases: all similar values
    if edge_densities.std() < 0.001:
        # All tiles have similar edge density - use a relative threshold
        threshold = edge_densities.mean() * 0.5
        print(f"  Low variance in densities, using 50% of mean: {threshold:.4f}")
        return max(threshold, 0.01)

    # Fit 2-component GMM to separate tissue (high edge density) from background (low)
    try:
        gmm = _make_gmm()
        gmm.fit(edge_densities.reshape(-1, 1))

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())

        # Sort by mean (lower mean = background, higher mean = tissue)
        sorted_idx = np.argsort(means)
        low_mean = means[sorted_idx[0]]
        high_mean = means[sorted_idx[1]]
        low_std = stds[sorted_idx[0]]
        high_std = stds[sorted_idx[1]]

        print(f"  GMM components:")
        print(f"    Background: mean={low_mean:.4f}, std={low_std:.4f}")
        print(f"    Tissue: mean={high_mean:.4f}, std={high_std:.4f}")

        # Threshold = midpoint between means
        # This separates the two populations
        threshold = (low_mean + high_mean) / 2

        # Sanity checks
        if threshold < 0.005:
            print(f"  Threshold too low ({threshold:.4f}), clamping to 0.01")
            threshold = 0.01
        elif threshold > 0.2:
            print(f"  Threshold too high ({threshold:.4f}), clamping to 0.2")
            threshold = 0.2

        # If means are too close, components may not be well separated
        if abs(high_mean - low_mean) < 0.01:
            print(f"  GMM components not well separated, using percentile-based threshold")
            threshold = np.percentile(edge_densities, 25)
            threshold = max(threshold, 0.01)

        print(f"  Adaptive threshold: {threshold:.4f}")
        return threshold

    except Exception as e:
        print(f"  GMM fitting failed: {e}")
        print(f"  Using default threshold 0.05")
        return 0.05


def has_vessel_candidates(tile, min_edge_density=0.05):
    """
    Fast Sobel-based pre-filter to detect if tile has vessel candidates.

    Vessels have strong edges (vessel walls), so tiles without significant
    edge density are unlikely to contain vessels and can be skipped.

    Args:
        tile: 2D numpy array (grayscale tile data)
        min_edge_density: Minimum mean gradient magnitude to consider
                         tile as having vessel candidates (default: 0.05)

    Returns:
        True if tile has sufficient edge density (potential vessels), False otherwise.
    """
    edge_density = compute_edge_density(tile)
    return edge_density > min_edge_density


def compute_lumen_ratios(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor=1):
    """
    Compute intensity ratios for a candidate lumen mask.

    Returns None if mask fails basic area/surrounding checks, otherwise returns
    dict with ratios and stats for later GMM-based classification.
    """
    area = mask.sum()
    # Scale area thresholds: base values for 1/1 scale, adjust for current scale
    # At coarser scales, fewer pixels represent same physical area
    min_area = 50 // (scale_factor * scale_factor) if scale_factor > 1 else 50
    max_area = 500000 // (scale_factor * scale_factor) if scale_factor > 1 else 500000
    min_area = max(10, min_area)  # floor to avoid rejecting everything

    if area < min_area or area > max_area:
        return None

    # Measure intensity inside mask for all 4 channels
    sma_inside = sma_norm[mask].mean()
    nuclear_inside = nuclear_norm[mask].mean()
    cd31_inside = cd31_norm[mask].mean()
    pm_inside = pm_norm[mask].mean()

    # Scale dilation kernel: 15px at 1/1 -> smaller at coarser scales
    kernel_size = max(3, 15 // scale_factor)
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    surrounding = dilated.astype(bool) & ~mask

    min_surrounding = max(20, 100 // (scale_factor * scale_factor))
    if surrounding.sum() < min_surrounding:
        return None

    # Measure intensity in surrounding ring for all 4 channels
    sma_surrounding = sma_norm[surrounding].mean()
    nuclear_surrounding = nuclear_norm[surrounding].mean()
    cd31_surrounding = cd31_norm[surrounding].mean()
    pm_surrounding = pm_norm[surrounding].mean()

    # Compute ratios (inside / surrounding)
    sma_ratio = sma_inside / max(sma_surrounding, 1e-6)
    nuclear_ratio = nuclear_inside / max(nuclear_surrounding, 1e-6)
    cd31_ratio = cd31_inside / max(cd31_surrounding, 1e-6)
    pm_ratio = pm_inside / max(pm_surrounding, 1e-6)

    return {
        'area': int(area),
        'sma_inside': float(sma_inside),
        'sma_wall': float(sma_surrounding),
        'sma_ratio': float(sma_ratio),
        'cd31_ratio': float(cd31_ratio),
        'nuclear_ratio': float(nuclear_ratio),
        'pm_ratio': float(pm_ratio),
    }


def verify_lumen_multichannel(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor=1):
    """Verify region is true lumen by checking all 4 channels.

    True lumens (blood vessel interior) should be darker than surrounding
    tissue in all channels since they contain blood/empty space.

    Note: This function uses fixed thresholds for single-mask verification.
    For batch processing with GMM-based adaptive thresholds, use
    verify_lumens_batch() instead.
    """
    stats = compute_lumen_ratios(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor)
    if stats is None:
        return False, {}

    # Lumen validation: inside should be darker than surrounding in all channels
    # All ratios < 1.0 means lumen is darker than surrounding wall
    sma_ratio = stats['sma_ratio']
    cd31_ratio = stats['cd31_ratio']
    nuclear_ratio = stats['nuclear_ratio']
    pm_ratio = stats['pm_ratio']

    is_valid = (sma_ratio < 1.0) and (cd31_ratio < 1.0) and (nuclear_ratio < 1.0) and (pm_ratio < 1.0)

    return is_valid, stats


def verify_lumens_batch(masks, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor=1, threshold_ppv=0.5):
    """
    Verify multiple candidate lumen masks using GMM-based adaptive thresholding.

    Instead of fixed ratio < 1.0 thresholds, this:
    1. Computes ratios for all masks
    2. Fits a 2-component GMM to each channel's ratios
    3. Classifies masks based on their probability of being in the "low ratio" class

    This is more robust to staining intensity variation across the slide.

    Args:
        masks: List of boolean masks (SAM2 segmentation outputs)
        sma_norm, nuclear_norm, cd31_norm, pm_norm: Normalized channel images
        scale_factor: Current processing scale
        threshold_ppv: Probability threshold for GMM classification (default 0.5)

    Returns:
        List of (is_valid, stats) tuples for each mask
    """
    # First pass: compute ratios for all masks
    all_stats = []
    for mask in masks:
        stats = compute_lumen_ratios(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor)
        all_stats.append(stats)

    # Collect ratios from valid masks for GMM fitting
    valid_indices = [i for i, s in enumerate(all_stats) if s is not None]
    if len(valid_indices) == 0:
        return [(False, {}) for _ in masks]

    valid_stats = [all_stats[i] for i in valid_indices]

    # Extract ratio arrays for each channel
    sma_ratios = np.array([s['sma_ratio'] for s in valid_stats])
    cd31_ratios = np.array([s['cd31_ratio'] for s in valid_stats])
    nuclear_ratios = np.array([s['nuclear_ratio'] for s in valid_stats])
    pm_ratios = np.array([s['pm_ratio'] for s in valid_stats])

    # Apply GMM classification to each channel
    # For lumens, we want the LOWER ratio class (darker inside than outside)
    sma_valid = gmm_classify_ratios(sma_ratios, threshold_ppv, min_samples=50, lower_is_positive=True)
    cd31_valid = gmm_classify_ratios(cd31_ratios, threshold_ppv, min_samples=50, lower_is_positive=True)
    nuclear_valid = gmm_classify_ratios(nuclear_ratios, threshold_ppv, min_samples=50, lower_is_positive=True)
    pm_valid = gmm_classify_ratios(pm_ratios, threshold_ppv, min_samples=50, lower_is_positive=True)

    # A mask is valid only if ALL channels classify it as a lumen
    combined_valid = sma_valid & cd31_valid & nuclear_valid & pm_valid

    # Build results list
    results = []
    valid_idx = 0
    for i, stats in enumerate(all_stats):
        if stats is None:
            results.append((False, {}))
        else:
            is_valid = bool(combined_valid[valid_idx])
            results.append((is_valid, stats))
            valid_idx += 1

    return results


# Minimum CD31 edge coverage to consider a lumen as a real vessel (not a tear)
CD31_EDGE_COVERAGE_THRESHOLD = 0.40  # 40% of perimeter must have CD31+ lining


def compute_cd31_edge_coverage(lumen_mask, cd31_norm, ring_width=3, n_samples=36):
    """
    Compute fraction of lumen perimeter that has CD31+ endothelial lining.

    Real vessels have CD31+ endothelium lining the lumen edge.
    Tissue tears have no endothelial lining.

    Args:
        lumen_mask: Binary mask of the lumen
        cd31_norm: Normalized CD31 channel (0-1 float)
        ring_width: Width of ring to sample at lumen edge (pixels)
        n_samples: Number of points to sample around perimeter

    Returns:
        Tuple of (coverage_fraction, edge_intensity_mean, background_intensity)
        coverage_fraction: Fraction of perimeter with elevated CD31 (0-1)
    """
    # Find lumen contour
    contours, _ = cv2.findContours(
        lumen_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        return 0.0, 0.0, 0.0

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 10:
        return 0.0, 0.0, 0.0

    # Create thin ring mask just outside the lumen (the endothelial zone)
    lumen_dilated = cv2.dilate(lumen_mask.astype(np.uint8),
                                np.ones((ring_width*2+1, ring_width*2+1), np.uint8))
    edge_ring = lumen_dilated.astype(bool) & ~lumen_mask.astype(bool)

    if edge_ring.sum() < 10:
        return 0.0, 0.0, 0.0

    # Get CD31 intensity in the edge ring
    edge_intensities = cd31_norm[edge_ring]
    edge_mean = edge_intensities.mean()

    # Get background intensity (outside the dilated region, nearby)
    bg_dilated = cv2.dilate(lumen_dilated, np.ones((20, 20), np.uint8))
    bg_mask = bg_dilated.astype(bool) & ~lumen_dilated.astype(bool)
    if bg_mask.sum() > 0:
        bg_intensity = cd31_norm[bg_mask].mean()
    else:
        bg_intensity = cd31_norm.mean()

    # Sample points around the contour perimeter
    contour_points = contour.reshape(-1, 2)
    step = max(1, len(contour_points) // n_samples)
    sample_indices = range(0, len(contour_points), step)

    # For each sample point, check if CD31 is elevated in the adjacent ring
    n_positive = 0
    n_checked = 0

    # Threshold: CD31 must be at least 1.5x background to count as "lined"
    cd31_threshold = max(bg_intensity * 1.5, 0.1)

    for i in sample_indices:
        px, py = contour_points[i]

        # Sample a small region just outside this point
        # Move outward from contour by ring_width/2
        # Use gradient direction or just sample nearby
        y1 = max(0, py - ring_width)
        y2 = min(cd31_norm.shape[0], py + ring_width + 1)
        x1 = max(0, px - ring_width)
        x2 = min(cd31_norm.shape[1], px + ring_width + 1)

        if y2 <= y1 or x2 <= x1:
            continue

        # Get intensity in this small patch, but only the ring portion
        patch_ring = edge_ring[y1:y2, x1:x2]
        if patch_ring.sum() == 0:
            continue

        patch_cd31 = cd31_norm[y1:y2, x1:x2]
        local_intensity = patch_cd31[patch_ring].mean()

        n_checked += 1
        if local_intensity >= cd31_threshold:
            n_positive += 1

    if n_checked == 0:
        return 0.0, edge_mean, bg_intensity

    coverage = n_positive / n_checked
    return coverage, edge_mean, bg_intensity


def dilate_until_signal_drops(lumens, cd31_norm, scale_factor=1, drop_ratio=0.9,
                              max_iterations=50, mode='adaptive',
                              min_wall_intensity=None):
    """
    Expand from lumens by dilating until signal drops.

    Works for any channel (CD31, SMA, etc.). Expands outward from lumen masks
    until the signal intensity drops below a threshold relative to the initial
    wall intensity.

    When min_wall_intensity is set, lumens whose initial ring intensity is below
    this value are returned as-is (no expansion). This is used for SMA ring
    detection where weak signal means no smooth muscle layer (vein/capillary).

    Two modes:
    - 'adaptive' (default): Per-pixel region growing. Each candidate pixel is
      accepted only if its local intensity exceeds the threshold. Produces
      irregular contours that follow the actual signal boundary.
    - 'uniform': Global ring check. Expands uniformly in all directions; stops
      when the mean intensity of the entire new ring drops below threshold.

    Returns labels array where each lumen's expanded region has a unique label.
    """
    labels = np.zeros(cd31_norm.shape, dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)

    # For adaptive mode, smooth slightly to reduce single-pixel noise
    if mode == 'adaptive':
        cd31_smooth = cv2.GaussianBlur(cd31_norm, (3, 3), 0)
    else:
        cd31_smooth = cd31_norm

    for idx, lumen in enumerate(lumens):
        mask = lumen['mask'].astype(np.uint8)
        label_id = idx + 1

        # Get initial wall intensity (the CD31+ ring around lumen)
        dilated_once = cv2.dilate(mask, kernel, iterations=1)
        initial_ring = (dilated_once > 0) & (mask == 0)
        if initial_ring.sum() == 0:
            labels[mask > 0] = label_id
            continue
        wall_intensity = cd31_smooth[initial_ring].mean()

        # If min_wall_intensity is set and signal is too weak, return lumen only
        if min_wall_intensity is not None and wall_intensity < min_wall_intensity:
            labels[mask > 0] = label_id
            continue

        # Threshold for stopping: when ring drops below this
        stop_threshold = wall_intensity * drop_ratio

        if mode == 'adaptive':
            # Per-pixel region growing: only accept pixels where CD31 is above threshold
            current_mask = dilated_once.copy()
            initial_reject = initial_ring & (cd31_smooth < stop_threshold)
            current_mask[initial_reject] = 0
            current_mask[mask > 0] = 1

            for i in range(max_iterations - 1):
                dilated = cv2.dilate(current_mask, kernel, iterations=1)
                new_pixels = (dilated > 0) & (current_mask == 0)

                if new_pixels.sum() == 0:
                    break

                accept = new_pixels & (cd31_smooth >= stop_threshold)

                if accept.sum() == 0:
                    break

                current_mask[accept] = 1
        else:
            # Uniform mode: global ring check (original behavior)
            current_mask = dilated_once.copy()
            for i in range(max_iterations - 1):
                dilated = cv2.dilate(current_mask, kernel, iterations=1)
                new_ring = (dilated > 0) & (current_mask == 0)

                if new_ring.sum() == 0:
                    break

                ring_intensity = cd31_smooth[new_ring].mean()

                if ring_intensity < stop_threshold:
                    break

                current_mask = dilated

        # Assign label to final expanded region
        labels[current_mask > 0] = label_id

    return labels


# ==============================================================================
# Contour Refinement at Full Resolution
# ==============================================================================

def smooth_contour_spline(contour, smoothing=3.0, n_points=0):
    """
    Smooth a closed contour using periodic B-spline interpolation.

    Fits a smooth curve through the contour points, eliminating stair-step
    artifacts from coarse-scale detection and upscaling.

    Args:
        contour: Contour array, shape (N, 1, 2) or (N, 2), integer pixel coords
        smoothing: Spline smoothing factor. Higher = smoother (less faithful to
            original points). 0 = interpolating spline (passes through all points).
            Default 3.0 works well for removing scaling stair-steps.
        n_points: Number of output points. 0 = same as input. Set higher for
            smoother visual appearance (e.g., 2*N).

    Returns:
        Smoothed contour as int32 array, shape (M, 1, 2) suitable for cv2
    """
    from scipy.interpolate import splprep, splev

    pts = np.array(contour).reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return contour  # Too few points to smooth

    # Remove duplicate consecutive points (can cause splprep to fail)
    diffs = np.diff(pts, axis=0)
    mask = np.any(diffs != 0, axis=1)
    mask = np.append(mask, True)  # Keep last point
    pts = pts[mask]
    if len(pts) < 5:
        return contour

    x, y = pts[:, 0], pts[:, 1]

    try:
        # Fit periodic B-spline (closed contour)
        tck, u = splprep([x, y], s=smoothing, per=True, k=3)

        # Evaluate at desired number of points
        if n_points <= 0:
            n_points = len(pts)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = splev(u_new, tck)

        smoothed = np.stack([x_new, y_new], axis=-1).astype(np.int32)
        return smoothed.reshape(-1, 1, 2)
    except Exception:
        # If spline fitting fails (degenerate contour), return original
        return np.array(contour).reshape(-1, 1, 2).astype(np.int32)


def refine_vessel_contours_fullres(vessel, channel_cache, spline=False,
                                   spline_smoothing=3.0,
                                   cd31_drop_ratio=0.9, cd31_mode='adaptive',
                                   sma_drop_ratio=0.9,
                                   sma_min_wall_intensity=30, sma_mode='adaptive',
                                   pixel_size_um=0.1725, padding_factor=2.0):
    """
    Refine vessel contours at full resolution using CZI data.

    Reads a small ROI from the CZI at native resolution around the vessel,
    reconstructs the lumen mask, re-runs CD31 and SMA dilation at full res,
    and extracts pixel-perfect contours with no scaling artifacts.

    Optionally applies spline smoothing for extra polish.

    Args:
        vessel: Vessel dict with 'inner_contour', 'outer_contour', 'tile_x', etc.
        channel_cache: DownsampledChannelCache (used for its .loader reference)
        spline: Apply spline smoothing to refined contours (default False)
        spline_smoothing: Spline smoothing factor (default 3.0)
        cd31_drop_ratio: Drop ratio for CD31 dilation (default 0.9)
        cd31_mode: CD31 dilation mode, 'adaptive' or 'uniform' (default 'adaptive')
        sma_drop_ratio: Drop ratio for SMA dilation (default 0.9)
        sma_min_wall_intensity: Min SMA wall intensity threshold (default 30)
        sma_mode: SMA dilation mode, 'adaptive' or 'uniform' (default 'adaptive')
        pixel_size_um: Physical pixel size at full resolution (default 0.1725)
        padding_factor: Padding around vessel bbox as multiple of vessel size (default 2.0)

    Returns:
        True if refinement succeeded, False otherwise.
        Modifies vessel dict in-place with refined contours and updated measurements.
    """
    scale_factor = vessel['scale_factor']
    tile_x = vessel['tile_x']
    tile_y = vessel['tile_y']

    # Get coarse contours in full-res tile-local coordinates
    inner_pts = np.array(vessel['inner_contour']).reshape(-1, 2)
    outer_pts = np.array(vessel['outer_contour']).reshape(-1, 2)

    if len(inner_pts) < 3 or len(outer_pts) < 3:
        return False

    # Include SMA contour in bbox calculation if present
    all_pts = [inner_pts, outer_pts]
    sma_pts = None
    if vessel.get('sma_contour') and len(vessel['sma_contour']) >= 3:
        sma_pts = np.array(vessel['sma_contour']).reshape(-1, 2)
        all_pts.append(sma_pts)
    all_pts_arr = np.vstack(all_pts)

    # Bounding box in full-res tile-local coords
    x_min, y_min = all_pts_arr.min(axis=0)
    x_max, y_max = all_pts_arr.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min

    # Add padding so dilation has room to expand
    pad_x = int(w * padding_factor / 2)
    pad_y = int(h * padding_factor / 2)
    roi_x1 = max(0, int(x_min) - pad_x)
    roi_y1 = max(0, int(y_min) - pad_y)
    roi_x2 = int(x_max) + pad_x
    roi_y2 = int(y_max) + pad_y

    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    if roi_w <= 0 or roi_h <= 0:
        return False

    # Convert ROI from tile-local full-res coords to global CZI mosaic coords
    tile_origin_x_fullres = tile_x * scale_factor
    tile_origin_y_fullres = tile_y * scale_factor
    global_roi_x = tile_origin_x_fullres + roi_x1
    global_roi_y = tile_origin_y_fullres + roi_y1

    loader = channel_cache.loader

    # Read CD31 and SMA channels at full resolution from CZI
    try:
        cd31_roi = loader.reader.read_mosaic(
            region=(loader.x_start + global_roi_x, loader.y_start + global_roi_y, roi_w, roi_h),
            scale_factor=1, C=CD31
        )
        cd31_roi = np.squeeze(cd31_roi)

        sma_roi = loader.reader.read_mosaic(
            region=(loader.x_start + global_roi_x, loader.y_start + global_roi_y, roi_w, roi_h),
            scale_factor=1, C=SMA
        )
        sma_roi = np.squeeze(sma_roi)
    except Exception:
        return False

    if cd31_roi.size == 0 or sma_roi.size == 0:
        return False

    # Validate ROI shapes match expected (roi_h, roi_w); resize if CZI returned different size
    if cd31_roi.shape != (roi_h, roi_w):
        cd31_roi = cv2.resize(cd31_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    if sma_roi.shape != (roi_h, roi_w):
        sma_roi = cv2.resize(sma_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    # Normalize to uint8
    cd31_norm = normalize_channel(cd31_roi)
    sma_norm = normalize_channel(sma_roi)

    # Reconstruct lumen mask at full res within ROI
    lumen_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    inner_in_roi = inner_pts - np.array([roi_x1, roi_y1])
    cv2.drawContours(lumen_mask, [inner_in_roi.reshape(-1, 1, 2).astype(np.int32)], -1, 1, -1)

    if lumen_mask.sum() == 0:
        return False

    lumens = [{'mask': lumen_mask.astype(bool)}]

    # Re-run CD31 dilation at full resolution (scale_factor=1)
    cd31_labels = dilate_until_signal_drops(lumens, cd31_norm, scale_factor=1,
                                            drop_ratio=cd31_drop_ratio,
                                            mode=cd31_mode)

    # Re-run SMA dilation at full resolution
    sma_labels = dilate_until_signal_drops(lumens, sma_norm, scale_factor=1,
                                           drop_ratio=sma_drop_ratio,
                                           min_wall_intensity=sma_min_wall_intensity,
                                           mode=sma_mode)

    # Extract refined contours in ROI coordinates
    # Inner (lumen) contour — spline-smooth the original since we don't re-detect it
    inner_in_roi_cv = inner_in_roi.reshape(-1, 1, 2).astype(np.int32)
    if spline:
        inner_in_roi_cv = smooth_contour_spline(inner_in_roi_cv, smoothing=spline_smoothing)

    # Outer (CD31) contour
    cd31_mask = (cd31_labels == 1).astype(np.uint8)
    cd31_contours, _ = cv2.findContours(cd31_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cd31_contours) == 0:
        return False
    outer_in_roi = max(cd31_contours, key=cv2.contourArea)
    if spline:
        outer_in_roi = smooth_contour_spline(outer_in_roi, smoothing=spline_smoothing)

    # SMA contour
    sma_mask = (sma_labels == 1).astype(np.uint8)
    sma_contours_found, _ = cv2.findContours(sma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(sma_contours_found) > 0:
        sma_in_roi = max(sma_contours_found, key=cv2.contourArea)
    else:
        sma_in_roi = inner_in_roi_cv  # Fallback

    sma_area_roi = cv2.contourArea(sma_in_roi)
    inner_area_roi = cv2.contourArea(inner_in_roi_cv)
    has_sma_ring = sma_area_roi > inner_area_roi * 1.05

    if spline and has_sma_ring:
        sma_in_roi = smooth_contour_spline(sma_in_roi, smoothing=spline_smoothing)

    # Convert back to tile-local full-res coordinates
    offset = np.array([roi_x1, roi_y1])
    inner_full = (np.array(inner_in_roi_cv).reshape(-1, 2) + offset).astype(np.int32)
    outer_full = (np.array(outer_in_roi).reshape(-1, 2) + offset).astype(np.int32)
    sma_full = (np.array(sma_in_roi).reshape(-1, 2) + offset).astype(np.int32)

    # Recompute measurements at full resolution
    outer_area = cv2.contourArea(outer_full.reshape(-1, 1, 2))
    inner_area = cv2.contourArea(inner_full.reshape(-1, 1, 2))
    sma_area = cv2.contourArea(sma_full.reshape(-1, 1, 2))

    if len(outer_full) >= 5:
        outer_ellipse = cv2.fitEllipse(outer_full.reshape(-1, 1, 2))
        outer_diameter = (outer_ellipse[1][0] + outer_ellipse[1][1]) / 2 * pixel_size_um
    else:
        outer_diameter = np.sqrt(outer_area / np.pi) * 2 * pixel_size_um

    if len(inner_full) >= 5:
        inner_ellipse = cv2.fitEllipse(inner_full.reshape(-1, 1, 2))
        inner_diameter = (inner_ellipse[1][0] + inner_ellipse[1][1]) / 2 * pixel_size_um
    else:
        inner_diameter = np.sqrt(inner_area / np.pi) * 2 * pixel_size_um

    if len(sma_full) >= 5 and has_sma_ring:
        sma_ellipse = cv2.fitEllipse(sma_full.reshape(-1, 1, 2))
        sma_diameter = (sma_ellipse[1][0] + sma_ellipse[1][1]) / 2 * pixel_size_um
    else:
        sma_diameter = outer_diameter if not has_sma_ring else np.sqrt(sma_area / np.pi) * 2 * pixel_size_um

    wall_thickness = (outer_diameter - inner_diameter) / 2
    sma_thickness = (sma_diameter - outer_diameter) / 2 if has_sma_ring else 0.0

    # Update vessel dict in-place
    vessel['outer_contour'] = outer_full.tolist()
    vessel['inner_contour'] = inner_full.tolist()
    vessel['sma_contour'] = sma_full.tolist()
    vessel['outer_diameter_um'] = float(outer_diameter)
    vessel['inner_diameter_um'] = float(inner_diameter)
    vessel['sma_diameter_um'] = float(sma_diameter)
    vessel['sma_thickness_um'] = float(sma_thickness)
    vessel['has_sma_ring'] = bool(has_sma_ring)
    vessel['wall_thickness_um'] = float(wall_thickness)
    vessel['outer_area_px'] = float(outer_area)
    vessel['inner_area_px'] = float(inner_area)
    vessel['wall_area_px'] = float(outer_area - inner_area)
    vessel['refined'] = True

    # Cleanup
    del cd31_roi, sma_roi, cd31_norm, sma_norm, cd31_labels, sma_labels, lumen_mask
    gc.collect()

    return True


def process_tile_at_scale(tile_x, tile_y, channel_cache, sam2_generator, scale_config, pixel_size_um=0.1725, adaptive_sobel_threshold=0.05):
    """
    Process a tile at given scale using downsampled channel cache.

    tile_x, tile_y are in SCALED coordinates (not full-res).
    The channel_cache holds data at BASE_SCALE and downsamples further as needed.

    Args:
        tile_x, tile_y: Tile origin in scaled coordinates
        channel_cache: DownsampledChannelCache with loaded channels
        sam2_generator: SAM2AutomaticMaskGenerator instance
        scale_config: Dict with scale parameters
        pixel_size_um: Physical pixel size in micrometers
        adaptive_sobel_threshold: Adaptive edge density threshold for pre-filtering
    """
    scale_factor = scale_config['scale_factor']
    min_diam = scale_config['min_diam_um']
    max_diam = scale_config['max_diam_um']
    tile_size = scale_config.get('tile_size', TILE_SIZE)
    dilation_mode = scale_config.get('dilation_mode', DILATION_MODE)
    sma_min_wall_intensity = scale_config.get('sma_min_wall_intensity', SMA_MIN_WALL_INTENSITY)

    # Effective pixel size at this scale (larger pixels at coarser scales)
    effective_pixel_size = pixel_size_um * scale_factor

    # Get all channels at this scale from cache
    tiles = {}
    for ch in [NUCLEAR, CD31, SMA, PM]:
        tiles[ch] = channel_cache.get_tile(tile_x, tile_y, tile_size, ch, scale_factor)

    # Check for valid tile data (all channels must be present)
    if any(tiles[ch] is None or tiles[ch].size == 0 for ch in [NUCLEAR, CD31, SMA, PM]):
        return []

    # Sobel-based pre-filter: skip tiles without vessel candidates
    # Vessels have strong edges (walls), so low edge density = no vessels
    global TILES_SKIPPED_BY_SOBEL
    sma_has_candidates = has_vessel_candidates(tiles[SMA], min_edge_density=adaptive_sobel_threshold)
    cd31_has_candidates = has_vessel_candidates(tiles[CD31], min_edge_density=adaptive_sobel_threshold)

    if not sma_has_candidates and not cd31_has_candidates:
        # Neither channel has sufficient edge density - skip expensive SAM2 inference
        TILES_SKIPPED_BY_SOBEL += 1
        del tiles
        gc.collect()
        return []

    # Normalize channels to uint8 (slide-wide photobleaching already applied to channel cache)
    sma_norm = normalize_channel(tiles[SMA])
    nuclear_norm = normalize_channel(tiles[NUCLEAR])
    cd31_norm = normalize_channel(tiles[CD31])
    pm_norm = normalize_channel(tiles[PM])

    # SAM2 input: grayscale as RGB (for detection)
    sma_rgb = cv2.cvtColor(sma_norm, cv2.COLOR_GRAY2RGB)
    cd31_rgb = cv2.cvtColor(cd31_norm, cv2.COLOR_GRAY2RGB)

    # Display RGB: multi-channel for visualization (R=SMA, G=CD31, B=nuclear)
    display_rgb = np.stack([sma_norm, cd31_norm, nuclear_norm], axis=-1)

    # Run SAM2 on both SMA and CD31 channels to find more vessels
    masks_sma = sam2_generator.generate(sma_rgb)
    masks_cd31 = sam2_generator.generate(cd31_rgb)

    # Tag masks with source channel
    for m in masks_sma:
        m['source'] = 'SMA'
    for m in masks_cd31:
        m['source'] = 'CD31'

    # Combine masks from both channels
    # Deduplication happens at the final merge step (merge_vessels_across_scales)
    masks = masks_sma + masks_cd31

    # Filter to lumens using GMM-based adaptive thresholding
    # This is more robust to staining intensity variation than fixed thresholds
    mask_arrays = [m['segmentation'] for m in masks]
    verification_results = verify_lumens_batch(
        mask_arrays, sma_norm, nuclear_norm, cd31_norm, pm_norm,
        scale_factor=scale_factor, threshold_ppv=0.5
    )

    lumens = []
    for i, (is_valid, stats) in enumerate(verification_results):
        if is_valid:
            lumens.append({'idx': i, 'mask': masks[i]['segmentation'], 'stats': stats, 'source': masks[i]['source']})

    if len(lumens) == 0:
        # Cleanup before returning
        del tiles, sma_norm, nuclear_norm, cd31_norm, pm_norm, sma_rgb, cd31_rgb, display_rgb, masks
        gc.collect()
        return []

    # Watershed expansion (CD31 outer wall)
    labels = dilate_until_signal_drops(lumens, cd31_norm, scale_factor, mode=dilation_mode)

    # SMA ring detection (3rd contour) — expands from lumen on SMA channel
    sma_labels = dilate_until_signal_drops(lumens, sma_norm, scale_factor,
                                           min_wall_intensity=sma_min_wall_intensity,
                                           mode=dilation_mode)

    # First pass: collect all candidate vessels with their CD31 enrichment ratios
    # for GMM-based adaptive thresholding
    candidates = []
    cd31_enrichment_ratios = []
    n_filtered_no_cd31_lining = 0  # Count lumens filtered for lacking CD31 endothelial lining

    for idx, lumen in enumerate(lumens):
        label_id = idx + 1

        wall_mask = (labels == label_id).astype(np.uint8)
        outer_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(outer_contours) == 0:
            continue
        outer_contour = max(outer_contours, key=cv2.contourArea)

        inner_mask = lumen['mask'].astype(np.uint8)
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(inner_contours) == 0:
            continue
        inner_contour = max(inner_contours, key=cv2.contourArea)

        # Extract SMA contour from sma_labels
        sma_mask = (sma_labels == label_id).astype(np.uint8)
        sma_contours_found, _ = cv2.findContours(sma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(sma_contours_found) > 0:
            sma_contour = max(sma_contours_found, key=cv2.contourArea)
        else:
            sma_contour = inner_contour  # Fallback: no SMA = lumen boundary

        # Determine if SMA ring is meaningful (expanded beyond lumen)
        sma_area_tile = cv2.contourArea(sma_contour)
        inner_area_tile_check = cv2.contourArea(inner_contour)
        has_sma_ring = sma_area_tile > inner_area_tile_check * 1.05  # >5% larger than lumen

        # Verify outer contour actually surrounds inner contour
        # Check that inner bounding box is contained within outer bounding box
        outer_bbox = cv2.boundingRect(outer_contour)  # (x, y, w, h)
        inner_bbox = cv2.boundingRect(inner_contour)

        outer_x1, outer_y1 = outer_bbox[0], outer_bbox[1]
        outer_x2, outer_y2 = outer_bbox[0] + outer_bbox[2], outer_bbox[1] + outer_bbox[3]
        inner_x1, inner_y1 = inner_bbox[0], inner_bbox[1]
        inner_x2, inner_y2 = inner_bbox[0] + inner_bbox[2], inner_bbox[1] + inner_bbox[3]

        # Inner must be fully inside outer (with small tolerance for edge effects)
        tolerance = 2  # pixels
        if not (inner_x1 >= outer_x1 - tolerance and inner_y1 >= outer_y1 - tolerance and
                inner_x2 <= outer_x2 + tolerance and inner_y2 <= outer_y2 + tolerance):
            continue  # Skip - outer doesn't properly surround inner

        # Compute CD31 wall enrichment (wall intensity / lumen intensity)
        # Wall region = expanded mask minus lumen
        wall_only = wall_mask.astype(bool) & ~lumen['mask']
        if wall_only.sum() < 10:
            continue  # No wall region

        cd31_in_wall = cd31_norm[wall_only].mean()
        cd31_in_lumen = cd31_norm[lumen['mask']].mean()
        cd31_enrichment = cd31_in_wall / max(cd31_in_lumen, 1e-6)

        # Check for CD31+ endothelial lining at lumen edge
        # This distinguishes real vessels (with endothelium) from tissue tears
        cd31_edge_coverage, cd31_edge_mean, cd31_bg = compute_cd31_edge_coverage(
            lumen['mask'], cd31_norm, ring_width=3
        )

        # Skip if insufficient CD31 lining (likely a tear, not a vessel)
        if cd31_edge_coverage < CD31_EDGE_COVERAGE_THRESHOLD:
            n_filtered_no_cd31_lining += 1
            continue

        # Store candidate for second pass
        candidates.append({
            'lumen': lumen,
            'wall_mask': wall_mask,
            'outer_contour': outer_contour,
            'inner_contour': inner_contour,
            'sma_contour': sma_contour,
            'has_sma_ring': has_sma_ring,
            'cd31_in_wall': cd31_in_wall,
            'cd31_in_lumen': cd31_in_lumen,
            'cd31_enrichment': cd31_enrichment,
            'cd31_edge_coverage': cd31_edge_coverage,
        })
        cd31_enrichment_ratios.append(cd31_enrichment)

    # Log how many were filtered for lacking CD31 lining
    if n_filtered_no_cd31_lining > 0:
        print(f"    Filtered {n_filtered_no_cd31_lining} lumens lacking CD31+ endothelial lining (likely tears)")

    # Second pass: Apply GMM-based CD31 enrichment classification
    # For enrichment, we want the HIGHER ratio class (wall > lumen)
    if len(candidates) > 0:
        cd31_enrichment_ratios = np.array(cd31_enrichment_ratios)
        cd31_valid = gmm_classify_ratios(
            cd31_enrichment_ratios, threshold_ppv=0.5, min_samples=50, lower_is_positive=False
        )
    else:
        cd31_valid = np.array([], dtype=bool)

    # Extract vessels that pass CD31 enrichment check
    vessels = []
    for cand_idx, candidate in enumerate(candidates):
        # Skip if CD31 enrichment check fails
        if not cd31_valid[cand_idx]:
            continue

        lumen = candidate['lumen']
        wall_mask = candidate['wall_mask']
        outer_contour = candidate['outer_contour']
        inner_contour = candidate['inner_contour']
        sma_contour = candidate['sma_contour']
        has_sma_ring = candidate['has_sma_ring']

        # Compute measurements at tile scale, then convert to full resolution
        outer_area_tile = cv2.contourArea(outer_contour)
        inner_area_tile = cv2.contourArea(inner_contour)

        # Convert areas back to full resolution (multiply by scale_factor^2)
        area_scale = scale_factor * scale_factor
        outer_area = outer_area_tile * area_scale
        inner_area = inner_area_tile * area_scale
        wall_area = outer_area - inner_area

        # Fit ellipses
        if len(outer_contour) >= 5:
            outer_ellipse = cv2.fitEllipse(outer_contour)
            outer_diameter = (outer_ellipse[1][0] + outer_ellipse[1][1]) / 2 * effective_pixel_size
        else:
            outer_diameter = np.sqrt(outer_area / np.pi) * 2 * pixel_size_um

        if len(inner_contour) >= 5:
            inner_ellipse = cv2.fitEllipse(inner_contour)
            inner_diameter = (inner_ellipse[1][0] + inner_ellipse[1][1]) / 2 * effective_pixel_size
        else:
            inner_diameter = np.sqrt(inner_area / np.pi) * 2 * pixel_size_um

        # Filter by diameter range for this scale
        if outer_diameter < min_diam or outer_diameter > max_diam:
            continue

        # Skip invalid vessels (outer must be > inner)
        if outer_diameter <= inner_diameter:
            continue

        # Note: Diameter ratio filtering now happens adaptively after all vessels
        # are collected, using adaptive_diameter_ratio_filter() with GMM

        wall_thickness = (outer_diameter - inner_diameter) / 2

        # Get centroid at tile scale
        M = cv2.moments(outer_contour)
        if M['m00'] > 0:
            cx_tile = int(M['m10'] / M['m00'])
            cy_tile = int(M['m01'] / M['m00'])
        else:
            cx_tile, cy_tile = inner_contour.mean(axis=0)[0].astype(int)

        # Scale contours and centroid back to full resolution
        # tile coords * scale_factor = full res coords
        cx_full = int(cx_tile * scale_factor)
        cy_full = int(cy_tile * scale_factor)

        outer_full = (outer_contour * scale_factor).astype(int)
        inner_full = (inner_contour * scale_factor).astype(int)
        sma_full = (sma_contour * scale_factor).astype(int)

        # SMA measurements
        sma_area_px = cv2.contourArea(sma_contour) * area_scale
        if len(sma_contour) >= 5 and has_sma_ring:
            sma_ellipse = cv2.fitEllipse(sma_contour)
            sma_diameter = (sma_ellipse[1][0] + sma_ellipse[1][1]) / 2 * effective_pixel_size
        else:
            sma_diameter = float(outer_diameter) if not has_sma_ring else np.sqrt(sma_area_px / np.pi) * 2 * pixel_size_um
        sma_thickness = (sma_diameter - outer_diameter) / 2 if has_sma_ring else 0.0

        # Global coordinates (tile_x/tile_y are in scaled coords, convert to full res)
        global_x = tile_x * scale_factor + cx_full
        global_y = tile_y * scale_factor + cy_full

        vessel = {
            'uid': f'vessel_{global_x}_{global_y}',
            'scale': scale_config['name'],
            'scale_factor': scale_factor,
            'source_channel': lumen['source'],  # 'SMA' or 'CD31'
            'tile_x': tile_x,  # in scaled coordinates
            'tile_y': tile_y,
            'local_center': [int(cx_tile), int(cy_tile)],  # in tile coordinates
            'global_center': [int(global_x), int(global_y)],  # in full-res coordinates
            'outer_contour': outer_full.tolist(),
            'inner_contour': inner_full.tolist(),
            'sma_contour': sma_full.tolist(),
            'outer_diameter_um': float(outer_diameter),
            'inner_diameter_um': float(inner_diameter),
            'sma_diameter_um': float(sma_diameter),
            'sma_thickness_um': float(sma_thickness),
            'has_sma_ring': bool(has_sma_ring),
            'wall_thickness_um': float(wall_thickness),
            'outer_area_px': float(outer_area),
            'inner_area_px': float(inner_area),
            'wall_area_px': float(wall_area),
            'cd31_edge_coverage': float(candidate.get('cd31_edge_coverage', 0)),
            **lumen['stats']
        }

        # Save crop with contours (use tile-scale contours for drawing on tile-scale image)
        crop_path = save_vessel_crop(vessel, display_rgb, outer_contour, inner_contour, scale_factor,
                                     sma_contour=sma_contour if has_sma_ring else None)
        vessel['crop_path'] = crop_path

        vessels.append(vessel)

    # Cleanup large arrays before returning
    del tiles, sma_norm, nuclear_norm, cd31_norm, pm_norm, sma_rgb, cd31_rgb, display_rgb, masks, labels, sma_labels
    gc.collect()

    return vessels

def _vessel_contour_global(vessel):
    """
    Get vessel outer contour in global (mosaic) coordinates.

    Stored contours are tile-local at full-res scale. This adds the tile
    origin offset to produce true global coordinates for IoU computation.
    """
    c = np.array(vessel['outer_contour']).reshape(-1, 1, 2).astype(np.int32)
    tile_offset_x = vessel['tile_x'] * vessel['scale_factor']
    tile_offset_y = vessel['tile_y'] * vessel['scale_factor']
    c = c.copy()
    c[:, :, 0] += tile_offset_x
    c[:, :, 1] += tile_offset_y
    return c


def compute_vessel_iou(v1, v2):
    """
    Compute IoU between two vessels using rendered contour masks.

    Translates tile-local contours to global coordinates before comparison.

    Args:
        v1, v2: Vessel dicts with 'outer_contour' keys

    Returns:
        float: IoU value between 0 and 1
    """
    from segmentation.utils.multiscale import compute_iou_contours

    c1 = _vessel_contour_global(v1)
    c2 = _vessel_contour_global(v2)

    if c1.shape[0] < 3 or c2.shape[0] < 3:
        return 0.0

    return compute_iou_contours(c1, c2)


def deduplicate_vessels_within_scale(vessels, distance_threshold_px=None, pixel_size_um=PIXEL_SIZE_UM):
    """
    Remove duplicate vessels detected in overlapping tiles within the same scale.

    Uses centroid proximity (cKDTree) to find candidates, then contour IoU
    to confirm duplicates. Keeps the vessel with the largest outer area.

    Args:
        vessels: List of vessel dicts from a single scale
        distance_threshold_px: Max centroid distance for duplicate candidates.
            Default: half the median vessel outer diameter in pixels.
    Returns:
        List of deduplicated vessels
    """
    if len(vessels) <= 1:
        return vessels

    from segmentation.utils.multiscale import compute_iou_contours

    # Build spatial index from global centers
    centers = np.array([v['global_center'] for v in vessels])
    tree = cKDTree(centers)

    # Auto-compute distance threshold from median vessel size
    if distance_threshold_px is None:
        diameters_px = [v['outer_diameter_um'] / pixel_size_um for v in vessels]
        distance_threshold_px = np.median(diameters_px) * 0.5

    # Sort by outer area descending (largest-wins strategy)
    indexed_vessels = sorted(enumerate(vessels), key=lambda x: x[1].get('outer_area_px', 0), reverse=True)

    kept = []          # indices of kept vessels (ordered)
    kept_set = set()   # same indices for O(1) lookup
    removed = set()    # indices of removed vessels

    for orig_idx, vessel in indexed_vessels:
        if orig_idx in removed:
            continue
        kept.append(orig_idx)
        kept_set.add(orig_idx)

        # Find nearby vessels
        center = np.array(vessel['global_center'])
        nearby = tree.query_ball_point(center, distance_threshold_px)

        # Compute global contour once for this vessel
        c1 = _vessel_contour_global(vessel)

        for j in nearby:
            if j == orig_idx or j in removed or j in kept_set:
                continue
            c2 = _vessel_contour_global(vessels[j])
            iou = compute_iou_contours(c1, c2)
            if iou > 0.5:  # >50% contour overlap = same vessel
                removed.add(j)

    result = [vessels[i] for i in kept]
    if len(removed) > 0:
        print(f"    Within-scale dedup: {len(vessels)} -> {len(result)} ({len(removed)} duplicates removed)")
    return result


def merge_vessels_across_scales(vessels, iou_threshold=None, coverage_threshold=None, pixel_size_um=PIXEL_SIZE_UM):
    """
    Merge vessels detected at different scales, keeping the finest segmentation that captures the full vessel.

    Uses Union-Find clustering to group overlapping vessels, then selects the finest-scale
    detection that covers ≥90% of the coarsest detection's area (ensuring completeness).

    Special handling for 1/2 scale: These finest-scale detections are only kept if they
    overlap with at least one detection at a coarser scale (1/4 or coarser). This prevents
    false positives from noise that only appears at the finest scale.

    Uses adaptive IoU thresholding via Otsu's method when iou_threshold is None.
    Uses cKDTree spatial indexing for O(n log n) performance instead of O(n²).

    Args:
        vessels: List of vessel dicts
        iou_threshold: Fixed IoU threshold for merging, or None for adaptive
        coverage_threshold: Min coverage ratio to consider fine-scale complete (default: COVERAGE_THRESHOLD)

    Returns:
        List of merged vessels
    """
    if len(vessels) == 0:
        return []

    if len(vessels) == 1:
        return vessels

    if coverage_threshold is None:
        coverage_threshold = COVERAGE_THRESHOLD

    # Sort by scale (FINEST FIRST - lower denominator = finer)
    sorted_vessels = sorted(
        vessels,
        key=lambda v: float(v['scale'].split('/')[1]) if '/' in v['scale'] else 1,
        reverse=False  # Finest first (1/4 before 1/8 before 1/16, etc.)
    )
    n = len(sorted_vessels)

    # Build spatial index from vessel centers
    centers = np.array([v['global_center'] for v in sorted_vessels])
    tree = cKDTree(centers)

    # Compute max search radius (largest vessel diameter in pixels * 1.5)
    max_diam_px = max(v['outer_diameter_um'] / pixel_size_um for v in sorted_vessels)
    search_radius = max_diam_px * 1.5

    print(f"  Building spatial index for {n} vessels (search radius: {search_radius:.0f}px)...")

    # Pre-compute all neighbor lists (avoids duplicate queries)
    neighbor_lists = tree.query_ball_point(centers, search_radius)

    # First pass: collect pairwise IoUs using cached neighbor lists
    all_ious = []
    iou_cache = {}  # Cache IoU values to avoid recomputation in Union-Find pass

    for i in range(n):
        v1 = sorted_vessels[i]
        nearby_indices = neighbor_lists[i]

        for j in nearby_indices:
            if j <= i:  # Skip self and already-checked pairs
                continue

            pair_key = (i, j)
            if pair_key in iou_cache:
                continue

            v2 = sorted_vessels[j]
            iou = compute_vessel_iou(v1, v2)
            iou_cache[pair_key] = iou
            if iou > 0:
                all_ious.append(iou)

    # Compute adaptive threshold if not provided
    if iou_threshold is None:
        if len(all_ious) > 0:
            all_ious_arr = np.array(all_ious)
            adaptive_threshold = compute_adaptive_iou_threshold(all_ious_arr)
            print(f"  Adaptive IoU threshold (Otsu): {adaptive_threshold:.3f}")
            print(f"    IoU stats: min={all_ious_arr.min():.3f}, max={all_ious_arr.max():.3f}, mean={all_ious_arr.mean():.3f}")
            iou_threshold = adaptive_threshold
        else:
            iou_threshold = 0.3
            print(f"  No overlapping vessels found, using default IoU threshold: {iou_threshold}")
    else:
        print(f"  Using fixed IoU threshold: {iou_threshold}")

    # Union-Find data structure for clustering overlapping vessels
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Group vessels with IoU > threshold using Union-Find (reuse cached IoUs)
    for i in range(n):
        for j in neighbor_lists[i]:
            if j <= i:
                continue
            pair_key = (i, j)
            iou = iou_cache.get(pair_key)
            if iou is None:
                iou = compute_vessel_iou(sorted_vessels[i], sorted_vessels[j])
            if iou > iou_threshold:
                union(i, j)

    # Build clusters from Union-Find
    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Select best vessel from each cluster
    # Note: 1/2 scale vessels require corroboration from coarser scales
    merged = []
    uncorroborated_half_scale = 0

    for root, indices in clusters.items():
        cluster_vessels = [sorted_vessels[i] for i in indices]

        # Check if cluster has any coarser-scale corroboration (scale_factor > 2)
        has_coarser_corroboration = any(v.get('scale_factor', 1) > 2 for v in cluster_vessels)

        if len(indices) == 1:
            vessel = cluster_vessels[0]
            # Skip 1/2 scale vessels without coarser-scale corroboration
            if vessel.get('scale_factor', 1) == 2 and not has_coarser_corroboration:
                uncorroborated_half_scale += 1
                continue
            merged.append(vessel)
            continue

        # For multi-vessel clusters: skip if ALL vessels are 1/2 scale (no corroboration)
        if not has_coarser_corroboration:
            uncorroborated_half_scale += len(cluster_vessels)
            continue

        # Find coarsest vessel in cluster (reference for coverage check)
        # Higher scale_factor = coarser (e.g., 64 > 32 > 16 > 8 > 4 > 2)
        coarsest = max(cluster_vessels, key=lambda v: v.get('scale_factor', 1))
        coarsest_area = coarsest.get('outer_area_px', 0)

        # Find finest vessel that covers ≥coverage_threshold of coarsest
        # cluster_vessels is already sorted finest-first (from sorted_vessels)
        selected = None
        for v in cluster_vessels:
            if coarsest_area > 0:
                coverage = v.get('outer_area_px', 0) / coarsest_area
            else:
                coverage = 1.0

            if coverage >= coverage_threshold:
                selected = v
                break

        # Fallback to coarsest if none meet threshold
        if selected is None:
            selected = coarsest

        merged.append(selected)

    if uncorroborated_half_scale > 0:
        print(f"    Filtered {uncorroborated_half_scale} uncorroborated 1/2 scale vessels")

    # Report scale distribution in merged results
    scale_counts = {}
    for v in merged:
        scale = v.get('scale', 'unknown')
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
    scale_summary = ', '.join(f"{s}: {c}" for s, c in sorted(scale_counts.items()))

    print(f"  Merged {len(vessels)} -> {len(merged)} vessels (coverage_threshold={coverage_threshold:.0%})")
    print(f"    Scale distribution: {scale_summary}")
    return merged


def analyze_vessel_network(vessels, mosaic_size, pixel_size_um=0.1725):
    """
    Analyze vessel network topology by skeletonizing merged vessel masks.

    Creates a combined mask from all detected vessel outer contours,
    applies skeletonization, and extracts network metrics.

    Args:
        vessels: List of vessel dictionaries with 'outer_contour' in full-res coordinates
        mosaic_size: Tuple (width, height) of full resolution mosaic
        pixel_size_um: Physical pixel size in micrometers

    Returns:
        dict with network metrics and skeleton array
    """
    print("\n" + "=" * 40)
    print("Analyzing vessel network topology")
    print("=" * 40)

    if len(vessels) == 0:
        print("No vessels to analyze")
        return {
            'total_length_um': 0.0,
            'total_length_mm': 0.0,
            'total_length_px': 0,
            'num_branch_points': 0,
            'num_endpoints': 0,
            'num_vessels': 0,
            'skeleton': None
        }

    # Determine working scale - use 1/8 scale to keep memory manageable
    # Full res could be 100k+ pixels which would be too large
    work_scale = 8
    work_width = mosaic_size[0] // work_scale
    work_height = mosaic_size[1] // work_scale
    effective_pixel_size = pixel_size_um * work_scale

    print(f"Working at 1/{work_scale} scale: {work_width} x {work_height}")
    print(f"Effective pixel size: {effective_pixel_size:.3f} um")

    # Create combined vessel mask at working scale
    print("Creating combined vessel mask...")
    combined_mask = np.zeros((work_height, work_width), dtype=np.uint8)

    for v in tqdm(vessels, desc="Drawing contours"):
        # Get contour in global coordinates (stored contours are tile-local)
        global_contour = _vessel_contour_global(v)
        if global_contour.size == 0:
            continue

        # Scale contour from full-res global to working scale
        scaled_contour = (global_contour / work_scale).astype(np.int32)

        # Draw filled contour on mask
        cv2.drawContours(combined_mask, [scaled_contour], -1, 255, thickness=cv2.FILLED)

    # Check if we have any vessel pixels
    vessel_pixels = combined_mask.sum() // 255
    print(f"Total vessel pixels in mask: {vessel_pixels:,}")

    if vessel_pixels == 0:
        print("No vessel pixels found in mask")
        return {
            'total_length_um': 0.0,
            'total_length_mm': 0.0,
            'total_length_px': 0,
            'num_branch_points': 0,
            'num_endpoints': 0,
            'num_vessels': len(vessels),
            'skeleton': None
        }

    # Apply skeletonization
    print("Applying skeletonization...")
    binary_mask = combined_mask > 0
    skeleton = skeletonize(binary_mask)

    # Count skeleton pixels (= total length in pixels)
    skeleton_pixels = skeleton.sum()
    total_length_px = int(skeleton_pixels)
    total_length_um = skeleton_pixels * effective_pixel_size

    print(f"Skeleton pixels: {skeleton_pixels:,}")
    print(f"Total vessel length: {total_length_um:.1f} um ({total_length_um/1000:.2f} mm)")

    # Analyze network topology using neighbor counting
    # A 3x3 kernel to count neighbors for each skeleton pixel
    neighbor_kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    # Convert skeleton to uint8 for convolution
    skeleton_uint8 = skeleton.astype(np.uint8)

    # Count neighbors for each pixel
    neighbor_count = convolve(skeleton_uint8, neighbor_kernel, mode='constant', cval=0)

    # Branch points: skeleton pixels with more than 2 neighbors
    # These are junction points where vessels meet
    branch_point_mask = skeleton & (neighbor_count > 2)
    num_branch_points = int(branch_point_mask.sum())

    # Endpoints: skeleton pixels with exactly 1 neighbor
    # These are terminal points of vessel segments
    endpoint_mask = skeleton & (neighbor_count == 1)
    num_endpoints = int(endpoint_mask.sum())

    print(f"Branch points (>2 neighbors): {num_branch_points}")
    print(f"Endpoints (1 neighbor): {num_endpoints}")

    # Save skeleton as PNG
    skeleton_path = os.path.join(OUTPUT_DIR, 'vessel_skeleton.png')
    skeleton_img = (skeleton * 255).astype(np.uint8)
    cv2.imwrite(skeleton_path, skeleton_img)
    print(f"Saved skeleton to: {skeleton_path}")

    # Save skeleton as numpy array for further analysis
    skeleton_npy_path = os.path.join(OUTPUT_DIR, 'vessel_skeleton.npy')
    np.save(skeleton_npy_path, skeleton)
    print(f"Saved skeleton array to: {skeleton_npy_path}")

    # Optional: save an overlay image showing skeleton on top of mask
    overlay_path = os.path.join(OUTPUT_DIR, 'vessel_skeleton_overlay.png')
    overlay = np.zeros((work_height, work_width, 3), dtype=np.uint8)
    overlay[binary_mask] = [50, 50, 50]  # Gray for vessel area
    overlay[skeleton] = [0, 255, 0]  # Green for skeleton
    overlay[branch_point_mask] = [255, 0, 0]  # Red for branch points
    overlay[endpoint_mask] = [0, 0, 255]  # Blue for endpoints
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved skeleton overlay to: {overlay_path}")

    # Build results dictionary
    network_metrics = {
        'total_length_um': float(total_length_um),
        'total_length_mm': float(total_length_um / 1000),
        'total_length_px': total_length_px,
        'num_branch_points': num_branch_points,
        'num_endpoints': num_endpoints,
        'num_vessels': len(vessels),
        'work_scale': work_scale,
        'effective_pixel_size_um': effective_pixel_size,
        'skeleton_path': skeleton_path,
        'skeleton_npy_path': skeleton_npy_path,
        'overlay_path': overlay_path,
    }

    print("\nNetwork metrics summary:")
    print(f"  Total length: {network_metrics['total_length_mm']:.2f} mm")
    print(f"  Branch points: {num_branch_points}")
    print(f"  Endpoints: {num_endpoints}")
    print(f"  Vessels analyzed: {len(vessels)}")

    return network_metrics


# ==============================================================================
# Multi-GPU Vessel Processing
# ==============================================================================

def _vessel_gpu_worker(gpu_id, input_queue, output_queue, stop_event,
                       channel_shm_info, base_scale, full_res_size,
                       output_dir, sam2_checkpoint_path, sam2_config,
                       sam2_generator_params, adaptive_sobel_threshold,
                       tile_sleep, reset_interval, pixel_size_um=0.1725):
    """
    GPU worker process for vessel detection.

    Each worker:
    1. Pins to assigned GPU via CUDA_VISIBLE_DEVICES
    2. Attaches to shared memory channel arrays
    3. Loads SAM2 model on cuda:0 (from local /tmp copy)
    4. Processes tiles from input_queue, puts vessels on output_queue

    Workers are started once and reused across all scales to avoid
    repeated 30s SAM2 load times.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import queue as queue_module
    from multiprocessing.shared_memory import SharedMemory

    worker_name = f"GPU-{gpu_id}"
    print(f"[{worker_name}] Starting vessel worker...", flush=True)

    # Attach to shared memory channels
    shared_memories = []
    channel_arrays = {}
    try:
        for ch_idx_str, info in channel_shm_info.items():
            ch_idx = int(ch_idx_str) if isinstance(ch_idx_str, str) else ch_idx_str
            shm = SharedMemory(name=info['shm_name'])
            arr = np.ndarray(tuple(info['shape']), dtype=np.dtype(info['dtype']), buffer=shm.buf)
            shared_memories.append(shm)
            channel_arrays[ch_idx] = arr
        print(f"[{worker_name}] Attached to {len(channel_arrays)} shared memory channels", flush=True)
    except Exception as e:
        print(f"[{worker_name}] Failed to attach shared memory: {e}", flush=True)
        output_queue.put({'status': 'init_error', 'gpu_id': gpu_id, 'error': str(e)})
        return

    # Create SharedChannelCache
    channel_cache = SharedChannelCache(channel_arrays, base_scale, tuple(full_res_size))

    # Ensure crops dir exists (workers save crops concurrently)
    os.makedirs(os.path.join(output_dir, 'crops'), exist_ok=True)

    # Load SAM2 model (from local /tmp copy for speed)
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        print(f"[{worker_name}] Loading SAM2 from {sam2_checkpoint_path}...", flush=True)
        sam2_model = build_sam2(sam2_config, sam2_checkpoint_path, device="cuda")
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model, **sam2_generator_params)
        print(f"[{worker_name}] SAM2 loaded on GPU {gpu_id}", flush=True)
    except Exception as e:
        print(f"[{worker_name}] Failed to load SAM2: {e}", flush=True)
        output_queue.put({'status': 'init_error', 'gpu_id': gpu_id, 'error': str(e)})
        for shm in shared_memories:
            shm.close()
        return

    # Signal ready
    output_queue.put({'status': 'ready', 'gpu_id': gpu_id})

    tiles_processed = 0
    tiles_since_reset = 0

    while not stop_event.is_set():
        try:
            try:
                work_item = input_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue

            if work_item is None:
                print(f"[{worker_name}] Received shutdown signal", flush=True)
                break

            tile_x, tile_y, scale_config = work_item

            try:
                sobel_before = TILES_SKIPPED_BY_SOBEL
                vessels = process_tile_at_scale(
                    tile_x, tile_y, channel_cache, mask_generator, scale_config,
                    pixel_size_um=pixel_size_um,
                    adaptive_sobel_threshold=adaptive_sobel_threshold
                )

                output_queue.put({
                    'status': 'success',
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'scale': scale_config['name'],
                    'vessels': vessels,
                    'sobel_skipped': TILES_SKIPPED_BY_SOBEL > sobel_before,
                })
            except Exception as e:
                print(f"[{worker_name}] Error at ({tile_x}, {tile_y}): {e}", flush=True)
                import traceback
                traceback.print_exc()
                output_queue.put({
                    'status': 'error',
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'scale': scale_config['name'],
                    'error': str(e),
                })

            tiles_processed += 1
            tiles_since_reset += 1

            # Memory cleanup after each tile
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # WSL2 sleep workaround
            if tile_sleep > 0:
                time.sleep(tile_sleep)

            # Periodic SAM2 predictor reset to prevent VRAM fragmentation
            if reset_interval > 0 and tiles_since_reset >= reset_interval:
                if hasattr(mask_generator, 'predictor') and hasattr(mask_generator.predictor, 'reset_predictor'):
                    mask_generator.predictor.reset_predictor()
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                tiles_since_reset = 0
                if tile_sleep > 0:
                    time.sleep(tile_sleep * 3)  # Longer pause on reset
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_cached = torch.cuda.memory_reserved() / 1e9
                print(f"[{worker_name}] VRAM reset after {tiles_processed} tiles: "
                      f"alloc={vram_used:.1f}GB, cached={vram_cached:.1f}GB", flush=True)

        except Exception as e:
            print(f"[{worker_name}] Worker loop error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # Cleanup
    print(f"[{worker_name}] Shutting down, processed {tiles_processed} tiles", flush=True)
    del mask_generator, sam2_model
    gc.collect()
    torch.cuda.empty_cache()

    channel_cache.release()
    for shm in shared_memories:
        shm.close()


class VesselMultiGPUProcessor:
    """
    Multi-GPU processor for vessel detection.

    Workers are started once and reused across all scales (avoids 30s SAM2 reload).
    Uses shared memory for channel data (zero-copy tile reads).

    Follows the same patterns as MultiGPUTileProcessor:
    - Staggered worker init (start one, wait for ready, start next)
    - SAM2 checkpoint copied to /tmp for fast loading
    - Proper queue.Empty handling, _workers_started guard

    Usage:
        with VesselMultiGPUProcessor(...) as processor:
            for scale_config in scales:
                for tile_x, tile_y in tissue_tiles:
                    processor.submit_tile(tile_x, tile_y, scale_config)
                for _ in range(n_tiles):
                    result = processor.collect_result()
    """

    def __init__(self, num_gpus, channel_shm_info, base_scale, full_res_size,
                 output_dir, sam2_checkpoint_path, sam2_config, sam2_generator_params,
                 adaptive_sobel_threshold, tile_sleep=0.0, reset_interval=0,
                 pixel_size_um=0.1725):
        self.num_gpus = num_gpus
        self.channel_shm_info = channel_shm_info
        self.base_scale = base_scale
        self.full_res_size = full_res_size
        self.output_dir = output_dir
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_config = sam2_config
        self.sam2_generator_params = sam2_generator_params
        self.adaptive_sobel_threshold = adaptive_sobel_threshold
        self.tile_sleep = tile_sleep
        self.reset_interval = reset_interval
        self.pixel_size_um = pixel_size_um

        self.workers = []
        self.input_queue = None
        self.output_queue = None
        self.stop_event = None
        self.tiles_submitted = 0
        self._local_checkpoint_path = None
        self._workers_started = False

    def _copy_checkpoint_to_local(self):
        """Copy SAM2 checkpoint to local /tmp for faster loading from network mounts."""
        import shutil
        from pathlib import Path

        checkpoint_path = Path(self.sam2_checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        local_dir = Path("/tmp") / f"sam2_vessel_cache_{os.getpid()}"
        local_dir.mkdir(exist_ok=True)
        local_path = local_dir / checkpoint_path.name

        if local_path.exists():
            print(f"  SAM2 checkpoint already cached at {local_path}", flush=True)
            self._local_checkpoint_path = local_path
            return str(local_path)

        print(f"  Copying SAM2 checkpoint to /tmp...", flush=True)
        start = time.time()
        shutil.copy2(checkpoint_path, local_path)
        print(f"  Copied in {time.time() - start:.1f}s", flush=True)
        self._local_checkpoint_path = local_path
        return str(local_path)

    def _cleanup_local_checkpoint(self):
        """Remove local checkpoint copy."""
        import shutil
        if self._local_checkpoint_path and self._local_checkpoint_path.exists():
            try:
                shutil.rmtree(self._local_checkpoint_path.parent)
            except Exception as e:
                print(f"  Warning: failed to cleanup checkpoint: {e}", flush=True)

    def start(self):
        """Start worker processes with staggered initialization.

        Starts workers one at a time, waiting for each to signal ready before
        starting the next. This avoids GPU contention during SAM2 model loading.
        """
        import multiprocessing
        import queue as queue_module
        ctx = multiprocessing.get_context('spawn')

        # Copy SAM2 checkpoint to local /tmp (once, shared by all workers)
        local_checkpoint = self._copy_checkpoint_to_local()

        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.stop_event = ctx.Event()

        print(f"Starting {self.num_gpus} vessel GPU workers (staggered)...", flush=True)

        ready_count = 0
        errors = []

        for gpu_id in range(self.num_gpus):
            p = ctx.Process(
                target=_vessel_gpu_worker,
                args=(
                    gpu_id,
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.channel_shm_info,
                    self.base_scale,
                    self.full_res_size,
                    self.output_dir,
                    local_checkpoint,
                    self.sam2_config,
                    self.sam2_generator_params,
                    self.adaptive_sobel_threshold,
                    self.tile_sleep,
                    self.reset_interval,
                    self.pixel_size_um,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            print(f"  Started vessel worker GPU-{gpu_id} (PID: {p.pid})", flush=True)

            # Staggered: wait for THIS worker to be ready before starting next
            try:
                msg = self.output_queue.get(timeout=180)  # 3 min for SAM2 load
                if msg.get('status') == 'ready':
                    ready_count += 1
                    print(f"  Worker GPU-{msg['gpu_id']} ready ({ready_count}/{self.num_gpus})", flush=True)
                elif msg.get('status') == 'init_error':
                    errors.append(f"GPU-{msg['gpu_id']}: {msg['error']}")
            except queue_module.Empty:
                errors.append(f"GPU-{gpu_id}: timeout waiting for ready")

        if errors:
            self.stop()
            raise RuntimeError(f"Worker init errors: {errors}")

        if ready_count < self.num_gpus:
            self.stop()
            raise RuntimeError(f"Only {ready_count}/{self.num_gpus} workers ready")

        print(f"All {self.num_gpus} vessel workers ready", flush=True)
        self._workers_started = True

    def submit_tile(self, tile_x, tile_y, scale_config):
        """Submit a tile for processing."""
        self.input_queue.put((tile_x, tile_y, scale_config))
        self.tiles_submitted += 1

    def collect_result(self, timeout=300):
        """Collect one tile result from workers, filtering stray init messages."""
        if not self._workers_started:
            raise RuntimeError("Call start() first")

        import queue as queue_module
        start = time.time()
        remaining = timeout

        while remaining > 0:
            try:
                result = self.output_queue.get(timeout=min(remaining, 1.0))
                status = result.get('status')
                if status in ('success', 'error'):
                    return result
                if status in ('ready', 'init_error'):
                    continue  # Skip stray init messages
                return result  # Unknown status, pass through
            except queue_module.Empty:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    return None
                continue
        return None

    def stop(self):
        """Stop all workers."""
        # Send sentinels before setting stop_event (avoids race)
        if self.input_queue:
            for _ in range(self.num_gpus):
                try:
                    self.input_queue.put(None, timeout=1.0)
                except Exception:
                    pass

        if self.stop_event:
            self.stop_event.set()

        for p in self.workers:
            p.join(timeout=15)
            if p.is_alive():
                print(f"  Worker {p.pid} did not stop, terminating", flush=True)
                p.terminate()

        self.workers.clear()
        self._cleanup_local_checkpoint()
        self._workers_started = False
        print("All vessel workers stopped", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def _scan_tissue_tiles(channel_cache, scale_factor, tile_size, stride, mosaic_size):
    """Scan tile grid and return list of (tile_x, tile_y) for tissue-containing tiles.

    Args:
        channel_cache: Channel cache with get_tile() method
        scale_factor: Current processing scale
        tile_size: Tile size at this scale
        stride: Tile stride (tile_size * (1 - overlap))
        mosaic_size: (width, height) at full resolution

    Returns:
        List of (tile_x, tile_y) tuples for tiles that contain tissue
    """
    scaled_mosaic_x = mosaic_size[0] // scale_factor
    scaled_mosaic_y = mosaic_size[1] // scale_factor
    n_tiles_x = max(1, (scaled_mosaic_x - tile_size) // stride + 1)
    n_tiles_y = max(1, (scaled_mosaic_y - tile_size) // stride + 1)

    print(f"Scaled mosaic: {scaled_mosaic_x} x {scaled_mosaic_y}", flush=True)
    print(f"Tile grid: {n_tiles_x} x {n_tiles_y} = {n_tiles_x * n_tiles_y} tiles", flush=True)

    print("Identifying tissue tiles...", flush=True)
    tissue_tiles = []
    for ty in tqdm(range(n_tiles_y), desc="Scanning"):
        for tx in range(n_tiles_x):
            tile_x = tx * stride
            tile_y = ty * stride
            tile = channel_cache.get_tile(tile_x, tile_y, tile_size, SMA, scale_factor)
            if tile is not None and is_tissue_tile(tile):
                tissue_tiles.append((tile_x, tile_y))

    print(f"Found {len(tissue_tiles)} tissue tiles", flush=True)
    return tissue_tiles


def _save_scale_checkpoint(scale_config, scale_vessels, all_vessels, output_dir):
    """Save per-scale and cumulative checkpoints atomically.

    Uses tempfile + os.replace for atomic writes to prevent corruption
    if the process is interrupted during the write.

    Args:
        scale_config: Scale configuration dict
        scale_vessels: Vessels found at this scale
        all_vessels: All vessels found so far (cumulative)
        output_dir: Output directory for checkpoint files
    """
    # Per-scale checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_scale_{scale_config['name'].replace('/', '_')}.json")
    checkpoint_data = {
        'scale': scale_config['name'],
        'vessels': scale_vessels,
        'num_vessels': len(scale_vessels),
    }
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(checkpoint_data, f)
        os.replace(tmp_path, checkpoint_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"  Checkpoint saved: {checkpoint_path} ({len(scale_vessels)} vessels)", flush=True)

    # Cumulative checkpoint
    cumulative_path = os.path.join(output_dir, 'vessel_detections_checkpoint.json')
    current_idx = next((j for j, sc in enumerate(SCALES) if sc['name'] == scale_config['name']), 0)
    completed_scales = [s['name'] for i, s in enumerate(SCALES) if i <= current_idx]
    fd2, tmp_path2 = tempfile.mkstemp(dir=output_dir, suffix='.json.tmp')
    try:
        with os.fdopen(fd2, 'w') as f:
            json.dump({'vessels': all_vessels, 'scales_completed': completed_scales, 'num_vessels': len(all_vessels)}, f)
        os.replace(tmp_path2, cumulative_path)
    except Exception:
        if os.path.exists(tmp_path2):
            os.unlink(tmp_path2)
        raise
    print(f"  Cumulative checkpoint: {len(all_vessels)} vessels total", flush=True)


def _save_postprocess_checkpoint(stage, vessels, output_dir, extra=None):
    """Save a post-processing stage checkpoint (atomic write)."""
    path = os.path.join(output_dir, f'postprocess_stage_{stage}.json')
    data = {'stage': stage, 'vessels': vessels, 'num_vessels': len(vessels)}
    if extra:
        data.update(extra)
    fd, tmp = tempfile.mkstemp(dir=output_dir, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"  Post-processing checkpoint saved: {stage} ({len(vessels)} vessels)", flush=True)


def _load_postprocess_stage(output_dir):
    """Detect latest completed post-processing stage and load its checkpoint."""
    for stage in ['cells', 'refined', 'merged']:
        path = os.path.join(output_dir, f'postprocess_stage_{stage}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return stage, data['vessels']
    return None, None


def _run_post_processing(all_vessels, channel_cache, mosaic_size, args, output_dir,
                         loader, num_gpus, _main_shm_handles=None, shm_manager=None):
    """Run all post-processing stages with checkpoint/resume support.

    Stages:
      1. merged  — cross-scale merge + diameter ratio filter
      2. refined — full-res contour refinement + crop generation
      3. cells   — cell composition analysis (Cellpose)
      4. output  — final JSON/GeoPackage/HTML/histograms (always re-run)
    """
    # Detect resume point from existing stage checkpoints
    completed_stage, staged_vessels = _load_postprocess_stage(output_dir)

    if completed_stage:
        print(f"\nResuming post-processing after stage '{completed_stage}' "
              f"({len(staged_vessels)} vessels)", flush=True)

    # Report tiles skipped by Sobel pre-filter
    print(f"Tiles skipped by Sobel pre-filter: {TILES_SKIPPED_BY_SOBEL}")

    # ---- Stage 1: Merge + diameter filter ----
    if completed_stage is None:
        print(f"\nTotal vessels before merge: {len(all_vessels)}")
        merged_vessels = merge_vessels_across_scales(all_vessels, pixel_size_um=PIXEL_SIZE_UM)
        print(f"After merge: {len(merged_vessels)}")

        print(f"\nApplying adaptive diameter ratio filter...")
        filtered_vessels = adaptive_diameter_ratio_filter(merged_vessels)
        print(f"After diameter ratio filter: {len(filtered_vessels)}")
        merged_vessels = filtered_vessels

        _save_postprocess_checkpoint('merged', merged_vessels, output_dir)
    else:
        # completed_stage is 'merged', 'refined', or 'cells' — load from checkpoint
        merged_vessels = staged_vessels
        print(f"  Skipping merge (already done at stage '{completed_stage}')")

    # ---- Stage 2: Full-res contour refinement + crops ----
    if completed_stage in (None, 'merged'):
        if not args.no_refine:
            print(f"\n{'=' * 60}")
            print(f"Full-Resolution Contour Refinement")
            print(f"{'=' * 60}")
            print(f"  Dilation mode: {args.dilation_mode} (CD31 + SMA)")
            print(f"  Spline smoothing: {'on' if args.spline else 'off'} (factor={args.spline_smoothing})")
            print(f"  SMA min_wall_intensity: {args.min_sma_intensity}")
            n_refined = 0
            n_failed = 0
            for vessel in tqdm(merged_vessels, desc="Refining contours at full res"):
                ok = refine_vessel_contours_fullres(
                    vessel, channel_cache,
                    spline=args.spline,
                    spline_smoothing=args.spline_smoothing,
                    cd31_mode=args.dilation_mode,
                    sma_min_wall_intensity=args.min_sma_intensity,
                    sma_mode=args.dilation_mode,
                    pixel_size_um=PIXEL_SIZE_UM,
                )
                if ok:
                    n_refined += 1
                else:
                    n_failed += 1
            print(f"  Refined {n_refined} vessels, {n_failed} failed")

            # Regenerate ALL crops at full resolution with refined contours
            print(f"\nRegenerating crops at full resolution...")
            n_crops = 0
            n_crop_fail = 0
            for vessel in tqdm(merged_vessels, desc="Full-res crops"):
                crop_path = save_vessel_crop_fullres(vessel, channel_cache)
                if crop_path:
                    n_crops += 1
                else:
                    n_crop_fail += 1
            print(f"  Generated {n_crops} full-res crops, {n_crop_fail} failed")
        else:
            print("\nSkipping full-resolution contour refinement (--no-refine)")
            regenerate_missing_crops(merged_vessels, channel_cache)

        _save_postprocess_checkpoint('refined', merged_vessels, output_dir)
    else:
        # completed_stage is 'refined' or 'cells' — already loaded from checkpoint
        print(f"  Skipping refinement (already done at stage '{completed_stage}')")

    # Analyze vessel network topology (skeletonization) — fast, always re-run
    network_metrics = analyze_vessel_network(merged_vessels, mosaic_size, pixel_size_um=PIXEL_SIZE_UM)

    # ---- Stage 3: Cell composition analysis ----
    gmm_info = {'error': 'Cell composition analysis skipped - no Cellpose'}

    if completed_stage != 'cells':
        if CELLPOSE_AVAILABLE and len(merged_vessels) > 0:
            print("\n" + "=" * 60)
            print("Cell Composition Analysis")
            print("=" * 60)
            print(f"Analyzing cell composition for {len(merged_vessels)} vessels...")

            # Pass 1: Segment cells in all vessels, collect intensities
            all_cell_data = []
            total_cells = 0

            for i, vessel in enumerate(tqdm(merged_vessels, desc="Pass 1: Segmenting cells")):
                try:
                    global_x, global_y = vessel['global_center']
                    outer_diam_um = vessel['outer_diameter_um']

                    roi_size_um = outer_diam_um * 1.5
                    roi_size_px = int(roi_size_um / (PIXEL_SIZE_UM * BASE_SCALE))
                    roi_size_px = max(roi_size_px, 50)

                    base_x = global_x // BASE_SCALE
                    base_y = global_y // BASE_SCALE

                    x1 = max(0, base_x - roi_size_px // 2)
                    y1 = max(0, base_y - roi_size_px // 2)
                    x2 = min(channel_cache.base_width, base_x + roi_size_px // 2)
                    y2 = min(channel_cache.base_height, base_y + roi_size_px // 2)

                    if x2 <= x1 or y2 <= y1:
                        all_cell_data.append({'cd31': np.array([]), 'sma': np.array([])})
                        continue

                    pm_roi = channel_cache.channels[PM][y1:y2, x1:x2]
                    nuclear_roi = channel_cache.channels[NUCLEAR][y1:y2, x1:x2]
                    cd31_roi = channel_cache.channels[CD31][y1:y2, x1:x2]
                    sma_roi = channel_cache.channels[SMA][y1:y2, x1:x2]

                    cell_masks, num_cells = segment_cells_in_vessel(pm_roi, nuclear_roi)

                    if num_cells == 0:
                        all_cell_data.append({'cd31': np.array([]), 'sma': np.array([])})
                        continue

                    cell_intensities = measure_cell_intensities(cell_masks, cd31_roi, sma_roi)
                    all_cell_data.append(cell_intensities)
                    total_cells += len(cell_intensities['cd31'])

                except Exception as e:
                    print(f"\n  Warning: Error processing vessel {i}: {e}")
                    all_cell_data.append({'cd31': np.array([]), 'sma': np.array([])})

            print(f"  Total cells segmented: {total_cells}")

            # Pass 2: Fit slide-wide GMM and classify
            if total_cells > 0:
                print("Pass 2: Fitting slide-wide GMM...")

                all_cd31 = np.concatenate([d['cd31'] for d in all_cell_data if len(d['cd31']) > 0])
                all_sma = np.concatenate([d['sma'] for d in all_cell_data if len(d['sma']) > 0])

                cd31_thresh, sma_thresh, gmm_info = classify_cells_gmm(all_cd31, all_sma)

                print(f"  CD31 threshold: {cd31_thresh:.1f}")
                print(f"  SMA threshold: {sma_thresh:.1f}")
                print(f"  GMM fallback used: {gmm_info.get('fallback_used', False)}")

                # Pass 3: Update vessel dicts with cell_composition
                print("Pass 3: Computing cell composition per vessel...")

                for vessel, cell_data in zip(merged_vessels, all_cell_data):
                    vessel['cell_composition'] = compute_cell_composition(
                        cell_data, cd31_thresh, sma_thresh
                    )

                vessels_with_cells = sum(1 for v in merged_vessels if v.get('cell_composition', {}).get('n_total', 0) > 0)
                total_endothelial = sum(v.get('cell_composition', {}).get('n_endothelial', 0) for v in merged_vessels)
                total_smooth_muscle = sum(v.get('cell_composition', {}).get('n_smooth_muscle', 0) for v in merged_vessels)

                print(f"\n  Cell composition summary:")
                print(f"    Vessels with cells: {vessels_with_cells}/{len(merged_vessels)}")
                print(f"    Total endothelial (CD31+): {total_endothelial}")
                print(f"    Total smooth muscle (SMA+): {total_smooth_muscle}")
            else:
                print("  No cells found across all vessels - skipping GMM classification")
                gmm_info = {'error': 'No cells found', 'n_cells_total': 0}
                for vessel in merged_vessels:
                    vessel['cell_composition'] = compute_cell_composition({}, 0, 0)
        else:
            if not CELLPOSE_AVAILABLE:
                print("Cellpose not available - skipping cell composition analysis")
            else:
                print("No vessels found - skipping cell composition analysis")
            for vessel in merged_vessels:
                vessel['cell_composition'] = compute_cell_composition({}, 0, 0)

        _save_postprocess_checkpoint('cells', merged_vessels, output_dir,
                                      extra={'gmm_info': gmm_info})
    else:
        # Resuming after cells — cell_composition already in vessel dicts
        print("\n  Cell composition already completed (loaded from checkpoint)", flush=True)
        # Recover gmm_info from checkpoint if available
        cells_path = os.path.join(output_dir, 'postprocess_stage_cells.json')
        with open(cells_path, 'r') as f:
            cells_ckpt = json.load(f)
        gmm_info = cells_ckpt.get('gmm_info', gmm_info)

    # ---- Stage 4: Final output (always re-run — fast, idempotent) ----
    output_data = {
        'vessels': merged_vessels,
        'network_metrics': network_metrics,
        'metadata': {
            'czi_path': CZI_PATH,
            'mosaic_size': list(mosaic_size),
            'num_vessels': len(merged_vessels),
            'scales': [s['name'] for s in SCALES],
            'cell_classification': gmm_info,
        }
    }

    # Save JSON (atomic write)
    output_path = os.path.join(output_dir, 'vessel_detections_multiscale.json')
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix='.json.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(output_data, f, indent=2)
        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"\nSaved to {output_path}")

    # Export to GeoPackage
    gpkg_path = os.path.join(output_dir, 'vessel_detections_multiscale.gpkg')
    export_to_geodataframe(merged_vessels, gpkg_path)

    # Stats
    if len(merged_vessels) > 0:
        diameters = [v['outer_diameter_um'] for v in merged_vessels]
        print(f"\nDiameter stats:")
        print(f"  min={min(diameters):.1f}, max={max(diameters):.1f}, mean={np.mean(diameters):.1f} µm")

        for scale_config in SCALES:
            count = sum(1 for v in merged_vessels if v['scale'] == scale_config['name'])
            print(f"  Scale {scale_config['name']}: {count} vessels")

    # Generate HTML
    generate_html(merged_vessels)

    # Generate histograms
    generate_histograms(merged_vessels, output_dir)

    # Final cleanup
    channel_cache.release()
    if num_gpus > 1 and _main_shm_handles is not None:
        for shm in _main_shm_handles:
            shm.close()
        shm_manager.cleanup()
        print("Shared memory cleaned up", flush=True)
    loader.close()

    print("\nDone!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Scale SAM2 Vessel Detection")
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from a specific scale (e.g., "1/4", "1/2"). '
                             'Loads previously saved scale results and continues from the specified scale.')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs for parallel tile processing (default: 1, sequential)')
    parser.add_argument('--tile-sleep', type=float, default=None,
                        help='Sleep seconds between tiles for GPU stability. '
                             'Default: auto (1.0 on WSL2, 0.0 on native Linux)')
    parser.add_argument('--predictor-reset-interval', type=int, default=None,
                        help='Reset SAM2 predictor every N tiles to prevent VRAM fragmentation. '
                             'Default: auto (50 on WSL2, 0/disabled on native Linux)')
    parser.add_argument('--no-refine', action='store_true',
                        help='Skip full-resolution contour refinement (faster, blockier contours)')
    parser.add_argument('--spline', action='store_true',
                        help='Enable spline smoothing during refinement (off by default)')
    parser.add_argument('--spline-smoothing', type=float, default=3.0,
                        help='Spline smoothing factor (default: 3.0). Higher = smoother.')
    parser.add_argument('--dilation-mode', choices=['adaptive', 'uniform'], default='adaptive',
                        help='Dilation mode for CD31 and SMA: adaptive = per-pixel region growing '
                             '(irregular contours following actual signal), uniform = global ring '
                             'check (uniform-width rings). Default: adaptive')
    parser.add_argument('--min-sma-intensity', type=float, default=30,
                        help='Min SMA wall intensity to detect SMA ring (default: 30)')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Override pixel size in micrometers (default: read from CZI metadata, fallback 0.1725)')
    parser.add_argument('--post-process-only', action='store_true',
                        help='Skip tile collection, load cumulative checkpoint, '
                             'and resume post-processing from last completed stage.')
    args = parser.parse_args()

    # Auto-detect WSL2 for sleep/reset defaults
    _is_wsl2 = False
    if os.path.exists('/proc/version'):
        with open('/proc/version') as f:
            _is_wsl2 = 'microsoft' in f.read().lower()
    tile_sleep = args.tile_sleep if args.tile_sleep is not None else (1.0 if _is_wsl2 else 0.0)
    reset_interval = args.predictor_reset_interval if args.predictor_reset_interval is not None else (50 if _is_wsl2 else 0)
    num_gpus = args.num_gpus

    # Set global dilation config from CLI args
    global DILATION_MODE, SMA_MIN_WALL_INTENSITY
    DILATION_MODE = args.dilation_mode
    SMA_MIN_WALL_INTENSITY = args.min_sma_intensity

    # Inject CLI config into each SCALES entry so multi-GPU workers receive them
    for sc in SCALES:
        sc['dilation_mode'] = DILATION_MODE
        sc['sma_min_wall_intensity'] = SMA_MIN_WALL_INTENSITY

    print("=" * 60, flush=True)
    print("Multi-Scale SAM2 Vessel Detection", flush=True)
    print("=" * 60, flush=True)

    # Create output directories (must be in main(), not module level, for multi-GPU workers)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'tiles'), exist_ok=True)
    import shutil
    crops_dir = os.path.join(OUTPUT_DIR, 'crops')
    if os.path.exists(crops_dir) and not args.resume_from and not args.post_process_only:
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir, exist_ok=True)

    # Clean up stale post-processing checkpoints on fresh runs
    if not args.resume_from and not args.post_process_only:
        import glob as _glob
        for stale in _glob.glob(os.path.join(OUTPUT_DIR, 'postprocess_stage_*.json')):
            os.unlink(stale)
            print(f"  Removed stale checkpoint: {os.path.basename(stale)}", flush=True)

    # Open CZI (don't load full-res into RAM)
    loader = CZILoader(CZI_PATH)
    mosaic_size = loader.mosaic_size
    print(f"\nMosaic size (full res): {mosaic_size}", flush=True)

    # Read pixel size from CZI metadata (or CLI override)
    global PIXEL_SIZE_UM
    if args.pixel_size is not None:
        PIXEL_SIZE_UM = args.pixel_size
        print(f"Pixel size (CLI override): {PIXEL_SIZE_UM} um", flush=True)
    else:
        czi_pixel_size = None
        if hasattr(loader, 'get_pixel_size'):
            czi_pixel_size = loader.get_pixel_size()
        # Validate: must be positive and reasonable (0.01 to 100 um)
        if czi_pixel_size is not None and 0.01 < czi_pixel_size < 100:
            PIXEL_SIZE_UM = czi_pixel_size
            print(f"Pixel size (CZI metadata): {PIXEL_SIZE_UM} um", flush=True)
        else:
            print(f"Pixel size (default fallback): {PIXEL_SIZE_UM} um", flush=True)

    # Load all channels at BASE_SCALE (1/2) - 4x smaller than full res
    channel_cache = DownsampledChannelCache(loader, [NUCLEAR, CD31, SMA, PM], BASE_SCALE)

    # NOTE: Slide-wide photobleaching correction disabled - too slow on large images
    # channel_cache.apply_photobleaching_correction()

    # SAM2 configuration (shared between single-GPU and multi-GPU paths)
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM2_CHECKPOINT = "/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/checkpoints/sam2.1_hiera_large.pt"
    SAM2_GENERATOR_PARAMS = {
        'points_per_side': 48,
        'pred_iou_thresh': 0.5,
        'stability_score_thresh': 0.7,
        'min_mask_region_area': 100,
    }

    # --post-process-only: skip all tile collection, jump to post-processing
    if args.post_process_only:
        checkpoint_path = os.path.join(OUTPUT_DIR, 'vessel_detections_checkpoint.json')
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: No checkpoint found at {checkpoint_path}", flush=True)
            sys.exit(1)
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        all_vessels = ckpt['vessels']
        print(f"\nLoaded {len(all_vessels)} vessels from checkpoint ({ckpt.get('scales_completed', '?')})")
        print("Skipping tile collection, jumping to post-processing...", flush=True)

        _run_post_processing(
            all_vessels, channel_cache, mosaic_size, args, OUTPUT_DIR, loader,
            num_gpus, _main_shm_handles=None, shm_manager=None,
        )
        return

    print(f"\nMode: {'Multi-GPU (' + str(num_gpus) + ' GPUs)' if num_gpus > 1 else 'Single-GPU'}", flush=True)
    print(f"  tile_sleep={tile_sleep:.1f}s, reset_interval={reset_interval}", flush=True)

    # Compute adaptive Sobel threshold (CPU, uses channel_cache directly)
    adaptive_sobel_threshold = compute_adaptive_sobel_threshold(channel_cache, SCALES[0]['scale_factor'])
    print(f"Adaptive Sobel threshold: {adaptive_sobel_threshold:.4f}")

    all_vessels = []

    # Resume support: load previously saved scale results and skip completed scales
    resume_from = args.resume_from
    scales_to_process = SCALES
    if resume_from:
        # Find the resume scale index
        scale_names = [s['name'] for s in SCALES]
        if resume_from not in scale_names:
            print(f"ERROR: Unknown scale '{resume_from}'. Available: {scale_names}", flush=True)
            sys.exit(1)
        resume_idx = scale_names.index(resume_from)
        # Load saved results from completed scales
        for i, sc in enumerate(SCALES[:resume_idx]):
            checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_scale_{sc['name'].replace('/', '_')}.json")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    saved = json.load(f)
                all_vessels.extend(saved['vessels'])
                print(f"  Loaded {len(saved['vessels'])} vessels from {sc['name']} checkpoint", flush=True)
            else:
                print(f"  WARNING: No checkpoint for scale {sc['name']}, skipping", flush=True)
        scales_to_process = SCALES[resume_idx:]
        print(f"\nResuming from scale {resume_from} with {len(all_vessels)} vessels from prior scales", flush=True)

    # These are set in multi-GPU path and passed to post-processing for cleanup
    _main_shm_handles = None
    shm_manager = None

    # ========================================================================
    # Multi-GPU path: shared memory + worker pool
    # ========================================================================
    if num_gpus > 1:
        from segmentation.processing.multigpu_shm import SharedSlideManager

        print(f"\nSetting up shared memory for {num_gpus} GPUs...", flush=True)
        shm_manager = SharedSlideManager()

        # Copy channel data into shared memory
        channel_shm_info = {}
        for ch_idx, ch_data in channel_cache.channels.items():
            info = shm_manager.add_slide(f"vessel_ch{ch_idx}", ch_data)
            channel_shm_info[ch_idx] = info

        total_shm_gb = sum(ch.nbytes for ch in channel_cache.channels.values()) / (1024**3)
        print(f"  Shared memory allocated: {total_shm_gb:.1f} GB for {len(channel_shm_info)} channels", flush=True)

        # Release original channel cache to save ~45GB RAM (shared memory has the copy)
        # But keep the reference for tissue tile scanning (reads from shm via SharedChannelCache)
        channel_cache.release()
        print("  Released original channel cache (data now in shared memory)", flush=True)

        # Create a SharedChannelCache for main process tissue scanning
        main_shm_channels = {}
        from multiprocessing.shared_memory import SharedMemory as _SharedMemory
        _main_shm_handles = []
        for ch_idx, info in channel_shm_info.items():
            shm = _SharedMemory(name=info['shm_name'])
            arr = np.ndarray(tuple(info['shape']), dtype=np.dtype(info['dtype']), buffer=shm.buf)
            main_shm_channels[ch_idx] = arr
            _main_shm_handles.append(shm)
        main_channel_cache = SharedChannelCache(main_shm_channels, BASE_SCALE, mosaic_size)

        # Start multi-GPU workers (once, reused across all scales)
        with VesselMultiGPUProcessor(
            num_gpus=num_gpus,
            channel_shm_info=channel_shm_info,
            base_scale=BASE_SCALE,
            full_res_size=list(mosaic_size),
            output_dir=OUTPUT_DIR,
            sam2_checkpoint_path=SAM2_CHECKPOINT,
            sam2_config=SAM2_CONFIG,
            sam2_generator_params=SAM2_GENERATOR_PARAMS,
            adaptive_sobel_threshold=adaptive_sobel_threshold,
            tile_sleep=tile_sleep,
            reset_interval=reset_interval,
            pixel_size_um=PIXEL_SIZE_UM,
        ) as processor:
            import queue as queue_module

            tiles_skipped_multigpu = 0

            for scale_config in scales_to_process:
                scale_factor = scale_config['scale_factor']
                tile_size = scale_config.get('tile_size', TILE_SIZE)
                stride = int(tile_size * (1 - TILE_OVERLAP))

                print(f"\n{'='*40}", flush=True)
                print(f"Processing scale {scale_config['name']} ({num_gpus} GPUs)", flush=True)
                print(f"  Scale factor: {scale_factor}x", flush=True)
                print(f"  Tile size: {tile_size}px (covers {tile_size * scale_factor}px at full res)", flush=True)
                print(f"  Stride: {stride}px ({int(TILE_OVERLAP*100)}% overlap)", flush=True)
                print(f"{'='*40}", flush=True)

                # Create tile grid and find tissue tiles (CPU, main process)
                tissue_tiles = _scan_tissue_tiles(main_channel_cache, scale_factor, tile_size, stride, mosaic_size)

                n_sample = max(1, int(len(tissue_tiles) * SAMPLE_FRACTION))
                sampled_tiles = random.sample(tissue_tiles, min(n_sample, len(tissue_tiles)))
                print(f"Processing {len(sampled_tiles)} tiles on {num_gpus} GPUs", flush=True)

                # Submit all tiles to workers
                for tile_x, tile_y in sampled_tiles:
                    processor.submit_tile(tile_x, tile_y, scale_config)

                # Collect all results
                scale_vessels = []
                for i in tqdm(range(len(sampled_tiles)), desc="Collecting"):
                    result = processor.collect_result(timeout=600)
                    if result is None:
                        print(f"\n  WARNING: Timeout collecting result {i+1}/{len(sampled_tiles)}", flush=True)
                        continue
                    if result.get('status') == 'success':
                        if result.get('scale') != scale_config['name']:
                            # Late result from a previous scale - add directly
                            all_vessels.extend(result.get('vessels', []))
                            print(f"\n  Note: Got late result from scale {result.get('scale')}", flush=True)
                        else:
                            scale_vessels.extend(result.get('vessels', []))
                        if result.get('sobel_skipped'):
                            tiles_skipped_multigpu += 1
                    elif result.get('status') == 'error':
                        print(f"\n  Error at ({result.get('tile_x')}, {result.get('tile_y')}): {result.get('error')}", flush=True)

                # Drain any late-arriving results before next scale
                drained = 0
                while True:
                    try:
                        result = processor.output_queue.get_nowait()
                        if result.get('status') == 'success':
                            if result.get('scale') == scale_config['name']:
                                scale_vessels.extend(result.get('vessels', []))
                            else:
                                all_vessels.extend(result.get('vessels', []))
                            if result.get('sobel_skipped'):
                                tiles_skipped_multigpu += 1
                        drained += 1
                    except queue_module.Empty:
                        break
                if drained > 0:
                    print(f"  Drained {drained} late-arriving results from queue", flush=True)

                scale_vessels = deduplicate_vessels_within_scale(scale_vessels, pixel_size_um=PIXEL_SIZE_UM)
                print(f"Found {len(scale_vessels)} vessels at scale {scale_config['name']}", flush=True)
                all_vessels.extend(scale_vessels)

                # Save checkpoint after each scale (atomic writes)
                _save_scale_checkpoint(scale_config, scale_vessels, all_vessels, OUTPUT_DIR)

        # Workers stopped by context manager
        # Keep shared memory alive for post-processing (crop regen, cell composition)
        # SharedChannelCache has same API as DownsampledChannelCache (get_tile, channels, base_width, etc.)
        channel_cache = main_channel_cache
        channel_cache.loader = loader
        # Propagate worker sobel skip count to global for reporting
        global TILES_SKIPPED_BY_SOBEL
        TILES_SKIPPED_BY_SOBEL = tiles_skipped_multigpu
        print("Workers stopped, shared memory kept alive for post-processing", flush=True)

    # ========================================================================
    # Single-GPU path: original sequential processing
    # ========================================================================
    else:
        # Load SAM2 on the single GPU
        print("\nLoading SAM2...", flush=True)
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
        mask_generator = SAM2AutomaticMaskGenerator(model=sam2, **SAM2_GENERATOR_PARAMS)

        # Process each scale sequentially
        for scale_config in scales_to_process:
            scale_factor = scale_config['scale_factor']
            tile_size = scale_config.get('tile_size', TILE_SIZE)
            stride = int(tile_size * (1 - TILE_OVERLAP))

            print(f"\n{'='*40}", flush=True)
            print(f"Processing scale {scale_config['name']}", flush=True)
            print(f"  Scale factor: {scale_factor}x", flush=True)
            print(f"  Tile size: {tile_size}px (covers {tile_size * scale_factor}px at full res)", flush=True)
            print(f"  Stride: {stride}px ({int(TILE_OVERLAP*100)}% overlap)", flush=True)
            print(f"  Diameter range: {scale_config['min_diam_um']}-{scale_config['max_diam_um']} µm", flush=True)
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_cached = torch.cuda.memory_reserved() / 1e9
            print(f"  VRAM: {vram_used:.1f}GB used, {vram_cached:.1f}GB cached", flush=True)
            print(f"{'='*40}", flush=True)

            tissue_tiles = _scan_tissue_tiles(channel_cache, scale_factor, tile_size, stride, mosaic_size)

            n_sample = max(1, int(len(tissue_tiles) * SAMPLE_FRACTION))
            sampled_tiles = random.sample(tissue_tiles, min(n_sample, len(tissue_tiles)))
            print(f"Processing {len(sampled_tiles)} tiles", flush=True)

            scale_vessels = []
            tiles_since_reset = 0

            for tile_x, tile_y in tqdm(sampled_tiles, desc="Processing"):
                try:
                    vessels = process_tile_at_scale(
                        tile_x, tile_y, channel_cache, mask_generator, scale_config,
                        pixel_size_um=PIXEL_SIZE_UM,
                        adaptive_sobel_threshold=adaptive_sobel_threshold
                    )
                    scale_vessels.extend(vessels)
                except Exception as e:
                    print(f"\nError at ({tile_x}, {tile_y}): {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                finally:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

                    if tile_sleep > 0:
                        time.sleep(tile_sleep)

                    tiles_since_reset += 1
                    if reset_interval > 0 and tiles_since_reset >= reset_interval:
                        if hasattr(mask_generator, 'predictor') and hasattr(mask_generator.predictor, 'reset_predictor'):
                            mask_generator.predictor.reset_predictor()
                        torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                        tiles_since_reset = 0
                        if tile_sleep > 0:
                            time.sleep(tile_sleep * 3)
                        vram_used = torch.cuda.memory_allocated() / 1e9
                        vram_cached = torch.cuda.memory_reserved() / 1e9
                        print(f"\n  [VRAM reset] allocated={vram_used:.1f}GB, cached={vram_cached:.1f}GB", flush=True)

            scale_vessels = deduplicate_vessels_within_scale(scale_vessels, pixel_size_um=PIXEL_SIZE_UM)
            print(f"Found {len(scale_vessels)} vessels at scale {scale_config['name']}", flush=True)
            all_vessels.extend(scale_vessels)

            # Save checkpoint after each scale (atomic writes)
            _save_scale_checkpoint(scale_config, scale_vessels, all_vessels, OUTPUT_DIR)

            # Aggressive cleanup between scales
            if hasattr(mask_generator, 'predictor') and hasattr(mask_generator.predictor, 'reset_predictor'):
                mask_generator.predictor.reset_predictor()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated() / 1e9
            print(f"  [Scale complete] VRAM after cleanup: {vram_after:.1f}GB", flush=True)

        # Free SAM2 model before post-processing (no longer needed)
        del mask_generator, sam2
        gc.collect()
        torch.cuda.empty_cache()
        print(f"SAM2 model freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)

    _run_post_processing(
        all_vessels, channel_cache, mosaic_size, args, OUTPUT_DIR, loader,
        num_gpus, _main_shm_handles=_main_shm_handles, shm_manager=shm_manager,
    )

def generate_histograms(vessels, output_dir):
    """
    Generate histograms for vessel morphometry.

    Creates:
    - histogram_diameters.png - outer diameter distribution
    - histogram_wall_thickness.png - wall thickness distribution
    - histogram_lumen_wall_ratio.png - lumen area / wall area ratio
    - histogram_combined.png - all three in one figure

    Args:
        vessels: List of vessel dicts
        output_dir: Directory to save histogram images
    """
    if len(vessels) == 0:
        print("No vessels for histogram generation")
        return

    print("\nGenerating histograms...")

    # Extract data
    diameters = np.array([v['outer_diameter_um'] for v in vessels])
    wall_thickness = np.array([v['wall_thickness_um'] for v in vessels])

    # Lumen/wall ratio (inner_area / wall_area)
    lumen_wall_ratios = []
    for v in vessels:
        inner_area = v.get('inner_area_px', 0)
        wall_area = v.get('wall_area_px', 1)  # avoid div by zero
        if wall_area > 0:
            lumen_wall_ratios.append(inner_area / wall_area)
    lumen_wall_ratios = np.array(lumen_wall_ratios)

    # Color by scale
    scale_colors = {
        '1/64': '#e41a1c',
        '1/32': '#377eb8',
        '1/16': '#4daf4a',
        '1/8': '#984ea3',
        '1/4': '#ff7f00',
        '1/2': '#ffff33',
    }

    # --- Individual histograms ---

    # 1. Diameter histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diameters, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Outer Diameter (µm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Vessel Diameter Distribution (n={len(diameters)})', fontsize=14)
    ax.axvline(np.median(diameters), color='red', linestyle='--', label=f'Median: {np.median(diameters):.1f} µm')
    ax.axvline(np.mean(diameters), color='orange', linestyle='--', label=f'Mean: {np.mean(diameters):.1f} µm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_diameters.png'), dpi=150)
    plt.close()

    # 2. Wall thickness histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(wall_thickness, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    ax.set_xlabel('Wall Thickness (µm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Vessel Wall Thickness Distribution (n={len(wall_thickness)})', fontsize=14)
    ax.axvline(np.median(wall_thickness), color='red', linestyle='--', label=f'Median: {np.median(wall_thickness):.1f} µm')
    ax.axvline(np.mean(wall_thickness), color='orange', linestyle='--', label=f'Mean: {np.mean(wall_thickness):.1f} µm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_wall_thickness.png'), dpi=150)
    plt.close()

    # 3. Lumen/wall ratio histogram
    if len(lumen_wall_ratios) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lumen_wall_ratios, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Lumen/Wall Area Ratio', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Lumen/Wall Ratio Distribution (n={len(lumen_wall_ratios)})', fontsize=14)
        ax.axvline(np.median(lumen_wall_ratios), color='red', linestyle='--', label=f'Median: {np.median(lumen_wall_ratios):.2f}')
        ax.axvline(np.mean(lumen_wall_ratios), color='orange', linestyle='--', label=f'Mean: {np.mean(lumen_wall_ratios):.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'histogram_lumen_wall_ratio.png'), dpi=150)
        plt.close()

    # 4. SMA thickness histogram (only vessels with SMA ring)
    sma_thickness = np.array([v['sma_thickness_um'] for v in vessels if v.get('has_sma_ring', False)])
    n_with_sma = len(sma_thickness)
    n_without_sma = len(vessels) - n_with_sma
    if len(sma_thickness) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sma_thickness, bins=50, edgecolor='black', alpha=0.7, color='magenta')
        ax.set_xlabel('SMA Thickness (µm)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'SMA Ring Thickness (n={n_with_sma} with SMA, {n_without_sma} without)', fontsize=14)
        ax.axvline(np.median(sma_thickness), color='red', linestyle='--', label=f'Median: {np.median(sma_thickness):.1f} µm')
        ax.axvline(np.mean(sma_thickness), color='orange', linestyle='--', label=f'Mean: {np.mean(sma_thickness):.1f} µm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'histogram_sma_thickness.png'), dpi=150)
        plt.close()

    # --- Combined figure ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Diameter
    ax = axes[0, 0]
    ax.hist(diameters, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Outer Diameter (µm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Diameter (n={len(diameters)}, median={np.median(diameters):.1f} µm)')
    ax.grid(True, alpha=0.3)

    # Wall thickness
    ax = axes[0, 1]
    ax.hist(wall_thickness, bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
    ax.set_xlabel('Wall Thickness (µm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Wall Thickness (median={np.median(wall_thickness):.1f} µm)')
    ax.grid(True, alpha=0.3)

    # Lumen/wall ratio
    ax = axes[1, 0]
    if len(lumen_wall_ratios) > 0:
        ax.hist(lumen_wall_ratios, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Lumen/Wall Area Ratio')
        ax.set_ylabel('Count')
        ax.set_title(f'Lumen/Wall Ratio (median={np.median(lumen_wall_ratios):.2f})')
        ax.grid(True, alpha=0.3)

    # SMA thickness
    ax = axes[1, 1]
    if len(sma_thickness) > 0:
        ax.hist(sma_thickness, bins=50, edgecolor='black', alpha=0.7, color='magenta')
        ax.set_xlabel('SMA Thickness (µm)')
        ax.set_ylabel('Count')
        ax.set_title(f'SMA Thickness (n={n_with_sma}, median={np.median(sma_thickness):.1f} µm)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No SMA rings detected', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('SMA Thickness')

    # Diameter by scale
    ax = axes[2, 0]
    scale_data = {}
    for v in vessels:
        scale = v.get('scale', 'unknown')
        if scale not in scale_data:
            scale_data[scale] = []
        scale_data[scale].append(v['outer_diameter_um'])

    for scale, data in sorted(scale_data.items()):
        color = scale_colors.get(scale, 'gray')
        ax.hist(data, bins=30, alpha=0.5, label=f'{scale} (n={len(data)})', color=color)
    ax.set_xlabel('Outer Diameter (µm)')
    ax.set_ylabel('Count')
    ax.set_title('Diameter by Detection Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SMA ring presence
    ax = axes[2, 1]
    sma_counts = [n_with_sma, n_without_sma]
    sma_labels_plot = [f'SMA+ (n={n_with_sma})', f'SMA- (n={n_without_sma})']
    sma_colors = ['magenta', 'lightgray']
    if sum(sma_counts) > 0:
        ax.bar(sma_labels_plot, sma_counts, color=sma_colors, edgecolor='black')
        ax.set_ylabel('Count')
        ax.set_title(f'SMA Ring Presence ({n_with_sma}/{len(vessels)} = {100*n_with_sma/len(vessels):.0f}%)')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_combined.png'), dpi=150)
    plt.close()

    print(f"  Saved histograms to {output_dir}")
    print(f"    - histogram_diameters.png")
    print(f"    - histogram_wall_thickness.png")
    print(f"    - histogram_lumen_wall_ratio.png")
    print(f"    - histogram_sma_thickness.png")
    print(f"    - histogram_combined.png")


def export_to_geodataframe(vessels, output_path):
    """
    Export vessels to a GeoPackage file for GIS compatibility.

    Creates three geometry columns:
    - outer_geometry: Polygon of vessel outer wall
    - inner_geometry: Polygon of vessel lumen
    - wall_geometry: outer.difference(inner) - the actual vessel wall

    Args:
        vessels: List of vessel dicts with outer_contour and inner_contour
        output_path: Path for output .gpkg file

    Returns:
        GeoDataFrame if successful, None otherwise
    """
    if not GEOPANDAS_AVAILABLE:
        print("Warning: geopandas not installed, skipping GeoDataFrame export")
        print("  Install with: pip install geopandas shapely")
        return None

    if not vessels:
        print("No vessels to export to GeoDataFrame")
        return None

    print(f"\nExporting {len(vessels)} vessels to GeoPackage...")

    records = []
    for v in vessels:
        try:
            # Convert contours to Shapely Polygons (in global coordinates)
            # Stored contours are tile-local; add tile origin offset
            tile_offset_x = v['tile_x'] * v['scale_factor']
            tile_offset_y = v['tile_y'] * v['scale_factor']
            outer_coords = np.array(v['outer_contour']).squeeze().copy()
            inner_coords = np.array(v['inner_contour']).squeeze().copy()
            outer_coords[..., 0] += tile_offset_x
            outer_coords[..., 1] += tile_offset_y
            inner_coords[..., 0] += tile_offset_x
            inner_coords[..., 1] += tile_offset_y

            # Need at least 3 points for a polygon
            if len(outer_coords) < 3 or len(inner_coords) < 3:
                continue

            # Create polygons (ensure they're closed)
            outer_poly = Polygon(outer_coords)
            inner_poly = Polygon(inner_coords)

            # Make valid if needed (handles self-intersections)
            if not outer_poly.is_valid:
                outer_poly = make_valid(outer_poly)
            if not inner_poly.is_valid:
                inner_poly = make_valid(inner_poly)

            # Create wall geometry (outer minus inner)
            try:
                wall_poly = outer_poly.difference(inner_poly)
            except Exception:
                wall_poly = outer_poly  # Fallback if difference fails

            # SMA geometry (if present)
            sma_poly = None
            sma_contour_data = v.get('sma_contour')
            if sma_contour_data and len(sma_contour_data) >= 3:
                sma_coords = np.array(sma_contour_data).squeeze().copy()
                sma_coords[..., 0] += tile_offset_x
                sma_coords[..., 1] += tile_offset_y
                if len(sma_coords) >= 3:
                    sma_poly = Polygon(sma_coords)
                    if not sma_poly.is_valid:
                        sma_poly = make_valid(sma_poly)

            # Build record with all vessel attributes
            record = {
                'uid': v.get('uid', ''),
                'scale': v.get('scale', ''),
                'scale_factor': v.get('scale_factor', 1),
                'source_channel': v.get('source_channel', ''),
                'global_x': v.get('global_center', [0, 0])[0],
                'global_y': v.get('global_center', [0, 0])[1],
                'outer_diameter_um': v.get('outer_diameter_um', 0),
                'inner_diameter_um': v.get('inner_diameter_um', 0),
                'wall_thickness_um': v.get('wall_thickness_um', 0),
                'sma_diameter_um': v.get('sma_diameter_um', 0),
                'sma_thickness_um': v.get('sma_thickness_um', 0),
                'has_sma_ring': v.get('has_sma_ring', False),
                'outer_area_px': v.get('outer_area_px', 0),
                'inner_area_px': v.get('inner_area_px', 0),
                'wall_area_px': v.get('wall_area_px', 0),
                'sma_inside': v.get('sma_inside', 0),
                'sma_wall': v.get('sma_wall', 0),
                'sma_ratio': v.get('sma_ratio', 0),
                'cd31_ratio': v.get('cd31_ratio', 0),
                'nuclear_ratio': v.get('nuclear_ratio', 0),
                'pm_ratio': v.get('pm_ratio', 0),
                'outer_geometry': outer_poly,
                'inner_geometry': inner_poly,
                'wall_geometry': wall_poly,
            }
            if sma_poly is not None:
                record['sma_geometry'] = sma_poly
            records.append(record)

        except Exception as e:
            print(f"  Warning: Could not process vessel {v.get('uid', 'unknown')}: {e}")
            continue

    if not records:
        print("No valid vessel geometries to export")
        return None

    # Create GeoDataFrame with wall_geometry as the active geometry
    gdf = gpd.GeoDataFrame(records, geometry='wall_geometry')

    # Save to GeoPackage (supports multiple geometry columns)
    gdf.to_file(output_path, driver='GPKG')
    print(f"Saved GeoPackage to {output_path}")
    print(f"  {len(gdf)} vessels with outer, inner, and wall geometries")

    return gdf


def generate_html(vessels):
    """Generate HTML viewer using package template."""
    import base64
    from segmentation.io.html_export import export_vessel_samples_to_html

    print("\nGenerating HTML...")

    # Convert vessels to sample format expected by export_vessel_samples_to_html
    samples = []
    for v in vessels:
        crop_path = v.get('crop_path', '')
        if not crop_path or not os.path.exists(crop_path):
            continue

        # Read contoured crop and convert to base64
        with open(crop_path, 'rb') as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Read raw crop (no contours) if available
        raw_path = v.get('crop_path_raw', '')
        img_raw_base64 = None
        if raw_path and os.path.exists(raw_path):
            with open(raw_path, 'rb') as f:
                raw_data = f.read()
            img_raw_base64 = base64.b64encode(raw_data).decode('utf-8')

        # Build features dict (package expects specific keys)
        features = {
            'outer_diameter_um': v.get('outer_diameter_um', 0),
            'inner_diameter_um': v.get('inner_diameter_um', 0),
            'wall_thickness_mean_um': v.get('wall_thickness_um', 0),
            'sma_diameter_um': v.get('sma_diameter_um', 0),
            'sma_thickness_um': v.get('sma_thickness_um', 0),
            'has_sma_ring': v.get('has_sma_ring', False),
            'outer_area_px': v.get('outer_area_px', 0),
            'inner_area_px': v.get('inner_area_px', 0),
            'wall_area_px': v.get('wall_area_px', 0),
            'sma_ratio': v.get('sma_ratio', 0),
            'scale': v.get('scale', '?'),
            'global_x': v.get('global_center', [0, 0])[0],
            'global_y': v.get('global_center', [0, 0])[1],
        }

        sample = {
            'uid': v.get('uid', ''),
            'image': img_base64,  # Contoured image
            'features': features,
        }
        if img_raw_base64:
            sample['image_raw'] = img_raw_base64  # Raw image (no contours)

        samples.append(sample)

    if not samples:
        print("No samples to export")
        return

    # Sort by diameter descending
    samples = sorted(samples, key=lambda s: -s['features'].get('outer_diameter_um', 0))

    # Export using package template
    export_vessel_samples_to_html(
        samples=samples,
        output_dir=OUTPUT_DIR,
        cell_type='vessel',
        samples_per_page=200,
        title='SAM2 Multiscale Vessel Detection',
        subtitle=f"Scales: {', '.join(s['name'] for s in SCALES)}",
        experiment_name='sam2_multiscale',
        file_name=os.path.basename(CZI_PATH),
    )

if __name__ == '__main__':
    main()
