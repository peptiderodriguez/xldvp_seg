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
# watershed removed - using dilate_until_signal_drops instead
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
import torch
from tqdm import tqdm

# Global counter for tiles skipped by Sobel pre-filter
TILES_SKIPPED_BY_SOBEL = 0

# ==============================================================================
# GMM-based Adaptive Thresholding
# ==============================================================================
# More robust to staining intensity variation across the slide than fixed thresholds.
# Instead of `ratio < 1.0`, fits a 2-component GMM to separate populations.

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
        self.gmm = GaussianMixture(
            n_components=2,
            covariance_type='full',
            max_iter=100,
            n_init=3,
            random_state=42
        )
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
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        max_iter=100,
        n_init=3,
        random_state=42
    )
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

    # Fit GMM on all values
    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        max_iter=100,
        n_init=3,
        random_state=42
    )
    gmm.fit(all_values.reshape(-1, 1))

    # Select positive component
    means = gmm.means_.flatten()
    if lower_is_positive:
        positive_component = int(np.argmin(means))
    else:
        positive_component = int(np.argmax(means))

    # Get probability for the single value
    proba = gmm.predict_proba(np.array([[value]]))
    return proba[0, positive_component] >= threshold_ppv


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

sys.path.insert(0, '/home/dude/code/vessel_seg')
from segmentation.io.czi_loader import CZILoader
from segmentation.preprocessing.illumination import correct_photobleaching

# Configuration
CZI_PATH = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT_DIR = "/home/dude/vessel_output/sam2_multiscale"
SAMPLE_FRACTION = 0.1  # 10% of tiles
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
        if scale_factor < self.base_scale:
            raise ValueError(f"scale_factor {scale_factor} < base_scale {self.base_scale}")

        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not loaded")

        # Convert tile coords from target scale to base scale
        # tile_x is in target-scale space, need to convert to base-scale space
        relative_scale = scale_factor // self.base_scale

        # In target scale: tile covers tile_size pixels
        # In full res: tile covers tile_size * scale_factor pixels
        # In base scale: tile covers tile_size * scale_factor / base_scale = tile_size * relative_scale
        base_tile_size = tile_size * relative_scale

        # Convert tile origin from target-scale coords to base-scale coords
        # tile_x (target) * scale_factor = full_res_x
        # full_res_x / base_scale = base_x
        base_x = (tile_x * scale_factor) // self.base_scale
        base_y = (tile_y * scale_factor) // self.base_scale

        # Extract region from base-scale data
        data = self.channels[channel]
        y2 = min(base_y + base_tile_size, data.shape[0])
        x2 = min(base_x + base_tile_size, data.shape[1])

        if base_x >= data.shape[1] or base_y >= data.shape[0]:
            return None

        region = data[base_y:y2, base_x:x2]

        if region.size == 0:
            return None

        # Downsample from base scale to target scale if needed
        if relative_scale > 1:
            target_h = region.shape[0] // relative_scale
            target_w = region.shape[1] // relative_scale
            if target_h == 0 or target_w == 0:
                return None
            region = cv2.resize(region, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return region

    def release(self):
        """Release all channel data."""
        self.channels.clear()
        gc.collect()

# Channel indices
NUCLEAR = 0
CD31 = 1
SMA = 2
PM = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'tiles'), exist_ok=True)

# Clear crops folder at start of each run to avoid mixing with previous runs
crops_dir = os.path.join(OUTPUT_DIR, 'crops')
if os.path.exists(crops_dir):
    import shutil
    shutil.rmtree(crops_dir)
os.makedirs(crops_dir, exist_ok=True)

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
    cd31_float = cd31_roi.astype(np.float32)
    sma_float = sma_roi.astype(np.float32)

    # Measure intensities for each cell (single pass)
    cd31_intensities = []
    sma_intensities = []
    valid_cell_ids = []

    for cell_id in cell_ids:
        cell_mask = cell_masks == cell_id
        n_pixels = cell_mask.sum()

        # Skip cells with very few pixels (likely noise)
        if n_pixels < 5:
            continue

        cd31_intensities.append(cd31_float[cell_mask].mean())
        sma_intensities.append(sma_float[cell_mask].mean())
        valid_cell_ids.append(cell_id)

    return {
        'cd31': np.array(cd31_intensities, dtype=np.float32),
        'sma': np.array(sma_intensities, dtype=np.float32),
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
    cd31_gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        max_iter=100,
        n_init=3,
        random_state=42
    )
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
    sma_gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        max_iter=100,
        n_init=3,
        random_state=42
    )
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


def save_vessel_crop(vessel, display_rgb, outer_contour, inner_contour, tile_x, tile_y, tile_size, scale_factor):
    """Save a crop of the vessel with contours drawn.

    Crop is centered on the mask centroid and sized to be 2x the mask bounding box.
    """
    # Get bounding box of both contours combined
    all_points = np.vstack([outer_contour, inner_contour])
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

    crop = display_rgb[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    # Draw contours on crop (translate to crop coordinates)
    outer_in_crop = outer_contour - np.array([x1, y1])
    inner_in_crop = inner_contour - np.array([x1, y1])

    cv2.drawContours(crop, [outer_in_crop], -1, (0, 255, 0), 2)  # Green for outer wall
    cv2.drawContours(crop, [inner_in_crop], -1, (0, 255, 255), 2)  # Cyan for inner lumen

    # Store crop offset for reference
    vessel['crop_offset'] = [int(x1), int(y1)]
    vessel['crop_scale_factor'] = scale_factor

    crop_path = os.path.join(OUTPUT_DIR, 'crops', f"{vessel['uid']}.jpg")
    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    return crop_path

def is_tissue_tile(tile, threshold=TISSUE_VARIANCE_THRESHOLD):
    return np.var(tile) > threshold


def compute_edge_density(tile, threshold_percentile=80):
    """
    Compute Sobel edge density for a tile.

    Args:
        tile: 2D numpy array (grayscale tile data)
        threshold_percentile: Percentile for edge thresholding (default: 80)

    Returns:
        float: Edge density (fraction of pixels above threshold)
    """
    # Convert to float32 for processing
    tile_float = tile.astype(np.float32)

    # Apply Gaussian smoothing to reduce noise (sigma=2)
    smoothed = gaussian_filter(tile_float, sigma=2)

    # Compute Sobel gradient magnitude
    sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Threshold at given percentile
    threshold = np.percentile(gradient_magnitude, threshold_percentile)
    edge_mask = gradient_magnitude > threshold

    # Calculate edge density (fraction of pixels above threshold)
    edge_density = edge_mask.sum() / edge_mask.size

    return edge_density


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
        gmm = GaussianMixture(
            n_components=2,
            covariance_type='full',
            max_iter=100,
            n_init=3,
            random_state=42
        )
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


def has_vessel_candidates(tile, threshold_percentile=80, min_edge_density=0.05):
    """
    Fast Sobel-based pre-filter to detect if tile has vessel candidates.

    Vessels have strong edges (vessel walls), so tiles without significant
    edge density are unlikely to contain vessels and can be skipped.

    Args:
        tile: 2D numpy array (grayscale tile data)
        threshold_percentile: Percentile for edge thresholding (default: 80)
        min_edge_density: Minimum fraction of pixels above threshold to consider
                         tile as having vessel candidates (default: 0.05 = 5%)

    Returns:
        True if tile has sufficient edge density (potential vessels), False otherwise.
    """
    edge_density = compute_edge_density(tile, threshold_percentile)
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
    sma_ratio = sma_inside / (sma_surrounding + 1)
    nuclear_ratio = nuclear_inside / (nuclear_surrounding + 1)
    cd31_ratio = cd31_inside / (cd31_surrounding + 1)
    pm_ratio = pm_inside / (pm_surrounding + 1)

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

def dilate_until_signal_drops(lumens, cd31_norm, scale_factor=1, drop_ratio=0.85, max_iterations=50):
    """
    Expand from lumens by dilating until CD31 signal drops.

    CD31 marks endothelium (vessel lining), so this ensures we're capturing
    actual blood vessel boundaries.

    For each lumen:
    1. Start with the lumen mask
    2. Dilate iteratively
    3. Check mean CD31 intensity in the newly added ring
    4. Stop when ring intensity drops below drop_ratio * wall intensity

    Returns labels array where each lumen's expanded region has a unique label.
    """
    labels = np.zeros(cd31_norm.shape, dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)

    for idx, lumen in enumerate(lumens):
        mask = lumen['mask'].astype(np.uint8)
        label_id = idx + 1

        # Get initial wall intensity (the CD31+ ring around lumen)
        dilated_once = cv2.dilate(mask, kernel, iterations=1)
        initial_ring = (dilated_once > 0) & (mask == 0)
        if initial_ring.sum() == 0:
            labels[mask > 0] = label_id
            continue
        wall_intensity = cd31_norm[initial_ring].mean()

        # Threshold for stopping: when ring drops below this
        stop_threshold = wall_intensity * drop_ratio

        # Iteratively dilate - always do at least one dilation
        current_mask = dilated_once.copy()  # Start with one dilation already done
        for i in range(max_iterations - 1):  # -1 since we already did one
            dilated = cv2.dilate(current_mask, kernel, iterations=1)
            new_ring = (dilated > 0) & (current_mask == 0)

            if new_ring.sum() == 0:
                break

            ring_intensity = cd31_norm[new_ring].mean()

            # Stop if signal dropped
            if ring_intensity < stop_threshold:
                break

            # Only expand if signal is still strong
            current_mask = dilated

        # Assign label to final expanded region
        labels[current_mask > 0] = label_id

    return labels

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

    # Apply photobleaching correction + normalization to all channels
    sma_corrected = correct_photobleaching(tiles[SMA].astype(np.float32))
    nuclear_corrected = correct_photobleaching(tiles[NUCLEAR].astype(np.float32))
    cd31_corrected = correct_photobleaching(tiles[CD31].astype(np.float32))
    pm_corrected = correct_photobleaching(tiles[PM].astype(np.float32))

    sma_norm = normalize_channel(sma_corrected)
    nuclear_norm = normalize_channel(nuclear_corrected)
    cd31_norm = normalize_channel(cd31_corrected)
    pm_norm = normalize_channel(pm_corrected)

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
        del tiles, sma_corrected, nuclear_corrected, cd31_corrected, pm_corrected
        del sma_norm, nuclear_norm, cd31_norm, pm_norm, sma_rgb, display_rgb, masks
        gc.collect()
        return []

    # Watershed expansion
    labels = dilate_until_signal_drops(lumens, cd31_norm, scale_factor)

    # First pass: collect all candidate vessels with their CD31 enrichment ratios
    # for GMM-based adaptive thresholding
    candidates = []
    cd31_enrichment_ratios = []

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
        cd31_enrichment = cd31_in_wall / (cd31_in_lumen + 1e-6)

        # Store candidate for second pass
        candidates.append({
            'lumen': lumen,
            'wall_mask': wall_mask,
            'outer_contour': outer_contour,
            'inner_contour': inner_contour,
            'cd31_in_wall': cd31_in_wall,
            'cd31_in_lumen': cd31_in_lumen,
            'cd31_enrichment': cd31_enrichment,
        })
        cd31_enrichment_ratios.append(cd31_enrichment)

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
            'outer_diameter_um': float(outer_diameter),
            'inner_diameter_um': float(inner_diameter),
            'wall_thickness_um': float(wall_thickness),
            'outer_area_px': float(outer_area),
            'inner_area_px': float(inner_area),
            'wall_area_px': float(wall_area),
            **lumen['stats']
        }

        # Save crop with contours (use tile-scale contours for drawing on tile-scale image)
        crop_path = save_vessel_crop(vessel, display_rgb, outer_contour, inner_contour, tile_x, tile_y, tile_size, scale_factor)
        vessel['crop_path'] = crop_path

        vessels.append(vessel)

    # Cleanup large arrays before returning
    del tiles, sma_corrected, nuclear_corrected, cd31_corrected, pm_corrected
    del sma_norm, nuclear_norm, cd31_norm, pm_norm, sma_rgb, display_rgb, masks, labels
    gc.collect()

    return vessels

def compute_vessel_iou(v1, v2):
    """
    Compute IoU between two vessels based on their bounding boxes.

    Uses bounding box IoU as a fast approximation. For more accurate
    polygon IoU, use shapely.

    Args:
        v1, v2: Vessel dicts with 'outer_contour' keys

    Returns:
        float: IoU value between 0 and 1
    """
    # Get bounding boxes from contours
    c1 = np.array(v1['outer_contour']).reshape(-1, 2)  # Flatten to (N, 2)
    c2 = np.array(v2['outer_contour']).reshape(-1, 2)

    if c1.shape[0] < 3 or c2.shape[0] < 3:
        return 0.0  # Need at least 3 points for a valid contour

    # Bounding boxes: [x_min, y_min, x_max, y_max]
    box1 = [c1[:, 0].min(), c1[:, 1].min(), c1[:, 0].max(), c1[:, 1].max()]
    box2 = [c2[:, 0].min(), c2[:, 1].min(), c2[:, 0].max(), c2[:, 1].max()]

    # Intersection
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def merge_vessels_across_scales(vessels, iou_threshold=None, coverage_threshold=None):
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
    max_diam_px = max(v['outer_diameter_um'] / 0.1725 for v in sorted_vessels)
    search_radius = max_diam_px * 1.5

    print(f"  Building spatial index for {n} vessels (search radius: {search_radius:.0f}px)...")

    # Pre-compute all neighbor lists (avoids duplicate queries)
    neighbor_lists = tree.query_ball_point(centers, search_radius)

    # First pass: collect pairwise IoUs using cached neighbor lists
    all_ious = []
    checked_pairs = set()

    for i in range(n):
        v1 = sorted_vessels[i]
        nearby_indices = neighbor_lists[i]

        for j in nearby_indices:
            if j <= i:  # Skip self and already-checked pairs
                continue

            pair_key = (i, j)
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            v2 = sorted_vessels[j]
            iou = compute_vessel_iou(v1, v2)
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

    # Group vessels with IoU > threshold using Union-Find
    for i in range(n):
        for j in neighbor_lists[i]:
            if j <= i:
                continue
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
        outer_contour = np.array(v['outer_contour'])
        if outer_contour.size == 0:
            continue

        # Scale contour from full-res to working scale
        scaled_contour = (outer_contour / work_scale).astype(np.int32)

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


def main():
    import sys
    print("=" * 60, flush=True)
    print("Multi-Scale SAM2 Vessel Detection", flush=True)
    print("=" * 60, flush=True)

    # Open CZI (don't load full-res into RAM)
    loader = CZILoader(CZI_PATH)
    mosaic_size = loader.mosaic_size
    print(f"\nMosaic size (full res): {mosaic_size}", flush=True)

    # Load all channels at BASE_SCALE (1/2) - 4x smaller than full res
    channel_cache = DownsampledChannelCache(loader, [NUCLEAR, CD31, SMA, PM], BASE_SCALE)

    # Load SAM2
    print("\nLoading SAM2...", flush=True)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "/home/dude/code/xldvp_seg_repo/checkpoints/sam2.1_hiera_large.pt",
        device="cuda"
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=48,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.7,
        min_mask_region_area=100,
    )

    # Compute adaptive Sobel threshold using the first (coarsest) scale
    adaptive_sobel_threshold = compute_adaptive_sobel_threshold(channel_cache, SCALES[0]['scale_factor'])
    print(f"Adaptive Sobel threshold: {adaptive_sobel_threshold:.4f}")

    all_vessels = []

    # Process each scale
    for scale_config in SCALES:
        scale_factor = scale_config['scale_factor']
        tile_size = scale_config.get('tile_size', TILE_SIZE)
        stride = int(tile_size * (1 - TILE_OVERLAP))  # With 50% overlap, stride = tile_size/2

        print(f"\n{'='*40}", flush=True)
        print(f"Processing scale {scale_config['name']}", flush=True)
        print(f"  Scale factor: {scale_factor}x", flush=True)
        print(f"  Tile size: {tile_size}px (covers {tile_size * scale_factor}px at full res)", flush=True)
        print(f"  Stride: {stride}px ({int(TILE_OVERLAP*100)}% overlap)", flush=True)
        print(f"  Diameter range: {scale_config['min_diam_um']}-{scale_config['max_diam_um']} µm", flush=True)
        print(f"{'='*40}", flush=True)

        # Create tile grid for this scale (in SCALED coordinates)
        # With overlap, we use stride instead of tile_size for stepping
        scaled_mosaic_x = mosaic_size[0] // scale_factor
        scaled_mosaic_y = mosaic_size[1] // scale_factor
        n_tiles_x = max(1, (scaled_mosaic_x - tile_size) // stride + 1)
        n_tiles_y = max(1, (scaled_mosaic_y - tile_size) // stride + 1)
        total_tiles = n_tiles_x * n_tiles_y
        print(f"Scaled mosaic: {scaled_mosaic_x} x {scaled_mosaic_y}", flush=True)
        print(f"Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles", flush=True)

        # Find tissue tiles (coordinates in scaled space)
        print("Identifying tissue tiles...", flush=True)
        tissue_tiles = []

        for ty in tqdm(range(n_tiles_y), desc="Scanning"):
            for tx in range(n_tiles_x):
                # tile_x, tile_y are in SCALED coordinates (using stride for stepping)
                tile_x = tx * stride
                tile_y = ty * stride

                # Check for tissue using the scaled tile (from cache)
                tile = channel_cache.get_tile(tile_x, tile_y, tile_size, SMA, scale_factor)
                if tile is not None and is_tissue_tile(tile):
                    tissue_tiles.append((tile_x, tile_y))

        print(f"Found {len(tissue_tiles)} tissue tiles", flush=True)

        # Sample (at 100%, this just takes all tiles)
        n_sample = max(1, int(len(tissue_tiles) * SAMPLE_FRACTION))
        sampled_tiles = random.sample(tissue_tiles, min(n_sample, len(tissue_tiles)))
        print(f"Processing {len(sampled_tiles)} tiles", flush=True)

        # Process
        scale_vessels = []
        for tile_x, tile_y in tqdm(sampled_tiles, desc="Processing"):
            try:
                vessels = process_tile_at_scale(
                    tile_x, tile_y, channel_cache, mask_generator, scale_config,
                    adaptive_sobel_threshold=adaptive_sobel_threshold
                )
                scale_vessels.extend(vessels)
            except Exception as e:
                print(f"\nError at ({tile_x}, {tile_y}): {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                # Memory cleanup after each tile
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Found {len(scale_vessels)} vessels at scale {scale_config['name']}", flush=True)
        all_vessels.extend(scale_vessels)

        # Cleanup between scales
        gc.collect()
        torch.cuda.empty_cache()

    # Merge across scales (uses adaptive IoU threshold)
    print(f"\nTotal vessels before merge: {len(all_vessels)}")
    merged_vessels = merge_vessels_across_scales(all_vessels)
    print(f"After merge: {len(merged_vessels)}")

    # Apply adaptive diameter ratio filter (GMM-based)
    print(f"\nApplying adaptive diameter ratio filter...")
    filtered_vessels = adaptive_diameter_ratio_filter(merged_vessels)
    print(f"After diameter ratio filter: {len(filtered_vessels)}")
    merged_vessels = filtered_vessels

    # Report tiles skipped by Sobel pre-filter
    print(f"Tiles skipped by Sobel pre-filter: {TILES_SKIPPED_BY_SOBEL}")

    # Analyze vessel network topology (skeletonization)
    network_metrics = analyze_vessel_network(merged_vessels, mosaic_size)

    # ==========================================================================
    # Cell Composition Analysis - 3-Pass Approach
    # ==========================================================================
    # Pass 1: Segment cells in all vessels, collect intensities
    # Pass 2: Fit slide-wide GMM and classify
    # Pass 3: Update vessel dicts with cell_composition
    # ==========================================================================

    print("\n" + "=" * 60)
    print("Cell Composition Analysis")
    print("=" * 60)

    gmm_info = {'error': 'Cell composition analysis skipped - no Cellpose'}

    if CELLPOSE_AVAILABLE and len(merged_vessels) > 0:
        print(f"Analyzing cell composition for {len(merged_vessels)} vessels...")

        # Pass 1: Segment cells in all vessels, collect intensities
        all_cell_data = []
        total_cells = 0

        for i, vessel in enumerate(tqdm(merged_vessels, desc="Pass 1: Segmenting cells")):
            try:
                # Get vessel ROI center and size
                global_x, global_y = vessel['global_center']
                outer_diam_um = vessel['outer_diameter_um']

                # Convert to pixels at base scale
                # ROI size = 1.5x outer diameter to capture full vessel
                roi_size_um = outer_diam_um * 1.5
                roi_size_px = int(roi_size_um / (0.1725 * BASE_SCALE))  # pixels at base scale
                roi_size_px = max(roi_size_px, 50)  # minimum size

                # Convert global coords to base scale
                base_x = global_x // BASE_SCALE
                base_y = global_y // BASE_SCALE

                # Calculate ROI bounds at base scale
                x1 = max(0, base_x - roi_size_px // 2)
                y1 = max(0, base_y - roi_size_px // 2)
                x2 = min(channel_cache.base_width, base_x + roi_size_px // 2)
                y2 = min(channel_cache.base_height, base_y + roi_size_px // 2)

                if x2 <= x1 or y2 <= y1:
                    all_cell_data.append({'cd31': np.array([]), 'sma': np.array([])})
                    continue

                # Extract ROIs from channel cache (at base scale)
                pm_roi = channel_cache.channels[PM][y1:y2, x1:x2]
                nuclear_roi = channel_cache.channels[NUCLEAR][y1:y2, x1:x2]
                cd31_roi = channel_cache.channels[CD31][y1:y2, x1:x2]
                sma_roi = channel_cache.channels[SMA][y1:y2, x1:x2]

                # Segment cells using Cellpose
                cell_masks, num_cells = segment_cells_in_vessel(pm_roi, nuclear_roi)

                if num_cells == 0:
                    all_cell_data.append({'cd31': np.array([]), 'sma': np.array([])})
                    continue

                # Measure intensities
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

            # Collect all intensities
            all_cd31 = np.concatenate([d['cd31'] for d in all_cell_data if len(d['cd31']) > 0])
            all_sma = np.concatenate([d['sma'] for d in all_cell_data if len(d['sma']) > 0])

            # Fit GMM
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

            # Summary stats
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

            # Add empty cell_composition to all vessels
            for vessel in merged_vessels:
                vessel['cell_composition'] = compute_cell_composition({}, 0, 0)
    else:
        if not CELLPOSE_AVAILABLE:
            print("Cellpose not available - skipping cell composition analysis")
        else:
            print("No vessels found - skipping cell composition analysis")

        # Add empty cell_composition to all vessels
        for vessel in merged_vessels:
            vessel['cell_composition'] = compute_cell_composition({}, 0, 0)

    # Build output data with both vessels and network metrics
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

    # Save JSON with vessels and network metrics
    output_path = os.path.join(OUTPUT_DIR, 'vessel_detections_multiscale.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Export to GeoPackage for GIS compatibility
    gpkg_path = os.path.join(OUTPUT_DIR, 'vessel_detections_multiscale.gpkg')
    export_to_geodataframe(merged_vessels, gpkg_path)

    # Stats
    if len(merged_vessels) > 0:
        diameters = [v['outer_diameter_um'] for v in merged_vessels]
        print(f"\nDiameter stats:")
        print(f"  min={min(diameters):.1f}, max={max(diameters):.1f}, mean={np.mean(diameters):.1f} µm")

        # By scale
        for scale_config in SCALES:
            count = sum(1 for v in merged_vessels if v['scale'] == scale_config['name'])
            print(f"  Scale {scale_config['name']}: {count} vessels")

    # Cleanup
    channel_cache.release()
    loader.close()

    # Generate HTML
    generate_html(merged_vessels)

    # Generate histograms
    generate_histograms(merged_vessels, OUTPUT_DIR)

    print("\nDone!")

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

    # --- Combined figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

    # Diameter by scale
    ax = axes[1, 1]
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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_combined.png'), dpi=150)
    plt.close()

    print(f"  Saved histograms to {output_dir}")
    print(f"    - histogram_diameters.png")
    print(f"    - histogram_wall_thickness.png")
    print(f"    - histogram_lumen_wall_ratio.png")
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
            # Convert contours to Shapely Polygons
            # Contours are stored as [[[x, y]], [[x, y]], ...] from OpenCV
            outer_coords = np.array(v['outer_contour']).squeeze()
            inner_coords = np.array(v['inner_contour']).squeeze()

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

        # Read crop and convert to base64
        with open(crop_path, 'rb') as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Build features dict (package expects specific keys)
        features = {
            'outer_diameter_um': v.get('outer_diameter_um', 0),
            'inner_diameter_um': v.get('inner_diameter_um', 0),
            'wall_thickness_mean_um': v.get('wall_thickness_um', 0),
            'outer_area_px': v.get('outer_area_px', 0),
            'inner_area_px': v.get('inner_area_px', 0),
            'wall_area_px': v.get('wall_area_px', 0),
            'sma_ratio': v.get('sma_ratio', 0),
            'scale': v.get('scale', '?'),
            'global_x': v.get('global_center', [0, 0])[0],
            'global_y': v.get('global_center', [0, 0])[1],
        }

        samples.append({
            'uid': v.get('uid', ''),
            'image': img_base64,  # Just base64, export function adds prefix
            'features': features,
        })

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
