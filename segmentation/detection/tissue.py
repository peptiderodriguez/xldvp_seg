"""
Tissue detection module for cell segmentation pipelines.

Uses block-based variance analysis with K-means clustering to
automatically detect tissue-containing regions in microscopy images.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def compute_otsu_threshold(gray_image, max_subsample=10_000_000):
    """Compute Otsu threshold on a grayscale image, excluding black padding.

    Subsamples up to max_subsample non-zero pixels to avoid allocating a
    full thresholded copy of the image.

    Args:
        gray_image: 2D uint8 grayscale array
        max_subsample: Maximum pixels to feed to Otsu (default 10M)

    Returns:
        float: Otsu threshold value
    """
    # Subsample random pixels (avoids full-image allocation)
    h, w = gray_image.shape
    n_total = h * w
    n_sample = min(max_subsample, n_total)

    if n_sample < n_total:
        rows = np.random.randint(0, h, size=n_sample)
        cols = np.random.randint(0, w, size=n_sample)
        pixels = gray_image[rows, cols]
    else:
        pixels = gray_image.ravel()

    # Exclude black padding (gray == 0)
    pixels = pixels[pixels > 0]
    if len(pixels) < 100:
        logger.warning("  compute_otsu_threshold: <100 non-zero pixels, returning default 200")
        return 200.0

    otsu_val, _ = cv2.threshold(pixels.reshape(1, -1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(otsu_val)


def _normalize_to_uint8(data):
    """Normalize non-uint8 data to uint8, computing percentiles on non-zero pixels only.

    CZI padding is exactly 0; zeros are preserved in the output.

    Args:
        data: Image array (2D or 3D, any dtype except uint8)

    Returns:
        Tuple of (normalized_uint8, success). success=False if insufficient non-zero pixels.
    """
    zero_mask = data == 0
    nonzero = data[~zero_mask]
    if len(nonzero) < 10:
        return np.zeros_like(data, dtype=np.uint8), False
    p_low, p_high = np.percentile(nonzero, [1, 99])
    if p_high <= p_low:
        return np.zeros_like(data, dtype=np.uint8), False
    result = np.clip((data.astype(np.float64) - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
    result[zero_mask] = 0
    return result, True


def calculate_block_variances(gray_image, block_size=512):
    """
    Calculate variance and mean intensity for each block in the image.

    Args:
        gray_image: 2D grayscale image array
        block_size: Size of blocks to analyze

    Returns:
        Tuple of (variances, means) — lists of variance and mean intensity
        values for each block
    """
    variances = []
    means = []
    height, width = gray_image.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = gray_image[y:y+block_size, x:x+block_size]
            # Skip very small edge blocks (less than ~1.5% of full size)
            # This prevents near-degenerate blocks from skewing statistics
            # while still including most edge tissue
            if block.size < (block_size * block_size) // 64:
                continue
            # Exclude zero pixels (CZI padding) from statistics
            valid = block[block > 0]
            if len(valid) < 10:
                continue
            variances.append(np.var(valid.astype(np.float64)))
            means.append(np.mean(valid.astype(np.float64)))

    return variances, means


def is_tissue_block(gray_block, variance_threshold, modality=None,
                    intensity_threshold=220, min_tissue_pixel_frac=0.20):
    """Shared block-level tissue criterion.

    Brightfield: Otsu-only — >=20% of non-zero pixels below intensity_threshold.
        variance_threshold is ignored (pass 0).
    Fluorescence/default: variance-only (zero pixels excluded from computation).

    Args:
        gray_block: 2D grayscale array (single block)
        variance_threshold: Variance threshold (used for fluorescence only;
            ignored for brightfield)
        modality: 'brightfield' for H&E, None/other for fluorescence (OR logic)
        intensity_threshold: Pixel intensity cutoff (brightfield: Otsu threshold;
            fluorescence: Otsu on block means or hardcoded 220)
        min_tissue_pixel_frac: Minimum fraction of pixels below intensity_threshold
            to count as tissue (brightfield only, default 0.20)

    Returns:
        bool: True if the block is classified as tissue
    """
    # Exclude zero pixels (CZI padding) from variance computation
    valid = gray_block[gray_block > 0]
    if len(valid) < 10:
        return False
    if modality == 'brightfield':
        # Otsu-only: fraction of non-zero pixels below intensity_threshold
        tissue_frac = np.sum((gray_block > 0) & (gray_block < intensity_threshold)) / gray_block.size
        return tissue_frac >= min_tissue_pixel_frac
    else:
        # Fluorescence: variance-only on non-zero pixels
        var = np.var(valid.astype(np.float64))
        return var >= variance_threshold


def compute_pixel_level_tissue_mask(
    image: np.ndarray,
    variance_threshold: float,
    block_size: int = 7
) -> np.ndarray:
    """
    Compute pixel-level tissue mask using local variance.

    Identifies tissue pixels (high local variance) vs background pixels (low variance).
    Uses same variance threshold as tile-level tissue detection for consistency.

    Args:
        image: RGB or grayscale image (H, W, 3) or (H, W)
        variance_threshold: Variance threshold for tissue detection
        block_size: Local neighborhood size for variance computation (default: 7×7)

    Returns:
        Boolean mask (H, W) where True = tissue, False = background
    """
    # Convert to grayscale if needed
    # For uint16: use percentile normalization (consistent with calibrate_tissue_threshold)
    if image.dtype == np.uint16:
        if image.ndim == 3:
            gray_raw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_raw = image
        gray, _ = _normalize_to_uint8(gray_raw)
    elif image.ndim == 3:
        from segmentation.utils.detection_utils import safe_to_uint8
        gray = cv2.cvtColor(safe_to_uint8(image), cv2.COLOR_RGB2GRAY)
    else:
        from segmentation.utils.detection_utils import safe_to_uint8
        gray = safe_to_uint8(image)

    # Compute local variance using uniform filter
    # This is equivalent to computing variance in each block_size×block_size neighborhood
    from scipy.ndimage import uniform_filter

    gray_f = gray.astype(np.float32)

    # Local variance: E[X²] - E[X]²
    local_mean = uniform_filter(gray_f, size=block_size)
    local_mean_sq = uniform_filter(gray_f ** 2, size=block_size)
    local_variance = local_mean_sq - local_mean ** 2

    # Threshold: pixels with variance above threshold are tissue
    tissue_mask = local_variance >= variance_threshold

    return tissue_mask


def has_tissue(tile_image, variance_threshold, min_tissue_fraction=0.10, block_size=512,
               intensity_threshold=220, modality=None, min_tissue_pixel_frac=0.20,
               max_bg_intensity=None):
    """
    Check if a tile contains tissue using block-based variance and intensity.

    When modality='brightfield': Otsu-only — a block is tissue if >=20% of its
    non-black pixels are below the Otsu threshold. Variance is ignored.

    When modality is None/other (default, fluorescence): uses variance-only.
    Zero pixels (CZI padding) are excluded from variance computation.

    Args:
        tile_image: RGB or grayscale image array (uint8 or uint16)
        variance_threshold: Variance threshold for tissue detection
            (calibrated on percentile-normalized uint8 data)
        min_tissue_fraction: Minimum fraction of blocks that must be tissue
        block_size: Size of blocks for variance calculation
        intensity_threshold: Intensity cutoff for tissue detection (uint8 scale).
            Brightfield: Otsu on pixel grayscale. Fluorescence: Otsu on block means.
        modality: 'brightfield' for H&E, None/other for fluorescence
        min_tissue_pixel_frac: Min fraction of dark pixels per block (brightfield only)
        max_bg_intensity: Deprecated alias for intensity_threshold (backward compat)

    Returns:
        Tuple of (has_tissue: bool, tissue_fraction: float)
    """
    # Backward compatibility: max_bg_intensity overrides intensity_threshold
    if max_bg_intensity is not None:
        intensity_threshold = max_bg_intensity

    # Handle all-black tiles (empty CZI regions)
    if tile_image.max() == 0:
        return False, 0.0

    # Convert to grayscale if needed
    # For uint16: use percentile normalization (consistent with calibrate_tissue_threshold)
    # Simple /256 crushes dynamic range for fluorescence images where signal is in low uint16 band
    if tile_image.dtype == np.uint16:
        if tile_image.ndim == 3:
            gray_raw = cv2.cvtColor(tile_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_raw = tile_image
        gray, ok = _normalize_to_uint8(gray_raw)
        if not ok:
            return False, 0.0
    elif tile_image.ndim == 3:
        from segmentation.utils.detection_utils import safe_to_uint8
        gray = cv2.cvtColor(safe_to_uint8(tile_image), cv2.COLOR_RGB2GRAY)
    else:
        from segmentation.utils.detection_utils import safe_to_uint8
        gray = safe_to_uint8(tile_image)

    # Count tissue blocks using shared is_tissue_block() logic
    height, width = gray.shape
    tissue_blocks = 0
    total_blocks = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            if block.size < (block_size * block_size) // 64:
                continue
            total_blocks += 1
            if is_tissue_block(block, variance_threshold, modality=modality,
                               intensity_threshold=intensity_threshold,
                               min_tissue_pixel_frac=min_tissue_pixel_frac):
                tissue_blocks += 1

    if total_blocks == 0:
        return False, 0.0

    tissue_fraction = tissue_blocks / total_blocks

    return tissue_fraction >= min_tissue_fraction, tissue_fraction


def compute_variance_threshold(variances, default=15.0):
    """Compute tissue detection threshold from pre-collected variance samples using K-means.

    Uses 3-cluster K-means: background (low), tissue (medium), outliers (high).
    Returns max variance of the background cluster as the threshold.

    Args:
        variances: array-like of variance values from calculate_block_variances()
        default: fallback threshold if insufficient samples

    Returns:
        float: variance threshold for tissue detection
    """
    variances = np.asarray(variances, dtype=float)

    if len(variances) < 10:
        logger.warning(f"Not enough variance samples ({len(variances)}), using default threshold {default}")
        return default

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(variances.reshape(-1, 1))

    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_

    # Background cluster = lowest center
    bg_cluster_idx = np.argmin(centers)
    bg_variances = variances[labels == bg_cluster_idx]

    threshold = float(np.max(bg_variances)) if len(bg_variances) > 0 else default

    sorted_centers = sorted(centers)
    logger.info(f"  K-means centers: {sorted_centers[0]:.1f} (bg), {sorted_centers[1]:.1f} (tissue), {sorted_centers[2]:.1f} (outliers)")
    logger.info(f"  Threshold (bg cluster max): {threshold:.1f}")

    return threshold


def compute_tissue_thresholds(variances, means, default_var=15.0, default_intensity=220,
                              modality=None, pixel_samples=None):
    """Compute variance and intensity thresholds from block-level calibration data.

    When modality='brightfield':
        - Variance threshold: K-means 3-cluster → ÷3 reduction
        - Intensity threshold: Otsu on pixel_samples (grayscale pixel values)
    When modality is None/other (default):
        - Variance threshold: K-means 3-cluster (no reduction)
        - Intensity threshold: Otsu on block mean intensities

    Args:
        variances: array-like of block variance values
        means: array-like of block mean intensity values (uint8 scale)
        default_var: fallback variance threshold
        default_intensity: fallback intensity threshold if Otsu fails
        modality: 'brightfield' for H&E, None/other for fluorescence
        pixel_samples: 1D array of grayscale pixel values for Otsu
            (required when modality='brightfield', ignored otherwise)

    Returns:
        tuple: (variance_threshold, intensity_threshold)
    """
    variance_threshold = compute_variance_threshold(variances, default=default_var)

    if modality == 'brightfield':
        # Brightfield: reduce variance threshold by 3× to catch lighter-stained tissue
        variance_threshold = variance_threshold / 3.0
        logger.info(f"  Variance threshold (÷3 brightfield reduction): {variance_threshold:.1f}")

        # Brightfield: Otsu on pixel grayscale values (not block means)
        if pixel_samples is not None and len(pixel_samples) >= 100:
            pixel_u8 = np.clip(np.asarray(pixel_samples, dtype=float), 0, 255).astype(np.uint8)
            otsu_val, _ = cv2.threshold(pixel_u8.reshape(1, -1), 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            intensity_threshold = float(otsu_val)
            logger.info(f"  Intensity threshold (Otsu on pixel grayscale): {intensity_threshold:.1f}")
            logger.info(f"  Pixel sample range: {pixel_u8.min():.0f} – "
                        f"{np.median(pixel_u8):.0f} (median) – {pixel_u8.max():.0f}")
        else:
            logger.warning(f"  No pixel_samples for brightfield Otsu, using default {default_intensity}")
            intensity_threshold = default_intensity

        return variance_threshold, intensity_threshold

    # Default (fluorescence): Otsu on block mean intensities
    means = np.asarray(means, dtype=float)
    if len(means) < 10:
        logger.warning(f"Not enough mean samples ({len(means)}), using default intensity threshold {default_intensity}")
        return variance_threshold, default_intensity

    # Otsu on block mean intensities to find tissue/background boundary
    means_uint8 = np.clip(means, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(means_uint8.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intensity_threshold = float(otsu_val)

    logger.info(f"  Intensity threshold (Otsu on block means): {intensity_threshold:.1f}")
    logger.info(f"  Block mean range: {means.min():.1f} – {np.median(means):.1f} (median) – {means.max():.1f}")

    return variance_threshold, intensity_threshold


def calibrate_tissue_threshold(
    tiles,
    reader=None,
    x_start=0,
    y_start=0,
    calibration_samples=50,
    block_size=512,
    image_array=None,
    channel=0,
    tile_size=3000,
    loader=None,
):
    """
    Auto-detect variance threshold using K-means clustering.

    Uses 3 clusters to separate: background (low var), tissue (medium var),
    and artifacts/noise (high var). Returns max variance of background cluster
    as the threshold.

    Priority for tile data source: image_array > loader > reader

    Args:
        tiles: List of tile coordinates [(x, y), ...] or tile dicts
        reader: CZI reader object (deprecated, use loader or image_array instead)
        x_start, y_start: CZI ROI offset (only used with reader)
        calibration_samples: Number of tiles to sample for calibration
        block_size: Size of blocks for variance calculation
        image_array: Pre-loaded image array (recommended, from loader.channel_data)
        channel: Channel index for CZI reader or loader
        tile_size: Size of tiles
        loader: CZILoader instance (preferred over reader for RAM-first architecture)

    Returns:
        float: Variance threshold for tissue detection
    """
    logger.info(f"Calibrating tissue threshold (K-means 3-cluster on {calibration_samples} random tiles)...")

    # Sample tiles for calibration
    n_tiles = len(tiles)
    n_samples = min(calibration_samples, n_tiles)
    sample_indices = np.random.choice(n_tiles, n_samples, replace=False)

    all_variances = []

    for idx in tqdm(sample_indices, desc="Calibrating"):
        tile = tiles[idx]

        # Handle different tile formats
        if isinstance(tile, dict):
            tile_x = tile.get('x', tile.get('tile_x', 0))
            tile_y = tile.get('y', tile.get('tile_y', 0))
        else:
            tile_x, tile_y = tile

        # Get tile image (priority: image_array > loader > reader)
        if image_array is not None:
            # Extract from pre-loaded array
            tile_img = image_array[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
        elif loader is not None:
            # Use CZILoader (RAM-first architecture)
            try:
                tile_img = loader.get_tile(tile_x, tile_y, tile_size, channel=channel)
                if tile_img is None or tile_img.size == 0:
                    continue
            except Exception as e:
                logger.debug(f"Failed to load tile ({tile_x}, {tile_y}) via loader: {e}")
                continue
        elif reader is not None:
            # Read from CZI (deprecated path)
            try:
                tile_img = reader.read_mosaic(
                    region=(x_start + tile_x, y_start + tile_y, tile_size, tile_size),
                    scale_factor=1,
                    C=channel
                )
                if tile_img is None or tile_img.size == 0:
                    continue
                tile_img = np.squeeze(tile_img)
            except Exception as e:
                logger.debug(f"Failed to read tile ({tile_x}, {tile_y}) via CZI reader: {e}")
                continue
        else:
            raise ValueError("Either reader, loader, or image_array must be provided")

        # Skip empty tiles
        if tile_img.max() == 0:
            continue

        # Normalize to uint8 for variance calculation
        if tile_img.dtype != np.uint8:
            tile_img, ok = _normalize_to_uint8(tile_img)
            if not ok:
                continue

        # Convert to grayscale if needed
        if tile_img.ndim == 3:
            gray = cv2.cvtColor(tile_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile_img

        # Calculate block variances
        variances, _ = calculate_block_variances(gray, block_size)
        all_variances.extend(variances)

    if len(all_variances) < 10:
        logger.warning("Not enough samples, using default threshold 15.0")
        return 15.0

    return compute_variance_threshold(np.array(all_variances))


def filter_tissue_tiles(
    tiles,
    variance_threshold,
    reader=None,
    x_start=0,
    y_start=0,
    image_array=None,
    channel=0,
    tile_size=3000,
    block_size=512,
    n_workers=None,
    show_progress=True,
    loader=None,
    modality=None,
    min_tissue_pixel_frac=0.20,
    intensity_threshold=220,
):
    """
    Filter tiles to only those containing tissue.

    Priority for tile data source: image_array > loader > reader

    Args:
        tiles: List of tile coordinates or tile dicts
        variance_threshold: Threshold from calibrate_tissue_threshold
        reader: CZI reader (deprecated, use loader or image_array instead)
        x_start, y_start: CZI ROI offset (only used with reader)
        image_array: Pre-loaded image array (recommended, from loader.channel_data)
        channel: Channel index for CZI reader or loader
        tile_size: Size of tiles
        block_size: Block size for variance calculation
        n_workers: Number of parallel workers (default: 80% of CPUs)
        show_progress: Whether to show progress bar
        loader: CZILoader instance (preferred over reader for RAM-first architecture)
        modality: 'brightfield' for H&E, None/other for fluorescence
        min_tissue_pixel_frac: Min fraction of dark pixels per block (brightfield only)
        intensity_threshold: Intensity cutoff passed to has_tissue()

    Returns:
        List of tiles that contain tissue
    """
    import os

    if n_workers is None:
        n_workers = max(1, int(os.cpu_count() * 0.8))

    logger.info("Filtering tiles to tissue-containing only...")
    logger.info(f"  Using {n_workers} workers for parallel checking")

    def check_tile(tile):
        """Check if a single tile contains tissue."""
        # Handle different tile formats
        if isinstance(tile, dict):
            tile_x = tile.get('x', tile.get('tile_x', 0))
            tile_y = tile.get('y', tile.get('tile_y', 0))
        else:
            tile_x, tile_y = tile

        # Get tile image (priority: image_array > loader > reader)
        try:
            if image_array is not None:
                tile_img = image_array[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
            elif loader is not None:
                # Use CZILoader (RAM-first architecture)
                tile_img = loader.get_tile(tile_x, tile_y, tile_size, channel=channel)
                if tile_img is None or tile_img.size == 0:
                    return None
            elif reader is not None:
                # Read from CZI (deprecated path)
                tile_img = reader.read_mosaic(
                    region=(x_start + tile_x, y_start + tile_y, tile_size, tile_size),
                    scale_factor=1,
                    C=channel
                )
                if tile_img is None or tile_img.size == 0:
                    return None
                tile_img = np.squeeze(tile_img)
            else:
                return None

            # Skip empty tiles
            if tile_img.max() == 0:
                return None

            # Normalize to uint8
            if tile_img.dtype != np.uint8:
                tile_img, ok = _normalize_to_uint8(tile_img)
                if not ok:
                    return None

            has_tissue_flag, _ = has_tissue(tile_img, variance_threshold, block_size=block_size,
                                             modality=modality, intensity_threshold=intensity_threshold,
                                             min_tissue_pixel_frac=min_tissue_pixel_frac)

            if has_tissue_flag:
                return tile
            return None

        except Exception as e:
            logger.debug(f"Error checking tissue in tile: {e}")
            return None

    tissue_tiles = []

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(check_tile, t): t for t in tiles}
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(tiles), desc="Checking tissue")

            for future in iterator:
                result = future.result()
                if result is not None:
                    tissue_tiles.append(result)
    else:
        iterator = tiles
        if show_progress:
            iterator = tqdm(tiles, desc="Checking tissue")

        for tile in iterator:
            result = check_tile(tile)
            if result is not None:
                tissue_tiles.append(result)

    # Sort by tile coordinates for deterministic order — as_completed() returns
    # results in completion order which varies between runs, making deduplication
    # and sampling non-reproducible.
    def _tile_sort_key(tile):
        if isinstance(tile, dict):
            return (tile.get('x', tile.get('tile_x', 0)),
                    tile.get('y', tile.get('tile_y', 0)))
        return tuple(tile)

    tissue_tiles.sort(key=_tile_sort_key)

    logger.info(f"  Tissue tiles: {len(tissue_tiles)} / {len(tiles)} ({100*len(tissue_tiles)/len(tiles):.1f}%)")

    return tissue_tiles
