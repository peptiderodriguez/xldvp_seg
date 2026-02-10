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
            variances.append(np.var(block))
            means.append(np.mean(block))

    return variances, means


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
            gray_raw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        else:
            gray_raw = image.astype(np.float64)
        p_low, p_high = np.percentile(gray_raw, [1, 99])
        if p_high > p_low:
            gray = np.clip((gray_raw - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        else:
            gray = np.zeros(gray_raw.shape, dtype=np.uint8)
    elif image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

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


def has_tissue(tile_image, variance_threshold, min_tissue_fraction=0.10, block_size=512, max_bg_intensity=220):
    """
    Check if a tile contains tissue using block-based variance and intensity.

    A block is classified as tissue if it has high variance (textured) OR
    low mean intensity (darker than background). This dual criterion catches
    uniform tissue blocks that have low variance but are clearly not background.

    Args:
        tile_image: RGB or grayscale image array (uint8 or uint16)
        variance_threshold: Variance threshold for tissue detection
            (calibrated on percentile-normalized uint8 data)
        min_tissue_fraction: Minimum fraction of blocks that must be tissue
        block_size: Size of blocks for variance calculation
        max_bg_intensity: Maximum mean intensity for background blocks (uint8 scale).
            Blocks with mean intensity below this are considered tissue regardless
            of variance. Default 220 is conservative for H&E (background ~230-250).

    Returns:
        Tuple of (has_tissue: bool, tissue_fraction: float)
    """
    # Handle all-black tiles (empty CZI regions)
    if tile_image.max() == 0:
        return False, 0.0

    # Reject scan boundary tiles (mix of black CZI padding + background)
    # CZI padding is exactly 0; use == 0 (not < 5) so fluorescence tiles
    # with near-zero background are not rejected
    if tile_image.ndim == 3:
        raw_check = np.min(tile_image, axis=2)  # any channel being 0
    else:
        raw_check = tile_image
    black_fraction = np.mean(raw_check == 0)
    if black_fraction > 0.02:  # >2% exactly-zero pixels = scan boundary
        return False, 0.0

    # Convert to grayscale if needed
    # For uint16: use percentile normalization (consistent with calibrate_tissue_threshold)
    # Simple /256 crushes dynamic range for fluorescence images where signal is in low uint16 band
    if tile_image.dtype == np.uint16:
        if tile_image.ndim == 3:
            gray_raw = cv2.cvtColor(tile_image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        else:
            gray_raw = tile_image.astype(np.float64)
        p_low, p_high = np.percentile(gray_raw, [1, 99])
        if p_high > p_low:
            gray = np.clip((gray_raw - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        else:
            return False, 0.0
    elif tile_image.ndim == 3:
        gray = cv2.cvtColor(tile_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = tile_image.astype(np.uint8)

    variances, means = calculate_block_variances(gray, block_size)

    if len(variances) == 0:
        return False, 0.0

    tissue_blocks = sum(
        1 for v, m in zip(variances, means)
        if v >= variance_threshold or m < max_bg_intensity
    )
    tissue_fraction = tissue_blocks / len(variances)

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


def compute_tissue_thresholds(variances, means, default_var=15.0, default_intensity=220):
    """Compute variance and intensity thresholds from block-level calibration data.

    Runs K-means 3-cluster on variances to get variance threshold.
    Runs Otsu on block mean intensities to find the tissue/background boundary,
    replacing the hardcoded max_bg_intensity=220.

    Args:
        variances: array-like of block variance values
        means: array-like of block mean intensity values (uint8 scale)
        default_var: fallback variance threshold
        default_intensity: fallback intensity threshold if Otsu fails

    Returns:
        tuple: (variance_threshold, intensity_threshold)
    """
    variance_threshold = compute_variance_threshold(variances, default=default_var)

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

        # Skip scan boundary tiles (mix of black CZI padding + background)
        # Use == 0 (not < 5) so fluorescence tiles with near-zero background pass
        if tile_img.ndim == 3:
            raw_check = np.min(tile_img, axis=2)
        else:
            raw_check = tile_img
        if np.mean(raw_check == 0) > 0.02:
            continue

        # Normalize to uint8 for variance calculation
        if tile_img.dtype != np.uint8:
            p_low, p_high = np.percentile(tile_img, [1, 99])
            if p_high > p_low:
                tile_img = np.clip((tile_img - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
            else:
                tile_img = np.zeros_like(tile_img, dtype=np.uint8)

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
                p_low, p_high = np.percentile(tile_img, [1, 99])
                if p_high > p_low:
                    tile_img = np.clip((tile_img - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
                else:
                    return None

            has_tissue_flag, _ = has_tissue(tile_img, variance_threshold, block_size=block_size)

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

    logger.info(f"  Tissue tiles: {len(tissue_tiles)} / {len(tiles)} ({100*len(tissue_tiles)/len(tiles):.1f}%)")

    return tissue_tiles
