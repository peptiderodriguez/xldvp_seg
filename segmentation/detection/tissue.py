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
    Calculate variance for each block in the image.

    Args:
        gray_image: 2D grayscale image array
        block_size: Size of blocks to analyze

    Returns:
        List of variance values for each block
    """
    variances = []
    height, width = gray_image.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = gray_image[y:y+block_size, x:x+block_size]
            # Skip blocks that are too small (less than 25% of full size)
            if block.size < (block_size * block_size) / 4:
                continue
            variances.append(np.var(block))

    return variances


def has_tissue(tile_image, variance_threshold, min_tissue_fraction=0.15, block_size=512):
    """
    Check if a tile contains tissue using block-based variance.

    Args:
        tile_image: RGB or grayscale image array (uint8)
        variance_threshold: Variance threshold for tissue detection
        min_tissue_fraction: Minimum fraction of blocks that must be tissue
        block_size: Size of blocks for variance calculation

    Returns:
        Tuple of (has_tissue: bool, tissue_fraction: float)
    """
    # Handle all-black tiles (empty CZI regions)
    if tile_image.max() == 0:
        return False, 0.0

    # Convert to grayscale if needed
    if tile_image.ndim == 3:
        gray = cv2.cvtColor(tile_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = tile_image.astype(np.uint8)

    variances = calculate_block_variances(gray, block_size)

    if len(variances) == 0:
        return False, 0.0

    tissue_blocks = sum(1 for v in variances if v >= variance_threshold)
    tissue_fraction = tissue_blocks / len(variances)

    return tissue_fraction >= min_tissue_fraction, tissue_fraction


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
        variances = calculate_block_variances(gray, block_size)
        all_variances.extend(variances)

    if len(all_variances) < 10:
        logger.warning("Not enough samples, using default threshold 15.0")
        return 15.0

    # K-means with 3 clusters: background (low var), tissue (medium var), artifacts (high var)
    variances_array = np.array(all_variances).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(variances_array)

    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_

    # Find the cluster with lowest center (background)
    bg_cluster_idx = np.argmin(centers)
    bottom_cluster_variances = variances_array[labels == bg_cluster_idx].flatten()

    threshold = float(np.max(bottom_cluster_variances)) if len(bottom_cluster_variances) > 0 else 15.0

    logger.info(f"  K-means centers: {sorted(centers)[0]:.1f} (bg), {sorted(centers)[1]:.1f} (tissue), {sorted(centers)[2]:.1f} (outliers)")
    logger.info(f"  Threshold (bg cluster max): {threshold:.1f}")

    return threshold


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
