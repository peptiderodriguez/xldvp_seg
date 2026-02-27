#!/usr/bin/env python3
"""
Convert CZI mosaic files to OME-Zarr format with multi-resolution pyramids.

Uses aicspylibczi directly (NOT AICSImage) to avoid the edge tile stitching bug
where smaller edge tiles get broadcast incorrectly.

Key features:
- Handles edge tiles with different sizes than interior tiles
- Tile-by-tile writing to avoid loading entire mosaic into RAM
- Multi-resolution pyramid generation for efficient Napari viewing
- Proper OME-NGFF metadata

Usage:
    python czi_to_ome_zarr.py /path/to/input.czi /path/to/output.zarr

    # With options
    python czi_to_ome_zarr.py input.czi output.zarr \
        --channels 0 1 2 \
        --pyramid-levels 5 \
        --chunk-size 1024

Author: Claude (Anthropic)
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

try:
    from aicspylibczi import CziFile
except ImportError:
    print("ERROR: aicspylibczi is required. Install with: pip install aicspylibczi")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_mosaic_info(czi: CziFile, scene: int = 0) -> Dict:
    """
    Extract mosaic information from CZI file.

    Args:
        czi: CziFile object
        scene: Scene index (0-based, default 0)

    Returns dict with:
        - bbox: full mosaic bounding box (x, y, w, h)
        - dims_shape: dimension info
        - n_channels: number of channels
        - n_tiles: number of M tiles
        - tiles_per_row: tiles in X direction
        - tiles_per_col: tiles in Y direction
        - nominal_tile_size: expected tile size (may differ for edge tiles)
        - pixel_type: data type (e.g., 'uint16')
    """
    # Get scene-specific bounding box
    try:
        bbox = czi.get_mosaic_scene_bounding_box(index=scene)
    except TypeError:
        # Older aicspylibczi without scene support — fall back to global bbox
        bbox = czi.get_mosaic_bounding_box()
    dims_shape = czi.get_dims_shape()[0]

    # Extract dimensions
    n_channels = dims_shape['C'][1] - dims_shape['C'][0]
    n_tiles = dims_shape['M'][1] - dims_shape['M'][0]

    # Get nominal tile size from first tile
    # Build kwargs based on available dimensions
    tile_kwargs = {'M': 0}
    if 'S' in dims_shape:
        tile_kwargs['S'] = scene
    if 'H' in dims_shape:
        tile_kwargs['H'] = 0
    if 'C' in dims_shape:
        tile_kwargs['C'] = 0

    try:
        first_tile_bbox = czi.get_mosaic_tile_bounding_box(**tile_kwargs)
        nominal_tile_size = (first_tile_bbox.w, first_tile_bbox.h)
    except Exception as e:
        logger.warning(f"Could not get tile bounding box: {e}, estimating from mosaic")
        # Estimate from typical Zeiss tile sizes
        nominal_tile_size = (2048, 2048)

    # Calculate grid dimensions
    tiles_per_row = (bbox.w + nominal_tile_size[0] - 1) // nominal_tile_size[0]
    tiles_per_col = (bbox.h + nominal_tile_size[1] - 1) // nominal_tile_size[1]

    # Get pixel type
    pixel_type = czi.pixel_type

    return {
        'bbox': {'x': bbox.x, 'y': bbox.y, 'w': bbox.w, 'h': bbox.h},
        'dims_shape': dims_shape,
        'n_channels': n_channels,
        'n_tiles': n_tiles,
        'tiles_per_row': tiles_per_row,
        'tiles_per_col': tiles_per_col,
        'nominal_tile_size': nominal_tile_size,
        'pixel_type': pixel_type,
        'scene': scene,
    }


def get_pixel_size_um(czi: CziFile) -> float:
    """Extract pixel size in micrometers from CZI metadata."""
    try:
        metadata = czi.meta
        scaling = metadata.find('.//Scaling/Items/Distance[@Id="X"]/Value')
        if scaling is not None:
            return float(scaling.text) * 1e6  # Convert meters to micrometers
    except Exception as e:
        logger.warning(f"Could not read pixel size from metadata: {e}")
    return 0.22  # Default pixel size


def preflight_check(
    czi: CziFile,
    mosaic_info: Dict,
    channels: List[int],
    output_path: Path,
    pyramid_levels: int,
    chunk_size: int,
) -> Dict:
    """
    Pre-flight validation of CZI before conversion.

    Performs:
    1. Structure validation
    2. Test read of a small region
    3. Output size estimation
    4. Disk space check

    Returns:
        Dict with validation results and estimates

    Raises:
        ValueError: If validation fails
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'estimates': {},
    }

    bbox = mosaic_info['bbox']
    n_channels = len(channels)

    logger.info("=" * 60)
    logger.info("PRE-FLIGHT CHECK")
    logger.info("=" * 60)

    # 1. Validate mosaic structure
    logger.info("\n[1/4] Validating mosaic structure...")

    if bbox['w'] <= 0 or bbox['h'] <= 0:
        results['errors'].append(f"Invalid bounding box: {bbox['w']}x{bbox['h']}")
        results['valid'] = False

    if mosaic_info['n_tiles'] == 0:
        results['errors'].append("No mosaic tiles found")
        results['valid'] = False

    # Check for unusually small/large dimensions
    if bbox['w'] > 500000 or bbox['h'] > 500000:
        results['warnings'].append(
            f"Very large image ({bbox['w']}x{bbox['h']}). "
            "Conversion may take several hours."
        )

    logger.info(f"  Mosaic: {bbox['w']:,} x {bbox['h']:,} pixels")
    logger.info(f"  Tiles: {mosaic_info['n_tiles']} ({mosaic_info['tiles_per_row']}x{mosaic_info['tiles_per_col']})")
    logger.info(f"  Tile size: {mosaic_info['nominal_tile_size']}")
    logger.info(f"  Channels to convert: {channels}")

    # 2. Test read a small region
    logger.info("\n[2/4] Testing read_mosaic()...")

    test_size = 512
    test_x = bbox['x']
    test_y = bbox['y']
    test_w = min(test_size, bbox['w'])
    test_h = min(test_size, bbox['h'])

    try:
        test_data = czi.read_mosaic(
            region=(test_x, test_y, test_w, test_h),
            scale_factor=1,
            C=channels[0],
            # Note: do NOT pass S= to read_mosaic on mosaic CZIs — use region= instead
        )
        test_data = np.squeeze(test_data)

        logger.info(f"  Test read successful: {test_data.shape}, dtype={test_data.dtype}")

        # Validate dtype
        if test_data.dtype not in [np.uint8, np.uint16, np.float32]:
            results['warnings'].append(
                f"Unusual dtype: {test_data.dtype}. May need adjustment."
            )

        # Check for all-zero data (potential read issue)
        if test_data.max() == 0:
            results['warnings'].append(
                "Test region is all zeros. May indicate read issue or empty region."
            )

    except Exception as e:
        results['errors'].append(f"Test read failed: {e}")
        results['valid'] = False
        logger.error(f"  Test read FAILED: {e}")

    # 3. Estimate output size
    logger.info("\n[3/4] Estimating output size...")

    # Determine dtype size
    pixel_type = str(mosaic_info['pixel_type']).lower()
    if 'uint16' in pixel_type or 'gray16' in pixel_type:
        bytes_per_pixel = 2
    elif 'uint8' in pixel_type or 'gray8' in pixel_type:
        bytes_per_pixel = 1
    else:
        bytes_per_pixel = 2  # Default

    # Raw size
    raw_size_bytes = bbox['w'] * bbox['h'] * n_channels * bytes_per_pixel

    # Pyramid overhead (~33% extra for all levels)
    pyramid_overhead = 1.33

    # Compression ratio estimate (zstd typically 2-4x for microscopy)
    compression_ratio = 2.5

    estimated_size_bytes = int(raw_size_bytes * pyramid_overhead / compression_ratio)

    results['estimates'] = {
        'raw_size_gb': raw_size_bytes / (1024**3),
        'estimated_output_gb': estimated_size_bytes / (1024**3),
        'n_strips': (bbox['h'] + 5000 - 1) // 5000,  # Default strip height
        'n_pyramid_levels': pyramid_levels,
    }

    logger.info(f"  Raw data size: {results['estimates']['raw_size_gb']:.1f} GB")
    logger.info(f"  Estimated output: {results['estimates']['estimated_output_gb']:.1f} GB (compressed)")
    logger.info(f"  Processing strips: {results['estimates']['n_strips']}")

    # 4. Check output path
    logger.info("\n[4/4] Checking output path...")

    output_parent = output_path.parent
    if not output_parent.exists():
        results['errors'].append(f"Output directory does not exist: {output_parent}")
        results['valid'] = False
    else:
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(output_parent)
            free_gb = free / (1024**3)
            needed_gb = results['estimates']['estimated_output_gb'] * 1.2  # 20% buffer

            logger.info(f"  Output directory: {output_parent}")
            logger.info(f"  Free space: {free_gb:.1f} GB")
            logger.info(f"  Estimated need: {needed_gb:.1f} GB")

            if free_gb < needed_gb:
                results['warnings'].append(
                    f"Low disk space: {free_gb:.1f} GB free, need ~{needed_gb:.1f} GB"
                )
        except Exception as e:
            results['warnings'].append(f"Could not check disk space: {e}")

    if output_path.exists():
        results['warnings'].append(f"Output path exists: {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)

    if results['errors']:
        logger.error("ERRORS:")
        for err in results['errors']:
            logger.error(f"  - {err}")
        results['valid'] = False

    if results['warnings']:
        logger.warning("WARNINGS:")
        for warn in results['warnings']:
            logger.warning(f"  - {warn}")

    if results['valid']:
        logger.info("PRE-FLIGHT CHECK: PASSED")
    else:
        logger.error("PRE-FLIGHT CHECK: FAILED")

    logger.info("=" * 60 + "\n")

    return results


def create_zarr_store(
    output_path: Path,
    shape: Tuple[int, ...],
    chunks: Tuple[int, ...],
    dtype: np.dtype,
    n_levels: int = 5,
    overwrite: bool = False,
) -> zarr.hierarchy.Group:
    """
    Create a zarr store with pyramid structure.

    Args:
        output_path: Path to output .zarr directory
        shape: Full resolution shape (C, Y, X)
        chunks: Chunk size for each dimension
        dtype: Data type
        n_levels: Number of pyramid levels
        overwrite: Overwrite existing store

    Returns:
        zarr Group for writing
    """
    if output_path.exists():
        if overwrite:
            import shutil
            shutil.rmtree(output_path)
            logger.info(f"Removed existing zarr store: {output_path}")
        else:
            raise FileExistsError(
                f"Output path exists: {output_path}. Use --overwrite to replace."
            )

    # Create store with Blosc compression
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)

    # Create arrays for each pyramid level
    current_shape = shape
    for level in range(n_levels):
        # Adjust chunks for smaller levels
        level_chunks = tuple(
            min(c, s) for c, s in zip(chunks, current_shape)
        )

        arr = root.create_dataset(
            str(level),
            shape=current_shape,
            chunks=level_chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=0,
        )
        logger.info(
            f"Created level {level}: shape={current_shape}, chunks={level_chunks}"
        )

        # Calculate next level shape (downsample by 2)
        current_shape = (
            current_shape[0],  # Keep channels
            max(1, current_shape[1] // 2),
            max(1, current_shape[2] // 2),
        )

        # Stop if we've reached a small enough size
        if current_shape[1] < 256 and current_shape[2] < 256:
            break

    return root


def write_ome_ngff_metadata(
    root: zarr.hierarchy.Group,
    n_levels: int,
    pixel_size_um: float,
    channel_names: Optional[List[str]] = None,
    n_channels: int = 1,
) -> None:
    """
    Write OME-NGFF metadata to zarr store.

    Args:
        root: zarr group
        n_levels: Number of pyramid levels
        pixel_size_um: Pixel size in micrometers
        channel_names: Optional list of channel names
        n_channels: Number of channels
    """
    # Build datasets list with coordinate transformations
    datasets = []
    for level in range(n_levels):
        if str(level) not in root:
            break
        scale_factor = 2 ** level
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [
                        1.0,  # channel
                        pixel_size_um * scale_factor,  # y
                        pixel_size_um * scale_factor,  # x
                    ]
                }
            ]
        })

    # Build axes
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    # Build multiscales metadata
    multiscales = [{
        "version": "0.4",
        "name": "image",
        "axes": axes,
        "datasets": datasets,
        "type": "gaussian",  # Downsampling method
        "metadata": {
            "method": "skimage.transform.downscale_local_mean",
            "version": "0.4"
        }
    }]

    root.attrs["multiscales"] = multiscales

    # Add OMERO metadata for channel visualization
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(n_channels)]

    # Define some default colors for channels
    colors = ["00FF00", "FF0000", "0000FF", "FFFF00", "FF00FF", "00FFFF"]

    omero = {
        "name": "image",
        "version": "0.4",
        "channels": [
            {
                "active": True,
                "coefficient": 1,
                "color": colors[i % len(colors)],
                "family": "linear",
                "inverted": False,
                "label": channel_names[i] if i < len(channel_names) else f"Ch{i}",
                "window": {
                    "start": 0,
                    "end": 65535,  # Assume 16-bit, Napari adjusts automatically
                    "min": 0,
                    "max": 65535,
                }
            }
            for i in range(n_channels)
        ]
    }
    root.attrs["omero"] = omero

    logger.info(f"Wrote OME-NGFF metadata: {len(datasets)} levels, {n_channels} channels")


def _process_strip(
    czi_path: str,
    ch: int,
    ch_idx: int,
    strip_idx: int,
    strip_height: int,
    bbox: Dict,
    zarr_path: str,
    scene: int = 0,
) -> Tuple[int, int, bool]:
    """Process a single strip (for parallel execution)."""
    from aicspylibczi import CziFile

    total_height = bbox['h']
    total_width = bbox['w']
    x_start = bbox['x']
    y_start = bbox['y']

    y_off = strip_idx * strip_height
    h = min(strip_height, total_height - y_off)

    # Open CZI (each worker needs its own handle)
    czi = CziFile(czi_path)

    # Read strip from CZI
    # Scene selection is implicit: x_start/y_start come from
    # get_mosaic_scene_bounding_box(index=scene), so the region
    # coordinates naturally read from the correct scene.
    strip_data = czi.read_mosaic(
        region=(x_start, y_start + y_off, total_width, h),
        scale_factor=1,
        C=ch,
    )

    # Remove singleton dimensions
    strip_data = np.squeeze(strip_data)

    # Handle various output shapes
    if strip_data.ndim == 2:
        actual_h, actual_w = strip_data.shape
    elif strip_data.ndim == 3:
        if strip_data.shape[0] == 1:
            strip_data = strip_data[0]
        elif strip_data.shape[-1] == 1:
            strip_data = strip_data[..., 0]
        else:
            strip_data = strip_data[..., 0]
        actual_h, actual_w = strip_data.shape
    else:
        actual_h, actual_w = h, total_width

    # Open zarr and write (each worker opens independently)
    root = zarr.open(zarr_path, mode='r+')
    root['0'][ch_idx, y_off:y_off + actual_h, :actual_w] = strip_data

    return ch_idx, strip_idx, True


def copy_tiles_to_zarr(
    czi: CziFile,
    zarr_array: zarr.Array,
    mosaic_info: Dict,
    channels: List[int],
    strip_height: int = 5000,
    num_workers: int = 1,
    czi_path: Optional[str] = None,
    zarr_path: Optional[str] = None,
) -> None:
    """
    Copy tiles from CZI to zarr array, handling edge tiles correctly.

    Uses strip-based reading via read_mosaic() for efficiency.
    Supports parallel processing with num_workers > 1.

    Args:
        czi: CziFile object
        zarr_array: Level 0 zarr array to write to
        mosaic_info: Mosaic info dict
        channels: List of channel indices to copy
        strip_height: Height of strips to read at a time
        num_workers: Number of parallel workers (1 = sequential)
        czi_path: Path to CZI file (required for parallel mode)
        zarr_path: Path to zarr store (required for parallel mode)
    """
    bbox = mosaic_info['bbox']
    total_height = bbox['h']
    total_width = bbox['w']
    x_start = bbox['x']
    y_start = bbox['y']

    n_strips = (total_height + strip_height - 1) // strip_height
    total_work = len(channels) * n_strips

    logger.info(f"Writing {len(channels)} channels, {n_strips} strips of {strip_height}px")

    scene = mosaic_info.get('scene', 0)

    if num_workers > 1 and czi_path and zarr_path:
        # Parallel mode
        logger.info(f"Using {num_workers} parallel workers")

        # Build list of all work items
        work_items = []
        for ch_idx, ch in enumerate(channels):
            for strip_idx in range(n_strips):
                work_items.append((ch, ch_idx, strip_idx))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_strip,
                    czi_path, ch, ch_idx, strip_idx, strip_height, bbox, zarr_path, scene
                ): (ch_idx, strip_idx)
                for ch, ch_idx, strip_idx in work_items
            }

            with tqdm(total=total_work, desc="Writing strips") as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        ch_idx, strip_idx = futures[future]
                        logger.error(f"Strip {ch_idx},{strip_idx} failed: {e}")
                    pbar.update(1)
    else:
        # Sequential mode (original behavior)
        for ch_idx, ch in enumerate(channels):
            logger.info(f"Processing channel {ch} ({ch_idx + 1}/{len(channels)})")

            for strip_idx in tqdm(range(n_strips), desc=f"Channel {ch}"):
                y_off = strip_idx * strip_height
                h = min(strip_height, total_height - y_off)

                # Read strip from CZI
                strip_data = czi.read_mosaic(
                    region=(x_start, y_start + y_off, total_width, h),
                    scale_factor=1,
                    C=ch,
                )

                # Remove singleton dimensions
                strip_data = np.squeeze(strip_data)

                # Handle various output shapes from read_mosaic
                if strip_data.ndim == 2:
                    actual_h, actual_w = strip_data.shape
                elif strip_data.ndim == 3:
                    if strip_data.shape[0] == 1:
                        strip_data = strip_data[0]
                        actual_h, actual_w = strip_data.shape
                    elif strip_data.shape[-1] == 1:
                        strip_data = strip_data[..., 0]
                        actual_h, actual_w = strip_data.shape
                    else:
                        logger.warning(f"Unexpected 3D shape {strip_data.shape}, taking first slice")
                        strip_data = strip_data[..., 0]
                        actual_h, actual_w = strip_data.shape
                else:
                    logger.warning(f"Unexpected strip shape: {strip_data.shape}")
                    actual_h, actual_w = h, total_width

                # Write to zarr
                zarr_array[ch_idx, y_off:y_off + actual_h, :actual_w] = strip_data

    logger.info("Finished writing level 0")


def generate_pyramid_level(
    source_array: zarr.Array,
    target_array: zarr.Array,
    block_size: int = 4096,
) -> None:
    """
    Generate a pyramid level by downsampling the source array by 2x.

    Uses local mean downsampling for better quality.

    Args:
        source_array: Source zarr array (higher resolution)
        target_array: Target zarr array (lower resolution, 2x smaller)
        block_size: Size of blocks to process at a time
    """
    from skimage.transform import downscale_local_mean

    n_channels = source_array.shape[0]
    src_h, src_w = source_array.shape[1], source_array.shape[2]
    tgt_h, tgt_w = target_array.shape[1], target_array.shape[2]

    # Process in blocks to manage memory
    n_blocks_y = (src_h + block_size - 1) // block_size
    n_blocks_x = (src_w + block_size - 1) // block_size

    total_blocks = n_blocks_y * n_blocks_x * n_channels

    with tqdm(total=total_blocks, desc="Downsampling") as pbar:
        for ch in range(n_channels):
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    # Source coordinates
                    src_y0 = by * block_size
                    src_y1 = min(src_y0 + block_size, src_h)
                    src_x0 = bx * block_size
                    src_x1 = min(src_x0 + block_size, src_w)

                    # Ensure even dimensions for downsampling
                    # Pad to even if needed
                    actual_h = src_y1 - src_y0
                    actual_w = src_x1 - src_x0

                    # Read block
                    block = source_array[ch, src_y0:src_y1, src_x0:src_x1]

                    # Pad to even dimensions if necessary
                    pad_h = actual_h % 2
                    pad_w = actual_w % 2
                    if pad_h or pad_w:
                        block = np.pad(
                            block,
                            ((0, pad_h), (0, pad_w)),
                            mode='edge'
                        )

                    # Downsample by 2x using local mean
                    downsampled = downscale_local_mean(block, (2, 2))

                    # Convert back to original dtype
                    downsampled = downsampled.astype(source_array.dtype)

                    # Target coordinates
                    tgt_y0 = src_y0 // 2
                    tgt_x0 = src_x0 // 2
                    tgt_y1 = min(tgt_y0 + downsampled.shape[0], tgt_h)
                    tgt_x1 = min(tgt_x0 + downsampled.shape[1], tgt_w)

                    # Trim downsampled data if needed
                    ds_h = tgt_y1 - tgt_y0
                    ds_w = tgt_x1 - tgt_x0

                    # Write to target
                    target_array[ch, tgt_y0:tgt_y1, tgt_x0:tgt_x1] = downsampled[:ds_h, :ds_w]

                    pbar.update(1)


def convert_czi_to_ome_zarr(
    czi_path: Path,
    output_path: Path,
    channels: Optional[List[int]] = None,
    pyramid_levels: int = 5,
    chunk_size: int = 1024,
    strip_height: int = 5000,
    channel_names: Optional[List[str]] = None,
    overwrite: bool = False,
    dry_run: bool = False,
    num_workers: int = 1,
    scene: int = 0,
) -> Optional[Dict]:
    """
    Convert a CZI mosaic file to OME-Zarr format.

    Args:
        czi_path: Path to input CZI file
        output_path: Path to output .zarr directory
        channels: List of channels to convert (None = all)
        pyramid_levels: Number of pyramid levels to generate
        chunk_size: Chunk size for zarr arrays
        strip_height: Height of strips for reading
        channel_names: Optional channel names for metadata
        overwrite: Overwrite existing output
        dry_run: Only run preflight check, don't convert
        num_workers: Number of parallel workers for strip processing
        scene: CZI scene index (0-based, default 0)

    Returns:
        None on success, or preflight results dict if dry_run=True
    """
    start_time = time.time()

    logger.info(f"Opening CZI: {czi_path}")
    czi = CziFile(str(czi_path))

    # Get mosaic info for the specified scene
    mosaic_info = get_mosaic_info(czi, scene=scene)
    logger.info(f"Mosaic size: {mosaic_info['bbox']['w']} x {mosaic_info['bbox']['h']} pixels")
    logger.info(f"Channels: {mosaic_info['n_channels']}")
    logger.info(f"Tiles: {mosaic_info['n_tiles']} ({mosaic_info['tiles_per_row']} x {mosaic_info['tiles_per_col']})")
    logger.info(f"Nominal tile size: {mosaic_info['nominal_tile_size']}")
    logger.info(f"Pixel type: {mosaic_info['pixel_type']}")

    # Determine channels to convert
    if channels is None:
        channels = list(range(mosaic_info['n_channels']))
    n_channels = len(channels)
    logger.info(f"Converting channels: {channels}")

    # Run preflight check
    preflight_results = preflight_check(
        czi=czi,
        mosaic_info=mosaic_info,
        channels=channels,
        output_path=output_path,
        pyramid_levels=pyramid_levels,
        chunk_size=chunk_size,
    )

    if dry_run:
        logger.info("Dry run complete. No conversion performed.")
        return preflight_results

    if not preflight_results['valid']:
        raise ValueError(
            "Preflight check failed. Fix errors above or use --skip-preflight to bypass."
        )

    # Get pixel size
    pixel_size_um = get_pixel_size_um(czi)
    logger.info(f"Pixel size: {pixel_size_um:.4f} um")

    # Determine dtype
    if 'uint16' in str(mosaic_info['pixel_type']).lower() or 'gray16' in str(mosaic_info['pixel_type']).lower():
        dtype = np.uint16
    elif 'uint8' in str(mosaic_info['pixel_type']).lower() or 'gray8' in str(mosaic_info['pixel_type']).lower():
        dtype = np.uint8
    else:
        dtype = np.uint16  # Default
        logger.warning(f"Unknown pixel type '{mosaic_info['pixel_type']}', defaulting to uint16")

    # Create zarr store
    shape = (n_channels, mosaic_info['bbox']['h'], mosaic_info['bbox']['w'])
    chunks = (1, chunk_size, chunk_size)

    logger.info(f"Creating zarr store: {output_path}")
    logger.info(f"Shape: {shape}, Chunks: {chunks}, Dtype: {dtype}")

    root = create_zarr_store(
        output_path,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        n_levels=pyramid_levels,
        overwrite=overwrite,
    )

    # Align strip_height to chunk_size to prevent race conditions when
    # parallel workers write to the same zarr chunk from adjacent strips.
    # Without alignment, strip boundaries (e.g., every 5000px) can fall
    # inside a chunk (e.g., 1024px), causing concurrent writes to overlap.
    if num_workers > 1 and strip_height % chunk_size != 0:
        aligned_strip_height = (strip_height // chunk_size) * chunk_size
        if aligned_strip_height == 0:
            aligned_strip_height = chunk_size
        logger.info(
            f"Aligned strip_height {strip_height} -> {aligned_strip_height} "
            f"(multiple of chunk_size={chunk_size}) to prevent parallel write conflicts"
        )
        strip_height = aligned_strip_height

    # Copy tiles to level 0
    logger.info("Writing level 0 (full resolution)...")
    copy_tiles_to_zarr(
        czi,
        root['0'],
        mosaic_info,
        channels,
        strip_height=strip_height,
        num_workers=num_workers,
        czi_path=str(czi_path),
        zarr_path=str(output_path),
    )

    # Generate pyramid levels
    actual_levels = len([k for k in root.array_keys()])
    for level in range(1, actual_levels):
        logger.info(f"Generating pyramid level {level}...")
        # Use larger blocks (8x chunk_size) for better I/O efficiency on slow storage
        generate_pyramid_level(
            root[str(level - 1)],
            root[str(level)],
            block_size=chunk_size * 8,
        )

    # Write OME-NGFF metadata
    write_ome_ngff_metadata(
        root,
        n_levels=actual_levels,
        pixel_size_um=pixel_size_um,
        channel_names=channel_names,
        n_channels=n_channels,
    )

    elapsed = time.time() - start_time
    logger.info(f"Conversion complete in {elapsed / 60:.1f} minutes")
    logger.info(f"Output: {output_path}")

    # Print summary
    total_size = sum(
        root[str(level)].nbytes_stored
        for level in range(actual_levels)
        if str(level) in root
    )
    logger.info(f"Total size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CZI mosaic to OME-Zarr with pyramids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all channels
    python czi_to_ome_zarr.py input.czi output.zarr

    # Convert specific channels with names
    python czi_to_ome_zarr.py input.czi output.zarr \\
        --channels 0 1 2 \\
        --channel-names "DAPI" "GFP" "RFP"

    # Fewer pyramid levels, larger chunks
    python czi_to_ome_zarr.py input.czi output.zarr \\
        --pyramid-levels 4 \\
        --chunk-size 2048
        """,
    )

    parser.add_argument(
        "czi_path",
        type=Path,
        help="Input CZI file path",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output .zarr directory path",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Channel indices to convert (default: all)",
    )
    parser.add_argument(
        "--channel-names",
        type=str,
        nargs="+",
        default=None,
        help="Channel names for metadata",
    )
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        default=5,
        help="Number of pyramid levels (default: 5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size for zarr arrays (default: 1024)",
    )
    parser.add_argument(
        "--strip-height",
        type=int,
        default=5000,
        help="Height of strips for reading CZI (default: 5000)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=1,
        help="Number of parallel workers for strip processing (default: 1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run preflight check, don't convert",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=0,
        help="CZI scene index (0-based, default: 0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not args.czi_path.exists():
        logger.error(f"Input file not found: {args.czi_path}")
        sys.exit(1)

    # Run conversion
    result = convert_czi_to_ome_zarr(
        czi_path=args.czi_path,
        output_path=args.output_path,
        channels=args.channels,
        pyramid_levels=args.pyramid_levels,
        chunk_size=args.chunk_size,
        strip_height=args.strip_height,
        channel_names=args.channel_names,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        num_workers=args.workers,
        scene=args.scene,
    )

    if args.dry_run and result:
        if not result['valid']:
            sys.exit(1)


if __name__ == "__main__":
    main()
