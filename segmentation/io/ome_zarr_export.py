"""Export SHM slide data to OME-Zarr format with multi-resolution pyramids.

Provides a fast path for OME-Zarr generation at the end of the pipeline,
reading directly from the shared memory array (already preprocessed with
flat-field + photobleach correction). This avoids re-reading the CZI file.

For standalone CZI-to-Zarr conversion (without SHM), use scripts/czi_to_ome_zarr.py.

Usage (from pipeline):
    from segmentation.io.ome_zarr_export import export_shm_to_ome_zarr

    export_shm_to_ome_zarr(
        shm_array=slide_shm_arr,      # (H, W, C) uint16
        ch_to_slot={0: 0, 1: 1, ...},
        pixel_size_um=0.22,
        czi_path='/path/to/slide.czi',
        output_path='/path/to/output.ome.zarr',
    )
"""

import time
from pathlib import Path

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def _generate_pyramid_level_from_array(
    source_array,
    target_array,
    block_size: int = 4096,
) -> None:
    """Generate a pyramid level by 2x downsampling using local mean.

    Args:
        source_array: Source zarr array (C, Y, X)
        target_array: Target zarr array (C, Y//2, X//2)
        block_size: Processing block size in pixels
    """
    from skimage.transform import downscale_local_mean

    n_channels = source_array.shape[0]
    src_h, src_w = source_array.shape[1], source_array.shape[2]
    tgt_h, tgt_w = target_array.shape[1], target_array.shape[2]

    n_blocks_y = (src_h + block_size - 1) // block_size
    n_blocks_x = (src_w + block_size - 1) // block_size

    for ch in range(n_channels):
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                src_y0 = by * block_size
                src_y1 = min(src_y0 + block_size, src_h)
                src_x0 = bx * block_size
                src_x1 = min(src_x0 + block_size, src_w)

                block = source_array[ch, src_y0:src_y1, src_x0:src_x1]

                # Pad to even dimensions for 2x downsampling
                actual_h = src_y1 - src_y0
                actual_w = src_x1 - src_x0
                pad_h = actual_h % 2
                pad_w = actual_w % 2
                if pad_h or pad_w:
                    block = np.pad(block, ((0, pad_h), (0, pad_w)), mode="edge")

                downsampled = downscale_local_mean(block, (2, 2)).astype(source_array.dtype)

                tgt_y0 = src_y0 // 2
                tgt_x0 = src_x0 // 2
                tgt_y1 = min(tgt_y0 + downsampled.shape[0], tgt_h)
                tgt_x1 = min(tgt_x0 + downsampled.shape[1], tgt_w)

                target_array[ch, tgt_y0:tgt_y1, tgt_x0:tgt_x1] = downsampled[
                    : tgt_y1 - tgt_y0, : tgt_x1 - tgt_x0
                ]


def export_shm_to_ome_zarr(
    shm_array: np.ndarray,
    ch_to_slot: dict[int, int],
    pixel_size_um: float,
    output_path,
    czi_path=None,
    channel_names: list[str] | None = None,
    pyramid_levels: int = 5,
    chunk_size: int = 1024,
    overwrite: bool = False,
) -> Path | None:
    """Export SHM slide array to OME-Zarr with pyramids.

    Reads directly from the shared memory array (H, W, C) and writes
    to OME-Zarr format (C, Y, X) with multi-resolution pyramids.

    Args:
        shm_array: Shared memory array, shape (H, W, C), typically uint16.
        ch_to_slot: Mapping of CZI channel index -> SHM slot index.
            E.g. {0: 0, 1: 1, 2: 2} for 3 channels.
        pixel_size_um: Pixel size in micrometers.
        output_path: Path to output .ome.zarr directory.
        czi_path: Optional CZI path for reading channel names from metadata.
        channel_names: Optional list of channel names. If None and czi_path
            is provided, names are read from CZI metadata.
        pyramid_levels: Number of pyramid levels (default: 5).
        chunk_size: Zarr chunk size (default: 1024).
        overwrite: If True, overwrite existing zarr store.

    Returns:
        Path to the created zarr store, or None on failure.
    """
    import numcodecs
    import zarr

    _zarr_v3 = int(zarr.__version__.split(".")[0]) >= 3

    output_path = Path(output_path)
    start_time = time.time()

    if output_path.exists() and not overwrite:
        logger.info(f"OME-Zarr already exists, skipping: {output_path}")
        return output_path

    # Validate input
    if shm_array.ndim != 3:
        logger.error(f"Expected 3D array (H, W, C), got {shm_array.ndim}D")
        return None

    # Get dimensions
    height, width, n_shm_slots = shm_array.shape
    n_channels = len(ch_to_slot)

    for czi_ch, slot_idx in ch_to_slot.items():
        if slot_idx >= n_shm_slots:
            logger.error(f"Slot index {slot_idx} exceeds array channels {n_shm_slots}")
            return None

    # Sort channels by CZI index for consistent ordering
    sorted_channels = sorted(ch_to_slot.items(), key=lambda x: x[0])

    logger.info(f"Exporting SHM to OME-Zarr: {output_path}")
    logger.info(f"  Shape: ({n_channels}, {height}, {width}), dtype={shm_array.dtype}")
    logger.info(f"  Channels: {[f'C{czi_ch}->slot{slot}' for czi_ch, slot in sorted_channels]}")

    # Resolve channel names
    if channel_names is None and czi_path is not None:
        try:
            from segmentation.io.czi_loader import get_czi_metadata

            meta = get_czi_metadata(str(czi_path))
            channel_names = []
            for czi_ch, _ in sorted_channels:
                if czi_ch < len(meta["channels"]):
                    ch_meta = meta["channels"][czi_ch]
                    name = ch_meta.get("fluorophore", "") or ch_meta.get("name", "")
                    em = ch_meta.get("emission_nm")
                    if em:
                        name = f"{name.strip()} ({em:.0f}nm)"
                    channel_names.append(name.strip() or f"Channel {czi_ch}")
                else:
                    channel_names.append(f"Channel {czi_ch}")
        except Exception as e:
            logger.warning(f"Could not read channel names from CZI: {e}")

    if channel_names is None:
        channel_names = [f"Channel {czi_ch}" for czi_ch, _ in sorted_channels]

    # Create zarr v2-format store (compatible with ome-zarr, napari, spatialdata)
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    if _zarr_v3:
        root = zarr.open_group(str(output_path), mode="w", zarr_format=2)
    else:
        store = zarr.DirectoryStore(str(output_path))
        root = zarr.group(store, overwrite=True)

    def _create_arr(name, **kwargs):
        if _zarr_v3:
            return root.create_array(name, compressors=[kwargs.pop("compressor")], **kwargs)
        else:
            return root.create_dataset(name, **kwargs)

    # Create pyramid arrays
    shape = (n_channels, height, width)
    current_shape = shape
    actual_levels = 0

    for level in range(pyramid_levels):
        level_chunks = tuple(min(c, s) for c, s in zip((1, chunk_size, chunk_size), current_shape))

        _create_arr(
            str(level),
            shape=current_shape,
            chunks=level_chunks,
            dtype=shm_array.dtype,
            compressor=compressor,
            fill_value=0,
        )
        actual_levels += 1

        # Use ceil division so pyramid targets match downscale_local_mean output
        current_shape = (
            current_shape[0],
            max(1, (current_shape[1] + 1) // 2),
            max(1, (current_shape[2] + 1) // 2),
        )
        if current_shape[1] < 256 and current_shape[2] < 256:
            break

    # Write level 0: copy SHM (H, W, C) -> zarr (C, Y, X)
    logger.info("  Writing level 0 (full resolution)...")
    level0 = root["0"]
    for out_idx, (czi_ch, slot_idx) in enumerate(sorted_channels):
        # Write in strips to manage memory and show progress
        strip_h = 5000
        n_strips = (height + strip_h - 1) // strip_h
        for s in range(n_strips):
            y0 = s * strip_h
            y1 = min(y0 + strip_h, height)
            level0[out_idx, y0:y1, :] = shm_array[y0:y1, :, slot_idx]

    # Generate pyramid levels
    for level in range(1, actual_levels):
        logger.info(f"  Generating pyramid level {level}...")
        _generate_pyramid_level_from_array(
            root[str(level - 1)],
            root[str(level)],
            block_size=chunk_size * 8,
        )

    # Write OME-NGFF metadata
    datasets = []
    for level in range(actual_levels):
        scale_factor = 2**level
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, pixel_size_um * scale_factor, pixel_size_um * scale_factor],
                    }
                ],
            }
        )

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "image",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
            "type": "local_mean",
            "metadata": {
                "method": "skimage.transform.downscale_local_mean",
                "version": "0.4",
            },
        }
    ]

    # OMERO channel visualization metadata
    colors = ["00FF00", "FF0000", "0000FF", "FFFF00", "FF00FF", "00FFFF"]
    max_val = 65535 if shm_array.dtype == np.uint16 else 255

    root.attrs["omero"] = {
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
                "window": {"start": 0, "end": max_val, "min": 0, "max": max_val},
            }
            for i in range(n_channels)
        ],
    }

    elapsed = time.time() - start_time
    logger.info(f"  OME-Zarr export complete in {elapsed:.1f}s: {output_path}")

    return output_path
