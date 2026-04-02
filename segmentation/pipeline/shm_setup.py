"""Shared memory setup, channel loading, tissue filtering, and tile sampling.

Encapsulates the direct-to-SHM channel loading, tile grid generation, tissue
detection/calibration, tile sampling, and shard manifest logic from run_pipeline().
"""

import sys
from pathlib import Path

import numpy as np

from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles
from segmentation.pipeline.preprocessing import apply_slide_preprocessing
from segmentation.pipeline.samples import generate_tile_grid
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def setup_shared_memory(
    args,
    loader,
    mosaic_info,
    pixel_size_um,
    ch_list,
    x_start,
    y_start,
    output_dir,
    run_timestamp,
    cached_resume_info=None,
    czi_meta=None,
):
    """Create shared memory, load CZI channels, filter tissue tiles, and apply sampling.

    Encapsulates the following stages from run_pipeline():
      - Channel assignment logging
      - SharedSlideManager creation + per-channel direct-to-SHM loading
      - Slide-wide preprocessing (flat-field, photobleach) on SHM views
      - Tile grid generation
      - Tissue threshold calibration and tissue tile filtering
      - Tile sampling (sample_fraction) and deterministic sort
      - Output directory setup (timestamped or resume path)
      - Multi-node shard tile list sharing and round-robin assignment

    Args:
        args: Parsed CLI namespace.
        loader: CZI loader (metadata already loaded; channels will be loaded here).
        mosaic_info: Dict with keys x, y, width, height (global pixel coords).
        pixel_size_um: Pixel size in microns.
        ch_list: Sorted list of CZI channel indices to load.
        x_start: Mosaic x origin (pixels).
        y_start: Mosaic y origin (pixels).
        output_dir: Base output directory (Path or str).
        run_timestamp: Timestamp string for naming new run dirs.
        cached_resume_info: Pre-computed result of detect_resume_stage(), or None.
        czi_meta: CZI metadata dict (from get_czi_metadata()), used for channel
            label logging only. May be None — labels will show '?' if absent.

    Returns:
        dict with keys:
            slide_shm_arr   - numpy array backed by shared memory (h, w, n_channels)
            ch_to_slot      - dict mapping CZI channel index -> SHM slot index
            shm_manager     - SharedSlideManager instance (caller must call .cleanup())
            all_channel_data - dict mapping CZI channel index -> SHM view array
            all_tiles       - all tiles in mosaic (pre-tissue-filter)
            tissue_tiles    - tiles passing tissue threshold
            sampled_tiles   - final tile list (after sampling + shard filtering)
            variance_threshold - calibrated or manual tissue variance threshold
            pct             - int(sample_fraction * 100)
            slide_output_dir - Path to this run's output directory
            tiles_dir       - Path to tiles/ subdirectory
    """
    from segmentation.processing.multigpu_shm import SharedSlideManager

    output_dir = Path(output_dir)
    slide_name = Path(getattr(args, "czi_path", "slide")).stem

    # ---- Channel assignment summary — prominent, self-documenting ----
    logger.info("=" * 70)
    logger.info("CHANNEL ASSIGNMENTS")
    logger.info("=" * 70)
    _ch_label_map = {}
    try:
        if czi_meta is not None:
            for _ch in czi_meta["channels"]:
                em = _ch.get("emission_nm")
                fluor = (_ch.get("fluorophore") or _ch.get("name") or "").strip()
                em_str = f" em={em:.0f}nm" if em else ""
                _ch_label_map[_ch["index"]] = f"{fluor}{em_str}"
    except Exception:
        pass

    def _ch_lbl(idx):
        return f"C={idx} ({_ch_label_map.get(idx, '?')})"

    if args.cell_type == "cell" and getattr(args, "cellpose_input_channels", None):
        _parts = str(args.cellpose_input_channels).split(",")
        _cyto = int(_parts[0].strip())
        _nuc = int(_parts[1].strip()) if len(_parts) > 1 else None
        logger.info(
            f"  Segmentation (Cellpose):  cyto={_ch_lbl(_cyto)}"
            + (f"  |  nuc={_ch_lbl(_nuc)}" if _nuc is not None else "")
        )
    elif args.cell_type == "islet":
        _mem = getattr(args, "membrane_channel", 1)
        _nuc = getattr(args, "nuclear_channel", 4)
        logger.info(
            f"  Segmentation (Cellpose islet):  membrane={_ch_lbl(_mem)}  |  nuc={_ch_lbl(_nuc)}"
        )
    else:
        logger.info(f"  Segmentation:  primary={_ch_lbl(args.channel)}")

    logger.info(f"  Loaded channels:  {', '.join(_ch_lbl(c) for c in ch_list)}")
    logger.info("=" * 70)

    # ---- Direct-to-SHM: create shared memory and load channels in one pass ----
    shm_manager = SharedSlideManager()
    h, w = loader.height, loader.width
    n_ch_total = len(ch_list)
    logger.info(f"Creating shared memory for {n_ch_total} channels ({h}x{w})...")
    logger.info(f"  Channel list: {ch_list}")

    slide_shm_arr = shm_manager.create_slide_buffer(slide_name, (h, w, n_ch_total), np.uint16)
    ch_to_slot = {ch: i for i, ch in enumerate(ch_list)}

    # Load each channel directly from CZI into SHM slot (no RAM intermediate)
    for ch in ch_list:
        slot = ch_to_slot[ch]
        shm_view = slide_shm_arr[:, :, slot]
        loader.load_to_shared_memory(ch, shm_view)
        # Register the SHM view as the loader's channel data (for get_tile, tissue detection)
        loader.set_channel_data(ch, shm_view)
    logger.info(f"  All {n_ch_total} channels loaded directly to shared memory")

    # Build all_channel_data as views into SHM (for preprocessing, resume HTML, etc.)
    all_channel_data = {ch: slide_shm_arr[:, :, ch_to_slot[ch]] for ch in ch_list}

    # Apply slide-wide preprocessing on SHM views (modifies data in-place)
    apply_slide_preprocessing(args, all_channel_data, loader)

    # Generate tile grid (using global coordinates)
    overlap = args.tile_overlap
    logger.info(f"Generating tile grid (size={args.tile_size}, overlap={overlap*100:.0f}%)...")
    all_tiles = generate_tile_grid(mosaic_info, args.tile_size, overlap_fraction=overlap)
    logger.info(f"  Total tiles: {len(all_tiles)}")

    # Determine tissue detection channel BEFORE calibration
    # For islet/tissue_pattern: use nuclear channel (universal cell marker)
    tissue_channel = args.channel
    if args.cell_type == "islet":
        tissue_channel = getattr(args, "nuclear_channel", 4)
        logger.info(f"Islet: using DAPI (ch{tissue_channel}) for tissue detection")
    elif args.cell_type == "tissue_pattern":
        tissue_channel = getattr(args, "tp_nuclear_channel", 4)
        logger.info(f"Tissue pattern: using nuclear (ch{tissue_channel}) for tissue detection")

    # Calibrate tissue threshold on the SAME channel used for filtering
    manual_threshold = getattr(args, "variance_threshold", None)
    if manual_threshold is not None:
        variance_threshold = manual_threshold
        logger.info(
            f"Using manual variance threshold: {variance_threshold:.1f} (skipping K-means calibration)"
        )
    else:
        logger.info("Calibrating tissue threshold...")
        variance_threshold = calibrate_tissue_threshold(
            all_tiles,
            calibration_samples=min(50, len(all_tiles)),
            channel=tissue_channel,
            tile_size=args.tile_size,
            loader=loader,  # Loader handles mosaic origin offset correctly
        )
    logger.info("Filtering to tissue-containing tiles...")
    tissue_tiles = filter_tissue_tiles(
        all_tiles,
        variance_threshold,
        channel=tissue_channel,
        tile_size=args.tile_size,
        loader=loader,  # Loader handles mosaic origin offset correctly
    )

    if len(tissue_tiles) == 0:
        logger.error(
            "No tissue-containing tiles found! Check CZI file and tissue detection thresholds."
        )
        sys.exit(1)

    # Sample from tissue tiles
    n_sample = max(1, int(len(tissue_tiles) * args.sample_fraction))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    logger.info(
        f"Sampled {len(sampled_tiles)} tiles ({args.sample_fraction*100:.0f}% of {len(tissue_tiles)} tissue tiles)"
    )

    # Sort sampled tiles deterministically (by position) before sharding
    # so all nodes agree on the same ordering regardless of np.random.choice order
    sampled_tiles.sort(key=lambda t: (t["y"], t["x"]))  # sort by (y, x)

    # Setup output directories (timestamped to avoid overwriting previous runs)
    pct = int(args.sample_fraction * 100)
    if getattr(args, "resume", None):
        slide_output_dir = Path(args.resume)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    elif getattr(args, "resume_from", None):
        slide_output_dir = Path(args.resume_from)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    else:
        slide_output_dir = output_dir / f"{slide_name}_{run_timestamp}_{pct}pct"
    tiles_dir = slide_output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Multi-node tile list sharing: write/read sampled_tiles.json so all shards
    # process the exact same tile list even if tissue calibration diverges slightly.
    # Tiles are dicts with 'x' and 'y' keys.
    if getattr(args, "tile_shard", None):
        tile_list_file = slide_output_dir / "sampled_tiles.json"
        if tile_list_file.exists():
            # Another shard already wrote the tile list -- use it
            sampled_tiles = fast_json_load(tile_list_file)
            logger.info(
                f"Loaded shared tile list from {tile_list_file} ({len(sampled_tiles)} tiles)"
            )
        else:
            # First shard to arrive -- write the tile list atomically
            try:
                atomic_json_dump(sampled_tiles, tile_list_file)
                logger.info(
                    f"Wrote shared tile list to {tile_list_file} ({len(sampled_tiles)} tiles)"
                )
            except OSError:
                # Another shard beat us in a race -- read theirs
                if tile_list_file.exists():
                    sampled_tiles = fast_json_load(tile_list_file)
                    logger.info(
                        f"Race: loaded shared tile list from {tile_list_file} ({len(sampled_tiles)} tiles)"
                    )

        # Round-robin shard assignment
        # NOTE: Per-tile resume not implemented for shard mode.
        # If a shard crashes, it re-processes all tiles in its shard.
        # For large slides, consider using smaller shard counts.
        shard_idx, shard_total = args.tile_shard
        total_before = len(sampled_tiles)
        sampled_tiles = [t for i, t in enumerate(sampled_tiles) if i % shard_total == shard_idx]
        logger.info(
            f"Tile shard {shard_idx}/{shard_total}: processing {len(sampled_tiles)}/{total_before} tiles"
        )

        # Write shard manifest for auditability
        manifest = {
            "shard_idx": shard_idx,
            "shard_total": shard_total,
            "tiles": sampled_tiles,
            "total_sampled": total_before,
            "random_seed": getattr(args, "random_seed", 42),
        }
        manifest_file = slide_output_dir / f"shard_{shard_idx}_manifest.json"
        atomic_json_dump(manifest, manifest_file)

    return {
        "slide_shm_arr": slide_shm_arr,
        "ch_to_slot": ch_to_slot,
        "shm_manager": shm_manager,
        "all_channel_data": all_channel_data,
        "all_tiles": all_tiles,
        "tissue_tiles": tissue_tiles,
        "sampled_tiles": sampled_tiles,
        "variance_threshold": variance_threshold,
        "pct": pct,
        "slide_output_dir": slide_output_dir,
        "tiles_dir": tiles_dir,
        "tissue_channel": tissue_channel,
        "h": h,
        "w": w,
    }
