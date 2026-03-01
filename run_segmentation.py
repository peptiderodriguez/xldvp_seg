#!/usr/bin/env python3
"""
Unified Cell Segmentation Pipeline

A general-purpose pipeline for detecting and classifying cells in CZI microscopy images.
Supports multiple cell types with shared infrastructure and full feature extraction.

Cell Types:
    - nmj: Neuromuscular junctions (intensity + elongation filter)
    - mk: Megakaryocytes (SAM2 automatic mask generation)
    - cell: Hematopoietic stem/progenitor cells (Cellpose-SAM + SAM2 refinement)
    - vessel: Blood vessel cross-sections (ring detection via contour hierarchy)
    - islet: Pancreatic islet cells (Cellpose membrane+nuclear + SAM2)

Features extracted per cell: up to 6478 total (with --extract-deep-features and --all-channels)
    - ~78 morphological features (22 base + NMJ-specific + multi-channel stats)
    - 256 SAM2 embedding features (always)
    - 4096 ResNet-50 features (2x2048 masked+context, opt-in via --extract-deep-features)
    - 2048 DINOv2-L features (2x1024 masked+context, opt-in via --extract-deep-features)
    + Cell-type specific features (elongation for NMJ, wall thickness for vessel, etc.)

Outputs:
    - {cell_type}_detections.json: All detections with universal IDs and global coordinates
    - {cell_type}_coordinates.csv: Quick export with center coordinates in pixels and um
    - {cell_type}_masks.h5: Per-tile mask arrays
    - html/: Interactive HTML viewer for annotation

After processing, automatically starts HTTP server and Cloudflare tunnel for remote viewing.
Server runs in background by default - script exits but server keeps running.

Server options:
    --serve-background  Start server in background (default) - script exits, server persists
    --serve             Start server in foreground - wait for Ctrl+C to stop
    --no-serve          Don't start server
    --port PORT         HTTP server port (default: 8081)
    --stop-server       Stop any running background server and exit
    --server-status     Show status of running server (including public URL)

Usage:
    # NMJ detection
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --channel 1

    # Vessel detection (SMA staining)
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --min-vessel-diameter 10 --max-vessel-diameter 500

    # Vessel with CD31 validation
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --cd31-channel 1
"""

import os
import gc
import json
import numpy as np
from pathlib import Path
import h5py
import torch

# Segmentation modules
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
)
from segmentation.io.html_export import (
    percentile_normalize,
    image_to_base64,
    create_hdf5_dataset,
)
from segmentation.utils.logging import get_logger, setup_logging
from segmentation.io.czi_loader import get_loader, get_czi_metadata, print_czi_metadata
from segmentation.utils.islet_utils import classify_islet_marker, compute_islet_marker_thresholds
from segmentation.utils.json_utils import NumpyEncoder
from segmentation.detection.cell_detector import CellDetector
# tile_processing functions now called from within pipeline modules

# Pipeline modules (extracted from this file)
from segmentation.pipeline.server import stop_background_server, show_server_status
from segmentation.pipeline.resume import (
    detect_resume_stage, reload_detections_from_tiles, _resume_generate_html_samples,
)
from segmentation.pipeline.samples import (
    _compute_tile_percentiles, calibrate_islet_marker_gmm,
    filter_and_create_html_samples, generate_tile_grid,
)
from segmentation.pipeline.finalize import _finish_pipeline
from segmentation.pipeline.detection_setup import (
    create_strategy_for_cell_type, apply_vessel_classifiers,
    load_classifier_into_detector, build_detection_params, load_vessel_classifiers,
)
from segmentation.pipeline.preprocessing import apply_slide_preprocessing
from segmentation.pipeline.cli import build_parser, postprocess_args

logger = get_logger(__name__)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args):
    """Main pipeline execution."""
    # Setup logging
    setup_logging(level="DEBUG" if getattr(args, 'verbose', False) else "INFO")

    from datetime import datetime
    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    slide_name = czi_path.stem
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set random seed early -- ensures all multi-node shards get identical
    # tissue calibration and tile sampling (same tile list on every node)
    np.random.seed(getattr(args, 'random_seed', 42))

    logger.info("=" * 60)
    logger.info("UNIFIED SEGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Slide: {slide_name}")
    logger.info(f"Run: {run_timestamp}")
    logger.info(f"Cell type: {args.cell_type}")
    logger.info(f"Channel: {args.channel}")
    if args.scene != 0:
        logger.info(f"Scene: {args.scene}")
    if getattr(args, 'multi_marker', False):
        logger.info("Multi-marker mode: ENABLED (auto-enabled --all-channels and --parallel-detection)")
    if getattr(args, 'tile_shard', None):
        shard_idx, shard_total = args.tile_shard
        logger.info(f"Tile shard: {shard_idx}/{shard_total} (detection-only)")
    elif getattr(args, 'detection_only', False):
        logger.info("Detection-only mode (skipping dedup/HTML/CSV)")
    logger.info(f"Random seed: {getattr(args, 'random_seed', 42)}")
    logger.info("=" * 60)

    # ---- Resume early-exit: skip CZI loading if ALL stages are done ----
    if getattr(args, 'resume', None):
        resume_dir = Path(args.resume)
        resume_info = detect_resume_stage(resume_dir, args.cell_type)
        _has_det = resume_info['has_detections'] and not getattr(args, 'force_detect', False)
        _has_html = resume_info['has_html'] and not getattr(args, 'force_html', False)
        _has_tiles = resume_info['has_tiles'] and not getattr(args, 'force_detect', False)

        logger.info(f"Resume state: tiles={resume_info['tile_count']}, "
                     f"detections={'yes' if resume_info['has_detections'] else 'no'}"
                     f" ({resume_info['detection_count']}), "
                     f"html={'yes' if resume_info['has_html'] else 'no'}")

        if _has_det and _has_html:
            # Everything done -- just regenerate CSV/summary without loading CZI
            logger.info("All stages complete -- regenerating CSV and summary only (no CZI load needed)")
            slide_output_dir = resume_dir
            det_file = slide_output_dir / f"{args.cell_type}_detections.json"
            with open(det_file) as f:
                all_detections = json.load(f)

            # Load pipeline config for metadata
            config_file = slide_output_dir / 'pipeline_config.json'
            if config_file.exists():
                with open(config_file) as f:
                    cfg = json.load(f)
                pixel_size_um = cfg.get('pixel_size_um')
                if pixel_size_um is None:
                    raise ValueError(
                        f"pipeline_config.json is missing 'pixel_size_um'. "
                        f"Re-run detection or add pixel_size_um to {config_file}"
                    )
                mosaic_info = {'width': cfg.get('width', 0), 'height': cfg.get('height', 0),
                               'x': cfg.get('x_start', 0), 'y': cfg.get('y_start', 0)}
            else:
                # Minimal load: just metadata from CZI
                loader = get_loader(czi_path, load_to_ram=False, channel=args.channel, scene=args.scene)
                pixel_size_um = loader.get_pixel_size()
                mosaic_info = {'width': loader.mosaic_size[0], 'height': loader.mosaic_size[1],
                               'x': loader.mosaic_origin[0], 'y': loader.mosaic_origin[1]}

            pct = int(args.sample_fraction * 100)
            _finish_pipeline(
                args, all_detections, [], slide_output_dir, slide_output_dir / "tiles",
                pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
                skip_html=True, resumed=True,
            )
            return

    # RAM-first architecture: Load CZI channel into RAM ONCE at pipeline start
    # This eliminates repeated network I/O for files on network mounts
    # Default is RAM loading for single slides (best performance on network mounts)
    if not args.load_to_ram:
        logger.warning("--no-ram is deprecated and ignored. All data is loaded to RAM.")
        args.load_to_ram = True
    use_ram = True

    logger.info("Loading CZI file with get_loader() (RAM-first architecture)...")
    loader = get_loader(
        czi_path,
        load_to_ram=use_ram,
        channel=args.channel,
        quiet=False,
        scene=args.scene
    )

    # Get mosaic bounds from loader properties
    x_start, y_start = loader.mosaic_origin
    width, height = loader.mosaic_size

    # Build mosaic_info dict for compatibility with existing functions
    mosaic_info = {
        'x': x_start,
        'y': y_start,
        'width': width,
        'height': height,
    }
    pixel_size_um = loader.get_pixel_size()

    logger.info(f"  Mosaic: {mosaic_info['width']} x {mosaic_info['height']} px")
    logger.info(f"  Origin: ({mosaic_info['x']}, {mosaic_info['y']})")
    logger.info(f"  Pixel size: {pixel_size_um:.4f} um/px")
    if use_ram:
        logger.info(f"  Channel {args.channel} loaded to RAM")

    # Always log channel metadata so the log is self-documenting
    try:
        _czi_meta = get_czi_metadata(czi_path, scene=args.scene)
        logger.info(f"  CZI channels ({_czi_meta['n_channels']}):")
        for _ch in _czi_meta['channels']:
            _ex = f"{_ch['excitation_nm']:.0f}" if _ch['excitation_nm'] else "?"
            _em = f"{_ch['emission_nm']:.0f}" if _ch['emission_nm'] else "?"
            _label = _ch['fluorophore'] if _ch['fluorophore'] != 'N/A' else _ch['name']
            logger.info(f"    [{_ch['index']}] {_ch['name']:<20s}  Ex {_ex} -> Em {_em} nm  ({_label})")
    except Exception as _e:
        logger.warning(f"  Could not read channel metadata: {_e}")

    # Load additional channels if --all-channels specified (for NMJ specificity checking)
    all_channel_data = {args.channel: loader.channel_data}  # Primary channel
    if getattr(args, 'all_channels', False) and use_ram:
        # Determine which channels to load
        if getattr(args, 'channels', None):
            ch_list = [int(x.strip()) for x in args.channels.split(',')]
            logger.info(f"Loading specified channels {ch_list} for multi-channel analysis...")
        else:
            # Load all channels from CZI
            try:
                dims = loader.reader.get_dims_shape()[0]
                n_channels = dims.get('C', (0, 3))[1]  # Default to 3 channels
            except Exception:
                n_channels = 3  # Fallback
            ch_list = list(range(n_channels))
            logger.info(f"Loading all {len(ch_list)} channels for multi-channel analysis...")

        for ch in ch_list:
            if ch != args.channel:
                logger.info(f"  Loading channel {ch}...")
                ch_loader = get_loader(czi_path, load_to_ram=True, channel=ch, quiet=True, scene=args.scene)
                all_channel_data[ch] = ch_loader.get_channel_data(ch)
        logger.info(f"  Loaded channels: {sorted(all_channel_data.keys())}")

    # Also load CD31 channel to RAM if specified (for vessel validation)
    # Note: get_loader() with the same path returns the cached loader and just adds the new channel
    if args.cell_type == 'vessel' and args.cd31_channel is not None:
        logger.info(f"Loading CD31 channel {args.cd31_channel} to RAM...")
        # Use get_loader to add channel to the same cached loader instance
        loader = get_loader(
            czi_path,
            load_to_ram=use_ram,
            channel=args.cd31_channel,
            quiet=False,
            scene=args.scene
        )

    # Apply slide-wide preprocessing (photobleach, flat-field, Reinhard)
    apply_slide_preprocessing(args, all_channel_data, loader)

    # Generate tile grid (using global coordinates)
    overlap = getattr(args, 'tile_overlap', 0.0)
    logger.info(f"Generating tile grid (size={args.tile_size}, overlap={overlap*100:.0f}%)...")
    all_tiles = generate_tile_grid(mosaic_info, args.tile_size, overlap_fraction=overlap)
    logger.info(f"  Total tiles: {len(all_tiles)}")

    # Determine tissue detection channel BEFORE calibration
    # For islet/tissue_pattern: use nuclear channel (universal cell marker)
    tissue_channel = args.channel
    if args.cell_type == 'islet':
        tissue_channel = getattr(args, 'nuclear_channel', 4)
        logger.info(f"Islet: using DAPI (ch{tissue_channel}) for tissue detection")
    elif args.cell_type == 'tissue_pattern':
        tissue_channel = getattr(args, 'tp_nuclear_channel', 4)
        logger.info(f"Tissue pattern: using nuclear (ch{tissue_channel}) for tissue detection")

    # Calibrate tissue threshold on the SAME channel used for filtering
    manual_threshold = getattr(args, 'variance_threshold', None)
    if manual_threshold is not None:
        variance_threshold = manual_threshold
        logger.info(f"Using manual variance threshold: {variance_threshold:.1f} (skipping K-means calibration)")
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
        logger.error("No tissue-containing tiles found!")
        return

    # Sample from tissue tiles
    n_sample = max(1, int(len(tissue_tiles) * args.sample_fraction))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    logger.info(f"Sampled {len(sampled_tiles)} tiles ({args.sample_fraction*100:.0f}% of {len(tissue_tiles)} tissue tiles)")

    # Sort sampled tiles deterministically (by position) before sharding
    # so all nodes agree on the same ordering regardless of np.random.choice order
    sampled_tiles.sort(key=lambda t: (t['y'], t['x']))  # sort by (y, x)

    # Setup output directories (timestamped to avoid overwriting previous runs)
    pct = int(args.sample_fraction * 100)
    if getattr(args, 'resume', None):
        slide_output_dir = Path(args.resume)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    elif getattr(args, 'resume_from', None):
        slide_output_dir = Path(args.resume_from)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    else:
        slide_output_dir = output_dir / f'{slide_name}_{run_timestamp}_{pct}pct'
    tiles_dir = slide_output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Multi-node tile list sharing: write/read sampled_tiles.json so all shards
    # process the exact same tile list even if tissue calibration diverges slightly.
    # Tiles are dicts with 'x' and 'y' keys.
    if getattr(args, 'tile_shard', None):
        tile_list_file = slide_output_dir / 'sampled_tiles.json'
        if tile_list_file.exists():
            # Another shard already wrote the tile list -- use it
            with open(tile_list_file) as f:
                sampled_tiles = json.load(f)
            logger.info(f"Loaded shared tile list from {tile_list_file} ({len(sampled_tiles)} tiles)")
        else:
            # First shard to arrive -- write the tile list (atomic via rename)
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(dir=slide_output_dir, suffix='.json')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(sampled_tiles, f)
                os.rename(tmp_path, tile_list_file)
                logger.info(f"Wrote shared tile list to {tile_list_file} ({len(sampled_tiles)} tiles)")
            except OSError:
                # Another shard beat us in a race -- read theirs
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                if tile_list_file.exists():
                    with open(tile_list_file) as f:
                        sampled_tiles = json.load(f)
                    logger.info(f"Race: loaded shared tile list from {tile_list_file} ({len(sampled_tiles)} tiles)")

        # Round-robin shard assignment
        # NOTE: Per-tile resume not implemented for shard mode.
        # If a shard crashes, it re-processes all tiles in its shard.
        # For large slides, consider using smaller shard counts.
        shard_idx, shard_total = args.tile_shard
        total_before = len(sampled_tiles)
        sampled_tiles = [t for i, t in enumerate(sampled_tiles) if i % shard_total == shard_idx]
        logger.info(f"Tile shard {shard_idx}/{shard_total}: processing {len(sampled_tiles)}/{total_before} tiles")

        # Write shard manifest for auditability
        manifest = {
            'shard_idx': shard_idx, 'shard_total': shard_total,
            'tiles': sampled_tiles,
            'total_sampled': total_before,
            'random_seed': getattr(args, 'random_seed', 42),
        }
        manifest_file = slide_output_dir / f'shard_{shard_idx}_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)

    # ---- Resume: detect completed stages and set skip flags ----
    skip_detection = False
    skip_dedup = False
    skip_html = False

    if getattr(args, 'resume', None) and not getattr(args, 'detection_only', False):
        resume_info = detect_resume_stage(slide_output_dir, args.cell_type)
        logger.info(f"Resume state: tiles={resume_info['tile_count']}, "
                     f"detections={'yes' if resume_info['has_detections'] else 'no'}"
                     f" ({resume_info['detection_count']}), "
                     f"html={'yes' if resume_info['has_html'] else 'no'}")

        # --merge-shards: always reload from tiles (shard detections are pre-dedup),
        # always run dedup, always generate HTML. Checkpointed at each stage.
        if getattr(args, 'merge_shards', False):
            # Validate shard completeness
            shard_manifests = list(slide_output_dir.glob('shard_*_manifest.json'))
            if shard_manifests:
                with open(shard_manifests[0]) as f:
                    m0 = json.load(f)
                expected_shards = m0.get('shard_total', len(shard_manifests))
                if len(shard_manifests) < expected_shards:
                    logger.warning(f"Only {len(shard_manifests)}/{expected_shards} shard manifests found -- "
                                   f"some shards may not have completed. Merge will use available data only.")

            merged_det_file = slide_output_dir / f'{args.cell_type}_detections_merged.json'
            deduped_det_file = slide_output_dir / f'{args.cell_type}_detections.json'

            # Checkpoint 1: merged detections (all shards concatenated)
            if merged_det_file.exists() and not getattr(args, 'force_detect', False):
                with open(merged_det_file) as f:
                    all_detections = json.load(f)
                logger.info(f"Checkpoint: loaded {len(all_detections)} merged detections from {merged_det_file.name}")
            elif resume_info['has_tiles']:
                all_detections = reload_detections_from_tiles(tiles_dir, args.cell_type)
                logger.info(f"Merged {len(all_detections)} detections from {resume_info['tile_count']} tile dirs")
                # Save checkpoint
                with open(merged_det_file, 'w') as f:
                    json.dump(all_detections, f, cls=NumpyEncoder)
                logger.info(f"Checkpoint saved: {merged_det_file.name}")
            else:
                logger.error("No tile dirs found -- nothing to merge")
                return

            # Checkpoint 2: deduped detections
            if deduped_det_file.exists() and not getattr(args, 'force_detect', False):
                with open(deduped_det_file) as f:
                    all_detections = json.load(f)
                logger.info(f"Checkpoint: loaded {len(all_detections)} deduped detections from {deduped_det_file.name}")
                skip_dedup = True
            skip_detection = True
            skip_html = False  # Always regenerate HTML for merge
            args.force_html = True
        elif resume_info['has_detections'] and not getattr(args, 'force_detect', False):
            skip_detection = True
            skip_dedup = True
            det_file = slide_output_dir / f"{args.cell_type}_detections.json"
            with open(det_file) as f:
                all_detections = json.load(f)
            logger.info(f"Loaded {len(all_detections)} detections from {det_file} (skipping detection + dedup)")
        elif resume_info['has_tiles'] and not getattr(args, 'force_detect', False):
            skip_detection = True
            all_detections = reload_detections_from_tiles(tiles_dir, args.cell_type)
            logger.info(f"Reloaded {len(all_detections)} detections from {resume_info['tile_count']} tile dirs (skipping detection, will dedup)")

        if resume_info['has_html'] and not getattr(args, 'force_html', False):
            skip_html = True
            logger.info("HTML exists -- skipping HTML generation (use --force-html to regenerate)")

    # Save pipeline config for resume/regeneration
    pipeline_config = {
        'czi_path': str(czi_path),
        'cell_type': args.cell_type,
        'tile_size': args.tile_size,
        'pixel_size_um': pixel_size_um,
        'scene': args.scene,
        'width': width,
        'height': height,
        'x_start': x_start,
        'y_start': y_start,
        'sample_fraction': args.sample_fraction,
        'tile_overlap': getattr(args, 'tile_overlap', 0.0),
        'channel': args.channel,
    }
    # Add display channel config
    if args.cell_type == 'islet':
        pipeline_config['display_channels'] = getattr(args, 'islet_display_chs', [2, 3, 5])
        pipeline_config['marker_channels'] = getattr(args, 'islet_marker_channels', 'gcg:2,ins:3,sst:5')
    elif args.cell_type == 'tissue_pattern':
        pipeline_config['display_channels'] = getattr(args, 'tp_display_channels_list', [0, 3, 1])
    config_file = slide_output_dir / 'pipeline_config.json'
    if not config_file.exists() or not getattr(args, 'resume', None):
        with open(config_file, 'w') as f:
            json.dump(pipeline_config, f)

    # ---- Resume fast-path: skip detection, go straight to dedup/HTML/CSV ----
    if skip_detection:
        is_multiscale = args.cell_type == 'vessel' and getattr(args, 'multi_scale', False)

        # Run dedup if reloaded from tiles (not from deduped detections JSON)
        if not skip_dedup and getattr(args, 'tile_overlap', 0) > 0 and len(all_detections) > 0 and not is_multiscale:
            from segmentation.processing.deduplication import deduplicate_by_mask_overlap
            pre_dedup = len(all_detections)
            mask_fn = f'{args.cell_type}_masks.h5'
            dedup_sort = 'confidence' if getattr(args, 'dedup_by_confidence', False) else 'area'
            all_detections = deduplicate_by_mask_overlap(
                all_detections, tiles_dir, min_overlap_fraction=0.1,
                mask_filename=mask_fn, sort_by=dedup_sort,
            )
            logger.info(f"Dedup (resume): {pre_dedup} -> {len(all_detections)}")

        # Regenerate HTML if needed (requires CZI + tile masks)
        all_samples = []
        if not skip_html:
            # Ensure all channels are loaded for HTML generation
            if args.all_channels or (args.cell_type == 'cell' and getattr(args, 'cellpose_input_channels', None)):
                try:
                    dims = loader.reader.get_dims_shape()[0]
                    _n_ch = dims.get('C', (0, 3))[1]
                except Exception:
                    _n_ch = 3
                for ch in range(_n_ch):
                    if ch not in all_channel_data:
                        logger.info(f"  Loading channel {ch} for HTML generation (resume)...")
                        loader.load_channel(ch)
                        all_channel_data[ch] = loader.get_channel_data(ch)

            logger.info(f"Regenerating HTML for {len(all_detections)} detections from saved tiles...")
            all_samples = _resume_generate_html_samples(
                args, all_detections, tiles_dir,
                all_channel_data, loader, pixel_size_um, slide_name,
                x_start, y_start,
            )
            logger.info(f"Generated {len(all_samples)} HTML samples from resume path")

        # Run the same post-processing as the normal path (HTML export, CSV, summary)
        _finish_pipeline(
            args, all_detections, all_samples, slide_output_dir, tiles_dir,
            pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
            skip_html=skip_html, resumed=True,
            all_tiles=all_tiles, tissue_tiles=tissue_tiles, sampled_tiles=sampled_tiles,
        )
        return

    # ---- Normal path: full detection pipeline ----
    # Initialize detector + classifiers + params
    logger.info("Initializing detector...")
    detector = CellDetector(device="cuda")

    classifier_loaded = load_classifier_into_detector(args, detector)

    # Auto-detect annotation run: no classifier -> show ALL candidates in HTML
    if args.cell_type in ('nmj', 'islet', 'tissue_pattern') and not classifier_loaded and args.html_score_threshold > 0:
        logger.info(f"No classifier loaded -- annotation run detected. "
                     f"Overriding --html-score-threshold from {args.html_score_threshold} to 0.0 "
                     f"(will show ALL candidates for annotation)")
        args.html_score_threshold = 0.0

    params = build_detection_params(args, pixel_size_um)
    logger.info(f"Detection params: {params}")

    strategy = create_strategy_for_cell_type(args.cell_type, params, pixel_size_um)
    logger.info(f"Using {strategy.name} strategy: {strategy.get_config()}")

    vessel_classifier, vessel_type_classifier = load_vessel_classifiers(args)

    # Process tiles
    logger.info("Processing tiles...")
    all_samples = []
    all_detections = []  # Universal list with global coordinates
    deferred_html_tiles = []  # For islet: defer HTML until marker thresholds computed
    is_multiscale = args.cell_type == 'vessel' and getattr(args, 'multi_scale', False)

    # ---- Shared memory creation (used by BOTH regular and multiscale paths) ----
    num_gpus = getattr(args, 'num_gpus', 1)

    if len(all_channel_data) < 2:
        logger.warning("Pipeline works best with --all-channels for multi-channel features")

    from segmentation.processing.multigpu_shm import SharedSlideManager
    from segmentation.processing.multigpu_worker import MultiGPUTileProcessor

    shm_manager = SharedSlideManager()

    try:
        # Load ALL channels to shared memory (RGB display uses first 3, feature extraction needs all)
        ch_keys = sorted(all_channel_data.keys())
        n_channels = len(ch_keys)
        h, w = all_channel_data[ch_keys[0]].shape
        logger.info(f"Creating shared memory for {n_channels} channels ({h}x{w})...")
        logger.info(f"  Channel mapping: {ch_keys}")

        slide_shm_arr = shm_manager.create_slide_buffer(
            slide_name, (h, w, n_channels), all_channel_data[ch_keys[0]].dtype
        )
        for i, ch_key in enumerate(ch_keys):
            slide_shm_arr[:, :, i] = all_channel_data[ch_key]
            logger.info(f"  Loaded channel {ch_key} to shared memory slot {i}")

        ch_to_slot = {ch_key: i for i, ch_key in enumerate(ch_keys)}

        # --- Islet marker calibration (pilot phase, before freeing channel data) ---
        islet_gmm_thresholds = {}
        if args.cell_type == 'islet':
            marker_chs_str = getattr(args, 'islet_marker_channels', 'gcg:2,ins:3,sst:5')
            try:
                marker_chs_list = [int(pair.split(':')[1]) for pair in marker_chs_str.split(',')]
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Invalid --islet-marker-channels format: '{marker_chs_str}'. "
                    f"Expected 'name:ch_idx,...' e.g. 'gcg:2,ins:3,sst:5'. Error: {e}"
                )
            n_pilot = max(1, int(len(sampled_tiles) * 0.05))
            pilot_indices = np.random.choice(len(sampled_tiles), n_pilot, replace=False)
            pilot_tiles = [sampled_tiles[i] for i in pilot_indices]
            nuclei_only = getattr(args, 'nuclei_only', False)
            nuc_ch = getattr(args, 'nuclear_channel', 4)
            mem_ch = nuc_ch if nuclei_only else getattr(args, 'membrane_channel', 1)
            islet_gmm_thresholds = calibrate_islet_marker_gmm(
                pilot_tiles=pilot_tiles,
                loader=loader,
                all_channel_data=all_channel_data,
                slide_shm_arr=None,  # Use all_channel_data directly (still in RAM)
                ch_to_slot=None,
                marker_channels=marker_chs_list,
                membrane_channel=mem_ch,
                nuclear_channel=nuc_ch,
                tile_size=args.tile_size,
                pixel_size_um=pixel_size_um,
                nuclei_only=nuclei_only,
                mosaic_origin=(x_start, y_start),
            )

        # Free original channel data -- everything is now in shared memory
        mem_freed_gb = sum(arr.nbytes for arr in all_channel_data.values()) / (1024**3)
        del all_channel_data
        # Clear ALL loader channel data (not just primary) to free memory
        if hasattr(loader, 'clear_all_channels'):
            loader.clear_all_channels()
        else:
            loader.channel_data = None
        gc.collect()
        logger.info(f"Freed all_channel_data ({mem_freed_gb:.1f} GB) -- using shared memory for all reads")

        # Build strategy parameters from the already-constructed params dict
        strategy_params = dict(params)
        if islet_gmm_thresholds:
            strategy_params['gmm_prefilter_thresholds'] = islet_gmm_thresholds

        # Get classifier path
        classifier_path = None
        if args.cell_type == 'nmj':
            classifier_path = getattr(args, 'nmj_classifier', None)
            if classifier_path:
                logger.info(f"Using specified NMJ classifier: {classifier_path}")
            else:
                logger.info("No --nmj-classifier specified -- will return all candidates (annotation run)")
        elif args.cell_type == 'islet':
            classifier_path = getattr(args, 'islet_classifier', None)
            if classifier_path:
                logger.info(f"Using specified islet classifier: {classifier_path}")
            else:
                logger.info("No --islet-classifier specified -- will return all candidates (annotation run)")
        elif args.cell_type == 'tissue_pattern':
            classifier_path = getattr(args, 'tp_classifier', None)
            if classifier_path:
                logger.info(f"Using specified tissue_pattern classifier: {classifier_path}")
            else:
                logger.info("No --tp-classifier specified -- will return all candidates (annotation run)")

        extract_deep = getattr(args, 'extract_deep_features', False)

        # Vessel-specific params for multi-GPU
        mgpu_cd31_channel = getattr(args, 'cd31_channel', None) if args.cell_type == 'vessel' else None
        mgpu_channel_names = None
        if args.cell_type == 'vessel' and getattr(args, 'channel_names', None):
            names = args.channel_names.split(',')
            mgpu_channel_names = {ch_keys[i]: name.strip()
                                  for i, name in enumerate(names)
                                  if i < len(ch_keys)}

        # Add mosaic origin to slide_info so workers can convert global->relative coords
        mgpu_slide_info = shm_manager.get_slide_info()
        mgpu_slide_info[slide_name]['mosaic_origin'] = (x_start, y_start)

        # ---- Multi-scale vessel detection mode ----
        if is_multiscale:
            logger.info("=" * 60)
            logger.info(f"MULTI-SCALE VESSEL DETECTION -- {num_gpus} GPU(s)")
            logger.info("=" * 60)

            from segmentation.utils.multiscale import (
                get_scale_params, generate_tile_grid_at_scale,
                convert_detection_to_full_res, merge_detections_across_scales,
            )
            from tqdm import tqdm as tqdm_progress

            scales = [int(s.strip()) for s in args.scales.split(',')]
            iou_threshold = getattr(args, 'multiscale_iou_threshold', 0.3)
            tile_size = args.tile_size

            logger.info(f"Scales: {scales} (coarse to fine)")
            logger.info(f"IoU threshold for deduplication: {iou_threshold}")

            # One MultiGPUTileProcessor for all scales (workers stay alive, models stay loaded)
            with MultiGPUTileProcessor(
                num_gpus=num_gpus,
                slide_info=mgpu_slide_info,
                cell_type='vessel',
                strategy_params=strategy_params,
                pixel_size_um=pixel_size_um,
                classifier_path=classifier_path,
                extract_deep_features=extract_deep,
                extract_sam2_embeddings=True,
                detection_channel=tissue_channel,
                cd31_channel=mgpu_cd31_channel,
                channel_names=mgpu_channel_names,
                variance_threshold=variance_threshold,
                channel_keys=ch_keys,
            ) as processor:

                all_scale_detections = []  # Accumulate full-res detections across scales
                total_tiles_submitted = 0

                # Resume from checkpoints if available
                completed_scales = set()
                if getattr(args, 'resume_from', None):
                    checkpoint_dir = Path(args.resume_from) / "checkpoints"
                    if checkpoint_dir.exists():
                        # Sort by modification time (not lexicographic -- scale_8x > scale_16x lex)
                        checkpoint_files = sorted(
                            checkpoint_dir.glob("scale_*x.json"),
                            key=lambda p: p.stat().st_mtime,
                        )
                        if checkpoint_files:
                            latest = checkpoint_files[-1]
                            with open(latest) as f:
                                all_scale_detections = json.load(f)
                            # Restore numpy arrays for contours (json.load produces lists)
                            for det in all_scale_detections:
                                for key in ('outer', 'inner', 'outer_contour', 'inner_contour'):
                                    if key in det and det[key] is not None:
                                        det[key] = np.array(det[key], dtype=np.int32)
                            for cf in checkpoint_files:
                                # Parse scale from filename like "scale_32x.json"
                                try:
                                    s = int(cf.stem.split('_')[1].rstrip('x'))
                                    completed_scales.add(s)
                                except (IndexError, ValueError):
                                    logger.warning(f"Skipping unrecognized checkpoint file: {cf.name}")
                            logger.info(
                                f"Resumed from {latest}: {len(all_scale_detections)} detections, "
                                f"completed scales: {sorted(completed_scales)}"
                            )

                for scale in scales:
                    if scale in completed_scales:
                        logger.info(f"Scale 1/{scale}x: skipping (checkpointed)")
                        continue

                    scale_params = get_scale_params(scale)
                    scale_tiles = generate_tile_grid_at_scale(
                        mosaic_info['width'], mosaic_info['height'],
                        tile_size, scale, overlap=0,
                    )

                    # Sample tiles if requested
                    if args.sample_fraction < 1.0:
                        n_sample = max(1, int(len(scale_tiles) * args.sample_fraction))
                        indices = np.random.choice(len(scale_tiles), n_sample, replace=False)
                        scale_tiles = [scale_tiles[i] for i in indices]

                    logger.info(
                        f"Scale 1/{scale}x: {len(scale_tiles)} tiles, "
                        f"pixel_size={pixel_size_um * scale:.3f} um, "
                        f"target: {scale_params.get('description', '')}"
                    )

                    # Submit tiles for this scale with scale metadata
                    for tx_s, ty_s in scale_tiles:
                        tile_with_dims = {
                            'x': x_start + tx_s * scale,
                            'y': y_start + ty_s * scale,
                            'w': tile_size * scale,
                            'h': tile_size * scale,
                            'scale_factor': scale,
                            'scale_params': scale_params,
                            'tile_x_scaled': tx_s,
                            'tile_y_scaled': ty_s,
                        }
                        processor.submit_tile(slide_name, tile_with_dims)

                    # Collect results for this scale
                    pbar = tqdm_progress(total=len(scale_tiles), desc=f"Scale 1/{scale}x")
                    scale_det_count = 0
                    results_collected = 0

                    while results_collected < len(scale_tiles):
                        result = processor.collect_result(timeout=3600)
                        if result is None:
                            logger.warning(f"Timeout at scale 1/{scale}x")
                            break

                        results_collected += 1
                        pbar.update(1)

                        if result['status'] == 'success':
                            try:
                                tile_dict = result['tile']
                                features_list = result['features_list']
                                sf = result.get('scale_factor', scale)
                                tx_s = tile_dict.get('tile_x_scaled', 0)
                                ty_s = tile_dict.get('tile_y_scaled', 0)

                                # Apply vessel classifiers
                                apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                                # Convert each detection from downscaled-local to full-res global
                                for feat in features_list:
                                    if 'outer_contour' in feat and 'outer' not in feat:
                                        feat['outer'] = feat.pop('outer_contour')
                                    if 'inner_contour' in feat and 'inner' not in feat:
                                        feat['inner'] = feat.pop('inner_contour')
                                    if feat.get('features', {}).get('detection_type') == 'arc':
                                        feat['is_arc'] = True

                                    det_fullres = convert_detection_to_full_res(
                                        feat, sf, tx_s, ty_s,
                                        smooth=True,
                                        smooth_base_factor=getattr(args, 'smooth_contours_factor', 3.0),
                                    )

                                    # Add mosaic origin for CZI-global coords
                                    for key in ('center', 'centroid'):
                                        if key in det_fullres:
                                            det_fullres[key][0] += x_start
                                            det_fullres[key][1] += y_start
                                    feats_d = det_fullres.get('features', {})
                                    if isinstance(feats_d, dict):
                                        if 'center' in feats_d and feats_d['center'] is not None:
                                            fc = feats_d['center']
                                            feats_d['center'] = [
                                                (fc[0] + tx_s) * sf + x_start,
                                                (fc[1] + ty_s) * sf + y_start,
                                            ]
                                        for ck in ('outer_center', 'inner_center'):
                                            if ck in feats_d and feats_d[ck] is not None:
                                                feats_d[ck][0] += x_start
                                                feats_d[ck][1] += y_start
                                    mosaic_offset = np.array([x_start, y_start], dtype=np.int32)
                                    if 'outer' in det_fullres and det_fullres['outer'] is not None:
                                        det_fullres['outer'] = det_fullres['outer'] + mosaic_offset
                                    if 'inner' in det_fullres and det_fullres['inner'] is not None:
                                        det_fullres['inner'] = det_fullres['inner'] + mosaic_offset

                                    for ckey in ('outer', 'inner'):
                                        if ckey in det_fullres and det_fullres[ckey] is not None:
                                            det_fullres[f'{ckey}_contour'] = det_fullres[ckey]
                                            det_fullres[f'{ckey}_contour_global'] = [
                                                [int(pt[0][0]), int(pt[0][1])]
                                                for pt in det_fullres[ckey]
                                            ]

                                    det_fullres['scale_detected'] = sf
                                    all_scale_detections.append(det_fullres)
                                    scale_det_count += 1

                            except Exception as e:
                                import traceback
                                _tid = result.get('tid', '?')
                                logger.error(f"Error post-processing multiscale tile {_tid}: {e}")
                                logger.error(f"Traceback:\n{traceback.format_exc()}")

                        elif result['status'] == 'error':
                            logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

                    pbar.close()
                    total_tiles_submitted += results_collected
                    logger.info(f"Scale 1/{scale}x: {scale_det_count} detections")

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Save checkpoint after each scale
                    checkpoint_dir = slide_output_dir / "checkpoints"
                    checkpoint_dir.mkdir(exist_ok=True)
                    checkpoint_file = checkpoint_dir / f"scale_{scale}x.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(all_scale_detections, f, cls=NumpyEncoder)
                    logger.info(f"Checkpoint saved: {checkpoint_file} ({len(all_scale_detections)} detections)")

            # Merge across scales (contour-based IoU dedup)
            logger.info(f"Merging {len(all_scale_detections)} detections across scales...")
            merged_detections = merge_detections_across_scales(
                all_scale_detections, iou_threshold=iou_threshold,
                tile_size=args.tile_size,
            )
            logger.info(f"After merge: {len(merged_detections)} vessels")

            # Regenerate UIDs from full-res global coords and build all_detections
            for det in merged_detections:
                features_dict = det.get('features', {})
                center = features_dict.get('center', det.get('center', [0, 0]))
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    cx, cy = int(center[0]), int(center[1])
                else:
                    cx, cy = 0, 0

                uid = f"{slide_name}_vessel_{cx}_{cy}"
                det['uid'] = uid
                det['slide'] = slide_name
                det['center'] = [cx, cy]
                det['center_um'] = [cx * pixel_size_um, cy * pixel_size_um]
                det['global_center'] = [cx, cy]
                det['global_center_um'] = [cx * pixel_size_um, cy * pixel_size_um]
                all_detections.append(det)

            # Generate HTML crops from shared memory with percentile normalization
            logger.info(f"Generating HTML crops for {len(all_detections)} multiscale detections...")

            for det in all_detections:
                features_dict = det.get('features', {})
                cx, cy = det['center']

                diameter_um = features_dict.get('outer_diameter_um', 50)
                diameter_px = int(diameter_um / pixel_size_um)
                crop_size = max(300, min(800, int(diameter_px * 2)))
                half = crop_size // 2

                # Crop from shared memory (0-indexed, subtract mosaic origin)
                rel_cy = cy - y_start
                rel_cx = cx - x_start
                y1 = max(0, rel_cy - half)
                x1 = max(0, rel_cx - half)
                y2 = min(h, rel_cy + half)
                x2 = min(w, rel_cx + half)

                if n_channels >= 3:
                    crop_rgb = np.stack([
                        slide_shm_arr[y1:y2, x1:x2, i] for i in range(3)
                    ], axis=-1)
                else:
                    crop_rgb = np.stack([slide_shm_arr[y1:y2, x1:x2, 0]] * 3, axis=-1)

                if crop_rgb.size == 0:
                    continue

                # Percentile normalize (not /256) for proper dynamic range
                crop_rgb = percentile_normalize(crop_rgb, p_low=1, p_high=99.5)

                b64_str, _ = image_to_base64(crop_rgb)
                sample = {
                    'uid': det['uid'],
                    'image': b64_str,
                    'stats': features_dict,
                }
                all_samples.append(sample)

            logger.info(f"Multi-scale mode: {len(all_detections)} detections, {len(all_samples)} HTML samples "
                        f"from {total_tiles_submitted} tiles on {num_gpus} GPUs")

        # ---- Regular (non-multiscale) tile processing ----
        else:
            logger.info("=" * 60)
            logger.info(f"{args.cell_type.upper()} DETECTION -- {num_gpus} GPU(s)")
            logger.info("=" * 60)

            with MultiGPUTileProcessor(
                num_gpus=num_gpus,
                slide_info=mgpu_slide_info,
                cell_type=args.cell_type,
                strategy_params=strategy_params,
                pixel_size_um=pixel_size_um,
                classifier_path=classifier_path,
                extract_deep_features=extract_deep,
                extract_sam2_embeddings=True,
                detection_channel=tissue_channel,
                cd31_channel=mgpu_cd31_channel,
                channel_names=mgpu_channel_names,
                variance_threshold=variance_threshold,
                channel_keys=ch_keys,
                islet_display_channels=getattr(args, 'islet_display_chs', None),
            ) as processor:

                # Submit all tiles (add tile dimensions for worker)
                logger.info(f"Submitting {len(sampled_tiles)} tiles to {num_gpus} GPUs...")
                tile_size = args.tile_size
                for tile in sampled_tiles:
                    # Worker expects 'x', 'y', 'w', 'h' keys
                    tile_with_dims = {
                        'x': tile['x'],
                        'y': tile['y'],
                        'w': tile_size,
                        'h': tile_size,
                    }
                    processor.submit_tile(slide_name, tile_with_dims)

                # Collect results with progress bar
                from tqdm import tqdm as tqdm_progress
                pbar = tqdm_progress(total=len(sampled_tiles), desc="Processing tiles")

                results_collected = 0
                while results_collected < len(sampled_tiles):
                    result = processor.collect_result(timeout=14400)  # 4h timeout per tile
                    if result is None:
                        logger.warning("Timeout waiting for result")
                        break

                    results_collected += 1
                    pbar.update(1)

                    if result['status'] == 'success':
                        try:
                            tile = result['tile']
                            tile_x, tile_y = tile['x'], tile['y']
                            masks = result['masks']
                            features_list = result['features_list']

                            # Skip tiles with no detections (masks=None from worker)
                            if masks is None or len(features_list) == 0:
                                continue

                            # Apply vessel classifier post-processing BEFORE saving
                            if args.cell_type == 'vessel':
                                apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                            # Save tile outputs
                            tile_id = f"tile_{tile_x}_{tile_y}"
                            tile_out = tiles_dir / tile_id
                            tile_out.mkdir(exist_ok=True)

                            # Save masks
                            with h5py.File(tile_out / f"{args.cell_type}_masks.h5", 'w') as f:
                                create_hdf5_dataset(f, 'masks', masks)

                            # Save features (includes vessel classification if applicable)
                            with open(tile_out / f"{args.cell_type}_features.json", 'w') as f:
                                json.dump(features_list, f, cls=NumpyEncoder)

                            # Add detections to global list
                            for feat in features_list:
                                all_detections.append(feat)

                            # Create samples for HTML
                            # Convert global tile coords to 0-based array indices
                            rel_tx = tile_x - x_start
                            rel_ty = tile_y - y_start
                            # Use masks.shape to handle edge tiles (smaller than tile_size at boundaries)
                            tile_h, tile_w = masks.shape[:2]
                            # Read HTML crops from shared memory (all_channel_data freed after shm creation)
                            if args.cell_type == 'islet' and hasattr(args, 'islet_display_chs'):
                                # Islet: display channels from --islet-display-channels (R, G, B)
                                _islet_disp = args.islet_display_chs
                                _shm_dtype = slide_shm_arr.dtype
                                rgb_channels = []
                                for _ci in range(3):
                                    if _ci < len(_islet_disp) and _islet_disp[_ci] in ch_to_slot:
                                        rgb_channels.append(
                                            slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, ch_to_slot[_islet_disp[_ci]]]
                                        )
                                    else:
                                        rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_shm_dtype))
                                tile_rgb_html = np.stack(rgb_channels, axis=-1)
                            elif args.cell_type == 'tissue_pattern':
                                # Tissue pattern: configurable R/G/B display channels
                                tp_disp = getattr(args, 'tp_display_channels_list', [0, 3, 1])
                                _shm_dtype = slide_shm_arr.dtype
                                rgb_channels = []
                                for _ci in range(3):
                                    if _ci < len(tp_disp) and tp_disp[_ci] in ch_to_slot:
                                        rgb_channels.append(
                                            slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, ch_to_slot[tp_disp[_ci]]]
                                        )
                                    else:
                                        rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_shm_dtype))
                                tile_rgb_html = np.stack(rgb_channels, axis=-1)
                            elif n_channels >= 3:
                                tile_rgb_html = np.stack([
                                    slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, i]
                                    for i in range(3)
                                ], axis=-1)
                            else:
                                tile_rgb_html = np.stack([
                                    slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, 0]
                                ] * 3, axis=-1)

                            tile_pct = _compute_tile_percentiles(tile_rgb_html) if getattr(args, 'html_normalization', 'crop') == 'tile' else None

                            if args.cell_type == 'islet':
                                # Flush tile data to disk to avoid OOM
                                np.save(tile_out / 'tile_rgb_html.npy', tile_rgb_html)
                                if tile_pct is not None:
                                    with open(tile_out / 'tile_pct.json', 'w') as f_pct:
                                        json.dump(tile_pct, f_pct)
                                deferred_html_tiles.append({
                                    'tile_dir': str(tile_out),
                                    'tile_x': tile_x, 'tile_y': tile_y,
                                    'tile_pct': tile_pct,
                                })
                                del masks, tile_rgb_html, features_list
                                result['masks'] = None
                                result['features_list'] = None
                                gc.collect()
                            else:
                                _max_html = getattr(args, 'max_html_samples', 0)
                                if _max_html > 0 and len(all_samples) >= _max_html:
                                    pass  # Skip HTML crop generation -- cap reached
                                else:
                                    html_samples = filter_and_create_html_samples(
                                        features_list, tile_x, tile_y, tile_rgb_html, masks,
                                        pixel_size_um, slide_name, args.cell_type,
                                        args.html_score_threshold,
                                        tile_percentiles=tile_pct,
                                        candidate_mode=args.candidate_mode,
                                        vessel_params=params if args.cell_type == 'vessel' else None,
                                    )
                                    all_samples.extend(html_samples)

                        except Exception as e:
                            import traceback
                            logger.error(f"Error post-processing tile ({tile_x}, {tile_y}): {e}")
                            logger.error(f"Traceback:\n{traceback.format_exc()}")

                    elif result['status'] in ('empty', 'no_tissue'):
                        pass  # Normal - no tissue in tile
                    elif result['status'] == 'error':
                        logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

                pbar.close()

                # Deferred HTML generation for islet (needs population-level marker thresholds)
                if deferred_html_tiles and args.cell_type == 'islet':
                    _islet_mm = getattr(args, 'islet_marker_map', None)
                    _marker_top_pct = getattr(args, 'marker_top_pct', 5)
                    _pct_chs_str = getattr(args, 'marker_pct_channels', 'sst')
                    _pct_channels = set(s.strip() for s in _pct_chs_str.split(',')) if _pct_chs_str else set()
                    _gmm_p = getattr(args, 'gmm_p_cutoff', 0.75)
                    _ratio_min = getattr(args, 'ratio_min', 1.5)
                    marker_thresholds = compute_islet_marker_thresholds(
                        all_detections, marker_map=_islet_mm,
                        marker_top_pct=_marker_top_pct,
                        pct_channels=_pct_channels,
                        gmm_p_cutoff=_gmm_p,
                        ratio_min=_ratio_min) if all_detections else None
                    # Add marker_class to each detection for JSON export
                    if marker_thresholds:
                        counts = {}
                        for det in all_detections:
                            mc, _ = classify_islet_marker(
                                det.get('features', {}), marker_thresholds, marker_map=_islet_mm)
                            det['marker_class'] = mc
                            counts[mc] = counts.get(mc, 0) + 1
                        logger.info(f"Islet marker classification: {counts}")
                    uid_to_marker = {d.get('uid', ''): d.get('marker_class') for d in all_detections}
                    try:
                        for dt in deferred_html_tiles:
                            _td = Path(dt['tile_dir'])
                            _tile_rgb = np.load(_td / 'tile_rgb_html.npy')
                            with h5py.File(_td / f'{args.cell_type}_masks.h5', 'r') as _hf:
                                _tile_masks = _hf['masks'][:]
                            with open(_td / f'{args.cell_type}_features.json', 'r') as _ff:
                                _tile_feats = json.load(_ff)
                            for _feat in _tile_feats:
                                _mc = uid_to_marker.get(_feat.get('uid', ''))
                                if _mc:
                                    _feat['marker_class'] = _mc
                            html_samples = filter_and_create_html_samples(
                                _tile_feats, dt['tile_x'], dt['tile_y'],
                                _tile_rgb, _tile_masks,
                                pixel_size_um, slide_name, args.cell_type,
                                args.html_score_threshold,
                                tile_percentiles=dt['tile_pct'],
                                marker_thresholds=marker_thresholds,
                                marker_map=_islet_mm,
                                candidate_mode=args.candidate_mode,
                                vessel_params=params if args.cell_type == 'vessel' else None,
                            )
                            all_samples.extend(html_samples)
                            # Clean up deferred temp files
                            _npy_path = _td / 'tile_rgb_html.npy'
                            if _npy_path.exists():
                                _npy_path.unlink()
                            _pct_path = _td / 'tile_pct.json'
                            if _pct_path.exists():
                                _pct_path.unlink()
                            del _tile_rgb, _tile_masks, _tile_feats
                            gc.collect()
                    finally:
                        # Ensure ALL deferred .npy files are cleaned up even on error
                        for dt in deferred_html_tiles:
                            _td = Path(dt['tile_dir'])
                            for _tmp in ('tile_rgb_html.npy', 'tile_pct.json'):
                                _p = _td / _tmp
                                if _p.exists():
                                    _p.unlink()
                        deferred_html_tiles = []
                    gc.collect()

            logger.info(f"Processing complete: {len(all_detections)} {args.cell_type} detections from {results_collected} tiles")

    finally:
        # Cleanup shared memory
        shm_manager.cleanup()

    logger.info(f"Total detections (pre-dedup): {len(all_detections)}")

    # Detection-only mode: skip dedup, HTML, CSV -- just save per-tile results and exit
    if getattr(args, 'detection_only', False):
        logger.info(f"Detection-only mode: {len(all_detections)} detections saved to tile dirs. Exiting.")
        if getattr(args, 'tile_shard', None):
            shard_idx, shard_total = args.tile_shard
            logger.info(f"Shard {shard_idx}/{shard_total} complete.")
        return

    # Deduplication: tile overlap causes same detection in adjacent tiles
    # Uses actual mask pixel overlap (loads HDF5 mask files) for accurate dedup
    # Skip for multiscale -- already deduped by contour IoU in merge_detections_across_scales()
    if not is_multiscale and getattr(args, 'tile_overlap', 0) > 0 and len(all_detections) > 0:
        from segmentation.processing.deduplication import deduplicate_by_mask_overlap
        pre_dedup = len(all_detections)
        mask_fn = f'{args.cell_type}_masks.h5'
        dedup_sort = 'confidence' if getattr(args, 'dedup_by_confidence', False) else 'area'
        all_detections = deduplicate_by_mask_overlap(
            all_detections, tiles_dir, min_overlap_fraction=0.1,
            mask_filename=mask_fn, sort_by=dedup_sort,
        )

        # Filter HTML samples to match deduped detections and remove duplicate UIDs
        deduped_uids = {det.get('uid', det.get('id', '')) for det in all_detections}
        seen_uids = set()
        unique_samples = []
        for s in all_samples:
            uid = s.get('uid', '')
            if uid in deduped_uids and uid not in seen_uids:
                seen_uids.add(uid)
                unique_samples.append(s)
        logger.info(f"Dedup: {len(all_samples)} HTML samples -> {len(unique_samples)} (removed {len(all_samples) - len(unique_samples)} duplicate UIDs)")
        all_samples = unique_samples

    # ---- Shared post-processing: CSV, JSON, HTML, summary, server ----
    _finish_pipeline(
        args, all_detections, all_samples, slide_output_dir, tiles_dir,
        pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
        all_tiles=all_tiles, tissue_tiles=tissue_tiles, sampled_tiles=sampled_tiles,
        resumed=False, params=params, classifier_loaded=classifier_loaded,
        is_multiscale=is_multiscale, detector=detector,
    )


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Early-exit commands
    if args.stop_server:
        setup_logging()
        stop_background_server()
        return

    if args.server_status:
        show_server_status()
        return

    if args.show_metadata:
        print_czi_metadata(args.czi_path, scene=args.scene)
        return

    # Require --czi-path for actual pipeline runs
    if args.czi_path is None:
        parser.error("--czi-path is required (unless using --stop-server, --server-status, or --show-metadata)")

    # Require --cell-type if not showing metadata
    if args.cell_type is None:
        parser.error("--cell-type is required unless using --show-metadata")

    args = postprocess_args(args, parser)
    run_pipeline(args)


if __name__ == '__main__':
    main()
