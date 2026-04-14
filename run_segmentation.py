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

from pathlib import Path

import numpy as np

from xldvp_seg.io.czi_loader import get_czi_metadata, get_loader, print_czi_metadata
from xldvp_seg.pipeline.cli import build_parser, postprocess_args
from xldvp_seg.pipeline.detection_loop import run_detection_loop
from xldvp_seg.pipeline.finalize import _finish_pipeline
from xldvp_seg.pipeline.resume import (
    _resume_generate_html_samples,
    detect_resume_stage,
    reload_detections_from_tiles,
)
from xldvp_seg.pipeline.server import show_server_status, stop_background_server
from xldvp_seg.pipeline.shm_setup import setup_shared_memory
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _maybe_export_ome_zarr(
    args, slide_shm_arr, ch_to_slot, pixel_size_um, slide_output_dir, slide_name, czi_path
):
    if not getattr(args, "ome_zarr", True):
        return
    try:
        from xldvp_seg.io.ome_zarr_export import export_shm_to_ome_zarr

        zarr_path = slide_output_dir / f"{slide_name}.ome.zarr"
        export_shm_to_ome_zarr(
            shm_array=slide_shm_arr,
            ch_to_slot=ch_to_slot,
            pixel_size_um=pixel_size_um,
            output_path=zarr_path,
            czi_path=str(czi_path),
            pyramid_levels=getattr(args, "zarr_levels", 5),
            overwrite=getattr(args, "force_zarr", False),
        )
    except (MemoryError, OSError):
        raise
    except Exception as e:
        logger.warning(f"OME-Zarr export failed (non-fatal): {e}")


def _apply_marker_snr_classification(args, detections, slide_output_dir):
    """Classify marker+/- from pre-computed SNR features (zero extra cost).

    Called after ``process_detections_post_dedup`` which writes ``ch{N}_snr``
    into each detection's features dict.  This function simply applies a
    threshold comparison and stores ``{name}_class`` / ``marker_profile``.
    """
    spec = getattr(args, "marker_snr_channels", None)
    if not spec:
        return

    from xldvp_seg.analysis.marker_classification import classify_single_marker

    marker_specs = [s.strip() for s in spec.split(",")]
    marker_names = []
    for ms in marker_specs:
        if ":" not in ms:
            logger.warning("--marker-snr-channels: skipping '%s' (expected NAME:CHANNEL)", ms)
            continue
        name, ch_str = ms.split(":", 1)
        try:
            ch = int(ch_str.strip())
        except ValueError:
            logger.warning("--marker-snr-channels: skipping '%s' (channel must be integer)", ms)
            continue
        name = name.strip()
        # Verify channel has SNR features before attempting classification
        has_snr = (
            any(d.get("features", {}).get(f"ch{ch}_snr") for d in detections[:5])
            if detections
            else False
        )
        if not has_snr:
            logger.warning(
                "--marker-snr-channels: skipping %s — ch%d_snr not found in features "
                "(background correction may have been disabled or channel does not exist)",
                name,
                ch,
            )
            continue
        marker_names.append(name)
        logger.info("Auto-classifying marker %s (ch%d, SNR >= 1.5)...", name, ch)
        try:
            classify_single_marker(
                detections,
                channel=ch,
                marker_name=name,
                method="snr",
                output_dir=Path(slide_output_dir),
                snr_threshold=1.5,
            )
        except Exception as e:
            logger.warning("Marker SNR classification failed for %s (ch%d): %s", name, ch, e)

    # Build marker_profile from classified markers
    for det in detections:
        feat = det.setdefault("features", {})
        parts = []
        for mn in marker_names:
            cls = feat.get(f"{mn}_class", "negative")
            parts.append(f"{mn}+" if cls == "positive" else f"{mn}-")
        feat["marker_profile"] = "/".join(parts)

    logger.info(
        "Marker classification complete: %s",
        ", ".join(f"{n} (ch{s.split(':')[1].strip()})" for n, s in zip(marker_names, marker_specs)),
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_pipeline(args):
    """Main pipeline execution."""
    # Setup logging
    setup_logging(level="DEBUG" if getattr(args, "verbose", False) else "INFO")

    from datetime import datetime

    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    slide_name = czi_path.stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed early -- ensures all multi-node shards get identical
    # tissue calibration and tile sampling (same tile list on every node)
    import random as _random_mod

    _seed = getattr(args, "random_seed", 42)
    np.random.seed(_seed)
    _random_mod.seed(_seed)
    try:
        import torch

        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        logger.info("Random seed: %d (numpy + stdlib + torch)", _seed)
    except ImportError:
        logger.info("Random seed: %d (numpy + stdlib; torch not available)", _seed)

    logger.info("=" * 60)
    logger.info("UNIFIED SEGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Slide: {slide_name}")
    logger.info(f"Run: {run_timestamp}")
    logger.info(f"Cell type: {args.cell_type}")
    logger.info(f"Channel: {args.channel}")
    if args.scene != 0:
        logger.info(f"Scene: {args.scene}")
    if getattr(args, "multi_marker", False):
        logger.info(
            "Multi-marker mode: ENABLED (auto-enabled --all-channels and --parallel-detection)"
        )
    if getattr(args, "tile_shard", None):
        shard_idx, shard_total = args.tile_shard
        if not getattr(args, "resume", None):
            logger.error(
                "--tile-shard REQUIRES --resume <shared-dir> so all shards write to "
                "the same location and share the tile list. Without --resume, each shard "
                "creates its own directory with independently-ordered tiles, causing "
                "gaps in coverage. Aborting."
            )
            raise SystemExit(1)
        logger.info(f"Tile shard: {shard_idx}/{shard_total} (detection-only)")
    elif getattr(args, "detection_only", False):
        logger.info("Detection-only mode (skipping dedup/HTML/CSV)")
    logger.info(f"Random seed: {getattr(args, 'random_seed', 42)}")
    logger.info("=" * 60)

    # ---- Resume early-exit: skip CZI loading if ALL stages are done ----
    _cached_resume_info = None
    if getattr(args, "resume", None):
        resume_dir = Path(args.resume)
        resume_info = detect_resume_stage(resume_dir, args.cell_type)
        _cached_resume_info = resume_info
        _has_det = resume_info["has_detections"] and not args.force_detect
        _has_html = resume_info["has_html"] and not args.force_html
        _has_tiles = resume_info["has_tiles"] and not args.force_detect

        logger.info(
            f"Resume state: tiles={resume_info['tile_count']}, "
            f"detections={'yes' if resume_info['has_detections'] else 'no'}"
            f" ({resume_info['detection_count']}), "
            f"html={'yes' if resume_info['has_html'] else 'no'}"
        )

        if _has_det and _has_html:
            # Everything done -- just regenerate CSV/summary without loading CZI
            logger.info(
                "All stages complete -- regenerating CSV and summary only (no CZI load needed)"
            )
            slide_output_dir = resume_dir
            det_file = slide_output_dir / f"{args.cell_type}_detections.json"
            all_detections = fast_json_load(det_file)

            # Load pipeline config for metadata
            config_file = slide_output_dir / "pipeline_config.json"
            if config_file.exists():
                cfg = fast_json_load(config_file)
                pixel_size_um = cfg.get("pixel_size_um")
                if pixel_size_um is None:
                    raise ValueError(
                        f"pipeline_config.json is missing 'pixel_size_um'. "
                        f"Re-run detection or add pixel_size_um to {config_file}"
                    )
                mosaic_info = {
                    "width": cfg.get("width", 0),
                    "height": cfg.get("height", 0),
                    "x": cfg.get("x_start", 0),
                    "y": cfg.get("y_start", 0),
                }
            else:
                # Minimal load: just metadata from CZI
                loader = get_loader(
                    czi_path, load_to_ram=False, channel=args.channel, scene=args.scene
                )
                pixel_size_um = loader.get_pixel_size()
                mosaic_info = {
                    "width": loader.mosaic_size[0],
                    "height": loader.mosaic_size[1],
                    "x": loader.mosaic_origin[0],
                    "y": loader.mosaic_origin[1],
                }

            # Apply marker SNR classification if requested (uses pre-computed SNR)
            _apply_marker_snr_classification(args, all_detections, slide_output_dir)

            pct = int(args.sample_fraction * 100)
            _finish_pipeline(
                args,
                all_detections,
                [],
                slide_output_dir,
                slide_output_dir / "tiles",
                pixel_size_um,
                slide_name,
                mosaic_info,
                run_timestamp,
                pct,
                skip_html=True,
                resumed=True,
            )
            return

    # Direct-to-SHM architecture: Load CZI channels directly into shared memory,
    # bypassing RAM allocation. Eliminates the RAM→SHM copy step and reduces peak memory.
    if not args.load_to_ram:
        logger.warning("--no-ram is deprecated and ignored. All data is loaded to shared memory.")
        args.load_to_ram = True

    logger.info("Loading CZI metadata (direct-to-SHM architecture)...")
    loader = get_loader(
        czi_path,
        load_to_ram=False,  # metadata only — channels loaded directly to SHM below
        channel=args.channel,
        quiet=False,
        scene=args.scene,
    )

    # Get mosaic bounds from loader properties
    x_start, y_start = loader.mosaic_origin
    width, height = loader.mosaic_size

    # Build mosaic_info dict for compatibility with existing functions
    mosaic_info = {
        "x": x_start,
        "y": y_start,
        "width": width,
        "height": height,
    }
    pixel_size_um = loader.get_pixel_size()

    logger.info(f"  Mosaic: {mosaic_info['width']} x {mosaic_info['height']} px")
    logger.info(f"  Origin: ({mosaic_info['x']}, {mosaic_info['y']})")
    logger.info(f"  Pixel size: {pixel_size_um:.4f} um/px")

    # Always log channel metadata so the log is self-documenting
    _czi_meta = None
    try:
        _czi_meta = get_czi_metadata(czi_path, scene=args.scene)
        logger.info(f"  CZI channels ({_czi_meta['n_channels']}):")
        for _ch in _czi_meta["channels"]:
            _ex = f"{_ch['excitation_nm']:.0f}" if _ch["excitation_nm"] else "?"
            _em = f"{_ch['emission_nm']:.0f}" if _ch["emission_nm"] else "?"
            _label = _ch["fluorophore"] if _ch["fluorophore"] != "N/A" else _ch["name"]
            logger.info(
                f"    [{_ch['index']}] {_ch['name']:<20s}  Ex {_ex} -> Em {_em} nm  ({_label})"
            )
        # Validate --channel against actual CZI channel count
        n_czi_channels = _czi_meta["n_channels"]
        _channels_to_check = [("--channel", args.channel)]
        # Only validate cell-type-specific channels when that cell type is active
        if getattr(args, "cd31_channel", None) is not None:
            _channels_to_check.append(("--cd31-channel", args.cd31_channel))
        if args.cell_type == "islet":
            _channels_to_check.append(("--membrane-channel", getattr(args, "membrane_channel", 1)))
            _channels_to_check.append(("--nuclear-channel", getattr(args, "nuclear_channel", 4)))
        for _flag, _ch_val in _channels_to_check:
            if _ch_val >= n_czi_channels:
                raise ValueError(
                    f"{_flag} {_ch_val} is out of range: CZI has {n_czi_channels} channels (0-{n_czi_channels - 1})"
                )
    except ValueError:
        raise  # Re-raise channel validation errors
    except Exception as _e:
        logger.warning(f"  Could not read channel metadata: {_e}")

    # Determine all channels to load
    ch_list = [args.channel]
    if getattr(args, "all_channels", False):
        if getattr(args, "channels", None):
            ch_list = [int(x.strip()) for x in args.channels.split(",")]
            logger.info(f"Will load specified channels {ch_list} for multi-channel analysis...")
        else:
            try:
                dims = loader.reader.get_dims_shape()[0]
                _n_ch = dims.get("C", (0, 3))[1]
            except Exception:
                _n_ch = 3
            ch_list = list(range(_n_ch))
            logger.info(f"Will load all {len(ch_list)} channels for multi-channel analysis...")
    # Also include CD31 channel for vessel validation
    if args.cell_type == "vessel" and args.cd31_channel is not None:
        if args.cd31_channel not in ch_list:
            ch_list.append(args.cd31_channel)
    ch_list = sorted(set(ch_list))

    # ---- SHM setup: load channels, filter tissue tiles, set up output dirs ----
    _shm = setup_shared_memory(
        args=args,
        loader=loader,
        mosaic_info=mosaic_info,
        pixel_size_um=pixel_size_um,
        ch_list=ch_list,
        x_start=x_start,
        y_start=y_start,
        output_dir=output_dir,
        run_timestamp=run_timestamp,
        cached_resume_info=_cached_resume_info,
        czi_meta=_czi_meta,
    )
    slide_shm_arr = _shm["slide_shm_arr"]
    ch_to_slot = _shm["ch_to_slot"]
    shm_manager = _shm["shm_manager"]
    all_channel_data = _shm["all_channel_data"]
    all_tiles = _shm["all_tiles"]
    tissue_tiles = _shm["tissue_tiles"]
    sampled_tiles = _shm["sampled_tiles"]
    variance_threshold = _shm["variance_threshold"]
    pct = _shm["pct"]
    slide_output_dir = _shm["slide_output_dir"]
    tiles_dir = _shm["tiles_dir"]
    tissue_channel = _shm["tissue_channel"]
    h = _shm["h"]
    w = _shm["w"]

    # ---- Resume: detect completed stages and set skip flags ----
    skip_detection = False
    skip_dedup = False
    skip_html = False

    if getattr(args, "resume", None) and not getattr(args, "detection_only", False):
        resume_info = (
            _cached_resume_info
            if _cached_resume_info is not None
            else detect_resume_stage(slide_output_dir, args.cell_type)
        )
        logger.info(
            f"Resume state: tiles={resume_info['tile_count']}, "
            f"detections={'yes' if resume_info['has_detections'] else 'no'}"
            f" ({resume_info['detection_count']}), "
            f"html={'yes' if resume_info['has_html'] else 'no'}"
        )

        # --merge-shards: always reload from tiles (shard detections are pre-dedup),
        # always run dedup, always generate HTML. Checkpointed at each stage.
        if getattr(args, "merge_shards", False):
            # Validate shard completeness
            shard_manifests = list(slide_output_dir.glob("shard_*_manifest.json"))
            if shard_manifests:
                m0 = fast_json_load(shard_manifests[0])
                expected_shards = m0.get("shard_total", len(shard_manifests))
                if len(shard_manifests) < expected_shards:
                    logger.warning(
                        f"Only {len(shard_manifests)}/{expected_shards} shard manifests found -- "
                        f"some shards may not have completed. Merge will use available data only."
                    )

            merged_det_file = slide_output_dir / f"{args.cell_type}_detections_merged.json"
            deduped_det_file = slide_output_dir / f"{args.cell_type}_detections.json"

            # Checkpoint 1: merged detections (all shards concatenated)
            if merged_det_file.exists() and not args.force_detect:
                all_detections = fast_json_load(merged_det_file)
                logger.info(
                    f"Checkpoint: loaded {len(all_detections)} merged detections from {merged_det_file.name}"
                )
            elif resume_info["has_tiles"]:
                all_detections = reload_detections_from_tiles(tiles_dir, args.cell_type)
                logger.info(
                    f"Merged {len(all_detections)} detections from {resume_info['tile_count']} tile dirs"
                )
                # Save checkpoint (atomic to prevent corruption on SLURM timeout)
                atomic_json_dump(all_detections, merged_det_file)
                logger.info(f"Checkpoint saved: {merged_det_file.name}")

                # Verify tile coverage: compare processed tiles vs expected tissue tiles
                sampled_tiles_file = slide_output_dir / "sampled_tiles.json"
                if sampled_tiles_file.exists():
                    expected_tiles = fast_json_load(sampled_tiles_file)
                    actual_tile_count = resume_info["tile_count"]
                    expected_count = len(expected_tiles)
                    coverage_pct = (
                        100 * actual_tile_count / expected_count if expected_count > 0 else 0
                    )
                    logger.info(
                        f"Tile coverage: {actual_tile_count}/{expected_count} "
                        f"({coverage_pct:.1f}%)"
                    )
                    if actual_tile_count < expected_count:
                        missing = expected_count - actual_tile_count
                        logger.warning(
                            f"INCOMPLETE COVERAGE: {missing} tissue tiles have no detections. "
                            f"This may indicate shard failures or tissue detection inconsistency. "
                            f"Expected {expected_count} tiles, found {actual_tile_count}."
                        )
            else:
                logger.error("No tile dirs found -- nothing to merge")
                return

            # Checkpoint 2: deduped detections
            # Checkpoint 3: post-dedup processed detections (contours + bg correction)
            postdedup_file = slide_output_dir / f"{args.cell_type}_detections_postdedup.json"
            if postdedup_file.exists() and not args.force_detect:
                all_detections = fast_json_load(postdedup_file)
                logger.info(
                    f"Checkpoint: loaded {len(all_detections)} post-dedup detections from {postdedup_file.name}"
                )
                skip_dedup = True
            elif deduped_det_file.exists() and not args.force_detect:
                all_detections = fast_json_load(deduped_det_file)
                logger.info(
                    f"Checkpoint: loaded {len(all_detections)} deduped detections from {deduped_det_file.name}"
                )
                skip_dedup = True
            skip_detection = True
            skip_html = False  # Always regenerate HTML for merge
            args.force_html = True
        elif (resume_info["has_detections"] or resume_info["has_tiles"]) and not args.force_detect:
            # Always fall through to detection — per-tile resume will skip
            # completed tiles and only process missing ones. This prevents
            # trusting incomplete checkpoint JSONs from partial runs.
            skip_detection = False
            skip_dedup = False
            all_detections = []
            logger.info(
                f"Resume: {resume_info['tile_count']} tiles found. "
                f"Per-tile check will skip completed tiles and detect missing ones."
            )

        if resume_info["has_html"] and not args.force_html:
            skip_html = True
            logger.info("HTML exists -- skipping HTML generation (use --force-html to regenerate)")

    # Save pipeline config for resume/regeneration
    pipeline_config = {
        "czi_path": str(czi_path),
        "cell_type": args.cell_type,
        "tile_size": args.tile_size,
        "pixel_size_um": pixel_size_um,
        "scene": args.scene,
        "width": width,
        "height": height,
        "x_start": x_start,
        "y_start": y_start,
        "sample_fraction": args.sample_fraction,
        "tile_overlap": args.tile_overlap,
        "channel": args.channel,
    }
    # Add display channel config
    if args.cell_type == "islet":
        pipeline_config["display_channels"] = getattr(args, "islet_display_chs", [2, 3, 5])
        pipeline_config["marker_channels"] = getattr(
            args, "islet_marker_channels", "gcg:2,ins:3,sst:5"
        )
    elif args.cell_type == "tissue_pattern":
        pipeline_config["display_channels"] = getattr(args, "tp_display_channels_list", [0, 3, 1])
    config_file = slide_output_dir / "pipeline_config.json"
    if not config_file.exists() or not getattr(args, "resume", None):
        atomic_json_dump(pipeline_config, config_file)

    # ---- Resume fast-path: skip detection, go straight to dedup/HTML/CSV ----
    if skip_detection:
        try:
            is_multiscale = args.cell_type == "vessel" and getattr(args, "multi_scale", False)

            # Run dedup if reloaded from tiles (not from deduped detections JSON)
            if (
                not skip_dedup
                and args.tile_overlap > 0
                and len(all_detections) > 0
                and not is_multiscale
            ):
                pre_dedup = len(all_detections)
                mask_fn = f"{args.cell_type}_masks.h5"
                dedup_sort = "confidence" if getattr(args, "dedup_by_confidence", False) else "area"
                if getattr(args, "dedup_method", "mask_overlap") == "iou_nms":
                    from xldvp_seg.processing.deduplication import deduplicate_by_iou_nms

                    all_detections = deduplicate_by_iou_nms(
                        all_detections,
                        tiles_dir,
                        iou_threshold=getattr(args, "iou_threshold", 0.2),
                        mask_filename=mask_fn,
                        sort_by=dedup_sort,
                    )
                else:
                    from xldvp_seg.processing.deduplication import deduplicate_by_mask_overlap

                    all_detections = deduplicate_by_mask_overlap(
                        all_detections,
                        tiles_dir,
                        min_overlap_fraction=0.1,
                        mask_filename=mask_fn,
                        sort_by=dedup_sort,
                    )
                logger.info(f"Dedup (resume): {pre_dedup} -> {len(all_detections)}")

            # Post-dedup processing (resume path) — skip already-completed steps
            _has_bg = False
            _has_contour = False
            if all_detections:
                _sample_feat = all_detections[0].get("features", {})
                _has_bg = any(k.endswith("_background") for k in _sample_feat)
                _has_contour = (
                    all_detections[0].get("contour_px") is not None
                    or all_detections[0].get("contour_dilated_px") is not None
                )

            _want_contour = args.contour_processing and not _has_contour
            _want_bg = args.background_correction and not _has_bg

            if len(all_detections) > 0 and (_want_contour or _want_bg):
                from xldvp_seg.pipeline.post_detection import process_detections_post_dedup

                mask_fn = f"{args.cell_type}_masks.h5"

                # Ensure all channels from the original run are loaded (not just primary).
                # Discover channels from detection features (ch{N}_mean keys).
                _needed_channels: set[int] = set()
                _sample_keys = all_detections[0].get("features", {}).keys()
                for _k in _sample_keys:
                    if _k.startswith("ch") and _k.endswith("_mean"):
                        _ch_str = _k[2:].replace("_mean", "")
                        try:
                            _needed_channels.add(int(_ch_str))
                        except ValueError:
                            pass
                if not _needed_channels:
                    # Fallback: at least load all CZI channels
                    try:
                        _dims = loader.reader.get_dims_shape()[0]
                        _n_ch = _dims.get("C", (0, 3))[1]
                        _needed_channels = set(range(_n_ch))
                    except Exception:
                        _needed_channels = {args.channel}

                for _ch in sorted(_needed_channels):
                    if _ch not in all_channel_data:
                        logger.info(
                            f"  Loading channel {_ch} for post-dedup processing (resume)..."
                        )
                        loader.load_channel(_ch)
                        all_channel_data[_ch] = loader.get_channel_data(_ch)

                # If extra channels were loaded outside SHM, fall back to loader
                # (SHM reader only sees channels in ch_to_slot)
                _extra_loaded = _needed_channels - set(ch_to_slot.keys())
                _use_shm = not _extra_loaded
                if _extra_loaded:
                    logger.info(
                        f"  Extra channels {sorted(_extra_loaded)} loaded to RAM (not SHM). "
                        f"Using loader fallback for post-dedup."
                    )

                if _has_contour and not _has_bg:
                    logger.info("Contours already processed — running background correction only")
                elif _has_bg and not _has_contour:
                    logger.info("Background already corrected — running contour processing only")

                # Nuclear counting: resolve nuclear channel.
                # When SHM is available we use the multi-GPU Phase 4 orchestrator,
                # which loads Cellpose+SAM2 inside each worker process — no need
                # to preload in the main process. Single-process fallback is used
                # only when SHM is unavailable (loader path).
                _count_nuclei = getattr(args, "count_nuclei", False)
                _nuc_ch_for_counting = getattr(args, "nuc_channel_for_counting", None)
                _cp_model_for_nuc = None
                _sam2_for_nuc = None
                _nuc_model_mgr = None
                _phase4_num_gpus = int(getattr(args, "num_gpus", 0)) if _use_shm else 0
                if _count_nuclei:
                    if _nuc_ch_for_counting is None:
                        cp_input = getattr(args, "cellpose_input_channels", None)
                        if cp_input:
                            parts = str(cp_input).split(",")
                            if len(parts) >= 2:
                                try:
                                    _nuc_ch_for_counting = int(parts[1].strip())
                                except ValueError:
                                    pass
                    if _nuc_ch_for_counting is None and args.cell_type == "islet":
                        _nuc_ch_for_counting = getattr(args, "nuclear_channel", None)
                    if _nuc_ch_for_counting is None and args.cell_type == "tissue_pattern":
                        _nuc_ch_for_counting = getattr(args, "tp_nuclear_channel", None)

                    if _nuc_ch_for_counting is not None:
                        if _phase4_num_gpus >= 1:
                            logger.info(
                                "Nuclear counting enabled: nuc_channel=%s, "
                                "multi-GPU Phase 4 with %d workers",
                                _nuc_ch_for_counting,
                                _phase4_num_gpus,
                            )
                        else:
                            from xldvp_seg.models.manager import ModelManager

                            _nuc_model_mgr = ModelManager()
                            _cp_model_for_nuc = _nuc_model_mgr.cellpose
                            _sam2_for_nuc = _nuc_model_mgr.sam2_predictor
                            logger.info(
                                "Nuclear counting enabled (single-process fallback): "
                                "nuc_channel=%s",
                                _nuc_ch_for_counting,
                            )
                    else:
                        logger.warning(
                            "--count-nuclei requires --nuc-channel-for-counting or "
                            "--channel-spec with nuc=... to identify the nuclear channel"
                        )

                _shm_name_for_phase4 = (
                    shm_manager.slide_info[slide_name]["shm_name"]
                    if (_use_shm and _phase4_num_gpus >= 1)
                    else None
                )

                process_detections_post_dedup(
                    all_detections,
                    tiles_dir,
                    pixel_size_um,
                    mask_filename=mask_fn,
                    slide_shm_arr=slide_shm_arr if _use_shm else None,
                    ch_to_slot=ch_to_slot if _use_shm else None,
                    x_start=x_start,
                    y_start=y_start,
                    loader=loader,
                    ch_indices=sorted(all_channel_data.keys()),
                    tile_size=args.tile_size,
                    contour_processing=_want_contour,
                    dilation_um=getattr(args, "dilation_um", 0.5),
                    rdp_epsilon=getattr(args, "rdp_epsilon", 5.0),
                    background_correction=_want_bg,
                    bg_neighbors=getattr(args, "bg_neighbors", 30),
                    count_nuclei=_count_nuclei,
                    nuc_channel_idx=_nuc_ch_for_counting,
                    min_nuclear_area=getattr(args, "min_nuclear_area", 50),
                    cellpose_model=_cp_model_for_nuc,
                    sam2_predictor=_sam2_for_nuc,
                    num_gpus=_phase4_num_gpus,
                    shm_name=_shm_name_for_phase4,
                )

                # Cleanup nuclear counting models (free GPU memory before HTML generation)
                if _nuc_model_mgr is not None:
                    _nuc_model_mgr.cleanup()
                    _nuc_model_mgr = None
                    _cp_model_for_nuc = None
                    _sam2_for_nuc = None

                # Checkpoint: save post-processed detections
                _postdedup_file = slide_output_dir / f"{args.cell_type}_detections_postdedup.json"
                atomic_json_dump(all_detections, _postdedup_file)
                logger.info(
                    f"Checkpoint: saved {len(all_detections)} post-dedup detections to {_postdedup_file.name}"
                )
            elif _has_bg and _has_contour:
                logger.info("Detections already post-processed — skipping")

            # ---- Built-in marker SNR classification (resume path) ----
            _apply_marker_snr_classification(args, all_detections, slide_output_dir)

            # Regenerate HTML if needed (requires CZI + tile masks)
            all_samples = []
            if not skip_html:
                # Ensure all channels are loaded for HTML generation
                if args.all_channels or (
                    args.cell_type == "cell" and getattr(args, "cellpose_input_channels", None)
                ):
                    try:
                        dims = loader.reader.get_dims_shape()[0]
                        _n_ch = dims.get("C", (0, 3))[1]
                    except Exception:
                        _n_ch = 3
                    for ch in range(_n_ch):
                        if ch not in all_channel_data:
                            logger.info(f"  Loading channel {ch} for HTML generation (resume)...")
                            loader.load_channel(ch)
                            all_channel_data[ch] = loader.get_channel_data(ch)

                logger.info(
                    f"Regenerating HTML for {len(all_detections)} detections from saved tiles..."
                )
                all_samples = _resume_generate_html_samples(
                    args,
                    all_detections,
                    tiles_dir,
                    all_channel_data,
                    loader,
                    pixel_size_um,
                    slide_name,
                    x_start,
                    y_start,
                )
                logger.info(f"Generated {len(all_samples)} HTML samples from resume path")

            # Subsample HTML by fraction if configured
            _html_frac = args.html_sample_fraction
            if _html_frac > 0 and len(all_samples) > 0 and len(all_detections) > 0:
                import random

                target = max(100, int(len(all_detections) * _html_frac))
                if len(all_samples) > target:
                    logger.info(
                        f"HTML sample fraction {_html_frac}: subsampling {len(all_samples)} -> {target}"
                    )
                    all_samples = random.sample(all_samples, target)

            # Run the same post-processing as the normal path (HTML export, CSV, summary)
            _finish_pipeline(
                args,
                all_detections,
                all_samples,
                slide_output_dir,
                tiles_dir,
                pixel_size_um,
                slide_name,
                mosaic_info,
                run_timestamp,
                pct,
                skip_html=skip_html,
                resumed=True,
                all_tiles=all_tiles,
                tissue_tiles=tissue_tiles,
                sampled_tiles=sampled_tiles,
            )
        finally:
            _maybe_export_ome_zarr(
                args,
                slide_shm_arr,
                ch_to_slot,
                pixel_size_um,
                slide_output_dir,
                slide_name,
                czi_path,
            )
            shm_manager.cleanup()
        return

    # ---- Normal path: full detection pipeline ----
    _det = run_detection_loop(
        args=args,
        sampled_tiles=sampled_tiles,
        slide_shm_arr=slide_shm_arr,
        ch_to_slot=ch_to_slot,
        shm_manager=shm_manager,
        x_start=x_start,
        y_start=y_start,
        slide_output_dir=slide_output_dir,
        tiles_dir=tiles_dir,
        pixel_size_um=pixel_size_um,
        slide_name=slide_name,
        mosaic_info=mosaic_info,
        tissue_channel=tissue_channel,
        variance_threshold=variance_threshold,
        ch_list=ch_list,
        loader=loader,
        h=h,
        w=w,
    )
    if _det["exit_early"]:
        return
    all_detections = _det["all_detections"]
    all_samples = _det["all_samples"]
    params = _det["params"]
    detector = _det["detector"]
    classifier_loaded = _det["classifier_loaded"]
    is_multiscale = _det["is_multiscale"]

    # Everything below uses SHM — wrap in try/finally to ensure cleanup on crash
    try:

        # Deduplication: tile overlap causes same detection in adjacent tiles
        # Uses actual mask pixel overlap (loads HDF5 mask files) for accurate dedup
        # Skip for multiscale -- already deduped by contour IoU in merge_detections_across_scales()
        if not is_multiscale and args.tile_overlap > 0 and len(all_detections) > 0:
            pre_dedup = len(all_detections)
            mask_fn = f"{args.cell_type}_masks.h5"
            dedup_sort = "confidence" if getattr(args, "dedup_by_confidence", False) else "area"
            if getattr(args, "dedup_method", "mask_overlap") == "iou_nms":
                from xldvp_seg.processing.deduplication import deduplicate_by_iou_nms

                all_detections = deduplicate_by_iou_nms(
                    all_detections,
                    tiles_dir,
                    iou_threshold=getattr(args, "iou_threshold", 0.2),
                    mask_filename=mask_fn,
                    sort_by=dedup_sort,
                )
            else:
                from xldvp_seg.processing.deduplication import deduplicate_by_mask_overlap

                all_detections = deduplicate_by_mask_overlap(
                    all_detections,
                    tiles_dir,
                    min_overlap_fraction=0.1,
                    mask_filename=mask_fn,
                    sort_by=dedup_sort,
                )

            # Filter HTML samples to match deduped detections and remove duplicate UIDs
            deduped_uids = {det.get("uid", det.get("id", "")) for det in all_detections}
            seen_uids = set()
            unique_samples = []
            for s in all_samples:
                uid = s.get("uid", "")
                if uid in deduped_uids and uid not in seen_uids:
                    seen_uids.add(uid)
                    unique_samples.append(s)
            logger.info(
                f"Dedup: {len(all_samples)} HTML samples -> {len(unique_samples)} (removed {len(all_samples) - len(unique_samples)} duplicate UIDs)"
            )
            all_samples = unique_samples

        # ---- Post-dedup: contour dilation + feature re-extraction + bg correction ----
        if len(all_detections) > 0 and (args.contour_processing or args.background_correction):
            from xldvp_seg.pipeline.post_detection import process_detections_post_dedup

            mask_fn = f"{args.cell_type}_masks.h5"

            # Nuclear counting (normal detection path).
            # Multi-GPU Phase 4: workers load Cellpose/SAM2; no main-process preload.
            _count_nuclei_normal = getattr(args, "count_nuclei", False)
            _nuc_ch_normal = getattr(args, "nuc_channel_for_counting", None)
            _cp_model_normal = None
            _sam2_normal = None
            _nuc_mgr_normal = None
            _phase4_num_gpus_normal = int(getattr(args, "num_gpus", 0))
            if _count_nuclei_normal:
                if _nuc_ch_normal is None:
                    cp_input = getattr(args, "cellpose_input_channels", None)
                    if cp_input:
                        parts = str(cp_input).split(",")
                        if len(parts) >= 2:
                            try:
                                _nuc_ch_normal = int(parts[1].strip())
                            except ValueError:
                                pass
                if _nuc_ch_normal is None and args.cell_type == "islet":
                    _nuc_ch_normal = getattr(args, "nuclear_channel", None)
                if _nuc_ch_normal is None and args.cell_type == "tissue_pattern":
                    _nuc_ch_normal = getattr(args, "tp_nuclear_channel", None)
                if _nuc_ch_normal is not None:
                    if _phase4_num_gpus_normal >= 1:
                        logger.info(
                            "Nuclear counting enabled: nuc_channel=%s, "
                            "multi-GPU Phase 4 with %d workers",
                            _nuc_ch_normal,
                            _phase4_num_gpus_normal,
                        )
                    else:
                        from xldvp_seg.models.manager import ModelManager

                        _nuc_mgr_normal = ModelManager()
                        _cp_model_normal = _nuc_mgr_normal.cellpose
                        _sam2_normal = _nuc_mgr_normal.sam2_predictor
                        logger.info(
                            "Nuclear counting enabled (single-process fallback): " "nuc_channel=%s",
                            _nuc_ch_normal,
                        )
                else:
                    logger.warning("--count-nuclei: could not determine nuclear channel")

            _shm_name_for_phase4 = (
                shm_manager.slide_info[slide_name]["shm_name"]
                if _phase4_num_gpus_normal >= 1
                else None
            )

            process_detections_post_dedup(
                all_detections,
                tiles_dir,
                pixel_size_um,
                mask_filename=mask_fn,
                slide_shm_arr=slide_shm_arr,
                ch_to_slot=ch_to_slot,
                x_start=x_start,
                y_start=y_start,
                contour_processing=args.contour_processing,
                dilation_um=getattr(args, "dilation_um", 0.5),
                rdp_epsilon=getattr(args, "rdp_epsilon", 5.0),
                background_correction=args.background_correction,
                bg_neighbors=getattr(args, "bg_neighbors", 30),
                count_nuclei=_count_nuclei_normal,
                nuc_channel_idx=_nuc_ch_normal,
                min_nuclear_area=getattr(args, "min_nuclear_area", 50),
                cellpose_model=_cp_model_normal,
                sam2_predictor=_sam2_normal,
                num_gpus=_phase4_num_gpus_normal,
                shm_name=_shm_name_for_phase4,
            )

            if _nuc_mgr_normal is not None:
                _nuc_mgr_normal.cleanup()

            # Checkpoint: save post-processed detections before HTML/CSV generation
            _postdedup_file = slide_output_dir / f"{args.cell_type}_detections_postdedup.json"
            atomic_json_dump(all_detections, _postdedup_file)
            logger.info(
                f"Checkpoint: saved {len(all_detections)} post-dedup detections to {_postdedup_file.name}"
            )

        # ---- Built-in marker SNR classification (normal path) ----
        _apply_marker_snr_classification(args, all_detections, slide_output_dir)

        # ---- Subsample HTML samples by fraction (after dedup, before export) ----
        _html_frac = args.html_sample_fraction
        if _html_frac > 0 and len(all_samples) > 0 and len(all_detections) > 0:
            import random

            target = max(100, int(len(all_detections) * _html_frac))
            if len(all_samples) > target:
                logger.info(
                    f"HTML sample fraction {_html_frac}: subsampling {len(all_samples)} -> {target} "
                    f"({_html_frac*100:.0f}% of {len(all_detections)} detections)"
                )
                all_samples = random.sample(all_samples, target)

        # ---- Shared post-processing: CSV, JSON, HTML, summary, server ----
        _finish_pipeline(
            args,
            all_detections,
            all_samples,
            slide_output_dir,
            tiles_dir,
            pixel_size_um,
            slide_name,
            mosaic_info,
            run_timestamp,
            pct,
            all_tiles=all_tiles,
            tissue_tiles=tissue_tiles,
            sampled_tiles=sampled_tiles,
            resumed=False,
            params=params,
            classifier_loaded=classifier_loaded,
            is_multiscale=is_multiscale,
            detector=detector,
        )
    finally:
        _maybe_export_ome_zarr(
            args, slide_shm_arr, ch_to_slot, pixel_size_um, slide_output_dir, slide_name, czi_path
        )
        # All SHM-dependent work is done — safe to free shared memory
        shm_manager.cleanup()


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
        parser.error(
            "--czi-path is required (unless using --stop-server, --server-status, or --show-metadata)"
        )

    # Require --cell-type if not showing metadata
    if args.cell_type is None:
        parser.error("--cell-type is required unless using --show-metadata")

    args = postprocess_args(args, parser)
    run_pipeline(args)


if __name__ == "__main__":
    main()
