"""Detection loop: GPU tile processing for the normal (non-resume) pipeline path.

Encapsulates the full-detection block from run_pipeline():
  - Detector/strategy/classifier initialization
  - Islet GMM marker calibration
  - Multi-scale vessel detection (optional)
  - Regular per-tile MultiGPUTileProcessor loop
  - Deferred islet HTML generation
  - Tile coverage verification
  - Cross-tile vessel merge
  - Detection-only early exit

The main function ``run_detection_loop`` delegates to four sub-functions:
  - ``_initialize_detector`` — detector, classifiers, strategy params, islet GMM
  - ``_run_multiscale_tiles`` — multi-scale vessel detection loop
  - ``_run_regular_tiles`` — standard per-tile processing with resume + deferred islet HTML
  - ``_postprocess_detections`` — tile coverage, cross-tile vessel merge, detection-only exit
"""

import gc
from pathlib import Path

import h5py
import numpy as np

from xldvp_seg.detection.cell_detector import CellDetector
from xldvp_seg.io.html_export import image_to_base64, percentile_normalize
from xldvp_seg.pipeline.detection_setup import (
    apply_vessel_classifiers,
    build_detection_params,
    load_classifier_into_detector,
    load_vessel_classifiers,
)
from xldvp_seg.pipeline.resume import (
    compose_tile_rgb,
    compute_and_apply_islet_markers,
    reload_detections_from_tiles,
)
from xldvp_seg.pipeline.samples import (
    _compute_tile_percentiles,
    calibrate_islet_marker_gmm,
    filter_and_create_html_samples,
)
from xldvp_seg.processing.multigpu_worker import MultiGPUTileProcessor
from xldvp_seg.processing.strategy_factory import create_strategy
from xldvp_seg.utils.device import empty_cache
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sub-function 1: Detector / classifier / strategy initialization
# ---------------------------------------------------------------------------


def _initialize_detector(
    args,
    loader,
    n_channels,
    ch_to_slot,
    slide_shm_arr,
    sampled_tiles,
    pixel_size_um,
    shm_manager,
    x_start,
    y_start,
    tile_size,
    ch_keys,
    slide_name,
):
    """Create CellDetector, resolve classifiers, build strategy params, islet GMM calibration.

    Returns a dict with keys:
        detector, classifier_loaded, params, strategy_params, classifier_path,
        is_multiscale, vessel_classifier, vessel_type_classifier, extract_deep,
        num_gpus, mgpu_slide_info, mgpu_cd31_channel, mgpu_channel_names.
    """
    logger.info("Initializing detector...")
    detector = CellDetector()

    classifier_loaded = load_classifier_into_detector(args, detector)

    # Auto-detect annotation run: no classifier -> show ALL candidates in HTML
    if (
        args.cell_type in ("nmj", "islet", "tissue_pattern")
        and not classifier_loaded
        and args.html_score_threshold > 0
    ):
        logger.info(
            f"No classifier loaded -- annotation run detected. "
            f"Overriding --html-score-threshold from {args.html_score_threshold} to 0.0 "
            f"(will show ALL candidates for annotation)"
        )
        args.html_score_threshold = 0.0

    params = build_detection_params(args, pixel_size_um)
    params["segmenter"] = getattr(args, "segmenter", "cellpose")
    logger.info(f"Detection params: {params}")

    strategy = create_strategy(
        cell_type=args.cell_type,
        strategy_params=params,
        extract_deep_features=params.get("extract_deep_features", False),
        extract_sam2_embeddings=params.get("extract_sam2_embeddings", True),
        pixel_size_um=pixel_size_um,
    )
    logger.info(f"Using {strategy.name} strategy: {strategy.get_config()}")

    vessel_classifier, vessel_type_classifier = load_vessel_classifiers(args)

    is_multiscale = args.cell_type == "vessel" and getattr(args, "multi_scale", False)

    num_gpus = getattr(args, "num_gpus", 1)

    if n_channels < 2:
        logger.warning("Pipeline works best with --all-channels for multi-channel features")

    # --- Islet marker calibration (pilot phase, using shared memory) ---
    islet_gmm_thresholds = {}
    if args.cell_type == "islet":
        marker_chs_str = getattr(args, "islet_marker_channels", "gcg:2,ins:3,sst:5")
        try:
            marker_chs_list = [int(pair.split(":")[1]) for pair in marker_chs_str.split(",")]
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid --islet-marker-channels format: '{marker_chs_str}'. "
                f"Expected 'name:ch_idx,...' e.g. 'gcg:2,ins:3,sst:5'. Error: {e}"
            )
        n_pilot = max(1, int(len(sampled_tiles) * 0.05))
        pilot_indices = np.random.choice(len(sampled_tiles), n_pilot, replace=False)
        pilot_tiles = [sampled_tiles[i] for i in pilot_indices]
        nuclei_only = getattr(args, "nuclei_only", False)
        nuc_ch = getattr(args, "nuclear_channel", 4)
        mem_ch = nuc_ch if nuclei_only else getattr(args, "membrane_channel", 1)
        islet_gmm_thresholds = calibrate_islet_marker_gmm(
            pilot_tiles=pilot_tiles,
            loader=loader,
            all_channel_data=None,
            slide_shm_arr=slide_shm_arr,
            ch_to_slot=ch_to_slot,
            marker_channels=marker_chs_list,
            membrane_channel=mem_ch,
            nuclear_channel=nuc_ch,
            tile_size=tile_size,
            pixel_size_um=pixel_size_um,
            nuclei_only=nuclei_only,
            mosaic_origin=(x_start, y_start),
        )

    # Build strategy parameters from the already-constructed params dict
    strategy_params = dict(params)
    if islet_gmm_thresholds:
        strategy_params["gmm_prefilter_thresholds"] = islet_gmm_thresholds

    # Get classifier path
    classifier_path = None
    if args.cell_type == "nmj":
        classifier_path = getattr(args, "nmj_classifier", None)
        if classifier_path:
            logger.info(f"Using specified NMJ classifier: {classifier_path}")
        else:
            logger.info(
                "No --nmj-classifier specified -- will return all candidates (annotation run)"
            )
    elif args.cell_type == "islet":
        classifier_path = getattr(args, "islet_classifier", None)
        if classifier_path:
            logger.info(f"Using specified islet classifier: {classifier_path}")
        else:
            logger.info(
                "No --islet-classifier specified -- will return all candidates (annotation run)"
            )
    elif args.cell_type == "tissue_pattern":
        classifier_path = getattr(args, "tp_classifier", None)
        if classifier_path:
            logger.info(f"Using specified tissue_pattern classifier: {classifier_path}")
        else:
            logger.info(
                "No --tp-classifier specified -- will return all candidates (annotation run)"
            )

    extract_deep = getattr(args, "extract_deep_features", False)

    # Vessel-specific params for multi-GPU
    mgpu_cd31_channel = getattr(args, "cd31_channel", None) if args.cell_type == "vessel" else None
    mgpu_channel_names = None
    if args.cell_type == "vessel" and getattr(args, "channel_names", None):
        names = args.channel_names.split(",")
        mgpu_channel_names = {
            ch_keys[i]: name.strip() for i, name in enumerate(names) if i < len(ch_keys)
        }

    # Add mosaic origin to slide_info so workers can convert global->relative coords
    mgpu_slide_info = shm_manager.get_slide_info()
    mgpu_slide_info[slide_name]["mosaic_origin"] = (x_start, y_start)

    return {
        "detector": detector,
        "classifier_loaded": classifier_loaded,
        "params": params,
        "strategy_params": strategy_params,
        "classifier_path": classifier_path,
        "is_multiscale": is_multiscale,
        "vessel_classifier": vessel_classifier,
        "vessel_type_classifier": vessel_type_classifier,
        "extract_deep": extract_deep,
        "num_gpus": num_gpus,
        "mgpu_slide_info": mgpu_slide_info,
        "mgpu_cd31_channel": mgpu_cd31_channel,
        "mgpu_channel_names": mgpu_channel_names,
    }


# ---------------------------------------------------------------------------
# Sub-function 2: Multi-scale vessel detection loop
# ---------------------------------------------------------------------------


def _run_multiscale_tiles(processor, args, ctx, init, sampled_tiles):
    """Run the multi-scale vessel detection loop.

    Args:
        processor: Active MultiGPUTileProcessor (inside ``with`` block).
        args: Parsed CLI namespace.
        ctx: Shared geometry/SHM context dict.
        init: Dict returned by ``_initialize_detector``.
        sampled_tiles: List of tile dicts.

    Returns:
        (all_detections, all_samples) — both lists.
    """
    from tqdm import tqdm as tqdm_progress

    from xldvp_seg.utils.multiscale import (
        convert_detection_to_full_res,
        generate_tile_grid_at_scale,
        get_scale_params,
        merge_detections_across_scales,
    )

    slide_shm_arr = ctx["slide_shm_arr"]
    ch_to_slot = ctx["ch_to_slot"]
    pixel_size_um = ctx["pixel_size_um"]
    x_start = ctx["x_start"]
    y_start = ctx["y_start"]
    slide_name = ctx["slide_name"]
    slide_output_dir = ctx["slide_output_dir"]
    h = ctx["h"]
    w = ctx["w"]
    tile_size = ctx["tile_size"]

    vessel_classifier = init["vessel_classifier"]
    vessel_type_classifier = init["vessel_type_classifier"]
    num_gpus = init["num_gpus"]

    scales = [int(s.strip()) for s in args.scales.split(",")]
    iou_threshold = getattr(args, "multiscale_iou_threshold", 0.3)

    logger.info("=" * 60)
    logger.info(f"MULTI-SCALE VESSEL DETECTION -- {num_gpus} GPU(s)")
    logger.info("=" * 60)
    logger.info(f"Scales: {scales} (coarse to fine)")
    logger.info(f"IoU threshold for deduplication: {iou_threshold}")

    all_scale_detections = []  # Accumulate full-res detections across scales
    total_tiles_submitted = 0

    # Resume from checkpoints if available
    completed_scales = set()
    if getattr(args, "resume_from", None):
        checkpoint_dir = Path(args.resume_from) / "checkpoints"
        if checkpoint_dir.exists():
            # Sort by modification time (not lexicographic -- scale_8x > scale_16x lex)
            checkpoint_files = sorted(
                checkpoint_dir.glob("scale_*x.json"),
                key=lambda p: p.stat().st_mtime,
            )
            if checkpoint_files:
                latest = checkpoint_files[-1]
                all_scale_detections = fast_json_load(latest)
                # Restore numpy arrays for contours (json.load produces lists)
                for det in all_scale_detections:
                    for key in ("outer", "inner", "outer_contour", "inner_contour"):
                        if key in det and det[key] is not None:
                            det[key] = np.array(det[key], dtype=np.int32)
                for cf in checkpoint_files:
                    # Parse scale from filename like "scale_32x.json"
                    try:
                        s = int(cf.stem.split("_")[1].rstrip("x"))
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
            ctx["mosaic_width"],
            ctx["mosaic_height"],
            tile_size,
            scale,
            overlap=0,
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
                "x": x_start + tx_s * scale,
                "y": y_start + ty_s * scale,
                "w": tile_size * scale,
                "h": tile_size * scale,
                "scale_factor": scale,
                "scale_params": scale_params,
                "tile_x_scaled": tx_s,
                "tile_y_scaled": ty_s,
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

            if result["status"] == "success":
                try:
                    tile_dict = result["tile"]
                    features_list = result["features_list"]
                    sf = result.get("scale_factor", scale)
                    tx_s = tile_dict.get("tile_x_scaled", 0)
                    ty_s = tile_dict.get("tile_y_scaled", 0)

                    # Apply vessel classifiers
                    apply_vessel_classifiers(
                        features_list, vessel_classifier, vessel_type_classifier
                    )

                    # Convert each detection from downscaled-local to full-res global
                    for feat in features_list:
                        if "outer_contour" in feat and "outer" not in feat:
                            feat["outer"] = feat.pop("outer_contour")
                        if "inner_contour" in feat and "inner" not in feat:
                            feat["inner"] = feat.pop("inner_contour")
                        if feat.get("features", {}).get("detection_type") == "arc":
                            feat["is_arc"] = True

                        det_fullres = convert_detection_to_full_res(
                            feat,
                            sf,
                            tx_s,
                            ty_s,
                            smooth=True,
                            smooth_base_factor=getattr(args, "smooth_contours_factor", 3.0),
                        )

                        # Add mosaic origin for CZI-global coords
                        # convert_detection_to_full_res already scaled
                        # det['center'], det['centroid'], feats['outer_center'],
                        # feats['inner_center'] to full-res array-local coords.
                        # feats['center'] is NOT scaled by that function, so
                        # scale it here first to match.
                        feats_d = det_fullres.get("features", {})
                        if isinstance(feats_d, dict):
                            if "center" in feats_d and feats_d["center"] is not None:
                                fc = feats_d["center"]
                                feats_d["center"] = [
                                    (fc[0] + tx_s) * sf,
                                    (fc[1] + ty_s) * sf,
                                ]
                        for key in ("center", "centroid"):
                            if key in det_fullres:
                                det_fullres[key] = list(det_fullres[key])  # defensive copy
                                det_fullres[key][0] += x_start
                                det_fullres[key][1] += y_start
                        if isinstance(feats_d, dict):
                            for ck in ("center", "outer_center", "inner_center"):
                                if ck in feats_d and feats_d[ck] is not None:
                                    feats_d[ck][0] += x_start
                                    feats_d[ck][1] += y_start
                        mosaic_offset = np.array([x_start, y_start], dtype=np.int32)
                        if "outer" in det_fullres and det_fullres["outer"] is not None:
                            det_fullres["outer"] = det_fullres["outer"] + mosaic_offset
                        if "inner" in det_fullres and det_fullres["inner"] is not None:
                            det_fullres["inner"] = det_fullres["inner"] + mosaic_offset

                        for ckey in ("outer", "inner"):
                            if ckey in det_fullres and det_fullres[ckey] is not None:
                                det_fullres[f"{ckey}_contour"] = det_fullres[ckey]
                                det_fullres[f"{ckey}_contour_global"] = [
                                    [int(pt[0][0]), int(pt[0][1])] for pt in det_fullres[ckey]
                                ]

                        det_fullres["scale_detected"] = sf
                        all_scale_detections.append(det_fullres)
                        scale_det_count += 1

                except Exception as e:
                    import traceback

                    _tid = result.get("tid", "?")
                    logger.error(f"Error post-processing multiscale tile {_tid}: {e}")
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

            elif result["status"] == "error":
                logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

        pbar.close()
        total_tiles_submitted += results_collected
        logger.info(f"Scale 1/{scale}x: {scale_det_count} detections")

        gc.collect()
        empty_cache()

        # Save checkpoint after each scale
        checkpoint_dir = slide_output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / f"scale_{scale}x.json"
        atomic_json_dump(all_scale_detections, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file} ({len(all_scale_detections)} detections)")

    # Merge across scales (contour-based IoU dedup)
    logger.info(f"Merging {len(all_scale_detections)} detections across scales...")
    merged_detections = merge_detections_across_scales(
        all_scale_detections,
        iou_threshold=iou_threshold,
        tile_size=tile_size,
    )
    logger.info(f"After merge: {len(merged_detections)} vessels")

    # Regenerate UIDs from full-res global coords and build all_detections
    all_detections = []
    for det in merged_detections:
        features_dict = det.get("features", {})
        center = features_dict.get("center", det.get("center", [0, 0]))
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            cx, cy = int(center[0]), int(center[1])
        else:
            cx, cy = 0, 0

        uid = f"{slide_name}_vessel_{cx}_{cy}"
        det["uid"] = uid
        det["slide"] = slide_name
        det["center"] = [cx, cy]
        det["center_um"] = [cx * pixel_size_um, cy * pixel_size_um]
        det["global_center"] = [cx, cy]
        det["global_center_um"] = [cx * pixel_size_um, cy * pixel_size_um]
        all_detections.append(det)

    # Generate HTML crops from shared memory with percentile normalization
    logger.info(f"Generating HTML crops for {len(all_detections)} multiscale detections...")
    all_samples = []

    for det in all_detections:
        features_dict = det.get("features", {})
        cx, cy = det["center"]

        diameter_um = features_dict.get("outer_diameter_um", 50)
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

        crop_rgb = compose_tile_rgb(
            None,
            (y1, x1, y2 - y1, x2 - x1),
            args.cell_type,
            slide_shm_arr=slide_shm_arr,
            ch_to_slot=ch_to_slot,
        )

        if crop_rgb.size == 0:
            continue

        # Percentile normalize (not /256) for proper dynamic range
        crop_rgb = percentile_normalize(crop_rgb, p_low=1, p_high=99.5)

        b64_str, _ = image_to_base64(crop_rgb)
        sample = {
            "uid": det["uid"],
            "image": b64_str,
            "stats": features_dict,
        }
        all_samples.append(sample)

    logger.info(
        f"Multi-scale mode: {len(all_detections)} detections, {len(all_samples)} HTML samples "
        f"from {total_tiles_submitted} tiles on {num_gpus} GPUs"
    )

    return all_detections, all_samples


# ---------------------------------------------------------------------------
# Sub-function 3: Regular (non-multiscale) tile processing
# ---------------------------------------------------------------------------


def _run_regular_tiles(processor, args, ctx, init, sampled_tiles, all_samples_ref):
    """Run standard per-tile processing with resume, deferred islet HTML, etc.

    Args:
        processor: Active MultiGPUTileProcessor (inside ``with`` block).
        args: Parsed CLI namespace.
        ctx: Shared geometry/SHM context dict.
        init: Dict returned by ``_initialize_detector``.
        sampled_tiles: List of tile dicts.
        all_samples_ref: Mutable list to extend with HTML samples.

    Returns:
        (all_detections, all_samples, collected_partial_vessels)
    """
    slide_shm_arr = ctx["slide_shm_arr"]
    ch_to_slot = ctx["ch_to_slot"]
    pixel_size_um = ctx["pixel_size_um"]
    x_start = ctx["x_start"]
    y_start = ctx["y_start"]
    tiles_dir = ctx["tiles_dir"]
    slide_name = ctx["slide_name"]
    tile_size = ctx["tile_size"]

    params = init["params"]
    vessel_classifier = init["vessel_classifier"]
    vessel_type_classifier = init["vessel_type_classifier"]
    num_gpus = init["num_gpus"]

    all_detections = []
    all_samples = all_samples_ref
    deferred_html_tiles = []
    collected_partial_vessels = {}

    logger.info("=" * 60)
    logger.info(f"{args.cell_type.upper()} DETECTION -- {num_gpus} GPU(s)")
    logger.info("=" * 60)

    # Filter to incomplete tiles on resume (skip tiles with existing features)
    tiles_to_process = sampled_tiles
    if getattr(args, "resume", None) and tiles_dir.exists():
        completed = 0
        remaining = []
        feat_name = f"{args.cell_type}_features.json"
        for tile in sampled_tiles:
            tile_feat = tiles_dir / f"tile_{tile['x']}_{tile['y']}" / feat_name
            if tile_feat.exists():
                completed += 1
            else:
                remaining.append(tile)
        if completed > 0:
            logger.info(
                f"Resume: {completed}/{len(sampled_tiles)} tiles already completed, "
                f"processing {len(remaining)} remaining"
            )
            tiles_to_process = remaining
        if not remaining:
            logger.info("All tiles already completed — skipping to dedup")
            # Reload from tiles instead of re-processing
            all_detections = reload_detections_from_tiles(tiles_dir, args.cell_type)

    # Submit tiles (add tile dimensions for worker)
    logger.info(f"Submitting {len(tiles_to_process)} tiles to {num_gpus} GPUs...")
    for tile in tiles_to_process:
        # Worker expects 'x', 'y', 'w', 'h' keys
        tile_with_dims = {
            "x": tile["x"],
            "y": tile["y"],
            "w": tile_size,
            "h": tile_size,
        }
        processor.submit_tile(slide_name, tile_with_dims)

    # Collect results with progress bar
    from tqdm import tqdm as tqdm_progress

    pbar = tqdm_progress(total=len(tiles_to_process), desc="Processing tiles")

    results_collected = 0
    while results_collected < len(tiles_to_process):
        result = processor.collect_result(timeout=14400)  # 4h timeout per tile
        if result is None:
            logger.warning("Timeout waiting for result")
            break

        results_collected += 1
        pbar.update(1)

        if result["status"] == "success":
            try:
                tile = result["tile"]
                tile_x, tile_y = tile["x"], tile["y"]
                features_list = result["features_list"]

                # Collect partial vessels from worker for cross-tile merge
                pv = result.get("partial_vessels")
                if pv:
                    for tk, pvl in pv.items():
                        if tk not in collected_partial_vessels:
                            collected_partial_vessels[tk] = []
                        collected_partial_vessels[tk].extend(pvl)

                # Masks: worker saves to disk when tiles_dir provided,
                # sends None through Queue to keep it lightweight.
                # Read from disk for HTML generation.
                tile_id = f"tile_{tile_x}_{tile_y}"
                tile_out = tiles_dir / tile_id
                masks = result["masks"]

                if masks is None and len(features_list) > 0:
                    # Worker saved to disk — read back for HTML
                    masks_file = tile_out / f"{args.cell_type}_masks.h5"
                    if masks_file.exists():
                        with h5py.File(masks_file, "r") as hf:
                            masks = hf["masks"][:]

                # Skip tiles with no detections
                if masks is None or len(features_list) == 0:
                    continue

                # Apply vessel classifier post-processing BEFORE saving
                if args.cell_type == "vessel":
                    apply_vessel_classifiers(
                        features_list,
                        vessel_classifier,
                        vessel_type_classifier,
                    )

                # Save tile masks (skip if worker already saved)
                tile_out.mkdir(exist_ok=True)
                masks_file = tile_out / f"{args.cell_type}_masks.h5"
                if not masks_file.exists():
                    from xldvp_seg.io.html_export import create_hdf5_dataset

                    with h5py.File(masks_file, "w") as f:
                        create_hdf5_dataset(f, "masks", masks)

                # Save features (includes vessel classification if applicable)
                atomic_json_dump(
                    features_list,
                    tile_out / f"{args.cell_type}_features.json",
                )

                # Add detections to global list
                for feat in features_list:
                    all_detections.append(feat)

                # Create samples for HTML
                # Convert global tile coords to 0-based array indices
                rel_tx = tile_x - x_start
                rel_ty = tile_y - y_start
                # Use masks.shape to handle edge tiles (smaller at boundaries)
                tile_h, tile_w = masks.shape[:2]
                # Read HTML crops from shared memory
                _disp_chs = None
                if args.cell_type == "islet":
                    _disp_chs = getattr(args, "islet_display_chs", None)
                elif args.cell_type == "tissue_pattern":
                    _disp_chs = getattr(args, "tp_display_channels_list", [0, 3, 1])
                tile_rgb_html = compose_tile_rgb(
                    _disp_chs,
                    (rel_ty, rel_tx, tile_h, tile_w),
                    args.cell_type,
                    slide_shm_arr=slide_shm_arr,
                    ch_to_slot=ch_to_slot,
                )

                tile_pct = (
                    _compute_tile_percentiles(tile_rgb_html)
                    if getattr(args, "html_normalization", "crop") == "tile"
                    else None
                )

                if args.cell_type == "islet":
                    # Flush tile data to disk to avoid OOM
                    np.save(tile_out / "tile_rgb_html.npy", tile_rgb_html)
                    if tile_pct is not None:
                        atomic_json_dump(tile_pct, tile_out / "tile_pct.json")
                    deferred_html_tiles.append(
                        {
                            "tile_dir": str(tile_out),
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "tile_pct": tile_pct,
                        }
                    )
                    del masks, tile_rgb_html, features_list
                    result.pop("masks", None)
                    result["features_list"] = None
                    gc.collect()
                else:
                    _max_html = args.max_html_samples
                    if _max_html > 0 and len(all_samples) >= _max_html:
                        if len(all_samples) == _max_html:
                            logger.info(
                                f"HTML sample cap reached ({_max_html}). "
                                f"Remaining tiles will not have HTML crops. "
                                f"Use --max-html-samples 0 for unlimited."
                            )
                    else:
                        html_samples = filter_and_create_html_samples(
                            features_list,
                            tile_x,
                            tile_y,
                            tile_rgb_html,
                            masks,
                            pixel_size_um,
                            slide_name,
                            args.cell_type,
                            args.html_score_threshold,
                            tile_percentiles=tile_pct,
                            candidate_mode=args.candidate_mode,
                            vessel_params=(params if args.cell_type == "vessel" else None),
                        )
                        all_samples.extend(html_samples)
                        # Cache HTML samples to disk for fast resume
                        if html_samples:
                            _cache_path = tile_out / f"{args.cell_type}_html_samples.json"
                            atomic_json_dump(html_samples, _cache_path)

            except Exception as e:
                import traceback

                logger.error(f"Error post-processing tile ({tile_x}, {tile_y}): {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")

        elif result["status"] in ("empty", "no_tissue"):
            pass  # Normal - no tissue in tile
        elif result["status"] == "error":
            logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

    pbar.close()

    # Deferred HTML generation for islet (needs population-level marker thresholds)
    if deferred_html_tiles and args.cell_type == "islet":
        marker_thresholds, _islet_mm = compute_and_apply_islet_markers(args, all_detections)
        uid_to_marker = {d.get("uid", ""): d.get("marker_class") for d in all_detections}
        try:
            for dt in deferred_html_tiles:
                _td = Path(dt["tile_dir"])
                _tile_rgb = np.load(_td / "tile_rgb_html.npy")
                with h5py.File(_td / f"{args.cell_type}_masks.h5", "r") as _hf:
                    _tile_masks = _hf["masks"][:]
                _tile_feats = fast_json_load(_td / f"{args.cell_type}_features.json")
                for _feat in _tile_feats:
                    _mc = uid_to_marker.get(_feat.get("uid", ""))
                    if _mc:
                        _feat["marker_class"] = _mc
                html_samples = filter_and_create_html_samples(
                    _tile_feats,
                    dt["tile_x"],
                    dt["tile_y"],
                    _tile_rgb,
                    _tile_masks,
                    pixel_size_um,
                    slide_name,
                    args.cell_type,
                    args.html_score_threshold,
                    tile_percentiles=dt["tile_pct"],
                    marker_thresholds=marker_thresholds,
                    marker_map=_islet_mm,
                    candidate_mode=args.candidate_mode,
                    vessel_params=(params if args.cell_type == "vessel" else None),
                )
                all_samples.extend(html_samples)
                # Cache HTML samples to disk for fast resume
                if html_samples:
                    atomic_json_dump(
                        html_samples,
                        _td / f"{args.cell_type}_html_samples.json",
                    )
                # Clean up deferred temp files
                _npy_path = _td / "tile_rgb_html.npy"
                if _npy_path.exists():
                    _npy_path.unlink()
                _pct_path = _td / "tile_pct.json"
                if _pct_path.exists():
                    _pct_path.unlink()
                del _tile_rgb, _tile_masks, _tile_feats
                gc.collect()
        finally:
            # Ensure ALL deferred .npy files are cleaned up even on error
            for dt in deferred_html_tiles:
                _td = Path(dt["tile_dir"])
                for _tmp in ("tile_rgb_html.npy", "tile_pct.json"):
                    _p = _td / _tmp
                    if _p.exists():
                        _p.unlink()
            deferred_html_tiles = []
        gc.collect()

    logger.info(
        f"Processing complete: {len(all_detections)} {args.cell_type} detections "
        f"from {results_collected} tiles"
    )

    return all_detections, all_samples, collected_partial_vessels


# ---------------------------------------------------------------------------
# Sub-function 4: Post-detection processing
# ---------------------------------------------------------------------------


def _postprocess_detections(
    all_detections, sampled_tiles, args, collected_partial_vessels, ctx, is_multiscale, shm_manager
):
    """Coverage verification, cross-tile vessel merge, detection-only early exit.

    Args:
        all_detections: List of detection dicts (mutated in place for vessel merge).
        sampled_tiles: List of tile dicts (for coverage check).
        args: Parsed CLI namespace.
        collected_partial_vessels: Dict of partial vessels from workers.
        ctx: Shared geometry/SHM context dict.
        is_multiscale: Whether multi-scale mode was used.
        shm_manager: SharedSlideManager (cleanup on detection-only exit).

    Returns:
        dict with ``exit_early=True`` if detection_only mode, else ``None``.
    """
    pixel_size_um = ctx["pixel_size_um"]
    slide_name = ctx["slide_name"]
    tile_size = ctx["tile_size"]

    logger.info(f"Total detections (pre-dedup): {len(all_detections)}")

    # ---- Tile coverage verification ----
    # After ALL detection paths (single-node, multi-GPU, shard merge), verify
    # that every tissue tile was processed. Catches shard gaps, tile skips, crashes.
    tiles_processed = set()
    for det in all_detections:
        to = det.get("tile_origin")
        if to:
            tiles_processed.add(tuple(to))
    n_processed = len(tiles_processed)
    n_expected = len(sampled_tiles) if sampled_tiles else 0
    if n_expected > 0:
        coverage_pct = 100 * n_processed / n_expected
        logger.info(
            f"Tile coverage: {n_processed}/{n_expected} tiles with detections ({coverage_pct:.1f}%)"
        )
        if n_processed < n_expected:
            # Some tiles may legitimately have 0 detections (empty tissue), but
            # a large gap (>10%) suggests a real problem
            gap_pct = 100 * (n_expected - n_processed) / n_expected
            if gap_pct > 10:
                logger.warning(
                    f"LOW COVERAGE: {n_expected - n_processed} tissue tiles ({gap_pct:.1f}%) "
                    f"produced no detections. This may indicate detection issues."
                )
            elif gap_pct > 0:
                logger.info(
                    f"Note: {n_expected - n_processed} tissue tiles had no detections "
                    f"(normal for sparse tissue regions)"
                )

    # Cross-tile vessel merge: reconstruct partial vessels from all workers and merge
    if args.cell_type == "vessel" and not is_multiscale and collected_partial_vessels:
        from xldvp_seg.detection.strategies.vessel import VesselStrategy

        n_partials = sum(len(v) for v in collected_partial_vessels.values())
        n_tiles_with = len(collected_partial_vessels)
        logger.info(
            f"Cross-tile vessel merge: {n_partials} partial vessels from {n_tiles_with} tiles"
        )
        merge_strategy = VesselStrategy()
        merge_strategy.import_partial_vessels(collected_partial_vessels)
        tile_overlap_px = int(tile_size * args.tile_overlap)
        merged_vessels = merge_strategy.merge_cross_tile_vessels(
            tile_size=tile_size,
            overlap=tile_overlap_px,
            match_threshold=0.6,
        )
        if merged_vessels:
            logger.info(f"Cross-tile merge produced {len(merged_vessels)} merged vessels")
            # Add merged vessels as new detections with proper UIDs
            for mv in merged_vessels:
                outer = mv.get("outer")
                if outer is None:
                    continue
                pts = outer.reshape(-1, 2)
                cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                uid = f"{slide_name}_vessel_{int(round(cx))}_{int(round(cy))}"
                feat = mv.get("features", {})
                feat["center"] = [cx, cy]
                feat["is_cross_tile_merged"] = True
                feat["merge_score"] = mv.get("merge_score", 0)
                det = {
                    "uid": uid,
                    "slide_name": slide_name,
                    "global_center": [cx, cy],
                    "global_center_um": [cx * pixel_size_um, cy * pixel_size_um],
                    "features": feat,
                    "tile_origin": list(mv.get("source_tiles", [(0, 0)])[0]),
                    "is_cross_tile_merged": True,
                }
                all_detections.append(det)
            logger.info(f"Total detections after cross-tile merge: {len(all_detections)}")
        else:
            logger.info("Cross-tile merge: no matches found")
        del merge_strategy, collected_partial_vessels
        gc.collect()

    # Detection-only mode: skip dedup, HTML, CSV -- just save per-tile results and exit
    if getattr(args, "detection_only", False):
        logger.info(
            f"Detection-only mode: {len(all_detections)} detections saved to tile dirs. Exiting."
        )
        if getattr(args, "tile_shard", None):
            shard_idx, shard_total = args.tile_shard
            logger.info(f"Shard {shard_idx}/{shard_total} complete.")
        shm_manager.cleanup()
        return {"exit_early": True}

    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_detection_loop(
    args,
    sampled_tiles,
    slide_shm_arr,
    ch_to_slot,
    shm_manager,
    x_start,
    y_start,
    slide_output_dir,
    tiles_dir,
    pixel_size_um,
    slide_name,
    mosaic_info,
    tissue_channel,
    variance_threshold,
    ch_list,
    loader,
    h,
    w,
):
    """Run the full detection pipeline on the normal (non-resume) path.

    Initializes detector, strategy, and classifiers, then launches the
    MultiGPUTileProcessor loop.  Also handles:
      - Islet GMM marker calibration (pilot phase)
      - Multi-scale vessel detection
      - Deferred islet HTML generation
      - Tile coverage verification
      - Cross-tile vessel merging
      - Detection-only early exit (--detection-only / --tile-shard)

    When ``detection_only`` is True (or a tile shard finishes), the function
    calls ``shm_manager.cleanup()`` itself and returns ``{"exit_early": True}``.
    The orchestrator must check this flag and return immediately.

    Args:
        args: Parsed CLI namespace.
        sampled_tiles: List of tile dicts (keys: x, y) to process.
        slide_shm_arr: Shared-memory array (h, w, n_channels) of dtype uint16.
        ch_to_slot: Dict mapping CZI channel index -> SHM slot.
        shm_manager: SharedSlideManager; caller retains ownership (cleanup on exit).
        x_start: Mosaic x origin (pixels).
        y_start: Mosaic y origin (pixels).
        slide_output_dir: Path to this run's output directory.
        tiles_dir: Path to tiles/ subdirectory.
        pixel_size_um: Pixel size in microns.
        slide_name: CZI stem name (used for UIDs and logging).
        mosaic_info: Dict with keys x, y, width, height.
        tissue_channel: CZI channel index used for tissue detection.
        variance_threshold: Calibrated tissue variance threshold.
        ch_list: Sorted list of CZI channel indices in SHM slot order.
        loader: CZI loader (channel refs cleared at start of this function).
        h: Slide height in pixels.
        w: Slide width in pixels.

    Returns:
        dict with keys:
            exit_early        - True if the pipeline should return immediately
                                (detection_only mode or tile shard finished).
            all_detections    - List of detection dicts with global coordinates.
            all_samples       - List of HTML sample dicts.
            params            - Detection params dict (needed by _finish_pipeline).
            detector          - CellDetector instance (needed by _finish_pipeline).
            classifier_loaded - Bool (needed by _finish_pipeline).
            is_multiscale     - Bool (vessel multi-scale mode).
    """
    slide_output_dir = Path(slide_output_dir)
    tiles_dir = Path(tiles_dir)

    n_channels = len(ch_to_slot)
    ch_keys = ch_list  # CZI channel indices in SHM slot order
    tile_size = args.tile_size

    # Build shared context dict for sub-functions
    ctx = {
        "slide_shm_arr": slide_shm_arr,
        "ch_to_slot": ch_to_slot,
        "pixel_size_um": pixel_size_um,
        "x_start": x_start,
        "y_start": y_start,
        "tiles_dir": tiles_dir,
        "slide_name": slide_name,
        "slide_output_dir": slide_output_dir,
        "h": h,
        "w": w,
        "tile_size": tile_size,
        "mosaic_width": mosaic_info["width"],
        "mosaic_height": mosaic_info["height"],
    }

    # INNER TRY: init + tile processing — SHM cleanup on failure
    try:
        # SHM is already created and populated from the direct-to-SHM loading above.
        # Loader channel_data points to SHM views (set during loading).
        # Clear loader references now — workers use SHM directly.
        if hasattr(loader, "clear_all_channels"):
            loader.clear_all_channels()
        else:
            loader.channel_data = None
        gc.collect()
        logger.info(
            f"Shared memory ready: {n_channels} channels, {slide_shm_arr.nbytes / (1024**3):.1f} GB"
        )

        init = _initialize_detector(
            args=args,
            loader=loader,
            n_channels=n_channels,
            ch_to_slot=ch_to_slot,
            slide_shm_arr=slide_shm_arr,
            sampled_tiles=sampled_tiles,
            pixel_size_um=pixel_size_um,
            shm_manager=shm_manager,
            x_start=x_start,
            y_start=y_start,
            tile_size=tile_size,
            ch_keys=ch_keys,
            slide_name=slide_name,
        )

        collected_partial_vessels = {}
        all_samples = []

        # Common processor kwargs (shared between multiscale and regular)
        processor_kwargs = {
            "num_gpus": init["num_gpus"],
            "slide_info": init["mgpu_slide_info"],
            "cell_type": args.cell_type,
            "strategy_params": init["strategy_params"],
            "pixel_size_um": pixel_size_um,
            "classifier_path": init["classifier_path"],
            "extract_deep_features": init["extract_deep"],
            "extract_sam2_embeddings": True,
            "detection_channel": tissue_channel,
            "cd31_channel": init["mgpu_cd31_channel"],
            "channel_names": init["mgpu_channel_names"],
            "variance_threshold": variance_threshold,
            "channel_keys": ch_keys,
            "tiles_dir": str(tiles_dir),
        }

        if init["is_multiscale"]:
            with MultiGPUTileProcessor(**processor_kwargs) as processor:
                all_detections, all_samples = _run_multiscale_tiles(
                    processor, args, ctx, init, sampled_tiles
                )
        else:
            # Pass display channels for tile_rgb override (generic, not islet-only)
            processor_kwargs["islet_display_channels"] = getattr(
                args, "display_channel_list", None
            ) or getattr(args, "islet_display_chs", None)
            with MultiGPUTileProcessor(**processor_kwargs) as processor:
                all_detections, all_samples, collected_partial_vessels = _run_regular_tiles(
                    processor, args, ctx, init, sampled_tiles, all_samples
                )

    except Exception:
        # Cleanup shared memory on detection failure before re-raising
        shm_manager.cleanup()
        raise

    # NOTE: shared memory is still alive here — post-dedup and HTML generation need it.
    # Cleanup is deferred to after _finish_pipeline().
    # Post-detection code (tile coverage, vessel merge) is wrapped in its own
    # try/except to prevent SHM leaks if these steps crash.

    # OUTER TRY: post-detection — separate SHM cleanup
    try:
        early = _postprocess_detections(
            all_detections,
            sampled_tiles,
            args,
            collected_partial_vessels,
            ctx,
            init["is_multiscale"],
            shm_manager,
        )
        if early:
            return early
    except Exception:
        shm_manager.cleanup()
        raise

    return {
        "exit_early": False,
        "all_detections": all_detections,
        "all_samples": all_samples,
        "params": init["params"],
        "detector": init["detector"],
        "classifier_loaded": init["classifier_loaded"],
        "is_multiscale": init["is_multiscale"],
    }
