"""Post-processing: channel legend, CSV/JSON/HTML export, summary, and server launch.

Functions for building channel legend metadata from CZI files, parsing channel
info from filenames, and the shared _finish_pipeline() that handles all
post-detection output: detections JSON, coordinates CSV, HTML export, summary
JSON, and HTTP server startup.
"""

from pathlib import Path

from segmentation.io.czi_loader import get_czi_metadata
from segmentation.io.html_export import export_samples_to_html
from segmentation.pipeline.server import (
    show_server_status,
    start_server_and_tunnel,
    wait_for_server_shutdown,
)
from segmentation.utils.json_utils import atomic_json_dump
from segmentation.utils.logging import get_logger
from segmentation.utils.timestamps import timestamped_path, update_symlink

logger = get_logger(__name__)


def _build_channel_legend(cell_type, args, czi_path, slide_name=None):
    """Build channel legend dict from CZI metadata for HTML export.

    Args:
        cell_type: Detection cell type string
        args: Parsed arguments (for display channel config)
        czi_path: Path to CZI file (for metadata extraction)
        slide_name: Slide name (fallback for NMJ filename parsing)

    Returns:
        Dict with 'red', 'green', 'blue' keys, or None on failure.
    """
    try:
        _czi_meta = get_czi_metadata(czi_path, scene=getattr(args, "scene", 0))

        def _channel_label(ch_idx):
            for ch in _czi_meta["channels"]:
                if ch["index"] == ch_idx:
                    name = ch["name"]
                    em = f" ({ch['emission_nm']:.0f}nm)" if ch.get("emission_nm") else ""
                    return f"{name}{em}"
            return f"Ch{ch_idx}"

        if cell_type == "islet":
            disp = getattr(args, "islet_display_chs", [2, 3, 5])
        elif cell_type == "tissue_pattern":
            disp = getattr(args, "tp_display_channels_list", [0, 3, 1])
        else:
            disp = [0, 1, 2]

        color_names = ("red", "green", "blue")
        return {
            color_names[i]: _channel_label(disp[i]) if i < len(disp) else "none" for i in range(3)
        }
    except (OSError, KeyError, IndexError) as e:
        logger.warning("Could not build channel legend from CZI metadata: %s", e)
        return None


def _export_spatialdata(
    args, all_detections, cell_type, pixel_size_um, slide_output_dir, tiles_dir
):
    """Export detections to SpatialData format if dependencies are available.

    Silently skips if spatialdata/anndata/geopandas are not installed.
    """
    try:
        import spatialdata  # noqa: F401
    except ImportError:
        return

    try:
        from scripts.convert_to_spatialdata import export_spatialdata

        output_zarr = slide_output_dir / f"{cell_type}_spatialdata.zarr"
        tiles_path = str(tiles_dir) if tiles_dir and Path(tiles_dir).exists() else None

        # Check for OME-Zarr image next to the CZI
        zarr_image = None
        czi_path = Path(args.czi_path)
        candidate_zarr = czi_path.with_suffix(".ome.zarr")
        if candidate_zarr.exists():
            zarr_image = str(candidate_zarr)

        export_spatialdata(
            detections=all_detections,
            output_path=output_zarr,
            cell_type=cell_type,
            tiles_dir=tiles_path,
            zarr_image=zarr_image,
            pixel_size_um=pixel_size_um,
            run_squidpy=False,
            overwrite=True,
        )
    except Exception as e:
        logger.warning("SpatialData export failed (non-fatal): %s", e)


def _finish_pipeline(
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
    skip_html=False,
    all_tiles=None,
    tissue_tiles=None,
    sampled_tiles=None,
    resumed=False,
    params=None,
    classifier_loaded=False,
    is_multiscale=False,
    detector=None,
):
    """Shared post-processing: HTML export, CSV, summary, server (used by both normal and resume paths).

    Args:
        resumed: Whether this is a resumed pipeline run (affects title/summary).
        params: Detection parameters dict (for summary, normal path only).
        classifier_loaded: Whether a classifier was loaded (for sort order, normal path).
        is_multiscale: Whether multiscale mode was used (for checkpoint cleanup).
        detector: CellDetector instance to cleanup (normal path only).
    """
    cell_type = args.cell_type
    czi_path = Path(args.czi_path)

    # ---- Save detections JSON + CSV FIRST (before HTML) ----
    # This ensures dedup results are persisted even if HTML generation crashes/hangs.
    # On resume, the pipeline will find the detections JSON and skip detection+dedup.
    for det in all_detections:
        det["tile_mask_label"] = det.get("mask_label", 0)
        _gc = det.get("global_center", [0, 0])
        det["global_id"] = f"{int(round(_gc[0]))}_{int(round(_gc[1]))}"

    detections_file = slide_output_dir / f"{cell_type}_detections.json"
    ts_detections = timestamped_path(detections_file)
    atomic_json_dump(all_detections, ts_detections)
    update_symlink(detections_file, ts_detections)
    logger.info(f"Saved {len(all_detections)} detections to {ts_detections}")

    # Log classifier provenance status
    from segmentation.utils.classifier_registry import extract_classifier_info

    scored_count, prov_count, sample_clf_info = extract_classifier_info(all_detections)
    if scored_count > 0:
        logger.info(f"Classifier scores present on {scored_count}/{len(all_detections)} detections")
        if prov_count > 0 and sample_clf_info:
            logger.info(
                f"  Classifier: {sample_clf_info.get('classifier_name', 'unknown')} "
                f"(F1={sample_clf_info.get('cv_f1', '?')}, "
                f"scored {sample_clf_info.get('scored_at', '?')})"
            )
        elif prov_count == 0:
            logger.warning("Scores have NO provenance -- origin unknown")

    csv_file = slide_output_dir / f"{cell_type}_coordinates.csv"
    ts_csv = timestamped_path(csv_file)
    import tempfile as _tmpmod

    _csv_fd, _csv_tmp = _tmpmod.mkstemp(dir=ts_csv.parent, suffix=".csv.tmp")
    try:
        import os as _os_mod

        with _os_mod.fdopen(_csv_fd, "w") as f:
            if cell_type == "vessel":
                f.write(
                    "uid,global_x_px,global_y_px,global_x_um,global_y_um,outer_diameter_um,wall_thickness_um,confidence,classifier\n"
                )
            else:
                f.write("uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2,classifier\n")
            for det in all_detections:
                g_center = det.get("global_center")
                g_center_um = det.get("global_center_um")
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get("features", {})
                _clf_name = det.get("classifier_info", {}).get("classifier_name", "")
                if not _clf_name and det.get("rf_prediction") is not None:
                    _clf_name = "unknown"
                base = (
                    f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                    f"{g_center_um[0]:.2f},{g_center_um[1]:.2f}"
                )
                if cell_type == "vessel":
                    f.write(
                        f"{base},"
                        f"{feat.get('outer_diameter_um', 0):.2f},{feat.get('wall_thickness_mean_um', 0):.2f},"
                        f"{feat.get('confidence', 'unknown')},{_clf_name}\n"
                    )
                else:
                    area_um2 = feat.get("area", 0) * (pixel_size_um**2)
                    f.write(f"{base},{area_um2:.2f},{_clf_name}\n")
        _os_mod.replace(_csv_tmp, ts_csv)
    except BaseException:
        try:
            import os as _os_mod2

            _os_mod2.unlink(_csv_tmp)
        except OSError:
            pass
        raise
    update_symlink(csv_file, ts_csv)
    logger.info(f"Saved coordinates to {ts_csv}")

    # Sort samples: classifier runs -> RF score descending; else -> area ascending
    _has_classifier = classifier_loaded or (
        (cell_type == "nmj" and getattr(args, "nmj_classifier", None))
        or (cell_type == "islet" and getattr(args, "islet_classifier", None))
        or (cell_type == "tissue_pattern" and getattr(args, "tp_classifier", None))
    )
    if _has_classifier:
        all_samples.sort(key=lambda x: x["stats"].get("rf_prediction") or 0, reverse=True)
    else:
        all_samples.sort(key=lambda x: x["stats"].get("area_um2", 0))

    # Export to HTML (unless skipped)
    if not skip_html and all_samples:
        if cell_type in ("nmj", "islet", "tissue_pattern") and len(all_detections) > len(
            all_samples
        ):
            logger.info(
                f"Total detections (all scores): {len(all_detections)}, "
                f"shown in HTML (rf_prediction >= {args.html_score_threshold}): {len(all_samples)}"
            )
        logger.info(f"Exporting to HTML ({len(all_samples)} samples)...")
        html_dir = slide_output_dir / "html"

        channel_legend = _build_channel_legend(cell_type, args, czi_path, slide_name=slide_name)

        prior_ann = getattr(args, "prior_annotations", None)
        experiment_name = f"{slide_name}_{run_timestamp}_{pct}pct"
        _title_suffix = " (resumed)" if resumed else ""
        _clf_subtitle = None
        if sample_clf_info:
            _f1 = sample_clf_info.get("cv_f1")
            _f1_str = f", F1={_f1:.3f}" if _f1 else ""
            _clf_subtitle = (
                f"Classifier: {sample_clf_info.get('classifier_name', 'unknown')}{_f1_str}"
            )
        export_samples_to_html(
            all_samples,
            html_dir,
            cell_type,
            samples_per_page=args.samples_per_page,
            title=f"{cell_type.upper()} Annotation Review{_title_suffix}",
            subtitle=_clf_subtitle,
            page_prefix=f"{cell_type}_page",
            experiment_name=experiment_name,
            file_name=f"{czi_path.name}" if resumed else f"{slide_name}.czi",
            pixel_size_um=pixel_size_um,
            tiles_processed=len(sampled_tiles) if sampled_tiles else 0,
            tiles_total=len(all_tiles) if all_tiles else 0,
            channel_legend=channel_legend,
            prior_annotations=prior_ann,
        )
    elif skip_html:
        logger.info("HTML export skipped (already exists)")

    # Clean up multiscale checkpoints after successful completion
    if is_multiscale:
        checkpoint_dir = slide_output_dir / "checkpoints"
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)
            logger.info("Multiscale checkpoints cleaned up after successful completion")

    # Save summary
    # Check if post-dedup processing was applied
    _sample_feat = all_detections[0].get("features", {}) if all_detections else {}
    _bg_corrected = any(k.endswith("_background") for k in _sample_feat)
    _contour_processed = any(d.get("contour_dilated_px") is not None for d in all_detections[:1])

    summary = {
        "slide_name": slide_name,
        "cell_type": cell_type,
        "pixel_size_um": pixel_size_um,
        "mosaic_width": mosaic_info["width"],
        "mosaic_height": mosaic_info["height"],
        "total_tiles": len(all_tiles) if all_tiles else 0,
        "tissue_tiles": len(tissue_tiles) if tissue_tiles else 0,
        "sampled_tiles": len(sampled_tiles) if sampled_tiles else 0,
        "total_detections": len(all_detections),
        "html_displayed": len(all_samples),
        "resumed": resumed,
        "params": params if params else {},
        "detections_file": str(detections_file),
        "coordinates_file": str(csv_file),
        "background_corrected": _bg_corrected,
        "contour_processed": _contour_processed,
    }
    if sample_clf_info:
        summary["classifier"] = {
            "name": sample_clf_info.get("classifier_name"),
            "feature_set": sample_clf_info.get("feature_set"),
            "cv_f1": sample_clf_info.get("cv_f1"),
            "scored_at": sample_clf_info.get("scored_at"),
        }
    atomic_json_dump(summary, slide_output_dir / "summary.json")

    # Cleanup detector resources
    if detector is not None:
        detector.cleanup()

    _status_label = "COMPLETE (resumed)" if resumed else "COMPLETE"
    logger.info("=" * 60)
    logger.info(_status_label)
    logger.info("=" * 60)
    logger.info(f"Total detections: {len(all_detections)}")
    logger.info(f"Displayed in HTML: {len(all_samples)} (score >= {args.html_score_threshold})")
    logger.info(f"Output: {slide_output_dir}")
    html_dir = slide_output_dir / "html"
    if html_dir.exists():
        logger.info(f"HTML viewer: {html_dir / 'index.html'}")

    # SpatialData export (optional — skipped silently if deps not installed)
    _export_spatialdata(args, all_detections, cell_type, pixel_size_um, slide_output_dir, tiles_dir)

    # Start HTTP server
    no_serve = getattr(args, "no_serve", False)
    serve_foreground = getattr(args, "serve", False)
    serve_background = getattr(args, "serve_background", True)
    port = getattr(args, "port", 8081)

    if no_serve:
        logger.info("Server disabled (--no-serve)")
    elif html_dir.exists() and not skip_html:
        if serve_foreground:
            http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
                html_dir, port, background=False, slide_name=slide_name, cell_type=cell_type
            )
            if http_proc is not None:
                wait_for_server_shutdown(http_proc, tunnel_proc)
        elif serve_background:
            start_server_and_tunnel(
                html_dir, port, background=True, slide_name=slide_name, cell_type=cell_type
            )
            print("")
            show_server_status()
