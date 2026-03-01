"""Post-processing: channel legend, CSV/JSON/HTML export, summary, and server launch.

Functions for building channel legend metadata from CZI files, parsing channel
info from filenames, and the shared _finish_pipeline() that handles all
post-detection output: detections JSON, coordinates CSV, HTML export, summary
JSON, and HTTP server startup.
"""

import re
import json
from pathlib import Path

from segmentation.utils.logging import get_logger
from segmentation.utils.json_utils import NumpyEncoder, atomic_json_dump
from segmentation.utils.timestamps import timestamped_path, update_symlink
from segmentation.io.czi_loader import get_czi_metadata
from segmentation.io.html_export import export_samples_to_html
from segmentation.pipeline.server import (
    start_server_and_tunnel, wait_for_server_shutdown, show_server_status,
)

logger = get_logger(__name__)


def parse_channel_legend_from_filename(filename: str) -> dict:
    """Parse channel information from filename to create legend.

    Looks for patterns like:
    - nuc488, nuc405 -> nuclear stain (keeps original like 'nuc488')
    - Bgtx647, BTX647 -> bungarotoxin (keeps original)
    - NfL750, NFL750 -> neurofilament (keeps original)
    - DAPI -> nuclear
    - SMA -> smooth muscle actin
    - _647_ -> standalone wavelength

    Args:
        filename: Slide filename (e.g., '20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch')

    Returns:
        Dict with 'red', 'green', 'blue' keys mapping to channel names,
        or None if no channels detected.
    """
    channels = []

    # Specific channel patterns - use original text from filename
    # Order: patterns that include wavelength first, then standalone wavelengths
    patterns = [
        # Patterns with wavelength embedded (capture the whole thing)
        r'nuc\d{3}',           # nuc488, nuc405
        r'bgtx\d{3}',          # Bgtx647
        r'btx\d{3}',           # BTX647
        r'nfl?\d{3}',          # NfL750, NFL750
        r'sma\d*',             # SMA, SMA488
        r'cd\d+',              # CD31, CD34
        # Named stains without wavelength
        r'dapi',
        r'bungarotoxin',
        r'neurofilament',
        # Standalone wavelengths (must be bounded by _ or - or start/end)
        r'(?:^|[_-])(\d{3})(?:[_-]|$)',  # _647_, -488-
    ]

    # Find all channel mentions with their positions
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, filename, re.IGNORECASE):
            # For grouped patterns, use group(1) if it exists
            if match.lastindex:
                text = match.group(1)
                pos = match.start(1)
            else:
                text = match.group(0)
                pos = match.start()
            found.append((pos, text))

    # Sort by position in filename and deduplicate
    found.sort(key=lambda x: x[0])
    seen = set()
    for pos, name in found:
        name_lower = name.lower()
        if name_lower not in seen:
            channels.append(name)
            seen.add(name_lower)

    if len(channels) >= 3:
        return {
            'red': channels[0],
            'green': channels[1],
            'blue': channels[2]
        }
    elif len(channels) == 2:
        return {
            'red': channels[0],
            'green': channels[1]
        }
    elif len(channels) == 1:
        return {
            'green': channels[0]  # Single channel shown as green
        }

    return None


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
        _czi_meta = get_czi_metadata(czi_path, scene=getattr(args, 'scene', 0))

        def _channel_label(ch_idx):
            for ch in _czi_meta['channels']:
                if ch['index'] == ch_idx:
                    name = ch['name']
                    em = f" ({ch['emission_nm']:.0f}nm)" if ch.get('emission_nm') else ''
                    return f'{name}{em}'
            return f'Ch{ch_idx}'

        if cell_type == 'islet':
            _islet_disp = getattr(args, 'islet_display_chs', [2, 3, 5])
            return {
                'red': _channel_label(_islet_disp[0]) if len(_islet_disp) > 0 else 'none',
                'green': _channel_label(_islet_disp[1]) if len(_islet_disp) > 1 else 'none',
                'blue': _channel_label(_islet_disp[2]) if len(_islet_disp) > 2 else 'none',
            }
        elif cell_type == 'tissue_pattern':
            tp_disp = getattr(args, 'tp_display_channels_list', [0, 3, 1])
            return {
                'red': _channel_label(tp_disp[0]) if len(tp_disp) > 0 else 'none',
                'green': _channel_label(tp_disp[1]) if len(tp_disp) > 1 else 'none',
                'blue': _channel_label(tp_disp[2]) if len(tp_disp) > 2 else 'none',
            }
        elif cell_type == 'nmj' and getattr(args, 'all_channels', False):
            try:
                return {
                    'red': _channel_label(0),
                    'green': _channel_label(1),
                    'blue': _channel_label(2),
                }
            except Exception:
                return parse_channel_legend_from_filename(slide_name) if slide_name else None
        else:
            return {
                'red': _channel_label(0),
                'green': _channel_label(1),
                'blue': _channel_label(2),
            }
    except Exception:
        return None


def _finish_pipeline(args, all_detections, all_samples, slide_output_dir, tiles_dir,
                     pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
                     skip_html=False, all_tiles=None, tissue_tiles=None, sampled_tiles=None,
                     resumed=False, params=None, classifier_loaded=False,
                     is_multiscale=False, detector=None):
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
        det['tile_mask_label'] = det.get('mask_label', 0)
        _gc = det.get('global_center', det.get('center', [0, 0]))
        det['global_id'] = f"{int(round(_gc[0]))}_{int(round(_gc[1]))}"

    detections_file = slide_output_dir / f'{cell_type}_detections.json'
    ts_detections = timestamped_path(detections_file)
    atomic_json_dump(all_detections, ts_detections)
    update_symlink(detections_file, ts_detections)
    logger.info(f"Saved {len(all_detections)} detections to {ts_detections}")

    csv_file = slide_output_dir / f'{cell_type}_coordinates.csv'
    ts_csv = timestamped_path(csv_file)
    with open(ts_csv, 'w') as f:
        if cell_type == 'vessel':
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,outer_diameter_um,wall_thickness_um,confidence\n')
            for det in all_detections:
                g_center = det.get('global_center')
                g_center_um = det.get('global_center_um')
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get('features', {})
                f.write(f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                        f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},"
                        f"{feat.get('outer_diameter_um', 0):.2f},{feat.get('wall_thickness_mean_um', 0):.2f},"
                        f"{feat.get('confidence', 'unknown')}\n")
        else:
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2\n')
            for det in all_detections:
                g_center = det.get('global_center')
                g_center_um = det.get('global_center_um')
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get('features', {})
                area_um2 = feat.get('area', 0) * (pixel_size_um ** 2)
                f.write(f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                        f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},{area_um2:.2f}\n")
    update_symlink(csv_file, ts_csv)
    logger.info(f"Saved coordinates to {ts_csv}")

    # Sort samples: classifier runs -> RF score descending; else -> area ascending
    _has_classifier = classifier_loaded or (
        (cell_type == 'nmj' and getattr(args, 'nmj_classifier', None)) or
        (cell_type == 'islet' and getattr(args, 'islet_classifier', None)) or
        (cell_type == 'tissue_pattern' and getattr(args, 'tp_classifier', None))
    )
    if _has_classifier:
        all_samples.sort(key=lambda x: x['stats'].get('rf_prediction') or 0, reverse=True)
    else:
        all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0))

    # Export to HTML (unless skipped)
    if not skip_html and all_samples:
        if cell_type in ('nmj', 'islet', 'tissue_pattern') and len(all_detections) > len(all_samples):
            logger.info(f"Total detections (all scores): {len(all_detections)}, "
                         f"shown in HTML (rf_prediction >= {args.html_score_threshold}): {len(all_samples)}")
        logger.info(f"Exporting to HTML ({len(all_samples)} samples)...")
        html_dir = slide_output_dir / "html"

        channel_legend = _build_channel_legend(cell_type, args, czi_path, slide_name=slide_name)

        prior_ann = getattr(args, 'prior_annotations', None)
        experiment_name = f"{slide_name}_{run_timestamp}_{pct}pct"
        _title_suffix = " (resumed)" if resumed else ""
        export_samples_to_html(
            all_samples,
            html_dir,
            cell_type,
            samples_per_page=args.samples_per_page,
            title=f"{cell_type.upper()} Annotation Review{_title_suffix}",
            page_prefix=f'{cell_type}_page',
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
    summary = {
        'slide_name': slide_name,
        'cell_type': cell_type,
        'pixel_size_um': pixel_size_um,
        'mosaic_width': mosaic_info['width'],
        'mosaic_height': mosaic_info['height'],
        'total_tiles': len(all_tiles) if all_tiles else 0,
        'tissue_tiles': len(tissue_tiles) if tissue_tiles else 0,
        'sampled_tiles': len(sampled_tiles) if sampled_tiles else 0,
        'total_detections': len(all_detections),
        'html_displayed': len(all_samples),
        'resumed': resumed,
        'params': params if params else {},
        'detections_file': str(detections_file),
        'coordinates_file': str(csv_file),
    }
    atomic_json_dump(summary, slide_output_dir / 'summary.json')

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

    # Start HTTP server
    no_serve = getattr(args, 'no_serve', False)
    serve_foreground = getattr(args, 'serve', False)
    serve_background = getattr(args, 'serve_background', True)
    port = getattr(args, 'port', 8081)

    if no_serve:
        logger.info("Server disabled (--no-serve)")
    elif html_dir.exists() and not skip_html:
        if serve_foreground:
            http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
                html_dir, port, background=False,
                slide_name=slide_name, cell_type=cell_type)
            if http_proc is not None:
                wait_for_server_shutdown(http_proc, tunnel_proc)
        elif serve_background:
            start_server_and_tunnel(
                html_dir, port, background=True,
                slide_name=slide_name, cell_type=cell_type)
            print("")
            show_server_status()
