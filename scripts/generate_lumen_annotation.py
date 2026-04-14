#!/usr/bin/env python
"""Generate paginated card-grid annotation HTML for vessel lumens.

Reads zarr crops centered on each lumen, builds fluorescence + contour-overlay
cards, and generates the standard Y/N/? annotation pages with channel toggles,
keyboard shortcuts, and localStorage persistence.

Usage:
    python scripts/generate_lumen_annotation.py \\
        --lumens vessel_lumens_threshold.json \\
        --zarr-path slide.ome.zarr \\
        --output-dir annotation_v1/ \\
        --display-channels 1,2,0 \\
        --channel-names "SMA,LYVE1,nuc" \\
        --sort-by rf_score --sort-descending \\
        --title "Fig7 Vessel Annotation"
"""

import argparse
import base64
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.io.html_export import generate_annotation_page, generate_index_page  # noqa: E402
from xldvp_seg.utils.json_utils import fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402
from xldvp_seg.utils.zarr_io import read_all_channels_crop, resolve_zarr_scales  # noqa: E402

logger = get_logger(__name__)

_AVAILABLE_SCALES = [4, 8, 16, 64]
_MIN_CROP_PX = 50  # minimum crop diameter in pixels


def pick_read_scale(diameter_um: float, base_pixel_size: float) -> int:
    """Pick coarsest scale that gives at least _MIN_CROP_PX diameter.

    Args:
        diameter_um: Lumen equivalent diameter in microns.
        base_pixel_size: Native pixel size in microns (CZI level 0).

    Returns:
        Scale factor (e.g. 4, 8, 16, 64).
    """
    for s in sorted(_AVAILABLE_SCALES, reverse=True):
        px_size = base_pixel_size * s
        if diameter_um / px_size >= _MIN_CROP_PX:
            return s
    return min(_AVAILABLE_SCALES)


def _encode_jpg(img_bgr: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


def _encode_png(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode("ascii")


def build_lumen_card(
    lumen: dict,
    zarr_root,
    scale_lookup: dict,
    base_pixel_size: float,
    display_channels: list[int],
    card_size: int,
) -> dict | None:
    """Build a single annotation card from a lumen dict.

    Returns:
        Sample dict with uid, image, image_clean, image_contour_only, stats.
        None if the crop could not be read.
    """
    uid = lumen.get("uid", "")
    cx_um = lumen.get("centroid_x_um")
    cy_um = lumen.get("centroid_y_um")
    if cx_um is None or cy_um is None:
        return None

    diam_um = lumen.get("equiv_diameter_um", 100)
    # Crop = 2× diameter (lumen centered with equal padding)
    half_um = max(diam_um, 20.0)  # minimum 20um half-window

    read_scale = pick_read_scale(diam_um, base_pixel_size)
    si = scale_lookup.get(read_scale)
    if si is None:
        return None
    scale_val, _level, level_key, extra_ds = si
    level_data = zarr_root[level_key]
    pixel_size = base_pixel_size * scale_val

    half_px = max(int(half_um / pixel_size), 25)
    cx_px = int(cx_um / pixel_size)
    cy_px = int(cy_um / pixel_size)
    y0 = max(0, cy_px - half_px)
    x0 = max(0, cx_px - half_px)
    h = w = 2 * half_px

    try:
        channels = read_all_channels_crop(level_data, extra_ds, y0, x0, h, w)
    except Exception:
        return None

    # Build RGB (clean — fluorescence only)
    ch0 = channels[0]
    rgb = np.zeros((ch0.shape[0], ch0.shape[1], 3), dtype=np.uint8)
    for ci, ch_idx in enumerate(display_channels):
        if ch_idx >= len(channels):
            continue
        ch = channels[ch_idx].astype(np.float32)
        nz = ch[ch > 0]
        if len(nz) > 10:
            p1, p99 = np.percentile(nz, [1, 99])
        else:
            p1, p99 = 0.0, 1.0
        if p99 <= p1:
            p99 = p1 + 1
        norm = np.clip((ch - p1) / (p99 - p1), 0, 1)
        rgb[:, :, ci] = (norm * 255).astype(np.uint8)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Build contour-only layer (green on black)
    contour_layer = np.zeros_like(rgb_bgr)
    contour_um = lumen.get("contour_global_um")
    if contour_um:
        pts_px = np.array(contour_um) / pixel_size - np.array([x0, y0])
        pts_int = pts_px.astype(np.int32).reshape(-1, 1, 2)
        thickness = max(1, min(3, rgb_bgr.shape[0] // 100))
        cv2.drawContours(contour_layer, [pts_int], -1, (0, 255, 0), thickness)

    clean_resized = cv2.resize(rgb_bgr, (card_size, card_size), interpolation=cv2.INTER_AREA)
    contour_resized = cv2.resize(
        contour_layer, (card_size, card_size), interpolation=cv2.INTER_AREA
    )

    stats: dict = {
        "area_um2": round(lumen.get("area_um2", 0), 1),
        "diameter_um": round(diam_um, 1),
        "scale": f"{lumen.get('discovery_scale', '?')}x",
    }
    rf_score = lumen.get("rf_score")
    if rf_score is not None:
        stats["score"] = round(rf_score, 3)

    return {
        "uid": uid,
        "image": _encode_jpg(clean_resized),
        "image_clean": _encode_jpg(clean_resized),
        "image_contour_only": _encode_png(contour_resized),
        "stats": stats,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--lumens", type=Path, required=True, help="Vessel lumens JSON")
    p.add_argument("--zarr-path", type=Path, required=True, help="OME-Zarr pyramid path")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for HTML pages")
    p.add_argument(
        "--display-channels",
        type=str,
        default="0,1,2",
        help="Comma-separated channel indices for R,G,B (default: 0,1,2)",
    )
    p.add_argument(
        "--channel-names",
        type=str,
        default=None,
        help="Comma-separated channel names (e.g. 'SMA,LYVE1,nuc')",
    )
    p.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Base pixel size in um (auto-detected from lumens if not provided)",
    )
    p.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help="Field to sort lumens by (e.g. rf_score, area_um2)",
    )
    p.add_argument(
        "--sort-descending", action="store_true", help="Sort descending (default: ascending)"
    )
    p.add_argument("--per-page", type=int, default=50, help="Lumens per page (default: 50)")
    p.add_argument(
        "--card-size", type=int, default=250, help="Card image size in pixels (default: 250)"
    )
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for localStorage isolation",
    )
    p.add_argument("--title", type=str, default="Vessel Lumen Annotation", help="Page title")
    args = p.parse_args()

    display_channels = [int(c.strip()) for c in args.display_channels.split(",")]
    channel_names = (
        [n.strip() for n in args.channel_names.split(",")] if args.channel_names else None
    )

    # --- Load lumens ---
    logger.info("Loading lumens: %s", args.lumens)
    lumens = fast_json_load(str(args.lumens))
    logger.info("  %d lumens", len(lumens))

    # --- Sort ---
    if args.sort_by:

        def _sort_key(l):
            v = l.get(args.sort_by)
            if v is None:
                v = l.get("features", {}).get(args.sort_by)
            return float(v) if isinstance(v, (int, float)) else 0.0

        lumens.sort(key=_sort_key, reverse=args.sort_descending)
        logger.info("Sorted by %s (%s)", args.sort_by, "desc" if args.sort_descending else "asc")

    # --- Open zarr ---
    import zarr

    zarr_root = zarr.open(str(args.zarr_path), mode="r")
    scale_infos = resolve_zarr_scales(zarr_root, _AVAILABLE_SCALES)
    scale_lookup = {si[0]: si for si in scale_infos}

    # --- Detect pixel size ---
    base_pixel_size = args.pixel_size_um
    if base_pixel_size is None:
        # Infer from equiv_diameter_um and bbox geometry at refined_scale.
        # For a round lumen, equiv_diameter_um ≈ bbox_side_px * pixel_size_at_scale,
        # where pixel_size_at_scale = base_pixel_size * refined_scale.
        # Use mean(bbox_w, bbox_h) as approximation of diameter in pixels.
        ratios = []
        for l in lumens[:200]:
            diam = l.get("equiv_diameter_um", 0)
            bw = l.get("bbox_w", 0)
            bh = l.get("bbox_h", 0)
            rs = l.get("refined_scale", 0)
            if diam > 0 and bw > 0 and bh > 0 and rs > 0:
                bbox_mean_px = (bw + bh) / 2.0
                if bbox_mean_px > 10:
                    ratios.append(diam / (bbox_mean_px * rs))
        if ratios:
            # Use median for robustness to elongated lumens
            base_pixel_size = float(np.median(ratios))
    if base_pixel_size is None:
        raise SystemExit(
            "ERROR: Could not determine pixel size. Provide --pixel-size-um explicitly."
        )
    if args.pixel_size_um is None:
        logger.info(
            "Base pixel size: %.4f um (inferred from %d lumens)", base_pixel_size, len(ratios)
        )
    else:
        logger.info("Base pixel size: %.4f um (from --pixel-size-um)", base_pixel_size)

    # --- Build cards ---
    samples = []
    for i, l in enumerate(lumens):
        card = build_lumen_card(
            l, zarr_root, scale_lookup, base_pixel_size, display_channels, args.card_size
        )
        if card is not None:
            samples.append(card)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d lumens processed (%d cards)", i + 1, len(lumens), len(samples))

    logger.info("Generated %d cards from %d lumens", len(samples), len(lumens))

    # --- Generate HTML pages ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_pages = max(1, math.ceil(len(samples) / args.per_page))

    channel_legend = {}
    color_names = ["red", "green", "blue"]
    if channel_names:
        for i, name in enumerate(channel_names[:3]):
            channel_legend[color_names[i]] = name

    for pg in range(1, n_pages + 1):
        batch = samples[(pg - 1) * args.per_page : pg * args.per_page]
        html = generate_annotation_page(
            samples=batch,
            cell_type="vessel_lumen",
            page_num=pg,
            total_pages=n_pages,
            title=args.title,
            page_prefix="annotation",
            experiment_name=args.experiment_name,
            channel_legend=channel_legend if channel_legend else None,
            subtitle=f"{len(samples)} lumens | {args.channel_names or 'default channels'}",
        )
        (args.output_dir / f"annotation_{pg}.html").write_text(html, encoding="utf-8")

    sort_desc = (
        f"sorted by {args.sort_by} ({'desc' if args.sort_descending else 'asc'})"
        if args.sort_by
        else "detection order"
    )
    index_html = generate_index_page(
        cell_type="vessel_lumen",
        total_samples=len(samples),
        total_pages=n_pages,
        title=args.title,
        subtitle=sort_desc,
        page_prefix="annotation",
        experiment_name=args.experiment_name,
        pixel_size_um=base_pixel_size,
        extra_stats={
            "Display": args.channel_names or args.display_channels,
            "Sort": sort_desc,
            "Card size": f"{args.card_size}px",
        },
    )
    (args.output_dir / "index.html").write_text(index_html, encoding="utf-8")
    logger.info("Wrote %d pages + index.html to %s/", n_pages, args.output_dir)


if __name__ == "__main__":
    main()
