#!/usr/bin/env python
"""Generate interactive HTML viewer for region label maps.

Single-layer mode: shows per-region cell/nuclear stats with distribution bars.
Multi-layer mode: toggleable layers for comparing segmentation runs.

Usage:
    # Single layer with stats
    python scripts/generate_region_viewer.py \\
        --label-maps regions/labels_pts64_filled.npy \\
        --czi-path slide.czi --display-channels "4" \\
        --nuc-stats regions/region_nuc_stats.json \\
        --min-cells 1000 \\
        --output viewer.html

    # Multi-layer comparison
    python scripts/generate_region_viewer.py \\
        --label-maps regions/labels_*_filled.npy \\
        --czi-path slide.czi --display-channels "4,2" \\
        --output comparison.html
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--label-maps", nargs="+", required=True, help="One or more .npy label map files"
    )
    parser.add_argument("--czi-path", help="CZI file for fluorescence background")
    parser.add_argument(
        "--display-channels",
        default="4",
        help="Channel indices for background (e.g., '4' or '4,2')",
    )
    parser.add_argument(
        "--scale", type=float, default=1 / 256, help="Thumbnail scale (default: 1/256)"
    )
    parser.add_argument("--scene", type=int, default=0, help="CZI scene index")
    parser.add_argument(
        "--nuc-stats", help="Pre-computed region_nuc_stats.json (avoids loading large detections)"
    )
    parser.add_argument(
        "--detections",
        help="Detections JSON with organ_id (alternative to --nuc-stats, needs lots of RAM)",
    )
    parser.add_argument(
        "--min-cells", type=int, default=0, help="Min nucleated cells per region (default: 0)"
    )
    parser.add_argument("--title", default="", help="Viewer title")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument(
        "--highlight-regions",
        default=None,
        help="Comma-separated region IDs OR path to a JSON file containing a list "
        "of region IDs. Highlighted regions render with a bold outline + colored "
        "fill; others get a thin outline only. Useful for marking statistical "
        "outliers (e.g., Tukey+ multinucleated regions).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    from xldvp_seg.utils.image_utils import percentile_normalize
    from xldvp_seg.visualization.fluorescence import read_czi_thumbnail_channels
    from xldvp_seg.visualization.region_viewer import (
        generate_multi_layer_viewer,
        generate_region_viewer,
    )

    # Load label maps
    label_maps = []
    for p in args.label_maps:
        path = Path(p)
        if not path.exists():
            logger.error("Label map not found: %s", path)
            sys.exit(1)
        name = path.stem.replace("labels_fluor_sam2_", "").replace("labels_", "")
        label_maps.append((name, np.load(path)))
    logger.info("Loaded %d label maps", len(label_maps))

    # Load fluorescence thumbnails
    fluor_thumbnails = []
    if args.czi_path:
        import io

        from PIL import Image

        channels = [int(c.strip()) for c in args.display_channels.split(",")]
        for ch in channels:
            ch_data, _, _, _ = read_czi_thumbnail_channels(
                args.czi_path, display_channels=[ch], scale_factor=args.scale, scene=args.scene
            )
            norm = percentile_normalize(ch_data[0])
            buf = io.BytesIO()
            Image.fromarray(norm).save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            fluor_thumbnails.append((f"ch{ch}", b64))
    else:
        logger.warning("No --czi-path provided — viewer will have no background image")

    # Load region stats
    region_stats = None
    if args.nuc_stats:
        with open(args.nuc_stats) as f:
            raw = json.load(f)
        region_stats = {int(k): v for k, v in raw.items()}
        logger.info("Loaded pre-computed stats for %d regions", len(region_stats))
    elif args.detections:
        from xldvp_seg.analysis.region_segmentation import compute_region_nuc_stats
        from xldvp_seg.utils.json_utils import fast_json_load

        logger.info("Loading detections (this may take a while for large files)...")
        dets = fast_json_load(args.detections)
        region_stats = compute_region_nuc_stats(dets)
        logger.info(
            "Computed stats for %d regions from %d detections", len(region_stats), len(dets)
        )
        del dets

    # Parse --highlight-regions (comma-separated IDs or JSON file path)
    highlight_ids = None
    if args.highlight_regions:
        hr = args.highlight_regions
        if Path(hr).exists():
            with open(hr) as f:
                highlight_ids = {int(x) for x in json.load(f)}
        else:
            highlight_ids = {int(x.strip()) for x in hr.split(",") if x.strip()}
        logger.info("Highlighting %d regions", len(highlight_ids))

    # Generate viewer
    if len(label_maps) == 1 and region_stats is not None:
        name, lbl = label_maps[0]
        generate_region_viewer(
            lbl,
            fluor_thumbnails,
            args.output,
            region_stats=region_stats,
            min_cells=args.min_cells,
            title=args.title,
            highlight_ids=highlight_ids,
        )
    else:
        generate_multi_layer_viewer(
            label_maps,
            fluor_thumbnails,
            args.output,
            title=args.title,
        )

    logger.info("Done: %s", args.output)


if __name__ == "__main__":
    main()
