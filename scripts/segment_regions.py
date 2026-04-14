#!/usr/bin/env python
"""Segment tissue into anatomical regions using SAM2 on fluorescence thumbnails.

Runs SAM2 auto-mask on CZI fluorescence channel thumbnails, builds a
non-overlapping label map, cleans peripheral junk, and optionally fills
gaps so every tissue pixel belongs to a region.

Usage:
    # Single run
    python scripts/segment_regions.py \\
        --czi-path slide.czi \\
        --display-channels "4,2" \\
        --points-per-side 64 \\
        --fill --fill-interstitial \\
        --output-dir regions/

    # Point density series (parallel SLURM jobs)
    python scripts/segment_regions.py \\
        --czi-path slide.czi \\
        --display-channels "4,2" \\
        --points-series 32,64,128,256,512 \\
        --fill --fill-interstitial \\
        --slurm --partition p.hpcl93 --mem 556G --time 1-00:00:00 \\
        --output-dir regions/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--czi-path", required=True, help="CZI file")
    parser.add_argument(
        "--display-channels",
        required=True,
        help="Comma-separated channel indices for RGB composite (e.g., '4,2')",
    )
    parser.add_argument(
        "--scale", type=float, default=1 / 256, help="Thumbnail scale (default: 1/256)"
    )
    parser.add_argument("--scene", type=int, default=0, help="CZI scene index")

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--points-per-side", type=int, help="SAM2 point density (single run)")
    g.add_argument(
        "--points-series", type=str, help="Comma-separated densities (e.g., '32,64,128')"
    )

    parser.add_argument("--pred-iou-thresh", type=float, default=0.0, help="SAM2 IoU threshold")
    parser.add_argument(
        "--stability-thresh", type=float, default=0.0, help="SAM2 stability threshold"
    )
    parser.add_argument("--sigma", type=float, default=5.0, help="Gaussian smoothing sigma")
    parser.add_argument("--min-area", type=int, default=300, help="Minimum region area (px)")
    parser.add_argument(
        "--min-tissue-overlap", type=float, default=0.75, help="Min tissue overlap fraction"
    )
    parser.add_argument(
        "--tissue-erode", type=int, default=12, help="Tissue mask erosion iterations"
    )
    parser.add_argument("--fill", action="store_true", help="Fill gaps with expand_labels")
    parser.add_argument(
        "--fill-interstitial", action="store_true", help="Fill enclosed holes as own regions"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="auto", help="SAM2 device"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # SLURM
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job(s)")
    parser.add_argument("--partition", default="p.hpcl93", help="SLURM partition")
    parser.add_argument("--mem", default="556G", help="SLURM memory")
    parser.add_argument("--time", default="1-00:00:00", help="SLURM time limit")
    parser.add_argument("--cpus", type=int, default=32, help="SLURM CPUs")
    parser.add_argument("--dry-run", action="store_true", help="Write sbatch but don't submit")
    parser.add_argument(
        "--viewer-after", action="store_true", help="Chain viewer regen after each SLURM job"
    )

    return parser.parse_args(argv)


def _resolve_device(device_str, points_per_side):
    if device_str == "auto":
        return "cpu" if points_per_side >= 256 else "cuda"
    return device_str


def _channel_tag(channels):
    """Build a filename tag from channel indices."""
    return "_".join(str(c) for c in channels)


def run_single(args, points_per_side):
    """Run a single segmentation at the given point density."""
    import numpy as np

    from xldvp_seg.analysis.region_segmentation import (
        build_tissue_mask,
        clean_labels,
        fill_labels,
        segment_regions,
    )
    from xldvp_seg.utils.image_utils import percentile_normalize
    from xldvp_seg.visualization.fluorescence import read_czi_thumbnail_channels

    channels = [int(c.strip()) for c in args.display_channels.split(",")]
    device = _resolve_device(args.device, points_per_side)

    logger.info("Loading CZI thumbnails (channels=%s, scale=%.4f)...", channels, args.scale)
    channel_arrays = []
    for ch in channels:
        ch_data, _, _, _ = read_czi_thumbnail_channels(
            args.czi_path, display_channels=[ch], scale_factor=args.scale, scene=args.scene
        )
        channel_arrays.append(percentile_normalize(ch_data[0]))

    # Build RGB composite (pad to 3 channels if needed)
    while len(channel_arrays) < 3:
        channel_arrays.append(np.zeros_like(channel_arrays[0]))
    rgb = np.stack(channel_arrays[:3], axis=-1)

    # Tissue mask from first channel
    tissue = build_tissue_mask(channel_arrays[0], erode=args.tissue_erode)

    logger.info("Running SAM2 (pts=%d, device=%s)...", points_per_side, device)
    labels, raw_masks = segment_regions(
        rgb,
        points_per_side=points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=args.min_area,
        device=device,
        sigma=args.sigma,
    )

    labels = clean_labels(
        labels, tissue, min_area=args.min_area, min_tissue_overlap=args.min_tissue_overlap
    )

    tag = _channel_tag(channels)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"labels_{tag}_pts{points_per_side}.npy", labels)

    if args.fill or args.fill_interstitial:
        filled = fill_labels(labels, tissue, fill_interstitial=args.fill_interstitial)
        np.save(out_dir / f"labels_{tag}_pts{points_per_side}_filled.npy", filled)

    logger.info("Done: pts=%d, output=%s", points_per_side, out_dir)


def submit_slurm(args):
    """Generate and submit sbatch files for a point density series."""
    from datetime import datetime

    points = [int(p.strip()) for p in args.points_series.split(",")]
    channels = [int(c.strip()) for c in args.display_channels.split(",")]
    tag = _channel_tag(channels)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    xldvp_python = os.environ.get(
        "XLDVP_PYTHON",
        str(Path(sys.executable)),
    )

    job_ids = []
    for pts in points:
        device = _resolve_device(args.device, pts)
        sbatch_path = out_dir / f"segment_pts{pts}_{ts}.sbatch"

        cmd = (
            f'"{xldvp_python}" {REPO / "scripts" / "segment_regions.py"}'
            f" --czi-path {args.czi_path}"
            f" --display-channels {args.display_channels}"
            f" --scale {args.scale}"
            f" --scene {args.scene}"
            f" --points-per-side {pts}"
            f" --pred-iou-thresh {args.pred_iou_thresh}"
            f" --stability-thresh {args.stability_thresh}"
            f" --sigma {args.sigma}"
            f" --min-area {args.min_area}"
            f" --min-tissue-overlap {args.min_tissue_overlap}"
            f" --tissue-erode {args.tissue_erode}"
            f" --device {device}"
            f" --output-dir {args.output_dir}"
        )
        if args.fill:
            cmd += " --fill"
        if args.fill_interstitial:
            cmd += " --fill-interstitial"

        gpu_line = ""
        if device in ("cuda", "auto"):
            gpu_line = "#SBATCH --gres=gpu:1\n"

        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=seg_pts{pts}
#SBATCH --partition={args.partition}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.mem}
{gpu_line}#SBATCH --time={args.time}
#SBATCH --output={out_dir}/segment_pts{pts}_%j.out
#SBATCH --error={out_dir}/segment_pts{pts}_%j.err

set -euo pipefail
export PYTHONPATH="{REPO}"
{cmd}
"""
        sbatch_path.write_text(sbatch_content)
        logger.info("Wrote %s", sbatch_path)

        if not args.dry_run:
            result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                job_ids.append((pts, job_id))
                logger.info("Submitted pts=%d as job %s", pts, job_id)
            else:
                logger.error("sbatch failed for pts=%d: %s", pts, result.stderr)

    # Chain viewer regen after each job (explicit filename, not glob)
    if args.viewer_after and not args.dry_run:
        viewer_script = REPO / "scripts" / "generate_region_viewer.py"
        if viewer_script.exists():
            for pts, job_id in job_ids:
                label_file = out_dir / f"labels_{tag}_pts{pts}_filled.npy"
                viewer_cmd = (
                    f'"{xldvp_python}" {viewer_script}'
                    f" --label-maps {label_file}"
                    f" --czi-path {args.czi_path}"
                    f" --display-channels {args.display_channels}"
                    f" --output {out_dir}/viewer_pts{pts}.html"
                )
                subprocess.run(
                    [
                        "sbatch",
                        f"--dependency=afterany:{job_id}",
                        "--partition=p.hpcl8",
                        "--cpus-per-task=4",
                        "--mem=16G",
                        "--time=00:10:00",
                        "--job-name=regen_view",
                        f"--output={out_dir}/regen_viewer_%j.out",
                        f"--error={out_dir}/regen_viewer_%j.err",
                        f"--wrap=set -euo pipefail; export PYTHONPATH={REPO}; {viewer_cmd}",
                    ],
                    capture_output=True,
                )

    if job_ids:
        logger.info(
            "Submitted %d jobs: %s", len(job_ids), ", ".join(f"pts={p}→{j}" for p, j in job_ids)
        )


def main():
    args = parse_args()

    if args.points_series and args.slurm:
        submit_slurm(args)
    elif args.points_series:
        # Sequential local run
        points = [int(p.strip()) for p in args.points_series.split(",")]
        for pts in points:
            run_single(args, pts)
    else:
        run_single(args, args.points_per_side)


if __name__ == "__main__":
    main()
