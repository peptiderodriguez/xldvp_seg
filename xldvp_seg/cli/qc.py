"""Quick quality check for pipeline output.

Prints detection stats, feature summary, area distribution, classifier scores,
marker profiles, per-channel SNR, and nuclear counting summary without needing
the HTML viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def run_qc(args: argparse.Namespace) -> None:
    """Print quality summary for a detection output directory."""
    output_dir = Path(args.output_dir)

    # Find detection JSON files
    det_files = sorted(output_dir.glob("*_detections*.json"))
    if not det_files:
        print(f"No detection files found in {output_dir}")
        return

    det_path = det_files[-1]  # most recent
    print(f"Loading {det_path.name}...")
    detections = fast_json_load(str(det_path))

    print(f"\n{'=' * 60}")
    print(f"Quality Check: {output_dir.name}")
    print(f"{'=' * 60}")
    print(f"Detections: {len(detections):,}")

    if not detections:
        return

    # Feature summary
    sample = detections[0].get("features", {})
    n_features = len([k for k, v in sample.items() if isinstance(v, (int, float))])
    print(f"Features per cell: {n_features}")

    # Area distribution
    areas = [
        d.get("features", {}).get("area_um2", d.get("features", {}).get("area", 0))
        for d in detections
    ]
    areas = [a for a in areas if a and a > 0]
    if areas:
        areas_arr = np.array(areas)
        print(
            f"\nArea (um2): median={np.median(areas_arr):.1f}, "
            f"mean={np.mean(areas_arr):.1f}, "
            f"range=[{np.min(areas_arr):.1f}, {np.max(areas_arr):.1f}]"
        )

    # RF scores
    scores = [d.get("rf_prediction") for d in detections if d.get("rf_prediction") is not None]
    if scores:
        scores_arr = np.array(scores)
        n_above = int((scores_arr >= 0.5).sum())
        print(
            f"\nClassifier: {len(scores):,} scored, "
            f"{n_above:,} above 0.5 ({100 * n_above / len(scores):.1f}%)"
        )

    # Marker profiles
    profiles: dict[str, int] = {}
    for d in detections:
        mp = d.get("features", {}).get("marker_profile")
        if mp:
            profiles[mp] = profiles.get(mp, 0) + 1
    if profiles:
        print("\nMarker profiles:")
        for profile, count in sorted(profiles.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / len(detections)
            print(f"  {profile}: {count:,} ({pct:.1f}%)")

    # Channel SNR summary
    channels: set[str] = set()
    for k in sample:
        if k.startswith("ch") and k.endswith("_snr"):
            ch = k.split("_")[0]  # e.g., "ch0"
            channels.add(ch)
    if channels:
        print("\nPer-channel SNR (median):")
        for ch in sorted(channels):
            snr_key = f"{ch}_snr"
            vals = [d.get("features", {}).get(snr_key, 0) for d in detections]
            vals = [v for v in vals if v > 0]
            if vals:
                med = np.median(vals)
                pct_above = 100 * sum(1 for v in vals if v >= 1.5) / len(vals)
                print(f"  {ch}: {med:.2f} ({pct_above:.0f}% above 1.5)")

    # Nuclear counting
    n_nuclei = [d.get("features", {}).get("n_nuclei") for d in detections]
    n_nuclei = [n for n in n_nuclei if n is not None]
    if n_nuclei:
        n_nuclei_arr = np.array(n_nuclei)
        print(
            f"\nNuclei/cell: median={np.median(n_nuclei_arr):.1f}, "
            f"range=[{np.min(n_nuclei_arr)}, {np.max(n_nuclei_arr)}]"
        )

    print(f"\n{'=' * 60}")


def main() -> None:
    """CLI entry point for xlseg qc."""
    parser = argparse.ArgumentParser(description="Quick quality check for pipeline output")
    parser.add_argument("output_dir", help="Pipeline output directory")
    args = parser.parse_args()
    run_qc(args)


if __name__ == "__main__":
    main()
