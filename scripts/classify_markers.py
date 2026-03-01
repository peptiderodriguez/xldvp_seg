#!/usr/bin/env python3
"""
Post-detection marker classification.

Classifies cells as marker-positive or marker-negative based on per-channel
intensity features already present in the detections JSON.

Methods:
  otsu       Otsu threshold on background-subtracted intensities (recommended)
  otsu_half  Otsu threshold / 2 (permissive, no background subtraction)
  gmm        2-component GMM on log1p intensities (P >= 0.75 = positive)

Background subtraction (--background-subtract, default with 'otsu' method):
  For each cell, estimates local background as the median intensity of the
  k=30 nearest neighboring cells. Subtracts this per-cell background from
  the cell's own intensity, then thresholds on the corrected values.
  This removes local autofluorescence variation and improves signal separation.

Usage:
    # Recommended: background-subtracted Otsu for NeuN + tdTomato
    python scripts/classify_markers.py \
        --detections cell_detections.json \
        --marker-channel 1,2 --marker-name NeuN,tdTomato \
        --method otsu

    python scripts/classify_markers.py \
        --detections cell_detections.json \
        --marker-channel 2 --marker-name tdTomato

    python scripts/classify_markers.py \
        --detections cell_detections.json \
        --marker-channel 1,3 --marker-name SMA,CD31 \
        --method gmm --output-dir output/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.json_utils import NumpyEncoder, atomic_json_dump
from segmentation.pipeline.background import (
    local_background_subtract,
    correct_all_channels,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Classification methods
# ---------------------------------------------------------------------------

def classify_otsu(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Full Otsu threshold on (background-subtracted) intensity values.

    Use after background subtraction — the noise floor is already removed,
    so the full Otsu threshold cleanly separates signal from residual noise.

    Returns (threshold, boolean mask where True = positive).
    """
    from skimage.filters import threshold_otsu

    nonzero = values[values > 0]
    if len(nonzero) < 10:
        logger.warning("Too few non-zero values for Otsu; defaulting threshold to 0")
        return 0.0, np.zeros(len(values), dtype=bool)

    if np.std(nonzero) < 1e-6:
        logger.warning("Near-zero variance in marker values; all cells classified as negative")
        return 0.0, np.zeros(len(values), dtype=bool)

    threshold = float(threshold_otsu(nonzero))
    positive = (values >= threshold) & (values > 0)
    return threshold, positive


def classify_otsu_half(values: np.ndarray) -> tuple[float, np.ndarray]:
    """Otsu threshold / 2 on raw intensity values.

    Returns (threshold, boolean mask where True = positive).
    """
    from skimage.filters import threshold_otsu

    nonzero = values[values > 0]
    if len(nonzero) < 10:
        logger.warning("Too few non-zero values for Otsu; defaulting threshold to 0")
        return 0.0, np.zeros(len(values), dtype=bool)

    if np.std(nonzero) < 1e-6:
        logger.warning("Near-zero variance in marker values; all cells classified as negative")
        return 0.0, np.zeros(len(values), dtype=bool)

    otsu_t = threshold_otsu(nonzero)
    # Otsu/2: permissive cutoff to capture weakly-expressing marker-positive cells,
    # reducing false negatives at the cost of more false positives.
    threshold = otsu_t / 2.0
    positive = (values >= threshold) & (values > 0)
    return float(threshold), positive


def classify_gmm(values: np.ndarray) -> tuple[float, np.ndarray]:
    """2-component GMM on log1p intensities.

    The component with the higher mean is treated as the 'high' (signal) class.
    Cells with posterior probability >= 0.75 for that component are positive.

    Returns (threshold, boolean mask where True = positive).
    The threshold is the value where the two component posteriors cross
    (approximate decision boundary), used for display only.
    """
    from sklearn.mixture import GaussianMixture

    if len(values) < 20:
        logger.warning("Too few values for GMM; defaulting all to negative")
        return 0.0, np.zeros(len(values), dtype=bool)

    log_vals = np.log1p(values).reshape(-1, 1)

    if np.std(log_vals) < 1e-6:
        logger.warning(f"Near-zero variance in log1p values; all cells classified as negative")
        return 0.0, np.zeros(len(values), dtype=bool)

    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(log_vals)

    # Identify which component has the higher mean
    high_idx = int(np.argmax(gmm.means_.flatten()))

    # Check if components are well-separated (warn if unimodal)
    means = gmm.means_.flatten()
    variances = gmm.covariances_.reshape(2, -1)[:, 0]
    stds = np.sqrt(variances)
    separation = abs(means[1] - means[0]) / max(stds[0] + stds[1], 1e-6)
    if separation < 0.5:
        logger.warning(f"GMM components poorly separated (separation={separation:.2f}). "
                       f"Data may be unimodal. Consider otsu_half method instead.")

    # Posterior probability for the high component
    probs = gmm.predict_proba(log_vals)[:, high_idx]
    positive = probs >= 0.75

    # Approximate threshold: crossing point in log space, convert back.
    # This weighted-midpoint formula is exact for equal priors (weights)
    # and approximate for unequal priors.
    log_threshold = (means[0] * stds[1] + means[1] * stds[0]) / (stds[0] + stds[1])
    threshold = float(np.expm1(log_threshold))

    return threshold, positive


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distribution(values: np.ndarray, threshold: float, marker_name: str,
                      method: str, n_positive: int, n_negative: int,
                      output_path: Path) -> None:
    """Histogram of marker values with threshold line."""
    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use log1p scale for x-axis if range is large
    use_log = values.max() > 10 * max(np.median(values), 1)
    plot_vals = np.log1p(values) if use_log else values
    plot_thresh = np.log1p(threshold) if use_log else threshold

    n_bins = min(100, max(30, len(values) // 20))
    ax.hist(plot_vals, bins=n_bins, color='steelblue', alpha=0.7, edgecolor='white',
            linewidth=0.5)
    ax.axvline(plot_thresh, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.1f}')

    pct = 100 * n_positive / max(n_positive + n_negative, 1)
    ax.set_title(f'{marker_name} classification ({method})\n'
                 f'{n_positive:,} positive ({pct:.1f}%) / {n_negative:,} negative')
    ax.set_xlabel('log1p(intensity)' if use_log else 'Intensity')
    ax.set_ylabel('Count')
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved distribution plot: {output_path}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def extract_marker_values(detections: list[dict], channel: int) -> np.ndarray:
    """Extract ch{N}_mean from each detection's features dict."""
    key = f'ch{channel}_mean'
    values = np.array([
        d.get('features', {}).get(key, 0.0) for d in detections
    ], dtype=np.float64)
    return values


def extract_centroids(detections: list[dict]) -> np.ndarray:
    """Extract **global** [x, y] centroids from detections.

    Uses ``global_center`` (slide-level), NOT ``features["centroid"]``
    (tile-local).  Imported ``_extract_centroids`` is the canonical impl.
    """
    from segmentation.pipeline.background import _extract_centroids
    return _extract_centroids(detections)


def classify_single_marker(detections: list[dict], channel: int,
                           marker_name: str, method: str,
                           output_dir: Path,
                           bg_subtract: bool = False,
                           centroids: np.ndarray | None = None,
                           n_neighbors: int = 30) -> dict:
    """Classify detections for one marker. Returns summary row dict.

    Mutates detections in-place: adds {marker}_class, {marker}_value,
    {marker}_threshold, and optionally {marker}_background and {marker}_snr
    to each detection's features dict.
    """
    logger.info(f"Classifying marker '{marker_name}' (ch{channel}) with method '{method}'"
                f"{f' + local background subtraction (k={n_neighbors})' if bg_subtract else ''}")

    raw_values = extract_marker_values(detections, channel)
    if len(raw_values) == 0:
        logger.warning(f"No values extracted for marker '{marker_name}' — skipping")
        return {'marker': marker_name, 'method': method, 'threshold': 0,
                'n_positive': 0, 'n_negative': 0, 'pct_positive': 0,
                'background_median': 0}

    logger.info(f"  Extracted {len(raw_values):,} values, "
                f"range [{raw_values.min():.1f}, {raw_values.max():.1f}], "
                f"median {np.median(raw_values):.1f}")

    # Local background subtraction using neighboring cells
    per_cell_bg = None
    if bg_subtract:
        if centroids is None:
            centroids = extract_centroids(detections)
        values, per_cell_bg = local_background_subtract(raw_values, centroids, n_neighbors)
        median_bg = float(np.median(per_cell_bg))
        logger.info(f"  Local background: median {median_bg:.1f}, "
                    f"range [{per_cell_bg.min():.1f}, {per_cell_bg.max():.1f}]")
        nonzero_corrected = values[values > 0]
        if len(nonzero_corrected) > 0:
            logger.info(f"  After subtraction: {(nonzero_corrected > 0).sum():,} cells with signal, "
                        f"range [{nonzero_corrected.min():.1f}, {nonzero_corrected.max():.1f}], "
                        f"median {np.median(nonzero_corrected):.1f}")
        else:
            logger.warning(f"  All values zero after background subtraction")
    else:
        values = raw_values

    # Classify
    if method == 'otsu':
        threshold, positive_mask = classify_otsu(values)
    elif method == 'otsu_half':
        threshold, positive_mask = classify_otsu_half(values)
    elif method == 'gmm':
        threshold, positive_mask = classify_gmm(values)
    else:
        raise ValueError(f"Unknown method: {method}")

    n_positive = int(positive_mask.sum())
    n_negative = len(positive_mask) - n_positive
    pct = 100 * n_positive / max(len(positive_mask), 1)

    logger.info(f"  Threshold: {threshold:.2f}"
                f"{' (on bg-subtracted values)' if bg_subtract else ''}")
    logger.info(f"  Positive: {n_positive:,} ({pct:.1f}%)  "
                f"Negative: {n_negative:,} ({100 - pct:.1f}%)")

    # Enrich detections in-place
    class_key = f'{marker_name}_class'
    value_key = f'{marker_name}_value'
    raw_key = f'{marker_name}_raw'
    thresh_key = f'{marker_name}_threshold'
    bg_key = f'{marker_name}_background'
    snr_key = f'{marker_name}_snr'

    for i, det in enumerate(detections):
        feat = det.setdefault('features', {})
        feat[class_key] = 'positive' if positive_mask[i] else 'negative'
        feat[value_key] = float(values[i])  # bg-subtracted if enabled
        feat[thresh_key] = float(threshold)
        if bg_subtract and per_cell_bg is not None:
            feat[raw_key] = float(raw_values[i])
            feat[bg_key] = float(per_cell_bg[i])
            # SNR = signal / background
            feat[snr_key] = float(raw_values[i] / per_cell_bg[i]) if per_cell_bg[i] > 0 else 0.0

    # Plot (use bg-subtracted values for the histogram)
    plot_path = output_dir / f'{marker_name}_distribution.png'
    plot_distribution(values, threshold, marker_name,
                      f'{method}+bgsub' if bg_subtract else method,
                      n_positive, n_negative, plot_path)

    return {
        'marker': marker_name,
        'method': method,
        'threshold': threshold,
        'background_median': float(np.median(per_cell_bg)) if per_cell_bg is not None else 0,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'pct_positive': round(pct, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Post-detection marker classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--detections', required=True,
                        help='Path to detections JSON file')
    parser.add_argument('--marker-channel', default=None,
                        help='Comma-separated channel indices (e.g. 2 or 1,3). '
                             'Required unless --marker-wavelength is used.')
    parser.add_argument('--marker-wavelength', default=None,
                        help='Comma-separated wavelengths instead of indices (e.g. "647,555"). '
                             'Resolved to channel indices via CZI metadata. '
                             'Requires --czi-path. Alternative to --marker-channel.')
    parser.add_argument('--czi-path', default=None,
                        help='CZI file path (required for --marker-wavelength resolution)')
    parser.add_argument('--marker-name', required=True,
                        help='Comma-separated marker names matching channels '
                             '(e.g. tdTomato or SMA,CD31)')
    parser.add_argument('--method', default='otsu',
                        choices=['otsu', 'otsu_half', 'gmm'],
                        help='Classification method (default: otsu)')
    parser.add_argument('--background-subtract', action='store_true', default=None,
                        help='Subtract estimated background before thresholding. '
                             'Default: on for otsu, off for otsu_half/gmm.')
    parser.add_argument('--no-background-subtract', action='store_true',
                        help='Disable background subtraction even for otsu method.')
    parser.add_argument('--correct-all-channels', action='store_true',
                        help='Background-correct ALL ch{N}_mean features (not just marker channels). '
                             'Overwrites ch{N}_mean with corrected values, saves originals as ch{N}_mean_raw.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same dir as detections)')
    args = parser.parse_args()

    # Background subtraction: default on for otsu, off for others
    if args.no_background_subtract:
        args.bg_subtract = False
    elif args.background_subtract is not None:
        args.bg_subtract = args.background_subtract
    else:
        args.bg_subtract = (args.method == 'otsu')

    # Validate: must have either --marker-channel or --marker-wavelength
    if not args.marker_channel and not args.marker_wavelength:
        parser.error("Either --marker-channel or --marker-wavelength is required")
    if args.marker_wavelength and not args.czi_path:
        parser.error("--marker-wavelength requires --czi-path for wavelength-to-index resolution")

    return args


def main():
    args = parse_args()
    setup_logging(level="INFO")

    # Validate paths
    det_path = Path(args.detections)
    if not det_path.exists():
        logger.error(f"Detections file not found: {det_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else det_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse marker channels — resolve wavelengths if specified
    if args.marker_wavelength:
        from segmentation.io.czi_loader import get_czi_metadata, resolve_channel_indices, ChannelResolutionError
        try:
            meta = get_czi_metadata(args.czi_path)
            wavelengths = [w.strip() for w in args.marker_wavelength.split(',')]
            filename = Path(args.czi_path).stem
            resolved = resolve_channel_indices(meta, wavelengths, filename)
            channels = [resolved[w] for w in wavelengths]
            logger.info(f"Resolved wavelengths {wavelengths} -> channel indices {channels}")
        except ChannelResolutionError as e:
            logger.error(f"Wavelength resolution failed: {e}")
            sys.exit(1)
    else:
        try:
            channels = [int(c.strip()) for c in args.marker_channel.split(',')]
        except ValueError:
            logger.error(f"Invalid --marker-channel: {args.marker_channel!r} "
                         f"(expected comma-separated integers)")
            sys.exit(1)

    names = [n.strip() for n in args.marker_name.split(',')]

    if len(channels) != len(names):
        logger.error(f"Mismatch: {len(channels)} channel(s) but {len(names)} name(s). "
                     f"Channels: {channels}, Names: {names}")
        sys.exit(1)

    # Load detections
    logger.info(f"Loading detections from {det_path}...")
    with open(det_path) as f:
        detections = json.load(f)
    logger.info(f"Loaded {len(detections):,} detections")

    if not detections:
        logger.error("No detections found in file")
        sys.exit(1)

    # Verify that requested channels have data
    sample_feat = detections[0].get('features', {})
    for ch, name in zip(channels, names):
        key = f'ch{ch}_mean'
        if key not in sample_feat:
            available = [k for k in sample_feat if k.startswith('ch') and k.endswith('_mean')]
            logger.warning(f"Channel key '{key}' not found in first detection's features. "
                           f"Available: {available}. Values will default to 0.")

    # Extract centroids if any background subtraction is needed
    centroids = None
    if args.bg_subtract or args.correct_all_channels:
        centroids = extract_centroids(detections)
        logger.info(f"Extracted {len(centroids):,} centroids for local background subtraction")

    # Background-correct ALL channels if requested (before marker classification)
    # Guard: skip if pipeline already did background correction
    if args.correct_all_channels:
        sample_feat = detections[0].get('features', {})
        if any(k.endswith('_background') for k in sample_feat):
            logger.info("Detections already background-corrected by pipeline -- skipping correct_all_channels")
            args.bg_subtract = False
        else:
            corrected_channels = correct_all_channels(detections, centroids=centroids)
            logger.info(f"Corrected {len(corrected_channels)} channels: {corrected_channels}")
            # Marker classification will now operate on already-corrected ch{N}_mean values,
            # so disable per-marker background subtraction to avoid double-correction
            args.bg_subtract = False

    # Process each marker
    summaries = []
    for ch, name in zip(channels, names):
        row = classify_single_marker(detections, ch, name, args.method, output_dir,
                                     bg_subtract=args.bg_subtract,
                                     centroids=centroids)
        summaries.append(row)

    # Create combined marker_profile when multiple markers are classified
    if len(names) > 1:
        logger.info("Creating combined marker_profile field...")
        class_keys = [f'{name}_class' for name in names]
        profile_counts = {}
        for det in detections:
            feat = det.get('features', {})
            parts = []
            for name, key in zip(names, class_keys):
                cls = feat.get(key, 'negative')
                symbol = '+' if cls == 'positive' else '-'
                parts.append(f'{name}{symbol}')
            profile = '/'.join(parts)
            feat['marker_profile'] = profile
            profile_counts[profile] = profile_counts.get(profile, 0) + 1

        logger.info("  Marker profiles:")
        for profile, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(detections)
            logger.info(f"    {profile}: {count:,} ({pct:.1f}%)")

    # Save enriched detections
    out_json = output_dir / f'{det_path.stem}_classified.json'
    logger.info(f"Saving classified detections to {out_json}...")
    atomic_json_dump(detections, out_json)
    logger.info(f"  Wrote {len(detections):,} detections")

    # Save summary CSV
    summary_path = output_dir / 'marker_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'marker', 'method', 'threshold', 'background_median',
            'n_positive', 'n_negative', 'pct_positive',
        ])
        writer.writeheader()
        writer.writerows(summaries)
    logger.info(f"  Wrote summary: {summary_path}")

    # Final summary
    logger.info("--- Classification complete ---")
    for row in summaries:
        bg_info = f", bg={row['background_median']:.1f}" if row.get('background_median') else ''
        logger.info(f"  {row['marker']}: {row['n_positive']:,} positive "
                     f"({row['pct_positive']:.1f}%), "
                     f"threshold={row['threshold']:.2f} ({row['method']}{bg_info})")


if __name__ == '__main__':
    main()
