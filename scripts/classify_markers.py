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
    _extract_centroids,
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

def extract_marker_values(detections: list[dict], channel: int,
                          feature: str = 'mean',
                          use_raw: bool = False) -> np.ndarray:
    """Extract per-cell intensity values from detection features.

    Args:
        channel: CZI channel index.
        feature: Intensity statistic — 'mean', 'median', 'p75', 'p95', etc.
        use_raw: If True, read ``ch{N}_{feature}_raw`` (pre-bg-correction).
    """
    suffix = f'{feature}_raw' if use_raw else feature
    key = f'ch{channel}_{suffix}'
    values = np.array([
        d.get('features', {}).get(key, 0.0) for d in detections
    ], dtype=np.float64)

    # Warn if all zeros (likely missing feature key)
    if values.max() == 0:
        fallback_key = f'ch{channel}_{feature}'
        if use_raw:
            logger.warning(f"Key '{key}' not found or all zeros — "
                           f"falling back to '{fallback_key}'")
            values = np.array([
                d.get('features', {}).get(fallback_key, 0.0) for d in detections
            ], dtype=np.float64)
    return values


def classify_single_marker(detections: list[dict], channel: int,
                           marker_name: str, method: str,
                           output_dir: Path,
                           bg_subtract: bool = False,
                           global_background: bool = False,
                           centroids: np.ndarray | None = None,
                           n_neighbors: int = 30,
                           snr_threshold: float = 1.5,
                           intensity_feature: str = 'mean',
                           use_raw: bool = False,
                           cv_max: float | None = None) -> dict:
    """Classify detections for one marker. Returns summary row dict.

    Mutates detections in-place: adds {marker}_class, {marker}_value,
    {marker}_threshold, and optionally {marker}_background and {marker}_snr
    to each detection's features dict.

    Args:
        intensity_feature: Which stat to threshold on ('mean', 'median', 'p75', 'p95').
        use_raw: Use pre-bg-correction values (``ch{N}_{feat}_raw``).
        global_background: Subtract slide-wide median (not local neighbors).
            Avoids inflating background in expression zones.
        cv_max: If set, cells with ``ch{N}_cv > cv_max`` are classified negative
            regardless of intensity (filters out particulate noise).
    """
    feat_label = f"{intensity_feature}{'_raw' if use_raw else ''}"
    bg_label = (' + global bg subtraction' if global_background
                else f' + local bg subtraction (k={n_neighbors})' if bg_subtract
                else '')
    logger.info(f"Classifying marker '{marker_name}' (ch{channel}) with method '{method}' "
                f"on {feat_label}{bg_label}"
                f"{f' + CV filter (max={cv_max})' if cv_max else ''}")

    raw_values = extract_marker_values(detections, channel,
                                       feature=intensity_feature, use_raw=use_raw)
    if len(raw_values) == 0:
        logger.warning(f"No values extracted for marker '{marker_name}' — skipping")
        return {'marker': marker_name, 'method': method, 'threshold': 0,
                'n_positive': 0, 'n_negative': 0, 'pct_positive': 0,
                'background_median': 0, 'cv_filtered': 0}

    logger.info(f"  Extracted {len(raw_values):,} values, "
                f"range [{raw_values.min():.1f}, {raw_values.max():.1f}], "
                f"median {np.median(raw_values):.1f}")

    # Background subtraction
    per_cell_bg = None
    if global_background:
        # Global: subtract slide-wide median from all cells.
        # Avoids local-neighbor inflation for regional markers.
        global_bg = float(np.median(raw_values))
        values = np.maximum(raw_values - global_bg, 0.0)
        per_cell_bg = np.full(len(raw_values), global_bg)
        nonzero = values[values > 0]
        logger.info(f"  Global background: {global_bg:.1f} (slide-wide median)")
        logger.info(f"  After subtraction: {len(nonzero):,}/{len(values):,} cells with signal, "
                    f"range [{nonzero.min():.1f}, {nonzero.max():.1f}], "
                    f"median {np.median(nonzero):.1f}" if len(nonzero) > 0 else
                    "  All values zero after subtraction")
    elif bg_subtract:
        if centroids is None:
            centroids = _extract_centroids(detections)
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

    # CV filter: read coefficient of variation for this channel
    cv_mask = None
    n_cv_filtered = 0
    if cv_max is not None:
        cv_suffix = 'cv_raw' if use_raw else 'cv'
        cv_key = f'ch{channel}_{cv_suffix}'
        cv_values = np.array([
            d.get('features', {}).get(cv_key, 0.0) for d in detections
        ], dtype=np.float64)
        # Fallback: if _raw version missing, try plain cv
        if cv_values.max() == 0 and use_raw:
            cv_key = f'ch{channel}_cv'
            cv_values = np.array([
                d.get('features', {}).get(cv_key, 0.0) for d in detections
            ], dtype=np.float64)
        cv_mask = cv_values <= cv_max
        n_cv_filtered = int((~cv_mask).sum())
        logger.info(f"  CV filter: {n_cv_filtered:,} cells with CV > {cv_max:.1f} "
                    f"(will be forced negative)")

    # Classify
    if method == 'snr':
        # SNR-based: positive if channel SNR >= snr_threshold
        # First try pipeline-computed ch{N}_snr, then compute from raw/bg
        snr_key_ch = f'ch{channel}_snr'
        pipeline_snr = np.array([
            d.get('features', {}).get(snr_key_ch, 0.0) for d in detections
        ], dtype=np.float64)
        if np.any(pipeline_snr > 0):
            snr_values = pipeline_snr
            logger.info(f"  Using pipeline-computed {snr_key_ch} for SNR classification")
        elif per_cell_bg is not None:
            snr_values = np.where(per_cell_bg > 0, raw_values / per_cell_bg, 0.0)
            logger.info(f"  Computing SNR from raw_values / per_cell_bg")
        else:
            raise ValueError(
                f"SNR method requires either pipeline bg correction (ch{channel}_snr in features) "
                "or --background-subtract. Neither found.")
        threshold = snr_threshold
        positive_mask = snr_values >= snr_threshold
    elif method == 'otsu':
        threshold, positive_mask = classify_otsu(values)
    elif method == 'otsu_half':
        threshold, positive_mask = classify_otsu_half(values)
    elif method == 'gmm':
        threshold, positive_mask = classify_gmm(values)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply CV filter (force high-CV cells to negative)
    if cv_mask is not None:
        positive_mask = positive_mask & cv_mask

    n_positive = int(positive_mask.sum())
    n_negative = len(positive_mask) - n_positive
    pct = 100 * n_positive / max(len(positive_mask), 1)

    logger.info(f"  Threshold: {threshold:.2f}"
                f"{' (on bg-subtracted values)' if bg_subtract else ''}")
    logger.info(f"  Positive: {n_positive:,} ({pct:.1f}%)  "
                f"Negative: {n_negative:,} ({100 - pct:.1f}%)"
                f"{f'  (CV filtered: {n_cv_filtered:,})' if n_cv_filtered else ''}")

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
            feat[snr_key] = float(raw_values[i] / per_cell_bg[i]) if per_cell_bg[i] > 0 else 0.0

    # Plot (use the values that were actually thresholded)
    plot_path = output_dir / f'{marker_name}_distribution.png'
    plot_distribution(values, threshold, marker_name,
                      f'{method} on {feat_label}'
                      + ('+bgsub' if bg_subtract else '')
                      + (f'+cv<={cv_max}' if cv_max else ''),
                      n_positive, n_negative, plot_path)

    return {
        'marker': marker_name,
        'method': method,
        'feature': feat_label,
        'threshold': threshold,
        'background_median': float(np.median(per_cell_bg)) if per_cell_bg is not None else 0,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'pct_positive': round(pct, 2),
        'cv_filtered': n_cv_filtered,
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
                        choices=['otsu', 'otsu_half', 'gmm', 'snr'],
                        help='Default classification method (default: otsu). '
                             'Override per-marker with --methods.')
    parser.add_argument('--methods', default=None,
                        help='Per-marker methods, comma-separated matching --marker-name order '
                             '(e.g. "snr,otsu,otsu"). Choices: otsu, otsu_half, gmm, snr. '
                             'Falls back to --method for unspecified.')
    parser.add_argument('--snr-threshold', type=float, default=1.5,
                        help='Default SNR threshold for --method snr (default: 1.5). '
                             'Overridden per-marker by --snr-thresholds if provided.')
    parser.add_argument('--snr-thresholds', default=None,
                        help='Per-marker SNR thresholds, comma-separated matching --marker-name order '
                             '(e.g. "3.0,4.0,2.5" for DCN>=3, GluI>=4, Pck1>=2.5). '
                             'Falls back to --snr-threshold for any unspecified markers.')
    parser.add_argument('--intensity-feature', default='mean',
                        choices=['mean', 'median', 'p25', 'p75', 'p95'],
                        help='Default intensity statistic to threshold on (default: mean). '
                             'Override per-marker with --intensity-features.')
    parser.add_argument('--intensity-features', default=None,
                        help='Per-marker intensity features, comma-separated matching --marker-name '
                             '(e.g. "p95,median,p95"). Choices: mean, median, p25, p75, p95.')
    parser.add_argument('--use-raw', action='store_true',
                        help='Use pre-bg-correction values (ch{N}_*_raw). '
                             'Avoids local background inflation for regional markers.')
    parser.add_argument('--cv-max', type=float, default=None,
                        help='Max coefficient of variation — cells with CV above this '
                             'are forced negative (filters particulate noise). '
                             'Typical: 1.0-2.0 for diffuse markers.')
    parser.add_argument('--global-background', action='store_true',
                        help='Subtract slide-wide median (global) instead of local neighbors. '
                             'Avoids inflating background for regional markers (e.g. GluI, Pck1). '
                             'Best combined with --use-raw --intensity-feature median.')
    parser.add_argument('--background-subtract', action='store_true', default=None,
                        help='Subtract local (k-nearest neighbor) background before thresholding. '
                             'Default: on for otsu, off for otsu_half/gmm.')
    parser.add_argument('--no-background-subtract', action='store_true',
                        help='Disable background subtraction even for otsu method.')
    parser.add_argument('--correct-all-channels', action='store_true',
                        help='Background-correct ALL ch{N}_mean features (not just marker channels). '
                             'Overwrites ch{N}_mean with corrected values, saves originals as ch{N}_mean_raw.')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same dir as detections)')
    args = parser.parse_args()

    # Background subtraction: default on for otsu and snr, off for others.
    # --use-raw implies no bg subtraction (raw values are pre-correction).
    if args.no_background_subtract or args.use_raw:
        args.bg_subtract = False
    elif args.background_subtract is not None:
        args.bg_subtract = args.background_subtract
    else:
        args.bg_subtract = (args.method in ('otsu', 'snr'))

    # Auto-detect pipeline-corrected detections and disable bg subtraction
    # to prevent double correction.  This is set after loading detections
    # (see below, before marker classification loop).

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
    from segmentation.io.czi_loader import get_czi_metadata, resolve_channel_indices, ChannelResolutionError
    if args.marker_wavelength:
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

    # Build channel label map from CZI metadata (if available)
    _czi_ch_labels = {}
    if args.czi_path:
        try:
            _meta = get_czi_metadata(args.czi_path)
            for _ch in _meta['channels']:
                fluor = (_ch.get('fluorophore') or _ch.get('name') or '').strip()
                em = _ch.get('emission_nm')
                _czi_ch_labels[_ch['index']] = f"{fluor} em={em:.0f}nm" if em else fluor
        except Exception as _e:
            logger.debug(f"Could not load CZI channel metadata: {_e}")

    # Print marker classification summary
    logger.info("=" * 70)
    logger.info("MARKER CLASSIFICATION")
    logger.info("=" * 70)
    feat_label = f"{args.intensity_feature}{'_raw' if args.use_raw else ''}"
    logger.info(f"  Feature: {feat_label}")
    if args.cv_max:
        logger.info(f"  CV filter: max {args.cv_max}")
    for ch, name in zip(channels, names):
        ch_detail = _czi_ch_labels.get(ch, '?')
        logger.info(f"  '{name}':  C={ch} ({ch_detail})  method={args.method}")
    logger.info("=" * 70)

    # Load detections (use fast_json_load for large files — orjson is 3-5x faster)
    from segmentation.utils.json_utils import fast_json_load
    logger.info(f"Loading detections from {det_path}...")
    detections = fast_json_load(det_path)
    logger.info(f"Loaded {len(detections):,} detections")

    if not detections:
        logger.error("No detections found in file")
        sys.exit(1)

    # Verify that requested channels have data
    sample_feat = detections[0].get('features', {})
    feat_suffix = f'{args.intensity_feature}_raw' if args.use_raw else args.intensity_feature
    for ch, name in zip(channels, names):
        key = f'ch{ch}_{feat_suffix}'
        if key not in sample_feat:
            available = [k for k in sample_feat if k.startswith(f'ch{ch}_')]
            logger.warning(f"Channel key '{key}' not found in first detection's features. "
                           f"Available ch{ch}_* keys: {available}. Values will default to 0.")

    # Extract centroids if any background subtraction is needed
    centroids = None
    if args.bg_subtract or args.correct_all_channels:
        centroids = _extract_centroids(detections)
        logger.info(f"Extracted {len(centroids):,} centroids for local background subtraction")

    # Guard: detect pipeline-corrected detections and disable ALL bg subtraction
    # to prevent double correction.  Pipeline post-dedup writes ch{N}_background keys.
    _pipeline_corrected = False
    if detections:
        sample_feat = detections[0].get('features', {})
        _pipeline_corrected = any(k.endswith('_background') for k in sample_feat)
    if _pipeline_corrected:
        if args.bg_subtract:
            logger.info("Detections already background-corrected by pipeline — "
                        "disabling per-marker bg subtraction to prevent double correction")
            args.bg_subtract = False
        if args.correct_all_channels:
            logger.info("Detections already background-corrected by pipeline — "
                        "skipping --correct-all-channels")
            args.correct_all_channels = False

    # Background-correct ALL channels if requested (for older detections only)
    if args.correct_all_channels:
        corrected_channels = correct_all_channels(detections, centroids=centroids)
        if corrected_channels:
            logger.info(f"Corrected {len(corrected_channels)} channels: {corrected_channels}")
            # Marker classification now operates on corrected ch{N}_mean values,
            # so disable per-marker bg subtraction to avoid double-correction
            args.bg_subtract = False

    # Parse per-marker SNR thresholds
    per_marker_snr = {}
    if args.snr_thresholds:
        snr_vals = [float(x.strip()) for x in args.snr_thresholds.split(',')]
        if len(snr_vals) != len(names):
            raise ValueError(f"--snr-thresholds has {len(snr_vals)} values but "
                             f"--marker-name has {len(names)} markers: {names}")
        per_marker_snr = dict(zip(names, snr_vals))
        logger.info(f"Per-marker SNR thresholds: {per_marker_snr}")

    # Parse per-marker methods
    per_marker_method = {}
    if args.methods:
        method_list = [m.strip() for m in args.methods.split(',')]
        if len(method_list) != len(names):
            raise ValueError(f"--methods has {len(method_list)} values but "
                             f"--marker-name has {len(names)} markers: {names}")
        valid_methods = {'otsu', 'otsu_half', 'gmm', 'snr'}
        for m in method_list:
            if m not in valid_methods:
                raise ValueError(f"Invalid method '{m}' in --methods. Choices: {valid_methods}")
        per_marker_method = dict(zip(names, method_list))
        logger.info(f"Per-marker methods: {per_marker_method}")

    # Parse per-marker intensity features
    per_marker_feature = {}
    if args.intensity_features:
        feat_list = [f.strip() for f in args.intensity_features.split(',')]
        if len(feat_list) != len(names):
            raise ValueError(f"--intensity-features has {len(feat_list)} values but "
                             f"--marker-name has {len(names)} markers: {names}")
        valid_feats = {'mean', 'median', 'p25', 'p75', 'p95'}
        for f in feat_list:
            if f not in valid_feats:
                raise ValueError(f"Invalid feature '{f}' in --intensity-features. Choices: {valid_feats}")
        per_marker_feature = dict(zip(names, feat_list))
        logger.info(f"Per-marker intensity features: {per_marker_feature}")

    # Process each marker
    summaries = []
    for ch, name in zip(channels, names):
        marker_snr = per_marker_snr.get(name, args.snr_threshold)
        marker_method = per_marker_method.get(name, args.method)
        marker_feature = per_marker_feature.get(name, args.intensity_feature)

        # For SNR method, disable bg_subtract (uses pipeline SNR directly)
        # For otsu with --use-raw, disable local bg_subtract (raw values bypass correction)
        # Global bg is always allowed (it's a simple slide-wide median)
        # Derive bg_subtract from per-marker method, not global default
        if args.no_background_subtract or args.use_raw:
            marker_bg_subtract = False
        elif args.background_subtract is not None:
            marker_bg_subtract = args.background_subtract
        else:
            marker_bg_subtract = (marker_method in ('otsu', 'snr'))
        marker_global_bg = args.global_background
        if marker_method == 'snr':
            marker_bg_subtract = False  # SNR reads from pipeline features
            marker_global_bg = False    # SNR has its own bg handling
        elif args.use_raw and not args.global_background:
            marker_bg_subtract = False  # raw values are pre-correction

        row = classify_single_marker(detections, ch, name, marker_method, output_dir,
                                     bg_subtract=marker_bg_subtract,
                                     global_background=marker_global_bg,
                                     centroids=centroids,
                                     snr_threshold=marker_snr,
                                     intensity_feature=marker_feature,
                                     use_raw=args.use_raw,
                                     cv_max=args.cv_max)
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

    # Save enriched detections — strip existing _classified suffix to prevent
    # doubling (e.g., cell_detections_classified_classified.json) when chaining
    stem = det_path.stem
    if stem.endswith('_classified'):
        stem = stem[:-len('_classified')]
    out_json = output_dir / f'{stem}_classified.json'
    logger.info(f"Saving classified detections to {out_json}...")
    atomic_json_dump(detections, out_json)
    logger.info(f"  Wrote {len(detections):,} detections")

    # Save summary CSV
    summary_path = output_dir / 'marker_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'marker', 'method', 'feature', 'threshold', 'background_median',
            'n_positive', 'n_negative', 'pct_positive', 'cv_filtered',
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
