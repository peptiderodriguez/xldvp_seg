#!/usr/bin/env python3
"""
Post-detection marker classification.

Classifies cells as marker-positive or marker-negative based on per-channel
intensity features already present in the detections JSON.

Methods:
  snr        Median-based SNR >= threshold (default 1.5). Recommended.
             SNR = median_raw / median_of_neighbor_medians.
  otsu       Otsu threshold on background-subtracted intensities (fallback)
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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from xldvp_seg.pipeline.background import (
    _extract_centroids,
    correct_all_channels,
)
from xldvp_seg.utils.json_utils import atomic_json_dump
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Import reusable functions from package module
from xldvp_seg.analysis.marker_classification import (  # noqa: E402
    classify_gmm,
    classify_otsu,
    classify_otsu_half,
    classify_single_marker,
    extract_marker_values,
    plot_distribution,
)

# Re-export for backward compat (other scripts may import from here)
__all__ = [
    "classify_otsu",
    "classify_otsu_half",
    "classify_gmm",
    "extract_marker_values",
    "classify_single_marker",
    "plot_distribution",
]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-detection marker classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--detections", required=True, help="Path to detections JSON file")
    parser.add_argument(
        "--marker-channel",
        default=None,
        help="Comma-separated channel indices (e.g. 2 or 1,3). "
        "Required unless --marker-wavelength is used.",
    )
    parser.add_argument(
        "--marker-wavelength",
        default=None,
        help='Comma-separated wavelengths instead of indices (e.g. "647,555"). '
        "Resolved to channel indices via CZI metadata. "
        "Requires --czi-path. Alternative to --marker-channel.",
    )
    parser.add_argument(
        "--czi-path",
        default=None,
        help="CZI file path (required for --marker-wavelength resolution)",
    )
    parser.add_argument(
        "--marker-name",
        required=True,
        help="Comma-separated marker names matching channels " "(e.g. tdTomato or SMA,CD31)",
    )
    parser.add_argument(
        "--method",
        default="snr",
        choices=["snr", "otsu", "otsu_half", "gmm"],
        help="Default classification method (default: snr). " "Override per-marker with --methods.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Per-marker methods, comma-separated matching --marker-name order "
        '(e.g. "snr,otsu,otsu"). Choices: otsu, otsu_half, gmm, snr. '
        "Falls back to --method for unspecified.",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=1.5,
        help="Default SNR threshold for --method snr (default: 1.5). "
        "Overridden per-marker by --snr-thresholds if provided.",
    )
    parser.add_argument(
        "--snr-thresholds",
        default=None,
        help="Per-marker SNR thresholds, comma-separated matching --marker-name order "
        '(e.g. "3.0,4.0,2.5" for DCN>=3, GluI>=4, Pck1>=2.5). '
        "Falls back to --snr-threshold for any unspecified markers.",
    )
    parser.add_argument(
        "--intensity-feature",
        default="snr",
        choices=["snr", "mean", "median", "p25", "p75", "p95"],
        help="Default intensity statistic to threshold on (default: snr). "
        "Override per-marker with --intensity-features.",
    )
    parser.add_argument(
        "--intensity-features",
        default=None,
        help="Per-marker intensity features, comma-separated matching --marker-name "
        '(e.g. "p95,median,p95"). Choices: mean, median, p25, p75, p95.',
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use pre-bg-correction values (ch{N}_*_raw). "
        "Avoids local background inflation for regional markers.",
    )
    parser.add_argument(
        "--cv-max",
        type=float,
        default=None,
        help="Max coefficient of variation — cells with CV above this "
        "are forced negative (filters particulate noise). "
        "Typical: 1.0-2.0 for diffuse markers.",
    )
    parser.add_argument(
        "--global-background",
        action="store_true",
        help="Subtract slide-wide median (global) instead of local neighbors. "
        "Avoids inflating background for regional markers (e.g. GluI, Pck1). "
        "Best combined with --use-raw --intensity-feature median.",
    )
    parser.add_argument(
        "--background-subtract",
        action="store_true",
        default=None,
        help="Subtract local (k-nearest neighbor) background before thresholding. "
        "Default: on for otsu, off for otsu_half/gmm.",
    )
    parser.add_argument(
        "--no-background-subtract",
        action="store_true",
        help="Disable background subtraction even for otsu method.",
    )
    parser.add_argument(
        "--normalize-channel",
        default=None,
        help="Normalize each marker SNR by this channel's SNR before thresholding. "
        'Use wavelength (e.g. "647") or index (e.g. "1"). Requires --czi-path '
        "for wavelength resolution. Effective metric: marker_snr / normalize_snr. "
        "Filters autofluorescent cells that are bright in all channels.",
    )
    parser.add_argument(
        "--correct-all-channels",
        action="store_true",
        help="Background-correct ALL ch{N}_mean features (not just marker channels). "
        "Overwrites ch{N}_mean with corrected values, saves originals as ch{N}_mean_raw.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: same dir as detections)"
    )
    args = parser.parse_args()

    # Background subtraction: default on for otsu only.
    # SNR uses pipeline-computed SNR directly — no bg subtraction needed.
    if args.no_background_subtract or args.use_raw:
        args.bg_subtract = False
    elif args.background_subtract is not None:
        args.bg_subtract = args.background_subtract
    else:
        args.bg_subtract = args.method == "otsu"

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
    # Guard: if output_dir looks like a JSON file path, use its parent instead
    if output_dir.suffix == ".json":
        logger.warning(f"--output-dir looks like a file path ({output_dir}), using parent dir")
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse marker channels — resolve wavelengths if specified
    from xldvp_seg.io.czi_loader import (
        ChannelResolutionError,
        get_czi_metadata,
        resolve_channel_indices,
    )

    # Load CZI metadata once (used for wavelength resolution, channel labels, normalize-channel)
    czi_meta = get_czi_metadata(args.czi_path) if args.czi_path else None
    if args.marker_wavelength:
        try:
            meta = czi_meta
            wavelengths = [w.strip() for w in args.marker_wavelength.split(",")]
            filename = Path(args.czi_path).stem
            resolved = resolve_channel_indices(meta, wavelengths, filename)
            channels = [resolved[w] for w in wavelengths]
            logger.info(f"Resolved wavelengths {wavelengths} -> channel indices {channels}")
        except ChannelResolutionError as e:
            logger.error(f"Wavelength resolution failed: {e}")
            sys.exit(1)
    else:
        try:
            channels = [int(c.strip()) for c in args.marker_channel.split(",")]
        except ValueError:
            logger.error(
                f"Invalid --marker-channel: {args.marker_channel!r} "
                f"(expected comma-separated integers)"
            )
            sys.exit(1)

    names = [n.strip() for n in args.marker_name.split(",")]

    if len(channels) != len(names):
        logger.error(
            f"Mismatch: {len(channels)} channel(s) but {len(names)} name(s). "
            f"Channels: {channels}, Names: {names}"
        )
        sys.exit(1)

    # Build channel label map from CZI metadata (if available)
    _czi_ch_labels = {}
    if czi_meta:
        try:
            for _ch in czi_meta["channels"]:
                fluor = (_ch.get("fluorophore") or _ch.get("name") or "").strip()
                em = _ch.get("emission_nm")
                _czi_ch_labels[_ch["index"]] = f"{fluor} em={em:.0f}nm" if em else fluor
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
        ch_detail = _czi_ch_labels.get(ch, "?")
        logger.info(f"  '{name}':  C={ch} ({ch_detail})  method={args.method}")
    logger.info("=" * 70)

    # Load detections (use fast_json_load for large files — orjson is 3-5x faster)
    from xldvp_seg.utils.json_utils import fast_json_load

    logger.info(f"Loading detections from {det_path}...")
    detections = fast_json_load(det_path)
    logger.info(f"Loaded {len(detections):,} detections")

    if not detections:
        logger.error("No detections found in file")
        sys.exit(1)

    # Verify that requested channels have data
    sample_feat = detections[0].get("features", {})
    feat_suffix = f"{args.intensity_feature}_raw" if args.use_raw else args.intensity_feature
    for ch, name in zip(channels, names):
        key = f"ch{ch}_{feat_suffix}"
        if key not in sample_feat:
            available = [k for k in sample_feat if k.startswith(f"ch{ch}_")]
            logger.warning(
                f"Channel key '{key}' not found in first detection's features. "
                f"Available ch{ch}_* keys: {available}. Values will default to 0."
            )

    # Extract centroids if any background subtraction is needed
    centroids = None
    if args.bg_subtract or args.correct_all_channels:
        centroids = _extract_centroids(detections)
        logger.info(f"Extracted {len(centroids):,} centroids for local background subtraction")

    # Guard: detect pipeline-corrected detections and disable ALL bg subtraction
    # to prevent double correction.  Pipeline post-dedup writes ch{N}_background keys.
    _pipeline_corrected = False
    if detections:
        _sample_keys: set[str] = set()
        for d in detections[:10]:
            _sample_keys.update(d.get("features", {}).keys())
        _pipeline_corrected = any(k.endswith("_background") for k in _sample_keys)
    if _pipeline_corrected:
        if args.bg_subtract:
            logger.info(
                "Detections already background-corrected by pipeline — "
                "disabling per-marker bg subtraction to prevent double correction"
            )
            args.bg_subtract = False
        if args.correct_all_channels:
            logger.info(
                "Detections already background-corrected by pipeline — "
                "skipping --correct-all-channels"
            )
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
        snr_vals = [float(x.strip()) for x in args.snr_thresholds.split(",")]
        if len(snr_vals) != len(names):
            raise ValueError(
                f"--snr-thresholds has {len(snr_vals)} values but "
                f"--marker-name has {len(names)} markers: {names}"
            )
        per_marker_snr = dict(zip(names, snr_vals))
        logger.info(f"Per-marker SNR thresholds: {per_marker_snr}")

    # Parse per-marker methods
    per_marker_method = {}
    if args.methods:
        method_list = [m.strip() for m in args.methods.split(",")]
        if len(method_list) != len(names):
            raise ValueError(
                f"--methods has {len(method_list)} values but "
                f"--marker-name has {len(names)} markers: {names}"
            )
        valid_methods = {"otsu", "otsu_half", "gmm", "snr"}
        for m in method_list:
            if m not in valid_methods:
                raise ValueError(f"Invalid method '{m}' in --methods. Choices: {valid_methods}")
        per_marker_method = dict(zip(names, method_list))
        logger.info(f"Per-marker methods: {per_marker_method}")

    # Parse per-marker intensity features
    per_marker_feature = {}
    if args.intensity_features:
        feat_list = [f.strip() for f in args.intensity_features.split(",")]
        if len(feat_list) != len(names):
            raise ValueError(
                f"--intensity-features has {len(feat_list)} values but "
                f"--marker-name has {len(names)} markers: {names}"
            )
        valid_feats = {"snr", "mean", "median", "p25", "p75", "p95"}
        for f in feat_list:
            if f not in valid_feats:
                raise ValueError(
                    f"Invalid feature '{f}' in --intensity-features. Choices: {valid_feats}"
                )
        per_marker_feature = dict(zip(names, feat_list))
        logger.info(f"Per-marker intensity features: {per_marker_feature}")

    # Resolve normalize channel (for SNR normalization against a reference like PM)
    normalize_ch = None
    normalize_snr = None
    if args.normalize_channel:
        nc = args.normalize_channel.strip()
        # Always resolve through CZI metadata (handles both index and wavelength)
        if czi_meta:
            try:
                nc_resolved = resolve_channel_indices(czi_meta, [nc], Path(args.czi_path).stem)
                normalize_ch = nc_resolved[nc]
            except ChannelResolutionError as e:
                logger.error(f"Cannot resolve --normalize-channel '{nc}': {e}")
                sys.exit(1)
        elif nc.isdigit() and int(nc) < 20:
            normalize_ch = int(nc)
        else:
            logger.error("--normalize-channel requires --czi-path for wavelength resolution")
            sys.exit(1)
        logger.info(
            f"  Normalize channel: ch{normalize_ch} " f"({_czi_ch_labels.get(normalize_ch, '?')})"
        )
        # Extract normalize channel SNR for all detections
        norm_snr_key = f"ch{normalize_ch}_snr"
        normalize_snr = np.array(
            [d.get("features", {}).get(norm_snr_key, 0.0) for d in detections],
            dtype=np.float64,
        )
        # Avoid div by zero: where normalize SNR is 0, set to 1 (no normalization)
        normalize_snr = np.where(normalize_snr > 0, normalize_snr, 1.0)
        logger.info(
            f"  Normalize SNR (ch{normalize_ch}): "
            f"median={np.median(normalize_snr):.2f}, "
            f"p95={np.percentile(normalize_snr, 95):.2f}, "
            f"max={np.max(normalize_snr):.2f}"
        )

    # Process each marker
    summaries = []
    for ch, name in zip(channels, names):
        marker_snr = per_marker_snr.get(name, args.snr_threshold)
        marker_method = per_marker_method.get(name, args.method)
        marker_feature = per_marker_feature.get(name, args.intensity_feature)

        # bg_subtract: on for otsu only. SNR uses pipeline features directly.
        if marker_method == "snr":
            marker_bg_subtract = False
            marker_global_bg = False
        elif args.no_background_subtract or args.use_raw:
            marker_bg_subtract = False
            marker_global_bg = args.global_background
        elif args.background_subtract is not None:
            marker_bg_subtract = args.background_subtract
            marker_global_bg = args.global_background
        else:
            marker_bg_subtract = marker_method == "otsu"
            marker_global_bg = args.global_background

        # Skip normalization if this marker IS the normalize channel (self-normalization = degenerate)
        marker_normalize = (
            None if (normalize_ch is not None and ch == normalize_ch) else normalize_snr
        )
        if normalize_ch is not None and ch == normalize_ch:
            logger.warning(
                f"  Skipping normalization for '{name}' (ch{ch}) — same as normalize channel"
            )

        row = classify_single_marker(
            detections,
            ch,
            name,
            marker_method,
            output_dir,
            bg_subtract=marker_bg_subtract,
            global_background=marker_global_bg,
            centroids=centroids,
            snr_threshold=marker_snr,
            intensity_feature=marker_feature,
            use_raw=args.use_raw,
            cv_max=args.cv_max,
            normalize_snr=marker_normalize,
        )
        summaries.append(row)

    # Create combined marker_profile when multiple markers are classified
    if len(names) > 1:
        logger.info("Creating combined marker_profile field...")
        class_keys = [f"{name}_class" for name in names]
        profile_counts = {}
        for det in detections:
            feat = det.get("features", {})
            parts = []
            for name, key in zip(names, class_keys):
                cls = feat.get(key, "negative")
                symbol = "+" if cls == "positive" else "-"
                parts.append(f"{name}{symbol}")
            profile = "/".join(parts)
            feat["marker_profile"] = profile
            profile_counts[profile] = profile_counts.get(profile, 0) + 1

        logger.info("  Marker profiles:")
        for profile, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(detections)
            logger.info(f"    {profile}: {count:,} ({pct:.1f}%)")

    # Save enriched detections with provenance in filename.
    method_tag = args.method
    feat_tag = args.intensity_feature
    out_json = output_dir / f"cell_detections_classified_{method_tag}_{feat_tag}.json"
    logger.info(f"Saving classified detections to {out_json}...")
    atomic_json_dump(detections, out_json)
    logger.info(f"  Wrote {len(detections):,} detections")

    # Save classification metadata (method, feature, thresholds — not per-cell)
    meta = {
        "method": method_tag,
        "intensity_feature": feat_tag,
        "use_raw": args.use_raw,
        "markers": {},
    }
    for row in summaries:
        meta["markers"][row["marker"]] = {
            "channel": row.get("channel"),
            "method": row.get("method", method_tag),
            "threshold": row.get("threshold"),
            "n_positive": row.get("n_positive"),
            "n_negative": row.get("n_negative"),
            "pct_positive": row.get("pct_positive"),
        }
    meta_json = output_dir / f"classification_metadata_{method_tag}_{feat_tag}.json"
    atomic_json_dump(meta, meta_json)
    logger.info(f"  Wrote classification metadata to {meta_json.name}")

    # Save summary CSV
    summary_path = output_dir / "marker_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "marker",
                "method",
                "feature",
                "threshold",
                "background_median",
                "n_positive",
                "n_negative",
                "pct_positive",
                "cv_filtered",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)
    logger.info(f"  Wrote summary: {summary_path}")

    # Final summary
    logger.info("--- Classification complete ---")
    for row in summaries:
        bg_info = f", bg={row['background_median']:.1f}" if row.get("background_median") else ""
        logger.info(
            f"  {row['marker']}: {row['n_positive']:,} positive "
            f"({row['pct_positive']:.1f}%), "
            f"threshold={row['threshold']:.2f} ({row['method']}{bg_info})"
        )


if __name__ == "__main__":
    main()
