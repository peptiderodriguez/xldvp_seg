"""Marker classification: classify cells as marker-positive or marker-negative.

Provides threshold-based classification methods (SNR, Otsu, GMM) operating on
per-channel intensity features already present in detection dicts.

Functions:
    classify_otsu          Full Otsu threshold on bg-subtracted values
    classify_otsu_half     Permissive Otsu/2 threshold
    classify_gmm           2-component GMM on log1p intensities
    extract_marker_values  Extract per-cell intensity from features dict
    classify_single_marker Main worker: classify one marker, mutate detections
    plot_distribution      Histogram with threshold line
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xldvp_seg.exceptions import ConfigError, DataLoadError
from xldvp_seg.pipeline.background import (
    _extract_centroids,
    local_background_subtract,
)
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Classification methods
# ---------------------------------------------------------------------------


def classify_otsu(values: np.ndarray, include_zeros: bool = False) -> tuple[float, np.ndarray]:
    """Full Otsu threshold on (background-subtracted) intensity values.

    Use after background subtraction — the noise floor is already removed,
    so the full Otsu threshold cleanly separates signal from residual noise.

    Args:
        values: Per-cell intensity values.
        include_zeros: If True, include zero-valued pixels in the Otsu computation
            and allow zero-valued cells to be classified as positive (when above
            threshold). Default False because zeros are often CZI buffer/padding
            areas that would bias the threshold downward. Set True only when zeros
            represent genuine background-corrected signal (e.g., after local
            background subtraction where zeros are real).

    Returns (threshold, boolean mask where True = positive).
    """
    from skimage.filters import threshold_otsu

    vals_for_otsu = values if include_zeros else values[values > 0]
    n_zeros = len(values) - len(vals_for_otsu)
    if not include_zeros and n_zeros > len(values) * 0.5:
        logger.warning(
            "%.1f%% of values are zero (excluded from Otsu). Consider include_zeros=True "
            "if data is background-corrected.",
            100 * n_zeros / max(len(values), 1),
        )
    if len(vals_for_otsu) < 10:
        logger.warning(
            "Too few %svalues for Otsu; defaulting threshold to 0",
            "" if include_zeros else "non-zero ",
        )
        return 0.0, np.zeros(len(values), dtype=bool)

    if np.std(vals_for_otsu) < 1e-6:
        logger.warning("Near-zero variance in marker values; all cells classified as negative")
        return 0.0, np.zeros(len(values), dtype=bool)

    threshold = float(threshold_otsu(vals_for_otsu))
    if include_zeros:
        positive = values >= threshold
    else:
        positive = (values >= threshold) & (values > 0)
    return threshold, positive


def classify_otsu_half(values: np.ndarray, include_zeros: bool = False) -> tuple[float, np.ndarray]:
    """Otsu threshold / 2 on raw intensity values.

    Args:
        include_zeros: If True, include zero-valued cells in the Otsu
            computation.  Default False excludes zeros because they often
            represent CZI buffer/padding areas rather than real signal.

    Returns (threshold, boolean mask where True = positive).
    """
    from skimage.filters import threshold_otsu

    nonzero = values if include_zeros else values[values > 0]
    n_zeros = len(values) - len(nonzero)
    if not include_zeros and n_zeros > len(values) * 0.5:
        logger.warning(
            "%.1f%% of values are zero (excluded from Otsu). Consider include_zeros=True "
            "if data is background-corrected.",
            100 * n_zeros / max(len(values), 1),
        )
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
    if include_zeros:
        positive = values >= threshold
    else:
        positive = (values >= threshold) & (values > 0)
    return float(threshold), positive


def classify_gmm(values: np.ndarray, posterior_threshold: float = 0.75) -> tuple[float, np.ndarray]:
    """2-component GMM on log1p intensities with BIC model selection.

    First compares 1-component vs 2-component GMM using BIC. If the
    1-component model is preferred (unimodal distribution), all cells are
    classified as negative. Otherwise, the component with the higher mean
    is treated as the 'high' (signal) class. Cells with posterior
    probability >= *posterior_threshold* for that component are positive.

    Args:
        values: Per-cell intensity values.
        posterior_threshold: Minimum posterior probability for the high-signal
            component to classify a cell as positive (default 0.75).

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
        logger.warning("Near-zero variance in log1p values; all cells classified as negative")
        return 0.0, np.zeros(len(values), dtype=bool)

    # BIC model selection: prefer 1-component if data is unimodal
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(log_vals)
    gmm2 = GaussianMixture(n_components=2, random_state=42, max_iter=200).fit(log_vals)
    if gmm1.bic(log_vals) - gmm2.bic(log_vals) < 6:
        logger.info(
            "BIC does not strongly favor 2 components (delta=%.1f < 6). Returning all-negative.",
            gmm1.bic(log_vals) - gmm2.bic(log_vals),
        )
        return 0.0, np.zeros(len(values), dtype=bool)
    gmm = gmm2

    # Identify which component has the higher mean
    high_idx = int(np.argmax(gmm.means_.flatten()))

    # Check if components are well-separated (warn if unimodal)
    means = gmm.means_.flatten()
    variances = gmm.covariances_.reshape(2, -1)[:, 0]
    stds = np.sqrt(variances)
    separation = abs(means[1] - means[0]) / max(stds[0] + stds[1], 1e-6)
    if separation < 0.5:
        logger.warning(
            "GMM components poorly separated (separation=%.2f). "
            "Data may be unimodal. Consider otsu_half method instead.",
            separation,
        )
        if min(gmm.weights_) < 0.1:
            logger.info(
                "Poorly separated GMM with minor component weight %.2f < 0.1. "
                "Returning all-negative.",
                min(gmm.weights_),
            )
            return 0.0, np.zeros(len(values), dtype=bool)

    # Posterior probability for the high component
    probs = gmm.predict_proba(log_vals)[:, high_idx]
    positive = probs >= posterior_threshold

    # Approximate threshold: crossing point in log space, convert back.
    # This weighted-midpoint formula is exact for equal priors (weights)
    # and approximate for unequal priors.
    log_threshold = (means[0] * stds[1] + means[1] * stds[0]) / (stds[0] + stds[1])
    threshold = float(np.expm1(log_threshold))

    return threshold, positive


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distribution(
    values: np.ndarray,
    threshold: float,
    marker_name: str,
    method: str,
    n_positive: int,
    n_negative: int,
    output_path: Path,
) -> None:
    """Histogram of marker values with threshold line."""
    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use log1p scale for x-axis if range is large
    use_log = values.max() > 10 * max(np.median(values), 1)
    plot_vals = np.log1p(values) if use_log else values
    plot_thresh = np.log1p(threshold) if use_log else threshold

    n_bins = min(100, max(30, len(values) // 20))
    ax.hist(plot_vals, bins=n_bins, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(
        plot_thresh, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.1f}"
    )

    pct = 100 * n_positive / max(n_positive + n_negative, 1)
    ax.set_title(
        f"{marker_name} classification ({method})\n"
        f"{n_positive:,} positive ({pct:.1f}%) / {n_negative:,} negative"
    )
    ax.set_xlabel("log1p(intensity)" if use_log else "Intensity")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved distribution plot: %s", output_path)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def extract_marker_values(
    detections: list[dict], channel: int, feature: str = "mean", use_raw: bool = False
) -> np.ndarray:
    """Extract per-cell intensity values from detection features.

    Args:
        channel: CZI channel index.
        feature: Intensity statistic — 'mean', 'median', 'p75', 'p95', etc.
        use_raw: If True, read ``ch{N}_{feature}_raw`` (pre-bg-correction).
    """
    # SNR is already a ratio — use_raw doesn't apply
    if feature == "snr":
        key = f"ch{channel}_snr"
    else:
        suffix = f"{feature}_raw" if use_raw else feature
        key = f"ch{channel}_{suffix}"
    values = np.array([d.get("features", {}).get(key, 0.0) for d in detections], dtype=np.float64)

    # Warn if all zeros (likely missing feature key)
    if len(values) > 0 and values.max() == 0:
        fallback_key = f"ch{channel}_{feature}"
        if use_raw:
            logger.warning(
                "Key '%s' not found or all zeros — falling back to '%s'", key, fallback_key
            )
            values = np.array(
                [d.get("features", {}).get(fallback_key, 0.0) for d in detections], dtype=np.float64
            )
    return values


def classify_single_marker(
    detections: list[dict],
    channel: int,
    marker_name: str,
    method: str,
    output_dir: Path,
    bg_subtract: bool = False,
    global_background: bool = False,
    centroids: np.ndarray | None = None,
    n_neighbors: int = 30,
    snr_threshold: float = 1.5,
    intensity_feature: str = "mean",
    use_raw: bool = False,
    cv_max: float | None = None,
    normalize_snr: np.ndarray | None = None,
) -> dict:
    """Classify detections for one marker. Returns summary row dict.

    Mutates detections in-place: adds {marker}_class, {marker}_value,
    {marker}_raw, {marker}_background, and {marker}_snr
    to each detection's features dict. The threshold is returned in the
    summary dict, not stored per-detection.

    Args:
        intensity_feature: Which stat to threshold on ('mean', 'median', 'p75', 'p95').
        use_raw: Use pre-bg-correction values (``ch{N}_{feat}_raw``).
        global_background: Subtract slide-wide median (not local neighbors).
            Avoids inflating background in expression zones.
        snr_threshold: SNR cutoff for positive classification (default 1.5).
            An SNR of 1.5 means the signal is at least 50% above the local
            background — a standard threshold for positive fluorescence signal
            in tissue imaging. Lower values (e.g., 1.2) increase sensitivity
            at the cost of specificity. Validate per experiment.
        cv_max: If set, cells with ``ch{N}_cv > cv_max`` are classified negative
            regardless of intensity (filters out particulate noise).
    """
    feat_label = f"{intensity_feature}{'_raw' if use_raw else ''}"
    bg_label = (
        " + global bg subtraction"
        if global_background
        else f" + local bg subtraction (k={n_neighbors})" if bg_subtract else ""
    )
    logger.info(
        "Classifying marker '%s' (ch%d) with method '%s' on %s%s%s",
        marker_name,
        channel,
        method,
        feat_label,
        bg_label,
        f" + CV filter (max={cv_max})" if cv_max else "",
    )

    raw_values = extract_marker_values(
        detections, channel, feature=intensity_feature, use_raw=use_raw
    )
    if len(raw_values) == 0:
        logger.warning("No values extracted for marker '%s' — skipping", marker_name)
        return {
            "marker": marker_name,
            "method": method,
            "threshold": 0,
            "n_positive": 0,
            "n_negative": 0,
            "pct_positive": 0,
            "background_median": 0,
            "cv_filtered": 0,
        }

    logger.info(
        "  Extracted %s values, range [%.1f, %.1f], median %.1f",
        f"{len(raw_values):,}",
        raw_values.min(),
        raw_values.max(),
        np.median(raw_values),
    )

    # Background subtraction (skip for SNR — already a ratio)
    per_cell_bg = None
    if intensity_feature == "snr":
        values = raw_values
        if normalize_snr is not None:
            values = values / normalize_snr
            logger.info(
                "  SNR normalized by reference channel: median=%.3f, p95=%.3f, max=%.3f",
                np.median(values),
                np.percentile(values, 95),
                values.max(),
            )
        else:
            logger.info(
                "  SNR mode — skipping background subtraction (already signal/background ratio)"
            )
    elif global_background:
        # Global: subtract slide-wide median from all cells.
        # Avoids local-neighbor inflation for regional markers.
        global_bg = float(np.median(raw_values))
        values = np.maximum(raw_values - global_bg, 0.0)
        per_cell_bg = np.full(len(raw_values), global_bg)
        nonzero = values[values > 0]
        logger.info("  Global background: %.1f (slide-wide median)", global_bg)
        if len(nonzero) > 0:
            logger.info(
                "  After subtraction: %s/%s cells with signal, range [%.1f, %.1f], median %.1f",
                f"{len(nonzero):,}",
                f"{len(values):,}",
                nonzero.min(),
                nonzero.max(),
                np.median(nonzero),
            )
        else:
            logger.info("  All values zero after subtraction")
    elif bg_subtract:
        if centroids is None:
            centroids = _extract_centroids(detections)
        values, per_cell_bg, _ = local_background_subtract(raw_values, centroids, n_neighbors)
        median_bg = float(np.median(per_cell_bg))
        logger.info(
            "  Local background: median %.1f, range [%.1f, %.1f]",
            median_bg,
            per_cell_bg.min(),
            per_cell_bg.max(),
        )
        nonzero_corrected = values[values > 0]
        if len(nonzero_corrected) > 0:
            logger.info(
                "  After subtraction: %s cells with signal, range [%.1f, %.1f], median %.1f",
                f"{(nonzero_corrected > 0).sum():,}",
                nonzero_corrected.min(),
                nonzero_corrected.max(),
                np.median(nonzero_corrected),
            )
        else:
            logger.warning("  All values zero after background subtraction")
    else:
        values = raw_values

    # CV filter: read coefficient of variation for this channel
    cv_mask = None
    n_cv_filtered = 0
    if cv_max is not None:
        cv_suffix = "cv_raw" if use_raw else "cv"
        cv_key = f"ch{channel}_{cv_suffix}"
        cv_values = np.array(
            [d.get("features", {}).get(cv_key, 0.0) for d in detections], dtype=np.float64
        )
        # Fallback: if _raw version missing, try plain cv
        if cv_values.max() == 0 and use_raw:
            cv_key = f"ch{channel}_cv"
            cv_values = np.array(
                [d.get("features", {}).get(cv_key, 0.0) for d in detections], dtype=np.float64
            )
        cv_mask = cv_values <= cv_max
        n_cv_filtered = int((~cv_mask).sum())
        logger.info(
            "  CV filter: %s cells with CV > %.1f (will be forced negative)",
            f"{n_cv_filtered:,}",
            cv_max,
        )

    # Classify
    _include_zeros = bg_subtract or global_background
    if method == "snr":
        # SNR-based: positive if channel SNR >= snr_threshold
        # First try pipeline-computed ch{N}_snr, then compute from raw/bg
        snr_key_ch = f"ch{channel}_snr"
        pipeline_snr = np.array(
            [d.get("features", {}).get(snr_key_ch, 0.0) for d in detections], dtype=np.float64
        )
        if np.any(pipeline_snr > 0):
            snr_values = pipeline_snr
            logger.info("  Using pipeline-computed %s for SNR classification", snr_key_ch)
        elif per_cell_bg is not None:
            snr_values = np.where(per_cell_bg > 0, raw_values / per_cell_bg, 0.0)
            logger.info("  Computing SNR from raw_values / per_cell_bg")
        else:
            raise DataLoadError(
                f"SNR method requires either pipeline bg correction (ch{channel}_snr in features) "
                "or --background-subtract. Neither found."
            )
        # Apply normalize-channel if provided
        if normalize_snr is not None:
            snr_values = snr_values / normalize_snr
            logger.info(
                "  SNR normalized by reference channel: median=%.3f, p95=%.3f, max=%.3f",
                np.median(snr_values),
                np.percentile(snr_values, 95),
                snr_values.max(),
            )
        threshold = snr_threshold
        positive_mask = snr_values >= snr_threshold
    elif method == "otsu":
        threshold, positive_mask = classify_otsu(values, include_zeros=_include_zeros)
    elif method == "otsu_half":
        threshold, positive_mask = classify_otsu_half(values, include_zeros=_include_zeros)
    elif method == "gmm":
        threshold, positive_mask = classify_gmm(values)
    else:
        raise ConfigError(f"Unknown method: {method}")

    # Apply CV filter (force high-CV cells to negative)
    if cv_mask is not None:
        positive_mask = positive_mask & cv_mask

    n_positive = int(positive_mask.sum())
    n_negative = len(positive_mask) - n_positive
    pct = 100 * n_positive / max(len(positive_mask), 1)

    logger.info(
        "  Threshold: %.2f%s",
        threshold,
        " (on bg-subtracted values)" if bg_subtract else "",
    )
    logger.info(
        "  Positive: %s (%.1f%%)  Negative: %s (%.1f%%)%s",
        f"{n_positive:,}",
        pct,
        f"{n_negative:,}",
        100 - pct,
        f"  (CV filtered: {n_cv_filtered:,})" if n_cv_filtered else "",
    )

    # Enrich detections in-place
    class_key = f"{marker_name}_class"
    value_key = f"{marker_name}_value"
    raw_key = f"{marker_name}_raw"
    bg_key = f"{marker_name}_background"
    snr_key = f"{marker_name}_snr"

    for i, det in enumerate(detections):
        feat = det.setdefault("features", {})
        feat[class_key] = "positive" if positive_mask[i] else "negative"
        feat[value_key] = float(values[i])  # bg-subtracted if enabled
        if per_cell_bg is not None:
            feat[raw_key] = float(raw_values[i])
            feat[bg_key] = float(per_cell_bg[i])
            feat[snr_key] = float(raw_values[i] / per_cell_bg[i]) if per_cell_bg[i] > 0 else 0.0

    # Plot (use the values that were actually thresholded)
    plot_path = output_dir / f"{marker_name}_distribution.png"
    plot_distribution(
        values,
        threshold,
        marker_name,
        f"{method} on {feat_label}"
        + ("+bgsub" if bg_subtract else "")
        + (f"+cv<={cv_max}" if cv_max else ""),
        n_positive,
        n_negative,
        plot_path,
    )

    return {
        "marker": marker_name,
        "method": method,
        "feature": feat_label,
        "threshold": threshold,
        "background_median": float(np.median(per_cell_bg)) if per_cell_bg is not None else 0,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pct_positive": round(pct, 2),
        "cv_filtered": n_cv_filtered,
    }
