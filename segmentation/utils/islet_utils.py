"""
Islet marker classification utilities.

Functions for classifying islet cells by dominant hormone marker (alpha/beta/delta)
using GMM-based or percentile-based thresholding on per-channel intensities.

Usage:
    from segmentation.utils.islet_utils import classify_islet_marker, compute_islet_marker_thresholds

    # Compute thresholds from all detections
    thresholds = compute_islet_marker_thresholds(all_detections, marker_map={'gcg': 2, 'ins': 3, 'sst': 5})

    # Classify a single cell
    class_name, color = classify_islet_marker(features_dict, thresholds, marker_map={'gcg': 2, 'ins': 3, 'sst': 5})
"""

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def classify_islet_marker(features_dict, marker_thresholds=None, marker_map=None):
    """Classify an islet cell by dominant hormone marker.

    Uses NORMALIZED channel values (same as HTML display) so contour color
    matches what the user sees. Marker channels are defined by marker_map.

    Args:
        features_dict: dict with 'ch{N}_mean' feature keys
        marker_thresholds: (norm_ranges, ch_thresholds, ratio_min) from compute_islet_marker_thresholds()
        marker_map: dict mapping marker name -> CZI channel index, e.g. {'gcg': 2, 'ins': 3, 'sst': 5}

    Returns (class_name, contour_color_rgb).
    """
    if marker_map is None:
        marker_map = {'gcg': 2, 'ins': 3, 'sst': 5}

    # Build ordered marker list with colors (R, G, B for first 3 markers)
    _marker_colors = [
        (255, 50, 50), (50, 255, 50), (50, 50, 255),
        (255, 255, 50), (255, 50, 255), (50, 255, 255),
        (255, 128, 0), (128, 0, 255),
    ]
    marker_names = list(marker_map.keys())
    marker_vals = {}
    for name in marker_names:
        ch_idx = marker_map[name]
        marker_vals[name] = features_dict.get(f'ch{ch_idx}_mean', 0)

    if marker_thresholds is None:
        return 'none', (128, 128, 128)

    norm_ranges, ch_thresholds, ratio_min = marker_thresholds

    # Normalize to 0-1 using same percentiles as HTML display
    def _norm(val, ch_key):
        lo, hi = norm_ranges.get(ch_key, (0, 1))
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    normed = {}
    positive = {}
    for name in marker_names:
        ch_key = f'ch{marker_map[name]}'
        normed[name] = _norm(marker_vals[name], ch_key)
        positive[name] = normed[name] >= ch_thresholds.get(ch_key, 0.5)

    # Gate: must exceed at least one channel's threshold
    if not any(positive.values()):
        return 'none', (128, 128, 128)

    # Ratio classification -- only among channels that are actually positive
    pos_markers = []
    for i, name in enumerate(marker_names):
        if positive[name]:
            color = _marker_colors[i] if i < len(_marker_colors) else (200, 200, 200)
            pos_markers.append((name, normed[name], color))

    if len(pos_markers) == 1:
        return pos_markers[0][0], pos_markers[0][2]

    # Multiple positive channels -- check ratio among them
    pos_markers.sort(key=lambda x: x[1], reverse=True)
    best_name, best_val, best_color = pos_markers[0]
    second_val = pos_markers[1][1] if len(pos_markers) > 1 else 0

    if second_val > 0 and best_val / second_val < ratio_min:
        return 'multi', (255, 170, 0)  # orange

    return best_name, best_color


def compute_islet_marker_thresholds(all_detections, vis_threshold_overrides=None, ratio_min=1.5,
                                    marker_map=None, marker_top_pct=5,
                                    pct_channels=None, gmm_p_cutoff=0.75,
                                    threshold_factor=1.0):
    """Compute per-channel thresholds for islet marker classification.

    Two methods, selectable per channel:
    - **GMM** (default): 2-component Gaussian Mixture on log-transformed intensities.
      Finds the natural boundary between background and signal populations.
    - **Percentile** (for specified channels): top N% of the intensity distribution
      = positive. Simple, interpretable, biologically grounded for channels where
      GMM separation is too poor to be reliable.

    Args:
        all_detections: list of detection dicts with features
        vis_threshold_overrides: optional dict {ch_key: float} to manually override
            per-channel thresholds (e.g. {'ch5': 0.6})
        ratio_min: dominant marker must be >= ratio_min * second-highest
            to be classified as single-marker. Otherwise -> "multi".
        marker_map: dict mapping marker name -> CZI channel index,
            e.g. {'gcg': 2, 'ins': 3, 'sst': 5}
        marker_top_pct: for pct_channels, classify the top N% of cells
            as marker-positive (default 5 = 95th percentile)
        pct_channels: set of marker names that should use percentile-based
            thresholding instead of GMM (default {'sst'})
        gmm_p_cutoff: GMM posterior probability cutoff for signal classification.
            Higher = stricter (fewer false positives). (default 0.75)
        threshold_factor: Divisor for the raw threshold. Values > 1 make
            classification more permissive (e.g. 2.0 halves the threshold).
            Matches the pipeline pre-filter's marker_signal_factor. (default 1.0)

    Returns (norm_ranges, ch_thresholds, ratio_min) for classify_islet_marker(),
        or None if too few detections for reliable thresholds.
    """
    if marker_map is None:
        marker_map = {'gcg': 2, 'ins': 3, 'sst': 5}
    if pct_channels is None:
        pct_channels = {'sst'}

    if len(all_detections) < 10:
        logger.warning(f"Only {len(all_detections)} detections — too few for reliable "
                       "marker thresholds. Skipping marker classification.")
        return None

    from sklearn.mixture import GaussianMixture

    # Build arrays from features using marker_map channel indices
    marker_arrays = {}
    for name, ch_idx in marker_map.items():
        marker_arrays[name] = np.array([
            d.get('features', {}).get(f'ch{ch_idx}_mean', 0) for d in all_detections
        ])

    norm_ranges = {}
    ch_thresholds = {}

    for name, ch_idx in marker_map.items():
        ch_key = f'ch{ch_idx}'
        ch_name = name.capitalize()
        arr = marker_arrays[name]
        # Exclude zero-valued entries (cells with no signal or missing features)
        arr_pos = arr[arr > 0]
        if len(arr_pos) < 10:
            logger.warning(f"Only {len(arr_pos)} cells with nonzero {ch_name} — using full array for percentiles")
            arr_pos = arr
        lo = float(np.percentile(arr_pos, 1))
        hi = float(np.percentile(arr_pos, 99.5))
        norm_ranges[ch_key] = (lo, hi)

        if name in pct_channels:
            # --- Percentile-based threshold (for channels with poor GMM separation) ---
            # threshold_factor widens the top-N%: factor=2 -> top 5% becomes top 10%
            effective_pct = min(marker_top_pct * threshold_factor, 50)
            method = f'top-{effective_pct:.0f}%'
            pct_cutoff = 100 - effective_pct
            raw_cutoff = float(np.percentile(arr_pos, pct_cutoff))
            bg_mean = sig_mean = separation = 0  # not computed for percentile channels
            logger.info(f"  {ch_name}: using top {effective_pct:.0f}% "
                        f"(p{pct_cutoff:.0f}={raw_cutoff:.0f})")
        else:
            # --- GMM on log-transformed intensities ---
            # Log1p compresses dynamic range, making background vs signal more Gaussian.
            log_arr = np.log1p(arr_pos).reshape(-1, 1)

            gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
            gmm.fit(log_arr)

            signal_idx = int(np.argmax(gmm.means_.flatten()))
            bg_idx = 1 - signal_idx
            bg_mean = float(np.exp(gmm.means_.flatten()[bg_idx]))
            sig_mean = float(np.exp(gmm.means_.flatten()[signal_idx]))

            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            separation = abs(means[signal_idx] - means[bg_idx]) / max(stds[bg_idx], stds[signal_idx])

            if separation < 1.0:
                # GMM components overlap too much -- binary search for posterior
                # crossover can converge to wrong point. Fall back to percentile.
                logger.warning(f"  {ch_name}: GMM separation too low ({separation:.2f} sigma), "
                               "falling back to percentile method")
                method = f'GMM-fallback(top-{marker_top_pct}%)'
                effective_pct = min(marker_top_pct * threshold_factor, 50)
                pct_cutoff = 100 - effective_pct
                raw_cutoff = float(np.percentile(arr_pos, pct_cutoff))
            else:
                method = f'GMM(P>={gmm_p_cutoff})'
                # Find raw threshold where P(signal) = gmm_p_cutoff via binary search.
                # Higher than P=0.5 crossover -- requires stronger evidence of signal,
                # reducing false positives and multi-marker misclassification.
                lo_raw, hi_raw = 0.0, float(arr.max())
                # Check if the cutoff is achievable: if P(signal) at lo_raw is already
                # above cutoff, the threshold would converge to 0 (everything is signal).
                # If P(signal) at hi_raw is below cutoff, no threshold satisfies the
                # criterion and binary search converges to max (nothing is signal).
                p_at_lo = gmm.predict_proba(np.log1p(np.array([[lo_raw]])))[0, signal_idx]
                p_at_hi = gmm.predict_proba(np.log1p(np.array([[hi_raw]])))[0, signal_idx]
                if p_at_hi < gmm_p_cutoff and p_at_lo < gmm_p_cutoff:
                    # P(signal) never reaches cutoff -- fall back to percentile
                    logger.warning(f"  {ch_name}: GMM posterior never reaches {gmm_p_cutoff} "
                                   f"(max P(signal)={max(p_at_lo, p_at_hi):.3f}), "
                                   "falling back to percentile method")
                    method = f'GMM-fallback(top-{marker_top_pct}%)'
                    effective_pct = min(marker_top_pct * threshold_factor, 50)
                    pct_cutoff = 100 - effective_pct
                    raw_cutoff = float(np.percentile(arr_pos, pct_cutoff))
                else:
                    for _ in range(50):
                        mid = (lo_raw + hi_raw) / 2
                        p = gmm.predict_proba(np.log1p(np.array([[mid]])))
                        if p[0, signal_idx] > gmm_p_cutoff:
                            hi_raw = mid
                        else:
                            lo_raw = mid
                    raw_cutoff = (lo_raw + hi_raw) / 2

        # Apply threshold_factor to GMM channels only (>1 = more permissive).
        # Percentile channels already incorporate threshold_factor via widened percentage.
        if threshold_factor > 1.0 and name not in pct_channels:
            raw_cutoff = raw_cutoff / threshold_factor

        # Convert raw threshold to normalized [0,1] for classify_islet_marker()
        if hi > lo:
            auto_t = float(np.clip((raw_cutoff - lo) / (hi - lo), 0.0, 1.0))
        else:
            auto_t = 0.5

        # Count positive cells
        n_pos = int(np.sum(arr > raw_cutoff))
        pos_pct = 100 * n_pos / len(arr) if len(arr) > 0 else 0

        # Allow manual override
        if vis_threshold_overrides and ch_key in vis_threshold_overrides:
            ch_thresholds[ch_key] = vis_threshold_overrides[ch_key]
            override_raw = lo + vis_threshold_overrides[ch_key] * (hi - lo)
            n_override = int(np.sum(arr > override_raw))
            logger.info(f"Islet {ch_name}({ch_key}): {method} bg={bg_mean:.0f}, sig={sig_mean:.0f}, "
                        f"sep={separation:.2f}σ, auto={auto_t:.3f} (raw={raw_cutoff:.0f}, {pos_pct:.1f}%), "
                        f"OVERRIDE={vis_threshold_overrides[ch_key]:.3f} "
                        f"({100*n_override/len(arr):.1f}%)")
        else:
            ch_thresholds[ch_key] = auto_t
            logger.info(f"Islet {ch_name}({ch_key}): p1={lo:.1f}, p99.5={hi:.1f}, "
                        f"{method} bg={bg_mean:.0f}, sig={sig_mean:.0f}, sep={separation:.2f}σ, "
                        f"threshold={auto_t:.3f} (raw>{raw_cutoff:.0f})")
            logger.info(f"  {ch_name}-positive: {n_pos} cells ({pos_pct:.1f}%)")

    logger.info(f"Islet marker ratio_min: {ratio_min}x (dominant must be >= {ratio_min}x runner-up)")
    return (norm_ranges, ch_thresholds, ratio_min)
