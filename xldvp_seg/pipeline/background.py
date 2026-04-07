"""Local background correction for per-channel intensity features.

Estimates per-cell local background using a KD-tree of neighboring cells,
then subtracts that background from all per-channel intensity features.
Corrected features overwrite the originals; raw values are preserved as
``ch{N}_{suffix}_raw``.

This module is the single source of truth for background correction logic.
Both the main pipeline (``run_segmentation.py``) and the standalone
``scripts/classify_markers.py`` import from here.
"""

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core background subtraction
# ---------------------------------------------------------------------------


def local_background_subtract(
    values: np.ndarray,
    centroids: np.ndarray,
    n_neighbors: int = 30,
    tree_and_indices: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Per-cell local background subtraction using neighboring cells.

    For each cell, finds the *n_neighbors* nearest cells and uses their
    median intensity as the local background estimate.  Subtracts this
    from the cell's own intensity.

    Args:
        values: 1-D array of marker intensities per cell.
        centroids: (N, 2) array of ``[x, y]`` coordinates per cell.
        n_neighbors: Number of nearest neighbors for background estimate.
        tree_and_indices: Optional ``(tree, indices)`` tuple from a previous
            call.  When provided the KD-tree build and query are skipped,
            which avoids redundant O(N log N) work when correcting multiple
            channels on the same set of detections.

    Returns:
        ``(corrected_values, per_cell_background, (tree, indices))``.
        Corrected values are clipped >= 0.  The third element can be
        passed as *tree_and_indices* to subsequent calls.
    """
    from scipy.spatial import cKDTree

    n = len(values)
    if n < n_neighbors + 1:
        bg = np.median(values)
        logger.warning(
            "Too few cells (%d) for local background (need %d+1). " "Using global median %.1f",
            n,
            n_neighbors,
            bg,
        )
        corrected = np.maximum(values - bg, 0.0)
        return corrected, np.full(n, bg), tree_and_indices

    if tree_and_indices is not None:
        tree, indices = tree_and_indices
    else:
        tree = cKDTree(centroids)
        # k+1 because the closest neighbor is the cell itself (distance 0)
        _, indices = tree.query(centroids, k=n_neighbors + 1)

    neighbor_values = values[indices[:, 1:]]  # (n, n_neighbors)
    per_cell_bg = np.median(neighbor_values, axis=1)

    corrected = np.maximum(values - per_cell_bg, 0.0)
    return corrected, per_cell_bg, (tree, indices)


# ---------------------------------------------------------------------------
# Full channel correction
# ---------------------------------------------------------------------------

# Intensity suffixes whose values should be shifted by background.
_INTENSITY_SUFFIXES = [
    "mean",
    "median",
    "min",
    "max",
    "p5",
    "p25",
    "p75",
    "p95",
]


def _extract_centroids(detections: list[dict]) -> np.ndarray:
    """Return (N, 2) float64 array of **global** ``[x, y]`` centroids.

    Uses ``global_center`` (slide-level coordinates), NOT
    ``features["centroid"]`` which is local to the tile.  The KD-tree
    neighbourhood must use global coordinates to find spatially adjacent
    cells.
    """
    centroids = []
    for i, d in enumerate(detections):
        c = d.get("global_center")
        if c is None:
            logger.warning(
                "Detection %d has no global_center — falling back to [0, 0], "
                "which may corrupt KD-tree neighbor estimation.  "
                "Do NOT fall back to 'center' (tile-local).",
                i,
            )
            c = [0, 0]
        centroids.append(c)
    return np.array(centroids, dtype=np.float64)


def correct_all_channels(
    detections: list[dict],
    n_neighbors: int = 30,
    centroids: np.ndarray | None = None,
) -> list[int]:
    """Background-correct **all** per-channel intensity features in-place.

    For every channel discovered via ``ch{N}_mean`` keys in the first
    detection's features dict the following happens:

    * **Corrected** (bg subtracted, clipped >= 0):
      ``ch{N}_mean``, ``_median``, ``_min``, ``_max``,
      ``_p5``, ``_p25``, ``_p75``, ``_p95``
    * **Recomputed**:
      ``ch{N}_cv``  (std / corrected_mean);
      ``ch{N}_ch{M}_ratio``, ``ch{N}_ch{M}_diff``  (from corrected means)
    * **Unchanged** (shape/spread invariant to shift):
      ``ch{N}_std``, ``_variance``, ``_iqr``, ``_dynamic_range``,
      ``_skewness``, ``_kurtosis``
    * **Added**:
      ``ch{N}_{stat}_raw`` (raw copies), ``ch{N}_background`` (local bg),
      ``ch{N}_snr`` (median_raw / background)

    Args:
        detections: List of detection dicts (mutated in-place).
        n_neighbors: KD-tree neighbor count.
        centroids: Optional pre-computed (N, 2) centroid array.

    Returns:
        List of channel indices that were corrected.
    """
    if not detections:
        return []

    # Guard: refuse to double-correct — scan first 10 detections
    sample_keys: set[str] = set()
    for d in detections[:10]:
        sample_keys.update(d.get("features", {}).keys())
    if any(k.endswith("_background") for k in sample_keys):
        logger.info(
            "Detections already background-corrected (found *_background keys) "
            "— skipping correct_all_channels to prevent double correction"
        )
        return []

    if centroids is None:
        centroids = _extract_centroids(detections)

    # Discover channels — scan first 10 detections and take the union of keys
    # in case the first detection has incomplete features
    channels_set: set[int] = set()
    for d in detections[:10]:
        for key in d.get("features", {}):
            if key.startswith("ch") and key.endswith("_mean"):
                ch_str = key[2:].replace("_mean", "")
                try:
                    channels_set.add(int(ch_str))
                except ValueError:
                    continue
    channels: list[int] = sorted(channels_set)

    if not channels:
        logger.warning("No ch{N}_mean keys found in detections")
        return []

    logger.info("Background-correcting %d channels: %s", len(channels), channels)
    n_det = len(detections)

    # --- Step 1: per-cell background from ch{N}_median (or median_raw on re-run) ---
    channel_bg: dict[int, np.ndarray] = {}
    channel_corrected_median: dict[int, np.ndarray] = {}

    # Build the KD-tree once and reuse across all channels
    _cached_tree_and_indices = None

    for ch in channels:
        # Read raw (uncorrected) median pixel intensity for background estimation.
        # After correct_all_channels runs: ch{N}_median = corrected, ch{N}_median_raw = original.
        # Before correct_all_channels (first pipeline run): ch{N}_median = original, no _raw yet.
        has_raw = (
            any(f"ch{ch}_median_raw" in d.get("features", {}) for d in detections[:10])
            if detections
            else False
        )
        value_key = f"ch{ch}_median_raw" if has_raw else f"ch{ch}_median"
        values = np.array(
            [d.get("features", {}).get(value_key, 0.0) for d in detections],
            dtype=np.float64,
        )
        corrected_median, per_cell_bg, _cached_tree_and_indices = local_background_subtract(
            values,
            centroids,
            n_neighbors,
            tree_and_indices=_cached_tree_and_indices,
        )
        channel_bg[ch] = per_cell_bg
        channel_corrected_median[ch] = corrected_median

        bg_median_val = float(np.median(per_cell_bg))
        nonzero = corrected_median[corrected_median > 0]
        logger.info(
            "  ch%d: bg median=%.1f, %d/%d cells with signal after correction",
            ch,
            bg_median_val,
            len(nonzero),
            n_det,
        )

    # --- Step 2: correct intensity features & write back ---
    for ch in channels:
        per_cell_bg = channel_bg[ch]

        for suffix in _INTENSITY_SUFFIXES:
            key = f"ch{ch}_{suffix}"
            if key not in sample_keys:
                continue

            values = np.array(
                [d.get("features", {}).get(key, 0.0) for d in detections],
                dtype=np.float64,
            )
            corrected = np.maximum(values - per_cell_bg, 0.0)

            raw_key = f"ch{ch}_{suffix}_raw"
            for i, det in enumerate(detections):
                feat = det.setdefault("features", {})
                feat[raw_key] = float(values[i])
                feat[key] = float(corrected[i])

        # background & SNR (median / median — robust to bright outlier pixels)
        median_raw = np.array(
            [d.get("features", {}).get(f"ch{ch}_median_raw", 0.0) for d in detections],
            dtype=np.float64,
        )
        for i, det in enumerate(detections):
            feat = det.setdefault("features", {})
            feat[f"ch{ch}_background"] = float(per_cell_bg[i])
            feat[f"ch{ch}_snr"] = (
                float(median_raw[i] / per_cell_bg[i]) if per_cell_bg[i] > 0 else 0.0
            )

        # cv = std / corrected_mean (CV is defined as std/mean, not std/median)
        if f"ch{ch}_cv" in sample_keys and f"ch{ch}_std" in sample_keys:
            for i, det in enumerate(detections):
                feat = det.setdefault("features", {})
                corr_mean = feat.get(f"ch{ch}_mean", 0.0)
                std = feat.get(f"ch{ch}_std", 0.0)
                feat[f"ch{ch}_cv"] = float(std / corr_mean) if corr_mean > 0 else 0.0

    # --- Step 2b: recompute dynamic_range from corrected min/max ---
    for ch in channels:
        dr_key = f"ch{ch}_dynamic_range"
        if dr_key in sample_keys:
            for det in detections:
                feat = det.setdefault("features", {})
                cmin = feat.get(f"ch{ch}_min", 0.0)
                cmax = feat.get(f"ch{ch}_max", 0.0)
                feat[dr_key] = float(cmax - cmin)

    # --- Step 3: recompute cross-channel ratios/diffs ---
    for ch_a in channels:
        for ch_b in channels:
            if ch_a == ch_b:
                continue
            ratio_key = f"ch{ch_a}_ch{ch_b}_ratio"
            diff_key = f"ch{ch_a}_ch{ch_b}_diff"
            if ratio_key in sample_keys:
                for i, det in enumerate(detections):
                    feat = det.setdefault("features", {})
                    a = channel_corrected_median[ch_a][i]
                    b = channel_corrected_median[ch_b][i]
                    feat[ratio_key] = float(a / b) if b > 0 else 0.0
            if diff_key in sample_keys:
                for i, det in enumerate(detections):
                    feat = det.setdefault("features", {})
                    a = channel_corrected_median[ch_a][i]
                    b = channel_corrected_median[ch_b][i]
                    feat[diff_key] = float(a - b)

    return channels
