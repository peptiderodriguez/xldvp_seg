"""Rare cell-population discovery via HDBSCAN + Ward-linkage taxonomy.

Identifies small-but-stable morphological cell populations (default ≥1000 cells)
across a whole-slide detection set. Pipeline:

1. Quality pre-filter (drop Cellpose-suspect cells).
2. Build feature matrix from requested feature groups (morph + SAM2 by default).
3. Log-transform scale-spanning features (area, perimeter, axes) via log1p.
4. RobustScaler (median/IQR) — resilient to extreme single-cell outliers that
   would distort StandardScaler.
5. PCA to ≤30 dims (capture ≥95% variance but capped — HDBSCAN density
   estimation degrades in very high dimensions).
6. HDBSCAN with ``min_cluster_size=1000`` by default (GPU via cuML if
   available, else CPU ``hdbscan``).
7. Multi-run stability check across ``min_cluster_size`` values via Jaccard
   overlap. Clusters surviving ≥2 runs are ``stable``.
8. Vectorized Moran's I on a Delaunay k=10 neighbor graph — spatial cohesion
   per cluster (one sparse matvec, not per-cluster loop).
9. Ward-linkage hierarchical clustering on cluster centroids → dendrogram
   showing inter-type morphological distances.

Output: augmented detections with ``rare_pop_id`` + ``hdbscan_prob`` fields,
per-cluster summary table (size, persistence, moran_I, stable), PCA linkage
matrix, and cached intermediates (``X_pca_<hash>.npz``, kNN adjacency
``W_knn_k<k>_<pos_hash>.npz``) for fast re-runs with different HDBSCAN
parameters. The PCA cache key includes feature groups, pre-filter thresholds,
``max_pcs``, ``pca_variance``, seed, excluded channels, and group weights —
any change invalidates.

Field name ``rare_pop_id`` is namespaced to avoid collisions with existing
``global_cluster`` / ``cluster_id`` fields from other pipelines. Three-value
sentinel:

* ``0, 1, 2, ...`` — density-based cluster member
* ``-1`` — HDBSCAN noise (passed pre-filter, not dense enough)
* ``-2`` — pre-filter drop (``rare_pop_filter_reason`` gives the reason:
  ``"n_nuclei" | "nc_ratio" | "overlap" | "area" | "missing_features"``)

Default feature weighting is ``"equal"`` (each group contributes 1 unit of
squared Euclidean distance via 1/sqrt(group_dim) per column). Without this,
SAM2's 256 dims would dominate morphology's ~78 dims 3× purely by column count.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.cluster import hierarchy as scipy_hier

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Top-level fields to log-transform (log1p) before scaling. These span orders
# of magnitude and would otherwise dominate Euclidean distances.
_LOG_TRANSFORM_FEATURES = (
    "area_um2",
    "perimeter_um",
    "major_axis_um",
    "minor_axis_um",
    "area_px",
    "nuclear_area_um2",
    "largest_nucleus_um2",
)


@dataclass
class RareCellConfig:
    """Configuration for rare cell-population discovery.

    Fields:
        feature_groups: groups passed to ``select_feature_names``. Default
            is shape+color+sam2 (~334D).
        min_cluster_size: HDBSCAN primary-run population floor.
        min_samples: HDBSCAN density-sensitivity knob (NOT a cluster-count
            target). Higher → more noise, fewer marginal clusters. ``None``
            → ``min_cluster_size // 10``. Conservative: ``= min_cluster_size``.
        stability_sizes: alternate ``min_cluster_size`` values run for
            Jaccard stability. Values equal to ``min_cluster_size`` are
            skipped. Must contain at least ``stability_min_survive``
            distinct values (else ConfigError).
        stability_jaccard: per-alt Jaccard threshold for "match".
        stability_min_survive: a primary cluster is ``stable`` iff it
            matches in ≥ this many alt runs.
        max_pcs: cap on PCA components (HDBSCAN degrades above ~30D).
        pca_variance: cumulative-variance target (capped by max_pcs).
            Applied to BOTH GPU and CPU PCA paths for equivalence.
        feature_group_weights: how to scale columns so each group
            contributes equal squared Euclidean distance.
            - ``"equal"`` (default): each column scaled by 1/sqrt(group_dim)
              so group d² contribution ≈ 1 regardless of column count.
              Without this, SAM2 (256D) drowns morphology (~78D) 3×.
            - ``"raw"``: no group weighting.
            - ``dict[str, float]``: multiplier applied on top of ``"equal"``
              (e.g., ``{"shape": 1.0, "sam2": 0.5}`` de-emphasizes SAM2).
        nuc_filter_*: N:C / overlap / n_nuclei thresholds (fractions in
            [0, 1], not percentages).
        area_filter_*: raw cell area in µm² (not log-space).
        delaunay_k: k-NN neighbors for the spatial-autocorrelation graph
            (misleading historical name — not literal Delaunay).
        use_gpu: try cuML for PCA+HDBSCAN; falls back to sklearn+hdbscan.
        seed: RNG seed. Used via ``np.random.default_rng``; also passed
            to cuML ``random_state`` where supported.
        cache_dir: write X_pca + W_knn caches here; keyed so changes to
            feature groups, pre-filter, max_pcs, pca_variance, seed, or
            weights force re-computation.
        exclude_channels: channel indices to skip in the ``channel`` group.
    """

    feature_groups: tuple[str, ...] = ("shape", "color", "sam2")
    min_cluster_size: int = 1000
    min_samples: int | None = None  # None → min_cluster_size // 10
    stability_sizes: tuple[int, ...] = (500, 1000, 2000)
    stability_jaccard: float = 0.5
    stability_min_survive: int = 2  # must survive ≥N of the stability runs
    max_pcs: int = 30
    pca_variance: float = 0.95
    feature_group_weights: str | dict[str, float] = "equal"
    nuc_filter_min_n_nuclei: int = 1
    nuc_filter_nc_min: float = 0.02
    nuc_filter_nc_max: float = 0.95
    nuc_filter_min_overlap: float = 0.8
    area_filter_min_um2: float = 20.0
    area_filter_max_um2: float = 5000.0
    delaunay_k: int = 10
    use_gpu: bool = True
    seed: int = 42
    cache_dir: Path | None = None
    exclude_channels: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class EmbeddingResult:
    """Result of :func:`load_and_embed` — wraps the detections + PCA embedding
    for downstream analyses (manifold sampling, rare-cell discovery, etc.).

    Fields:
        kept: post-pre-filter detections (aligned row-wise with ``X_pca`` /
            ``X_scaled``).
        X_pca: ``(len(kept), n_components)`` float32 PCA embedding.
        feature_names: column names for the pre-PCA matrix.
        X_scaled: ``(len(kept), n_features)`` RobustScaler + group-weighted
            matrix (pre-PCA input).
        var_explained: cumulative variance captured by ``n_components``.
        n_components: actual PCs used (≤ ``cfg.max_pcs``).
        weights_by_group: per-feature-group multiplier applied during scaling
            (``{}`` if weighting was ``"raw"``).
        pca_cache_key: 12-char hash identifying the ``X_pca_<hash>.npz`` file
            on disk (or ``""`` if ``cache_dir`` was ``None``). Downstream
            consumers that need to find the sibling cache can compute
            ``cache_dir / f"X_pca_{pca_cache_key}.npz"`` directly instead of
            re-deriving the hash.
    """

    kept: list[dict]
    X_pca: np.ndarray
    feature_names: list[str]
    X_scaled: np.ndarray
    var_explained: float
    n_components: int
    weights_by_group: dict[str, float]
    pca_cache_key: str = ""


# ---------------------------------------------------------------------------
# Step 1 — Quality pre-filter
# ---------------------------------------------------------------------------


def pre_filter_cells(
    detections: list[dict], cfg: RareCellConfig
) -> tuple[list[dict], dict, list[str | None]]:
    """Drop Cellpose-suspect cells before clustering.

    Returns:
        ``(kept_detections, stats, drop_reasons)`` where:
          * ``stats`` has keys ``"input", "dropped_n_nuclei", "dropped_nc_ratio",
            "dropped_overlap", "dropped_area", "kept"``.
          * ``drop_reasons`` is a list the same length as ``detections``,
            with the reason string for each dropped cell (one of
            ``"n_nuclei", "nc_ratio", "overlap", "area"``) or ``None`` for
            cells that passed. This drives the ``-2`` sentinel + optional
            ``rare_pop_filter_reason`` field downstream.
    """
    stats = {
        "input": len(detections),
        "dropped_n_nuclei": 0,
        "dropped_nc_ratio": 0,
        "dropped_overlap": 0,
        "dropped_area": 0,
        "kept": 0,
    }
    kept: list[dict] = []
    drop_reasons: list[str | None] = [None] * len(detections)
    for i, det in enumerate(detections):
        feats = det.get("features", {})
        area_um2 = feats.get("area_um2", det.get("area_um2", 0.0))
        n_nuclei = feats.get("n_nuclei", det.get("n_nuclei", 0))
        nc_ratio = feats.get("nuclear_area_fraction", det.get("nuclear_area_fraction", 0.0))
        # overlap_fraction: average across all per-nucleus entries, if present
        nuclei = det.get("nuclei") or []
        if nuclei:
            overlap_vals = [
                float(n.get("overlap_fraction", 1.0)) for n in nuclei if isinstance(n, dict)
            ]
            overlap = float(np.mean(overlap_vals)) if overlap_vals else 1.0
        else:
            overlap = 1.0

        if n_nuclei < cfg.nuc_filter_min_n_nuclei:
            stats["dropped_n_nuclei"] += 1
            drop_reasons[i] = "n_nuclei"
            continue
        if not (cfg.nuc_filter_nc_min <= nc_ratio <= cfg.nuc_filter_nc_max):
            stats["dropped_nc_ratio"] += 1
            drop_reasons[i] = "nc_ratio"
            continue
        if overlap < cfg.nuc_filter_min_overlap:
            stats["dropped_overlap"] += 1
            drop_reasons[i] = "overlap"
            continue
        if not (cfg.area_filter_min_um2 <= area_um2 <= cfg.area_filter_max_um2):
            stats["dropped_area"] += 1
            drop_reasons[i] = "area"
            continue
        kept.append(det)

    stats["kept"] = len(kept)
    return kept, stats, drop_reasons


# ---------------------------------------------------------------------------
# Step 2-3 — Feature matrix, log transform, RobustScaler, PCA
# ---------------------------------------------------------------------------


def build_feature_matrix(
    detections: list[dict], feature_groups: tuple[str, ...], exclude_channels: tuple[int, ...] = ()
) -> tuple[np.ndarray, list[str], list[int]]:
    """Extract feature matrix using ``select_feature_names`` from cluster_features.

    Returns:
        ``(X, kept_feature_names, valid_indices)`` — X has shape
        ``(len(valid_indices), len(kept_feature_names))``. Detections with
        missing/non-finite features are silently skipped and excluded from
        ``valid_indices``.
    """
    from xldvp_seg.analysis.cluster_features import _extract_feature_matrix, select_feature_names

    names = select_feature_names(
        detections, set(feature_groups), exclude_channels=set(exclude_channels)
    )
    if not names:
        raise ConfigError(
            f"No features matched groups {feature_groups}; "
            f"check detections have features dict populated."
        )
    # _extract_feature_matrix returns (X, feature_names, valid_indices) — note
    # the order: feature_names comes BEFORE valid_indices.
    X, kept_names, valid_indices = _extract_feature_matrix(detections, names)
    if len(valid_indices) < len(detections):
        logger.warning(
            "Feature extraction dropped %d/%d detections with missing/non-finite values",
            len(detections) - len(valid_indices),
            len(detections),
        )
    return X, kept_names, list(valid_indices)


def log_transform_copy(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Return a float32 copy of ``X`` with ``np.log1p`` applied to scale-spanning
    columns (area/perimeter/axes, listed in ``_LOG_TRANSFORM_FEATURES``).

    Always copies — ``X`` is not modified. Named ``_copy`` to reflect behavior
    (the earlier ``log_transform_inplace`` name was misleading; that alias
    remains as a re-export for backward compatibility).
    """
    X = X.astype(np.float32, copy=True)
    for i, name in enumerate(feature_names):
        if name in _LOG_TRANSFORM_FEATURES:
            col = X[:, i]
            # Clamp negatives to 0 (shouldn't happen for areas, but defensive)
            X[:, i] = np.log1p(np.clip(col, 0, None))
    return X


# Backward-compatible alias — do not use in new code.
log_transform_inplace = log_transform_copy


def apply_group_weights(
    X: np.ndarray,
    feature_names: list[str],
    mode: str | dict[str, float] = "equal",
) -> tuple[np.ndarray, dict[str, float]]:
    """Weight columns so each feature group contributes ~equal squared
    Euclidean distance regardless of column count.

    Without this, a 256D SAM2 block would dominate a 78D morph block 3× in
    HDBSCAN density estimation purely by dimensionality count.

    Args:
        X: scaled (post-RobustScaler) feature matrix ``(N, D)``.
        feature_names: column names, length D.
        mode:
          * ``"equal"`` — multiply each column by ``1/sqrt(group_dim)`` so the
            group's expected d² contribution is ~1.
          * ``"raw"`` — no weighting (identity).
          * ``dict[str, float]`` — explicit per-group multiplier on top of
            ``"equal"``. Missing groups default to 1.0.

    Returns:
        ``(X_weighted, weights_by_group)`` — weights_by_group is the effective
        per-group column multiplier used (for caching + provenance).
    """
    from xldvp_seg.analysis.cluster_features import classify_feature_group

    if mode == "raw":
        return X.astype(np.float32, copy=True), {}

    groups = [classify_feature_group(n) or "other" for n in feature_names]
    dims_by_group: dict[str, int] = {}
    for g in groups:
        dims_by_group[g] = dims_by_group.get(g, 0) + 1

    if isinstance(mode, dict):
        explicit = {k: float(v) for k, v in mode.items()}
    elif mode == "equal":
        explicit = {}
    else:
        raise ConfigError(f"feature_group_weights must be 'equal', 'raw', or dict; got {mode!r}")

    weights_by_group = {
        g: explicit.get(g, 1.0) / float(np.sqrt(max(1, d))) for g, d in dims_by_group.items()
    }
    col_weights = np.array([weights_by_group[g] for g in groups], dtype=np.float32)
    X_w = (X * col_weights).astype(np.float32)
    logger.info(
        "Group weights (mode=%s): %s",
        mode if isinstance(mode, str) else "custom",
        ", ".join(
            f"{g}(d={dims_by_group[g]}, w={weights_by_group[g]:.3f})" for g in sorted(dims_by_group)
        ),
    )
    return X_w, weights_by_group


def scale_and_pca(
    X: np.ndarray, cfg: RareCellConfig, feature_names: list[str] | None = None
) -> tuple[np.ndarray, float, int, np.ndarray, dict[str, float]]:
    """RobustScaler → group weighting → PCA (GPU if available, else CPU).

    Both GPU (cuML) and CPU (sklearn) paths apply the same ``pca_variance``
    trim so the same config produces equivalent output on either device.

    Args:
        X: post-log1p feature matrix ``(N, D)``.
        cfg: RareCellConfig (``max_pcs``, ``pca_variance``, ``use_gpu``,
            ``seed``, ``feature_group_weights``).
        feature_names: column names (required for group weighting; pass
            ``None`` to skip weighting).

    Returns:
        ``(X_pca, var_explained, n_components, X_scaled, weights_by_group)``.
        ``X_scaled`` is the scaled+weighted matrix (used later for
        centroid-space feature interpretation — avoids the earlier double
        RobustScaler fit). ``weights_by_group`` goes into the cache-key hash.
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    if feature_names is not None:
        X_scaled, weights_by_group = apply_group_weights(
            X_scaled, feature_names, cfg.feature_group_weights
        )
    else:
        weights_by_group = {}

    n_pcs_cap = min(cfg.max_pcs, *X_scaled.shape)

    if cfg.use_gpu:
        try:
            import cupy as cp
            from cuml.decomposition import PCA  # noqa: N811

            X_gpu = cp.asarray(X_scaled)
            pca_gpu = PCA(n_components=n_pcs_cap)
            X_pca_gpu = pca_gpu.fit_transform(X_gpu)
            evr = cp.asnumpy(pca_gpu.explained_variance_ratio_)
            X_pca_host = cp.asnumpy(X_pca_gpu).astype(np.float32)
            cum = np.cumsum(evr)
            n_for_target = int(np.searchsorted(cum, cfg.pca_variance) + 1)
            n_used = min(n_for_target, n_pcs_cap)
            X_pca = X_pca_host[:, :n_used]
            var_used = float(cum[n_used - 1])
            logger.info(
                "PCA (cuML GPU): %d components used, %.1f%% variance "
                "(capped at max_pcs=%d; %d needed for %.0f%% target)",
                n_used,
                100 * var_used,
                cfg.max_pcs,
                n_for_target,
                100 * cfg.pca_variance,
            )
            return X_pca, var_used, n_used, X_scaled, weights_by_group
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            logger.info("cuML PCA unavailable (%s), falling back to sklearn", e)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_pcs_cap, random_state=cfg.seed)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_for_target = int(np.searchsorted(cum, cfg.pca_variance) + 1)
    n_used = min(n_for_target, n_pcs_cap)
    X_pca = X_pca[:, :n_used]
    var_used = float(cum[n_used - 1])
    logger.info(
        "PCA (sklearn CPU): %d components used, %.1f%% variance "
        "(capped at max_pcs=%d; %d needed for %.0f%% target)",
        n_used,
        100 * var_used,
        cfg.max_pcs,
        n_for_target,
        100 * cfg.pca_variance,
    )
    return X_pca, var_used, n_used, X_scaled, weights_by_group


# ---------------------------------------------------------------------------
# Step 4 — HDBSCAN
# ---------------------------------------------------------------------------


def run_hdbscan(
    X_pca: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    use_gpu: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run HDBSCAN. Returns (labels, probabilities, cluster_persistence).

    ``persistence`` is a 1D array of length ``n_clusters`` where index ``i``
    corresponds to the ``i``-th cluster in ascending-cluster-id order. The
    function explicitly aligns/pads so the caller can safely index
    ``persistence[i]`` against ``np.unique(labels[labels >= 0])[i]``.
    """
    if min_samples is None:
        min_samples = max(1, min_cluster_size // 10)

    labels: np.ndarray
    probs: np.ndarray
    persistence: np.ndarray

    if use_gpu:
        try:
            import cupy as cp
            from cuml.cluster import HDBSCAN  # noqa: N811

            X_gpu = cp.asarray(X_pca)
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method="eom",
                gen_min_span_tree=True,
            )
            clusterer.fit(X_gpu)
            labels = cp.asnumpy(clusterer.labels_).astype(np.int32)
            probs = (
                cp.asnumpy(clusterer.probabilities_).astype(np.float32)
                if clusterer.probabilities_ is not None
                else np.ones(len(labels), dtype=np.float32)
            )
            persistence = (
                cp.asnumpy(clusterer.cluster_persistence_).astype(np.float32)
                if hasattr(clusterer, "cluster_persistence_")
                and clusterer.cluster_persistence_ is not None
                else _fallback_persistence(labels)
            )
            persistence = _align_persistence(labels, persistence)
            logger.info(
                "HDBSCAN (cuML GPU): min_cluster_size=%d, min_samples=%d → %d clusters, %d noise",
                min_cluster_size,
                min_samples,
                len(persistence),
                int(np.sum(labels == -1)),
            )
            return labels, probs, persistence
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            logger.info("cuML HDBSCAN unavailable (%s), falling back to CPU hdbscan", e)

    import hdbscan as hdbscan_cpu

    clusterer = hdbscan_cpu.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    clusterer.fit(X_pca)
    labels = clusterer.labels_.astype(np.int32)
    probs = clusterer.probabilities_.astype(np.float32)
    persistence = np.asarray(clusterer.cluster_persistence_, dtype=np.float32)
    persistence = _align_persistence(labels, persistence)
    logger.info(
        "HDBSCAN (CPU): min_cluster_size=%d, min_samples=%d → %d clusters, %d noise",
        min_cluster_size,
        min_samples,
        len(persistence),
        int(np.sum(labels == -1)),
    )
    return labels, probs, persistence


def _fallback_persistence(labels: np.ndarray) -> np.ndarray:
    """Return zeros of length n_clusters when cluster_persistence_ unavailable."""
    n = int(labels.max()) + 1 if labels.size > 0 and labels.max() >= 0 else 0
    return np.zeros(n, dtype=np.float32)


def _align_persistence(labels: np.ndarray, persistence: np.ndarray) -> np.ndarray:
    """Pad or truncate ``persistence`` to match ``len(unique(labels[labels>=0]))``.

    HDBSCAN's ``cluster_persistence_`` is usually in ascending cluster-id order
    but we don't assume it — we just enforce length. Logs a warning on mismatch
    so ordering bugs surface loudly instead of silently mis-labeling.
    """
    n_clusters = int(np.unique(labels[labels >= 0]).size)
    if persistence.size == n_clusters:
        return persistence.astype(np.float32, copy=False)
    logger.warning(
        "HDBSCAN persistence length mismatch (%d vs %d clusters); padding/truncating",
        persistence.size,
        n_clusters,
    )
    out = np.zeros(n_clusters, dtype=np.float32)
    out[: min(n_clusters, persistence.size)] = persistence[
        : min(n_clusters, persistence.size)
    ].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Step 5 — Stability via Jaccard
# ---------------------------------------------------------------------------


def compute_stability(
    primary_labels: np.ndarray,
    alt_runs: list[np.ndarray],
    jaccard_threshold: float = 0.5,
    min_survive: int = 2,
) -> np.ndarray:
    """Return boolean array marking which primary clusters are stable.

    For each alt run, build a contingency matrix C[i, j] = |P_i ∩ A_j|, then
    compute the full Jaccard matrix in one pass. A primary cluster is credited
    with **one** survival per alt run via **reciprocal best-match**: cluster
    ``P_i`` survives iff some alt cluster ``A_j`` has ``argmax_i J(P_i, A_j) = i``
    and ``J(P_i, A_j) >= jaccard_threshold``. Each alt cluster can credit at
    most one primary, eliminating the old "any-match" bug where two primaries
    could both claim the same merged alt cluster.

    ``stable = survivals >= min_survive``.

    Raises:
        ConfigError: if ``alt_runs`` is empty or ``min_survive`` exceeds
            ``len(alt_runs)`` (unachievable threshold).
    """
    if not alt_runs:
        raise ConfigError(
            "compute_stability: alt_runs is empty; supply stability_sizes "
            "that include values != min_cluster_size."
        )
    if min_survive > len(alt_runs):
        raise ConfigError(
            f"compute_stability: min_survive={min_survive} exceeds "
            f"len(alt_runs)={len(alt_runs)}; threshold is unachievable."
        )

    primary_ids = np.unique(primary_labels[primary_labels >= 0])
    Kp = len(primary_ids)
    survivals = np.zeros(Kp, dtype=np.int32)
    if Kp == 0:
        return survivals.astype(bool)

    primary_remap = {int(cid): i for i, cid in enumerate(primary_ids)}
    # Full primary sizes (count cells regardless of what alt assigns them to).
    p_sizes_true = np.array(
        [int(np.sum(primary_labels == cid)) for cid in primary_ids], dtype=np.int64
    )

    for alt in alt_runs:
        alt_ids = np.unique(alt[alt >= 0])
        if alt_ids.size == 0:
            continue
        alt_remap = {int(cid): i for i, cid in enumerate(alt_ids)}

        # Contingency matrix C[i, j] = |P_i ∩ A_j| via one vectorized add.at
        # over cells that are non-noise in BOTH labelings (union with noise
        # has no overlap contribution).
        mask = (primary_labels >= 0) & (alt >= 0)
        p_idx = np.fromiter(
            (primary_remap[int(c)] for c in primary_labels[mask]),
            dtype=np.int64,
            count=int(mask.sum()),
        )
        a_idx = np.fromiter(
            (alt_remap[int(c)] for c in alt[mask]),
            dtype=np.int64,
            count=int(mask.sum()),
        )
        C = np.zeros((Kp, alt_ids.size), dtype=np.int64)
        np.add.at(C, (p_idx, a_idx), 1)

        # Full cluster sizes for the union denominator (a cell in P_i but
        # alt-noise still counts toward |P_i|, etc.).
        a_sizes_true = np.array([int(np.sum(alt == cid)) for cid in alt_ids], dtype=np.int64)
        union = p_sizes_true[:, None] + a_sizes_true[None, :] - C
        jaccard = np.where(union > 0, C / np.maximum(union, 1), 0.0)

        # Reciprocal best-match: for each alt cluster j, find its best primary
        # i_*(j) = argmax_i J(P_i, A_j). Credit primary i if its best alt k
        # (k_*(i) = argmax_j J(P_i, A_j)) has i_*(k) = i AND J >= threshold.
        if alt_ids.size == 0 or Kp == 0:
            continue
        best_alt_for_primary = jaccard.argmax(axis=1)  # shape (Kp,)
        best_j_for_primary = jaccard.max(axis=1)
        best_primary_for_alt = jaccard.argmax(axis=0)  # shape (Ka,)
        for i in range(Kp):
            if best_j_for_primary[i] < jaccard_threshold:
                continue
            k = int(best_alt_for_primary[i])
            if int(best_primary_for_alt[k]) == i:
                survivals[i] += 1

    return survivals >= min_survive


# ---------------------------------------------------------------------------
# Step 6 — Delaunay adjacency + vectorized Moran's I
# ---------------------------------------------------------------------------


def build_knn_adjacency(points_um: np.ndarray, k: int = 10) -> sparse.csr_matrix:
    """kNN adjacency on 2D points (used for spatial autocorrelation; not
    literal Delaunay despite the older ``build_delaunay_knn`` alias).

    At 500K points, scipy's Delaunay tessellation produces variable-degree
    graphs that complicate Moran's I normalization. A symmetric k-NN graph
    is denser and has near-uniform degree — simpler + faster for the
    Moran's I use case.

    Returns a row-normalized CSR matrix (each row sums to 1.0).
    """
    from scipy.spatial import KDTree

    tree = KDTree(points_um)
    _, indices = tree.query(points_um, k=k + 1)  # +1 to include self
    n = points_um.shape[0]
    # scipy may return fewer than k+1 neighbors when N < k+1; guard here.
    k_actual = indices.shape[1] - 1
    row = np.repeat(np.arange(n), k_actual)
    col = indices[:, 1:].ravel()  # skip self
    data = np.ones(n * k_actual, dtype=np.float32)
    W = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    # Symmetrize (union of directed edges). Binary graph so maximum == OR.
    W = W.maximum(W.T)
    # Row-normalize — standard convention for Moran's I with row-stochastic W.
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D_inv = sparse.diags(1.0 / row_sums)
    return (D_inv @ W).tocsr()


# Backward-compatible alias — do not use in new code.
build_delaunay_knn = build_knn_adjacency


def morans_i_vectorized(W: sparse.csr_matrix, memberships: np.ndarray) -> np.ndarray:
    """Compute Moran's I for many binary membership vectors in one pass.

    Args:
        W: row-normalized (N, N) sparse adjacency matrix.
        memberships: (N, K) 0/1 matrix (dense or sparse), one column per
            cluster.

    Returns:
        1D array of length K with Moran's I per cluster. Degenerate columns
        (all-0 or all-1 → ``z ≡ 0`` → ``denom ≈ 0``) return ``np.nan`` so
        downstream formatting can flag "spatial autocorrelation undefined"
        rather than silently emit 0.0.

    With a row-stochastic W the standard prefactor ``N/S0`` simplifies to 1
    since ``S0 = N``, so Moran's I reduces to ``(zᵀ W z) / (zᵀ z)``.
    """
    # Work in dense for centering; membership matrices are small (N × K) with
    # K typically <100. If sparse was passed, densify here.
    if sparse.issparse(memberships):
        M = memberships.toarray()
    else:
        M = np.asarray(memberships)
    means = M.mean(axis=0)
    Z = M - means
    WZ = W @ Z
    numer = (Z * WZ).sum(axis=0)
    denom = (Z * Z).sum(axis=0)
    eps = 1e-12
    out = np.where(denom > eps, numer / np.maximum(denom, eps), np.nan).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Step 8 — Ward linkage on cluster centroids
# ---------------------------------------------------------------------------


def compute_centroids(X_pca: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centroids (K, d), cluster_ids_sorted (K,))``.

    Single-pass vectorized via ``np.add.at`` — avoids K separate boolean-mask
    allocations over 500K-row labels.
    """
    cluster_ids = np.unique(labels[labels >= 0])
    d = X_pca.shape[1]
    if cluster_ids.size == 0:
        return np.zeros((0, d), dtype=np.float32), cluster_ids
    remap = {int(c): i for i, c in enumerate(cluster_ids)}
    dense = np.fromiter((remap.get(int(c), -1) for c in labels), dtype=np.int64, count=len(labels))
    valid = dense >= 0
    centroids = np.zeros((cluster_ids.size, d), dtype=np.float32)
    np.add.at(centroids, dense[valid], X_pca[valid])
    counts = np.bincount(dense[valid], minlength=cluster_ids.size).astype(np.float32)
    counts[counts == 0] = 1.0  # no-op guard; shouldn't happen since we took unique
    centroids /= counts[:, None]
    return centroids, cluster_ids


def ward_linkage_on_centroids(centroids: np.ndarray) -> np.ndarray:
    """Ward-linkage hierarchical clustering on cluster centroids.

    Returns the scipy linkage matrix (K-1, 4). Empty if fewer than 2 centroids.
    """
    if len(centroids) < 2:
        return np.zeros((0, 4), dtype=np.float32)
    return scipy_hier.linkage(centroids, method="ward")


# ---------------------------------------------------------------------------
# Step 9 — Assemble cluster summary
# ---------------------------------------------------------------------------


def summarize_clusters(
    labels: np.ndarray,
    persistence: np.ndarray,
    moran_i: np.ndarray,
    stable: np.ndarray,
    detections: list[dict],
    feature_names: list[str],
    X_scaled: np.ndarray,
) -> list[dict]:
    """Build per-cluster summary rows. ``moran_i`` values that are NaN
    (degenerate clusters with no spatial variance) are emitted as ``None``."""
    cluster_ids = np.unique(labels[labels >= 0])
    rows = []
    for i, cid in enumerate(cluster_ids):
        mask = labels == cid
        size = int(mask.sum())
        # Top regions (by count in this cluster)
        region_counts: dict[int, int] = {}
        for det_idx in np.flatnonzero(mask):
            rid = detections[det_idx].get("organ_id", 0)
            if rid:
                region_counts[int(rid)] = region_counts.get(int(rid), 0) + 1
        top_regions = sorted(region_counts.items(), key=lambda kv: -kv[1])[:5]
        # Top morph features: centroid z-score magnitude (scaled space)
        centroid = X_scaled[mask].mean(axis=0)
        top_idx = np.argsort(-np.abs(centroid))[:10]
        top_feats = [(feature_names[j], round(float(centroid[j]), 3)) for j in top_idx]
        m = float(moran_i[i])
        rows.append(
            {
                "cluster_id": int(cid),
                "size": size,
                "hdbscan_persistence": round(float(persistence[i]), 4),
                "moran_i": round(m, 4) if np.isfinite(m) else None,
                "stable": bool(stable[i]),
                "top_regions": ";".join(f"{r}:{c}" for r, c in top_regions),
                "top_morph_features": ";".join(f"{n}:{v}" for n, v in top_feats),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(feature_names: list[str], n_cells: int, cfg: RareCellConfig) -> str:
    """Hash identifying a unique (features × pre-filter × PCA × weighting)
    configuration. Any of these changing must invalidate the cache, else the
    user silently reads a stale ``X_pca`` array shaped for old config.
    """
    m = hashlib.sha256()
    m.update(str(sorted(feature_names)).encode())
    m.update(f"n={n_cells}".encode())
    m.update(
        f"filter={cfg.nuc_filter_min_n_nuclei},{cfg.nuc_filter_nc_min},"
        f"{cfg.nuc_filter_nc_max},{cfg.nuc_filter_min_overlap},"
        f"{cfg.area_filter_min_um2},{cfg.area_filter_max_um2}".encode()
    )
    m.update(f"pca={cfg.max_pcs},{cfg.pca_variance},seed={cfg.seed}".encode())
    m.update(f"excl={sorted(cfg.exclude_channels)}".encode())
    m.update(f"weights={cfg.feature_group_weights!r}".encode())
    return m.hexdigest()[:12]


from xldvp_seg.utils.json_utils import atomic_savez as _atomic_savez  # shared helper


def _atomic_save_npz(path: Path, matrix: sparse.csr_matrix) -> None:
    """``sparse.save_npz`` via temp file + ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        sparse.save_npz(f, matrix)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def discover_rare_cell_types(
    detections: list[dict], cfg: RareCellConfig | None = None
) -> dict[str, Any]:
    """Run the full rare-cell-population discovery pipeline.

    Mutates ``detections`` in place. Every detection ends up with a
    ``rare_pop_id`` field:

      * ``0, 1, 2, ...`` — member of a density-based cluster.
      * ``-1`` — HDBSCAN noise (passed pre-filter, not dense enough).
      * ``-2`` — dropped by pre-filter. Also sets ``rare_pop_filter_reason``
        to one of ``"n_nuclei" | "nc_ratio" | "overlap" | "area"`` so the
        reason is recoverable without re-running filters.

    Returns a dict with:
        - ``detections`` — original list (kept cells modified in place, dropped
          cells tagged ``-2``).
        - ``kept_detections`` — list of post-filter + post-feature-extraction
          cells (aligned with ``labels`` / ``probabilities``).
        - ``cluster_summary`` — list of per-cluster dicts.
        - ``linkage`` — Ward linkage on cluster centroids ``(K-1, 4)``.
        - ``cluster_ids`` — 1D array of cluster IDs ordered as centroid rows.
        - ``prefilter_stats`` — pre-filter counts dict.
        - ``pca_variance`` — cumulative variance captured.
        - ``pca_n_components`` — number of PCs actually used.
        - ``noise_pct`` — fraction of kept cells labeled noise.
        - ``labels`` — HDBSCAN labels over ``kept_detections``.
        - ``probabilities`` — HDBSCAN probs over ``kept_detections``.
        - ``feature_names`` — column names used for PCA input.
        - ``feature_group_weights`` — effective per-group multiplier used.
    """
    cfg = cfg or RareCellConfig()
    rng = np.random.default_rng(cfg.seed)  # noqa: F841  (reserved for future use)

    # --- 1. Pre-filter ---
    logger.info("Pre-filtering %d detections", len(detections))
    kept, pf_stats, drop_reasons = pre_filter_cells(detections, cfg)
    logger.info(
        "Pre-filter: kept %d/%d (dropped: n_nuclei=%d, nc=%d, overlap=%d, area=%d)",
        pf_stats["kept"],
        pf_stats["input"],
        pf_stats["dropped_n_nuclei"],
        pf_stats["dropped_nc_ratio"],
        pf_stats["dropped_overlap"],
        pf_stats["dropped_area"],
    )

    # Tag filter-dropped cells with -2 sentinel + reason immediately. Kept
    # cells will have their HDBSCAN label (including noise -1) written below.
    for det, reason in zip(detections, drop_reasons):
        if reason is not None:
            det["rare_pop_id"] = -2
            det["rare_pop_filter_reason"] = reason

    if pf_stats["kept"] < cfg.min_cluster_size * 2:
        raise ConfigError(
            f"Too few cells after pre-filter ({pf_stats['kept']}) for "
            f"min_cluster_size={cfg.min_cluster_size}. Need ≥ 2×min_cluster_size."
        )

    # --- 2-3. Features + log + scale + PCA (with caching) ---
    X_raw, feature_names, valid_indices = build_feature_matrix(
        kept, cfg.feature_groups, cfg.exclude_channels
    )
    # Filter kept to those with valid features (and sync drop_reasons).
    if len(valid_indices) < len(kept):
        valid_set = set(valid_indices)
        dropped_by_features = [kept[i] for i in range(len(kept)) if i not in valid_set]
        for det in dropped_by_features:
            # Treat missing-feature drops as a special pre-filter reason.
            det["rare_pop_id"] = -2
            det["rare_pop_filter_reason"] = "missing_features"
        kept = [kept[i] for i in valid_indices]
    X_log = log_transform_copy(X_raw, feature_names)

    ckey = _cache_key(feature_names, len(kept), cfg)
    X_pca = None
    var_explained = 0.0
    n_components = 0
    X_scaled: np.ndarray | None = None
    weights_by_group: dict[str, float] = {}
    cache_hit = False
    if cfg.cache_dir is not None:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        pca_cache = cfg.cache_dir / f"X_pca_{ckey}.npz"
        if pca_cache.exists():
            try:
                data = np.load(pca_cache)
                X_pca = data["X_pca"]
                X_scaled = data["X_scaled"]
                var_explained = float(data["var_explained"])
                n_components = int(data["n_components"])
                cache_hit = True
                logger.info(
                    "PCA cache hit: %s (%d components, %.1f%% var)",
                    pca_cache.name,
                    n_components,
                    100 * var_explained,
                )
            except (OSError, ValueError, KeyError) as e:
                logger.warning("PCA cache load failed (%s) — recomputing", e)
                cache_hit = False
    if not cache_hit:
        X_pca, var_explained, n_components, X_scaled, weights_by_group = scale_and_pca(
            X_log, cfg, feature_names
        )
        if cfg.cache_dir is not None:
            _atomic_savez(
                cfg.cache_dir / f"X_pca_{ckey}.npz",
                X_pca=X_pca,
                X_scaled=X_scaled,
                var_explained=np.float64(var_explained),
                n_components=np.int64(n_components),
            )

    assert X_pca is not None and X_scaled is not None

    # --- 4. Primary HDBSCAN ---
    labels, probs, persistence = run_hdbscan(
        X_pca,
        cfg.min_cluster_size,
        cfg.min_samples,
        use_gpu=cfg.use_gpu,
        random_state=cfg.seed,
    )

    # --- 5. Stability check ---
    alt_runs = []
    for alt_size in cfg.stability_sizes:
        if alt_size == cfg.min_cluster_size:
            continue
        alt_labels, _, _ = run_hdbscan(
            X_pca,
            alt_size,
            cfg.min_samples,
            use_gpu=cfg.use_gpu,
            random_state=cfg.seed,
        )
        alt_runs.append(alt_labels)

    n_primary = int(np.unique(labels[labels >= 0]).size)
    if alt_runs:
        stable = compute_stability(
            labels, alt_runs, cfg.stability_jaccard, cfg.stability_min_survive
        )
    else:
        logger.warning(
            "No alternate stability runs (stability_sizes is only %d) — "
            "all clusters flagged stable=False (cannot assess).",
            cfg.min_cluster_size,
        )
        stable = np.zeros(n_primary, dtype=bool)

    # --- 6. Moran's I (sparse one-hot membership; single matvec) ---
    positions = np.array([d.get("global_center_um", [0, 0]) for d in kept], dtype=np.float32)
    have_positions = positions.size > 0 and not np.all(positions == 0)
    if have_positions:
        pos_hash = hashlib.sha256(positions.tobytes()).hexdigest()[:12]
        W_cache = (
            cfg.cache_dir / f"W_knn_k{cfg.delaunay_k}_{pos_hash}.npz" if cfg.cache_dir else None
        )
        W = None
        if W_cache is not None and W_cache.exists():
            try:
                W = sparse.load_npz(W_cache)
                logger.info("Moran's I: loaded cached kNN W (%s)", W_cache.name)
            except (OSError, ValueError) as e:
                logger.warning("W cache load failed (%s) — rebuilding", e)
                W = None
        if W is None:
            W = build_knn_adjacency(positions, cfg.delaunay_k)
            if W_cache is not None:
                _atomic_save_npz(W_cache, W)

        cluster_ids_moran = np.unique(labels[labels >= 0])
        if cluster_ids_moran.size > 0:
            remap = {int(c): i for i, c in enumerate(cluster_ids_moran)}
            dense_idx = np.fromiter(
                (remap.get(int(c), -1) for c in labels), dtype=np.int64, count=len(labels)
            )
            v = dense_idx >= 0
            rows_idx = np.flatnonzero(v)
            M = sparse.csr_matrix(
                (
                    np.ones(rows_idx.size, dtype=np.float32),
                    (rows_idx, dense_idx[v]),
                ),
                shape=(len(labels), cluster_ids_moran.size),
            )
            moran_i = morans_i_vectorized(W, M)
        else:
            moran_i = np.zeros(0, dtype=np.float32)
    else:
        logger.warning("No global_center_um on detections; Moran's I set to NaN")
        moran_i = np.full(n_primary, np.nan, dtype=np.float32)

    # --- 7. Write rare_pop_id + hdbscan_prob back onto kept detections ---
    for det, lbl, p in zip(kept, labels, probs):
        det["rare_pop_id"] = int(lbl)
        det["hdbscan_prob"] = float(p)  # full precision; consumer rounds if needed.
        det.pop("rare_pop_filter_reason", None)

    # --- 8. Centroids + Ward linkage ---
    centroids, cluster_ids = compute_centroids(X_pca, labels)
    linkage = ward_linkage_on_centroids(centroids)

    # --- 9. Cluster summary ---
    summary = summarize_clusters(
        labels, persistence, moran_i, stable, kept, feature_names, X_scaled
    )

    noise_pct = float(np.mean(labels == -1)) if labels.size else 0.0
    for row in summary:
        row["noise_pct"] = round(noise_pct, 4)

    return {
        "detections": detections,  # original list — kept cells modified in place
        "kept_detections": kept,
        "cluster_summary": summary,
        "linkage": linkage,
        "cluster_ids": cluster_ids,
        "prefilter_stats": pf_stats,
        "pca_variance": var_explained,
        "pca_n_components": n_components,
        "noise_pct": noise_pct,
        "labels": labels,
        "probabilities": probs,
        "feature_names": feature_names,
        "feature_group_weights": weights_by_group,
    }


# ---------------------------------------------------------------------------
# Shared helper — load + pre-filter + feature matrix + scale + PCA (no HDBSCAN)
# ---------------------------------------------------------------------------


def load_and_embed(
    detections: list[dict] | str | Path,
    cfg: RareCellConfig,
) -> EmbeddingResult:
    """Load detections, pre-filter, build feature matrix, log1p, RobustScale +
    group-weight, and PCA — stopping before HDBSCAN.

    This is the shared front half of :func:`discover_rare_cell_types` packaged
    for downstream analyses (manifold sampling, rare-cell review, etc.) that
    need the same embedding but not the density clustering.

    Cache-aware: reuses the ``X_pca_<hash>.npz`` cache written by
    :func:`discover_rare_cell_types` when present (same cache-key logic as
    :func:`_cache_key`), and writes one on miss if ``cfg.cache_dir`` is set.

    Args:
        detections: list of detection dicts, OR a path (str / Path) to a JSON
            file loaded via :func:`~xldvp_seg.utils.json_utils.fast_json_load`.
        cfg: :class:`RareCellConfig` — uses ``feature_groups``,
            ``exclude_channels``, pre-filter thresholds, ``max_pcs``,
            ``pca_variance``, ``feature_group_weights``, ``use_gpu``, ``seed``,
            and ``cache_dir``.

    Returns:
        :class:`EmbeddingResult` with kept detections + PCA embedding.

    Raises:
        ConfigError: too few cells survived the pre-filter (requires
            ``≥ 2 × cfg.min_cluster_size``), or no features matched the
            requested groups.
    """
    # --- 0. Load (if path) ---
    if isinstance(detections, (str, Path)):
        from xldvp_seg.utils.json_utils import fast_json_load

        logger.info("Loading detections from %s", detections)
        detections = fast_json_load(str(detections))
    if not isinstance(detections, list):
        raise ConfigError(
            f"detections must be a list of dicts or a path; got {type(detections).__name__}"
        )

    # --- 1. Pre-filter ---
    logger.info("Pre-filtering %d detections", len(detections))
    kept, pf_stats, _drop_reasons = pre_filter_cells(detections, cfg)
    logger.info(
        "Pre-filter: kept %d/%d (dropped: n_nuclei=%d, nc=%d, overlap=%d, area=%d)",
        pf_stats["kept"],
        pf_stats["input"],
        pf_stats["dropped_n_nuclei"],
        pf_stats["dropped_nc_ratio"],
        pf_stats["dropped_overlap"],
        pf_stats["dropped_area"],
    )
    # NOTE: the HDBSCAN-specific ``kept >= 2 * min_cluster_size`` guard lives
    # in :func:`discover_rare_cell_types` — it's not a universal requirement
    # of the embedding pipeline, and other callers (e.g. manifold sampling)
    # legitimately use a much smaller anchor count.

    # --- 2. Feature matrix (drop rows with missing/non-finite values) ---
    X_raw, feature_names, valid_indices = build_feature_matrix(
        kept, cfg.feature_groups, cfg.exclude_channels
    )
    if len(valid_indices) < len(kept):
        kept = [kept[i] for i in valid_indices]

    # --- 3. log1p on scale-spanning columns ---
    X_log = log_transform_copy(X_raw, feature_names)

    # --- 4. Scale + PCA (with cache) ---
    ckey = _cache_key(feature_names, len(kept), cfg)
    X_pca: np.ndarray | None = None
    X_scaled: np.ndarray | None = None
    var_explained = 0.0
    n_components = 0
    weights_by_group: dict[str, float] = {}
    cache_hit = False
    if cfg.cache_dir is not None:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        pca_cache = cfg.cache_dir / f"X_pca_{ckey}.npz"
        if pca_cache.exists():
            try:
                data = np.load(pca_cache, allow_pickle=False)
                X_pca = data["X_pca"]
                X_scaled = data["X_scaled"]
                var_explained = float(data["var_explained"])
                n_components = int(data["n_components"])
                # weights_by_group is stored as parallel (names, values) arrays
                # so we avoid allow_pickle=True (safer + faster). Older caches
                # without the keys simply recompute via weights_by_group={}.
                if "weights_group_names" in data.files and "weights_group_values" in data.files:
                    names = [str(s) for s in data["weights_group_names"]]
                    values = [float(v) for v in data["weights_group_values"]]
                    weights_by_group = dict(zip(names, values))
                cache_hit = True
                logger.info(
                    "PCA cache hit: %s (%d components, %.1f%% var)",
                    pca_cache.name,
                    n_components,
                    100 * var_explained,
                )
            except (OSError, ValueError, KeyError) as e:
                logger.warning("PCA cache load failed (%s) — recomputing", e)
                cache_hit = False
    if not cache_hit:
        X_pca, var_explained, n_components, X_scaled, weights_by_group = scale_and_pca(
            X_log, cfg, feature_names
        )
        if cfg.cache_dir is not None:
            _names = np.array(list(weights_by_group.keys()), dtype="U32")
            _values = np.array(list(weights_by_group.values()), dtype=np.float32)
            _atomic_savez(
                cfg.cache_dir / f"X_pca_{ckey}.npz",
                X_pca=X_pca,
                X_scaled=X_scaled,
                var_explained=np.float64(var_explained),
                n_components=np.int64(n_components),
                weights_group_names=_names,
                weights_group_values=_values,
            )

    assert X_pca is not None and X_scaled is not None

    return EmbeddingResult(
        kept=kept,
        X_pca=X_pca,
        feature_names=feature_names,
        X_scaled=X_scaled,
        var_explained=var_explained,
        n_components=n_components,
        weights_by_group=weights_by_group,
        pca_cache_key=ckey if cfg.cache_dir is not None else "",
    )


__all__ = [
    "EmbeddingResult",
    "RareCellConfig",
    "apply_group_weights",
    "build_delaunay_knn",  # alias, backward-compat
    "build_feature_matrix",
    "build_knn_adjacency",
    "compute_centroids",
    "compute_stability",
    "discover_rare_cell_types",
    "load_and_embed",
    "log_transform_copy",
    "log_transform_inplace",  # alias, backward-compat
    "morans_i_vectorized",
    "pre_filter_cells",
    "run_hdbscan",
    "scale_and_pca",
    "summarize_clusters",
    "ward_linkage_on_centroids",
]
