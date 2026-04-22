"""Manifold-spanning cell sampling for LMD pool generation.

Builds a hierarchical ``(manifold_group, organ) -> spatial_replicate`` grouping.
Pipeline: reuse rare-cell embedding (pre-filter + PCA) -> FPS anchors +
Voronoi (``manifold_group_id`` + ``d_to_anchor``) -> outlier flag (global pct
or per-group MAD) -> Ward-linkage spatial clustering per ``(group, organ)``
pair into replicates of ``target_area_um2`` -> :func:`select_lmd_replicates`
cap + rank for plate allocation.

Cache ``manifold_state_<hash>.npz`` (atomic write) stores anchors + labels +
distances + outlier mask. Hash folds feature_names, k_anchors, seed, outlier
policy, PCA params, excluded channels, and group weighting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.cluster import hierarchy as scipy_hier
from scipy.spatial import cKDTree

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.utils.json_utils import atomic_savez
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config + data classes
# ---------------------------------------------------------------------------


@dataclass
class ManifoldSamplingConfig:
    """Configuration for manifold-spanning sampling.

    Attributes:
        k_anchors: FPS anchors (Level 1 manifold groups).
        target_area_um2: Target cumulative cell area per Level 2 replicate.
        target_n_cells: Optional per-replicate count target (applied alongside
            area — whichever wants more replicates wins).
        outlier_method: ``"global_pct"`` flags top ``(100-threshold)%`` of all
            ``d_to_anchor``; ``"per_group_mad"`` flags per-Voronoi-cell
            ``d > median + threshold * MAD``.
        outlier_threshold: Percentile (global_pct) or MAD multiplier.
        cap_per_group: Max replicates per manifold group.
        priority: ``anchor_dist``, ``spatial_tight``, or ``composite``.
        include_partial: If False, drop pairs whose total area < target.
        min_spread_replicate_radii: Force ``n_rep=1`` when xy extent (max of
            x/y ptp) is below this many target-area radii. Default 4 =
            "two replicate diameters" so a 2-split only emits when the
            bounding box fits two non-overlapping disks of ``target_area_um2``.
        chunk_size: Voronoi chunk; ``None`` autos from free GPU mem.
        ward_chunk_size: Above this pair size Ward runs on kd-tree bins.
        seed: RNG seed.
        use_gpu: Try cupy; falls back to numpy.
        cache_dir: Write ``manifold_state_<hash>.npz`` here.
        feature_groups: Feature-matrix groups (maps to ``RareCellConfig``).
        feature_group_weights: ``"equal"`` (default, 1/√dim per group),
            ``"raw"``, or per-group dict. Default flipped to ``"equal"`` so
            SAM2's 256 dims don't drown morph's ~78.
        max_pcs: PCA cap (maps to ``RareCellConfig``).
        pca_variance: PCA variance target (maps to ``RareCellConfig``).
        exclude_channels: Channel ids excluded from the feature matrix.
        nuc_filter_nc_min/max, nuc_filter_min_overlap, area_filter_*_um2:
            Pre-filter thresholds (map to ``RareCellConfig``).
        organ_field: Per-cell field providing organ id (default ``organ_id``).
        organ_drop_value: Cells with this organ id are excluded from Level 2.
        organ_required: If True, cells lacking ``organ_field`` abort;
            if False, all-unassigned triggers single-tier fallback (one
            replicate per manifold group, ``organ_id`` recorded as
            ``organ_drop_value``).
    """

    k_anchors: int = 1000
    target_area_um2: float = 2500.0
    target_n_cells: int | None = None
    outlier_method: Literal["global_pct", "per_group_mad"] = "global_pct"
    outlier_threshold: float = 98.0
    cap_per_group: int = 5
    priority: Literal["anchor_dist", "spatial_tight", "composite"] = "anchor_dist"
    include_partial: bool = False
    min_spread_replicate_radii: float = 4.0
    chunk_size: int | None = None
    ward_chunk_size: int = 2000
    seed: int = 42
    use_gpu: bool = True
    cache_dir: Path | None = None
    # Embedding passthrough (mapped to RareCellConfig in discover_manifold_replicates).
    feature_groups: tuple[str, ...] = ("shape", "color", "sam2")
    feature_group_weights: str | dict[str, float] = "equal"
    max_pcs: int = 30
    pca_variance: float = 0.95
    exclude_channels: tuple[int, ...] = ()
    # Pre-filter passthrough.
    nuc_filter_nc_min: float = 0.02
    nuc_filter_nc_max: float = 0.95
    nuc_filter_min_overlap: float = 0.8
    area_filter_min_um2: float = 20.0
    area_filter_max_um2: float = 5000.0
    # Organ handling (Level-2 grouping).
    organ_field: str = "organ_id"
    organ_drop_value: int = 0
    organ_required: bool = False


@dataclass
class Replicate:
    """A single LMD replicate (contiguous cell pool for microdissection).

    Attributes:
        replicate_id: ``g{group:04d}_o{organ:03d}_r{idx:03d}``.
        manifold_group_id: Level-1 FPS anchor id.
        organ_id: Level-2 organ (``0`` = unassigned; skipped).
        within_pair_replicate_idx: 0-based index inside the pair.
        cell_uids: Per-cell UIDs for LMD XML export.
        cell_indices: Row indices into the kept-cell array.
        n_cells: Member count.
        total_area_um2: Summed cell area — the effective LMD cut area.
        mean_anchor_distance: Mean Level-1 d_to_anchor (lower = tighter morph).
        mean_xy_um: Centroid of member positions.
        xy_spread_um: ``ptp(x) + ptp(y)`` — half bounding-box perimeter
            (monotone proxy for spatial tightness; used by
            :func:`select_lmd_replicates` priorities ``spatial_tight`` and
            ``composite``).
        partial: ``True`` when area < target but include_partial allowed emit.
    """

    replicate_id: str
    manifold_group_id: int
    organ_id: int
    within_pair_replicate_idx: int
    cell_uids: list[str]
    cell_indices: list[int]
    n_cells: int
    total_area_um2: float
    mean_anchor_distance: float
    mean_xy_um: tuple[float, float]
    xy_spread_um: float
    partial: bool = False


# ---------------------------------------------------------------------------
# Level 1 — farthest-point sampling (FPS)
# ---------------------------------------------------------------------------


def fps_anchors(X: np.ndarray, k: int, *, seed: int = 42, use_gpu: bool = True) -> np.ndarray:
    """Farthest-point sampling in Euclidean distance. ``O(k*N)``.

    First anchor drawn via ``np.random.default_rng(seed).integers(0, N)`` on
    CPU (NOT ``cupy.random`` — cupy RNG diverges from numpy for same seed)
    for cross-device reproducibility.

    Args:
        X: ``(N, d)`` feature matrix (typically PCA scores).
        k: Number of anchors.
        seed: RNG seed for the first pick.
        use_gpu: Try cupy; falls back to numpy.

    Returns:
        1D int64 array of length ``k`` with anchor row indices into ``X``.

    Raises:
        ConfigError: If ``k <= 0`` or ``k > N``.
    """
    N = int(X.shape[0])
    if k <= 0:
        raise ConfigError(f"fps_anchors: k must be positive, got {k}")
    if k > N:
        raise ConfigError(f"fps_anchors: k={k} exceeds N={N}")

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, N))

    if use_gpu:
        try:
            import cupy as cp  # noqa: N811

            X_gpu = cp.asarray(X)
            picked: list[int] = [first]
            # Accumulate in float64 to match the CPU path exactly — prevents
            # float32 ties from resolving differently across devices.
            min_d2 = cp.sum((X_gpu - X_gpu[first]) ** 2, axis=1, dtype=cp.float64)
            for _ in range(1, k):
                nxt = int(cp.argmax(min_d2).get())
                picked.append(nxt)
                d2_new = cp.sum((X_gpu - X_gpu[nxt]) ** 2, axis=1, dtype=cp.float64)
                min_d2 = cp.minimum(min_d2, d2_new)
            logger.info("FPS (cupy GPU): picked %d anchors out of %d cells", k, N)
            return np.asarray(picked, dtype=np.int64)
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            logger.info("cupy FPS unavailable (%s), falling back to numpy", e)

    picked_np: list[int] = [first]
    min_d2_np = np.sum((X - X[first]) ** 2, axis=1).astype(np.float64)
    for _ in range(1, k):
        nxt = int(np.argmax(min_d2_np))
        picked_np.append(nxt)
        min_d2_np = np.minimum(min_d2_np, np.sum((X - X[nxt]) ** 2, axis=1))
    logger.info("FPS (numpy CPU): picked %d anchors out of %d cells", k, N)
    return np.asarray(picked_np, dtype=np.int64)


# ---------------------------------------------------------------------------
# Voronoi assignment
# ---------------------------------------------------------------------------


def _auto_chunk_size(n_centroids: int, dim: int, use_gpu: bool) -> int:
    """Pick a Voronoi chunk size from free device memory; 50k fallback."""
    fallback = 50_000
    if not use_gpu:
        return fallback
    try:
        import cupy as cp  # noqa: N811

        free_bytes, _ = cp.cuda.runtime.memGetInfo()
    except (ImportError, ModuleNotFoundError, RuntimeError):
        return fallback
    bytes_per_row = 4 * (n_centroids + dim)
    if bytes_per_row <= 0:
        return fallback
    budget = int(0.25 * free_bytes / bytes_per_row)
    return max(1024, min(budget, 500_000)) or fallback


def voronoi_assign(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    chunk: int | None = None,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row of ``X`` to the nearest centroid.

    Uses ``||x||² + ||c||² - 2·x·c`` (single matmul + two norms) — 2x less
    memory, 3-5x faster than naive broadcasted difference at large N/K.

    Args:
        X: ``(N, d)`` feature matrix.
        centroids: ``(K, d)`` centroid matrix.
        chunk: Rows per chunk. ``None`` autos from free GPU mem (50k fallback).
        use_gpu: Try cupy; falls back to numpy.

    Returns:
        ``(labels, d_to_anchor)`` — int32 length N and float32 Euclidean
        distance (not squared).

    Raises:
        ConfigError: If centroid and feature dims disagree.
    """
    if X.shape[1] != centroids.shape[1]:
        raise ConfigError(
            f"voronoi_assign: dim mismatch X.shape[1]={X.shape[1]} vs "
            f"centroids.shape[1]={centroids.shape[1]}"
        )
    N = int(X.shape[0])
    K = int(centroids.shape[0])
    if chunk is None:
        chunk = _auto_chunk_size(K, X.shape[1], use_gpu)

    labels = np.empty(N, dtype=np.int32)
    d_to_anchor = np.empty(N, dtype=np.float32)

    if use_gpu:
        try:
            import cupy as cp  # noqa: N811

            X_gpu = cp.asarray(X)
            C_gpu = cp.asarray(centroids)
            c_norm2 = cp.sum(C_gpu * C_gpu, axis=1)
            for i in range(0, N, chunk):
                j = min(i + chunk, N)
                x_chunk = X_gpu[i:j]
                x_norm2 = cp.sum(x_chunk * x_chunk, axis=1, keepdims=True)
                d2 = x_norm2 + c_norm2[None, :] - 2.0 * cp.matmul(x_chunk, C_gpu.T)
                d2 = cp.maximum(d2, 0.0)  # clamp tiny negatives from subtraction
                lbl = cp.argmin(d2, axis=1).astype(cp.int32)
                d_min = cp.take_along_axis(d2, lbl[:, None], axis=1).ravel()
                labels[i:j] = cp.asnumpy(lbl)
                d_to_anchor[i:j] = cp.asnumpy(cp.sqrt(d_min)).astype(np.float32)
            logger.info("Voronoi (cupy GPU): %d cells → %d centroids, chunk=%d", N, K, chunk)
            return labels, d_to_anchor
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            logger.info("cupy Voronoi unavailable (%s), falling back to numpy", e)

    c_norm2_np = np.sum(centroids * centroids, axis=1)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        x_chunk = X[i:j]
        x_norm2 = np.sum(x_chunk * x_chunk, axis=1, keepdims=True)
        d2 = x_norm2 + c_norm2_np[None, :] - 2.0 * x_chunk @ centroids.T
        np.maximum(d2, 0.0, out=d2)
        lbl = np.argmin(d2, axis=1).astype(np.int32)
        d_min = np.take_along_axis(d2, lbl[:, None], axis=1).ravel()
        labels[i:j] = lbl
        d_to_anchor[i:j] = np.sqrt(d_min).astype(np.float32)
    logger.info("Voronoi (numpy CPU): %d cells → %d centroids, chunk=%d", N, K, chunk)
    return labels, d_to_anchor


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------


def flag_outliers(
    labels: np.ndarray,
    d_to_anchor: np.ndarray,
    *,
    method: Literal["global_pct", "per_group_mad"] = "global_pct",
    threshold: float = 98.0,
) -> np.ndarray:
    """Flag cells whose distance to their Voronoi anchor is anomalously large.

    Args:
        labels: ``(N,)`` manifold_group_id per cell.
        d_to_anchor: ``(N,)`` Euclidean distance to assigned centroid.
        method: ``"global_pct"`` — flag top ``(100-threshold)%`` across all
            cells; ``"per_group_mad"`` — per Voronoi cell, flag ``d > median +
            threshold * MAD`` (``MAD = median(|x-median(x)|)``; no 1.4826
            scaling — subsumed into ``threshold``).
        threshold: Percentile (global_pct) or MAD multiplier (per_group_mad).

    Returns:
        ``(N,)`` boolean mask; True = outlier.

    Raises:
        ConfigError: If ``method`` is unknown or threshold non-finite.
    """
    if not np.isfinite(threshold):
        raise ConfigError(f"flag_outliers: threshold must be finite, got {threshold}")

    if method == "global_pct":
        if not (0.0 < threshold < 100.0):
            raise ConfigError(
                f"flag_outliers[global_pct]: threshold must be in (0, 100), got {threshold}"
            )
        cutoff = float(np.percentile(d_to_anchor, threshold))
        mask = d_to_anchor > cutoff
        logger.info(
            "Outliers (global_pct p%.1f): cutoff=%.4f → %d/%d flagged (%.2f%%)",
            threshold,
            cutoff,
            int(mask.sum()),
            mask.size,
            100.0 * mask.mean() if mask.size else 0.0,
        )
        return mask

    if method == "per_group_mad":
        mask = np.zeros_like(d_to_anchor, dtype=bool)
        unique_groups = np.unique(labels)
        for gid in unique_groups:
            in_g = labels == gid
            if not in_g.any():
                continue
            d_g = d_to_anchor[in_g]
            med = float(np.median(d_g))
            mad = float(np.median(np.abs(d_g - med)))
            mask[in_g] = d_g > (med + threshold * mad)
        logger.info(
            "Outliers (per_group_mad k=%.2f): %d/%d flagged (%.2f%%) across %d groups",
            threshold,
            int(mask.sum()),
            mask.size,
            100.0 * mask.mean() if mask.size else 0.0,
            unique_groups.size,
        )
        return mask

    raise ConfigError(
        f"flag_outliers: method must be 'global_pct' or 'per_group_mad', got {method!r}"
    )


# ---------------------------------------------------------------------------
# Level 2 — spatial replicates (Ward-linkage clustering per pair)
# ---------------------------------------------------------------------------


def _chunked_ward_cluster(xy: np.ndarray, n_rep: int, chunk_size: int = 2000) -> np.ndarray:
    """Ward linkage + fcluster on 2D positions, chunked for large inputs.

    Ward memory is ``O(n²)`` so for ``n > chunk_size`` we tile xy via a
    kd-tree grid into ~``chunk_size`` bins and run Ward per bin. Labels are
    stitched with global offsets; each bin gets a proportional share of
    ``n_rep`` clusters (at least 1). Deterministic (no RNG).

    Args:
        xy: ``(N, 2)`` positions (µm).
        n_rep: Target cluster count globally.
        chunk_size: Max cells per Ward bin.

    Returns:
        ``(N,)`` int32 labels (0-based, contiguous).
    """
    N = int(xy.shape[0])
    if N == 0:
        return np.zeros(0, dtype=np.int32)
    if n_rep <= 1 or N < 2:
        return np.zeros(N, dtype=np.int32)

    if N <= chunk_size:
        Z = scipy_hier.linkage(xy, method="ward")
        raw = scipy_hier.fcluster(Z, t=n_rep, criterion="maxclust")
        _, inverse = np.unique(raw, return_inverse=True)
        return inverse.astype(np.int32)

    # Tile xy into a regular grid of seed bins; each cell joins its nearest
    # seed. Deterministic (no RNG) and keeps Ward memory bounded.
    n_bins = int(np.ceil(N / chunk_size))
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    n_side = max(1, int(np.ceil(np.sqrt(n_bins))))
    xs = np.linspace(x_min, x_max, n_side + 2)[1:-1]
    ys = np.linspace(y_min, y_max, n_side + 2)[1:-1]
    if xs.size == 0 or ys.size == 0:
        seed_pts = np.array([[(x_min + x_max) / 2, (y_min + y_max) / 2]], dtype=xy.dtype)
    else:
        seed_pts = np.array([(x, y) for x in xs for y in ys], dtype=xy.dtype)

    seed_tree = cKDTree(seed_pts)
    _, taken = seed_tree.query(xy, k=1)
    taken = taken.astype(np.int32)

    out = np.zeros(N, dtype=np.int32)
    offset = 0
    for bid in np.unique(taken):
        mask = taken == bid
        m = int(mask.sum())
        if m == 0:
            continue
        share = max(1, int(np.ceil(n_rep * m / N)))
        share = min(share, m)
        if share == 1 or m < 2:
            sub = np.zeros(m, dtype=np.int32)
        else:
            Z = scipy_hier.linkage(xy[mask], method="ward")
            raw = scipy_hier.fcluster(Z, t=share, criterion="maxclust")
            _, sub = np.unique(raw, return_inverse=True)
            sub = sub.astype(np.int32)
        out[mask] = sub + offset
        offset += (int(sub.max()) + 1) if sub.size else 0
    _, inv = np.unique(out, return_inverse=True)
    return inv.astype(np.int32)


def spatial_replicates(
    positions_um: np.ndarray,
    labels: np.ndarray,
    organ_ids: np.ndarray,
    areas_um2: np.ndarray,
    cell_uids: list[str],
    outlier_mask: np.ndarray,
    d_to_anchor: np.ndarray,
    cfg: ManifoldSamplingConfig,
) -> list[Replicate]:
    """Level-2 spatial splitter.

    For each ``(manifold_group_id, organ_id)`` pair: skip if all outliers or
    ``organ_id==0``; compute ``n_rep = max(1, round(total_area/target))``;
    drop + warn if total area < target and ``include_partial=False``; spread
    guard forces ``n_rep=1`` when ``max(ptp)`` < ``min_spread_replicate_radii
    * sqrt(target/π)``; run :func:`_chunked_ward_cluster`; emit one
    :class:`Replicate` per spatial label (under-target dropped unless
    ``include_partial`` — then tagged ``partial=True``).

    Args:
        positions_um: ``(N, 2)`` xy (µm).
        labels: ``(N,)`` manifold_group_id.
        organ_ids: ``(N,)`` integer organ (``0`` = unassigned).
        areas_um2: ``(N,)`` per-cell area.
        cell_uids: ``(N,)`` UIDs.
        outlier_mask: ``(N,)`` boolean — True excludes the cell.
        d_to_anchor: ``(N,)`` distance (for ``mean_anchor_distance``).
        cfg: :class:`ManifoldSamplingConfig`.

    Returns:
        List of :class:`Replicate`.

    Raises:
        ConfigError: If input arrays disagree in length or shape.
    """
    N = positions_um.shape[0]
    if not (
        labels.shape[0] == N
        and organ_ids.shape[0] == N
        and areas_um2.shape[0] == N
        and outlier_mask.shape[0] == N
        and d_to_anchor.shape[0] == N
        and len(cell_uids) == N
    ):
        raise ConfigError("spatial_replicates: input arrays disagree in length")
    if positions_um.ndim != 2 or positions_um.shape[1] != 2:
        raise ConfigError(
            f"spatial_replicates: positions_um must be (N, 2), got {positions_um.shape}"
        )

    drop_value = int(cfg.organ_drop_value)
    eligible_global = (~outlier_mask) & (organ_ids != drop_value)
    replicates: list[Replicate] = []
    target_radius = float(np.sqrt(cfg.target_area_um2 / np.pi))
    spread_cutoff = cfg.min_spread_replicate_radii * target_radius

    if not eligible_global.any():
        if cfg.organ_required:
            raise ConfigError(
                "spatial_replicates: no cells have an assigned organ_id — "
                "set organ_required=False to fall back to single-tier "
                "(one replicate per manifold_group)."
            )
        logger.warning(
            "spatial_replicates: all cells have organ_id=%d — falling back to "
            "single-tier mode (one replicate per manifold_group; organ_id "
            "recorded as %d).",
            drop_value,
            drop_value,
        )
        eligible_global = ~outlier_mask
        organ_ids_use = np.full(N, drop_value, dtype=np.int32)
    else:
        organ_ids_use = organ_ids.astype(np.int32, copy=False)

    # Pack (gid, oid) into a single hash for one-shot unique grouping.
    # max_oid is strict upper bound: organ ids 0..max_oid-1 fit in one slot.
    max_oid = int(organ_ids_use.max()) + 1 if organ_ids_use.size else 1
    pair_keys = labels.astype(np.int64) * max_oid + organ_ids_use.astype(np.int64)

    for pair_hash in np.unique(pair_keys[eligible_global]):
        pair_mask = (pair_keys == pair_hash) & eligible_global
        idx_arr = np.flatnonzero(pair_mask)
        if idx_arr.size == 0:
            continue
        gid = int(labels[idx_arr[0]])
        oid = int(organ_ids_use[idx_arr[0]])
        pair_area = float(areas_um2[idx_arr].sum())
        n_rep = max(1, int(round(pair_area / cfg.target_area_um2)))

        if pair_area < cfg.target_area_um2:
            if not cfg.include_partial:
                logger.warning(
                    "Pair (g=%d, o=%d): n_cells=%d total_area=%.0f µm² < target "
                    "%.0f µm² — dropped (enable include_partial to retain).",
                    gid,
                    oid,
                    idx_arr.size,
                    pair_area,
                    cfg.target_area_um2,
                )
                continue
            n_rep = 1

        xy = positions_um[idx_arr]
        spread = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])))
        if spread < spread_cutoff:
            n_rep = 1
        elif cfg.target_n_cells is not None:
            n_by_count = max(1, int(round(idx_arr.size / cfg.target_n_cells)))
            n_rep = max(n_rep, n_by_count)

        sub_labels = _chunked_ward_cluster(xy, n_rep=n_rep, chunk_size=cfg.ward_chunk_size)

        for k in range((int(sub_labels.max()) + 1) if sub_labels.size else 0):
            member_mask = sub_labels == k
            if not member_mask.any():
                continue
            member_idx = idx_arr[member_mask]
            m_area = float(areas_um2[member_idx].sum())
            partial = m_area < cfg.target_area_um2
            if partial and not cfg.include_partial:
                continue
            mean_xy = (
                float(positions_um[member_idx, 0].mean()),
                float(positions_um[member_idx, 1].mean()),
            )
            xy_spread = float(
                np.ptp(positions_um[member_idx, 0]) + np.ptp(positions_um[member_idx, 1])
            )
            replicates.append(
                Replicate(
                    replicate_id=f"g{gid:04d}_o{oid:03d}_r{k:03d}",
                    manifold_group_id=gid,
                    organ_id=oid,
                    within_pair_replicate_idx=int(k),
                    cell_uids=[cell_uids[i] for i in member_idx.tolist()],
                    cell_indices=member_idx.tolist(),
                    n_cells=int(member_idx.size),
                    total_area_um2=round(m_area, 2),
                    mean_anchor_distance=round(float(d_to_anchor[member_idx].mean()), 4),
                    mean_xy_um=(round(mean_xy[0], 1), round(mean_xy[1], 1)),
                    xy_spread_um=round(xy_spread, 1),
                    partial=bool(partial),
                )
            )

    logger.info("Spatial replicates: emitted %d total", len(replicates))
    return replicates


# ---------------------------------------------------------------------------
# Replicate ranking + selection
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score with zero-variance guard (returns zeros if std < 1e-12)."""
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def select_lmd_replicates(
    replicates: list[Replicate],
    *,
    cap_per_group: int = 5,
    priority: Literal["anchor_dist", "spatial_tight", "composite"] = "anchor_dist",
    plate_budget: int | None = None,
) -> list[Replicate]:
    """Cap and rank replicates for LMD plate allocation.

    Priority (ascending, smallest = best): ``anchor_dist`` = tightest morph
    match (mean_anchor_distance); ``spatial_tight`` = most compact
    neighborhood (xy_spread_um); ``composite`` = ``z(anchor_dist) + 0.5 *
    z(xy_spread_um)``.

    Args:
        replicates: Candidate replicates.
        cap_per_group: Max replicates per ``manifold_group_id``.
        priority: Ranking metric.
        plate_budget: If set, trim output to N replicates total (after cap).

    Returns:
        Ranked + capped list.

    Raises:
        ConfigError: If ``priority`` is unknown.
    """
    if not replicates:
        return []

    anchor = np.array([r.mean_anchor_distance for r in replicates], dtype=np.float64)
    spread = np.array([r.xy_spread_um for r in replicates], dtype=np.float64)

    if priority == "anchor_dist":
        score = anchor.copy()
    elif priority == "spatial_tight":
        score = spread.copy()
    elif priority == "composite":
        score = _zscore(anchor) + 0.5 * _zscore(spread)
    else:
        raise ConfigError(
            "select_lmd_replicates: priority must be 'anchor_dist', "
            f"'spatial_tight', or 'composite', got {priority!r}"
        )

    order = np.argsort(score, kind="stable")
    per_group_emitted: dict[int, int] = {}
    kept_idx: list[int] = []
    for i in order:
        gid = replicates[int(i)].manifold_group_id
        if per_group_emitted.get(gid, 0) >= cap_per_group:
            continue
        kept_idx.append(int(i))
        per_group_emitted[gid] = per_group_emitted.get(gid, 0) + 1

    if plate_budget is not None and len(kept_idx) > plate_budget:
        kept_idx = kept_idx[:plate_budget]

    logger.info(
        "select_lmd_replicates: %d → %d replicates (priority=%s, cap_per_group=%d, budget=%s)",
        len(replicates),
        len(kept_idx),
        priority,
        cap_per_group,
        plate_budget,
    )
    return [replicates[i] for i in kept_idx]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(
    feature_names: list[str],
    n_cells: int,
    cfg: ManifoldSamplingConfig,
    extra: dict | None = None,
) -> str:
    """Hash identifying a unique manifold-sampling configuration.

    Folds every input that changes ``picked_idx`` / ``labels`` /
    ``d_to_anchor`` / ``outlier_mask``: feature names, ``k_anchors``, seed,
    outlier policy, plus ``extra`` (PCA variance, max_pcs, excluded channels,
    group weighting) from the embedding stage.
    """
    m = hashlib.sha256()
    m.update(str(sorted(feature_names)).encode())
    m.update(f"n={n_cells}".encode())
    m.update(f"k={cfg.k_anchors},seed={cfg.seed}".encode())
    m.update(f"outlier={cfg.outlier_method},{cfg.outlier_threshold}".encode())
    if extra:
        for key in sorted(extra):
            m.update(f"{key}={extra[key]!r}".encode())
    return m.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def discover_manifold_replicates(
    detections: list[dict],
    cfg: ManifoldSamplingConfig | None = None,
) -> dict[str, Any]:
    """End-to-end manifold sampling + replicate building.

    Mirrors :func:`xldvp_seg.analysis.rare_cell_discovery.discover_rare_cell_types`.
    Delegates embedding (pre-filter + PCA) to
    ``rare_cell_discovery.load_and_embed`` if present; else the explicit
    ``pre_filter_cells + build_feature_matrix + log_transform_copy +
    scale_and_pca`` chain. Kept detections are mutated in place with a
    ``manifold_group_id`` field.

    Args:
        detections: Raw detection dicts.
        cfg: :class:`ManifoldSamplingConfig` (defaults if ``None``).

    Returns:
        Dict with ``replicates``, ``picked_idx``, ``labels``, ``d_to_anchor``,
        ``outlier_mask``, ``kept_detections``, ``feature_names``,
        ``pca_variance``, ``pca_n_components``, ``stats``.

    Raises:
        ConfigError: If no cells survive embedding or k_anchors > N_kept.
    """
    cfg = cfg or ManifoldSamplingConfig()

    from xldvp_seg.analysis import rare_cell_discovery as rcd

    # Embedding config is now built from the full passthrough on
    # ManifoldSamplingConfig so CLI flags (feature groups, weights, PCA,
    # pre-filter thresholds) reach the rare-cell embedding layer — rather
    # than being silently overridden with hardcoded defaults.
    rcd_cfg = rcd.RareCellConfig(
        feature_groups=tuple(cfg.feature_groups),
        feature_group_weights=cfg.feature_group_weights,
        max_pcs=cfg.max_pcs,
        pca_variance=cfg.pca_variance,
        exclude_channels=tuple(cfg.exclude_channels),
        nuc_filter_nc_min=cfg.nuc_filter_nc_min,
        nuc_filter_nc_max=cfg.nuc_filter_nc_max,
        nuc_filter_min_overlap=cfg.nuc_filter_min_overlap,
        area_filter_min_um2=cfg.area_filter_min_um2,
        area_filter_max_um2=cfg.area_filter_max_um2,
        use_gpu=cfg.use_gpu,
        seed=cfg.seed,
        cache_dir=cfg.cache_dir,
    )

    embed_fn = getattr(rcd, "load_and_embed", None)
    pca_cache_key = ""
    if embed_fn is not None:
        result = embed_fn(detections, rcd_cfg)
        kept = result.kept
        X_pca = result.X_pca
        feature_names = result.feature_names
        var_explained = float(result.var_explained)
        n_components = int(result.n_components)
        weights_by_group = getattr(result, "weights_by_group", {})
        pca_cache_key = getattr(result, "pca_cache_key", "")
        logger.info("Embedding via rare_cell_discovery.load_and_embed()")
    else:
        logger.info("rare_cell_discovery.load_and_embed() not found; inline embedding")
        kept, pf_stats, _ = rcd.pre_filter_cells(detections, rcd_cfg)
        logger.info("Pre-filter kept %d/%d cells", pf_stats["kept"], pf_stats["input"])
        X_raw, feature_names, valid_indices = rcd.build_feature_matrix(
            kept, rcd_cfg.feature_groups, rcd_cfg.exclude_channels
        )
        if len(valid_indices) < len(kept):
            kept = [kept[i] for i in valid_indices]
        X_log = rcd.log_transform_copy(X_raw, feature_names)
        X_pca, var_explained, n_components, _X_scaled, weights_by_group = rcd.scale_and_pca(
            X_log, rcd_cfg, feature_names
        )

    if X_pca.shape[0] == 0:
        raise ConfigError(
            "discover_manifold_replicates: no cells survived embedding — "
            "check pre-filter thresholds / feature availability."
        )
    if cfg.k_anchors > X_pca.shape[0]:
        raise ConfigError(
            f"discover_manifold_replicates: k_anchors={cfg.k_anchors} exceeds "
            f"kept cell count {X_pca.shape[0]}"
        )

    # Cache key must fold every input that affects picked_idx / labels /
    # d_to_anchor / outlier_mask — including the pre-filter thresholds, since
    # two different filter settings can leave the same N cells alive and
    # would otherwise silently reuse a stale cache.
    extra = {
        "pca_variance": rcd_cfg.pca_variance,
        "max_pcs": rcd_cfg.max_pcs,
        "exclude_channels": sorted(rcd_cfg.exclude_channels),
        "feature_group_weights": rcd_cfg.feature_group_weights,
        "feature_groups": sorted(rcd_cfg.feature_groups),
        "nuc_filter": (
            rcd_cfg.nuc_filter_nc_min,
            rcd_cfg.nuc_filter_nc_max,
            rcd_cfg.nuc_filter_min_overlap,
        ),
        "area_filter": (
            rcd_cfg.area_filter_min_um2,
            rcd_cfg.area_filter_max_um2,
        ),
        "organ_field": cfg.organ_field,
        "organ_drop_value": cfg.organ_drop_value,
    }
    ckey = _cache_key(feature_names, len(kept), cfg, extra=extra)
    cache_path = cfg.cache_dir / f"manifold_state_{ckey}.npz" if cfg.cache_dir is not None else None
    picked_idx: np.ndarray | None = None
    labels: np.ndarray | None = None
    d_to_anchor: np.ndarray | None = None
    outlier_mask: np.ndarray | None = None
    if cache_path is not None and cache_path.exists():
        try:
            data = np.load(cache_path)
            picked_idx = data["picked_idx"]
            labels = data["labels"]
            d_to_anchor = data["d_to_anchor"]
            outlier_mask = data["outlier_mask"]
            logger.info("Manifold cache hit: %s", cache_path.name)
        except (OSError, ValueError, KeyError) as e:
            logger.warning("Manifold cache load failed (%s) — recomputing", e)
            picked_idx = None

    if picked_idx is None:
        picked_idx = fps_anchors(X_pca, cfg.k_anchors, seed=cfg.seed, use_gpu=cfg.use_gpu)
        centroids = X_pca[picked_idx]
        labels, d_to_anchor = voronoi_assign(
            X_pca, centroids, chunk=cfg.chunk_size, use_gpu=cfg.use_gpu
        )
        outlier_mask = flag_outliers(
            labels,
            d_to_anchor,
            method=cfg.outlier_method,
            threshold=cfg.outlier_threshold,
        )
        if cache_path is not None:
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            atomic_savez(
                cache_path,
                picked_idx=picked_idx,
                labels=labels,
                d_to_anchor=d_to_anchor,
                outlier_mask=outlier_mask,
                # Cross-link to the matching X_pca_<hash>.npz so the viewer
                # can load the paired PCA cache without guessing.
                pca_cache_key=np.array(pca_cache_key, dtype="U32"),
            )

    assert labels is not None and d_to_anchor is not None and outlier_mask is not None

    for det, lbl in zip(kept, labels):
        det["manifold_group_id"] = int(lbl)

    positions_um = np.array([d.get("global_center_um", [0.0, 0.0]) for d in kept], dtype=np.float32)
    areas_um2 = np.array(
        [float(d.get("features", {}).get("area_um2", 0.0)) for d in kept], dtype=np.float32
    )
    organ_ids = np.array(
        [int(d.get(cfg.organ_field, cfg.organ_drop_value)) for d in kept], dtype=np.int32
    )
    cell_uids = [str(d.get("uid", f"cell_{i}")) for i, d in enumerate(kept)]

    replicates = spatial_replicates(
        positions_um=positions_um,
        labels=labels,
        organ_ids=organ_ids,
        areas_um2=areas_um2,
        cell_uids=cell_uids,
        outlier_mask=outlier_mask,
        d_to_anchor=d_to_anchor,
        cfg=cfg,
    )

    drop_value = int(cfg.organ_drop_value)
    eligible = (~outlier_mask) & (organ_ids != drop_value)
    stats: dict[str, Any] = {
        "n_kept_cells": int(len(kept)),
        "n_anchors": int(cfg.k_anchors),
        "n_replicates": len(replicates),
        "n_outliers": int(outlier_mask.sum()),
        "outlier_fraction": float(outlier_mask.mean()) if outlier_mask.size else 0.0,
        "n_eligible": int(eligible.sum()),
        "n_unassigned_organ": int((organ_ids == drop_value).sum()),
        "organ_field": cfg.organ_field,
        "organ_drop_value": drop_value,
        "pca_variance": var_explained,
        "pca_n_components": n_components,
        "feature_group_weights": weights_by_group,
        "outlier_method": cfg.outlier_method,
        "outlier_threshold": cfg.outlier_threshold,
        "target_area_um2": cfg.target_area_um2,
        "include_partial": cfg.include_partial,
    }

    return {
        "replicates": replicates,
        "picked_idx": picked_idx,
        "labels": labels,
        "d_to_anchor": d_to_anchor,
        "outlier_mask": outlier_mask,
        "kept_detections": kept,
        "feature_names": feature_names,
        "pca_variance": var_explained,
        "pca_n_components": n_components,
        "stats": stats,
    }


__all__ = [
    "ManifoldSamplingConfig",
    "Replicate",
    "discover_manifold_replicates",
    "flag_outliers",
    "fps_anchors",
    "select_lmd_replicates",
    "spatial_replicates",
    "voronoi_assign",
]
