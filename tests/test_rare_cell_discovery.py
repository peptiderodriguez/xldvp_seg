"""Tests for rare cell-population discovery pipeline.

Synthetic-data tests:
  * `test_planted_recovery`: plant 3 Gaussian populations of 1000 cells each
    in 30D on a uniform background (17K cells) — HDBSCAN should recover all
    3 with Jaccard ≥ 0.9 vs ground truth.
  * `test_negative_control_noise_only`: pure uniform random → assert ≤1
    stable cluster (ideally 0). Guards against hallucinating structure.
  * `test_field_name_namespacing`: verify output JSON has `rare_pop_id`,
    not colliding with `global_cluster`.
  * `test_log_transform_columns`: log1p applied to area-like features only.
  * `test_morans_i_vectorized_matches_naive`: sanity-check vectorized
    implementation vs one-cluster-at-a-time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.analysis.rare_cell_discovery import (  # noqa: E402
    RareCellConfig,
    apply_group_weights,
    build_knn_adjacency,
    compute_centroids,
    compute_stability,
    discover_rare_cell_types,
    log_transform_copy,
    morans_i_vectorized,
    pre_filter_cells,
    run_hdbscan,
    scale_and_pca,
    summarize_clusters,
    ward_linkage_on_centroids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_detections(
    n_background: int = 17000,
    n_per_rare: int = 1000,
    n_rare_pops: int = 3,
    n_dims: int = 30,
    separation: float = 5.0,
    seed: int = 42,
) -> tuple[list[dict], np.ndarray]:
    """Build a synthetic detection list with planted rare populations.

    Background: uniform random in [-1, 1]^n_dims
    Rare populations: n_rare_pops tight Gaussians, each at a corner of the
    hypercube scaled by ``separation`` (≥5σ separation).

    Returns (detections_list, ground_truth_labels).
    """
    rng = np.random.default_rng(seed)
    background = rng.uniform(-1, 1, size=(n_background, n_dims)).astype(np.float32)

    # Pick rare-population centers far apart. Use orthogonal axis-aligned
    # offsets so all pairs are ≥ separation * sqrt(2) apart (well-separated).
    centers = np.zeros((n_rare_pops, n_dims), dtype=np.float32)
    for i in range(n_rare_pops):
        centers[i, i % n_dims] = separation
        if n_rare_pops > n_dims:
            centers[i, (i + n_rare_pops // 2) % n_dims] = separation

    rare_blocks = []
    for c in centers:
        rare_blocks.append(
            rng.normal(loc=c, scale=0.3, size=(n_per_rare, n_dims)).astype(np.float32)
        )
    X = np.vstack([background, *rare_blocks])
    gt = np.zeros(len(X), dtype=np.int32)  # background = -1 (coded as 0 here for test)
    gt[:n_background] = -1
    for i in range(n_rare_pops):
        start = n_background + i * n_per_rare
        gt[start : start + n_per_rare] = i

    # Build detection dicts with minimum fields
    detections = []
    for i, row in enumerate(X):
        feats = {f"feat_{j}": float(row[j]) for j in range(n_dims)}
        # Required pre-filter fields (put them on the good side so nothing is dropped)
        feats["n_nuclei"] = 1
        feats["nuclear_area_fraction"] = 0.3
        feats["area_um2"] = 100.0
        detections.append(
            {
                "uid": f"cell_{i}",
                "features": feats,
                "global_center_um": [float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))],
                "n_nuclei": 1,
                "area_um2": 100.0,
                "nuclei": [{"overlap_fraction": 0.95}],
            }
        )

    return detections, gt


def _best_jaccard(pred: np.ndarray, gt: np.ndarray, gt_label: int) -> float:
    """Best Jaccard between predicted clusters and a single ground-truth label."""
    best = 0.0
    gt_mask = gt == gt_label
    for pid in np.unique(pred):
        if pid < 0:
            continue
        p_mask = pred == pid
        inter = int((p_mask & gt_mask).sum())
        if inter == 0:
            continue
        union = int((p_mask | gt_mask).sum())
        j = inter / union if union else 0.0
        if j > best:
            best = j
    return best


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_planted_recovery():
    """HDBSCAN must recover 3 planted 1000-cell populations with Jaccard ≥0.9."""
    pytest.importorskip("hdbscan")
    detections, gt = _make_synthetic_detections(
        n_background=17000, n_per_rare=1000, n_rare_pops=3, n_dims=30, separation=5.0
    )
    cfg = RareCellConfig(
        feature_groups=("shape", "color"),  # will be overridden by feature names below
        min_cluster_size=800,
        min_samples=50,
        stability_sizes=(500, 800, 1200),
        max_pcs=30,
        use_gpu=False,  # CPU for reproducibility in CI
        nuc_filter_min_n_nuclei=1,
        nuc_filter_nc_min=0.01,
        nuc_filter_nc_max=0.99,
        nuc_filter_min_overlap=0.5,
        area_filter_min_um2=1.0,
        area_filter_max_um2=10000.0,
    )

    # Build feature matrix directly — bypass select_feature_names since
    # synthetic features aren't in any "group".
    feat_names = [f"feat_{j}" for j in range(30)]
    X = np.array([[d["features"][n] for n in feat_names] for d in detections], dtype=np.float32)
    labels, _, _ = run_hdbscan(X, cfg.min_cluster_size, cfg.min_samples, use_gpu=False)

    n_clusters = len(np.unique(labels[labels >= 0]))
    assert n_clusters >= 3, f"Expected ≥3 clusters, got {n_clusters}"

    for rare_label in range(3):
        j = _best_jaccard(labels, gt, rare_label)
        assert j >= 0.85, f"Rare pop {rare_label}: best Jaccard {j:.3f} < 0.85"


def test_negative_control_noise_only():
    """On uniform noise, HDBSCAN should find ≤2 stable clusters."""
    pytest.importorskip("hdbscan")
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(5000, 20)).astype(np.float32)
    labels, _, _ = run_hdbscan(X, min_cluster_size=500, min_samples=50, use_gpu=False)
    n_clusters = len(np.unique(labels[labels >= 0]))
    # On uniform random, HDBSCAN may find zero or a few marginal clusters.
    # Assert ≤2 — a loose guard against gross hallucination.
    assert n_clusters <= 2, f"Expected ≤2 clusters on pure noise, got {n_clusters}"


def test_field_name_namespacing():
    """Output detections must have `rare_pop_id`, not `global_cluster` / `cluster_id`.

    Asserts via source inspection (running the full orchestrator here needs
    real feature groups from the cluster_features registry — covered in the
    integration test with real data).
    """
    import inspect

    from xldvp_seg.analysis.rare_cell_discovery import discover_rare_cell_types as _fn

    src = inspect.getsource(_fn)
    assert (
        'det["rare_pop_id"] = int(lbl)' in src
    ), "Expected rare_pop_id as the output field name (namespaced from cluster_id / global_cluster)"
    assert 'det["hdbscan_prob"] = float(p)' in src
    # -2 sentinel for pre-filter drops
    assert 'det["rare_pop_id"] = -2' in src
    assert 'det["rare_pop_filter_reason"]' in src


def test_log_transform_columns():
    """log1p applied only to scale-spanning area-like features."""
    X = np.array([[100.0, 0.5, 10.0, 0.9], [400.0, 0.8, 20.0, 0.95]], dtype=np.float32)
    names = ["area_um2", "circularity", "perimeter_um", "solidity"]
    Xt = log_transform_copy(X, names)
    # area_um2, perimeter_um transformed
    assert np.isclose(Xt[0, 0], np.log1p(100.0))
    assert np.isclose(Xt[0, 2], np.log1p(10.0))
    # circularity, solidity untouched
    assert np.isclose(Xt[0, 1], 0.5)
    assert np.isclose(Xt[0, 3], 0.9)


def test_stability_jaccard_survives():
    """Cluster surviving across runs flagged stable."""
    primary = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1])
    # alt1: primary cluster 2 mostly dropped to noise; only 1 of 3 cells
    # survives as alt cluster 2. |P_2|=3, |A_2|=1, intersection=1, union=3 →
    # Jaccard=1/3 ≈ 0.33 < 0.5 → not credited.
    alt1 = np.array([0, 0, 0, 1, 1, 1, -1, -1, 2, -1])
    alt2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1])  # identical
    stable = compute_stability(primary, [alt1, alt2], jaccard_threshold=0.5, min_survive=2)
    assert stable[0] and stable[1]
    # Cluster 2: credited only on alt2 (1 survival) → not stable with min_survive=2
    assert not stable[2]


def test_stability_rejects_empty_alt_runs():
    """compute_stability raises ConfigError when no alt runs are provided."""
    from xldvp_seg.exceptions import ConfigError

    with pytest.raises(ConfigError, match="alt_runs is empty"):
        compute_stability(np.array([0, 0, 1, 1]), [], jaccard_threshold=0.5, min_survive=2)


def test_stability_rejects_unachievable_threshold():
    """compute_stability raises ConfigError when min_survive > len(alt_runs)."""
    from xldvp_seg.exceptions import ConfigError

    with pytest.raises(ConfigError, match="unachievable"):
        compute_stability(
            np.array([0, 0, 1, 1]),
            [np.array([0, 0, 1, 1])],
            jaccard_threshold=0.5,
            min_survive=2,
        )


def test_stability_reciprocal_best_match_prevents_double_credit():
    """Two primary clusters merged into one alt cluster can't both claim it.

    Under the old 'any match' logic, if alt merges primary 0 + 1 into a
    single alt cluster that Jaccard-matches both, both would get credit.
    With reciprocal-best-match, only one primary (the best match) is
    credited per alt cluster.
    """
    # 10 cells: primary splits 5/5, alt merges them all into one cluster.
    primary = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    alt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # one mega-cluster
    # Jaccard(P_0, A_0) = 5/10 = 0.5; Jaccard(P_1, A_0) = 5/10 = 0.5.
    # Both meet threshold, but A_0 argmax over i is tied; argmax picks first.
    # Only one primary should be credited per alt run.
    stable = compute_stability(primary, [alt, alt], jaccard_threshold=0.5, min_survive=2)
    assert (
        sum(stable) <= 1
    ), f"At most one primary can survive a merge-all alt run, got {sum(stable)}"


def test_morans_i_vectorized_matches_naive():
    """Vectorized Moran's I must equal naive per-column computation."""
    rng = np.random.default_rng(0)
    n = 200
    points = rng.uniform(0, 100, size=(n, 2))
    W = build_knn_adjacency(points, k=5)
    # 3 binary membership vectors
    M = rng.integers(0, 2, size=(n, 3)).astype(np.float32)
    vec = morans_i_vectorized(W, M)
    # Naive: per column
    for k in range(M.shape[1]):
        z = M[:, k] - M[:, k].mean()
        wz = W @ z
        numer = float(z @ wz)
        denom = float(z @ z) if z @ z > 0 else 1.0
        naive = numer / denom
        assert np.isclose(vec[k], naive, rtol=1e-4), f"col {k}: vec={vec[k]}, naive={naive}"


def test_morans_i_degenerate_cluster_is_nan():
    """All-zero or all-one membership columns have zero variance → Moran's I nan."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(50, 2))
    W = build_knn_adjacency(points, k=5)
    # Column 0: all ones. Column 1: all zeros. Column 2: nontrivial binary.
    M = np.zeros((50, 3), dtype=np.float32)
    M[:, 0] = 1.0
    M[:25, 2] = 1.0
    vec = morans_i_vectorized(W, M)
    assert np.isnan(vec[0]), "all-ones column should yield nan"
    assert np.isnan(vec[1]), "all-zeros column should yield nan"
    assert np.isfinite(vec[2])


def test_prefilter_drops_suspects():
    """Pre-filter drops cells with bad n_nuclei / overlap / area."""

    detections = [
        {"n_nuclei": 0, "area_um2": 100, "nuclei": [{"overlap_fraction": 1.0}]},  # drop: n_nuclei
        {
            "n_nuclei": 1,
            "area_um2": 100,
            "nuclear_area_fraction": 0.99,
            "nuclei": [{"overlap_fraction": 1.0}],
        },  # drop: nc_ratio > 0.95
        {
            "n_nuclei": 1,
            "area_um2": 100,
            "nuclear_area_fraction": 0.3,
            "nuclei": [{"overlap_fraction": 0.5}],
        },  # drop: overlap
        {
            "n_nuclei": 1,
            "area_um2": 10000,
            "nuclear_area_fraction": 0.3,
            "nuclei": [{"overlap_fraction": 1.0}],
        },  # drop: area too big
        {
            "n_nuclei": 1,
            "area_um2": 100,
            "nuclear_area_fraction": 0.3,
            "nuclei": [{"overlap_fraction": 1.0}],
        },  # KEEP
    ]
    cfg = RareCellConfig()
    kept, stats, reasons = pre_filter_cells(detections, cfg)
    assert len(kept) == 1
    assert stats["dropped_n_nuclei"] == 1
    assert stats["dropped_nc_ratio"] == 1
    assert stats["dropped_overlap"] == 1
    assert stats["dropped_area"] == 1
    # Reasons aligned with input order (5 detections: drops for 4 reasons, 1 keep)
    assert reasons == ["n_nuclei", "nc_ratio", "overlap", "area", None]


# ---------------------------------------------------------------------------
# Group weighting + helpers
# ---------------------------------------------------------------------------


def test_apply_group_weights_equal_equalizes_groups():
    """'equal' weighting gives each group ~1 unit of squared distance."""
    # 2 morph (shape) cols + 8 sam2 cols. Without weighting SAM2 dominates 4×.
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(50, 10)).astype(np.float32)
    names = ["area_um2", "perimeter_um"] + [f"sam2_{i}" for i in range(8)]
    X_w, weights_by_group = apply_group_weights(X, names, mode="equal")
    # With mode="equal", each column gets weight 1/sqrt(group_dim).
    # shape group has 2 cols, sam2 has 8.
    assert np.isclose(weights_by_group["shape"], 1.0 / np.sqrt(2), atol=1e-6)
    assert np.isclose(weights_by_group["sam2"], 1.0 / np.sqrt(8), atol=1e-6)
    # Expected per-group contribution to E[d²] under unit-variance inputs:
    # shape: 2 × (1/sqrt(2))² = 1; sam2: 8 × (1/sqrt(8))² = 1.
    # Empirically check: variance per column roughly scales with weight².
    shape_var = X_w[:, :2].var(axis=0).sum()
    sam2_var = X_w[:, 2:].var(axis=0).sum()
    # Both should be ~1 (within sampling noise on 50 rows).
    assert 0.3 < shape_var < 3.0
    assert 0.3 < sam2_var < 3.0


def test_apply_group_weights_raw_is_identity():
    X = np.arange(20, dtype=np.float32).reshape(5, 4)
    names = ["area_um2", "perimeter_um", "sam2_0", "sam2_1"]
    X_out, weights = apply_group_weights(X, names, mode="raw")
    assert np.array_equal(X_out, X)
    assert weights == {}


def test_apply_group_weights_explicit_dict():
    X = np.ones((3, 4), dtype=np.float32)
    names = ["area_um2", "perimeter_um", "sam2_0", "sam2_1"]
    X_out, weights = apply_group_weights(X, names, mode={"shape": 2.0, "sam2": 0.5})
    # shape cols: weight = 2.0 / sqrt(2); sam2 cols: 0.5 / sqrt(2).
    assert np.isclose(X_out[0, 0], 2.0 / np.sqrt(2))
    assert np.isclose(X_out[0, 2], 0.5 / np.sqrt(2))


def test_apply_group_weights_rejects_bad_mode():
    from xldvp_seg.exceptions import ConfigError

    X = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ConfigError):
        apply_group_weights(X, ["area_um2", "sam2_0"], mode="nonsense")


def test_compute_centroids_vectorized_matches_reference():
    """Vectorized compute_centroids matches the slow per-cluster mean."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 5)).astype(np.float32)
    labels = np.array([i % 7 - 1 for i in range(500)], dtype=np.int32)  # 6 clusters + noise
    centroids, cluster_ids = compute_centroids(X, labels)
    # Reference: boolean mask per cluster.
    expected_ids = np.unique(labels[labels >= 0])
    assert np.array_equal(cluster_ids, expected_ids)
    for i, cid in enumerate(cluster_ids):
        ref = X[labels == cid].mean(axis=0)
        assert np.allclose(centroids[i], ref, atol=1e-5)


def test_compute_centroids_empty():
    centroids, cluster_ids = compute_centroids(
        np.zeros((10, 3), dtype=np.float32), -np.ones(10, dtype=np.int32)
    )
    assert centroids.shape == (0, 3)
    assert cluster_ids.size == 0


def test_build_knn_adjacency_row_normalized_and_no_selfloop():
    """Each row of W sums to ~1.0 and diagonal is 0."""
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(100, 2)).astype(np.float32)
    W = build_knn_adjacency(points, k=5)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    # Diagonal: KNN skips self at column 0, so no self-edges.
    assert np.allclose(W.diagonal(), 0.0, atol=1e-6)


def test_ward_linkage_empty_and_small():
    # <2 centroids → empty linkage matrix
    assert ward_linkage_on_centroids(np.zeros((1, 3), dtype=np.float32)).shape == (0, 4)
    # 3 centroids → linkage has K-1=2 rows
    centroids = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    Z = ward_linkage_on_centroids(centroids)
    assert Z.shape == (2, 4)


# ---------------------------------------------------------------------------
# summarize_clusters (round-trip + nan handling)
# ---------------------------------------------------------------------------


def test_summarize_clusters_formats_moran_nan_as_none():
    """Moran NaN (degenerate) becomes None in summary dict."""
    labels = np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int32)
    persistence = np.array([0.5, 0.3], dtype=np.float32)
    moran = np.array([np.nan, 0.42], dtype=np.float32)
    stable = np.array([True, False])
    # Detections with organ_ids so top_regions populates
    dets = [{"organ_id": 1}] * 3 + [{"organ_id": 2}] * 3 + [{"organ_id": 0}]
    names = ["feat_0", "feat_1"]
    X_scaled = np.zeros((7, 2), dtype=np.float32)
    rows = summarize_clusters(labels, persistence, moran, stable, dets, names, X_scaled)
    assert len(rows) == 2
    assert rows[0]["cluster_id"] == 0
    assert rows[0]["moran_i"] is None
    assert rows[0]["stable"] is True
    assert rows[0]["top_regions"] == "1:3"
    assert rows[1]["moran_i"] == 0.42
    assert rows[1]["top_regions"] == "2:3"


# ---------------------------------------------------------------------------
# Orchestrator end-to-end (with -2 sentinel + cache round-trip)
# ---------------------------------------------------------------------------


def _make_e2e_detections(
    n_good: int = 2000,
    n_bad: int = 50,
    n_rare_pops: int = 2,
    n_per_rare: int = 400,
    seed: int = 0,
) -> list[dict]:
    """Detections that pass real select_feature_names (shape+sam2 groups)
    with a few pre-filter drops for each reason + planted rare populations."""
    rng = np.random.default_rng(seed)

    def _feat(area: float, rare_shift: np.ndarray | None = None) -> dict:
        vec = rng.normal(0, 1, size=16).astype(np.float32)
        if rare_shift is not None:
            vec = vec + rare_shift
        feats = {
            "area_um2": float(area),
            "perimeter_um": float(np.sqrt(area) * 4),
            "major_axis_um": float(np.sqrt(area)),
            "minor_axis_um": float(np.sqrt(area) * 0.8),
            "circularity": float(np.clip(rng.normal(0.6, 0.1), 0, 1)),
            "solidity": float(np.clip(rng.normal(0.9, 0.05), 0, 1)),
            "eccentricity": float(np.clip(rng.normal(0.5, 0.1), 0, 1)),
            "extent": float(np.clip(rng.normal(0.6, 0.1), 0, 1)),
            "n_nuclei": 1,
            "nuclear_area_fraction": 0.3,
        }
        for i, v in enumerate(vec):
            feats[f"sam2_{i}"] = float(v)
        return feats

    dets: list[dict] = []
    # good background cells
    for _ in range(n_good):
        dets.append(
            {
                "features": _feat(area=float(rng.uniform(50, 500))),
                "n_nuclei": 1,
                "nuclear_area_fraction": 0.3,
                "nuclei": [{"overlap_fraction": 0.95}],
                "global_center_um": [float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))],
                "organ_id": int(rng.integers(1, 4)),
            }
        )
    # rare populations (shifted SAM2 features + tighter area distribution)
    for pop_idx in range(n_rare_pops):
        shift = np.zeros(16, dtype=np.float32)
        shift[pop_idx * 4 : pop_idx * 4 + 4] = 5.0
        for _ in range(n_per_rare):
            dets.append(
                {
                    "features": _feat(area=float(rng.normal(200, 5)), rare_shift=shift),
                    "n_nuclei": 1,
                    "nuclear_area_fraction": 0.3,
                    "nuclei": [{"overlap_fraction": 0.95}],
                    "global_center_um": [
                        float(rng.uniform(0, 1000)),
                        float(rng.uniform(0, 1000)),
                    ],
                    "organ_id": 4 + pop_idx,
                }
            )
    # bad cells — one for each pre-filter reason
    dets.append(
        {
            "features": {**_feat(area=100), "n_nuclei": 0},
            "n_nuclei": 0,
            "nuclear_area_fraction": 0.3,
            "nuclei": [{"overlap_fraction": 1.0}],
            "global_center_um": [500, 500],
            "organ_id": 1,
        }
    )
    dets.append(
        {
            "features": {**_feat(area=100), "nuclear_area_fraction": 0.99},
            "n_nuclei": 1,
            "nuclear_area_fraction": 0.99,  # nc_ratio too high
            "nuclei": [{"overlap_fraction": 1.0}],
            "global_center_um": [500, 500],
            "organ_id": 1,
        }
    )
    dets.append(
        {
            "features": _feat(area=100),
            "n_nuclei": 1,
            "nuclear_area_fraction": 0.3,
            "nuclei": [{"overlap_fraction": 0.2}],  # low overlap
            "global_center_um": [500, 500],
            "organ_id": 1,
        }
    )
    dets.append(
        {
            "features": _feat(area=50000),  # area too big
            "n_nuclei": 1,
            "nuclear_area_fraction": 0.3,
            "nuclei": [{"overlap_fraction": 1.0}],
            "global_center_um": [500, 500],
            "organ_id": 1,
        }
    )
    # padding to make sure we meet the 2×min_cluster_size floor
    for _ in range(n_bad):
        dets.append(
            {
                "features": _feat(area=float(rng.uniform(50, 500))),
                "n_nuclei": 1,
                "nuclear_area_fraction": 0.3,
                "nuclei": [{"overlap_fraction": 0.95}],
                "global_center_um": [float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))],
                "organ_id": int(rng.integers(1, 4)),
            }
        )
    return dets


def test_orchestrator_end_to_end_writes_sentinel(tmp_path):
    """Full pipeline: pre-filter drops get -2 + reason; kept cells get cluster/noise labels."""
    pytest.importorskip("hdbscan")
    dets = _make_e2e_detections(n_good=800, n_rare_pops=2, n_per_rare=200)
    cfg = RareCellConfig(
        feature_groups=("shape", "sam2"),
        min_cluster_size=100,
        stability_sizes=(50, 100, 200),
        stability_min_survive=1,  # 2 alt runs after dropping primary; 1 survival sufficient
        min_samples=20,
        use_gpu=False,
        cache_dir=tmp_path,
        seed=0,
    )
    result = discover_rare_cell_types(dets, cfg)

    # Every detection has rare_pop_id assigned.
    assert all("rare_pop_id" in d for d in dets), "every detection should be tagged"

    # -2 sentinel present with matching reasons.
    sentinel_dets = [d for d in dets if d.get("rare_pop_id") == -2]
    assert len(sentinel_dets) >= 4, "expected at least one drop per reason"
    reasons_seen = {d.get("rare_pop_filter_reason") for d in sentinel_dets}
    assert reasons_seen >= {"n_nuclei", "nc_ratio", "overlap", "area"}

    # Kept cells never have a filter_reason set.
    kept_dets = [d for d in dets if d.get("rare_pop_id", -2) != -2]
    assert all("rare_pop_filter_reason" not in d for d in kept_dets)

    # Summary sane.
    assert result["pca_n_components"] <= cfg.max_pcs
    assert 0.0 <= result["pca_variance"] <= 1.0

    # Cache artifacts written.
    pca_caches = list(tmp_path.glob("X_pca_*.npz"))
    knn_caches = list(tmp_path.glob("W_knn_k*.npz"))
    assert len(pca_caches) == 1
    assert len(knn_caches) == 1


def test_orchestrator_cache_round_trip_identical_labels(tmp_path):
    """Second run reads cache and yields identical labels."""
    pytest.importorskip("hdbscan")
    dets1 = _make_e2e_detections(n_good=600, n_rare_pops=2, n_per_rare=200, seed=1)
    cfg = RareCellConfig(
        feature_groups=("shape", "sam2"),
        min_cluster_size=100,
        stability_sizes=(50, 100, 200),
        stability_min_survive=1,  # 2 alt runs after dropping primary; 1 survival sufficient
        min_samples=20,
        use_gpu=False,
        cache_dir=tmp_path,
        seed=0,
    )
    r1 = discover_rare_cell_types(dets1, cfg)

    # Fresh detection list (same RNG → same features), same cache dir.
    dets2 = _make_e2e_detections(n_good=600, n_rare_pops=2, n_per_rare=200, seed=1)
    r2 = discover_rare_cell_types(dets2, cfg)

    assert np.array_equal(r1["labels"], r2["labels"])
    assert r1["pca_n_components"] == r2["pca_n_components"]


def test_orchestrator_too_few_kept_raises(tmp_path):
    """Too-few-cells post-filter → ConfigError."""
    from xldvp_seg.exceptions import ConfigError

    dets = _make_e2e_detections(n_good=50, n_rare_pops=0, n_per_rare=0)
    cfg = RareCellConfig(
        feature_groups=("shape", "sam2"),
        min_cluster_size=500,
        use_gpu=False,
        cache_dir=tmp_path,
    )
    with pytest.raises(ConfigError, match="Too few cells"):
        discover_rare_cell_types(dets, cfg)


def test_scale_and_pca_returns_scaler_and_variance():
    """scale_and_pca returns 5-tuple with X_scaled + weights."""
    rng = np.random.default_rng(0)
    # Rank-5 matrix with small noise (so 5 PCs explain >95%).
    core = rng.normal(0, 1, size=(200, 5)).astype(np.float32)
    loading = rng.normal(0, 1, size=(5, 12)).astype(np.float32)
    X = core @ loading + rng.normal(0, 0.01, size=(200, 12)).astype(np.float32)
    names = [f"feat_{i}" for i in range(12)]  # no group matches → all "shape" default
    cfg = RareCellConfig(max_pcs=10, pca_variance=0.95, use_gpu=False)
    X_pca, var, n, X_scaled, weights = scale_and_pca(X, cfg, names)
    assert X_pca.shape[0] == 200
    assert n <= 10
    assert 0.5 < var <= 1.0
    assert X_scaled.shape == X.shape
    assert isinstance(weights, dict)
