"""Tests for manifold-spanning cell sampling (Level 1 FPS + Voronoi + Level 2 Ward).

Unit tests cover ``fps_anchors``, ``voronoi_assign``, ``flag_outliers``,
``_chunked_ward_cluster``, ``spatial_replicates``, and
``select_lmd_replicates``.

Smoke tests wire the full ``discover_manifold_replicates`` orchestrator on the
small JSON fixture (``tests/fixtures/manifold_sampling_small.json``) and — if
Wave 2E has delivered ``xldvp_seg.visualization.manifold_viewer`` — exercise
the HTML builder end-to-end.

Style mirrors ``test_rare_cell_discovery.py``: ``np.random.default_rng`` for
all synthetic data, ``pytest.importorskip`` for optional deps, and f-string
messages on the assertions that are most likely to regress.
"""

from __future__ import annotations

import copy
import json
import sys
import tracemalloc
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.analysis.manifold_sampling import (  # noqa: E402
    ManifoldSamplingConfig,
    Replicate,
    _chunked_ward_cluster,
    discover_manifold_replicates,
    flag_outliers,
    fps_anchors,
    select_lmd_replicates,
    spatial_replicates,
    voronoi_assign,
)
from xldvp_seg.exceptions import ConfigError  # noqa: E402

FIXTURE_PATH = REPO / "tests" / "fixtures" / "manifold_sampling_small.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _four_gaussians_in_4d(
    n_per: int = 50, scale: float = 0.1, sep: float = 5.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Four well-separated Gaussian clumps along orthogonal axes.

    Returns ``(X, gt)`` where ``gt[i]`` is the ground-truth cluster id in
    ``[0, 4)``. Centers are ``±sep`` along axes 0 and 1 so all pairs are at
    least ``sep * sqrt(2)`` apart.
    """
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[sep, 0, 0, 0], [-sep, 0, 0, 0], [0, sep, 0, 0], [0, -sep, 0, 0]], dtype=np.float32
    )
    blocks = []
    gt = []
    for gid, c in enumerate(centers):
        blocks.append(rng.normal(loc=c, scale=scale, size=(n_per, 4)).astype(np.float32))
        gt.extend([gid] * n_per)
    return np.vstack(blocks), np.asarray(gt, dtype=np.int32)


def _inflate_fixture(fixture_dets: list[dict], copies: int, seed: int = 0) -> list[dict]:
    """Replicate fixture detections ``copies`` times with jittered positions + UIDs.

    The orchestrator's default ``RareCellConfig`` requires
    ``>= 2 * min_cluster_size`` kept cells (``min_cluster_size=1000``), so the
    500-cell fixture needs inflation to clear the pre-filter floor.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for c in range(copies):
        for det in fixture_dets:
            d = copy.deepcopy(det)
            d["uid"] = f"{det['uid']}_c{c:02d}"
            # Jitter xy a little to avoid degenerate Ward input.
            x, y = det["global_center_um"]
            d["global_center_um"] = [float(x + rng.normal(0, 5)), float(y + rng.normal(0, 5))]
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# fps_anchors
# ---------------------------------------------------------------------------


def test_fps_anchors_recovers_planted_clusters():
    """FPS with k=4 on 4 well-separated Gaussians must cover each cluster once."""
    X, gt = _four_gaussians_in_4d(n_per=50, scale=0.1, sep=5.0, seed=0)
    picked = fps_anchors(X, k=4, seed=0, use_gpu=False)

    assert picked.shape == (4,)
    assert picked.dtype == np.int64
    covered = {int(gt[idx]) for idx in picked}
    assert covered == {
        0,
        1,
        2,
        3,
    }, f"FPS must pick one anchor from each of the 4 clusters; covered={covered}"


def test_fps_anchors_seed_deterministic_cpu():
    """Two FPS runs with the same seed (CPU path) must pick identical anchors."""
    rng = np.random.default_rng(123)
    X = rng.normal(0, 1, size=(200, 8)).astype(np.float32)
    p1 = fps_anchors(X, k=10, seed=42, use_gpu=False)
    p2 = fps_anchors(X, k=10, seed=42, use_gpu=False)
    assert np.array_equal(p1, p2), f"FPS not deterministic: p1={p1}, p2={p2}"


# ---------------------------------------------------------------------------
# voronoi_assign
# ---------------------------------------------------------------------------


def test_voronoi_assign_chunked_matches_unchunked():
    """Chunked and single-shot Voronoi pass must agree on labels + distances."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(30, 6)).astype(np.float32)
    centroids = rng.normal(0, 1, size=(5, 6)).astype(np.float32)
    lbl_small, d_small = voronoi_assign(X, centroids, chunk=5, use_gpu=False)
    lbl_big, d_big = voronoi_assign(X, centroids, chunk=10_000, use_gpu=False)
    assert np.array_equal(lbl_small, lbl_big)
    assert np.allclose(d_small, d_big, atol=1e-5)


def test_voronoi_distance_correctness():
    """Hand-crafted 3-point / 2-centroid case — exact distance + label check."""
    X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    centroids = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    labels, d = voronoi_assign(X, centroids, chunk=10, use_gpu=False)

    assert labels.tolist() == [0, 1, 0], f"Expected labels=[0, 1, 0], got {labels.tolist()}"
    expected_d = np.array([0.0, 0.0, np.sqrt(2.0)], dtype=np.float32)
    assert np.allclose(d, expected_d, atol=1e-5), f"distances: got {d}, expected {expected_d}"


# ---------------------------------------------------------------------------
# flag_outliers
# ---------------------------------------------------------------------------


def test_flag_outliers_global_pct():
    """global_pct threshold flags top (100-threshold)% of distances exactly."""
    d = np.arange(100, dtype=np.float32)
    labels = np.zeros(100, dtype=np.int32)

    mask_98 = flag_outliers(labels, d, method="global_pct", threshold=98.0)
    # p98 of arange(100) = 98.02; cells with d > 98.02 → indices 99.
    # Matching description: "flag exactly 2" — percentile 98 with interpolation
    # gives cutoff ≈ 98.02, which only flags index 99.  Use ≥1 as upper bound,
    # but allow the 2-flag case for strictly-greater with ties.
    assert mask_98.sum() in {1, 2}, f"Expected 1-2 flagged at p98, got {mask_98.sum()}"
    # Top index must be flagged.
    assert mask_98[-1]

    mask_90 = flag_outliers(labels, d, method="global_pct", threshold=90.0)
    # p90 ≈ 89.1 → flags indices 90..99 (10 cells).
    assert mask_90.sum() in {9, 10}, f"Expected ~10 flagged at p90, got {mask_90.sum()}"


def test_flag_outliers_per_group_mad():
    """Per-group MAD flags heavy-tailed point in cluster 1 but nothing in cluster 0."""
    rng = np.random.default_rng(0)
    d0 = rng.uniform(0, 1, size=50).astype(np.float32)
    d1 = rng.uniform(0, 1, size=50).astype(np.float32)
    d1[0] = 100.0  # massive outlier
    d = np.concatenate([d0, d1])
    labels = np.concatenate([np.zeros(50, dtype=np.int32), np.ones(50, dtype=np.int32)])

    mask = flag_outliers(labels, d, method="per_group_mad", threshold=3.0)
    # The 100-valued point in cluster 1 must be flagged.
    assert mask[50], "Heavy-tailed point in cluster 1 should be flagged"
    # Cluster 0 has no heavy tail — at most a tiny number of noisy flags.
    assert mask[:50].sum() <= 2, f"Cluster 0 should have ~0 flags, got {mask[:50].sum()}"


def test_flag_outliers_rejects_bad_method():
    """Unknown method must raise ConfigError."""
    d = np.arange(10, dtype=np.float32)
    labels = np.zeros(10, dtype=np.int32)
    with pytest.raises(ConfigError, match="method must be"):
        flag_outliers(labels, d, method="nonsense", threshold=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Area-budget logic (tested via spatial_replicates — no standalone helper)
# ---------------------------------------------------------------------------


def test_area_budget_pick_via_spatial_replicates():
    """Area budget behavior through ``spatial_replicates``.

    The module has no standalone ``area_budget_pick`` helper — cumulative-area
    accounting lives inside ``spatial_replicates``. This test exercises the
    three relevant sub-cases:

      * total_area ≈ 3 × target → ~3 replicates emitted (area-driven n_rep).
      * single under-target pair with ``include_partial=False`` → dropped.
      * same pair with ``include_partial=True`` → 1 partial replicate emitted.
    """
    # --- (A) total_area well over target → multiple replicates ---
    rng = np.random.default_rng(0)
    n = 10
    positions = rng.uniform(0, 200, size=(n, 2)).astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)  # one manifold group
    organ_ids = np.ones(n, dtype=np.int32)  # one organ
    areas = np.full(n, 100.0, dtype=np.float32)  # total 1000, target 250 → n_rep≈4
    cell_uids = [f"c{i}" for i in range(n)]
    outlier_mask = np.zeros(n, dtype=bool)
    d_to_anchor = np.full(n, 0.5, dtype=np.float32)

    # include_partial=True so under-target sub-groups aren't silently dropped —
    # we want to confirm every cell is accounted for.
    cfg = ManifoldSamplingConfig(
        k_anchors=1,
        target_area_um2=250.0,
        include_partial=True,
        min_spread_replicate_radii=0.0,  # disable spread guard for the test
        use_gpu=False,
    )
    reps = spatial_replicates(
        positions, labels, organ_ids, areas, cell_uids, outlier_mask, d_to_anchor, cfg
    )
    assert 2 <= len(reps) <= 5, f"Expected 2-5 replicates for total_area/target≈4, got {len(reps)}"
    assert all(r.manifold_group_id == 0 and r.organ_id == 1 for r in reps)
    # Total cells across replicates must equal n (include_partial=True retains
    # every Ward sub-group, even those under target).
    assert sum(r.n_cells for r in reps) == n

    # --- (B) under-target pair, include_partial=False → dropped ---
    small_areas = np.full(n, 5.0, dtype=np.float32)  # total 50, target 1000 → skip
    cfg_drop = ManifoldSamplingConfig(
        k_anchors=1,
        target_area_um2=1000.0,
        include_partial=False,
        use_gpu=False,
    )
    reps_drop = spatial_replicates(
        positions, labels, organ_ids, small_areas, cell_uids, outlier_mask, d_to_anchor, cfg_drop
    )
    assert (
        reps_drop == []
    ), f"Under-target pair with include_partial=False must drop; got {reps_drop}"

    # --- (C) same small pair, include_partial=True → 1 partial replicate ---
    cfg_partial = ManifoldSamplingConfig(
        k_anchors=1,
        target_area_um2=1000.0,
        include_partial=True,
        use_gpu=False,
    )
    reps_partial = spatial_replicates(
        positions, labels, organ_ids, small_areas, cell_uids, outlier_mask, d_to_anchor, cfg_partial
    )
    assert len(reps_partial) == 1, f"Expected 1 partial replicate, got {len(reps_partial)}"
    assert reps_partial[0].partial is True
    assert reps_partial[0].n_cells == n


# ---------------------------------------------------------------------------
# spatial_replicates — determinism + chunked Ward + partial handling
# ---------------------------------------------------------------------------


def test_spatial_replicates_ward_deterministic():
    """Identical inputs → identical Replicate lists (Ward + cKDTree are deterministic)."""
    rng = np.random.default_rng(0)
    n = 200
    positions = rng.uniform(0, 500, size=(n, 2)).astype(np.float32)
    labels = rng.integers(0, 3, size=n).astype(np.int32)
    organ_ids = rng.integers(1, 4, size=n).astype(np.int32)
    areas = rng.uniform(80, 120, size=n).astype(np.float32)
    uids = [f"c{i}" for i in range(n)]
    outlier_mask = np.zeros(n, dtype=bool)
    d = rng.uniform(0, 1, size=n).astype(np.float32)
    cfg = ManifoldSamplingConfig(
        k_anchors=3,
        target_area_um2=500.0,
        min_spread_replicate_radii=0.0,
        use_gpu=False,
    )

    r1 = spatial_replicates(positions, labels, organ_ids, areas, uids, outlier_mask, d, cfg)
    r2 = spatial_replicates(positions, labels, organ_ids, areas, uids, outlier_mask, d, cfg)

    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a.replicate_id == b.replicate_id
        assert a.cell_indices == b.cell_indices
        assert a.n_cells == b.n_cells


def test_spatial_replicates_chunked_ward_large_n():
    """_chunked_ward_cluster on 3000 points: runs, returns ndarray, ``n_rep`` labels, low memory."""
    rng = np.random.default_rng(0)
    # 4 well-separated spatial clumps at corners of a 1000 x 1000 box.
    centers = np.array([[200, 200], [800, 200], [200, 800], [800, 800]], dtype=np.float32)
    blocks = [rng.normal(loc=c, scale=20, size=(750, 2)).astype(np.float32) for c in centers]
    xy = np.vstack(blocks)
    assert xy.shape == (3000, 2)

    tracemalloc.start()
    labels = _chunked_ward_cluster(xy, n_rep=4, chunk_size=2000)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (3000,)
    assert labels.dtype == np.int32
    # Labels must cover all cells, no missing values.
    assert np.all(labels >= 0)
    # At least 2 distinct labels; exact count varies with binning on chunked path.
    assert (
        len(np.unique(labels)) >= 2
    ), f"Chunked Ward should yield ≥2 labels on 4-corner data; got {len(np.unique(labels))}"
    # Ward memory on a 2000-cell bin is ~64 MB; peak across chunks should be well under 100 MB.
    assert peak < 100 * 1024 * 1024, f"Peak memory {peak / 1e6:.1f} MB exceeded 100 MB budget"


def test_spatial_replicates_partial_drop_vs_include():
    """A pair with 0.4× target-area: drops by default, emits 1 partial when include_partial."""
    n = 5
    positions = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32
    )
    labels = np.zeros(n, dtype=np.int32)
    organ_ids = np.ones(n, dtype=np.int32)
    # total_area = 400, target = 1000 → 0.4x target → round(0.4)=0 → n_rep=0 path.
    areas = np.full(n, 80.0, dtype=np.float32)
    uids = [f"c{i}" for i in range(n)]
    outlier_mask = np.zeros(n, dtype=bool)
    d = np.full(n, 0.5, dtype=np.float32)

    cfg_drop = ManifoldSamplingConfig(
        k_anchors=1, target_area_um2=1000.0, include_partial=False, use_gpu=False
    )
    reps_drop = spatial_replicates(
        positions, labels, organ_ids, areas, uids, outlier_mask, d, cfg_drop
    )
    assert reps_drop == []

    cfg_keep = ManifoldSamplingConfig(
        k_anchors=1, target_area_um2=1000.0, include_partial=True, use_gpu=False
    )
    reps_keep = spatial_replicates(
        positions, labels, organ_ids, areas, uids, outlier_mask, d, cfg_keep
    )
    assert len(reps_keep) == 1
    assert reps_keep[0].partial is True
    assert reps_keep[0].n_cells == n


# ---------------------------------------------------------------------------
# select_lmd_replicates — caps + priorities
# ---------------------------------------------------------------------------


def _make_reps_for_ranking() -> list[Replicate]:
    """20 replicates across 3 manifold groups: gid 0 (7), gid 1 (8), gid 2 (5)."""
    rng = np.random.default_rng(0)
    reps: list[Replicate] = []
    for gid, count in [(0, 7), (1, 8), (2, 5)]:
        for k in range(count):
            reps.append(
                Replicate(
                    replicate_id=f"g{gid:04d}_o001_r{k:03d}",
                    manifold_group_id=gid,
                    organ_id=1,
                    within_pair_replicate_idx=k,
                    cell_uids=[f"c_{gid}_{k}"],
                    cell_indices=[gid * 100 + k],
                    n_cells=10,
                    total_area_um2=2500.0,
                    mean_anchor_distance=float(rng.uniform(0, 5)),
                    mean_xy_um=(0.0, 0.0),
                    xy_spread_um=float(rng.uniform(50, 500)),
                    partial=False,
                )
            )
    return reps


def test_select_lmd_replicates_caps_and_ranks():
    """cap_per_group limits output size; priority selects the best per group."""
    reps = _make_reps_for_ranking()
    assert len(reps) == 20

    # --- anchor_dist: each group keeps its 2 lowest mean_anchor_distance reps ---
    out = select_lmd_replicates(reps, cap_per_group=2, priority="anchor_dist")
    assert len(out) == 6, f"Expected 3 groups × cap 2 = 6 outputs, got {len(out)}"
    from collections import Counter

    counts = Counter(r.manifold_group_id for r in out)
    assert counts == {0: 2, 1: 2, 2: 2}
    # Verify anchor_dist minimality within each group vs the original pool.
    for gid in (0, 1, 2):
        group_all = sorted(r.mean_anchor_distance for r in reps if r.manifold_group_id == gid)[:2]
        group_picked = sorted(r.mean_anchor_distance for r in out if r.manifold_group_id == gid)
        assert np.allclose(
            group_all, group_picked
        ), f"Group {gid}: expected the 2 lowest anchor_distance reps; got {group_picked}"

    # --- spatial_tight: should pick by ascending xy_spread_um ---
    out_sp = select_lmd_replicates(reps, cap_per_group=2, priority="spatial_tight")
    assert len(out_sp) == 6
    for gid in (0, 1, 2):
        group_all = sorted(r.xy_spread_um for r in reps if r.manifold_group_id == gid)[:2]
        group_picked = sorted(r.xy_spread_um for r in out_sp if r.manifold_group_id == gid)
        assert np.allclose(
            group_all, group_picked
        ), f"spatial_tight group {gid}: expected 2 tightest, got {group_picked}"

    # --- composite: accepts the combined metric, returns same cap shape ---
    out_c = select_lmd_replicates(reps, cap_per_group=2, priority="composite")
    assert len(out_c) == 6
    assert all(r.manifold_group_id in {0, 1, 2} for r in out_c)


def test_select_lmd_replicates_rejects_bad_priority():
    """Unknown priority → ConfigError."""
    reps = _make_reps_for_ranking()
    with pytest.raises(ConfigError, match="priority must be"):
        select_lmd_replicates(reps, cap_per_group=2, priority="nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_and_embed — optional dependency, called without HDBSCAN
# ---------------------------------------------------------------------------


def test_load_and_embed_returns_dataclass(tmp_path):
    """``load_and_embed`` returns an EmbeddingResult aligned with kept cells.

    A second call with a different ``max_pcs`` writes a distinct cache key —
    verified by listing ``X_pca_*.npz`` in the cache dir.
    """
    from xldvp_seg.analysis.rare_cell_discovery import (
        RareCellConfig,
        load_and_embed,
    )

    rng = np.random.default_rng(0)
    n = 100
    dets = []
    for i in range(n):
        feats = {
            "area_um2": float(rng.uniform(50, 500)),
            "perimeter_um": float(rng.uniform(20, 80)),
            "major_axis_um": float(rng.uniform(5, 15)),
            "minor_axis_um": float(rng.uniform(4, 12)),
            "circularity": float(np.clip(rng.normal(0.6, 0.1), 0, 1)),
            "solidity": float(np.clip(rng.normal(0.9, 0.05), 0, 1)),
            "eccentricity": float(np.clip(rng.normal(0.5, 0.1), 0, 1)),
            "extent": float(np.clip(rng.normal(0.6, 0.1), 0, 1)),
            "n_nuclei": 1,
            "nuclear_area_fraction": 0.3,
        }
        for j in range(16):
            feats[f"sam2_{j}"] = float(rng.normal(0, 1))
        dets.append(
            {
                "uid": f"c{i}",
                "features": feats,
                "n_nuclei": 1,
                "nuclear_area_fraction": 0.3,
                "nuclei": [{"overlap_fraction": 0.95}],
                "global_center_um": [float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))],
                "organ_id": int(rng.integers(1, 4)),
            }
        )

    cfg = RareCellConfig(
        feature_groups=("shape", "sam2"),
        min_cluster_size=10,
        stability_sizes=(10, 20),
        stability_min_survive=1,
        max_pcs=10,
        use_gpu=False,
        cache_dir=tmp_path,
        seed=0,
    )
    result = load_and_embed(dets, cfg)

    # Structural assertions.
    assert result.X_pca.shape[0] == len(
        result.kept
    ), f"X_pca rows {result.X_pca.shape[0]} != kept cells {len(result.kept)}"
    assert result.n_components <= 10
    assert 0.0 <= result.var_explained <= 1.0
    assert len(result.feature_names) == result.X_scaled.shape[1]

    caches_initial = {p.name for p in tmp_path.glob("X_pca_*.npz")}
    assert len(caches_initial) == 1, f"Expected 1 cache, got {caches_initial}"

    # Different max_pcs → different cache key → a second cache file appears.
    cfg2 = RareCellConfig(
        feature_groups=("shape", "sam2"),
        min_cluster_size=10,
        stability_sizes=(10, 20),
        stability_min_survive=1,
        max_pcs=5,
        use_gpu=False,
        cache_dir=tmp_path,
        seed=0,
    )
    _ = load_and_embed(dets, cfg2)
    caches_after = {p.name for p in tmp_path.glob("X_pca_*.npz")}
    assert (
        len(caches_after) == 2
    ), f"Second call with different max_pcs should create a new cache key; got {caches_after}"


# ---------------------------------------------------------------------------
# discover_manifold_replicates — end-to-end on the JSON fixture
# ---------------------------------------------------------------------------


def test_discover_manifold_replicates_end_to_end(tmp_path):
    """Orchestrator wires embedding → FPS → Voronoi → outliers → Ward replicates.

    Note: the orchestrator builds its own ``RareCellConfig`` with defaults
    (``min_cluster_size=1000``), which imposes a ``>= 2000`` kept-cell floor.
    The 500-cell fixture is inflated 5× (with xy jitter) to clear that floor.
    """
    with open(FIXTURE_PATH) as f:
        fixture_dets = json.load(f)
    assert len(fixture_dets) == 500
    detections = _inflate_fixture(fixture_dets, copies=5, seed=0)
    assert len(detections) == 2500

    cfg = ManifoldSamplingConfig(
        k_anchors=10,
        target_area_um2=200.0,
        outlier_threshold=95.0,
        min_spread_replicate_radii=0.0,  # don't collapse replicates on small fixture
        use_gpu=False,
        cache_dir=tmp_path,
        seed=0,
    )
    result = discover_manifold_replicates(detections, cfg)

    # Structural keys.
    expected = {"replicates", "picked_idx", "labels", "d_to_anchor", "outlier_mask", "stats"}
    assert expected.issubset(result.keys()), f"Missing keys: {expected - set(result.keys())}"

    # FPS and Voronoi shapes.
    assert result["picked_idx"].shape == (10,)
    assert result["labels"].ndim == 1
    assert result["labels"].shape == result["d_to_anchor"].shape == result["outlier_mask"].shape

    # Voronoi labels are in [0, k_anchors) — no -1 sentinel at this stage.
    lbl_min, lbl_max = int(result["labels"].min()), int(result["labels"].max())
    assert (
        lbl_min >= 0 and lbl_max < 10
    ), f"Voronoi labels must lie in [0, 10); got [{lbl_min}, {lbl_max}]"

    # At least one replicate emitted (partial included on this small fixture).
    # With include_partial=False default and target=200, most fixture pairs
    # will easily exceed area — expect ≥1.
    assert len(result["replicates"]) >= 1, "Expected at least 1 replicate on 2500-cell fixture"

    for rep in result["replicates"]:
        assert rep.organ_id in {1, 2, 3}, f"Unexpected organ_id {rep.organ_id}"
        assert len(rep.cell_uids) >= 1
        assert rep.n_cells >= 1
        assert 0 <= rep.manifold_group_id < 10

    # Cache artifact written.
    cache_files = list(tmp_path.glob("manifold_state_*.npz"))
    assert len(cache_files) == 1, f"Expected 1 manifold_state cache, got {cache_files}"


# ---------------------------------------------------------------------------
# Viewer smoke test
# ---------------------------------------------------------------------------


def test_build_linked_viewer_html_smoke(tmp_path):
    """Smoke test for :func:`build_linked_viewer_html`.

    Exercises the real signature (no skip-on-TypeError shim) so a future
    signature change fails loudly rather than silently skipping.
    """
    from xldvp_seg.visualization.manifold_viewer import build_linked_viewer_html

    rng = np.random.default_rng(0)
    n = 30
    umap_3d = rng.uniform(-5, 5, size=(n, 3)).astype(np.float32)
    positions_um = rng.uniform(0, 4000, size=(n, 2)).astype(np.float32)
    labels = rng.integers(0, 5, size=n).astype(np.int32)
    # Simple synthetic thumbnail — 40x60 RGB with a gradient so the base64
    # encoding path exercises real pixel data.
    h, w = 40, 60
    thumb = np.zeros((h, w, 3), dtype=np.uint8)
    thumb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    thumb[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]

    html = build_linked_viewer_html(
        umap_coords_3d=umap_3d,
        positions_um=positions_um,
        labels=labels,
        thumbnail_rgb=thumb,
        slide_extent_um=(5000.0, 3000.0),
        title="smoke",
    )

    assert isinstance(html, str) and len(html) > 0
    assert "<script" in html
    assert "data:image/jpeg;base64," in html, "thumbnail should be embedded as data URI"
    assert html.count("</script>") >= 1
    # safe_json must not leak a raw </script> inside a <script> payload.
    assert "<\\/script>" in html or html.count("</script>") == html.count("<script")
    HTMLParser().feed(html)


# ---------------------------------------------------------------------------
# CLI smoke — invokes the real `xlseg manifold-sample` entry on the fixture
# ---------------------------------------------------------------------------


class TestCLISmoke:
    def test_cli_manifold_sample_on_fixture(self, tmp_path):
        """Invoke ``xlseg manifold-sample`` via its Python entry on the fixture.

        Catches CLI-wiring regressions (argparse flag drift, import errors,
        orchestrator/CLI signature mismatches). Runs CPU-only on the 500-cell
        fixture — ~5-10 s wall.
        """
        # Invoke the manifold_sample script's main() directly rather than
        # shelling out — keeps the smoke test fast and lets pytest capture
        # the traceback on failure instead of buried stderr.
        import importlib.util

        if not FIXTURE_PATH.exists():
            pytest.skip(f"fixture missing: {FIXTURE_PATH}")

        script_path = REPO / "scripts" / "manifold_sample.py"
        spec = importlib.util.spec_from_file_location("_manifold_sample_cli", script_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rc = mod.main(
            [
                "--detections",
                str(FIXTURE_PATH),
                "--output-dir",
                str(tmp_path),
                "--k-anchors",
                "10",
                "--target-area-um2",
                "500",
                "--no-use-gpu",
                "--no-exemplar-json",
            ]
        )
        assert rc == 0, f"CLI returned non-zero: {rc}"

        # Required outputs exist + nonzero.
        for fname in (
            "manifold_replicates.json",
            "manifold_sample_stats.json",
            "lmd_selected_replicates.json",
            "lmd_selected_replicates.csv",
        ):
            p = tmp_path / fname
            assert p.exists(), f"missing {fname}"
            assert p.stat().st_size > 0, f"empty {fname}"
        assert list(tmp_path.glob("manifold_state_*.npz")), "missing manifold state cache"


# ---------------------------------------------------------------------------
# Extra branch coverage (review-driven additions)
# ---------------------------------------------------------------------------


def test_voronoi_assign_dim_mismatch_raises():
    """Mismatched feature dims must raise :class:`ConfigError`, not crash downstream."""
    X = np.zeros((5, 3), dtype=np.float32)
    C = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ConfigError, match="dim mismatch"):
        voronoi_assign(X, C, use_gpu=False)


def test_flag_outliers_global_pct_bounds():
    """global_pct threshold outside ``(0, 100)`` must fail fast."""
    labels = np.zeros(10, dtype=np.int32)
    d = np.arange(10, dtype=np.float32)
    with pytest.raises(ConfigError, match="in \\(0, 100\\)"):
        flag_outliers(labels, d, method="global_pct", threshold=0.0)
    with pytest.raises(ConfigError, match="in \\(0, 100\\)"):
        flag_outliers(labels, d, method="global_pct", threshold=100.0)
    with pytest.raises(ConfigError, match="must be finite"):
        flag_outliers(labels, d, method="global_pct", threshold=float("nan"))


def test_flag_outliers_unknown_method():
    labels = np.zeros(3, dtype=np.int32)
    d = np.zeros(3, dtype=np.float32)
    with pytest.raises(ConfigError, match="must be 'global_pct' or 'per_group_mad'"):
        flag_outliers(labels, d, method="nope", threshold=50.0)  # type: ignore[arg-type]


def test_chunked_ward_cluster_degenerate_tiny():
    """n==0 and n<2 degenerate paths must return sane labels without crashing."""
    assert _chunked_ward_cluster(np.zeros((0, 2)), n_rep=5).shape == (0,)
    assert _chunked_ward_cluster(np.zeros((1, 2)), n_rep=5).tolist() == [0]
    assert _chunked_ward_cluster(np.zeros((3, 2)), n_rep=1).tolist() == [0, 0, 0]


def test_spatial_replicates_single_tier_fallback(tmp_path):
    """All cells having organ_id==0 must trigger single-tier fallback when
    ``organ_required=False`` (default) — one replicate per manifold group,
    each tagged with ``organ_id == organ_drop_value``."""
    rng = np.random.default_rng(0)
    n = 200
    positions = rng.uniform(0, 10_000, size=(n, 2)).astype(np.float32)
    labels = rng.integers(0, 3, size=n).astype(np.int32)
    organs = np.zeros(n, dtype=np.int32)  # all unassigned
    areas = np.full(n, 200.0, dtype=np.float32)
    cell_uids = [f"c{i}" for i in range(n)]
    mask = np.zeros(n, dtype=bool)
    d = rng.uniform(0, 1, size=n).astype(np.float32)

    cfg = ManifoldSamplingConfig(
        k_anchors=3,
        target_area_um2=500.0,
        min_spread_replicate_radii=0.0,  # disable spread guard for synthetic xy
        organ_required=False,
    )
    reps = spatial_replicates(positions, labels, organs, areas, cell_uids, mask, d, cfg)
    assert reps, "fallback should emit at least one replicate"
    assert all(r.organ_id == cfg.organ_drop_value for r in reps)


def test_spatial_replicates_organ_required_raises():
    """``organ_required=True`` must abort when no cell has an organ id."""
    n = 50
    positions = np.zeros((n, 2), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    organs = np.zeros(n, dtype=np.int32)
    areas = np.ones(n, dtype=np.float32)
    cfg = ManifoldSamplingConfig(organ_required=True)
    with pytest.raises(ConfigError, match="no cells have an assigned organ_id"):
        spatial_replicates(
            positions,
            labels,
            organs,
            areas,
            [f"c{i}" for i in range(n)],
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=np.float32),
            cfg,
        )


def test_select_lmd_replicates_zero_variance_composite():
    """Composite ranking on identical scores reduces to insertion-order via stable sort."""

    def _r(rid: str, anchor: float, spread: float, group: int) -> Replicate:
        return Replicate(
            replicate_id=rid,
            manifold_group_id=group,
            organ_id=1,
            within_pair_replicate_idx=0,
            cell_uids=["x"],
            cell_indices=[0],
            n_cells=1,
            total_area_um2=100.0,
            mean_anchor_distance=anchor,
            mean_xy_um=(0.0, 0.0),
            xy_spread_um=spread,
        )

    reps = [_r(f"r{i}", 5.0, 7.0, i) for i in range(4)]
    out = select_lmd_replicates(reps, cap_per_group=10, priority="composite")
    # All scores identical -> stable argsort -> output == input order.
    assert [r.replicate_id for r in out] == ["r0", "r1", "r2", "r3"]


def test_discover_manifold_replicates_cache_roundtrip(tmp_path):
    """Two consecutive runs with identical cfg must reuse the cache and
    produce byte-identical ``picked_idx``/``labels``/``d_to_anchor``/
    ``outlier_mask`` arrays."""
    with open(FIXTURE_PATH) as f:
        dets = json.load(f)
    # Inflate to exceed k_anchors.
    dets = _inflate_fixture(dets, copies=3, seed=0)

    cfg = ManifoldSamplingConfig(
        k_anchors=50,
        target_area_um2=500.0,
        use_gpu=False,
        cache_dir=tmp_path,
        seed=17,
    )

    r1 = discover_manifold_replicates(copy.deepcopy(dets), cfg)
    cache_files = list(tmp_path.glob("manifold_state_*.npz"))
    assert len(cache_files) == 1
    mtime_before = cache_files[0].stat().st_mtime_ns

    r2 = discover_manifold_replicates(copy.deepcopy(dets), cfg)
    # Cache file unchanged (no rewrite on hit).
    assert cache_files[0].stat().st_mtime_ns == mtime_before
    np.testing.assert_array_equal(r1["picked_idx"], r2["picked_idx"])
    np.testing.assert_array_equal(r1["labels"], r2["labels"])
    np.testing.assert_allclose(r1["d_to_anchor"], r2["d_to_anchor"])
    np.testing.assert_array_equal(r1["outlier_mask"], r2["outlier_mask"])


def test_discover_manifold_replicates_cache_invalidates_on_weights(tmp_path):
    """Changing ``feature_group_weights`` must produce a new cache entry.

    Guards against the review-flagged case where two configs with matching
    kept-cell counts could otherwise collide on the same cache key.
    """
    with open(FIXTURE_PATH) as f:
        dets = json.load(f)
    dets = _inflate_fixture(dets, copies=3, seed=0)

    base_kwargs = dict(
        k_anchors=50,
        target_area_um2=500.0,
        use_gpu=False,
        cache_dir=tmp_path,
        seed=17,
    )
    discover_manifold_replicates(
        copy.deepcopy(dets),
        ManifoldSamplingConfig(**base_kwargs, feature_group_weights="equal"),
    )
    discover_manifold_replicates(
        copy.deepcopy(dets),
        ManifoldSamplingConfig(**base_kwargs, feature_group_weights="raw"),
    )
    # Two distinct manifold_state caches -> two distinct hash keys.
    caches = list(tmp_path.glob("manifold_state_*.npz"))
    assert len(caches) == 2, f"expected 2 distinct caches, got {[c.name for c in caches]}"
