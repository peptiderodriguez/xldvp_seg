"""Tests for xldvp_seg.analysis.region_clustering."""

from __future__ import annotations

import numpy as np
import pytest

from xldvp_seg.analysis.region_clustering import (
    cluster_hdbscan,
    cluster_leiden,
    find_optimal_k_elbow,
    hopkins_statistic,
    process_region,
)


def _make_blobs(n_per_blob=80, centers=None, spread=0.3, seed=0):
    """Make synthetic blobs at the given centers."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=float)
    X = np.vstack([rng.normal(c, spread, size=(n_per_blob, len(c))) for c in centers])
    y = np.concatenate([np.full(n_per_blob, i) for i in range(len(centers))])
    return X, y


# ---------------------------------------------------------------------------
# Hopkins statistic
# ---------------------------------------------------------------------------


def test_hopkins_on_uniform_is_around_half():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(500, 5))
    h = hopkins_statistic(X, rng=rng)
    # Uniform data has Hopkins ~ 0.5; allow generous tolerance for sampling noise
    assert 0.35 < h < 0.65, f"expected ~0.5 for uniform, got {h}"


def test_hopkins_on_tight_blobs_is_high():
    X, _ = _make_blobs(n_per_blob=100, centers=[[0, 0], [10, 10], [0, 10]], spread=0.1)
    h = hopkins_statistic(X, rng=np.random.default_rng(0))
    assert h > 0.8, f"expected H>0.8 for tight blobs, got {h}"


def test_hopkins_handles_zero_range():
    X = np.zeros((30, 3))  # all identical points: data range = 0
    h = hopkins_statistic(X, rng=np.random.default_rng(0))
    assert h == 0.5


# ---------------------------------------------------------------------------
# Elbow method for k-means
# ---------------------------------------------------------------------------


def test_elbow_three_blobs_picks_three():
    X, _ = _make_blobs(
        n_per_blob=80,
        centers=[[0, 0], [10, 0], [0, 10]],
        spread=0.4,
    )
    best_k, sil, labels, ch, sil_per_k, inertia_per_k = find_optimal_k_elbow(
        X, max_k=6, rng=np.random.default_rng(0)
    )
    assert best_k == 3, f"expected k=3, got {best_k} (sil per k: {sil_per_k})"
    # Silhouette should be high for well-separated blobs
    assert sil > 0.5
    # Inertia should be monotone decreasing
    ins = [inertia_per_k[k] for k in sorted(inertia_per_k)]
    assert all(ins[i] >= ins[i + 1] for i in range(len(ins) - 1))


def test_elbow_returns_degenerate_on_tiny_input():
    X = np.array([[0.0, 0.0]])  # n=1, can't even do k=2
    best_k, sil, labels, ch, sil_per_k, inertia_per_k = find_optimal_k_elbow(X, max_k=8)
    assert best_k == 1
    assert sil_per_k == {} and inertia_per_k == {}
    assert labels.shape == (1,)


# ---------------------------------------------------------------------------
# Leiden
# ---------------------------------------------------------------------------


def test_leiden_finds_two_blobs():
    X, y = _make_blobs(n_per_blob=60, centers=[[0, 0], [10, 10]], spread=0.4)
    labels = cluster_leiden(X, n_neighbors=10, resolution=1.0)
    # Leiden at default resolution often over-splits — but it should never
    # *merge* across truly separated blobs. Assert no Leiden cluster straddles
    # both true blobs: each Leiden label belongs predominantly to one blob.
    for lbl in np.unique(labels):
        members_true_blob = y[labels == lbl]
        _, counts = np.unique(members_true_blob, return_counts=True)
        purity = counts.max() / counts.sum()
        assert purity > 0.95, f"Leiden label {lbl} straddles blobs: {counts}"


def test_leiden_handles_small_input():
    X, _ = _make_blobs(n_per_blob=8, centers=[[0, 0], [5, 5]])
    # n_neighbors internally capped at n-1
    labels = cluster_leiden(X, n_neighbors=30, resolution=1.0)
    assert labels.shape == (16,)
    assert labels.dtype == np.int32


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------


def test_hdbscan_separates_blobs_from_noise():
    X_blobs, _ = _make_blobs(n_per_blob=80, centers=[[0, 0], [15, 15]], spread=0.3)
    rng = np.random.default_rng(0)
    X_noise = rng.uniform(-30, 30, size=(40, 2))
    X = np.vstack([X_blobs, X_noise])
    labels = cluster_hdbscan(X, min_cluster_size=20)
    # Should find at least 2 clusters
    assert len(np.unique(labels[labels >= 0])) >= 2
    # Noise label (-1) should be used for the scattered points
    assert (labels == -1).any()


def test_hdbscan_respects_min_cluster_size_cap():
    # With n=20 and requested min_cluster_size=500, should cap at n//5 = 4, floor at 5
    X, _ = _make_blobs(n_per_blob=10, centers=[[0, 0], [5, 5]])
    labels = cluster_hdbscan(X, min_cluster_size=500)
    assert labels.shape == (20,)


# ---------------------------------------------------------------------------
# process_region end-to-end
# ---------------------------------------------------------------------------


def _synthetic_detections(n_per_blob=60, centers=None, seed=0):
    """Synthetic detection dicts shaped like the real pipeline output."""
    rng = np.random.default_rng(seed)
    X, _ = _make_blobs(n_per_blob=n_per_blob, centers=centers, spread=0.4, seed=seed)
    detections = []
    for row in X:
        feats = {
            "area_um2": float(abs(row[0]) * 10 + 50),
            "circularity": float((np.tanh(row[1]) + 1) / 2),
            "aspect_ratio": float(1 + abs(row[0]) * 0.1),
            "solidity": float(0.7 + 0.25 * np.tanh(row[1])),
            "ch0_mean": float(row[0] * 100 + 500 + rng.normal() * 10),
            "ch1_mean": float(row[1] * 100 + 500 + rng.normal() * 10),
            "ch0_std": float(abs(row[0]) * 20 + 50),
            "ch1_std": float(abs(row[1]) * 20 + 50),
        }
        detections.append({"features": feats, "organ_id": 1})
    return detections


def test_process_region_end_to_end():
    detections = _synthetic_detections(n_per_blob=80, centers=[[0, 0], [3, 3], [0, 3]], seed=0)
    feature_names = [
        "area_um2",
        "circularity",
        "aspect_ratio",
        "solidity",
        "ch0_mean",
        "ch1_mean",
        "ch0_std",
        "ch1_std",
    ]
    result = process_region(
        detections,
        feature_names,
        max_k=6,
        max_points_plot=1000,
        rng=np.random.default_rng(0),
    )
    assert result is not None
    assert result["n_cells"] == 240
    assert result["n_features"] == 8
    assert len(result["umap_x"]) == 240
    assert len(result["pc1"]) == 240
    # All four label arrays same length
    for key in ("labels_kmeans", "labels_leiden", "labels_hdbscan_pca", "labels_hdbscan_umap"):
        assert len(result[key]) == 240, f"{key} wrong length"
    # Hopkins should be high for separated blobs
    assert result["hopkins"] > 0.7
    # n_clusters keys present for all methods
    for method in ("kmeans", "leiden", "hdbscan_pca", "hdbscan_umap"):
        assert method in result["n_clusters"]
        assert method in result["n_noise"]
    # var_explained always 3-entry (padded if fewer PCs)
    assert len(result["var_explained"]) == 3


def test_process_region_too_few_cells_returns_none():
    detections = _synthetic_detections(n_per_blob=5, centers=[[0, 0]], seed=0)
    result = process_region(detections, ["ch0_mean"], max_points_plot=500)
    assert result is None


def test_process_region_drops_constant_features():
    # n=60 synthetic — enough to pass the >=50 threshold
    dets = []
    for i in range(60):
        dets.append(
            {
                "organ_id": 1,
                "features": {
                    "varying": float(i),
                    "constant": 42.0,  # zero variance — must be dropped
                    "also_varying": float(i * 2),
                    "noise": float(np.sin(i)),
                },
            }
        )
    result = process_region(
        dets,
        ["varying", "constant", "also_varying", "noise"],
        max_points_plot=500,
        rng=np.random.default_rng(0),
    )
    assert result is not None
    # "constant" feature should have been dropped from active set
    assert result["n_features"] == 3
    loadings_pc1 = [item["feature"] for item in result["top_loadings"]["PC1"]]
    assert "constant" not in loadings_pc1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
