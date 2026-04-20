"""Tests for ``compute_cluster_spatial_stats`` from
``scripts/global_cluster_spatial_viewer.py``.

The function is currently a script-level helper (not promoted to the package
yet). Import by file path via ``importlib`` so we don't execute the full
``main()`` entry point.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "global_cluster_spatial_viewer.py"

# Ensure repo root is importable for the script's own ``from xldvp_seg...`` imports
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("_gcs_viewer", SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

compute_cluster_spatial_stats = _mod.compute_cluster_spatial_stats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_region_cluster_is_organ_specific():
    """A cluster whose cells all live in one region has focal=0, k_90=1."""
    cluster_labels = np.array([5, 5, 5, 5], dtype=np.int32)
    region_ids = np.array([1, 1, 1, 1], dtype=np.int64)
    stats = compute_cluster_spatial_stats(cluster_labels, region_ids)
    assert 5 in stats
    s = stats[5]
    assert s["focal_multimodal"] == 0.0, f"organ-specific cluster should be focal=0, got {s}"
    assert s["k_90"] == 1
    assert s["top_region_frac"] == 1.0
    assert s["n_major"] == 1
    assert s["n_regions_touched"] == 1


def test_bimodal_cluster_has_positive_focal_score():
    """50/50 split across two regions: focal_multimodal > 0, k_90=2."""
    cluster_labels = np.zeros(100, dtype=np.int32)
    region_ids = np.concatenate([np.full(50, 1), np.full(50, 2)]).astype(np.int64)
    stats = compute_cluster_spatial_stats(cluster_labels, region_ids)
    assert 0 in stats
    s = stats[0]
    assert s["focal_multimodal"] > 0
    assert s["k_90"] == 2
    assert s["top_region_frac"] == pytest.approx(0.5, abs=1e-6)
    assert s["n_major"] == 2


def test_ubiquitous_cluster_low_focal_score():
    """100 cells spread across 20 regions evenly (5 each).

    No region passes the 10% major threshold, so n_major=0 and focal=0.
    top3_frac = 3 * 0.05 = 0.15.
    """
    cluster_labels = np.zeros(100, dtype=np.int32)
    region_ids = np.repeat(np.arange(1, 21, dtype=np.int64), 5)
    stats = compute_cluster_spatial_stats(cluster_labels, region_ids)
    assert 0 in stats
    s = stats[0]
    assert s["top3_frac"] == pytest.approx(0.15, abs=1e-3)
    assert s["n_major"] == 0, f"5/100 per region == 5% < 10%, expected n_major=0 got {s['n_major']}"
    # focal_multimodal = top3_frac * min(n_major, 5) = 0.15 * 0 = 0
    assert s["focal_multimodal"] == 0.0


def test_entropy_normalization():
    """Entropy (normalized, stored as 'entropy') must be in [0, 1]."""
    rng = np.random.default_rng(0)
    for trial in range(5):
        n_regions = int(rng.integers(2, 20))
        n_cells = int(rng.integers(50, 500))
        cluster_labels = np.zeros(n_cells, dtype=np.int32)
        region_ids = rng.integers(1, n_regions + 1, size=n_cells).astype(np.int64)
        stats = compute_cluster_spatial_stats(cluster_labels, region_ids)
        assert 0 in stats
        h = stats[0]["entropy"]
        assert 0.0 <= h <= 1.0 + 1e-9, f"trial {trial}: entropy {h} out of [0,1]"


def test_noise_label_excluded():
    """Cells with cluster_label < 0 (HDBSCAN noise) must not produce an entry."""
    cluster_labels = np.concatenate([np.full(50, -1), np.full(50, 0)]).astype(np.int32)
    region_ids = np.ones(100, dtype=np.int64)
    stats = compute_cluster_spatial_stats(cluster_labels, region_ids)
    assert -1 not in stats, "noise label -1 should be excluded"
    assert 0 in stats
    assert stats[0]["n_cells"] == 50


def test_region_zero_excluded():
    """region_id == 0 (unassigned) cells must be excluded from the cluster stats."""
    cluster_labels = np.zeros(100, dtype=np.int32)
    region_ids = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
    stats = compute_cluster_spatial_stats(cluster_labels, region_ids)
    assert 0 in stats
    # Only 50 cells in region 1 count; the 50 with region_id=0 are dropped.
    assert stats[0]["n_cells"] == 50
    assert stats[0]["n_regions_touched"] == 1


def test_k_90_degenerate():
    """k_90 should be 1 for a single region and 2 for two regions at 50/50."""
    # Single region -> k_90 = 1
    cl1 = np.zeros(10, dtype=np.int32)
    rg1 = np.ones(10, dtype=np.int64)
    s1 = compute_cluster_spatial_stats(cl1, rg1)[0]
    assert s1["k_90"] == 1

    # Two regions 50/50 -> k_90 = 2
    cl2 = np.zeros(10, dtype=np.int32)
    rg2 = np.array([1] * 5 + [2] * 5, dtype=np.int64)
    s2 = compute_cluster_spatial_stats(cl2, rg2)[0]
    assert s2["k_90"] == 2


def test_divergence_rewards_spread():
    """Divergence = n_major * entropy_norm -- more major + higher entropy wins."""
    # Cluster A: 3 major regions evenly (100/100/100 -> entropy near 1)
    cluster_labels_a = np.zeros(300, dtype=np.int32)
    region_ids_a = np.concatenate([np.full(100, 1), np.full(100, 2), np.full(100, 3)]).astype(
        np.int64
    )
    stats_a = compute_cluster_spatial_stats(cluster_labels_a, region_ids_a)[0]

    # Cluster B: 1 major region (all in region 1) -> entropy = 0
    cluster_labels_b = np.zeros(300, dtype=np.int32)
    region_ids_b = np.ones(300, dtype=np.int64)
    stats_b = compute_cluster_spatial_stats(cluster_labels_b, region_ids_b)[0]

    # A: n_major=3 * entropy_norm ~ high; B: n_major=1 * entropy_norm=0.
    assert (
        stats_a["divergence"] > stats_b["divergence"]
    ), f"expected A.div > B.div, got A={stats_a['divergence']} B={stats_b['divergence']}"
    assert stats_a["n_major"] == 3
    assert stats_b["n_major"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
