"""Tests for the rare-cell CLI helpers (discover_rare_cell_types.py).

Loads the script as a module without running main() and exercises the pure
helpers: _pick_exemplars selection logic and parse_args flag semantics.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_cli_module():
    script = REPO / "scripts" / "discover_rare_cell_types.py"
    spec = importlib.util.spec_from_file_location("discover_rare_cell_types_cli", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["discover_rare_cell_types_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _pick_exemplars
# ---------------------------------------------------------------------------


def test_pick_exemplars_caps_at_cluster_size():
    mod = _load_cli_module()
    # 3 cells in cluster 0, 10 in cluster 1
    kept = [{"id": i} for i in range(13)]
    labels = np.array([0, 0, 0] + [1] * 10, dtype=np.int32)
    summary = [
        {"cluster_id": 0, "size": 3, "stable": True},
        {"cluster_id": 1, "size": 10, "stable": True},
    ]
    exemplars = mod._pick_exemplars(
        kept, labels, summary, n_per_cluster=5, top_n_clusters=10, stable_only=True, seed=0
    )
    # cluster 0 capped at 3, cluster 1 at 5 → total 8
    by_cid = {}
    for e in exemplars:
        by_cid.setdefault(e["_exemplar_cluster_id"], []).append(e)
    assert len(by_cid[0]) == 3
    assert len(by_cid[1]) == 5


def test_pick_exemplars_stable_only_filters_exploratory():
    mod = _load_cli_module()
    kept = [{"id": i} for i in range(20)]
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    summary = [
        {"cluster_id": 0, "size": 10, "stable": True},
        {"cluster_id": 1, "size": 10, "stable": False},  # exploratory → excluded
    ]
    exemplars = mod._pick_exemplars(
        kept, labels, summary, n_per_cluster=5, stable_only=True, seed=0
    )
    assert all(e["_exemplar_cluster_id"] == 0 for e in exemplars)


def test_pick_exemplars_top_n_selects_smallest():
    mod = _load_cli_module()
    kept = [{"id": i} for i in range(30)]
    labels = np.array([0] * 5 + [1] * 10 + [2] * 15, dtype=np.int32)
    summary = [
        {"cluster_id": 2, "size": 15, "stable": True},  # largest
        {"cluster_id": 1, "size": 10, "stable": True},
        {"cluster_id": 0, "size": 5, "stable": True},  # smallest
    ]
    exemplars = mod._pick_exemplars(
        kept, labels, summary, n_per_cluster=3, top_n_clusters=2, stable_only=True, seed=0
    )
    # Should pick the two smallest (clusters 0 and 1), not 2.
    picked = {e["_exemplar_cluster_id"] for e in exemplars}
    assert picked == {0, 1}


# ---------------------------------------------------------------------------
# CLI flag semantics
# ---------------------------------------------------------------------------


def test_parse_args_use_gpu_default_on():
    mod = _load_cli_module()
    args = mod.parse_args(["--detections", "x.json", "--output-dir", "o"])
    assert args.use_gpu is True


def test_parse_args_no_use_gpu_flips_off():
    mod = _load_cli_module()
    args = mod.parse_args(["--detections", "x.json", "--output-dir", "o", "--no-use-gpu"])
    assert args.use_gpu is False


def test_parse_args_feature_group_weights_default_equal():
    mod = _load_cli_module()
    args = mod.parse_args(["--detections", "x.json", "--output-dir", "o"])
    assert args.feature_group_weights == "equal"


def test_parse_args_feature_group_weights_raw():
    mod = _load_cli_module()
    args = mod.parse_args(
        ["--detections", "x.json", "--output-dir", "o", "--feature-group-weights", "raw"]
    )
    assert args.feature_group_weights == "raw"


# ---------------------------------------------------------------------------
# Atomic CSV writer
# ---------------------------------------------------------------------------


def test_write_cluster_summary_csv_handles_empty_and_none_moran(tmp_path):
    mod = _load_cli_module()
    # Empty summary still writes header (no tmp file left behind)
    empty_path = tmp_path / "empty.csv"
    mod.write_cluster_summary_csv([], empty_path)
    content = empty_path.read_text(encoding="utf-8")
    assert "cluster_id" in content
    # moran_i=None emitted as empty cell
    rows = [
        {
            "cluster_id": 0,
            "size": 100,
            "hdbscan_persistence": 0.4,
            "moran_i": None,
            "stable": True,
            "noise_pct": 0.1,
            "top_regions": "1:100",
            "top_morph_features": "",
        },
    ]
    p = tmp_path / "s.csv"
    mod.write_cluster_summary_csv(rows, p)
    out = p.read_text(encoding="utf-8").splitlines()
    # Second line (data) has empty moran_i field between two commas
    assert ",," in out[1], f"expected empty moran cell; got {out[1]!r}"
    # No leftover tmp file
    assert not (tmp_path / "s.csv.tmp").exists()


# ---------------------------------------------------------------------------
# Atomic npy writer
# ---------------------------------------------------------------------------


def test_atomic_np_save_round_trip(tmp_path):
    mod = _load_cli_module()
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    p = tmp_path / "data.npy"
    mod._atomic_np_save(p, arr)
    loaded = np.load(p)
    assert np.array_equal(loaded, arr)
    assert not (tmp_path / "data.npy.tmp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
