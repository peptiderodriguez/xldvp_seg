"""Tests for scripts/score_vessel_lumens.py."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.score_vessel_lumens import (
    _collect_feature_names,
    _filter_final,
    _promote_basic_fields,
)

# --- Fixtures ---


def _make_lumen(uid, area=100, rf_score=None, features=None, **kwargs):
    l = {"uid": uid, "area_um2": area, "equiv_diameter_um": 11.3, "features": features or {}}
    if rf_score is not None:
        l["rf_score"] = rf_score
    l.update(kwargs)
    return l


# --- Tests ---


class TestPromoteBasicFields:
    def test_promotes_numeric_fields(self):
        lumens = [_make_lumen("a", area=150, contrast_ratio=2.5)]
        _promote_basic_fields(lumens)
        assert lumens[0]["features"]["area_um2"] == 150
        assert lumens[0]["features"]["equiv_diameter_um"] == 11.3

    def test_does_not_overwrite_existing(self):
        lumens = [_make_lumen("a", area=150, features={"area_um2": 999})]
        _promote_basic_fields(lumens)
        assert lumens[0]["features"]["area_um2"] == 999  # not overwritten

    def test_skips_non_numeric(self):
        lumens = [{"uid": "a", "features": {}, "darkness_tier": "dark"}]
        _promote_basic_fields(lumens)
        assert "darkness_tier" not in lumens[0]["features"]


class TestCollectFeatureNames:
    def test_collects_numeric_keys(self):
        lumens = [
            _make_lumen("a", features={"morph_circ": 0.9, "ch0_mean": 100, "is_valid": True}),
            _make_lumen("b", features={"morph_circ": 0.8, "ch1_mean": 200}),
        ]
        names = _collect_feature_names(lumens, {"a", "b"})
        assert "morph_circ" in names
        assert "ch0_mean" in names
        assert "ch1_mean" in names
        assert "is_valid" not in names  # bool excluded

    def test_excludes_prefixes(self):
        lumens = [_make_lumen("a", features={"bbox_x": 10, "discovery_scale": 64, "area_um2": 100})]
        names = _collect_feature_names(lumens, {"a"})
        assert "bbox_x" not in names
        assert "discovery_scale" not in names
        assert "area_um2" in names

    def test_only_annotated_uids(self):
        lumens = [
            _make_lumen("a", features={"feat1": 1.0}),
            _make_lumen("b", features={"feat2": 2.0}),
        ]
        names = _collect_feature_names(lumens, {"a"})  # only uid "a"
        assert "feat1" in names
        assert "feat2" not in names


class TestFilterFinal:
    def _make_lumens_with_markers(self):
        return [
            _make_lumen("v1", rf_score=0.90, n_marker_wall=20, n_SMA_wall=12, n_LYVE1_wall=8),
            _make_lumen("v2", rf_score=0.80, n_marker_wall=15, n_SMA_wall=10, n_LYVE1_wall=5),
            _make_lumen("v3", rf_score=0.50, n_marker_wall=10, n_SMA_wall=5, n_LYVE1_wall=5),
            _make_lumen("v4", rf_score=0.30, n_marker_wall=5, n_SMA_wall=3, n_LYVE1_wall=2),
            _make_lumen("v5", rf_score=0.95, n_marker_wall=3, n_SMA_wall=2, n_LYVE1_wall=1),
        ]

    def test_rf_threshold_plus_marker(self):
        lumens = self._make_lumens_with_markers()
        result = _filter_final(lumens, 0.75, ["SMA", "LYVE1"], 8, set(), set())
        uids = {l["uid"] for l in result}
        assert "v1" in uids  # RF=0.90, SMA=12>=8
        assert "v2" in uids  # RF=0.80, SMA=10>=8
        assert "v3" not in uids  # RF=0.50 < 0.75
        assert "v4" not in uids  # RF=0.30 < 0.75
        assert "v5" not in uids  # RF=0.95 but SMA=2<8, LYVE1=1<8

    def test_annotation_positive_override(self):
        lumens = self._make_lumens_with_markers()
        result = _filter_final(lumens, 0.75, ["SMA", "LYVE1"], 8, {"v4"}, set())
        uids = {l["uid"] for l in result}
        assert "v4" in uids  # rescued by annotation

    def test_annotation_negative_override(self):
        lumens = self._make_lumens_with_markers()
        result = _filter_final(lumens, 0.75, ["SMA", "LYVE1"], 8, set(), {"v1"})
        uids = {l["uid"] for l in result}
        assert "v1" not in uids  # excluded by annotation

    def test_fallback_to_coarse_marker_wall(self):
        lumens = [
            _make_lumen("v1", rf_score=0.90, n_marker_wall=10),
            _make_lumen("v2", rf_score=0.90, n_marker_wall=5),
        ]
        # No per-marker fields → falls back to n_marker_wall
        result = _filter_final(lumens, 0.75, [], 8, set(), set())
        uids = {l["uid"] for l in result}
        assert "v1" in uids  # n_marker_wall=10>=8
        assert "v2" not in uids  # n_marker_wall=5<8
