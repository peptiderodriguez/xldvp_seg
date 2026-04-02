"""Tests for scripts/detect_vessel_structures.py — vessel structure detection."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from detect_vessel_structures import (
    analyze_marker_composition,
    assign_vessel_type,
    classify_vessel_morphology,
    compute_vessel_morphometry,
    select_multi_marker_cells,
    tag_detections,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_detection(features=None, **kwargs):
    """Create a minimal detection dict."""
    d = {"features": features or {}}
    d.update(kwargs)
    return d


@pytest.fixture
def sma_cd31_detections():
    """20 detections with SMA/CD31 classifications."""
    dets = []
    for i in range(20):
        feat = {}
        if i < 8:
            feat["SMA_class"] = "positive"
            feat["CD31_class"] = "negative"
        elif i < 14:
            feat["SMA_class"] = "negative"
            feat["CD31_class"] = "positive"
        else:
            feat["SMA_class"] = "negative"
            feat["CD31_class"] = "negative"
        dets.append(_make_detection(feat))
    return dets


@pytest.fixture
def ring_args():
    """Default args namespace for ring classification."""
    return SimpleNamespace(
        linearity_threshold=3.0,
        ring_threshold=0.5,
        arc_threshold=0.3,
    )


# ---------------------------------------------------------------------------
# Multi-marker cell selection
# ---------------------------------------------------------------------------


class TestSelectMultiMarkerCells:
    def test_or_logic(self, sma_cd31_detections):
        idx = select_multi_marker_cells(
            sma_cd31_detections,
            ["SMA_class==positive", "CD31_class==positive"],
            logic="or",
        )
        # 8 SMA+ + 6 CD31+ = 14
        assert len(idx) == 14

    def test_and_logic(self, sma_cd31_detections):
        idx = select_multi_marker_cells(
            sma_cd31_detections,
            ["SMA_class==positive", "CD31_class==positive"],
            logic="and",
        )
        # No cell is both SMA+ AND CD31+
        assert len(idx) == 0

    def test_single_filter(self, sma_cd31_detections):
        idx = select_multi_marker_cells(
            sma_cd31_detections,
            ["SMA_class==positive"],
            logic="or",
        )
        assert len(idx) == 8

    def test_empty_filters(self, sma_cd31_detections):
        idx = select_multi_marker_cells(sma_cd31_detections, [], logic="or")
        assert len(idx) == 0

    def test_boolean_coercion(self):
        dets = [_make_detection({"marker": True}), _make_detection({"marker": False})]
        idx = select_multi_marker_cells(dets, ["marker==True"], logic="or")
        assert len(idx) == 1

    def test_malformed_filter_exits(self):
        with pytest.raises(SystemExit):
            select_multi_marker_cells([], ["bad_filter"], logic="or")


# ---------------------------------------------------------------------------
# Morphology classification
# ---------------------------------------------------------------------------


class TestClassifyVesselMorphology:
    def test_strip(self, ring_args):
        metrics = {
            "linearity": 4.0,
            "elongation": 5.0,
            "ring_score": 0.1,
            "arc_fraction": 0.1,
            "circularity": 0.3,
            "hollowness": 0.4,
            "has_curvature": False,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "strip"

    def test_ring_via_graph(self, ring_args):
        metrics = {
            "linearity": 1.5,
            "elongation": 1.2,
            "ring_score": 0.7,
            "arc_fraction": 0.9,
            "circularity": 0.4,
            "hollowness": 0.7,
            "has_curvature": False,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "ring"

    def test_ring_via_pca(self, ring_args):
        metrics = {
            "linearity": 1.5,
            "elongation": 1.2,
            "ring_score": 0.3,
            "arc_fraction": 0.9,
            "circularity": 0.8,
            "hollowness": 0.7,
            "has_curvature": False,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "ring"

    def test_arc(self, ring_args):
        metrics = {
            "linearity": 2.0,
            "elongation": 2.5,
            "ring_score": 0.3,
            "arc_fraction": 0.5,
            "circularity": 0.4,
            "hollowness": 0.5,
            "has_curvature": False,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "arc"

    def test_arc_via_curvature(self, ring_args):
        metrics = {
            "linearity": 2.0,
            "elongation": 3.5,
            "ring_score": 0.2,
            "arc_fraction": 0.2,
            "circularity": 0.3,
            "hollowness": 0.3,
            "has_curvature": True,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "arc"

    def test_cluster(self, ring_args):
        metrics = {
            "linearity": 1.0,
            "elongation": 1.5,
            "ring_score": 0.2,
            "arc_fraction": 0.1,
            "circularity": 0.3,
            "hollowness": 0.3,
            "has_curvature": False,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "cluster"

    def test_strip_takes_precedence_over_arc(self, ring_args):
        """When both strip and arc conditions met, strip wins (checked first)."""
        metrics = {
            "linearity": 3.5,
            "elongation": 3.5,
            "ring_score": 0.0,
            "arc_fraction": 0.5,
            "circularity": 0.3,
            "hollowness": 0.5,
            "has_curvature": True,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "strip"

    def test_ring_takes_precedence_over_arc(self, ring_args):
        """When both ring and arc conditions met, ring wins (checked first)."""
        metrics = {
            "linearity": 1.5,
            "elongation": 1.5,
            "ring_score": 0.7,
            "arc_fraction": 0.5,
            "circularity": 0.8,
            "hollowness": 0.7,
            "has_curvature": False,
        }
        assert classify_vessel_morphology(metrics, ring_args) == "ring"


# ---------------------------------------------------------------------------
# Vessel morphometry
# ---------------------------------------------------------------------------


class TestComputeVesselMorphometry:
    def test_ring_morphometry(self):
        # 20 points in a ring of radius 50
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        positions = np.column_stack([50 * np.cos(angles), 50 * np.sin(angles)])
        comp_indices = list(range(20))

        m = compute_vessel_morphometry(positions, comp_indices, "ring")
        assert m["vessel_diameter_um"] > 80  # ~100
        assert m["lumen_diameter_um"] > 0
        assert m["wall_extent_um"] >= 0
        assert m["size_class"] in ("medium", "large")  # p95 of radius ~50 → diameter ~95-100

    def test_strip_morphometry(self):
        positions = np.column_stack([np.linspace(0, 200, 20), np.zeros(20)])
        comp_indices = list(range(20))

        m = compute_vessel_morphometry(positions, comp_indices, "strip")
        assert "cell_count" in m
        assert m["cell_count"] == 20
        # Strip doesn't compute ring-specific metrics
        assert "vessel_diameter_um" not in m

    def test_small_component(self):
        positions = np.array([[0, 0], [1, 0], [0, 1]])
        m = compute_vessel_morphometry(positions, [0, 1, 2], "ring")
        assert m["cell_count"] == 3
        # Too few points for ring morphometry
        assert "vessel_diameter_um" not in m


# ---------------------------------------------------------------------------
# Marker composition
# ---------------------------------------------------------------------------


class TestAnalyzeMarkerComposition:
    def test_mixed_markers(self, sma_cd31_detections):
        indices = list(range(14))  # 8 SMA+ + 6 CD31+
        comp = analyze_marker_composition(
            sma_cd31_detections,
            indices,
            ["SMA", "CD31"],
            ["SMA_class", "CD31_class"],
        )
        assert comp["n_sma"] == 8
        assert comp["n_cd31"] == 6
        assert comp["sma_frac"] == pytest.approx(8 / 14, abs=0.01)
        assert comp["dominant_marker"] == "SMA"

    def test_single_marker(self):
        dets = [_make_detection({"LYVE1_class": "positive"}) for _ in range(5)]
        comp = analyze_marker_composition(dets, list(range(5)), ["LYVE1"], ["LYVE1_class"])
        assert comp["n_lyve1"] == 5
        assert comp["lyve1_frac"] == 1.0

    def test_double_positive_cells(self):
        """Cells positive for both SMA and CD31 are counted in BOTH markers."""
        dets = [
            _make_detection({"SMA_class": "positive", "CD31_class": "positive"}) for _ in range(10)
        ]
        comp = analyze_marker_composition(
            dets, list(range(10)), ["SMA", "CD31"], ["SMA_class", "CD31_class"]
        )
        assert comp["n_sma"] == 0  # single-positive SMA = 0
        assert comp["n_cd31"] == 0  # single-positive CD31 = 0
        assert comp["n_double_pos"] == 10  # all are double-positive
        assert comp["double_pos_frac"] == 1.0


# ---------------------------------------------------------------------------
# Spatial layering
# ---------------------------------------------------------------------------


class TestDetectSpatialLayeringReverse:
    def test_cd31_outer_sma_inner(self):
        """Reversed layering: CD31 outer, SMA inner → vein-like arrangement."""
        from detect_vessel_structures import detect_spatial_layering

        n = 20
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.zeros((n, 2))
        # SMA at inner radius 30, CD31 at outer radius 50 (opposite of artery)
        positions[:10] = np.column_stack([30 * np.cos(angles[:10]), 30 * np.sin(angles[:10])])
        positions[10:] = np.column_stack([50 * np.cos(angles[10:]), 50 * np.sin(angles[10:])])

        detections = []
        for i in range(n):
            feat = {
                "SMA_class": "positive" if i < 10 else "negative",
                "CD31_class": "positive" if i >= 10 else "negative",
            }
            detections.append(_make_detection(feat))

        layering = detect_spatial_layering(
            positions,
            detections,
            list(range(n)),
            list(range(n)),
            {"SMA": "SMA_class", "CD31": "CD31_class"},
        )
        # Should detect CD31 as outer (not SMA)
        assert any("CD31_outer" in k for k in layering)
        cd31_outer_key = [k for k in layering if "CD31_outer" in k][0]
        assert layering[cd31_outer_key]["significant"]
        assert layering[cd31_outer_key]["score"] > 0

    def test_layering_with_index_offset(self):
        """Verify layering works when local indices don't start at 0."""
        from detect_vessel_structures import detect_spatial_layering

        # 10 cells, local indices 5..14, global indices 100..109
        n = 10
        # Full positions array with 15 entries (local indices up to 14)
        positions = np.zeros((15, 2))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # SMA at r=50 (indices 5-9), CD31 at r=30 (indices 10-14)
        positions[5:10] = np.column_stack([50 * np.cos(angles[:5]), 50 * np.sin(angles[:5])])
        positions[10:15] = np.column_stack([30 * np.cos(angles[5:]), 30 * np.sin(angles[5:])])

        detections = [_make_detection() for _ in range(110)]
        for i in range(100, 105):
            detections[i]["features"]["SMA_class"] = "positive"
            detections[i]["features"]["CD31_class"] = "negative"
        for i in range(105, 110):
            detections[i]["features"]["SMA_class"] = "negative"
            detections[i]["features"]["CD31_class"] = "positive"

        layering = detect_spatial_layering(
            positions,
            detections,
            list(range(5, 15)),
            list(range(100, 110)),
            {"SMA": "SMA_class", "CD31": "CD31_class"},
        )
        # Should find SMA outer
        assert any("SMA_outer" in k for k in layering)


# ---------------------------------------------------------------------------
# Vessel type assignment
# ---------------------------------------------------------------------------


class TestAssignVesselType:
    def test_artery_thick_wall(self):
        """Thick SMA wall (wall_cell_layers > 1.5) + large → artery."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.6, "cd31_frac": 0.3, "lyve1_frac": 0.0, "n_cells": 30},
            {},
            {"vessel_diameter_um": 150, "wall_extent_um": 50, "wall_cell_layers": 3.0},
        )
        assert vt == "artery"

    def test_arteriole_thick_wall_small(self):
        """Thick wall but small diameter → arteriole."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.6, "cd31_frac": 0.3, "lyve1_frac": 0.0, "n_cells": 10},
            {},
            {"vessel_diameter_um": 40, "wall_extent_um": 15, "wall_cell_layers": 2.0},
        )
        assert vt == "arteriole"

    def test_artery_sma_dominant_no_morphometry(self):
        """SMA dominant but no morphometry → defaults to artery."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.6, "cd31_frac": 0.1, "lyve1_frac": 0.0, "n_cells": 5},
            {},
            {},
        )
        assert vt == "artery"

    def test_vein_thin_wall(self):
        """CD31 dominant, thin wall → vein."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.2, "cd31_frac": 0.7, "lyve1_frac": 0.0, "n_cells": 20},
            {},
            {"vessel_diameter_um": 80, "wall_extent_um": 10, "wall_cell_layers": 1.0},
        )
        assert vt == "vein"

    def test_venule_thin_wall_small(self):
        """CD31 dominant, thin wall, small → venule."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.1, "cd31_frac": 0.7, "lyve1_frac": 0.0, "n_cells": 8},
            {},
            {"vessel_diameter_um": 30, "wall_extent_um": 5, "wall_cell_layers": 0.8},
        )
        assert vt == "venule"

    def test_lymphatic(self):
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.0, "cd31_frac": 0.1, "lyve1_frac": 0.5, "n_cells": 15},
            {},
            {"vessel_diameter_um": 30},
        )
        assert vt == "lymphatic"

    def test_collecting_lymphatic(self):
        """LYVE1+ with SMA smooth muscle → collecting lymphatic."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.2, "cd31_frac": 0.0, "lyve1_frac": 0.5, "n_cells": 25},
            {},
            {"vessel_diameter_um": 80},
        )
        assert vt == "collecting_lymphatic"

    def test_capillary(self):
        vt = assign_vessel_type(
            "cluster",
            {"sma_frac": 0.0, "cd31_frac": 0.8, "lyve1_frac": 0.0, "n_cells": 8},
            {},
            {},
        )
        assert vt == "capillary"

    def test_strip_artery(self):
        vt = assign_vessel_type(
            "strip",
            {"sma_frac": 0.5, "cd31_frac": 0.3, "lyve1_frac": 0.0, "n_cells": 40},
            {},
            {},
        )
        assert vt == "artery_longitudinal"

    def test_unclassified(self):
        vt = assign_vessel_type(
            "cluster",
            {"sma_frac": 0.1, "cd31_frac": 0.1, "lyve1_frac": 0.0, "n_cells": 50},
            {},
            {},
        )
        assert vt == "unclassified"

    def test_lymphatic_longitudinal_reachable(self):
        """LYVE1-dominant strip → lymphatic_longitudinal, not just lymphatic."""
        vt = assign_vessel_type(
            "strip",
            {"sma_frac": 0.0, "cd31_frac": 0.0, "lyve1_frac": 0.8, "n_cells": 20},
            {},
            {},
        )
        assert vt == "lymphatic_longitudinal"

    def test_lyve1_does_not_override_sma_dominant(self):
        """Fig3 case: 60% SMA + 35% LYVE1 → artery, not lymphatic."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.6, "cd31_frac": 0.0, "lyve1_frac": 0.35, "n_cells": 30},
            {},
            {},
        )
        assert vt == "artery"

    def test_lyve1_dominant_over_sma(self):
        """LYVE1 dominant → lymphatic."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.1, "cd31_frac": 0.0, "lyve1_frac": 0.6, "n_cells": 20},
            {},
            {},
        )
        assert vt == "lymphatic"


# ---------------------------------------------------------------------------
# Spatial layering
# ---------------------------------------------------------------------------


class TestDetectSpatialLayering:
    def test_sma_outer_cd31_inner(self):
        """SMA cells at larger radii than CD31 → significant layering."""
        from detect_vessel_structures import detect_spatial_layering

        # 20 cells in a ring: SMA at radius ~50, CD31 at radius ~30
        n = 20
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.zeros((n, 2))
        # First 10: SMA at r=50, next 10: CD31 at r=30
        positions[:10] = np.column_stack([50 * np.cos(angles[:10]), 50 * np.sin(angles[:10])])
        positions[10:] = np.column_stack([30 * np.cos(angles[10:]), 30 * np.sin(angles[10:])])

        detections = []
        for i in range(n):
            feat = {
                "SMA_class": "positive" if i < 10 else "negative",
                "CD31_class": "positive" if i >= 10 else "negative",
            }
            detections.append(_make_detection(feat))

        layering = detect_spatial_layering(
            positions,
            detections,
            list(range(n)),
            list(range(n)),
            {"SMA": "SMA_class", "CD31": "CD31_class"},
        )
        assert "SMA_outer_vs_CD31" in layering
        assert layering["SMA_outer_vs_CD31"]["score"] > 0
        assert layering["SMA_outer_vs_CD31"]["significant"]


# ---------------------------------------------------------------------------
# Morphometry with global indices
# ---------------------------------------------------------------------------


class TestMorphometryGlobalIndices:
    def test_wall_cell_layers_uses_global_indices(self):
        """Verify wall_cell_layers reads area_um2 from correct detections (global)."""
        # 10 cells in a thick ring (r=40..60), global indices are 100-109
        rng = np.random.default_rng(42)
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        radii = rng.uniform(40, 60, 10)  # varied radii → wall_extent > 0
        positions = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Create 110 detections, only indices 100-109 have area_um2
        all_dets = [_make_detection() for _ in range(110)]
        for i in range(100, 110):
            all_dets[i]["features"]["area_um2"] = 100.0  # ~11µm diameter cells

        # Local indices 0-9, global indices 100-109
        comp_local = list(range(10))
        comp_global = list(range(100, 110))

        m = compute_vessel_morphometry(positions, comp_local, "ring", all_dets, comp_global)
        # Should compute wall_cell_layers from the correct cells (global 100-109)
        assert "wall_cell_layers" in m
        assert m["wall_cell_layers"] > 0


# ---------------------------------------------------------------------------
# Tag detections
# ---------------------------------------------------------------------------


class TestTagDetections:
    def test_tags_vessel_cells(self):
        dets = [_make_detection() for _ in range(10)]
        positive_idx = list(range(5))
        assignments = {
            0: {
                "vessel_id": 0,
                "vessel_type": "artery",
                "morphology": "ring",
                "size_class": "large",
            },
            1: {
                "vessel_id": 0,
                "vessel_type": "artery",
                "morphology": "ring",
                "size_class": "large",
            },
        }
        tag_detections(dets, positive_idx, assignments)

        assert dets[0]["features"]["vessel_type"] == "artery"
        assert dets[0]["features"]["vessel_id"] == 0
        assert dets[5]["features"]["vessel_type"] == "none"
        assert dets[5]["features"]["vessel_id"] == -1

    def test_handles_empty_assignments(self):
        dets = [_make_detection() for _ in range(5)]
        tag_detections(dets, [], {})
        assert all(d["features"]["vessel_id"] == -1 for d in dets)
