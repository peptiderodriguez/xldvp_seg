"""Tests for xldvp_seg.analysis.vessel_characterization.

Covers:
- analyze_marker_composition with various marker configurations
- assign_vessel_type decision tree
"""

import pytest

from xldvp_seg.analysis.vessel_characterization import (
    analyze_marker_composition,
    assign_vessel_type,
)


def _make_detection(features=None):
    """Create a minimal detection dict."""
    return {"features": features or {}}


class TestAnalyzeMarkerComposition:
    def test_sma_dominant(self):
        """Mostly SMA+ cells -> dominant_marker=SMA."""
        dets = []
        for i in range(10):
            if i < 7:
                dets.append(_make_detection({"SMA_class": "positive", "CD31_class": "negative"}))
            else:
                dets.append(_make_detection({"SMA_class": "negative", "CD31_class": "positive"}))

        comp = analyze_marker_composition(
            dets, list(range(10)), ["SMA", "CD31"], ["SMA_class", "CD31_class"]
        )
        assert comp["n_cells"] == 10
        assert comp["n_sma"] == 7
        assert comp["n_cd31"] == 3
        assert comp["dominant_marker"] == "SMA"
        assert comp["sma_frac"] == pytest.approx(0.7, abs=0.01)
        assert comp["n_double_pos"] == 0

    def test_cd31_dominant(self):
        """Mostly CD31+ cells."""
        dets = [
            _make_detection({"SMA_class": "negative", "CD31_class": "positive"}) for _ in range(10)
        ]
        comp = analyze_marker_composition(
            dets, list(range(10)), ["SMA", "CD31"], ["SMA_class", "CD31_class"]
        )
        assert comp["dominant_marker"] == "CD31"
        assert comp["n_cd31"] == 10

    def test_double_positive(self):
        """All cells positive for both markers -> n_double_pos=N, single=0."""
        dets = [
            _make_detection({"SMA_class": "positive", "CD31_class": "positive"}) for _ in range(5)
        ]
        comp = analyze_marker_composition(
            dets, list(range(5)), ["SMA", "CD31"], ["SMA_class", "CD31_class"]
        )
        assert comp["n_double_pos"] == 5
        assert comp["n_sma"] == 0  # single-positive
        assert comp["n_cd31"] == 0
        assert comp["sma_total_frac"] == 1.0
        assert comp["cd31_total_frac"] == 1.0

    def test_empty_cell_list(self):
        """Empty component -> n_cells=0, no crash."""
        comp = analyze_marker_composition([], [], ["SMA"], ["SMA_class"])
        assert comp["n_cells"] == 0
        assert comp["n_sma"] == 0

    def test_all_negative(self):
        """All cells negative for all markers."""
        dets = [_make_detection({"SMA_class": "negative"}) for _ in range(5)]
        comp = analyze_marker_composition(dets, list(range(5)), ["SMA"], ["SMA_class"])
        assert comp["n_sma"] == 0
        assert comp["sma_frac"] == 0.0
        assert comp["dominant_marker"] == "none"

    def test_three_markers(self):
        """Three markers including LYVE1."""
        dets = [
            _make_detection(
                {"SMA_class": "positive", "CD31_class": "negative", "LYVE1_class": "negative"}
            ),
            _make_detection(
                {"SMA_class": "negative", "CD31_class": "positive", "LYVE1_class": "negative"}
            ),
            _make_detection(
                {"SMA_class": "negative", "CD31_class": "negative", "LYVE1_class": "positive"}
            ),
        ]
        comp = analyze_marker_composition(
            dets,
            list(range(3)),
            ["SMA", "CD31", "LYVE1"],
            ["SMA_class", "CD31_class", "LYVE1_class"],
        )
        assert comp["n_cells"] == 3
        assert comp["n_sma"] == 1
        assert comp["n_cd31"] == 1
        assert comp["n_lyve1"] == 1
        assert comp["n_double_pos"] == 0


class TestAssignVesselType:
    def test_artery_thick_wall(self):
        """SMA ring with thick wall -> artery."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.6, "cd31_frac": 0.2, "lyve1_frac": 0.0, "n_cells": 30},
            {},
            {"vessel_diameter_um": 150, "wall_extent_um": 50, "wall_cell_layers": 3.0},
        )
        assert vt == "artery"

    def test_arteriole_small(self):
        """SMA ring, thick wall, small -> arteriole."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.5, "cd31_frac": 0.2, "lyve1_frac": 0.0, "n_cells": 10},
            {},
            {"vessel_diameter_um": 50, "wall_extent_um": 15, "wall_cell_layers": 2.0},
        )
        assert vt == "arteriole"

    def test_vein_thin_wall(self):
        """CD31 dominant, thin wall -> vein."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.1, "cd31_frac": 0.7, "lyve1_frac": 0.0, "n_cells": 20},
            {},
            {"vessel_diameter_um": 80, "wall_extent_um": 8, "wall_cell_layers": 0.8},
        )
        assert vt == "vein"

    def test_venule(self):
        """CD31 dominant, thin wall, small -> venule."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.05, "cd31_frac": 0.8, "lyve1_frac": 0.0, "n_cells": 8},
            {},
            {"vessel_diameter_um": 25, "wall_extent_um": 3, "wall_cell_layers": 0.5},
        )
        assert vt == "venule"

    def test_lymphatic(self):
        """LYVE1 dominant -> lymphatic."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.05, "cd31_frac": 0.05, "lyve1_frac": 0.6, "n_cells": 15},
            {},
            {"vessel_diameter_um": 40},
        )
        assert vt == "lymphatic"

    def test_collecting_lymphatic(self):
        """LYVE1 dominant + SMA -> collecting_lymphatic."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.2, "cd31_frac": 0.0, "lyve1_frac": 0.5, "n_cells": 20},
            {},
            {"vessel_diameter_um": 60},
        )
        assert vt == "collecting_lymphatic"

    def test_capillary(self):
        """CD31 cluster with few cells -> capillary."""
        vt = assign_vessel_type(
            "cluster",
            {"sma_frac": 0.0, "cd31_frac": 0.8, "lyve1_frac": 0.0, "n_cells": 8},
            {},
            {},
        )
        assert vt == "capillary"

    def test_strip_artery_longitudinal(self):
        """SMA dominant strip -> artery_longitudinal."""
        vt = assign_vessel_type(
            "strip",
            {"sma_frac": 0.6, "cd31_frac": 0.1, "lyve1_frac": 0.0, "n_cells": 30},
            {},
            {},
        )
        assert vt == "artery_longitudinal"

    def test_strip_vein_longitudinal(self):
        """CD31 dominant strip -> vein_longitudinal."""
        vt = assign_vessel_type(
            "strip",
            {"sma_frac": 0.1, "cd31_frac": 0.6, "lyve1_frac": 0.0, "n_cells": 30},
            {},
            {},
        )
        assert vt == "vein_longitudinal"

    def test_unclassified_low_markers(self):
        """Low marker expression -> unclassified."""
        vt = assign_vessel_type(
            "ring",
            {"sma_frac": 0.05, "cd31_frac": 0.05, "lyve1_frac": 0.0, "n_cells": 20},
            {},
            {"vessel_diameter_um": 50, "wall_cell_layers": 1.0},
        )
        assert vt == "unclassified"

    def test_uses_total_frac_when_available(self):
        """Should prefer _total_frac keys over _frac keys."""
        vt = assign_vessel_type(
            "ring",
            {
                "sma_frac": 0.1,
                "cd31_frac": 0.1,
                "lyve1_frac": 0.0,
                "sma_total_frac": 0.7,
                "cd31_total_frac": 0.3,
                "lyve1_total_frac": 0.0,
                "n_cells": 20,
            },
            {},
            {"vessel_diameter_um": 150, "wall_extent_um": 50, "wall_cell_layers": 3.0},
        )
        assert vt == "artery"

    def test_strip_lymphatic_longitudinal(self):
        """LYVE1 dominant strip -> lymphatic_longitudinal."""
        vt = assign_vessel_type(
            "strip",
            {"sma_frac": 0.0, "cd31_frac": 0.0, "lyve1_frac": 0.6, "n_cells": 15},
            {},
            {},
        )
        assert vt == "lymphatic_longitudinal"

    def test_unknown_morphology(self):
        """Unknown morphology string -> unclassified."""
        vt = assign_vessel_type(
            "unknown",
            {"sma_frac": 0.5, "cd31_frac": 0.5, "lyve1_frac": 0.0, "n_cells": 20},
            {},
            {},
        )
        assert vt == "unclassified"
