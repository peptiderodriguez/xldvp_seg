"""Tests for curvilinear pattern detection (detect_curvilinear_patterns.py).

Tests cover:
- Component classification (strip vs blob via graph diameter linearity)
- Positive cell selection (SNR and marker-filter paths)
- Component width calculation (vectorized perpendicular distances)
- Per-cell refinement methods (betweenness, degree_ratio, k_core)
- Edge cases (empty input, single cell, all noise)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from xldvp_seg.analysis.pattern_detection import (
    classify_components,
    refine_strip_cells,
    select_positive_cells,
)
from xldvp_seg.utils.graph_topology import component_width

# Ensure scripts/ is importable for functions that remain in the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from detect_curvilinear_patterns import (
    extract_aligned_positions,
    tag_detections,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strip_positions():
    """30 cells in a line, spaced 25µm apart."""
    return np.array([[i * 25, 0] for i in range(30)], dtype=np.float64)


@pytest.fixture
def blob_positions():
    """30 cells in a tight cluster (circle, radius 40µm)."""
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    return np.column_stack([np.cos(angles) * 40, np.sin(angles) * 40]).astype(np.float64)


@pytest.fixture
def curved_strip_positions():
    """50 cells along a semicircle (radius 250µm, ~16µm spacing)."""
    angles = np.linspace(0, np.pi, 50)
    np.random.seed(42)
    pos = np.column_stack([np.cos(angles) * 250, np.sin(angles) * 250]).astype(np.float64)
    pos += np.random.normal(0, 5, pos.shape)
    return pos


@pytest.fixture
def sample_detections():
    """Minimal detection dicts with features for testing."""
    dets = []
    for i in range(20):
        dets.append(
            {
                "uid": f"cell_{i}",
                "global_center": [i * 50, 100],
                "features": {
                    "ch2_snr": 2.0 if i < 10 else 0.5,
                    "area": 500,
                    "area_um2": 15.0,
                    "MSLN_class": "positive" if i < 10 else "negative",
                },
            }
        )
    return dets


# ---------------------------------------------------------------------------
# classify_components tests
# ---------------------------------------------------------------------------


class TestClassifyComponents:
    def test_strip_has_high_linearity(self, strip_positions):
        """A linear arrangement should be classified as strip."""
        labels, stats, G = classify_components(
            strip_positions, radius=50, min_component_size=5, linearity_threshold=2.0
        )
        assert "strip" in labels
        strip_count = sum(1 for l in labels if l == "strip")
        assert strip_count == 30

    def test_blob_has_low_linearity(self, blob_positions):
        """A circular cluster should be classified as cluster, not strip."""
        labels, stats, G = classify_components(
            blob_positions, radius=100, min_component_size=5, linearity_threshold=2.0
        )
        strip_count = sum(1 for l in labels if l == "strip")
        assert strip_count == 0
        cluster_count = sum(1 for l in labels if l == "cluster")
        assert cluster_count == 30

    def test_curved_strip_detected(self, curved_strip_positions):
        """A curved semicircle should still be detected as a strip."""
        labels, stats, G = classify_components(
            curved_strip_positions,
            radius=50,
            min_component_size=5,
            linearity_threshold=2.0,
        )
        strip_count = sum(1 for l in labels if l == "strip")
        # Most cells should be in a strip (some at edges may fragment)
        assert strip_count > 30

    def test_small_components_are_noise(self, strip_positions):
        """Components below min_component_size should be labeled noise."""
        # Use a tiny radius so each cell is isolated
        labels, stats, G = classify_components(
            strip_positions, radius=1, min_component_size=5, linearity_threshold=2.0
        )
        assert all(l == "noise" for l in labels)

    def test_min_strip_cells_filter(self, strip_positions):
        """Components passing linearity but below min_strip_cells → cluster."""
        labels, stats, G = classify_components(
            strip_positions,
            radius=50,
            min_component_size=5,
            linearity_threshold=2.0,
            min_strip_cells=50,  # more than our 30 cells
        )
        # Should be cluster, not strip (too few cells)
        strip_count = sum(1 for l in labels if l == "strip")
        assert strip_count == 0

    def test_min_strip_length_filter(self, strip_positions):
        """Components below min_strip_length → cluster."""
        labels, stats, G = classify_components(
            strip_positions,
            radius=50,
            min_component_size=5,
            linearity_threshold=2.0,
            min_strip_length=10000,  # 10mm, way longer than our strip
        )
        strip_count = sum(1 for l in labels if l == "strip")
        assert strip_count == 0

    def test_max_strip_width_filter(self):
        """Wide components should be excluded by max_strip_width."""
        # 3-cell-wide strip
        positions = np.array(
            [[i * 25, j * 25] for i in range(20) for j in range(-1, 2)],
            dtype=np.float64,
        )
        labels, stats, G = classify_components(
            positions,
            radius=50,
            min_component_size=5,
            linearity_threshold=2.0,
            max_strip_width=10,  # very narrow — should exclude this strip
        )
        strip_count = sum(1 for l in labels if l == "strip")
        assert strip_count == 0

    def test_returns_graph(self, strip_positions):
        """classify_components should return the networkx graph."""
        labels, stats, G = classify_components(
            strip_positions, radius=50, min_component_size=5, linearity_threshold=2.0
        )
        assert G.number_of_nodes() == 30
        assert G.number_of_edges() > 0

    def test_component_stats_have_length_and_width(self, strip_positions):
        """Component stats should include length_um and width_um."""
        labels, stats, G = classify_components(
            strip_positions, radius=50, min_component_size=5, linearity_threshold=2.0
        )
        assert len(stats) > 0
        for s in stats:
            assert "length_um" in s
            assert "width_um" in s
            assert "linearity" in s
            assert "diameter" in s
            assert s["length_um"] > 0

    def test_empty_positions(self):
        """Empty input should return empty labels."""
        positions = np.empty((0, 2), dtype=np.float64)
        labels, stats, G = classify_components(
            positions, radius=50, min_component_size=5, linearity_threshold=2.0
        )
        assert labels == []
        assert stats == []


# ---------------------------------------------------------------------------
# component_width tests
# ---------------------------------------------------------------------------


class TestComponentWidth:
    def test_zero_width_for_collinear_points(self):
        """Points exactly on the path should have width 0."""
        positions = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float64)
        w = component_width(positions, [0, 1, 2, 3], [0, 1, 2, 3])
        assert w == pytest.approx(0.0, abs=0.1)

    def test_known_perpendicular_offset(self):
        """Points offset by known distance from path."""
        positions = np.array(
            [[0, 0], [10, 0], [20, 0], [5, 30], [15, 30]],
            dtype=np.float64,
        )
        path = [0, 1, 2]  # x-axis
        w = component_width(positions, [0, 1, 2, 3, 4], path)
        assert w == pytest.approx(30.0, abs=1.0)

    def test_short_path(self):
        """Path with < 2 nodes should return 0."""
        positions = np.array([[0, 0], [10, 0]], dtype=np.float64)
        assert component_width(positions, [0, 1], [0]) == 0.0
        assert component_width(positions, [0, 1], []) == 0.0

    def test_robust_to_outlier(self):
        """95th percentile width should be robust to single outlier."""
        # 19 points on the line + 1 outlier at 100µm offset
        positions = np.array(
            [[i * 10, 0] for i in range(19)] + [[100, 100]],
            dtype=np.float64,
        )
        path = list(range(19))
        w = component_width(positions, list(range(20)), path)
        # 95th percentile should be much less than 100 (the outlier)
        assert w < 100


# ---------------------------------------------------------------------------
# select_positive_cells tests
# ---------------------------------------------------------------------------


class TestSelectPositiveCells:
    def test_snr_filter(self, sample_detections):
        idx = select_positive_cells(sample_detections, snr_channel=2, snr_threshold=1.5)
        assert len(idx) == 10  # first 10 have snr=2.0

    def test_marker_filter_string(self, sample_detections):
        idx = select_positive_cells(sample_detections, marker_filter="MSLN_class==positive")
        assert len(idx) == 10

    def test_marker_filter_boolean(self):
        """Boolean True/False values should be matched by marker filter."""
        dets = [
            {"features": {"passed": True}},
            {"features": {"passed": False}},
            {"features": {"passed": True}},
        ]
        idx = select_positive_cells(dets, marker_filter="passed==True")
        assert len(idx) == 2
        assert idx == [0, 2]

    def test_snr_channel_zero(self):
        """SNR channel 0 should work (not treated as falsy)."""
        dets = [{"features": {"ch0_snr": 2.0}}, {"features": {"ch0_snr": 0.5}}]
        idx = select_positive_cells(dets, snr_channel=0, snr_threshold=1.0)
        assert len(idx) == 1

    def test_malformed_marker_filter(self):
        """Marker filter without == should raise ValueError."""
        with pytest.raises(ValueError):
            select_positive_cells([], marker_filter="MSLN_class:positive")


# ---------------------------------------------------------------------------
# refine_strip_cells tests
# ---------------------------------------------------------------------------


class TestRefineStripCells:
    @pytest.fixture
    def strip_with_hangers(self):
        """Strip (30 cells, 3 wide) + 10 hanger-on cells offset perpendicular."""
        np.random.seed(42)
        core = np.array(
            [[i * 25, j * 20] for i in range(30) for j in range(-1, 2)],
            dtype=np.float64,
        )
        core += np.random.normal(0, 3, core.shape)
        hangers = np.array(
            [[i * 80, 80 + np.random.uniform(0, 20)] for i in range(10)],
            dtype=np.float64,
        )
        positions = np.vstack([core, hangers])
        labels = ["strip"] * len(positions)
        return positions, labels

    def test_degree_ratio(self, strip_with_hangers):
        positions, labels = strip_with_hangers
        result = refine_strip_cells(
            positions, labels, radius=50, method="degree_ratio", threshold=0.5
        )
        # Hangers should be demoted
        hanger_labels = result[len(positions) - 10 :]
        demoted = sum(1 for l in hanger_labels if l == "cluster")
        assert demoted >= 5  # most hangers should be demoted

    def test_k_core(self, strip_with_hangers):
        positions, labels = strip_with_hangers
        result = refine_strip_cells(positions, labels, radius=50, method="k_core", threshold=2)
        # Some hangers should be demoted
        hanger_labels = result[len(positions) - 10 :]
        demoted = sum(1 for l in hanger_labels if l == "cluster")
        assert demoted >= 5

    def test_betweenness(self, strip_with_hangers):
        positions, labels = strip_with_hangers
        result = refine_strip_cells(
            positions, labels, radius=50, method="betweenness", threshold=20
        )
        total_demoted = sum(1 for l in result if l == "cluster")
        assert total_demoted > 0

    def test_no_strip_cells(self):
        """If no strip cells, should return labels unchanged."""
        positions = np.array([[0, 0], [10, 0]], dtype=np.float64)
        labels = ["cluster", "noise"]
        result = refine_strip_cells(
            positions, labels, radius=50, method="degree_ratio", threshold=0.5
        )
        assert result == labels

    def test_reuses_provided_graph(self, strip_with_hangers):
        """When G_all is provided, should not rebuild the graph."""
        import networkx as nx
        from scipy.spatial import KDTree

        positions, labels = strip_with_hangers
        tree = KDTree(positions)
        pairs = tree.query_pairs(r=50)
        G = nx.Graph()
        G.add_nodes_from(range(len(positions)))
        G.add_edges_from(pairs)

        # Pass the graph — should work without rebuilding
        result = refine_strip_cells(
            positions, labels, radius=50, method="degree_ratio", threshold=0.5, G_all=G
        )
        assert len(result) == len(labels)


# ---------------------------------------------------------------------------
# tag_detections tests
# ---------------------------------------------------------------------------


class TestTagDetections:
    def test_tags_all_detections(self, sample_detections):
        positive_idx = list(range(10))
        labels = ["strip"] * 5 + ["cluster"] * 3 + ["noise"] * 2
        field = tag_detections(sample_detections, positive_idx, labels, "msln")
        assert field == "msln_pattern"
        # Check tagged values
        for i in range(5):
            assert sample_detections[i]["features"]["msln_pattern"] == "strip"
        for i in range(5, 8):
            assert sample_detections[i]["features"]["msln_pattern"] == "cluster"
        for i in range(10, 20):
            assert sample_detections[i]["features"]["msln_pattern"] == "other"

    def test_handles_missing_features_dict(self):
        """Detections without features dict should get one created."""
        dets = [{"uid": "test"}]
        tag_detections(dets, [], [], "test")
        assert "features" in dets[0]
        assert dets[0]["features"]["test_pattern"] == "other"


# ---------------------------------------------------------------------------
# extract_aligned_positions tests
# ---------------------------------------------------------------------------


class TestExtractAlignedPositions:
    def test_basic_extraction(self, sample_detections):
        from xldvp_seg.utils.detection_utils import extract_positions_um

        positive_idx = list(range(10))
        valid_idx, positions = extract_aligned_positions(
            sample_detections, positive_idx, extract_positions_um
        )
        assert len(valid_idx) == len(positions)
        assert len(valid_idx) <= 10

    def test_skips_unresolvable(self):
        """Detections without coordinates should be skipped."""
        from xldvp_seg.utils.detection_utils import extract_positions_um

        dets = [
            {"features": {"area": 100, "area_um2": 3.0}, "global_center": [10, 20]},
            {"features": {"area": 100, "area_um2": 3.0}},  # no position
        ]
        valid_idx, positions = extract_aligned_positions(dets, [0, 1], extract_positions_um)
        assert len(valid_idx) == 1
        assert valid_idx[0] == 0
