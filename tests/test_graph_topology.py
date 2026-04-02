"""Tests for xldvp_seg.utils.graph_topology — shared graph topology analysis."""

import numpy as np
import pytest

from xldvp_seg.utils.graph_topology import (
    arc_fraction,
    bounding_box_aspect_ratio,
    build_component_subgraph,
    build_radius_graph_sparse,
    circularity,
    component_linearity,
    component_width,
    compute_all_metrics,
    double_bfs_diameter,
    elongation,
    has_curvature,
    hollowness,
    path_length_um,
    ring_score,
    safe_hull_area,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable point patterns
# ---------------------------------------------------------------------------


@pytest.fixture
def ring_points():
    """20 points arranged in a circle of radius 50."""
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    return np.column_stack([50 * np.cos(angles), 50 * np.sin(angles)])


@pytest.fixture
def strip_points():
    """30 points in a thin strip along x-axis."""
    x = np.linspace(0, 300, 30)
    y = np.random.default_rng(42).uniform(-5, 5, 30)
    return np.column_stack([x, y])


@pytest.fixture
def blob_points():
    """40 points in a compact blob."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 20, (40, 2))


@pytest.fixture
def arc_points():
    """15 points in a half-circle arc."""
    angles = np.linspace(0, np.pi, 15)
    return np.column_stack([50 * np.cos(angles), 50 * np.sin(angles)])


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestBuildRadiusGraphSparse:
    def test_ring_single_component(self, ring_points):
        n_comp, labels, pairs = build_radius_graph_sparse(ring_points, radius=30.0)
        # All points close enough to form one component
        assert n_comp == 1 or len(set(labels)) == 1

    def test_disconnected_at_small_radius(self, ring_points):
        # Very small radius — no edges
        n_comp, labels, pairs = build_radius_graph_sparse(ring_points, radius=0.1)
        assert n_comp == len(ring_points)

    def test_returns_pairs(self, strip_points):
        n_comp, labels, pairs = build_radius_graph_sparse(strip_points, radius=20.0)
        assert isinstance(pairs, set)
        assert len(pairs) > 0

    def test_empty_positions(self):
        positions = np.empty((0, 2))
        n_comp, labels, pairs = build_radius_graph_sparse(positions, radius=10.0)
        assert n_comp == 0


class TestBuildComponentSubgraph:
    def test_builds_subgraph(self, strip_points):
        _, labels, pairs = build_radius_graph_sparse(strip_points, radius=20.0)
        comp_nodes = set(np.where(labels == 0)[0])
        G = build_component_subgraph(pairs, comp_nodes)
        assert len(G.nodes()) == len(comp_nodes)
        assert G.number_of_edges() > 0


# ---------------------------------------------------------------------------
# Strip metrics
# ---------------------------------------------------------------------------


class TestBuildRadiusGraphSparseEdgeCases:
    def test_single_node(self):
        positions = np.array([[5.0, 5.0]])
        n_comp, labels, pairs = build_radius_graph_sparse(positions, radius=10.0)
        assert n_comp == 1
        assert len(labels) == 1
        assert len(pairs) == 0

    def test_two_nodes_connected(self):
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        n_comp, labels, pairs = build_radius_graph_sparse(positions, radius=10.0)
        assert n_comp == 1
        assert len(pairs) == 1

    def test_two_nodes_disconnected(self):
        positions = np.array([[0.0, 0.0], [100.0, 0.0]])
        n_comp, labels, pairs = build_radius_graph_sparse(positions, radius=10.0)
        assert n_comp == 2


class TestDoubleBfsDiameter:
    def test_single_node(self):
        positions = np.array([[0.0, 0.0]])
        pairs = set()
        G = build_component_subgraph(pairs, {0})
        diam, path = double_bfs_diameter(G, {0})
        assert diam == 0
        assert path == [0]

    def test_two_nodes(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        pairs = {(0, 1)}
        G = build_component_subgraph(pairs, {0, 1})
        diam, path = double_bfs_diameter(G, {0, 1})
        assert diam == 1
        assert len(path) == 2

    def test_triangle(self):
        positions = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
        pairs = {(0, 1), (1, 2), (0, 2)}
        G = build_component_subgraph(pairs, {0, 1, 2})
        diam, path = double_bfs_diameter(G, {0, 1, 2})
        assert diam == 1  # all nodes 1 hop from each other

    def test_strip_has_large_diameter(self, strip_points):
        _, labels, pairs = build_radius_graph_sparse(strip_points, radius=20.0)
        comp_nodes = set(np.where(labels == labels[0])[0])
        G = build_component_subgraph(pairs, comp_nodes)
        diam, path = double_bfs_diameter(G, comp_nodes)
        # Strip of 30 points should have diameter > 10
        assert diam > 10
        assert len(path) == diam + 1

    def test_blob_has_small_diameter(self, blob_points):
        _, labels, pairs = build_radius_graph_sparse(blob_points, radius=30.0)
        comp_nodes = set(np.where(labels == labels[0])[0])
        G = build_component_subgraph(pairs, comp_nodes)
        diam, path = double_bfs_diameter(G, comp_nodes)
        # Blob should have short diameter
        assert diam < 10


class TestComponentLinearity:
    def test_strip_is_linear(self):
        # diameter=20, n=25 → linearity = 20/5 = 4.0
        assert component_linearity(20, 25) == pytest.approx(4.0)

    def test_blob_is_compact(self):
        # diameter=4, n=25 → linearity = 4/5 = 0.8
        assert component_linearity(4, 25) == pytest.approx(0.8)

    def test_zero_nodes(self):
        assert component_linearity(5, 0) == 0.0


class TestComponentWidth:
    def test_collinear_zero_width(self):
        positions = np.array([[i, 0.0] for i in range(10)])
        path = list(range(10))
        w = component_width(positions, list(range(10)), path)
        assert w == pytest.approx(0.0, abs=1e-6)

    def test_known_offset(self):
        positions = np.array([[0, 0], [10, 0], [20, 0], [10, 5.0]])
        path = [0, 1, 2]
        w = component_width(positions, [0, 1, 2, 3], path, percentile=100)
        assert w == pytest.approx(5.0, abs=0.1)


class TestComponentWidthEdgeCases:
    def test_single_segment_path(self):
        positions = np.array([[0, 0], [10, 0], [5, 3.0]])
        w = component_width(positions, [0, 1, 2], [0, 1])
        assert w > 0

    def test_empty_path(self):
        positions = np.array([[0, 0], [10, 0]])
        assert component_width(positions, [0, 1], []) == 0.0

    def test_single_node_path(self):
        positions = np.array([[0, 0], [10, 0]])
        assert component_width(positions, [0, 1], [0]) == 0.0


class TestPathLengthUm:
    def test_empty_path(self):
        positions = np.array([[0, 0], [10, 0]])
        assert path_length_um(positions, []) == 0.0

    def test_single_node_path(self):
        positions = np.array([[0, 0]])
        assert path_length_um(positions, [0]) == 0.0

    def test_straight_path(self):
        positions = np.array([[0, 0], [10, 0], [20, 0]])
        length = path_length_um(positions, [0, 1, 2])
        assert length == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Ring / arc metrics — graph topology
# ---------------------------------------------------------------------------


class TestRingScore:
    def test_ring_high_score(self, ring_points):
        _, labels, pairs = build_radius_graph_sparse(ring_points, radius=25.0)
        comp_nodes = set(range(len(ring_points)))
        G = build_component_subgraph(pairs, comp_nodes)
        rs = ring_score(ring_points, G, comp_nodes)
        # Perfect ring should have very high score
        assert rs > 0.6

    def test_strip_low_score(self, strip_points):
        _, labels, pairs = build_radius_graph_sparse(strip_points, radius=20.0)
        comp_nodes = set(range(len(strip_points)))
        G = build_component_subgraph(pairs, comp_nodes)
        rs = ring_score(strip_points, G, comp_nodes)
        # Strip should have low ring score (or 0 if AR guard triggers)
        assert rs < 0.4

    def test_returns_zero_for_elongated(self):
        # Very elongated: AR > 3 → ring_score should be 0
        positions = np.array([[i, 0.0] for i in range(20)])
        pairs = {(i, i + 1) for i in range(19)}
        G = build_component_subgraph(pairs, set(range(20)))
        assert ring_score(positions, G, set(range(20))) == 0.0

    def test_small_component(self):
        positions = np.array([[0, 0], [1, 0], [0, 1]])
        assert ring_score(positions, None, set(range(3))) == 0.0


class TestArcFraction:
    def test_full_ring(self, ring_points):
        _, labels, pairs = build_radius_graph_sparse(ring_points, radius=25.0)
        comp_nodes = set(range(len(ring_points)))
        G = build_component_subgraph(pairs, comp_nodes)
        af = arc_fraction(ring_points, G, comp_nodes)
        assert af > 0.7  # should be close to 1.0

    def test_half_ring(self, arc_points):
        _, labels, pairs = build_radius_graph_sparse(arc_points, radius=25.0)
        comp_nodes = set(range(len(arc_points)))
        G = build_component_subgraph(pairs, comp_nodes)
        af = arc_fraction(arc_points, G, comp_nodes)
        # Half circle — arc_fraction should be less than a full ring
        # but still substantial (contiguous arc covers most of the angular span)
        assert 0.2 < af <= 1.0

    def test_small_component(self):
        positions = np.array([[0, 0], [1, 0], [0, 1]])
        assert arc_fraction(positions, None, set(range(3))) == 0.0


# ---------------------------------------------------------------------------
# Ring / arc metrics — geometric / PCA
# ---------------------------------------------------------------------------


class TestElongation:
    def test_strip_high_elongation(self, strip_points):
        assert elongation(strip_points) > 3.0

    def test_ring_low_elongation(self, ring_points):
        assert elongation(ring_points) < 2.0

    def test_small_set(self):
        assert elongation(np.array([[0, 0], [1, 1]])) == 1.0


class TestCircularity:
    def test_ring_high_circularity(self, ring_points):
        assert circularity(ring_points) > 0.6

    def test_blob_lower_circularity(self, blob_points, ring_points):
        # Random blob has variable distances — lower circularity than a ring
        assert circularity(blob_points) < circularity(ring_points)


class TestHollowness:
    def test_ring_high_hollowness(self, ring_points):
        # Ring: all points at same radius → median ≈ max → hollowness ≈ 1.0
        assert hollowness(ring_points) > 0.8

    def test_blob_moderate_hollowness(self, blob_points):
        # Random Gaussian blob has median < max
        h = hollowness(blob_points)
        assert 0.3 < h < 0.9


class TestHasCurvature:
    def test_curved_strip(self):
        # Parabolic curve y = x^2/100
        x = np.linspace(-50, 50, 30)
        y = x**2 / 100
        pts = np.column_stack([x, y])
        assert has_curvature(pts) is True

    def test_straight_strip(self, strip_points):
        # Nearly straight strip with tiny y-noise — should NOT detect curvature
        assert has_curvature(strip_points) is False

    def test_compact_blob_no_curvature(self, blob_points):
        assert has_curvature(blob_points) is False

    def test_fewer_than_10_points(self):
        pts = np.array([[i, 0.0] for i in range(9)])
        assert has_curvature(pts) is False


# ---------------------------------------------------------------------------
# Degenerate geometry — all points at same position
# ---------------------------------------------------------------------------


class TestDegenerateGeometry:
    """All points at the same position — should not crash."""

    def test_elongation_same_position(self):
        pts = np.full((10, 2), 5.0)
        e = elongation(pts)
        assert np.isfinite(e)

    def test_circularity_same_position(self):
        pts = np.full((10, 2), 5.0)
        assert circularity(pts) == 0.0  # mean_r ~ 0

    def test_hollowness_same_position(self):
        pts = np.full((10, 2), 5.0)
        assert hollowness(pts) == 0.0  # max_r ~ 0

    def test_safe_hull_area_same_position(self):
        pts = np.full((10, 2), 5.0)
        assert safe_hull_area(pts) == 0.0

    def test_bounding_box_ar_same_position(self):
        pts = np.full((10, 2), 5.0)
        ar = bounding_box_aspect_ratio(pts)
        assert np.isfinite(ar)


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


class TestSafeHullArea:
    def test_known_square(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert safe_hull_area(pts) == pytest.approx(100.0)

    def test_degenerate(self):
        assert safe_hull_area(np.array([[0, 0], [1, 1]])) == 0.0

    def test_collinear(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]])
        assert safe_hull_area(pts) == 0.0


class TestBoundingBoxAspectRatio:
    def test_square_ar_one(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        ar = bounding_box_aspect_ratio(pts)
        assert ar == pytest.approx(1.0, abs=0.3)

    def test_elongated_high_ar(self, strip_points):
        ar = bounding_box_aspect_ratio(strip_points)
        assert ar > 3.0


# ---------------------------------------------------------------------------
# Convenience: compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_ring_metrics(self, ring_points):
        _, labels, pairs = build_radius_graph_sparse(ring_points, radius=25.0)
        comp_nodes = set(range(len(ring_points)))
        G = build_component_subgraph(pairs, comp_nodes)
        m = compute_all_metrics(ring_points, G, comp_nodes)

        assert m["n_cells"] == 20
        assert m["ring_score"] > 0.5
        assert m["hollowness"] > 0.8
        assert m["linearity"] < 3.0  # ring has moderate linearity (not strip-level)
        assert "diameter" in m
        assert "hull_area_um2" in m

    def test_strip_metrics(self, strip_points):
        _, labels, pairs = build_radius_graph_sparse(strip_points, radius=20.0)
        comp_nodes = set(np.where(labels == labels[0])[0])
        G = build_component_subgraph(pairs, comp_nodes)
        m = compute_all_metrics(strip_points, G, comp_nodes)

        assert m["linearity"] > 2.0
        assert m["elongation"] > 3.0
        assert m["length_um"] > 100
