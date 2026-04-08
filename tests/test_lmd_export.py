"""Tests for xldvp_seg.lmd.export -- LMD export pure-logic functions.

Covers detection filtering, annotation loading, spatial ordering,
coordinate extraction, spatial control generation, and well assignment.

Run with: pytest tests/test_lmd_export.py -v
"""

import json

import numpy as np
import pytest

from xldvp_seg.lmd.export import (
    assign_wells_with_controls,
    filter_detections,
    generate_spatial_control,
    get_detection_coordinates,
    load_annotations,
    nearest_neighbor_order,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detections(n=5, with_scores=False, with_uids=True):
    """Create sample detection dicts for testing.

    Args:
        n: Number of detections.
        with_scores: If True, add rf_prediction to each detection.
        with_uids: If True, add uid to each detection.

    Returns:
        List of detection dicts.
    """
    dets = []
    for i in range(n):
        det = {
            "global_center": [i * 100 + 50, i * 200 + 50],
            "features": {"area": 500 + i * 100},
        }
        if with_uids:
            det["uid"] = f"cell_{i}"
        if with_scores:
            det["rf_prediction"] = 0.1 * i  # 0.0, 0.1, 0.2, 0.3, 0.4
        dets.append(det)
    return dets


def _write_json(path, data):
    """Write data as JSON file."""
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# filter_detections
# ---------------------------------------------------------------------------


class TestFilterDetections:
    """Tests for filter_detections by UID and/or score."""

    def test_filter_by_positive_uids(self):
        """Only detections whose uid is in positive_uids are kept."""
        dets = _make_detections(5)
        positive = {"cell_0", "cell_2", "cell_4"}

        result = filter_detections(dets, positive_uids=positive)

        assert len(result) == 3
        uids = {d["uid"] for d in result}
        assert uids == positive

    def test_filter_by_min_score(self):
        """Only detections with rf_prediction >= min_score are kept."""
        dets = _make_detections(5, with_scores=True)

        result = filter_detections(dets, min_score=0.25)

        # Scores: 0.0, 0.1, 0.2, 0.3, 0.4 -- only 0.3, 0.4 pass
        assert len(result) == 2
        scores = [d["rf_prediction"] for d in result]
        assert all(s >= 0.25 for s in scores)

    def test_combined_uid_and_score_filter(self):
        """Both uid and score filters are applied together."""
        dets = _make_detections(5, with_scores=True)
        positive = {"cell_0", "cell_3", "cell_4"}

        result = filter_detections(dets, positive_uids=positive, min_score=0.25)

        # cell_0 has score 0.0 (fails), cell_3 has 0.3 (passes), cell_4 has 0.4 (passes)
        assert len(result) == 2
        uids = {d["uid"] for d in result}
        assert uids == {"cell_3", "cell_4"}

    def test_empty_result(self):
        """Filters that match nothing return empty list."""
        dets = _make_detections(5)
        result = filter_detections(dets, positive_uids={"nonexistent_uid"})
        assert result == []

    def test_no_filters_returns_all(self):
        """No filters applied returns all detections."""
        dets = _make_detections(5)
        result = filter_detections(dets)
        assert len(result) == 5

    def test_score_fallback_to_score_key(self):
        """When rf_prediction is absent, falls back to 'score' key."""
        dets = [{"uid": "a", "score": 0.8}, {"uid": "b", "score": 0.3}]

        result = filter_detections(dets, min_score=0.5)

        assert len(result) == 1
        assert result[0]["uid"] == "a"

    def test_none_score_treated_as_zero(self):
        """Detections with score=None are treated as score=0."""
        dets = [{"uid": "a", "score": None}]

        result = filter_detections(dets, min_score=0.1)

        assert len(result) == 0

    def test_id_fallback_when_no_uid(self):
        """When uid is absent, 'id' is used instead."""
        dets = [{"id": "cell_x"}, {"id": "cell_y"}]
        positive = {"cell_x"}

        result = filter_detections(dets, positive_uids=positive)

        assert len(result) == 1
        assert result[0]["id"] == "cell_x"


# ---------------------------------------------------------------------------
# load_annotations
# ---------------------------------------------------------------------------


class TestLoadAnnotations:
    """Tests for load_annotations -- 3 annotation formats."""

    def test_positive_negative_format(self, tmp_path):
        """Format: {"positive": [...], "negative": [...]}."""
        ann = {"positive": ["uid_1", "uid_2"], "negative": ["uid_3"]}
        ann_path = tmp_path / "annotations.json"
        _write_json(ann_path, ann)

        result = load_annotations(ann_path)

        assert isinstance(result, set)
        assert result == {"uid_1", "uid_2"}

    def test_annotations_dict_format(self, tmp_path):
        """Format: {"annotations": {"uid": "yes/no", ...}}."""
        ann = {
            "annotations": {
                "uid_a": "yes",
                "uid_b": "no",
                "uid_c": "positive",
                "uid_d": "true",
                "uid_e": "1",
                "uid_f": "negative",
            }
        }
        ann_path = tmp_path / "annotations.json"
        _write_json(ann_path, ann)

        result = load_annotations(ann_path)

        assert result == {"uid_a", "uid_c", "uid_d", "uid_e"}

    def test_plain_list_format(self, tmp_path):
        """Format: plain list of UIDs (all treated as positive)."""
        ann = ["uid_1", "uid_2", "uid_3"]
        ann_path = tmp_path / "annotations.json"
        _write_json(ann_path, ann)

        result = load_annotations(ann_path)

        assert result == {"uid_1", "uid_2", "uid_3"}

    def test_empty_positive_list(self, tmp_path):
        """Empty positive list returns empty set."""
        ann = {"positive": [], "negative": ["uid_1"]}
        ann_path = tmp_path / "annotations.json"
        _write_json(ann_path, ann)

        result = load_annotations(ann_path)

        assert result == set()

    def test_annotations_case_insensitive_labels(self, tmp_path):
        """Labels like 'Yes', 'YES', 'Positive' are recognized."""
        ann = {
            "annotations": {
                "uid_a": "Yes",
                "uid_b": "YES",
                "uid_c": "Positive",
                "uid_d": "TRUE",
            }
        }
        ann_path = tmp_path / "annotations.json"
        _write_json(ann_path, ann)

        result = load_annotations(ann_path)

        # The function lowercases before comparing
        assert result == {"uid_a", "uid_b", "uid_c", "uid_d"}


# ---------------------------------------------------------------------------
# nearest_neighbor_order
# ---------------------------------------------------------------------------


class TestNearestNeighborOrder:
    """Tests for nearest_neighbor_order path ordering."""

    def test_known_square(self):
        """Four points in a square: ordering visits all and returns valid permutation."""
        points = [[0, 0], [10, 0], [10, 10], [0, 10]]

        order = nearest_neighbor_order(points)

        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}

    def test_single_point(self):
        """Single point returns [0]."""
        order = nearest_neighbor_order([[5, 5]])
        assert order == [0]

    def test_empty_returns_empty(self):
        """Empty input returns empty list."""
        order = nearest_neighbor_order([])
        assert order == []

    def test_valid_permutation(self):
        """Result is a valid permutation of all indices."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 1000, size=(20, 2)).tolist()

        order = nearest_neighbor_order(points)

        assert len(order) == 20
        assert set(order) == set(range(20))

    def test_start_idx_respected(self):
        """Custom start_idx is used as the first element."""
        points = [[0, 0], [100, 100], [200, 200], [50, 50]]

        order = nearest_neighbor_order(points, start_idx=2)

        assert order[0] == 2

    def test_collinear_points(self):
        """Collinear points produce a valid ordering."""
        points = [[i * 10, 0] for i in range(10)]

        order = nearest_neighbor_order(points)

        assert len(order) == 10
        assert set(order) == set(range(10))

    def test_two_points(self):
        """Two points produce a valid ordering [start, other]."""
        points = [[0, 0], [100, 100]]

        order = nearest_neighbor_order(points)

        assert len(order) == 2
        assert set(order) == {0, 1}


# ---------------------------------------------------------------------------
# get_detection_coordinates
# ---------------------------------------------------------------------------


class TestGetDetectionCoordinates:
    """Tests for get_detection_coordinates coordinate extraction."""

    def test_with_global_center(self):
        """Extracts [x, y] from global_center."""
        det = {"global_center": [150, 300]}

        coords = get_detection_coordinates(det)

        assert coords == [150, 300]

    def test_without_global_center_returns_none(self):
        """Returns None when global_center is absent."""
        det = {"center": [10, 20]}  # tile-local, not global

        coords = get_detection_coordinates(det)

        assert coords is None

    def test_does_not_fallback_to_center(self):
        """Never falls back to 'center' (tile-local)."""
        det = {"center": [10, 20], "features": {"area": 500}}

        coords = get_detection_coordinates(det)

        assert coords is None

    def test_returns_reference_not_copy(self):
        """Returns the same list object from the detection."""
        center = [100, 200]
        det = {"global_center": center}

        coords = get_detection_coordinates(det)

        assert coords is center


# ---------------------------------------------------------------------------
# generate_spatial_control
# ---------------------------------------------------------------------------


class TestGenerateSpatialControl:
    """Tests for generate_spatial_control direction-based shifting."""

    def _make_square_contour(self, cx=0, cy=0, size=10):
        """Create a square contour centered at (cx, cy)."""
        half = size / 2
        return [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]

    def test_no_overlap_returns_shifted_contour(self):
        """With no existing polygons, first direction (E) is used."""
        contour = self._make_square_contour(0, 0, 10)

        shifted, direction, offset = generate_spatial_control(
            contour, precomputed_polygons=[], offset_um=100.0
        )

        assert shifted is not None
        assert len(shifted) == len(contour)
        assert direction in (
            "E",
            "NE",
            "N",
            "NW",
            "W",
            "SW",
            "S",
            "SE",
        )
        assert offset >= 100.0

    def test_shifted_contour_offset_from_original(self):
        """Shifted contour center differs from original by approximately offset_um."""
        contour = self._make_square_contour(500, 500, 20)

        shifted, direction, offset = generate_spatial_control(
            contour, precomputed_polygons=[], offset_um=100.0
        )

        orig_center = np.mean(contour, axis=0)
        shifted_center = np.mean(shifted, axis=0)
        dist = np.linalg.norm(shifted_center - orig_center)
        # Distance should be approximately offset_um
        assert dist == pytest.approx(offset, rel=0.1)

    def test_always_returns_result(self):
        """Even with overlapping polygons, a fallback result is always returned."""
        pytest.importorskip("shapely")
        from shapely.geometry import Polygon

        # Place a huge polygon covering most of the area
        huge_poly = Polygon([(-10000, -10000), (10000, -10000), (10000, 10000), (-10000, 10000)])
        contour = self._make_square_contour(0, 0, 10)

        shifted, direction, offset = generate_spatial_control(
            contour,
            precomputed_polygons=[huge_poly],
            offset_um=50.0,
            max_attempts=2,
        )

        # Should fall back to E_fallback
        assert shifted is not None
        assert "fallback" in direction or direction in (
            "E",
            "NE",
            "N",
            "NW",
            "W",
            "SW",
            "S",
            "SE",
        )

    def test_returns_correct_tuple_structure(self):
        """Return value is (shifted_contour, direction_name, actual_offset)."""
        contour = self._make_square_contour(0, 0, 10)

        result = generate_spatial_control(contour, precomputed_polygons=[], offset_um=100.0)

        assert len(result) == 3
        shifted, direction, offset = result
        assert isinstance(shifted, list)
        assert isinstance(direction, str)
        assert isinstance(offset, float)


# ---------------------------------------------------------------------------
# assign_wells_with_controls
# ---------------------------------------------------------------------------


class TestAssignWellsWithControls:
    """Tests for assign_wells_with_controls serpentine well assignment."""

    def test_singles_get_alternating_wells(self):
        """Each single gets target well, then control well."""
        singles = [
            {"type": "single", "uid": "s1", "contour_um": [[0, 0], [1, 0], [1, 1]]},
            {"type": "single", "uid": "s2", "contour_um": [[2, 0], [3, 0], [3, 1]]},
        ]
        ctrls = [
            {"type": "single_control", "uid": "s1_ctrl", "contour_um": [[10, 0], [11, 0], [11, 1]]},
            {"type": "single_control", "uid": "s2_ctrl", "contour_um": [[12, 0], [13, 0], [13, 1]]},
        ]

        assignments, wells = assign_wells_with_controls(singles, ctrls, [], [])

        assert len(assignments) == 4
        # Target-control alternation
        assert assignments[0]["type"] == "single"
        assert assignments[1]["type"] == "single_control"
        assert assignments[2]["type"] == "single"
        assert assignments[3]["type"] == "single_control"
        # Each should have a well assigned
        for item in assignments:
            assert "well" in item

    def test_clusters_after_singles(self):
        """Clusters are assigned wells after all singles."""
        singles = [{"type": "single", "uid": "s1"}]
        single_ctrls = [{"type": "single_control", "uid": "s1_ctrl"}]
        clusters = [{"type": "cluster", "uid": "c1"}]
        cluster_ctrls = [{"type": "cluster_control", "uid": "c1_ctrl"}]

        assignments, wells = assign_wells_with_controls(
            singles, single_ctrls, clusters, cluster_ctrls
        )

        assert len(assignments) == 4
        # First 2 are single + ctrl, next 2 are cluster + ctrl
        assert assignments[0]["type"] == "single"
        assert assignments[1]["type"] == "single_control"
        assert assignments[2]["type"] == "cluster"
        assert assignments[3]["type"] == "cluster_control"

    def test_empty_inputs(self):
        """No detections returns empty assignments."""
        assignments, wells = assign_wells_with_controls([], [], [], [])

        assert assignments == []
        assert wells == []

    def test_well_names_are_plate_format(self):
        """Well names follow plate format (e.g., B2, D4)."""
        singles = [{"type": "single", "uid": f"s{i}"} for i in range(3)]
        ctrls = [{"type": "single_control", "uid": f"s{i}_ctrl"} for i in range(3)]

        assignments, wells = assign_wells_with_controls(singles, ctrls, [], [])

        # Wells should be alphanumeric plate addresses
        for item in assignments:
            well = item["well"]
            assert isinstance(well, str)
            # Basic check: starts with letter, followed by digits
            assert well[0].isalpha()
            assert len(well) >= 2 and well[1:].isdigit()

    def test_unique_wells_assigned(self):
        """Each assignment gets a unique well."""
        singles = [{"type": "single", "uid": f"s{i}"} for i in range(5)]
        ctrls = [{"type": "single_control", "uid": f"s{i}_ctrl"} for i in range(5)]

        assignments, wells = assign_wells_with_controls(singles, ctrls, [], [])

        assigned_wells = [item["well"] for item in assignments]
        assert len(assigned_wells) == len(set(assigned_wells)), "Wells should be unique"
