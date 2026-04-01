"""Tests for adaptive RDP simplification, adaptive dilation, and contour field helpers."""

import numpy as np

from segmentation.lmd.contour_processing import adaptive_dilate, adaptive_rdp_simplify
from segmentation.utils.detection_utils import get_contour_px, get_contour_um

# ---------------------------------------------------------------------------
# Helper: generate test contours
# ---------------------------------------------------------------------------


def _circle_contour(cx=100.0, cy=100.0, r=50.0, n=200):
    """Generate a circle contour with *n* points."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.float32)


def _star_contour(cx=100.0, cy=100.0, r_outer=50.0, r_inner=25.0, n_points=10):
    """Generate a star-shaped (concave) contour."""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    radii = np.where(np.arange(2 * n_points) % 2 == 0, r_outer, r_inner)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return np.column_stack([xs, ys]).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: adaptive_rdp_simplify
# ---------------------------------------------------------------------------


class TestAdaptiveRdpSimplify:
    def test_circle_within_tolerance(self):
        """A 200-point circle should simplify within 5% symmetric difference."""
        contour = _circle_contour(n=200)
        simplified, epsilon = adaptive_rdp_simplify(contour, max_area_change_pct=5.0)

        assert len(simplified) < len(contour), "Should reduce point count"
        assert epsilon > 0, "Should find a non-zero epsilon"

        # Verify symmetric difference within tolerance
        from shapely.geometry import Polygon

        orig_poly = Polygon(contour)
        simp_poly = Polygon(simplified)
        sym_diff = orig_poly.symmetric_difference(simp_poly).area
        deviation = sym_diff / orig_poly.area
        assert deviation <= 0.05, f"Symmetric difference {deviation:.3f} exceeds 5%"

    def test_tight_tolerance(self):
        """1% tolerance should still simplify but keep more points."""
        contour = _circle_contour(n=200)
        simplified_tight, _ = adaptive_rdp_simplify(contour, max_area_change_pct=1.0)
        simplified_loose, _ = adaptive_rdp_simplify(contour, max_area_change_pct=10.0)

        assert len(simplified_tight) >= len(
            simplified_loose
        ), "Tighter tolerance should keep more points"

    def test_concave_shape_within_tolerance(self):
        """A concave star shape should also stay within tolerance."""
        contour = _star_contour(n_points=50)
        simplified, epsilon = adaptive_rdp_simplify(contour, max_area_change_pct=5.0)

        from shapely.geometry import Polygon

        orig_poly = Polygon(contour)
        simp_poly = Polygon(simplified)
        sym_diff = orig_poly.symmetric_difference(simp_poly).area
        deviation = sym_diff / orig_poly.area
        assert deviation <= 0.05, f"Star deviation {deviation:.3f} exceeds 5%"

    def test_already_simple_passthrough(self):
        """Contours with < 10 points should pass through unchanged."""
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        result, epsilon = adaptive_rdp_simplify(square)
        assert len(result) == 4
        assert epsilon == 0.0

    def test_tiny_contour(self):
        """Degenerate contours (< 3 points) pass through."""
        line = np.array([[0, 0], [10, 10]], dtype=np.float32)
        result, epsilon = adaptive_rdp_simplify(line)
        assert len(result) == 2
        assert epsilon == 0.0

    def test_point_reduction(self):
        """Verify significant point reduction for dense contours."""
        contour = _circle_contour(n=500)
        simplified, _ = adaptive_rdp_simplify(contour, max_area_change_pct=5.0)
        reduction = 1 - len(simplified) / len(contour)
        assert reduction > 0.5, f"Expected >50% reduction, got {reduction:.1%}"


# ---------------------------------------------------------------------------
# Tests: adaptive_dilate
# ---------------------------------------------------------------------------


class TestAdaptiveDilate:
    def test_circle_within_tolerance(self):
        """Dilation of a circle should stay within 5% area increase."""
        from shapely.geometry import Polygon

        contour = _circle_contour(n=200)
        dilated, dilation = adaptive_dilate(contour, max_area_change_pct=5.0)

        assert dilation > 0, "Should find a non-zero dilation"

        orig_area = Polygon(contour).area
        dilated_area = Polygon(dilated).area
        increase = (dilated_area - orig_area) / orig_area
        assert increase <= 0.05, f"Area increase {increase:.3f} exceeds 5%"
        assert increase > 0, "Dilation should increase area"

    def test_tight_tolerance_smaller_dilation(self):
        """1% tolerance should produce smaller dilation than 10%."""
        contour = _circle_contour(n=200)
        _, dil_tight = adaptive_dilate(contour, max_area_change_pct=1.0)
        _, dil_loose = adaptive_dilate(contour, max_area_change_pct=10.0)
        assert dil_tight < dil_loose

    def test_small_cell_area(self):
        """Small cell (100 um² at 0.1735 um/px) should get ~0.8 px dilation at 5%."""
        # 100 um² / (0.1735²) ≈ 3322 px²  →  r ≈ 32.5 px
        contour = _circle_contour(r=32.5, n=200)
        _, dilation_px = adaptive_dilate(contour, max_area_change_pct=5.0)
        # For a circle: d ≈ r * (sqrt(1.05) - 1) ≈ 32.5 * 0.0247 ≈ 0.8 px
        assert 0.5 < dilation_px < 1.2, f"Expected ~0.8 px, got {dilation_px:.2f}"

    def test_simple_contour_passthrough(self):
        """Tiny contour (< 3 points) should pass through unchanged."""
        line = np.array([[0, 0], [10, 10]], dtype=np.float32)
        result, dilation = adaptive_dilate(line)
        assert len(result) == 2
        assert dilation == 0.0


# ---------------------------------------------------------------------------
# Tests: get_contour_px / get_contour_um helpers
# ---------------------------------------------------------------------------


class TestContourFieldHelpers:
    def test_new_field_name(self):
        det = {"contour_px": [[1, 2]], "contour_um": [[0.3, 0.6]]}
        assert get_contour_px(det) == [[1, 2]]
        assert get_contour_um(det) == [[0.3, 0.6]]

    def test_legacy_field_name(self):
        det = {"contour_dilated_px": [[1, 2]], "contour_dilated_um": [[0.3, 0.6]]}
        assert get_contour_px(det) == [[1, 2]]
        assert get_contour_um(det) == [[0.3, 0.6]]

    def test_new_takes_precedence(self):
        det = {
            "contour_px": [[1, 2]],
            "contour_dilated_px": [[99, 99]],
        }
        assert get_contour_px(det) == [[1, 2]]

    def test_missing_returns_none(self):
        det = {"features": {}}
        assert get_contour_px(det) is None
        assert get_contour_um(det) is None

    def test_empty_list_does_not_fall_through(self):
        """Empty contour_px should NOT fall through to contour_dilated_px."""
        det = {"contour_px": [], "contour_dilated_px": [[99, 99]]}
        assert get_contour_px(det) == []  # returns empty list, NOT [[99, 99]]

    def test_empty_list_um_does_not_fall_through(self):
        """Empty contour_um should NOT fall through to contour_dilated_um."""
        det = {"contour_um": [], "contour_dilated_um": [[0.3, 0.6]]}
        assert get_contour_um(det) == []


# ---------------------------------------------------------------------------
# Tests: process_contour with max_area_change_pct
# ---------------------------------------------------------------------------


class TestProcessContourAdaptive:
    def test_adaptive_rdp_via_process_contour(self):
        """process_contour with max_area_change_pct uses adaptive RDP."""
        from segmentation.lmd.contour_processing import process_contour

        contour = _circle_contour(n=200).tolist()
        result, stats = process_contour(
            contour,
            pixel_size_um=0.5,
            dilation_um=0.0,
            max_area_change_pct=5.0,
            return_stats=True,
        )
        assert result is not None
        assert stats["points_after"] < stats["points_before"]
        assert stats["valid"]

    def test_zero_pct_uses_fixed_epsilon(self):
        """max_area_change_pct=0 falls through to fixed rdp_epsilon."""
        from segmentation.lmd.contour_processing import process_contour

        contour = _circle_contour(n=200).tolist()
        result_fixed, stats_fixed = process_contour(
            contour,
            pixel_size_um=0.5,
            dilation_um=0.0,
            rdp_epsilon=2.0,
            max_area_change_pct=0,
            return_stats=True,
        )
        result_none, stats_none = process_contour(
            contour,
            pixel_size_um=0.5,
            dilation_um=0.0,
            rdp_epsilon=2.0,
            max_area_change_pct=None,
            return_stats=True,
        )
        # Both should produce the same result (fixed epsilon path)
        assert stats_fixed["points_after"] == stats_none["points_after"]
