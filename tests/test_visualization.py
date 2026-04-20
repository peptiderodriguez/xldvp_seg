"""Tests for xldvp_seg.visualization modules.

Covers encoding, colors, JS loading, and data_loading utilities.

Run with: pytest tests/test_visualization.py -v
"""

import base64
import json
import math

import numpy as np
import pytest

from xldvp_seg.visualization.colors import (
    AUTO_COLORS,
    BINARY_COLORS,
    QUAD_COLORS,
    assign_group_colors,
    hsl_palette,
)
from xldvp_seg.visualization.data_loading import compute_auto_eps
from xldvp_seg.visualization.encoding import (
    build_contour_js_data,
    encode_float32_base64,
    safe_json,
)
from xldvp_seg.visualization.js_loader import load_js

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_float32_base64(encoded):
    """Decode a base64 string back to a numpy float32 array."""
    raw = base64.b64decode(encoded)
    return np.frombuffer(raw, dtype=np.float32)


def _make_slides_data(group_labels):
    """Build a minimal slides_data list for assign_group_colors tests.

    Args:
        group_labels: list of label strings (all assigned to a single slide).

    Returns:
        List of (name, data) tuples compatible with assign_group_colors.
    """
    groups = [{"label": lbl, "n": 10} for lbl in group_labels]
    return [("slide_1", {"groups": groups})]


# ---------------------------------------------------------------------------
# encode_float32_base64
# ---------------------------------------------------------------------------


class TestEncodeFloat32Base64:
    """Tests for encode_float32_base64 roundtrip encoding."""

    def test_roundtrip_simple_array(self):
        """Encode a float32 array, decode base64, compare values."""
        arr = np.array([1.0, 2.5, -3.14, 0.0], dtype=np.float32)
        encoded = encode_float32_base64(arr)

        decoded = _decode_float32_base64(encoded)
        np.testing.assert_array_almost_equal(decoded, arr)

    def test_empty_array(self):
        """Empty array produces a valid (empty) base64 string that decodes back."""
        arr = np.array([], dtype=np.float32)
        encoded = encode_float32_base64(arr)

        assert isinstance(encoded, str)
        decoded = _decode_float32_base64(encoded)
        assert len(decoded) == 0

    def test_single_element(self):
        """Single-element array roundtrips correctly."""
        arr = np.array([42.0], dtype=np.float32)
        encoded = encode_float32_base64(arr)

        decoded = _decode_float32_base64(encoded)
        assert len(decoded) == 1
        assert decoded[0] == pytest.approx(42.0)

    def test_auto_casts_float64_to_float32(self):
        """Input float64 array is cast to float32 before encoding."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        encoded = encode_float32_base64(arr)

        decoded = _decode_float32_base64(encoded)
        assert decoded.dtype == np.float32
        np.testing.assert_array_almost_equal(decoded, arr.astype(np.float32))

    def test_large_array_roundtrip(self):
        """Roundtrip a larger array to verify no truncation."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(1000).astype(np.float32)
        encoded = encode_float32_base64(arr)

        decoded = _decode_float32_base64(encoded)
        np.testing.assert_array_equal(decoded, arr)


# ---------------------------------------------------------------------------
# safe_json
# ---------------------------------------------------------------------------


class TestSafeJson:
    """Tests for safe_json XSS-safe JSON encoding."""

    def test_normal_output(self):
        """Normal dict is valid JSON."""
        obj = {"key": "value", "n": 42}
        result = safe_json(obj)

        parsed = json.loads(result)
        assert parsed == obj

    def test_script_tag_escaped(self):
        """</script> sequences are escaped to prevent XSS."""
        obj = {"html": "<script>alert(1)</script>"}
        result = safe_json(obj)

        # The raw result must NOT contain literal </script>
        assert "</script>" not in result
        # It should contain the escaped form
        assert "<\\/script>" in result

    def test_closing_tag_in_various_contexts(self):
        """Closing tags in values and keys are all escaped."""
        obj = {"</tag>": "foo</bar>baz"}
        result = safe_json(obj)

        assert "</" not in result
        assert "<\\/" in result

    def test_nan_inf_sanitized_to_null(self):
        """safe_json sanitizes NaN/Inf to null for strict-JSON browser parsing.

        stdlib json.dumps emits NaN/Infinity by default (non-standard JSON
        that breaks JSON.parse in browsers). safe_json recursively replaces
        non-finite floats with None before serialization.
        """
        assert safe_json(float("nan")) == "null"
        assert safe_json(float("inf")) == "null"
        assert safe_json(float("-inf")) == "null"
        # Nested sanitization
        assert safe_json({"a": float("nan"), "b": [1.0, float("inf"), 3.0]}) == (
            '{"a": null, "b": [1.0, null, 3.0]}'
        )

    def test_list_output(self):
        """Lists are correctly serialized."""
        obj = [1, 2, "three"]
        result = safe_json(obj)

        parsed = json.loads(result)
        assert parsed == obj


# ---------------------------------------------------------------------------
# build_contour_js_data
# ---------------------------------------------------------------------------


class TestBuildContourJsData:
    """Tests for build_contour_js_data contour conversion."""

    def test_normal_contours(self):
        """Standard contours produce JS-ready dicts with bounding boxes."""
        # Triangle contour, pixel_size_um = 0.5
        contour = [[0, 0], [10, 0], [5, 10]]
        contours_raw = [(contour, 0.5)]

        result = build_contour_js_data(contours_raw)

        assert len(result) == 1
        item = result[0]
        assert "pts" in item
        assert "bx1" in item and "by1" in item
        assert "bx2" in item and "by2" in item
        # Pts should be in um: [0*0.5, 0*0.5, 10*0.5, 0*0.5, 5*0.5, 10*0.5]
        assert item["bx1"] == pytest.approx(0.0, abs=0.2)
        assert item["bx2"] == pytest.approx(5.0, abs=0.2)
        assert item["by1"] == pytest.approx(0.0, abs=0.2)
        assert item["by2"] == pytest.approx(5.0, abs=0.2)

    def test_empty_list(self):
        """Empty contours list returns empty list."""
        result = build_contour_js_data([])
        assert result == []

    def test_max_contours_cap(self):
        """max_contours limits the output to that many entries."""
        # Create 200 contours
        contours_raw = [([[0, 0], [10, 0], [5, 10]], 1.0) for _ in range(200)]

        result = build_contour_js_data(contours_raw, max_contours=50)

        # With step = max(1, 200//50) = 4, we get ceil(200/4) = 50 entries
        assert len(result) <= 50

    def test_too_few_points_skipped(self):
        """Contours with fewer than 3 points are skipped."""
        contours_raw = [
            ([[0, 0], [10, 0]], 1.0),  # only 2 points
            ([[0, 0], [10, 0], [5, 10]], 1.0),  # 3 points -- valid
        ]

        result = build_contour_js_data(contours_raw)

        assert len(result) == 1

    def test_invalid_contour_skipped(self):
        """Contours that cannot be parsed are silently skipped."""
        contours_raw = [
            ("not_an_array", 1.0),
            ([[0, 0], [10, 0], [5, 10]], 1.0),
        ]

        result = build_contour_js_data(contours_raw)

        assert len(result) == 1

    def test_pixel_size_conversion(self):
        """Coordinates are multiplied by pixel_size_um."""
        contour = [[100, 200], [300, 200], [200, 400]]
        pixel_size = 0.325
        contours_raw = [(contour, pixel_size)]

        result = build_contour_js_data(contours_raw)

        assert len(result) == 1
        # Check bounding box is in um coordinates
        assert result[0]["bx1"] == pytest.approx(100 * 0.325, abs=0.1)
        assert result[0]["bx2"] == pytest.approx(300 * 0.325, abs=0.1)


# ---------------------------------------------------------------------------
# hsl_palette
# ---------------------------------------------------------------------------


class TestHslPalette:
    """Tests for hsl_palette color generation."""

    def test_correct_count(self):
        """Requested number of colors is returned."""
        for n in [1, 5, 10, 25]:
            colors = hsl_palette(n)
            assert len(colors) == n

    def test_valid_hex_format(self):
        """All colors are valid #RRGGBB hex strings."""
        colors = hsl_palette(20)
        for color in colors:
            assert isinstance(color, str)
            assert len(color) == 7, f"Expected 7-char hex, got {color!r}"
            assert color[0] == "#"
            # Verify hex digits
            int(color[1:], 16)  # raises ValueError if invalid

    def test_unique_colors(self):
        """For moderate n, colors should be distinct."""
        colors = hsl_palette(10)
        assert len(set(colors)) == 10

    def test_single_color(self):
        """Single color request produces valid hex."""
        colors = hsl_palette(1)
        assert len(colors) == 1
        assert colors[0].startswith("#")

    def test_zero_colors(self):
        """Zero colors returns empty list."""
        colors = hsl_palette(0)
        assert colors == []


# ---------------------------------------------------------------------------
# assign_group_colors
# ---------------------------------------------------------------------------


class TestAssignGroupColors:
    """Tests for assign_group_colors palette selection logic."""

    def test_binary_positive_negative(self):
        """Two groups named positive/negative get BINARY_COLORS."""
        slides_data = _make_slides_data(["positive", "negative"])

        color_map = assign_group_colors(slides_data)

        assert color_map["positive"] == BINARY_COLORS["positive"]
        assert color_map["negative"] == BINARY_COLORS["negative"]

    def test_two_arbitrary_groups(self):
        """Two arbitrary groups get red/blue."""
        slides_data = _make_slides_data(["alpha", "beta"])

        color_map = assign_group_colors(slides_data)

        assert len(color_map) == 2
        # Should be red and blue (first two of palette)
        assert color_map["alpha"] == "#ff4444"
        assert color_map["beta"] == "#4488ff"

    def test_four_groups_use_quad_colors(self):
        """Four groups use QUAD_COLORS palette."""
        slides_data = _make_slides_data(["a", "b", "c", "d"])

        color_map = assign_group_colors(slides_data)

        assert len(color_map) == 4
        sorted_labels = sorted(["a", "b", "c", "d"])
        for i, label in enumerate(sorted_labels):
            assert color_map[label] == QUAD_COLORS[i]

    def test_five_to_twenty_groups_use_auto_colors(self):
        """5-20 groups use AUTO_COLORS palette."""
        labels = [f"group_{i}" for i in range(7)]
        slides_data = _make_slides_data(labels)

        color_map = assign_group_colors(slides_data)

        assert len(color_map) == 7
        sorted_labels = sorted(labels)
        for i, label in enumerate(sorted_labels):
            assert color_map[label] == AUTO_COLORS[i]

    def test_more_than_twenty_groups_use_hsl(self):
        """More than 20 groups triggers HSL palette generation."""
        labels = [f"g{i:02d}" for i in range(25)]
        slides_data = _make_slides_data(labels)

        color_map = assign_group_colors(slides_data)

        assert len(color_map) == 25
        # All values should be valid hex colors
        for color in color_map.values():
            assert color.startswith("#")
            assert len(color) == 7

    def test_colors_applied_to_group_dicts(self):
        """After assign_group_colors, each group dict has a 'color' key."""
        slides_data = _make_slides_data(["positive", "negative"])

        assign_group_colors(slides_data)

        for _, data in slides_data:
            for g in data["groups"]:
                assert "color" in g
                assert g["color"].startswith("#")

    def test_multi_slide_groups_merged(self):
        """Groups from multiple slides are merged correctly."""
        slides_data = [
            ("slide1", {"groups": [{"label": "A", "n": 10}, {"label": "B", "n": 5}]}),
            ("slide2", {"groups": [{"label": "B", "n": 3}, {"label": "C", "n": 8}]}),
        ]

        color_map = assign_group_colors(slides_data)

        assert len(color_map) == 3
        assert "A" in color_map and "B" in color_map and "C" in color_map


# ---------------------------------------------------------------------------
# load_js
# ---------------------------------------------------------------------------


class TestLoadJs:
    """Tests for load_js JavaScript component loading."""

    def test_load_existing_component(self):
        """Loading 'init' returns a non-empty string with JS code."""
        js = load_js("init")

        assert isinstance(js, str)
        assert len(js) > 0
        assert "init.js" in js  # Header comment

    def test_load_multiple_components(self):
        """Loading multiple components concatenates them."""
        js = load_js("init", "pan_zoom")

        assert "init.js" in js
        assert "pan_zoom.js" in js

    def test_missing_component_raises(self):
        """Loading a nonexistent component raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="JS component not found"):
            load_js("nonexistent_component_xyz")

    def test_component_content_is_javascript(self):
        """Loaded JS contains typical JavaScript tokens."""
        js = load_js("controls")

        # Should contain at least function/var/let/const or similar JS tokens
        assert any(
            keyword in js for keyword in ["function", "const", "let", "var", "=>"]
        ), "Expected JS content"

    def test_empty_call_returns_empty(self):
        """Calling load_js() with no arguments returns empty string."""
        js = load_js()
        assert js == ""


# ---------------------------------------------------------------------------
# compute_auto_eps
# ---------------------------------------------------------------------------


class TestComputeAutoEps:
    """Tests for compute_auto_eps DBSCAN eps estimation."""

    def test_clustered_positions(self):
        """Synthetic clustered data produces a finite positive eps."""
        rng = np.random.default_rng(42)
        # Two tight clusters far apart
        cluster1 = rng.normal(loc=[0, 0], scale=1.0, size=(50, 2))
        cluster2 = rng.normal(loc=[100, 100], scale=1.0, size=(50, 2))
        positions = np.vstack([cluster1, cluster2]).astype(np.float32)

        eps = compute_auto_eps(positions, k=10)

        assert eps is not None
        assert eps > 0
        assert math.isfinite(eps)

    def test_too_few_points_returns_none(self):
        """With fewer than k+1 points, returns None."""
        positions = np.array([[0, 0], [1, 1]], dtype=np.float32)

        eps = compute_auto_eps(positions, k=10)

        assert eps is None

    def test_exactly_k_plus_one_points(self):
        """With exactly k+1 points, produces a result."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((11, 2)).astype(np.float32)

        eps = compute_auto_eps(positions, k=10)

        assert eps is not None
        assert eps > 0

    def test_uniform_grid_returns_positive_eps(self):
        """Uniformly spaced grid returns a positive eps."""
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(x, y)
        positions = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

        eps = compute_auto_eps(positions, k=10)

        assert eps is not None
        assert eps > 0

    def test_identical_points_returns_floor(self):
        """All identical points: y_range ~ 0, returns floor of 1.0."""
        positions = np.ones((20, 2), dtype=np.float32)

        eps = compute_auto_eps(positions, k=10)

        assert eps is not None
        # When all distances are the same, y_range < 1e-9 branch returns max(dist, 1.0)
        assert eps >= 1.0
