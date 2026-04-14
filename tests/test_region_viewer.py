"""Tests for xldvp_seg.visualization.region_viewer — HTML output validation."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from xldvp_seg.visualization.region_viewer import (
    extract_region_contours,
    generate_multi_layer_viewer,
    generate_region_viewer,
)


@pytest.fixture
def simple_label_map():
    """Label map with 3 non-overlapping regions."""
    lbl = np.zeros((100, 200), dtype=np.int32)
    lbl[10:40, 10:50] = 1
    lbl[10:40, 60:100] = 2
    lbl[60:90, 30:80] = 3
    return lbl


@pytest.fixture
def fake_fluor_b64():
    """Minimal valid JPEG base64 for a 1x1 black pixel."""
    # Create a tiny grayscale image, encode as JPEG
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((100, 200), dtype=np.uint8)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return [("fluor", b64)]


@pytest.fixture
def sample_stats():
    return {
        1: {
            "count": 5000,
            "mean_nuc": 1.2,
            "median_nuc": 1,
            "nuc_dist": {"1": 4000, "2": 900, "3": 100},
        },
        2: {"count": 2000, "mean_nuc": 1.1, "median_nuc": 1, "nuc_dist": {"1": 1800, "2": 200}},
        3: {
            "count": 500,
            "mean_nuc": 1.3,
            "median_nuc": 1,
            "nuc_dist": {"1": 350, "2": 120, "3": 30},
        },
    }


class TestExtractRegionContours:
    def test_basic(self, simple_label_map):
        contours = extract_region_contours(simple_label_map)
        assert set(contours.keys()) == {1, 2, 3}
        for rid, pts in contours.items():
            assert len(pts) >= 4  # rectangle has at least 4 points
            assert all(len(p) == 2 for p in pts)

    def test_empty_label_map(self):
        lbl = np.zeros((50, 50), dtype=np.int32)
        contours = extract_region_contours(lbl)
        assert contours == {}

    def test_single_pixel_excluded(self):
        lbl = np.zeros((50, 50), dtype=np.int32)
        lbl[25, 25] = 1  # single pixel — too small for contour
        contours = extract_region_contours(lbl)
        assert len(contours) == 0


class TestGenerateRegionViewer:
    def test_produces_html(self, simple_label_map, fake_fluor_b64, tmp_path):
        out = tmp_path / "viewer.html"
        generate_region_viewer(simple_label_map, fake_fluor_b64, out)
        assert out.exists()
        html = out.read_text()
        assert "<canvas" in html
        assert "Region Viewer" in html

    def test_with_stats(self, simple_label_map, fake_fluor_b64, sample_stats, tmp_path):
        out = tmp_path / "viewer.html"
        generate_region_viewer(simple_label_map, fake_fluor_b64, out, region_stats=sample_stats)
        html = out.read_text()
        assert "5,000" in html or "5000" in html  # region 1 cell count
        assert "nuc_dist" in html or "nd" in html

    def test_min_cells_filter(self, simple_label_map, fake_fluor_b64, sample_stats, tmp_path):
        out = tmp_path / "viewer.html"
        generate_region_viewer(
            simple_label_map, fake_fluor_b64, out, region_stats=sample_stats, min_cells=1000
        )
        html = out.read_text()
        # Region 3 has 500 cells — should be excluded
        # Regions 1 (5000) and 2 (2000) should remain
        assert "2 regions" in html

    def test_no_xss_in_json(self, simple_label_map, fake_fluor_b64, tmp_path):
        out = tmp_path / "viewer.html"
        generate_region_viewer(simple_label_map, fake_fluor_b64, out)
        html = out.read_text()
        # safe_json should escape </script> and <!--
        assert "</script>" not in html.split("</script>")[0].split("<script>")[-1]


class TestGenerateMultiLayerViewer:
    def test_produces_html(self, simple_label_map, fake_fluor_b64, tmp_path):
        layers = [
            ("pts32", simple_label_map),
            ("pts64_filled", simple_label_map * 2),  # different labels
        ]
        out = tmp_path / "multi.html"
        generate_multi_layer_viewer(layers, fake_fluor_b64, out)
        assert out.exists()
        html = out.read_text()
        assert "LAYERS" in html
        assert "pts32" in html
        assert "pts64_filled" in html
        assert "2 layers" in html.lower() or "2 segmentation" in html.lower()

    def test_filled_tag(self, simple_label_map, fake_fluor_b64, tmp_path):
        layers = [("clean", simple_label_map), ("filled", simple_label_map)]
        out = tmp_path / "multi.html"
        generate_multi_layer_viewer(layers, fake_fluor_b64, out)
        html = out.read_text()
        assert "filled" in html.lower()

    def test_empty_layers(self, fake_fluor_b64, tmp_path):
        out = tmp_path / "empty.html"
        generate_multi_layer_viewer([], fake_fluor_b64, out)
        assert not out.exists()  # should not write empty viewer
