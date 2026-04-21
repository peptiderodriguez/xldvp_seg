"""Tests for global_cluster_spatial_viewer.py --rare-mode extension.

Loads the script as a module without running ``main()`` (per `tests/
test_cluster_spatial_stats.py` pattern), then exercises:

- ``_load_rare_cluster_summary`` parses CSV with correct types
- ``_build_rare_dendrogram_svg`` emits SVG with clickable leaves
- ``generate_rare_html`` produces HTML with expected sidebar columns +
  dendrogram block
- Edge case: <3 clusters → dendrogram skipped (banner shown)
- Edge case: 0 stable clusters → banner shown
"""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_viewer_module():
    """Load scripts/global_cluster_spatial_viewer.py without running main()."""
    script = REPO / "scripts" / "global_cluster_spatial_viewer.py"
    spec = importlib.util.spec_from_file_location("global_cluster_spatial_viewer", script)
    mod = importlib.util.module_from_spec(spec)
    # Inject into sys.modules so relative imports resolve
    sys.modules["global_cluster_spatial_viewer"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_fake_summary_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cluster_summary.csv"
    fieldnames = [
        "cluster_id",
        "size",
        "hdbscan_persistence",
        "moran_i",
        "stable",
        "noise_pct",
        "top_regions",
        "top_morph_features",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def test_load_rare_cluster_summary_handles_null_moran(tmp_path):
    """moran_i empty / non-finite → None in loaded dict; JS renders as '—'."""
    mod = _load_viewer_module()
    path = tmp_path / "cs.csv"
    path.write_text(
        "cluster_id,size,hdbscan_persistence,moran_i,stable,noise_pct,"
        "top_regions,top_morph_features\n"
        "0,100,0.5,,True,0.1,1:100,area:1.0\n"  # empty moran
        "1,50,0.3,nan,False,0.1,,\n"  # literal 'nan' string → non-finite
        "2,80,0.4,0.42,True,0.1,,\n"  # normal numeric
    )
    rows = mod._load_rare_cluster_summary(path)
    assert rows[0]["moran_i"] is None
    assert rows[1]["moran_i"] is None  # "nan" parses but not finite → None
    assert rows[2]["moran_i"] == 0.42


def test_load_rare_cluster_summary_typed(tmp_path):
    mod = _load_viewer_module()
    rows_in = [
        {
            "cluster_id": 0,
            "size": 1200,
            "hdbscan_persistence": 0.45,
            "moran_i": 0.32,
            "stable": True,
            "noise_pct": 0.12,
            "top_regions": "1:400;2:300",
            "top_morph_features": "area:1.2",
        },
        {
            "cluster_id": 1,
            "size": 800,
            "hdbscan_persistence": 0.12,
            "moran_i": -0.02,
            "stable": False,
            "noise_pct": 0.12,
            "top_regions": "",
            "top_morph_features": "",
        },
    ]
    csv_path = _write_fake_summary_csv(tmp_path, rows_in)
    out = mod._load_rare_cluster_summary(csv_path)
    assert len(out) == 2
    assert out[0]["cluster_id"] == 0 and out[0]["size"] == 1200
    assert out[0]["stable"] is True
    assert out[1]["stable"] is False
    assert isinstance(out[0]["moran_i"], float)


def test_build_rare_dendrogram_svg():
    mod = _load_viewer_module()
    # Build a toy Ward linkage on 4 centroids
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    centroids = np.array([[0, 0], [0.1, 0.05], [5, 5], [5.1, 5.2]], dtype=np.float32)
    Z = scipy_linkage(centroids, method="ward")
    cluster_ids = np.array([10, 20, 30, 40])
    summary_by_id = {
        10: {"size": 1000, "stable": True},
        20: {"size": 2000, "stable": True},
        30: {"size": 8000, "stable": True},
        40: {"size": 5000, "stable": False},
    }
    svg = mod._build_rare_dendrogram_svg(Z, cluster_ids, summary_by_id)
    assert svg.startswith("<svg"), "Should produce SVG"
    assert 'data-cid="10"' in svg
    assert 'data-cid="40"' in svg
    assert "selectCluster(" in svg
    assert "<polyline" in svg


def test_build_dendrogram_too_few_clusters_returns_empty():
    mod = _load_viewer_module()
    Z = np.zeros((0, 4), dtype=np.float32)
    svg = mod._build_rare_dendrogram_svg(Z, np.array([0]), {0: {"size": 1000, "stable": True}})
    assert svg == ""


def test_generate_rare_html_contains_expected_sections(tmp_path):
    mod = _load_viewer_module()
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    # Fake detections
    detections = [
        {"organ_id": 1, "rare_pop_id": 0},
        {"organ_id": 1, "rare_pop_id": 0},
        {"organ_id": 2, "rare_pop_id": 1},
        {"organ_id": 2, "rare_pop_id": 1},
        {"organ_id": 2, "rare_pop_id": 2},
        {"organ_id": 3, "rare_pop_id": 2},
    ]
    region_ids = np.array([1, 1, 2, 2, 2, 3])
    summary = [
        {
            "cluster_id": 0,
            "size": 1200,
            "hdbscan_persistence": 0.45,
            "moran_i": 0.5,
            "stable": True,
            "noise_pct": 0.1,
            "top_regions": "1:1200",
            "top_morph_features": "area_um2:1.5",
        },
        {
            "cluster_id": 1,
            "size": 2500,
            "hdbscan_persistence": 0.30,
            "moran_i": 0.2,
            "stable": True,
            "noise_pct": 0.1,
            "top_regions": "2:2500",
            "top_morph_features": "",
        },
        {
            "cluster_id": 2,
            "size": 900,
            "hdbscan_persistence": 0.15,
            "moran_i": 0.0,
            "stable": False,
            "noise_pct": 0.1,
            "top_regions": "3:900",
            "top_morph_features": "",
        },
    ]
    centroids = np.array([[0, 0], [5, 0], [10, 10]], dtype=np.float32)
    Z = scipy_linkage(centroids, method="ward")
    cluster_ids = np.array([0, 1, 2])

    # Fake 1x1 JPEG
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    import base64

    fluor_b64 = base64.b64encode(buf.getvalue()).decode()

    out_path = tmp_path / "rare.html"
    mod.generate_rare_html(
        kept_detections=detections,
        region_ids=region_ids,
        summary=summary,
        linkage=Z,
        cluster_ids=cluster_ids,
        contours={},
        fluor_b64=fluor_b64,
        img_w=10,
        img_h=10,
        dendrogram_png_b64=None,
        output_path=out_path,
    )
    html = out_path.read_text()
    assert "Rare Cell Populations" in html
    assert "persistence" in html
    assert "moran_i" in html.lower() or "Moran" in html
    assert "stable" in html
    # Dendrogram SVG embedded
    assert "<svg" in html
    assert 'data-cid="0"' in html
    # All 3 clusters present in JSON payload
    assert '"cluster_id": 0' in html
    assert '"cluster_id": 1' in html
    assert '"cluster_id": 2' in html
    # No zero-stable banner (we have 2 stable)
    assert "No STABLE clusters found" not in html


def test_generate_rare_html_zero_stable_banner(tmp_path):
    mod = _load_viewer_module()
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    summary = [
        {
            "cluster_id": 0,
            "size": 1200,
            "hdbscan_persistence": 0.1,
            "moran_i": 0.0,
            "stable": False,
            "noise_pct": 0.1,
            "top_regions": "",
            "top_morph_features": "",
        },
        {
            "cluster_id": 1,
            "size": 900,
            "hdbscan_persistence": 0.1,
            "moran_i": 0.0,
            "stable": False,
            "noise_pct": 0.1,
            "top_regions": "",
            "top_morph_features": "",
        },
    ]
    centroids = np.array([[0, 0], [5, 0]], dtype=np.float32)
    Z = scipy_linkage(centroids, method="ward")

    mod.generate_rare_html(
        kept_detections=[{"organ_id": 1, "rare_pop_id": 0}],
        region_ids=np.array([1]),
        summary=summary,
        linkage=Z,
        cluster_ids=np.array([0, 1]),
        contours={},
        fluor_b64="",
        img_w=10,
        img_h=10,
        dendrogram_png_b64=None,
        output_path=tmp_path / "rare.html",
    )
    html = (tmp_path / "rare.html").read_text()
    assert "No STABLE clusters found" in html


def test_parse_args_rejects_rare_mode_without_summary():
    mod = _load_viewer_module()
    with pytest.raises(SystemExit):
        mod.parse_args(
            [
                "--detections",
                "x.json",
                "--label-map",
                "l.npy",
                "--czi-path",
                "c.czi",
                "--output",
                "out.html",
                "--rare-mode",
            ]
        )
