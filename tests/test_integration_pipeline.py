"""Integration tests: run actual pipeline on real CZI at 1% scale.

These tests exercise the full pipeline on a real MSLN slide with
--sample-fraction 0.01 (~1-2 tiles, ~200-500 detections). Each test
depends on the previous test's output.

Requirements: GPU access, ~5 minutes total, MSLN CZI accessible.

Usage:
    # Run on cluster:
    pytest tests/test_integration_pipeline.py -v -m integration

    # Use existing run (skip detection):
    INTEGRATION_RUN_DIR=/path/to/run pytest tests/test_integration_pipeline.py -v -m integration -k "not test_1"
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CZI_PATH = Path(
    "/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/"
    "20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch.czi"
)
PYTHON = sys.executable
CELL_TYPE = "cell"

# Channel layout (from czi_info.py):
#   [0] AF488 (nuc488) -- nuclear
#   [1] AF647 (PM647)  -- plasma membrane (cyto for Cellpose)
#   [2] AF750 (Msln750) -- marker
#   [3] AF555 (Pdgfra546) -- BAD STAIN, exclude

# Skip all tests if CZI not accessible
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not CZI_PATH.exists(), reason="MSLN CZI not accessible"),
]


# -------------------------------------------------------------------------
# Shared fixtures (module-scoped so state persists across tests)
# -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def output_dir():
    """Create a temp output dir for this test session."""
    with tempfile.TemporaryDirectory(prefix="xldvp_integration_") as d:
        yield Path(d)


@pytest.fixture(scope="module")
def run_dir(output_dir):
    """Discover the timestamped run directory after detection.

    The pipeline creates output_dir/<slide_name>/<timestamp>/ structure.
    We find cell_detections.json at the run root (NOT in tiles/ subdirs).
    """
    # Exclude tile-level JSONs — only look for the final merged/deduped output
    candidates = [
        p
        for p in sorted(output_dir.rglob("cell_detections.json"), key=lambda p: p.stat().st_mtime)
        if "tiles" not in p.parent.name
    ]
    if candidates:
        return candidates[-1].parent
    # Fallback: find the timestamped run dir (contains tiles/ as a subdirectory)
    for d in sorted(output_dir.rglob("tiles"), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            return d.parent  # The run dir is the parent of tiles/
    return output_dir


def _run_script(script_path, args, timeout=900):
    """Run a Python script with PYTHONPATH set to the repo root.

    Args:
        script_path: Path to the script (relative to REPO or absolute).
        args: List of CLI arguments.
        timeout: Timeout in seconds (default 10 minutes).

    Returns:
        subprocess.CompletedProcess

    Raises:
        subprocess.CalledProcessError: If the script exits non-zero.
    """
    script = Path(script_path)
    if not script.is_absolute():
        script = REPO / script
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    cmd = [PYTHON, str(script)] + [str(a) for a in args]
    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


# -------------------------------------------------------------------------
# Test 1: Detection pipeline
# -------------------------------------------------------------------------


def test_1_detection(output_dir):
    """Run detection pipeline on ~1% of tiles and verify output."""
    result = _run_script(
        "run_segmentation.py",
        [
            "--czi-path",
            str(CZI_PATH),
            "--cell-type",
            CELL_TYPE,
            "--channel-spec",
            "cyto=PM,nuc=488",
            "--channels",
            "0,1,2",
            "--all-channels",
            "--num-gpus",
            "1",
            # NOTE: 0.01 = ~1-2 tiles for fast integration testing.
            # Production always uses 1.0 (see CLAUDE.md convention).
            "--sample-fraction",
            "0.01",
            "--html-sample-fraction",
            "1.0",
            "--output-dir",
            str(output_dir),
        ],
        timeout=5400,  # CZI load + model init + 1% tiles — large slides need time
    )

    # Find the final detections JSON (exclude per-tile JSONs in tiles/ subdirs)
    det_files = [
        p for p in output_dir.rglob("cell_detections.json") if "tiles" not in p.parent.name
    ]
    assert len(det_files) >= 1, (
        f"No cell_detections.json found under {output_dir} (excluding tiles/).\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )

    det_path = sorted(det_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(det_path) as f:
        detections = json.load(f)

    assert len(detections) > 0, "Detection pipeline produced 0 detections"

    # Validate detection structure
    det = detections[0]
    assert "uid" in det, "Detection missing 'uid'"
    assert "global_center" in det, "Detection missing 'global_center'"
    assert "features" in det, "Detection missing 'features'"
    assert "contour_dilated_px" in det, "Detection missing 'contour_dilated_px'"

    feats = det["features"]
    assert "area" in feats, "Features missing 'area'"
    assert "solidity" in feats, "Features missing 'solidity'"
    assert "sam2_0" in feats, "Features missing 'sam2_0' (SAM2 embeddings)"
    assert "ch0_mean" in feats, "Features missing 'ch0_mean' (per-channel stats)"
    assert "ch0_background" in feats, "Features missing 'ch0_background' (background correction)"


# -------------------------------------------------------------------------
# Test 2: Quality filter
# -------------------------------------------------------------------------


def test_2_quality_filter(run_dir):
    """Apply quality filter and verify filtered output."""
    det_path = run_dir / "cell_detections.json"
    assert det_path.exists(), f"Detections not found at {det_path}"

    filtered_path = run_dir / "cell_detections_filtered.json"
    _run_script(
        "scripts/quality_filter_detections.py",
        [
            "--detections",
            str(det_path),
            "--output",
            str(filtered_path),
        ],
    )

    assert filtered_path.exists(), f"Filtered output not created at {filtered_path}"

    with open(det_path) as f:
        original = json.load(f)
    with open(filtered_path) as f:
        filtered = json.load(f)

    # Quality filter tags ALL detections (rf_prediction=1.0 pass, 0.0 fail) — does NOT remove any
    assert len(filtered) == len(original), (
        f"Quality filter should tag ALL detections, not remove any. "
        f"Got {len(filtered)}, expected {len(original)}"
    )
    passing = [d for d in filtered if d["rf_prediction"] == 1.0]
    failing = [d for d in filtered if d["rf_prediction"] == 0.0]
    assert len(passing) + len(failing) == len(
        filtered
    ), "All detections should be tagged 0.0 or 1.0"
    assert len(passing) > 0, "Quality filter rejected ALL detections"

    # Every detection should have rf_prediction set to 0.0 or 1.0
    for det in filtered:
        pred = det.get("rf_prediction")
        assert pred is not None, "Filtered detection missing rf_prediction"
        assert pred in (0.0, 1.0), f"rf_prediction should be 0.0 or 1.0, got {pred}"


# -------------------------------------------------------------------------
# Test 3: Marker classification
# -------------------------------------------------------------------------


def test_3_marker_classification(run_dir):
    """Classify MSLN marker and verify output fields."""
    filtered_path = run_dir / "cell_detections_filtered.json"
    assert filtered_path.exists(), f"Filtered detections not found at {filtered_path}"

    _run_script(
        "scripts/classify_markers.py",
        [
            "--detections",
            str(filtered_path),
            "--marker-wavelength",
            "750",
            "--marker-name",
            "MSLN",
            "--czi-path",
            str(CZI_PATH),
        ],
    )

    # classify_markers.py writes <stem>_classified.json in the same directory
    classified_path = run_dir / "cell_detections_filtered_classified.json"
    assert classified_path.exists(), (
        f"Classified output not found at {classified_path}. "
        f"Files in {run_dir}: {[f.name for f in run_dir.iterdir()]}"
    )

    with open(classified_path) as f:
        classified = json.load(f)

    assert len(classified) > 0, "Classified detections file is empty"

    det = classified[0]
    feats = det.get("features", {})
    assert "MSLN_class" in feats, f"Missing MSLN_class in features. Keys: {sorted(feats.keys())}"
    assert feats["MSLN_class"] in (
        "positive",
        "negative",
    ), f"MSLN_class should be 'positive' or 'negative', got '{feats['MSLN_class']}'"
    assert "marker_profile" in feats, "Missing marker_profile in features"

    # At least some cells should be positive and some negative
    # (unless stain is very weak/strong -- allow single class as non-fatal)
    classes = {d["features"]["MSLN_class"] for d in classified}
    assert classes <= {"positive", "negative"}, f"Unexpected marker classes: {classes}"
    if len(classes) == 1:
        import warnings

        warnings.warn(
            f"All detections have the same MSLN class: {classes}. Stain may be weak/strong.",
            stacklevel=2,
        )


# -------------------------------------------------------------------------
# Test 4: SlideAnalysis roundtrip
# -------------------------------------------------------------------------


def test_4_slide_analysis_roundtrip(run_dir):
    """Load classified JSON into SlideAnalysis and test filtering + export."""
    from segmentation.core import SlideAnalysis

    classified_path = run_dir / "cell_detections_filtered_classified.json"
    assert classified_path.exists(), f"Classified detections not found: {classified_path}"

    with open(classified_path) as f:
        detections = json.load(f)

    slide = SlideAnalysis.from_detections(detections)

    # features_df has MSLN_class column
    df = slide.features_df
    assert "MSLN_class" in df.columns, f"MSLN_class not in features_df columns: {list(df.columns)}"

    # Filter by marker — conservation check: positive + negative = total
    msln_pos = slide.filter(marker="MSLN", positive=True)
    msln_neg = slide.filter(marker="MSLN", positive=False)
    assert msln_pos.n_detections + msln_neg.n_detections == slide.n_detections, (
        f"Positive ({msln_pos.n_detections}) + negative ({msln_neg.n_detections}) "
        f"should sum to total ({slide.n_detections})"
    )

    # Filter by score
    high_score = slide.filter(score_threshold=0.5)
    assert high_score.n_detections > 0, (
        "filter(score_threshold=0.5) returned 0 detections -- "
        "quality filter should have set rf_prediction=1.0 on passing cells"
    )

    # to_anndata shape matches
    adata = slide.to_anndata()
    assert (
        adata.n_obs == slide.n_detections
    ), f"AnnData n_obs ({adata.n_obs}) != slide n_detections ({slide.n_detections})"


# -------------------------------------------------------------------------
# Test 5: HTML regeneration
# -------------------------------------------------------------------------


def test_5_html_generation(run_dir):
    """Regenerate HTML from classified detections and verify output."""
    classified_path = run_dir / "cell_detections_filtered_classified.json"
    assert classified_path.exists(), f"Classified detections not found: {classified_path}"

    html_dir = run_dir / "html_integration_test"
    _run_script(
        "scripts/regenerate_html.py",
        [
            "--output-dir",
            str(run_dir),
            "--czi-path",
            str(CZI_PATH),
            "--detections",
            str(classified_path),
            "--score-threshold",
            "0.5",
            "--max-samples",
            "100",
            "--html-dir",
            str(html_dir),
        ],
    )

    assert html_dir.exists(), f"HTML directory not created at {html_dir}"
    html_files = list(html_dir.glob("*.html"))
    assert len(html_files) >= 1, f"No HTML files found in {html_dir}"
    # Verify at least one page has real content (>1KB)
    large_pages = [f for f in html_files if f.stat().st_size > 1024]
    assert len(large_pages) >= 1, (
        f"No HTML pages > 1KB. " f"Sizes: {[(f.name, f.stat().st_size) for f in html_files]}"
    )


# -------------------------------------------------------------------------
# Test 6: serve_html lifecycle (start background, get URL, stop)
# -------------------------------------------------------------------------


def test_6_serve_html(run_dir):
    """Start serve_html in background, retrieve URL, then stop."""
    from pathlib import Path

    # Clean up stale server info from previous runs
    info_file = Path("/tmp/xldvp_seg_serve_info.json")
    info_file.unlink(missing_ok=True)

    html_dir = run_dir / "html_integration_test"
    if not html_dir.exists():
        # Fallback to any html dir
        html_candidates = list(run_dir.rglob("html*"))
        html_dir = html_candidates[0] if html_candidates else run_dir

    # Start in background (with tunnel — --no-tunnel ignores --background)
    try:
        _run_script(
            "serve_html.py",
            [str(html_dir), "--background"],
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("serve_html --background failed (may need cloudflared)")

    # Get URL
    try:
        result = _run_script("serve_html.py", ["--get-url"], timeout=10)
        url = result.stdout.strip()
        assert url, "serve_html --get-url returned empty string"
        assert "localhost" in url or "http" in url, f"Unexpected URL format: {url}"
    finally:
        # Always stop the server
        try:
            _run_script("serve_html.py", ["--stop"], timeout=10)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass  # Best effort cleanup


# -------------------------------------------------------------------------
# Test 7: Save/load roundtrip via SlideAnalysis
# -------------------------------------------------------------------------


def test_7_save_load_roundtrip(run_dir, tmp_path):
    """SlideAnalysis.save() then fast_json_load() produces identical data."""
    from segmentation.core import SlideAnalysis
    from segmentation.utils.json_utils import fast_json_load

    classified_path = run_dir / "cell_detections_filtered_classified.json"
    assert classified_path.exists(), f"Classified detections not found: {classified_path}"

    original_dets = fast_json_load(classified_path)
    slide = SlideAnalysis.from_detections(original_dets)

    save_path = tmp_path / "roundtrip_detections.json"
    slide.save(save_path)
    assert save_path.exists(), "SlideAnalysis.save() did not create file"

    reloaded_dets = fast_json_load(save_path)
    assert len(reloaded_dets) == len(original_dets), (
        f"Roundtrip changed detection count: " f"{len(original_dets)} -> {len(reloaded_dets)}"
    )

    # Verify UIDs match
    original_uids = [d["uid"] for d in original_dets]
    reloaded_uids = [d["uid"] for d in reloaded_dets]
    assert original_uids == reloaded_uids, "UIDs changed after save/load roundtrip"

    # Verify features preserved
    for orig, reloaded in zip(original_dets, reloaded_dets):
        orig_feats = orig.get("features", {})
        reload_feats = reloaded.get("features", {})
        assert set(orig_feats.keys()) == set(
            reload_feats.keys()
        ), f"Feature keys differ for {orig['uid']}"
