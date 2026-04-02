#!/usr/bin/env python3
"""One-command slide visualization: classify, spatial cluster, view.

Chains classify_markers → spatial_cell_analysis → generate_multi_slide_spatial_viewer
→ serve_html into a single command. Each step is optional and skipped if its output
already exists (use --force to re-run).

Usage:
    # Minimal — just serve existing viewer/HTML
    python scripts/view_slide.py /path/to/run_dir

    # Classify markers + generate viewer + serve
    python scripts/view_slide.py /path/to/run_dir \
        --markers "NeuN:1:otsu,tdTomato:2:otsu" \
        --group-field marker_profile

    # With spatial clustering
    python scripts/view_slide.py /path/to/run_dir \
        --markers "NeuN:1:otsu,tdTomato:2:otsu" \
        --spatial --spatial-filter "tdTomato_class==positive" \
        --group-field component_id

    # Read everything from YAML config
    python scripts/view_slide.py /path/to/run_dir --config configs/senescence.yaml
"""

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_detections(run_dir: Path, force: bool = False) -> Path:
    """Find the best available detection JSON in priority order."""
    cell_types = ["cell", "nmj", "mk", "vessel", "islet", "tissue_pattern", "mesothelium"]

    # Priority order per cell type
    suffixes = ["_detections_classified.json", "_detections_postdedup.json", "_detections.json"]
    if force:
        # When forcing, skip classified (we'll regenerate it)
        suffixes = ["_detections_postdedup.json", "_detections.json"]

    for suffix in suffixes:
        for ct in cell_types:
            candidate = run_dir / f"{ct}{suffix}"
            if candidate.exists():
                return candidate

    # Fallback: glob for any *_detections*.json
    candidates = sorted(run_dir.glob("*_detections*.json"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    print(f"ERROR: No detection JSON found in {run_dir}", file=sys.stderr)
    sys.exit(1)


def find_czi(run_dir: Path) -> Path | None:
    """Try to find the CZI path from pipeline_config.json."""
    config_file = run_dir / "pipeline_config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            czi_path = config.get("czi_path")
            if czi_path and Path(czi_path).exists():
                return Path(czi_path)
        except Exception:
            pass
    return None


def get_pixel_size(run_dir: Path) -> float | None:
    """Read pixel_size_um from pipeline_config.json."""
    config_file = run_dir / "pipeline_config.json"
    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            return config.get("pixel_size_um")
        except Exception:
            pass
    return None


def read_yaml_config(config_path: Path) -> dict:
    """Read YAML config file."""
    try:
        import yaml
    except ImportError:
        print(
            "ERROR: PyYAML required for --config. Install with: pip install pyyaml", file=sys.stderr
        )
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def port_is_free(port: int) -> bool:
    """Check if a TCP port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def find_free_port(start: int = 8080, max_tries: int = 20) -> int:
    """Find an available port starting from `start`."""
    for offset in range(max_tries):
        port = start + offset
        if port_is_free(port):
            return port
    # Fallback: let OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def detect_cell_type(det_path: Path) -> str | None:
    """Infer cell type from detection filename."""
    name = det_path.name
    for ct in ["cell", "nmj", "mk", "vessel", "islet", "tissue_pattern", "mesothelium"]:
        if name.startswith(ct + "_"):
            return ct
    return None


# ---------------------------------------------------------------------------
# Pipeline steps (each calls existing scripts via subprocess)
# ---------------------------------------------------------------------------


def run_classify_markers(
    det_json: Path, markers: list[dict], czi_path: Path | None, output_dir: Path
) -> Path:
    """Run classify_markers.py. Returns path to classified JSON.

    If all markers use the same method, runs a single invocation.
    If methods differ, groups by method and runs one invocation per group
    (each reads the previous output so classifications accumulate).
    """
    methods = set(m.get("method", "otsu") for m in markers)

    print(f"\n{'=' * 60}")
    print("Step 1: Marker classification")
    print(f"{'=' * 60}")
    for m in markers:
        method_desc = {
            "otsu": "Otsu threshold (auto bg subtraction on pipeline-corrected data)",
            "otsu_half": "Otsu/2 — permissive threshold (captures weak expressors)",
            "gmm": "2-component GMM (P >= 0.75 = positive)",
        }.get(m.get("method", "otsu"), m.get("method", "otsu"))
        print(f"  {m['name']} (ch{m['channel']}): {method_desc}")
    print()

    if len(methods) == 1:
        # All markers share the same method — single call
        groups = [markers]
    else:
        # Group by method, run sequentially (each reads previous output)
        by_method = {}
        for m in markers:
            by_method.setdefault(m.get("method", "otsu"), []).append(m)
        groups = list(by_method.values())

    current_json = det_json
    for group in groups:
        channels = ",".join(str(m["channel"]) for m in group)
        names = ",".join(m["name"] for m in group)
        method = group[0].get("method", "otsu")

        cmd = [
            sys.executable,
            str(REPO / "scripts" / "classify_markers.py"),
            "--detections",
            str(current_json),
            "--marker-channel",
            channels,
            "--marker-name",
            names,
            "--method",
            method,
            "--output-dir",
            str(output_dir),
        ]

        print(f"  Running: {names} (method={method})")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: classify_markers.py failed for {names}", file=sys.stderr)
            sys.exit(1)

        # Next group reads the classified output from this group
        cell_type = detect_cell_type(det_json) or "cell"
        classified = output_dir / f"{cell_type}_detections_classified.json"
        if classified.exists():
            current_json = classified

    # Find the classified output
    cell_type = detect_cell_type(det_json) or "cell"
    classified = output_dir / f"{cell_type}_detections_classified.json"
    if classified.exists():
        return classified

    # Fallback: look for any classified JSON
    candidates = sorted(output_dir.glob("*_classified.json"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    print("WARNING: classified JSON not found, using input detections")
    return det_json


def run_spatial_analysis(
    det_json: Path, pixel_size: float, spatial_filter: str | None, output_dir: Path
) -> Path:
    """Run spatial_cell_analysis.py --spatial-network. Returns path to enriched JSON."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "spatial_cell_analysis.py"),
        "--detections",
        str(det_json),
        "--output-dir",
        str(output_dir),
        "--pixel-size",
        str(pixel_size),
        "--spatial-network",
    ]
    if spatial_filter:
        cmd.extend(["--marker-filter", spatial_filter])

    print(f"\n{'=' * 60}")
    print("Step 2: Spatial network analysis")
    print(f"{'=' * 60}")
    print("  Method: Delaunay triangulation → edge pruning → connected components")
    print("  Max edge distance: 50 um (cells farther apart are not neighbors)")
    print("  Min component size: 3 cells (smaller groups ignored)")
    if spatial_filter:
        print(f"  Cell filter: {spatial_filter}")
        print("  (only filtered cells participate in the spatial graph)")
    print()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: spatial_cell_analysis.py failed", file=sys.stderr)
        sys.exit(1)

    # spatial_cell_analysis writes *_spatial.json alongside the input
    spatial = det_json.parent / det_json.name.replace(".json", "_spatial.json")
    if spatial.exists():
        return spatial

    # Also check output_dir
    spatial = output_dir / det_json.name.replace(".json", "_spatial.json")
    if spatial.exists():
        return spatial

    print("WARNING: spatial JSON not found, using input detections")
    return det_json


def run_spatial_viewer(det_json: Path, group_field: str, title: str, output_dir: Path) -> Path:
    """Run generate_multi_slide_spatial_viewer.py — unified viewer with DBSCAN + ROI.

    Features:
    - DBSCAN clustering with auto-eps (KNN knee method)
    - Interactive eps scale slider (adjust in browser, no regeneration)
    - Convex hull outlines around clusters
    - Min cells slider for DBSCAN min_samples
    - ROI drawing (circle, rectangle, polygon) with JSON export
    - Focus view (double-click to zoom in on one slide)
    - Per-group color legend with toggle
    - Pan/zoom on each slide panel
    """
    output_html = output_dir / "spatial_viewer.html"

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "generate_multi_slide_spatial_viewer.py"),
        "--detections",
        str(det_json),
        "--group-field",
        group_field,
        "--title",
        title,
        "--output",
        str(output_html),
    ]

    print(f"\n{'=' * 60}")
    print("Step 3: Generating interactive spatial viewer")
    print(f"{'=' * 60}")
    print(f"  Coloring cells by: {group_field}")
    print("  Viewer features:")
    print("    - DBSCAN spatial clustering with auto-eps (KNN knee method)")
    print("    - Eps scale slider: adjust cluster tightness in the browser")
    print("    - Convex hull outlines around clusters (min 24 cells)")
    print("    - Min cells slider for DBSCAN min_samples")
    print("    - ROI drawing (circle, rectangle, polygon) with JSON export")
    print("    - Per-group legend with show/hide toggle")
    print("    - Pan/zoom per slide panel, focus view on double-click")
    print()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: generate_multi_slide_spatial_viewer.py failed", file=sys.stderr)
        sys.exit(1)

    return output_html


def serve_directory(directory: Path, port: int):
    """Start HTTP server + Cloudflare tunnel (foreground, Ctrl+C to stop).

    Uses the pipeline's server module which tracks PIDs per-port and only
    cleans up THIS script's processes on exit — other tunnels are untouched.
    """
    from xldvp_seg.pipeline.server import start_server_and_tunnel, wait_for_server_shutdown

    # Resolve the directory to serve
    serve_dir = directory
    if (directory / "html").is_dir():
        # If there's an html/ subdir AND a spatial_viewer.html at the top level,
        # serve from the top level so both are accessible
        if (directory / "spatial_viewer.html").exists():
            serve_dir = directory
        else:
            serve_dir = directory / "html"

    slide_name = directory.parent.name if directory.parent != directory else directory.name

    print(f"\n{'=' * 60}")
    print(f"Serving: {serve_dir}")
    print(f"Port: {port}")
    print(f"{'=' * 60}")

    http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
        serve_dir,
        port=port,
        background=False,
        slide_name=slide_name,
        cell_type="view_slide",
    )
    if http_proc is None:
        print("ERROR: Failed to start server", file=sys.stderr)
        sys.exit(1)

    wait_for_server_shutdown(http_proc, tunnel_proc)


# ---------------------------------------------------------------------------
# Marker spec parsing
# ---------------------------------------------------------------------------


def parse_markers_inline(spec: str) -> list[dict]:
    """Parse inline marker spec: 'name:channel:method,...'"""
    markers = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) < 2:
            print(
                f"ERROR: Invalid marker spec '{item}'. Expected 'name:channel[:method]'",
                file=sys.stderr,
            )
            sys.exit(1)
        markers.append(
            {
                "name": parts[0].strip(),
                "channel": int(parts[1].strip()),
                "method": parts[2].strip() if len(parts) > 2 else "otsu",
            }
        )
    return markers


def parse_markers_yaml(config: dict) -> list[dict]:
    """Extract marker list from YAML config."""
    markers_raw = config.get("markers", [])
    markers = []
    for m in markers_raw:
        markers.append(
            {
                "name": m["name"],
                "channel": m["channel"],
                "method": m.get("method", "otsu"),
            }
        )
    return markers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="One-command slide visualization: classify, cluster, view",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir", type=Path, help="Run directory (timestamped, with tiles/ and *_detections.json)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config to read markers/spatial settings from",
    )
    parser.add_argument(
        "--markers",
        default=None,
        help='Inline marker spec: "name:channel:method,..." '
        '(e.g. "NeuN:1:otsu,tdTomato:2:otsu")',
    )
    parser.add_argument(
        "--spatial", action="store_true", help="Run spatial clustering (Delaunay + communities)"
    )
    parser.add_argument(
        "--spatial-filter",
        default=None,
        help='Filter expression for spatial (e.g. "tdTomato_class==positive")',
    )
    parser.add_argument(
        "--group-field",
        default="marker_profile",
        help="Field to color cells by in viewer (default: marker_profile)",
    )
    parser.add_argument("--title", default=None, help="Viewer title (default: slide name)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP server port (default: auto-detect free port from 8080)",
    )
    parser.add_argument(
        "--no-serve", action="store_true", help="Generate viewer HTML without serving"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run classification even if classified file exists"
    )

    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    if not run_dir.is_dir():
        print(f"ERROR: Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- Read YAML config (if provided) ----
    markers = None
    spatial = args.spatial
    spatial_filter = args.spatial_filter
    group_field = args.group_field
    title = args.title

    if args.config:
        config = read_yaml_config(args.config)

        # Markers from YAML (CLI --markers overrides)
        if not args.markers:
            yaml_markers = parse_markers_yaml(config)
            if yaml_markers:
                markers = yaml_markers

        # Spatial settings from YAML (CLI flags override)
        spatial_cfg = config.get("spatial_network", {})
        if spatial_cfg.get("enabled") and not args.spatial:
            spatial = True
        if not args.spatial_filter and spatial_cfg.get("marker_filter"):
            spatial_filter = spatial_cfg["marker_filter"]

        # Viewer settings from YAML
        viewer_cfg = config.get("spatial_viewer", {})
        if args.group_field == "marker_profile" and viewer_cfg.get("group_field"):
            group_field = viewer_cfg["group_field"]
        if not args.title and viewer_cfg.get("title"):
            title = viewer_cfg["title"]

    # CLI --markers overrides YAML
    if args.markers:
        markers = parse_markers_inline(args.markers)

    # Default title from directory name
    if not title:
        title = run_dir.name

    # ---- Find detection JSON ----
    det_json = find_detections(run_dir, force=args.force)
    print(f"Using detections: {det_json.name} ({det_json.parent})")

    czi_path = find_czi(run_dir)
    pixel_size = get_pixel_size(run_dir)

    # ---- Step 1: Classify markers ----
    if markers:
        classified = run_dir / f'{detect_cell_type(det_json) or "cell"}_detections_classified.json'
        if classified.exists() and not args.force:
            print(f"\nClassified detections already exist: {classified.name}")
            print("  (use --force to re-run classification)")
            det_json = classified
        else:
            det_json = run_classify_markers(det_json, markers, czi_path, run_dir)

    # ---- Step 2: Spatial analysis ----
    if spatial:
        if pixel_size is None:
            print(
                "ERROR: --spatial requires pixel_size_um but pipeline_config.json "
                "not found or missing pixel_size_um.",
                file=sys.stderr,
            )
            print("  Provide pixel size via YAML config (pixel_size_um key).", file=sys.stderr)
            sys.exit(1)
        det_json = run_spatial_analysis(det_json, pixel_size, spatial_filter, run_dir)

    # ---- Step 3: Generate spatial viewer ----
    viewer_html = run_spatial_viewer(det_json, group_field, title, run_dir)

    if viewer_html.exists():
        print(f"\nViewer written: {viewer_html}")
    else:
        print(f"\nWARNING: Expected viewer at {viewer_html} but file not found")

    # ---- Step 4: Serve ----
    if not args.no_serve:
        port = args.port if args.port else find_free_port(8080)
        serve_directory(run_dir, port)
    else:
        print("\n--no-serve specified, skipping server.")
        print(f"To view later: python scripts/view_slide.py {run_dir}")


if __name__ == "__main__":
    main()
