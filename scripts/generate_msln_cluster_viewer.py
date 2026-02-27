#!/usr/bin/env python3
"""Generate interactive HTML visualization for Msln+ clustering results.

Usage:
    python scripts/generate_msln_cluster_viewer.py
    python scripts/generate_msln_cluster_viewer.py --input /path/to/detections_clustered.json --output /path/to/out.html

Reads detections_clustered.json from the Msln+ output directory and generates
an interactive_clusters.html file with dual Plotly.js scatter plots (UMAP + spatial).
"""

import argparse
import json
import html as html_mod
import os
from collections import Counter

DEFAULT_INPUT = "/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/msln_plus/detections_clustered.json"
DEFAULT_OUTPUT = "/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/msln_plus/interactive_clusters.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML visualization for Msln+ clustering results."
    )
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help='Path to detections_clustered.json')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='Path for output HTML file')
    return parser.parse_args()


def main():
    args = parse_args()
    INPUT = args.input
    OUTPUT = args.output

    print(f"Loading {INPUT} ...")
    with open(INPUT) as f:
        detections = json.load(f)
    print(f"Loaded {len(detections)} detections")

    # -------------------------------------------------------------------
    # Extract the subset of fields we need for plotting
    # -------------------------------------------------------------------
    records = []
    for d in detections:
        feat = d.get("features", {})
        gc = d.get("global_center")
        ct = d.get("center", [0, 0])
        records.append({
            "uid": d.get("uid", ""),
            "umap_x": d.get("umap_x", 0),
            "umap_y": d.get("umap_y", 0),
            "gx": gc[0] if gc else ct[0],
            "gy": gc[1] if gc else ct[1],
            "cluster_label": d.get("cluster_label", "unknown"),
            "cluster_id": d.get("cluster_id", -1),
            "area": feat.get("area", 0),
            "area_um2": feat.get("area_um2", 0),
            "ch0_mean": round(feat.get("ch0_mean", 0), 1),
            "ch1_mean": round(feat.get("ch1_mean", 0), 1),
            "ch2_mean": round(feat.get("ch2_mean", 0), 1),
        })

    # -------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------
    labels = [r["cluster_label"] for r in records]
    label_counts = Counter(labels)
    total = len(records)
    n_clusters = len([l for l in label_counts if l != "noise"])
    noise_count = label_counts.get("noise", 0)

    # Sort labels: alphabetical clusters first, noise last
    sorted_labels = sorted([l for l in label_counts if l != "noise"])
    if noise_count > 0:
        sorted_labels.append("noise")

    # -------------------------------------------------------------------
    # Color assignment
    # -------------------------------------------------------------------
    COLOR_PALETTE = [
        "#636EFA",  # blue
        "#EF553B",  # red
        "#00CC96",  # green
        "#AB63FA",  # purple
        "#FFA15A",  # orange
        "#19D3F3",  # cyan
        "#FF6692",  # pink
        "#B6E880",  # lime
        "#FF97FF",  # magenta
        "#FECB52",  # yellow
    ]
    NOISE_COLOR = "#555555"

    color_map = {}
    ci = 0
    for label in sorted_labels:
        if label == "noise":
            color_map[label] = NOISE_COLOR
        else:
            color_map[label] = COLOR_PALETTE[ci % len(COLOR_PALETTE)]
            ci += 1

    # -------------------------------------------------------------------
    # Group records by label
    # -------------------------------------------------------------------
    data_by_label = {label: [] for label in sorted_labels}
    for r in records:
        data_by_label[r["cluster_label"]].append(r)

    # -------------------------------------------------------------------
    # Build Plotly trace data
    # -------------------------------------------------------------------
    traces_umap = []
    traces_spatial = []

    for label in sorted_labels:
        pts = data_by_label[label]
        count = len(pts)
        color = color_map[label]

        hover = []
        for p in pts:
            hover.append(
                f"uid: {p['uid']}<br>"
                f"cluster: {p['cluster_label']} (id={p['cluster_id']})<br>"
                f"area: {p['area']} px ({p['area_um2']:.1f} um\u00b2)<br>"
                f"ch0_mean: {p['ch0_mean']}<br>"
                f"ch1_mean: {p['ch1_mean']}<br>"
                f"ch2_mean: {p['ch2_mean']}"
            )

        msz = 2 if label == "noise" else 3
        mop = 0.4 if label == "noise" else 0.7

        trace_base = {
            "name": f"{label} ({count})",
            "mode": "markers",
            "marker": {"color": color, "size": msz, "opacity": mop},
            "hovertext": hover,
            "hoverinfo": "text",
        }

        tu = dict(trace_base)
        tu["x"] = [p["umap_x"] for p in pts]
        tu["y"] = [p["umap_y"] for p in pts]
        tu["type"] = "scattergl"
        traces_umap.append(tu)

        ts = dict(trace_base)
        ts["x"] = [p["gx"] for p in pts]
        ts["y"] = [p["gy"] for p in pts]
        ts["type"] = "scattergl"
        traces_spatial.append(ts)

    # -------------------------------------------------------------------
    # Summary breakdown HTML
    # -------------------------------------------------------------------
    summary_parts = []
    for label in sorted_labels:
        c = label_counts[label]
        pct = 100.0 * c / total
        clr = color_map[label]
        summary_parts.append(
            f'<span style="color:{clr}; font-weight:bold">'
            f"{html_mod.escape(label)}</span>: {c:,} ({pct:.1f}%)"
        )
    summary_row = " &nbsp;|&nbsp; ".join(summary_parts)

    # -------------------------------------------------------------------
    # Serialize trace data as JSON strings
    # -------------------------------------------------------------------
    tj_u = json.dumps(traces_umap)
    tj_s = json.dumps(traces_spatial)

    # -------------------------------------------------------------------
    # Assemble HTML
    # -------------------------------------------------------------------
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Msln+ Cells - Shape-based Clustering</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: #0a0a0a;
  color: #ddd;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
}}
.header {{
  padding: 16px 24px 8px;
  border-bottom: 1px solid #333;
}}
.header h1 {{
  font-size: 20px;
  color: #eee;
  margin-bottom: 8px;
  font-weight: 600;
}}
.stats {{
  display: flex;
  gap: 24px;
  margin-bottom: 6px;
  font-size: 14px;
}}
.stats .stat {{
  background: #1a1a1a;
  padding: 4px 12px;
  border-radius: 4px;
  border: 1px solid #333;
}}
.stats .stat .val {{
  color: #7cb3ff;
  font-weight: bold;
}}
.breakdown {{
  font-size: 12px;
  color: #aaa;
  margin-top: 4px;
  padding: 4px 0;
}}
.plots {{
  display: flex;
  width: 100%;
  height: calc(100vh - 120px);
}}
.plot-container {{
  flex: 1;
  min-width: 0;
}}
</style>
</head>
<body>

<div class="header">
  <h1>Msln+ Cells &mdash; Shape-based Clustering (top 10% ch2)</h1>
  <div class="stats">
    <div class="stat">Total cells: <span class="val">{total:,}</span></div>
    <div class="stat">Clusters: <span class="val">{n_clusters}</span></div>
    <div class="stat">Noise: <span class="val">{noise_count:,}</span> ({100.0 * noise_count / total:.1f}%)</div>
  </div>
  <div class="breakdown">{summary_row}</div>
</div>

<div class="plots">
  <div class="plot-container" id="umap-plot"></div>
  <div class="plot-container" id="spatial-plot"></div>
</div>

<script>
var tracesUMAP = {tj_u};
var tracesSpatial = {tj_s};

var layoutBase = {{
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#111',
  font: {{ family: 'Consolas, Monaco, Courier New, monospace', color: '#ddd', size: 11 }},
  margin: {{ l: 50, r: 20, t: 40, b: 50 }},
  legend: {{
    bgcolor: 'rgba(20,20,20,0.9)',
    bordercolor: '#444',
    borderwidth: 1,
    font: {{ size: 11 }},
    itemsizing: 'constant',
    tracegroupgap: 2
  }},
  xaxis: {{ gridcolor: '#222', zerolinecolor: '#333' }},
  yaxis: {{ gridcolor: '#222', zerolinecolor: '#333' }},
  hovermode: 'closest'
}};

var layoutUMAP = Object.assign({{}}, layoutBase, {{
  title: {{ text: 'UMAP Embedding', font: {{ size: 14 }} }},
  xaxis: Object.assign({{}}, layoutBase.xaxis, {{ title: 'UMAP 1' }}),
  yaxis: Object.assign({{}}, layoutBase.yaxis, {{ title: 'UMAP 2' }})
}});

var layoutSpatial = Object.assign({{}}, layoutBase, {{
  title: {{ text: 'Spatial Distribution (mosaic coords)', font: {{ size: 14 }} }},
  xaxis: Object.assign({{}}, layoutBase.xaxis, {{ title: 'X (px)' }}),
  yaxis: Object.assign({{}}, layoutBase.yaxis, {{ title: 'Y (px)', autorange: 'reversed' }})
}});

var config = {{
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['sendDataToCloud'],
  displaylogo: false
}};

Plotly.newPlot('umap-plot', tracesUMAP, layoutUMAP, config);
Plotly.newPlot('spatial-plot', tracesSpatial, layoutSpatial, config);
</script>

</body>
</html>"""

    with open(OUTPUT, "w") as f:
        f.write(html_out)

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"Written {OUTPUT}")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
