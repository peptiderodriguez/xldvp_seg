#!/usr/bin/env python3
"""
LMD Export Tool - Export annotated detections to Leica LMD format.

Workflow:
1. Run segmentation to get candidates
2. Annotate in HTML viewer (yes/no)
3. Train classifier if needed
4. Run this script to:
   a) Load detections + annotations
   b) Place reference crosses interactively
   c) Export to Leica LMD XML

Usage:
    # Step 1: Generate HTML for placing reference crosses
    python run_lmd_export.py \
        --detections /path/to/detections.json \
        --annotations /path/to/annotations.json \
        --output-dir /path/to/output \
        --generate-cross-html

    # Step 2: After placing crosses in HTML, export to LMD
    python run_lmd_export.py \
        --detections /path/to/detections.json \
        --annotations /path/to/annotations.json \
        --crosses /path/to/crosses.json \
        --output-dir /path/to/output \
        --export

    # With spatial clustering (100 detections per well)
    python run_lmd_export.py \
        --detections /path/to/detections.json \
        --annotations /path/to/annotations.json \
        --crosses /path/to/crosses.json \
        --output-dir /path/to/output \
        --export \
        --cluster-size 100 \
        --plate-format 384 \
        --clustering-method greedy
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


def load_detections(detections_path):
    """Load detections from JSON file."""
    with open(detections_path, 'r') as f:
        return json.load(f)


def load_annotations(annotations_path):
    """
    Load annotations and return set of positive UIDs.

    Supports multiple formats:
    - {"positive": [...], "negative": [...]}
    - {"annotations": {"uid": "yes/no", ...}}
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    positive_uids = set()

    if 'positive' in data:
        # Old format
        positive_uids.update(data['positive'])
    elif 'annotations' in data:
        # New format from HTML export
        for uid, label in data['annotations'].items():
            if label.lower() in ('yes', 'positive', 'true', '1'):
                positive_uids.add(uid)
    else:
        # Assume list of positive UIDs
        if isinstance(data, list):
            positive_uids.update(data)

    return positive_uids


def filter_detections(detections, positive_uids):
    """Filter detections to only include positively annotated ones."""
    filtered = []
    for det in detections:
        uid = det.get('uid', det.get('id', ''))
        if uid in positive_uids:
            filtered.append(det)
    return filtered


def get_detection_coordinates(det):
    """Extract (x, y) coordinates from detection."""
    if 'global_center' in det:
        return det['global_center']
    elif 'center' in det:
        return det['center']
    return None


def cluster_detections_spatially(detections, target_cluster_size=100, method='greedy'):
    """
    Cluster detections spatially for well assignment.

    Args:
        detections: List of detection dicts
        target_cluster_size: Target number of detections per cluster
        method: 'greedy' (nearest neighbor), 'kmeans', or 'dbscan'

    Returns:
        List of clusters, each cluster is a list of detection indices
    """
    # Extract coordinates
    coords = []
    valid_indices = []
    for i, det in enumerate(detections):
        xy = get_detection_coordinates(det)
        if xy is not None:
            coords.append(xy)
            valid_indices.append(i)

    if len(coords) == 0:
        return []

    coords = np.array(coords)
    n_detections = len(coords)
    n_clusters = max(1, n_detections // target_cluster_size)

    if method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(valid_indices[i])

    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        # Estimate eps from target cluster size
        # Assume roughly circular clusters
        area_per_cluster = (coords.max(axis=0) - coords.min(axis=0)).prod() / n_clusters
        eps = np.sqrt(area_per_cluster / np.pi) * 0.5

        dbscan = DBSCAN(eps=eps, min_samples=1)
        labels = dbscan.fit_predict(coords)

        n_labels = labels.max() + 1
        clusters = [[] for _ in range(n_labels)]
        noise_cluster = []

        for i, label in enumerate(labels):
            if label == -1:
                noise_cluster.append(valid_indices[i])
            else:
                clusters[label].append(valid_indices[i])

        # Add noise to nearest cluster or as separate cluster
        if noise_cluster:
            clusters.append(noise_cluster)

    else:  # greedy method
        # Greedy nearest-neighbor clustering
        remaining = set(range(len(coords)))
        clusters = []

        while remaining:
            # Start new cluster from first remaining point
            start_idx = min(remaining)
            cluster = [valid_indices[start_idx]]
            remaining.remove(start_idx)
            current_centroid = coords[start_idx].copy()

            # Add nearest neighbors until cluster is full
            while len(cluster) < target_cluster_size and remaining:
                # Find nearest remaining point to cluster centroid
                min_dist = float('inf')
                nearest_idx = None

                for idx in remaining:
                    dist = np.linalg.norm(coords[idx] - current_centroid)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx

                if nearest_idx is not None:
                    cluster.append(valid_indices[nearest_idx])
                    remaining.remove(nearest_idx)
                    # Update centroid
                    cluster_coords = coords[[i for i in range(len(coords))
                                            if valid_indices[i] in cluster]]
                    current_centroid = cluster_coords.mean(axis=0)

            clusters.append(cluster)

    # Sort clusters by centroid Y then X (top-to-bottom, left-to-right)
    cluster_centroids = []
    for cluster in clusters:
        cluster_coords = np.array([coords[valid_indices.index(i)] for i in cluster])
        centroid = cluster_coords.mean(axis=0)
        cluster_centroids.append(centroid)

    # Sort by Y first (row), then X (column)
    sorted_indices = sorted(range(len(clusters)),
                           key=lambda i: (cluster_centroids[i][1], cluster_centroids[i][0]))
    clusters = [clusters[i] for i in sorted_indices]

    return clusters


def assign_wells_384(n_clusters):
    """
    Generate well names for 384-well plate (A1-P24).

    Returns list of well names in order.
    """
    rows = 'ABCDEFGHIJKLMNOP'  # 16 rows
    cols = range(1, 25)  # 24 columns

    wells = []
    for col in cols:
        for row in rows:
            wells.append(f"{row}{col}")
            if len(wells) >= n_clusters:
                return wells

    return wells


def assign_wells_96(n_clusters):
    """
    Generate well names for 96-well plate (A1-H12).
    """
    rows = 'ABCDEFGH'  # 8 rows
    cols = range(1, 13)  # 12 columns

    wells = []
    for col in cols:
        for row in rows:
            wells.append(f"{row}{col}")
            if len(wells) >= n_clusters:
                return wells

    return wells


def generate_cross_placement_html(detections, output_dir, pixel_size_um,
                                  image_width_px, image_height_px,
                                  thumbnail_path=None):
    """
    Generate HTML page for interactively placing reference crosses.

    User clicks on the overview to place crosses, then saves positions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate detection bounds for overview
    all_x = []
    all_y = []
    for det in detections:
        if 'global_center' in det:
            all_x.append(det['global_center'][0])
            all_y.append(det['global_center'][1])
        elif 'center' in det:
            all_x.append(det['center'][0])
            all_y.append(det['center'][1])

    if not all_x:
        print("ERROR: No detection coordinates found")
        return None

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Create SVG overview of detections
    svg_width = 1200
    svg_height = int(svg_width * image_height_px / image_width_px)

    scale_x = svg_width / image_width_px
    scale_y = svg_height / image_height_px

    detection_circles = []
    for det in detections:
        if 'global_center' in det:
            cx, cy = det['global_center']
        elif 'center' in det:
            cx, cy = det['center']
        else:
            continue

        svg_x = cx * scale_x
        svg_y = cy * scale_y
        detection_circles.append(f'<circle cx="{svg_x:.1f}" cy="{svg_y:.1f}" r="3" fill="lime" opacity="0.7"/>')

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>LMD Reference Cross Placement</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00d4ff;
            margin-bottom: 10px;
        }}
        .instructions {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .instructions ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .canvas-container {{
            position: relative;
            border: 2px solid #333;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
        }}
        #mainSvg {{
            display: block;
            cursor: crosshair;
        }}
        .cross-list {{
            margin-top: 20px;
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
        }}
        .cross-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            margin: 5px 0;
            background: #0f3460;
            border-radius: 4px;
        }}
        .cross-item button {{
            background: #e94560;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .buttons {{
            margin-top: 20px;
            display: flex;
            gap: 10px;
        }}
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }}
        .btn-primary {{
            background: #00d4ff;
            color: #000;
        }}
        .btn-secondary {{
            background: #333;
            color: #fff;
        }}
        .btn-success {{
            background: #00ff88;
            color: #000;
        }}
        .info {{
            margin-top: 15px;
            color: #888;
            font-size: 14px;
        }}
        .cross-marker {{
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LMD Reference Cross Placement</h1>

        <div class="instructions">
            <strong>Instructions:</strong>
            <ul>
                <li>Click on the image to place reference crosses (minimum 3 required)</li>
                <li>Crosses should be placed at identifiable landmarks visible under the LMD</li>
                <li>Recommended: place crosses at tissue corners or distinctive features</li>
                <li>Green dots show detection locations for reference</li>
            </ul>
        </div>

        <div class="canvas-container">
            <svg id="mainSvg" width="{svg_width}" height="{svg_height}"
                 viewBox="0 0 {svg_width} {svg_height}">
                <!-- Background -->
                <rect width="100%" height="100%" fill="#111"/>

                <!-- Detection points -->
                <g id="detections">
                    {''.join(detection_circles)}
                </g>

                <!-- Reference crosses (added by clicks) -->
                <g id="crosses"></g>
            </svg>
        </div>

        <div class="cross-list">
            <h3>Reference Crosses: <span id="crossCount">0</span></h3>
            <div id="crossListItems"></div>
        </div>

        <div class="buttons">
            <button class="btn btn-secondary" onclick="clearCrosses()">Clear All</button>
            <button class="btn btn-primary" onclick="undoLast()">Undo Last</button>
            <button class="btn btn-success" onclick="saveCrosses()">Save Crosses</button>
        </div>

        <div class="info">
            <p>Image size: {image_width_px} x {image_height_px} px | Pixel size: {pixel_size_um:.4f} µm/px</p>
            <p>Total detections shown: {len(detections)}</p>
        </div>
    </div>

    <script>
        const imageWidth = {image_width_px};
        const imageHeight = {image_height_px};
        const svgWidth = {svg_width};
        const svgHeight = {svg_height};
        const pixelSize = {pixel_size_um};

        let crosses = [];

        document.getElementById('mainSvg').addEventListener('click', function(e) {{
            const rect = this.getBoundingClientRect();
            const svgX = e.clientX - rect.left;
            const svgY = e.clientY - rect.top;

            // Convert to image pixel coordinates
            const imgX = svgX / svgWidth * imageWidth;
            const imgY = svgY / svgHeight * imageHeight;

            // Convert to µm
            const umX = imgX * pixelSize;
            const umY = imgY * pixelSize;

            addCross(imgX, imgY, umX, umY, svgX, svgY);
        }});

        function addCross(imgX, imgY, umX, umY, svgX, svgY) {{
            const id = crosses.length + 1;
            crosses.push({{
                id: id,
                x_px: imgX,
                y_px: imgY,
                x_um: umX,
                y_um: umY
            }});

            // Add cross to SVG
            const crossGroup = document.getElementById('crosses');
            const crossSize = 15;
            const cross = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            cross.setAttribute('id', 'cross_' + id);
            cross.setAttribute('class', 'cross-marker');
            cross.innerHTML = `
                <line x1="${{svgX - crossSize}}" y1="${{svgY}}" x2="${{svgX + crossSize}}" y2="${{svgY}}"
                      stroke="red" stroke-width="3"/>
                <line x1="${{svgX}}" y1="${{svgY - crossSize}}" x2="${{svgX}}" y2="${{svgY + crossSize}}"
                      stroke="red" stroke-width="3"/>
                <circle cx="${{svgX}}" cy="${{svgY}}" r="20" fill="none" stroke="red" stroke-width="2"/>
                <text x="${{svgX + 25}}" y="${{svgY + 5}}" fill="red" font-size="14" font-weight="bold">${{id}}</text>
            `;
            crossGroup.appendChild(cross);

            updateCrossList();
        }}

        function updateCrossList() {{
            document.getElementById('crossCount').textContent = crosses.length;

            const listDiv = document.getElementById('crossListItems');
            listDiv.innerHTML = crosses.map((c, i) => `
                <div class="cross-item">
                    <span>Cross ${{c.id}}: (${{c.x_px.toFixed(0)}}, ${{c.y_px.toFixed(0)}}) px = (${{c.x_um.toFixed(1)}}, ${{c.y_um.toFixed(1)}}) µm</span>
                    <button onclick="removeCross(${{i}})">Remove</button>
                </div>
            `).join('');
        }}

        function removeCross(index) {{
            const cross = crosses[index];
            const elem = document.getElementById('cross_' + cross.id);
            if (elem) elem.remove();
            crosses.splice(index, 1);
            updateCrossList();
        }}

        function clearCrosses() {{
            document.getElementById('crosses').innerHTML = '';
            crosses = [];
            updateCrossList();
        }}

        function undoLast() {{
            if (crosses.length > 0) {{
                removeCross(crosses.length - 1);
            }}
        }}

        function saveCrosses() {{
            if (crosses.length < 3) {{
                alert('Please place at least 3 reference crosses!');
                return;
            }}

            const data = {{
                image_width_px: imageWidth,
                image_height_px: imageHeight,
                pixel_size_um: pixelSize,
                crosses: crosses.map(c => ({{
                    id: c.id,
                    x_px: c.x_px,
                    y_px: c.y_px,
                    x_um: c.x_um,
                    y_um: c.y_um
                }}))
            }};

            // Download as JSON
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'reference_crosses.json';
            a.click();
            URL.revokeObjectURL(url);

            alert('Saved ' + crosses.length + ' reference crosses!\\n\\nUse this file with:\\npython run_lmd_export.py --crosses reference_crosses.json --export');
        }}
    </script>
</body>
</html>'''

    html_path = output_dir / 'place_crosses.html'
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Generated cross placement HTML: {html_path}")
    print(f"Open this file in a browser, place crosses, and save the JSON.")

    return html_path


def export_to_lmd(detections, crosses_data, output_path, flip_y=True,
                  cluster_size=None, plate_format='384', clustering_method='greedy'):
    """
    Export detections to Leica LMD XML format using py-lmd.

    Args:
        detections: List of detection dicts with polygon_image or contours
        crosses_data: Dict with crosses list and image metadata
        output_path: Path to save XML file
        flip_y: Whether to flip Y axis for stage coordinates
        cluster_size: If set, spatially cluster detections into groups of this size
                      and assign each cluster to a well
        plate_format: '384' or '96' well plate format
        clustering_method: 'greedy', 'kmeans', or 'dbscan'
    """
    from lmd.lib import Collection, Shape
    from lmd.tools import makeCross

    pixel_size = crosses_data['pixel_size_um']
    image_height_px = crosses_data['image_height_px']
    image_width_px = crosses_data['image_width_px']
    image_height_um = image_height_px * pixel_size
    image_width_um = image_width_px * pixel_size

    # Get calibration points from crosses (first 3)
    crosses = crosses_data['crosses']
    if len(crosses) < 3:
        raise ValueError("Need at least 3 reference crosses for calibration")

    calibration_points = np.array([
        [c['x_um'], c['y_um'] if not flip_y else image_height_um - c['y_um']]
        for c in crosses[:3]
    ])

    # Create collection
    collection = Collection(calibration_points=calibration_points)

    # Add reference crosses
    for c in crosses:
        x_um = c['x_um']
        y_um = c['y_um'] if not flip_y else image_height_um - c['y_um']

        cross = makeCross(
            center=np.array([x_um, y_um]),
            arm_length=100,  # µm
            arm_width=10,    # µm
        )
        collection.add_shape(Shape(
            points=cross,
            well="CAL",
            name=f"RefCross_{c['id']}"
        ))

    # Cluster detections if requested
    if cluster_size and cluster_size > 0:
        print(f"  Clustering detections spatially (target {cluster_size} per cluster, method={clustering_method})...")
        clusters = cluster_detections_spatially(detections, target_cluster_size=cluster_size, method=clustering_method)

        # Assign wells
        if plate_format == '96':
            wells = assign_wells_96(len(clusters))
        else:
            wells = assign_wells_384(len(clusters))

        print(f"  Created {len(clusters)} clusters for {len(wells)} wells")

        # Build detection-to-well mapping
        det_to_well = {}
        det_to_cluster = {}
        for cluster_idx, cluster in enumerate(clusters):
            well = wells[cluster_idx] if cluster_idx < len(wells) else wells[-1]
            for det_idx in cluster:
                det_to_well[det_idx] = well
                det_to_cluster[det_idx] = cluster_idx
    else:
        # All detections go to A1
        det_to_well = {i: "A1" for i in range(len(detections))}
        det_to_cluster = {i: 0 for i in range(len(detections))}
        clusters = None

    # Add detection shapes
    for i, det in enumerate(detections):
        # Get polygon coordinates
        polygon_px = None

        if 'polygon_image' in det:
            polygon_px = np.array(det['polygon_image'])
        elif 'outer_contour_global' in det:
            polygon_px = np.array(det['outer_contour_global'])
        elif 'outer_contour' in det:
            # Local contour - need tile origin
            tile_origin = det.get('tile_origin', [0, 0])
            contour = np.array(det['outer_contour'])
            if len(contour.shape) == 3:
                contour = contour.reshape(-1, 2)
            polygon_px = contour + np.array(tile_origin)

        if polygon_px is None or len(polygon_px) < 3:
            continue

        # Convert to µm
        polygon_um = polygon_px * pixel_size

        # Flip Y if needed
        if flip_y:
            polygon_um[:, 1] = image_height_um - polygon_um[:, 1]

        # Close polygon if not closed
        if not np.allclose(polygon_um[0], polygon_um[-1]):
            polygon_um = np.vstack([polygon_um, polygon_um[0]])

        # Get well assignment
        well = det_to_well.get(i, "A1")

        # Add to collection
        name = det.get('uid', det.get('id', f'Shape_{i+1:04d}'))
        collection.new_shape(polygon_um, well=well, name=name)

    # Save
    output_path = Path(output_path)
    collection.save(str(output_path))

    if clusters:
        print(f"Exported {len(detections)} shapes in {len(clusters)} clusters + {len(crosses)} reference crosses to: {output_path}")
    else:
        print(f"Exported {len(detections)} shapes + {len(crosses)} reference crosses to: {output_path}")

    # Also save metadata CSV with cluster and well assignments
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write('name,type,x_um,y_um,well,cluster\n')
        for c in crosses:
            y_um = c['y_um'] if not flip_y else image_height_um - c['y_um']
            f.write(f"RefCross_{c['id']},calibration,{c['x_um']:.2f},{y_um:.2f},CAL,\n")
        for i, det in enumerate(detections):
            name = det.get('uid', det.get('id', ''))
            if 'global_center_um' in det:
                x, y = det['global_center_um']
            elif 'global_center' in det:
                x = det['global_center'][0] * pixel_size
                y = det['global_center'][1] * pixel_size
            elif 'center' in det:
                x = det['center'][0] * pixel_size
                y = det['center'][1] * pixel_size
            else:
                continue
            if flip_y:
                y = image_height_um - y
            well = det_to_well.get(i, 'A1')
            cluster_idx = det_to_cluster.get(i, 0)
            f.write(f"{name},shape,{x:.2f},{y:.2f},{well},{cluster_idx}\n")

    print(f"Saved coordinate metadata to: {csv_path}")

    # Save cluster summary if clustering was used
    if clusters:
        cluster_summary_path = output_path.with_name(output_path.stem + '_cluster_summary.json')
        cluster_summary = {
            'total_detections': len(detections),
            'total_clusters': len(clusters),
            'cluster_size_target': cluster_size,
            'clustering_method': clustering_method,
            'plate_format': plate_format,
            'clusters': []
        }
        for cluster_idx, cluster in enumerate(clusters):
            well = wells[cluster_idx] if cluster_idx < len(wells) else wells[-1]
            cluster_coords = np.array([
                get_detection_coordinates(detections[det_idx])
                for det_idx in cluster
                if get_detection_coordinates(detections[det_idx]) is not None
            ])
            centroid = cluster_coords.mean(axis=0) if len(cluster_coords) > 0 else [0, 0]
            cluster_summary['clusters'].append({
                'cluster_id': cluster_idx,
                'well': well,
                'n_detections': len(cluster),
                'centroid_px': [float(centroid[0]), float(centroid[1])],
                'centroid_um': [float(centroid[0] * pixel_size), float(centroid[1] * pixel_size)],
                'detection_uids': [detections[idx].get('uid', detections[idx].get('id', f'det_{idx}'))
                                   for idx in cluster]
            })
        with open(cluster_summary_path, 'w') as f:
            json.dump(cluster_summary, f, indent=2)
        print(f"Saved cluster summary to: {cluster_summary_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Export annotated detections to Leica LMD format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Step 1: Generate HTML for placing reference crosses
  python run_lmd_export.py \\
      --detections output/slide/detections.json \\
      --annotations output/slide/annotations.json \\
      --output-dir output/slide/lmd \\
      --generate-cross-html

  # Step 2: After placing crosses, export to LMD
  python run_lmd_export.py \\
      --detections output/slide/detections.json \\
      --annotations output/slide/annotations.json \\
      --crosses reference_crosses.json \\
      --output-dir output/slide/lmd \\
      --export
'''
    )

    # Input files
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to detections JSON file')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotations JSON (filters to positives only)')
    parser.add_argument('--crosses', type=str, default=None,
                        help='Path to reference crosses JSON (from HTML tool)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for LMD files')
    parser.add_argument('--output-name', type=str, default='shapes',
                        help='Base name for output files')

    # Actions
    parser.add_argument('--generate-cross-html', action='store_true',
                        help='Generate HTML for placing reference crosses')
    parser.add_argument('--export', action='store_true',
                        help='Export to LMD XML (requires --crosses)')

    # Image metadata (can be auto-detected from detections)
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in µm (auto-detect from detections if not set)')
    parser.add_argument('--image-width', type=int, default=None,
                        help='Image width in pixels')
    parser.add_argument('--image-height', type=int, default=None,
                        help='Image height in pixels')

    # Spatial clustering
    parser.add_argument('--cluster-size', type=int, default=None,
                        help='Target number of detections per well (enables clustering)')
    parser.add_argument('--plate-format', type=str, default='384',
                        choices=['384', '96'],
                        help='Well plate format (default: 384)')
    parser.add_argument('--clustering-method', type=str, default='greedy',
                        choices=['greedy', 'kmeans', 'dbscan'],
                        help='Spatial clustering method (default: greedy)')

    # Options
    parser.add_argument('--no-flip-y', action='store_true',
                        help='Do not flip Y axis for stage coordinates')

    args = parser.parse_args()

    # Load detections
    print(f"Loading detections from: {args.detections}")
    detections = load_detections(args.detections)
    print(f"  Loaded {len(detections)} detections")

    # Filter by annotations if provided
    if args.annotations:
        print(f"Loading annotations from: {args.annotations}")
        positive_uids = load_annotations(args.annotations)
        print(f"  Found {len(positive_uids)} positive annotations")

        detections = filter_detections(detections, positive_uids)
        print(f"  Filtered to {len(detections)} positive detections")

    if len(detections) == 0:
        print("ERROR: No detections to export!")
        return

    # Try to auto-detect image metadata
    pixel_size = args.pixel_size
    image_width = args.image_width
    image_height = args.image_height

    # Check first detection for metadata
    sample_det = detections[0]
    if pixel_size is None:
        if 'features' in sample_det and 'pixel_size_um' in sample_det['features']:
            pixel_size = sample_det['features']['pixel_size_um']
        else:
            pixel_size = 0.22  # Default
        print(f"  Using pixel size: {pixel_size} µm/px")

    # Estimate image size from detection coordinates
    if image_width is None or image_height is None:
        max_x = max_y = 0
        for det in detections:
            if 'global_center' in det:
                max_x = max(max_x, det['global_center'][0])
                max_y = max(max_y, det['global_center'][1])
            elif 'center' in det:
                max_x = max(max_x, det['center'][0])
                max_y = max(max_y, det['center'][1])

        # Add margin
        if image_width is None:
            image_width = int(max_x * 1.1)
        if image_height is None:
            image_height = int(max_y * 1.1)
        print(f"  Estimated image size: {image_width} x {image_height} px")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate cross placement HTML
    if args.generate_cross_html:
        generate_cross_placement_html(
            detections, output_dir, pixel_size,
            image_width, image_height
        )

    # Export to LMD
    if args.export:
        if not args.crosses:
            print("ERROR: --crosses required for export. First use --generate-cross-html")
            return

        print(f"Loading crosses from: {args.crosses}")
        with open(args.crosses, 'r') as f:
            crosses_data = json.load(f)

        # Override with command line args if provided
        if args.pixel_size:
            crosses_data['pixel_size_um'] = args.pixel_size
        if args.image_width:
            crosses_data['image_width_px'] = args.image_width
        if args.image_height:
            crosses_data['image_height_px'] = args.image_height

        output_path = output_dir / f"{args.output_name}.xml"
        export_to_lmd(
            detections, crosses_data, output_path,
            flip_y=not args.no_flip_y,
            cluster_size=args.cluster_size,
            plate_format=args.plate_format,
            clustering_method=args.clustering_method
        )

    if not args.generate_cross_html and not args.export:
        print("No action specified. Use --generate-cross-html or --export")


if __name__ == '__main__':
    main()
