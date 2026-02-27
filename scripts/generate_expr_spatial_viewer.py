#!/usr/bin/env python
"""Interactive HTML viewer for grouped spatial detection data.

Two input modes:

1. **Multi-scene directory** (brain FISH expression groups):
   Reads per-scene zone data (detections_zoned.json + zone_metadata.json) from
   the expression-gated spatial analysis pipeline.

2. **Direct detection file(s)** (any clustered/labeled data):
   Reads one or more detection JSON files directly, grouping by any field
   (cluster_label, cluster_id, zone_label, etc.).

Generates a single self-contained HTML file with:
- NxM grid of scene/file panels (canvas-based, independent pan/zoom)
- Clickable legend to toggle groups on/off
- Auto-eps via KNN knee method -- per-group optimal eps, multiplier slider
- Convex hull outlines + cluster labels for visible groups
- Dot size / opacity controls

Clustering (DBSCAN + convex hull) runs entirely in JavaScript so the user can
adjust eps interactively without regenerating the HTML.

Usage:
    # Mode 1: Multi-scene directory (brain FISH)
    python scripts/generate_expr_spatial_viewer.py \\
        --input-dir brain_fish_output \\
        --subdir zones_expr_spatial_v2 \\
        --group-field expression_group

    # Mode 2: Direct detection file(s)
    python scripts/generate_expr_spatial_viewer.py \\
        --detections clustering/detections_clustered.json \\
        --group-field cluster_label \\
        --exclude-groups noise --top-n 10
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree


# --- Auto-eps via KNN knee method ---

def compute_auto_eps(positions, k=10):
    """Compute optimal DBSCAN eps using KNN distance knee/elbow method.

    Builds a KDTree, queries the Kth nearest-neighbor distance for every point,
    sorts ascending, and finds the elbow (max deviation from the diagonal).
    """
    n = len(positions)
    if n < k + 1:
        return None

    tree = KDTree(positions)
    dists, _ = tree.query(positions, k=k + 1)  # +1 because self is distance 0
    knn_dists = np.sort(dists[:, -1])  # Kth neighbor distance, sorted ascending

    # Kneedle-style elbow: max perpendicular distance from line connecting
    # first point (0, knn_dists[0]) to last point (1, knn_dists[-1])
    x_norm = np.linspace(0, 1, n)
    y_range = knn_dists[-1] - knn_dists[0]
    if y_range < 1e-9:
        return float(knn_dists[0])  # all distances equal
    y_norm = (knn_dists - knn_dists[0]) / y_range

    # Distance from diagonal (0,0)→(1,1) = (y - x) / sqrt(2), max of that
    diffs = y_norm - x_norm
    elbow_idx = int(np.argmax(diffs))
    return float(knn_dists[elbow_idx])


# --- Expression group mapping ---

# Fixed color palette for 12 simplified expression groups.
# Slc17a7 family = warm, Gad1 family = cool, co-express/other = distinct.
GROUP_COLORS = {
    'Slc17a7 only':         '#e6194b',  # red
    'Slc17a7+/Htr2a+':     '#f58231',  # orange
    'Slc17a7+/Ntrk2+':     '#ffe119',  # yellow
    'Slc17a7+/Htr2a+/Ntrk2+': '#fabebe',  # light pink
    'Gad1 only':            '#4363d8',  # blue
    'Gad1+/Htr2a+':        '#42d4f4',  # cyan
    'Gad1+/Ntrk2+':        '#911eb4',  # purple
    'Gad1+/Htr2a+/Ntrk2+': '#aaffc3',  # mint
    'Slc17a7+/Gad1+':      '#f032e6',  # magenta
    'Htr2a only':           '#3cb44b',  # green
    'Ntrk2 only':           '#469990',  # teal
    'Htr2a+/Ntrk2+':       '#9a6324',  # brown
}

# Ordered list for consistent legend rendering
GROUP_ORDER = [
    'Slc17a7 only', 'Slc17a7+/Htr2a+', 'Slc17a7+/Ntrk2+', 'Slc17a7+/Htr2a+/Ntrk2+',
    'Gad1 only', 'Gad1+/Htr2a+', 'Gad1+/Ntrk2+', 'Gad1+/Htr2a+/Ntrk2+',
    'Slc17a7+/Gad1+', 'Htr2a only', 'Ntrk2 only', 'Htr2a+/Ntrk2+',
]

# 20-color maximally-distinct palette for auto-assignment (non-expression modes)
AUTO_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990',
    '#e6beff', '#9a6324', '#ffe119', '#aaffc3', '#800000',
    '#ffd8b1', '#000075', '#a9a9a9', '#808000', '#ff69b4',
]
OTHER_COLOR = '#555555'  # gray for lumped "other" group


def parse_expression_pattern(cell_type):
    """Parse 'Slc17a7+/Htr2a-/Ntrk2-/Gad1-' into dict of marker -> bool."""
    markers = {}
    for part in cell_type.split('/'):
        m = re.match(r'^(\w+)([+-])$', part)
        if m:
            markers[m.group(1)] = (m.group(2) == '+')
    return markers


def classify_expression_group(cell_type):
    """Map full expression pattern to simplified group label.

    Returns None for quad-negative and unrecognized patterns (excluded).
    """
    if cell_type == 'other':
        return None

    markers = parse_expression_pattern(cell_type)
    if len(markers) != 4:
        return None

    slc = markers.get('Slc17a7', False)
    htr = markers.get('Htr2a', False)
    ntrk = markers.get('Ntrk2', False)
    gad = markers.get('Gad1', False)

    # Quad-negative excluded
    if not any([slc, htr, ntrk, gad]):
        return None

    # Co-expression of both major markers — Slc17a7 (VGLUT1, excitatory) + Gad1
    # (GABAergic, inhibitory) is a distinct phenotype; lump all 4 Htr2a/Ntrk2 combos.
    if slc and gad:
        return 'Slc17a7+/Gad1+'

    # Slc17a7 family (no Gad1)
    if slc and not gad:
        if htr and ntrk:
            return 'Slc17a7+/Htr2a+/Ntrk2+'
        if htr:
            return 'Slc17a7+/Htr2a+'
        if ntrk:
            return 'Slc17a7+/Ntrk2+'
        return 'Slc17a7 only'

    # Gad1 family (no Slc17a7)
    if gad and not slc:
        if htr and ntrk:
            return 'Gad1+/Htr2a+/Ntrk2+'
        if htr:
            return 'Gad1+/Htr2a+'
        if ntrk:
            return 'Gad1+/Ntrk2+'
        return 'Gad1 only'

    # Neither Slc17a7 nor Gad1
    if htr and ntrk:
        return 'Htr2a+/Ntrk2+'
    if htr:
        return 'Htr2a only'
    if ntrk:
        return 'Ntrk2 only'

    return None


def extract_group(det, group_field, zone_cell_type=None):
    """Return group label for one detection, or None to skip.

    Args:
        det: Detection dict.
        group_field: Which field to use for grouping.
        zone_cell_type: Optional dict mapping zone_id -> cell_type (for expression_group mode).

    Returns:
        Group label string, or None to skip this detection.
    """
    if group_field == 'expression_group':
        zone_id = det.get('zone_id')
        if zone_id is None or zone_cell_type is None or zone_id not in zone_cell_type:
            return None
        return classify_expression_group(zone_cell_type[zone_id])
    elif group_field == 'cluster_label':
        val = det.get('cluster_label')
        return str(val) if val is not None else None
    elif group_field == 'cluster_id':
        val = det.get('cluster_id')
        if val is None:
            return None
        return str(val)
    elif group_field == 'zone_label':
        val = det.get('zone_label')
        return str(val) if val is not None else None
    else:
        # Generic: try the field name directly
        val = det.get(group_field)
        return str(val) if val is not None else None


def load_file_data(path, group_field):
    """Load a detection JSON file and group cells by the specified field.

    Args:
        path: Path to detection JSON file.
        group_field: Which field to group by.

    Returns:
        Dict with scene data or None if no valid data.
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        detections = json.load(f)

    group_all_cells = {}  # group_label -> list of (x, y, area_um2)

    for det in detections:
        group = extract_group(det, group_field)
        if group is None:
            continue

        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            continue

        area = det.get('features', {}).get('area_um2', 0.0)
        if not isinstance(area, (int, float)) or np.isnan(area):
            area = 0.0

        group_all_cells.setdefault(group, []).append((pos[0], pos[1], area))

    if not group_all_cells:
        return None

    # Build per-group output (order will be set later by top-N logic)
    groups_out = []
    for group_label, cells in group_all_cells.items():
        cells_arr = np.array(cells, dtype=np.float32)
        auto_eps = compute_auto_eps(cells_arr[:, :2], k=10)

        groups_out.append({
            'label': group_label,
            'color': None,  # assigned later by apply_top_n_colors()
            'n': len(cells),
            'x': cells_arr[:, 0].tolist(),
            'y': cells_arr[:, 1].tolist(),
            'a': cells_arr[:, 2].tolist(),
            'auto_eps': auto_eps,
        })

    if not groups_out:
        return None

    all_x = []
    all_y = []
    for g in groups_out:
        all_x.extend(g['x'])
        all_y.extend(g['y'])

    total_cells = sum(g['n'] for g in groups_out)

    return {
        'groups': groups_out,
        'n_cells': total_cells,
        'x_range': [float(min(all_x)), float(max(all_x))],
        'y_range': [float(min(all_y)), float(max(all_y))],
    }


def load_file_data_two_stage(path, umap_fields=('umap_x', 'umap_y')):
    """Load detection JSON for two-stage clustering (flat cell arrays + UMAP coords).

    Returns per-scene flat data with spatial + UMAP coordinates for client-side
    two-stage clustering (appearance DBSCAN on UMAP, then spatial DBSCAN per group).
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        detections = json.load(f)

    xs, ys, uxs, uys, areas = [], [], [], [], []
    skipped = 0

    for det in detections:
        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            skipped += 1
            continue

        ux = det.get(umap_fields[0])
        uy = det.get(umap_fields[1])
        if ux is None or uy is None:
            skipped += 1
            continue

        if not isinstance(ux, (int, float)) or not isinstance(uy, (int, float)):
            skipped += 1
            continue

        area = det.get('features', {}).get('area_um2', 0.0)
        if not isinstance(area, (int, float)) or np.isnan(area):
            area = 0.0

        xs.append(pos[0])
        ys.append(pos[1])
        uxs.append(float(ux))
        uys.append(float(uy))
        areas.append(area)

    if not xs:
        return None

    n = len(xs)
    xs_arr = np.array(xs, dtype=np.float32)
    ys_arr = np.array(ys, dtype=np.float32)
    uxs_arr = np.array(uxs, dtype=np.float32)
    uys_arr = np.array(uys, dtype=np.float32)
    areas_arr = np.array(areas, dtype=np.float32)

    # Auto-eps for UMAP space -- use 90th percentile of KNN-10 distances.
    # The knee method gives too-small eps for dense UMAP embeddings where
    # KNN distances are uniformly small.
    umap_pos = np.column_stack([uxs_arr, uys_arr])
    k_umap = min(10, n - 1)
    if k_umap >= 1:
        tree = KDTree(umap_pos)
        dists, _ = tree.query(umap_pos, k=k_umap + 1)
        knn_dists = dists[:, -1]
        auto_eps_umap = float(np.percentile(knn_dists, 90))
    else:
        auto_eps_umap = 1.0

    # Auto-eps for spatial space (knee method works fine here)
    spatial_pos = np.column_stack([xs_arr, ys_arr])
    auto_eps_spatial = compute_auto_eps(spatial_pos, k=10)

    return {
        'n': n,
        'x': xs_arr.tolist(),
        'y': ys_arr.tolist(),
        'ux': uxs_arr.tolist(),
        'uy': uys_arr.tolist(),
        'a': areas_arr.tolist(),
        'x_range': [float(xs_arr.min()), float(xs_arr.max())],
        'y_range': [float(ys_arr.min()), float(ys_arr.max())],
        'auto_eps_umap': auto_eps_umap or 1.0,
        'auto_eps_spatial': auto_eps_spatial or 100.0,
        'skipped': skipped,
    }


def apply_top_n_colors(scenes_data, top_n, exclude_groups, group_field):
    """Apply top-N grouping and color assignment across all scenes.

    For expression_group mode, uses the fixed GROUP_COLORS palette.
    For all other modes, counts cells globally, keeps top-N groups,
    lumps the rest into "other", and auto-assigns colors.

    Args:
        scenes_data: List of (label, data_dict) tuples. Modified in place.
        top_n: Number of top groups to keep (rest become "other").
        exclude_groups: Set of group labels to drop entirely.
        group_field: The group field mode.
    """
    if group_field == 'expression_group':
        # Use fixed palette, just apply exclusions
        for label, data in scenes_data:
            data['groups'] = [
                g for g in data['groups']
                if g['label'] not in exclude_groups
            ]
            for g in data['groups']:
                if g['color'] is None:
                    g['color'] = GROUP_COLORS.get(g['label'], OTHER_COLOR)
        return

    # Count cells per group across all scenes
    global_counts = {}
    for _, data in scenes_data:
        for g in data['groups']:
            if g['label'] in exclude_groups:
                continue
            global_counts[g['label']] = global_counts.get(g['label'], 0) + g['n']

    # Sort by count descending, keep top N
    sorted_groups = sorted(global_counts.items(), key=lambda x: -x[1])
    top_labels = set()
    for i, (lbl, _) in enumerate(sorted_groups):
        if i < top_n:
            top_labels.add(lbl)

    # Assign colors to top groups (sorted by count)
    color_map = {}
    for i, (lbl, _) in enumerate(sorted_groups):
        if lbl in top_labels:
            color_map[lbl] = AUTO_COLORS[i % len(AUTO_COLORS)]
    color_map['other'] = OTHER_COLOR

    # Remap groups per scene: keep top-N, lump rest into "other"
    for _, data in scenes_data:
        new_groups = {}
        for g in data['groups']:
            if g['label'] in exclude_groups:
                continue

            target = g['label'] if g['label'] in top_labels else 'other'

            if target not in new_groups:
                new_groups[target] = {
                    'label': target,
                    'color': color_map[target],
                    'n': 0,
                    'x': [],
                    'y': [],
                    'a': [],
                    'auto_eps': None,
                    '_all_positions': [],
                }
            ng = new_groups[target]
            ng['n'] += g['n']
            ng['x'].extend(g['x'])
            ng['y'].extend(g['y'])
            ng['a'].extend(g['a'])
            ng['_all_positions'].extend(zip(g['x'], g['y']))

        # Recompute auto-eps for merged groups
        for ng in new_groups.values():
            if ng['n'] > 0:
                positions = np.array(ng['_all_positions'], dtype=np.float32)
                ng['auto_eps'] = compute_auto_eps(positions, k=10)
            del ng['_all_positions']

        # Sort: top groups by global count, "other" last
        ordered = []
        for lbl, _ in sorted_groups:
            if lbl in new_groups and lbl in top_labels:
                ordered.append(new_groups[lbl])
        if 'other' in new_groups:
            ordered.append(new_groups['other'])

        data['groups'] = ordered
        data['n_cells'] = sum(g['n'] for g in ordered)

    return color_map


def load_scene_data(scene_dir, subdir, group_field='expression_group'):
    """Load scene zone data and collect cell positions per group.

    Clustering is done client-side in JavaScript so the user can adjust eps
    interactively. This function only collects and groups the cell positions.

    For expression_group mode, requires zone_metadata.json. For other modes,
    reads detections_zoned.json directly.

    Returns dict with scene data or None if files missing.
    """
    zone_dir = scene_dir / subdir
    det_path = zone_dir / 'detections_zoned.json'

    if not det_path.exists():
        return None

    # Zone metadata needed only for expression_group mode
    zone_cell_type = None
    if group_field == 'expression_group':
        meta_path = zone_dir / 'zone_metadata.json'
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        zone_cell_type = {}
        for z in meta['zones']:
            zone_cell_type[z['structure_id']] = z['cell_type']

    with open(det_path) as f:
        detections = json.load(f)

    # Group cells
    group_all_cells = {}  # group_label -> list of (x, y, area_um2)

    for det in detections:
        group = extract_group(det, group_field, zone_cell_type=zone_cell_type)
        if group is None:
            continue

        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            continue

        area = det.get('features', {}).get('area_um2', 0.0)
        if not isinstance(area, (int, float)) or np.isnan(area):
            area = 0.0

        group_all_cells.setdefault(group, []).append((pos[0], pos[1], area))

    # Build per-group output
    groups_out = []
    if group_field == 'expression_group':
        # Use fixed order for expression groups
        for group_label in GROUP_ORDER:
            cells = group_all_cells.get(group_label, [])
            if not cells:
                continue

            cells_arr = np.array(cells, dtype=np.float32)
            auto_eps = compute_auto_eps(cells_arr[:, :2], k=10)

            groups_out.append({
                'label': group_label,
                'color': GROUP_COLORS[group_label],
                'n': len(cells),
                'x': cells_arr[:, 0].tolist(),
                'y': cells_arr[:, 1].tolist(),
                'a': cells_arr[:, 2].tolist(),
                'auto_eps': auto_eps,
            })
    else:
        # Generic: all groups, colors assigned later by apply_top_n_colors()
        for group_label, cells in group_all_cells.items():
            cells_arr = np.array(cells, dtype=np.float32)
            auto_eps = compute_auto_eps(cells_arr[:, :2], k=10)

            groups_out.append({
                'label': group_label,
                'color': None,
                'n': len(cells),
                'x': cells_arr[:, 0].tolist(),
                'y': cells_arr[:, 1].tolist(),
                'a': cells_arr[:, 2].tolist(),
                'auto_eps': auto_eps,
            })

    if not groups_out:
        return None

    # Compute coordinate ranges for auto-fit
    all_x = []
    all_y = []
    for g in groups_out:
        all_x.extend(g['x'])
        all_y.extend(g['y'])

    total_cells = sum(g['n'] for g in groups_out)

    return {
        'groups': groups_out,
        'n_cells': total_cells,
        'x_range': [float(min(all_x)), float(max(all_x))],
        'y_range': [float(min(all_y)), float(max(all_y))],
    }


def discover_scenes(input_dir, subdir):
    """Discover all scene directories containing zone data.

    Returns list of (scene_label, scene_path) tuples sorted by name.
    """
    input_dir = Path(input_dir)
    scenes = []

    for sample_dir in sorted(input_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        for scene_dir in sorted(sample_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not scene_dir.name.startswith('scene'):
                continue
            zone_dir = scene_dir / subdir
            if zone_dir.exists():
                label = f"{sample_dir.name}/{scene_dir.name}"
                scenes.append((label, scene_dir))

    return scenes


def generate_html(scenes_data, output_path, legend_items=None,
                  title='Spatial Group Viewer', default_min_cells=10,
                  default_min_hull=24):
    """Generate self-contained interactive HTML with NxM scene grid.

    Args:
        scenes_data: List of (label, data_dict) tuples.
        output_path: Output HTML file path.
        legend_items: List of (label, color) tuples for the legend.
            If None, uses GROUP_ORDER/GROUP_COLORS (expression_group mode).
        title: Page title and sidebar heading.
        default_min_cells: Default DBSCAN min_samples.
        default_min_hull: Min cells in cluster to draw hull.
    """

    # Serialize scene data to compact JS (positions only, no clusters)
    scenes_js_parts = []
    for label, data in scenes_data:
        groups_js = []
        for g in data['groups']:
            x_str = ','.join(f'{v:.1f}' for v in g['x'])
            y_str = ','.join(f'{v:.1f}' for v in g['y'])
            a_str = ','.join(f'{v:.1f}' for v in g['a'])

            auto_eps = g.get('auto_eps')
            auto_eps_js = f'{auto_eps:.1f}' if auto_eps is not None else '100'

            groups_js.append(
                f'{{label:{json.dumps(g["label"])},color:"{g["color"]}",n:{g["n"]},'
                f'x:new Float32Array([{x_str}]),'
                f'y:new Float32Array([{y_str}]),'
                f'a:new Float32Array([{a_str}]),'
                f'autoEps:{auto_eps_js},'
                f'clusters:[]}}'
            )

        scene_js = (
            f'{{name:{json.dumps(label)},n:{data["n_cells"]},'
            f'xr:[{data["x_range"][0]:.1f},{data["x_range"][1]:.1f}],'
            f'yr:[{data["y_range"][0]:.1f},{data["y_range"][1]:.1f}],'
            f'groups:[{",".join(groups_js)}]}}'
        )
        scenes_js_parts.append(scene_js)

    scenes_js = ',\n'.join(scenes_js_parts)

    # Legend data (global)
    if legend_items is None:
        legend_items = [(label, GROUP_COLORS[label]) for label in GROUP_ORDER]

    legend_js_parts = []
    for lbl, color in legend_items:
        legend_js_parts.append(f'{{label:{json.dumps(lbl)},color:"{color}"}}')
    legend_js = ','.join(legend_js_parts)

    n_scenes = len(scenes_data)
    n_cols = min(4, n_scenes)
    n_rows = (n_scenes + n_cols - 1) // n_cols

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }}
  #main {{ display: flex; width: 100vw; height: 100vh; }}
  #grid {{
    flex: 1;
    display: grid;
    grid-template-columns: repeat({n_cols}, 1fr);
    grid-template-rows: repeat({n_rows}, 1fr);
    gap: 2px;
    padding: 2px;
  }}
  .panel {{
    position: relative;
    overflow: hidden;
    background: #111122;
    border: 1px solid #333;
    border-radius: 4px;
    cursor: grab;
  }}
  .panel.dragging {{ cursor: grabbing; }}
  .panel canvas {{ position: absolute; top: 0; left: 0; }}
  .panel-label {{
    position: absolute; top: 4px; left: 6px; z-index: 10;
    font-size: 11px; color: #ccc; background: rgba(17,17,34,0.8);
    padding: 2px 6px; border-radius: 3px; pointer-events: none;
  }}
  .panel-count {{
    position: absolute; bottom: 4px; left: 6px; z-index: 10;
    font-size: 10px; color: #888; pointer-events: none;
  }}
  #sidebar {{
    width: 230px; min-width: 190px; background: rgba(26,26,46,0.95);
    border-left: 1px solid #444; overflow-y: auto; padding: 10px;
    display: flex; flex-direction: column; gap: 10px;
  }}
  #sidebar h3 {{ font-size: 13px; color: #aaa; margin-bottom: 4px; }}
  .leg-item {{
    display: flex; align-items: center; gap: 6px; padding: 3px 4px;
    cursor: pointer; border-radius: 3px; user-select: none; font-size: 11px;
  }}
  .leg-item:hover {{ background: rgba(255,255,255,0.08); }}
  .leg-item.hidden {{ opacity: 0.25; text-decoration: line-through; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-label {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .ctrl-group {{ border-top: 1px solid #333; padding-top: 8px; }}
  .ctrl-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; margin-bottom: 4px; }}
  .ctrl-row input[type=range] {{ width: 90px; }}
  .btn {{ background: #2a2a4a; border: 1px solid #555; color: #ccc; padding: 4px 8px;
    border-radius: 3px; cursor: pointer; font-size: 11px; }}
  .btn:hover {{ background: #3a3a5a; }}
  #cluster-status {{ font-size: 10px; color: #888; margin-top: 2px; }}
</style>
</head>
<body>
<div id="main">
  <div id="grid"></div>
  <div id="sidebar">
    <div>
      <h3>{title}</h3>
      <div id="leg-items"></div>
      <div style="margin-top:6px;display:flex;gap:4px;">
        <button class="btn" id="btn-all">Show All</button>
        <button class="btn" id="btn-none">Hide All</button>
      </div>
    </div>
    <div class="ctrl-group">
      <h3>Clustering</h3>
      <div class="ctrl-row">
        <span>Eps scale</span>
        <input type="range" id="eps-slider" min="0.25" max="3.0" value="1.0" step="0.05">
        <span id="eps-val">1.00</span><span>x</span>
      </div>
      <div class="ctrl-row">
        <span>Min cells</span>
        <input type="range" id="min-cells-slider" min="3" max="50" value="{default_min_cells}" step="1">
        <span id="min-cells-val">{default_min_cells}</span>
      </div>
      <div id="cluster-status"></div>
    </div>
    <div class="ctrl-group">
      <h3>Display</h3>
      <div class="ctrl-row">
        <span>Dot size</span>
        <input type="range" id="dot-size" min="1" max="10" value="3" step="0.5">
        <span id="dot-val">3</span>
      </div>
      <div class="ctrl-row">
        <span>Opacity</span>
        <input type="range" id="opacity" min="0.1" max="1" value="0.8" step="0.05">
        <span id="op-val">0.80</span>
      </div>
      <div class="ctrl-row">
        <span>Hulls</span>
        <input type="checkbox" id="show-hulls" checked>
        <span>Labels</span>
        <input type="checkbox" id="show-labels" checked>
      </div>
      <div class="ctrl-row">
        <button class="btn" id="btn-reset">Reset Zoom</button>
      </div>
    </div>
    <div class="ctrl-group" style="font-size:10px;color:#666;">
      Scroll to zoom, drag to pan.<br>
      Click legend to toggle groups.<br>
      Eps auto-computed per group<br>
      via KNN knee method. Slider<br>
      scales all eps values together.
    </div>
  </div>
</div>
<script>
const SCENES = [{scenes_js}];
const LEGEND = [{legend_js}];
const MIN_HULL = {default_min_hull};

// --- DBSCAN with grid spatial index ---
function dbscan(x, y, n, eps, minPts) {{
  // Returns Int32Array of cluster labels (-1 = noise)
  const labels = new Int32Array(n).fill(-1);
  if (n === 0 || eps <= 0) return labels;

  // Build spatial grid (cell size = eps)
  const grid = new Map();
  for (let i = 0; i < n; i++) {{
    const gx = Math.floor(x[i] / eps);
    const gy = Math.floor(y[i] / eps);
    const key = gx * 100003 + gy;  // hash to single number (faster than string)
    let cell = grid.get(key);
    if (!cell) {{ cell = []; grid.set(key, cell); }}
    cell.push(i);
  }}

  const eps2 = eps * eps;  // compare squared distances
  function getNeighbors(idx) {{
    const px = x[idx], py = y[idx];
    const gx = Math.floor(px / eps);
    const gy = Math.floor(py / eps);
    const result = [];
    for (let dx = -1; dx <= 1; dx++) {{
      for (let dy = -1; dy <= 1; dy++) {{
        const cell = grid.get((gx + dx) * 100003 + (gy + dy));
        if (!cell) continue;
        for (let k = 0; k < cell.length; k++) {{
          const j = cell[k];
          const ddx = x[j] - px, ddy = y[j] - py;
          if (ddx * ddx + ddy * ddy <= eps2) result.push(j);
        }}
      }}
    }}
    return result;
  }}

  let clusterId = 0;
  const visited = new Uint8Array(n);

  for (let i = 0; i < n; i++) {{
    if (visited[i]) continue;
    visited[i] = 1;
    const nbrs = getNeighbors(i);
    if (nbrs.length < minPts) continue;  // not a core point

    // Expand cluster
    labels[i] = clusterId;
    const queue = [];
    for (let k = 0; k < nbrs.length; k++) {{
      if (nbrs[k] !== i) queue.push(nbrs[k]);
    }}
    let qi = 0;
    while (qi < queue.length) {{
      const j = queue[qi++];
      if (!visited[j]) {{
        visited[j] = 1;
        const jnbrs = getNeighbors(j);
        if (jnbrs.length >= minPts) {{
          for (let k = 0; k < jnbrs.length; k++) {{
            if (!visited[jnbrs[k]]) queue.push(jnbrs[k]);
          }}
        }}
      }}
      if (labels[j] === -1) labels[j] = clusterId;
    }}
    clusterId++;
  }}
  return labels;
}}

// --- Convex hull (Andrew's monotone chain) ---
function convexHull(points) {{
  // points: array of [x,y]. Returns hull vertices in order.
  const n = points.length;
  if (n < 3) return points.slice();
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  // Remove duplicates
  const pts = [points[0]];
  for (let i = 1; i < n; i++) {{
    if (points[i][0] !== points[i-1][0] || points[i][1] !== points[i-1][1])
      pts.push(points[i]);
  }}
  if (pts.length < 3) return pts;

  function cross(O, A, B) {{
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
  }}
  const lower = [];
  for (const p of pts) {{
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop();
    lower.push(p);
  }}
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {{
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop();
    upper.push(p);
  }}
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}}

// --- Re-cluster all groups in all scenes ---
function reclusterAll() {{
  const mult = parseFloat(document.getElementById('eps-slider').value);
  const minCells = parseInt(document.getElementById('min-cells-slider').value);
  let totalClusters = 0, totalHulls = 0;
  let epsMin = Infinity, epsMax = 0;
  const t0 = performance.now();

  for (const scene of SCENES) {{
    for (const group of scene.groups) {{
      const eps = group.autoEps * mult;
      if (eps < epsMin) epsMin = eps;
      if (eps > epsMax) epsMax = eps;
      const labels = dbscan(group.x, group.y, group.n, eps, minCells);

      // Collect clusters
      const clusterMap = new Map();  // clusterId -> [indices]
      for (let i = 0; i < group.n; i++) {{
        const cl = labels[i];
        if (cl === -1) continue;
        let arr = clusterMap.get(cl);
        if (!arr) {{ arr = []; clusterMap.set(cl, arr); }}
        arr.push(i);
      }}

      group.clusters = [];
      let num = 0;
      for (const [clId, indices] of clusterMap) {{
        num++;
        totalClusters++;
        // Collect positions for hull + sum area
        const pts = [];
        let sx = 0, sy = 0, totalArea = 0;
        for (const idx of indices) {{
          const px = group.x[idx], py = group.y[idx];
          pts.push([px, py]);
          sx += px; sy += py;
          totalArea += group.a[idx];
        }}
        const cx = sx / indices.length;
        const cy = sy / indices.length;

        let hull = [];
        if (indices.length >= MIN_HULL) {{
          hull = convexHull(pts);
          if (hull.length >= 3) totalHulls++;
          else hull = [];
        }}

        group.clusters.push({{
          label: group.label + ' #' + num,
          n: indices.length,
          areaUm2: totalArea,
          hull: hull,
          cx: cx,
          cy: cy,
        }});
      }}
    }}
  }}

  const dt = (performance.now() - t0).toFixed(0);
  const epsRange = epsMin === Infinity ? '' :
    ' | eps ' + Math.round(epsMin) + '-' + Math.round(epsMax) + ' um';
  document.getElementById('cluster-status').textContent =
    totalClusters + ' clusters (' + totalHulls + ' hulls) ' + dt + 'ms' + epsRange;
}}

// State
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.8, showHulls = true, showLabels = true;

// Panel state array
const panels = [];
let activePanel = null;
let rafId = 0;
let rafDirty = new Set();

function scheduleRender(p) {{
  rafDirty.add(p);
  if (!rafId) {{
    rafId = requestAnimationFrame(() => {{
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    }});
  }}
}}

function scheduleRenderAll() {{
  panels.forEach(p => rafDirty.add(p));
  if (!rafId) {{
    rafId = requestAnimationFrame(() => {{
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    }});
  }}
}}

function initPanels() {{
  const grid = document.getElementById('grid');
  SCENES.forEach((scene, idx) => {{
    const div = document.createElement('div');
    div.className = 'panel';
    const labelEl = document.createElement('div');
    labelEl.className = 'panel-label';
    labelEl.textContent = scene.name;
    const countEl = document.createElement('div');
    countEl.className = 'panel-count';
    const canvas = document.createElement('canvas');
    div.appendChild(labelEl);
    div.appendChild(countEl);
    div.appendChild(canvas);
    grid.appendChild(div);

    const ctx = canvas.getContext('2d');
    const state = {{
      div, canvas, ctx, countEl, scene,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
    }};
    panels.push(state);

    div.addEventListener('mousedown', e => {{
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      activePanel = state;
      div.classList.add('dragging');
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.panStartX = state.panX;
      state.panStartY = state.panY;
      e.preventDefault();
    }});

    div.addEventListener('wheel', e => {{
      e.preventDefault();
      const rect = div.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
      state.panX = mx - factor * (mx - state.panX);
      state.panY = my - factor * (my - state.panY);
      state.zoom *= factor;
      state.zoom = Math.max(0.01, Math.min(200, state.zoom));
      scheduleRender(state);
    }}, {{passive: false}});
  }});

  window.addEventListener('mousemove', e => {{
    if (!activePanel) return;
    activePanel.panX = activePanel.panStartX + (e.clientX - activePanel.dragStartX);
    activePanel.panY = activePanel.panStartY + (e.clientY - activePanel.dragStartY);
    scheduleRender(activePanel);
  }});
  window.addEventListener('mouseup', () => {{
    if (activePanel) {{
      activePanel.div.classList.remove('dragging');
      activePanel = null;
    }}
  }});
}}

function resizePanels() {{
  const dpr = window.devicePixelRatio || 1;
  panels.forEach(p => {{
    const rect = p.div.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    p.cw = w; p.ch = h;
    p.canvas.width = w * dpr;
    p.canvas.height = h * dpr;
    p.canvas.style.width = w + 'px';
    p.canvas.style.height = h + 'px';
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }});
}}

function fitPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const s = p.scene;
  const dataW = s.xr[1] - s.xr[0];
  const dataH = s.yr[1] - s.yr[0];
  if (dataW <= 0 || dataH <= 0) {{
    p.zoom = 1;
    p.panX = cw / 2 - s.xr[0];
    p.panY = ch / 2 - s.yr[0];
    return;
  }}
  const pad = 0.05;
  p.zoom = Math.min(cw / (dataW * (1 + 2*pad)), ch / (dataH * (1 + 2*pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}}

function renderPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const ctx = p.ctx;

  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);

  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  const r = dotSize / p.zoom;
  const halfR = r / 2;
  let total = 0;

  for (const group of p.scene.groups) {{
    if (hidden.has(group.label)) continue;

    // Draw dots
    ctx.globalAlpha = dotAlpha;
    ctx.fillStyle = group.color;
    const n = group.n;
    const gx = group.x, gy = group.y;
    for (let i = 0; i < n; i++) {{
      ctx.fillRect(gx[i] - halfR, gy[i] - halfR, r, r);
    }}
    total += n;

    // Draw cluster hulls
    if (showHulls && group.clusters) {{
      ctx.globalAlpha = 1;
      for (const cl of group.clusters) {{
        if (!cl.hull || cl.hull.length < 3) continue;
        const path = new Path2D();
        path.moveTo(cl.hull[0][0], cl.hull[0][1]);
        for (let i = 1; i < cl.hull.length; i++) {{
          path.lineTo(cl.hull[i][0], cl.hull[i][1]);
        }}
        path.closePath();

        ctx.setLineDash([6/p.zoom, 4/p.zoom]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2.5 / p.zoom;
        ctx.stroke(path);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.2 / p.zoom;
        ctx.stroke(path);
        ctx.setLineDash([]);

        if (showLabels) {{
          const fontSize = 11 / p.zoom;
          ctx.font = fontSize + 'px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const areaStr = cl.areaUm2 >= 1000
            ? (cl.areaUm2 / 1000).toFixed(1) + 'K'
            : Math.round(cl.areaUm2).toString();
          const line1 = cl.n + ' cells';
          const line2 = areaStr + ' um\u00B2';
          const lh = fontSize * 1.2;
          ctx.fillStyle = '#000';
          ctx.fillText(line1, cl.cx + 0.5/p.zoom, cl.cy - lh/2 + 0.5/p.zoom);
          ctx.fillText(line2, cl.cx + 0.5/p.zoom, cl.cy + lh/2 + 0.5/p.zoom);
          ctx.fillStyle = '#fff';
          ctx.fillText(line1, cl.cx, cl.cy - lh/2);
          ctx.fillText(line2, cl.cx, cl.cy + lh/2);
        }}
      }}
    }}
  }}

  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}}

function renderAll() {{
  panels.forEach(renderPanel);
}}

// --- Legend ---
function initLegend() {{
  const legDiv = document.getElementById('leg-items');
  const groupTotals = {{}};
  SCENES.forEach(s => {{
    s.groups.forEach(g => {{
      groupTotals[g.label] = (groupTotals[g.label] || 0) + g.n;
    }});
  }});

  LEGEND.forEach(leg => {{
    const item = document.createElement('div');
    item.className = 'leg-item';
    const count = groupTotals[leg.label] || 0;
    item.innerHTML =
      '<span class="leg-dot" style="background:' + leg.color + '"></span>' +
      '<span class="leg-label" title="' + leg.label + '">' + leg.label +
      ' (' + count.toLocaleString() + ')</span>';
    item.onclick = () => {{
      if (hidden.has(leg.label)) hidden.delete(leg.label);
      else hidden.add(leg.label);
      item.classList.toggle('hidden');
      renderAll();
    }};
    legDiv.appendChild(item);
  }});
}}

// --- Controls ---
let clusterTimer = null;
function initControls() {{
  // Clustering sliders — debounce re-clustering (runs DBSCAN on release)
  const epsSlider = document.getElementById('eps-slider');
  const minCellsSlider = document.getElementById('min-cells-slider');

  epsSlider.oninput = e => {{
    document.getElementById('eps-val').textContent = parseFloat(e.target.value).toFixed(2);
  }};
  epsSlider.onchange = e => {{
    reclusterAll();
    renderAll();
  }};
  minCellsSlider.oninput = e => {{
    document.getElementById('min-cells-val').textContent = e.target.value;
  }};
  minCellsSlider.onchange = e => {{
    reclusterAll();
    renderAll();
  }};

  document.getElementById('dot-size').oninput = e => {{
    dotSize = parseFloat(e.target.value);
    document.getElementById('dot-val').textContent = dotSize;
    scheduleRenderAll();
  }};
  document.getElementById('opacity').oninput = e => {{
    dotAlpha = parseFloat(e.target.value);
    document.getElementById('op-val').textContent = dotAlpha.toFixed(2);
    scheduleRenderAll();
  }};
  document.getElementById('show-hulls').onchange = e => {{
    showHulls = e.target.checked;
    renderAll();
  }};
  document.getElementById('show-labels').onchange = e => {{
    showLabels = e.target.checked;
    renderAll();
  }};
  document.getElementById('btn-all').onclick = () => {{
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    renderAll();
  }};
  document.getElementById('btn-none').onclick = () => {{
    LEGEND.forEach(l => hidden.add(l.label));
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    renderAll();
  }};
  document.getElementById('btn-reset').onclick = () => {{
    panels.forEach(fitPanel);
    resizePanels();
    renderAll();
  }};
}}

// --- Init ---
initPanels();
initLegend();
initControls();

function fullInit() {{
  resizePanels();
  panels.forEach(fitPanel);
  reclusterAll();
  renderAll();
}}

setTimeout(fullInit, 50);
window.addEventListener('resize', () => {{
  resizePanels();
  scheduleRenderAll();
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)


def build_legend_items(scenes_data, group_field):
    """Build ordered legend items from scenes_data.

    For expression_group mode, uses GROUP_ORDER/GROUP_COLORS.
    For other modes, collects all groups that appear in the data (post top-N).

    Returns list of (label, color) tuples.
    """
    if group_field == 'expression_group':
        return [(label, GROUP_COLORS[label]) for label in GROUP_ORDER]

    # Collect unique groups in order of first appearance (top groups first, "other" last)
    seen = {}
    for _, data in scenes_data:
        for g in data['groups']:
            if g['label'] not in seen:
                seen[g['label']] = g['color']

    # "other" always last
    items = [(lbl, color) for lbl, color in seen.items() if lbl != 'other']
    if 'other' in seen:
        items.append(('other', seen['other']))
    return items


def generate_html_two_stage(scenes_data, output_path, title='Two-Stage Cluster Viewer',
                            default_min_cells=10, default_min_hull=24):
    """Generate HTML with two-stage clustering: appearance (UMAP DBSCAN) then spatial.

    Data is embedded as flat per-scene cell arrays (x, y, ux, uy, area). All grouping
    happens client-side in JavaScript: stage 1 clusters on UMAP appearance coordinates,
    stage 2 clusters spatially within each appearance group.

    Args:
        scenes_data: List of (label, data_dict) tuples from load_file_data_two_stage().
        output_path: Output HTML file path.
        title: Page title and sidebar heading.
        default_min_cells: Default DBSCAN min_samples for both stages.
        default_min_hull: Min cells in spatial cluster to draw convex hull.
    """
    # Serialize scene data to compact JS (flat cell arrays)
    scenes_js_parts = []
    for label, data in scenes_data:
        x_str = ','.join(f'{v:.1f}' for v in data['x'])
        y_str = ','.join(f'{v:.1f}' for v in data['y'])
        ux_str = ','.join(f'{v:.4f}' for v in data['ux'])
        uy_str = ','.join(f'{v:.4f}' for v in data['uy'])
        a_str = ','.join(f'{v:.1f}' for v in data['a'])

        scenes_js_parts.append(
            f'{{name:{json.dumps(label)},n:{data["n"]},'
            f'x:new Float32Array([{x_str}]),'
            f'y:new Float32Array([{y_str}]),'
            f'ux:new Float32Array([{ux_str}]),'
            f'uy:new Float32Array([{uy_str}]),'
            f'a:new Float32Array([{a_str}]),'
            f'xr:[{data["x_range"][0]:.1f},{data["x_range"][1]:.1f}],'
            f'yr:[{data["y_range"][0]:.1f},{data["y_range"][1]:.1f}],'
            f'autoEpsUmap:{data["auto_eps_umap"]:.4f},'
            f'autoEpsSpatial:{data["auto_eps_spatial"]:.1f},'
            f'groups:[]}}'
        )

    scenes_js = ',\n'.join(scenes_js_parts)

    n_scenes = len(scenes_data)
    n_cols = min(4, n_scenes)
    n_rows = (n_scenes + n_cols - 1) // n_cols

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }}
  #main {{ display: flex; width: 100vw; height: 100vh; }}
  #grid {{
    flex: 1;
    display: grid;
    grid-template-columns: repeat({n_cols}, 1fr);
    grid-template-rows: repeat({n_rows}, 1fr);
    gap: 2px;
    padding: 2px;
  }}
  .panel {{
    position: relative;
    overflow: hidden;
    background: #111122;
    border: 1px solid #333;
    border-radius: 4px;
    cursor: grab;
  }}
  .panel.dragging {{ cursor: grabbing; }}
  .panel canvas {{ position: absolute; top: 0; left: 0; }}
  .panel-label {{
    position: absolute; top: 4px; left: 6px; z-index: 10;
    font-size: 11px; color: #ccc; background: rgba(17,17,34,0.8);
    padding: 2px 6px; border-radius: 3px; pointer-events: none;
  }}
  .panel-count {{
    position: absolute; bottom: 4px; left: 6px; z-index: 10;
    font-size: 10px; color: #888; pointer-events: none;
  }}
  #sidebar {{
    width: 240px; min-width: 200px; background: rgba(26,26,46,0.95);
    border-left: 1px solid #444; overflow-y: auto; padding: 10px;
    display: flex; flex-direction: column; gap: 10px;
  }}
  #sidebar h3 {{ font-size: 13px; color: #aaa; margin-bottom: 4px; }}
  .leg-item {{
    display: flex; align-items: center; gap: 6px; padding: 3px 4px;
    cursor: pointer; border-radius: 3px; user-select: none; font-size: 11px;
  }}
  .leg-item:hover {{ background: rgba(255,255,255,0.08); }}
  .leg-item.hidden {{ opacity: 0.25; text-decoration: line-through; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-label {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .ctrl-group {{ border-top: 1px solid #333; padding-top: 8px; }}
  .ctrl-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; margin-bottom: 4px; }}
  .ctrl-row input[type=range] {{ width: 90px; }}
  .ctrl-lbl {{ min-width: 55px; }}
  .btn {{ background: #2a2a4a; border: 1px solid #555; color: #ccc; padding: 4px 8px;
    border-radius: 3px; cursor: pointer; font-size: 11px; }}
  .btn:hover {{ background: #3a3a5a; }}
  .status-line {{ font-size: 10px; color: #888; margin-top: 2px; }}
</style>
</head>
<body>
<div id="main">
  <div id="grid"></div>
  <div id="sidebar">
    <div><h3>{title}</h3></div>
    <div class="ctrl-group">
      <h3>Appearance Clustering</h3>
      <div class="ctrl-row">
        <span class="ctrl-lbl">Eps</span>
        <input type="range" id="u-eps" min="0.10" max="20.00" value="1.00" step="0.10">
        <span id="u-eps-val">1.00</span><span>x</span>
      </div>
      <div class="ctrl-row">
        <span class="ctrl-lbl">Min cells</span>
        <input type="range" id="u-min" min="3" max="100" value="{default_min_cells}" step="1">
        <span id="u-min-val">{default_min_cells}</span>
      </div>
      <div id="u-status" class="status-line"></div>
    </div>
    <div class="ctrl-group">
      <h3>Spatial Clustering</h3>
      <div class="ctrl-row">
        <span class="ctrl-lbl">Eps scale</span>
        <input type="range" id="s-eps" min="0.10" max="20.00" value="1.00" step="0.10">
        <span id="s-eps-val">1.00</span><span>x</span>
      </div>
      <div class="ctrl-row">
        <span class="ctrl-lbl">Min cells</span>
        <input type="range" id="s-min" min="3" max="100" value="{default_min_cells}" step="1">
        <span id="s-min-val">{default_min_cells}</span>
      </div>
      <div id="s-status" class="status-line"></div>
    </div>
    <div class="ctrl-group">
      <h3>Display</h3>
      <div class="ctrl-row">
        <span class="ctrl-lbl">Dot size</span>
        <input type="range" id="dot-size" min="1" max="10" value="3" step="0.5">
        <span id="dot-val">3</span>
      </div>
      <div class="ctrl-row">
        <span class="ctrl-lbl">Opacity</span>
        <input type="range" id="opacity" min="0.1" max="1" value="0.8" step="0.05">
        <span id="op-val">0.80</span>
      </div>
      <div class="ctrl-row">
        <span>Hulls</span>
        <input type="checkbox" id="show-hulls" checked>
        <span>Labels</span>
        <input type="checkbox" id="show-labels" checked>
      </div>
      <div class="ctrl-row">
        <button class="btn" id="btn-reset">Reset Zoom</button>
      </div>
    </div>
    <div class="ctrl-group">
      <div id="leg-items"></div>
      <div style="margin-top:6px;display:flex;gap:4px;">
        <button class="btn" id="btn-all">Show All</button>
        <button class="btn" id="btn-none">Hide All</button>
      </div>
    </div>
    <div class="ctrl-group" style="font-size:10px;color:#666;">
      Stage 1: DBSCAN on UMAP appearance.<br>
      Stage 2: spatial DBSCAN per group.<br>
      Scroll to zoom, drag to pan.<br>
      Click legend to toggle groups.
    </div>
  </div>
</div>
<script>
"use strict";
const SCENES = [{scenes_js}];
const PALETTE = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
  '#42d4f4','#f032e6','#bfef45','#fabebe','#469990',
  '#e6beff','#9a6324','#ffe119','#aaffc3','#800000',
  '#ffd8b1','#000075','#a9a9a9','#808000','#ff69b4'];
const NOISE_COLOR = '#555555';
const MIN_HULL = {default_min_hull};

// --- DBSCAN with grid spatial index ---
function dbscan(x, y, n, eps, minPts) {{
  const labels = new Int32Array(n).fill(-1);
  if (n === 0 || eps <= 0) return labels;
  const grid = new Map();
  for (let i = 0; i < n; i++) {{
    const gx = Math.floor(x[i] / eps);
    const gy = Math.floor(y[i] / eps);
    const key = gx * 100003 + gy;
    let cell = grid.get(key);
    if (!cell) {{ cell = []; grid.set(key, cell); }}
    cell.push(i);
  }}
  const eps2 = eps * eps;
  function getNeighbors(idx) {{
    const px = x[idx], py = y[idx];
    const gx = Math.floor(px / eps);
    const gy = Math.floor(py / eps);
    const result = [];
    for (let dx = -1; dx <= 1; dx++) {{
      for (let dy = -1; dy <= 1; dy++) {{
        const cell = grid.get((gx + dx) * 100003 + (gy + dy));
        if (!cell) continue;
        for (let k = 0; k < cell.length; k++) {{
          const j = cell[k];
          const ddx = x[j] - px, ddy = y[j] - py;
          if (ddx * ddx + ddy * ddy <= eps2) result.push(j);
        }}
      }}
    }}
    return result;
  }}
  let clusterId = 0;
  const visited = new Uint8Array(n);
  for (let i = 0; i < n; i++) {{
    if (visited[i]) continue;
    visited[i] = 1;
    const nbrs = getNeighbors(i);
    if (nbrs.length < minPts) continue;
    labels[i] = clusterId;
    const queue = [];
    for (let k = 0; k < nbrs.length; k++) {{
      if (nbrs[k] !== i) queue.push(nbrs[k]);
    }}
    let qi = 0;
    while (qi < queue.length) {{
      const j = queue[qi++];
      if (!visited[j]) {{
        visited[j] = 1;
        const jnbrs = getNeighbors(j);
        if (jnbrs.length >= minPts) {{
          for (let k = 0; k < jnbrs.length; k++) {{
            if (!visited[jnbrs[k]]) queue.push(jnbrs[k]);
          }}
        }}
      }}
      if (labels[j] === -1) labels[j] = clusterId;
    }}
    clusterId++;
  }}
  return labels;
}}

// --- Convex hull (Andrew's monotone chain) ---
function convexHull(points) {{
  const n = points.length;
  if (n < 3) return points.slice();
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const pts = [points[0]];
  for (let i = 1; i < n; i++) {{
    if (points[i][0] !== points[i-1][0] || points[i][1] !== points[i-1][1])
      pts.push(points[i]);
  }}
  if (pts.length < 3) return pts;
  function cross(O, A, B) {{
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
  }}
  const lower = [];
  for (const p of pts) {{
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop();
    lower.push(p);
  }}
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {{
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop();
    upper.push(p);
  }}
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}}

// --- State ---
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.8, showHulls = true, showLabels = true;
const panels = [];
let activePanel = null;
let rafId = 0;
let rafDirty = new Set();

function scheduleRender(p) {{
  rafDirty.add(p);
  if (!rafId) {{
    rafId = requestAnimationFrame(() => {{
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    }});
  }}
}}

function scheduleRenderAll() {{
  panels.forEach(p => rafDirty.add(p));
  if (!rafId) {{
    rafId = requestAnimationFrame(() => {{
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    }});
  }}
}}

// --- Spatial DBSCAN (stage 2 only, preserves appearance groups) ---
function reclusterSpatial() {{
  const sMult = parseFloat(document.getElementById('s-eps').value);
  const sMinCells = parseInt(document.getElementById('s-min').value);
  let totalClusters = 0, totalHulls = 0;
  const t0 = performance.now();

  for (const scene of SCENES) {{
    const sEps = scene.autoEpsSpatial * sMult;
    for (const g of scene.groups) {{
      const gn = g.indices.length;
      const gx = new Float32Array(gn);
      const gy = new Float32Array(gn);
      const ga = new Float32Array(gn);
      for (let j = 0; j < gn; j++) {{
        const idx = g.indices[j];
        gx[j] = scene.x[idx];
        gy[j] = scene.y[idx];
        ga[j] = scene.a[idx];
      }}

      const sLabels = dbscan(gx, gy, gn, sEps, sMinCells);
      const clusterMap = new Map();
      for (let j = 0; j < gn; j++) {{
        if (sLabels[j] === -1) continue;
        let arr = clusterMap.get(sLabels[j]);
        if (!arr) {{ arr = []; clusterMap.set(sLabels[j], arr); }}
        arr.push(j);
      }}

      g.clusters = [];
      let num = 0;
      for (const [clId, localIndices] of clusterMap) {{
        num++;
        totalClusters++;
        const pts = [];
        let sx = 0, sy = 0, ta = 0;
        for (const li of localIndices) {{
          pts.push([gx[li], gy[li]]);
          sx += gx[li]; sy += gy[li];
          ta += ga[li];
        }}
        const cx = sx / localIndices.length;
        const cy = sy / localIndices.length;
        let hull = [];
        if (localIndices.length >= MIN_HULL) {{
          hull = convexHull(pts);
          if (hull.length >= 3) totalHulls++;
          else hull = [];
        }}
        g.clusters.push({{
          label: g.label + ' #' + num,
          n: localIndices.length,
          areaUm2: ta,
          hull: hull,
          cx: cx, cy: cy,
        }});
      }}
    }}
  }}

  const dt = (performance.now() - t0).toFixed(0);
  const sEpsDisplay = Math.round(SCENES[0].autoEpsSpatial * sMult);
  document.getElementById('s-status').textContent =
    totalClusters + ' clusters (' + totalHulls + ' hulls) eps=' + sEpsDisplay + 'um ' + dt + 'ms';
}}

// --- Full two-stage re-cluster ---
function reclusterAll() {{
  const uMult = parseFloat(document.getElementById('u-eps').value);
  const uMinCells = parseInt(document.getElementById('u-min').value);
  let totalGroups = 0, totalNoise = 0;

  for (const scene of SCENES) {{
    // Stage 1: appearance DBSCAN on UMAP coords
    const uEps = scene.autoEpsUmap * uMult;
    const labels = dbscan(scene.ux, scene.uy, scene.n, uEps, uMinCells);

    // Collect groups by appearance cluster label
    const groupMap = new Map();
    for (let i = 0; i < scene.n; i++) {{
      const cl = labels[i];
      let g = groupMap.get(cl);
      if (!g) {{
        g = {{label: cl === -1 ? 'noise' : '', indices: [], isNoise: cl === -1}};
        groupMap.set(cl, g);
      }}
      g.indices.push(i);
    }}

    // Sort non-noise groups by size descending
    const sorted = [];
    for (const [k, g] of groupMap) {{
      if (!g.isNoise) sorted.push(g);
    }}
    sorted.sort((a, b) => b.indices.length - a.indices.length);

    // Assign labels and colors
    scene.groups = [];
    for (let i = 0; i < sorted.length; i++) {{
      const g = sorted[i];
      g.label = 'Group ' + (i + 1);
      g.color = i < PALETTE.length ? PALETTE[i] : '#555555';
      scene.groups.push(g);
      totalGroups++;
    }}

    // Add noise group last
    const noiseGroup = groupMap.get(-1);
    if (noiseGroup && noiseGroup.indices.length > 0) {{
      noiseGroup.color = NOISE_COLOR;
      scene.groups.push(noiseGroup);
      totalNoise += noiseGroup.indices.length;
    }}
  }}

  const uEpsDisplay = (SCENES[0].autoEpsUmap * uMult).toFixed(3);
  document.getElementById('u-status').textContent =
    totalGroups + ' groups, ' + totalNoise + ' noise (eps=' + uEpsDisplay + ')';

  rebuildLegend();

  // Stage 2: spatial
  reclusterSpatial();
}}

// --- Dynamic legend ---
function rebuildLegend() {{
  const oldHidden = new Set(hidden);
  const legDiv = document.getElementById('leg-items');
  legDiv.innerHTML = '';
  hidden.clear();

  // Collect unique groups across all scenes (by label)
  const seen = new Map();
  for (const scene of SCENES) {{
    for (const g of scene.groups) {{
      const existing = seen.get(g.label);
      if (existing) {{
        existing.count += g.indices.length;
      }} else {{
        seen.set(g.label, {{label: g.label, color: g.color, count: g.indices.length}});
      }}
    }}
  }}

  // Re-apply hidden state for labels that survived re-clustering (e.g. "noise")
  for (const label of oldHidden) {{
    if (seen.has(label)) hidden.add(label);
  }}

  for (const [label, info] of seen) {{
    const item = document.createElement('div');
    item.className = 'leg-item' + (hidden.has(label) ? ' hidden' : '');
    item.innerHTML =
      '<span class="leg-dot" style="background:' + info.color + '"></span>' +
      '<span class="leg-label" title="' + label + '">' + label +
      ' (' + info.count.toLocaleString() + ')</span>';
    item.onclick = () => {{
      if (hidden.has(label)) hidden.delete(label);
      else hidden.add(label);
      item.classList.toggle('hidden');
      renderAll();
    }};
    legDiv.appendChild(item);
  }}
}}

// --- Panels ---
function initPanels() {{
  const grid = document.getElementById('grid');
  SCENES.forEach((scene, idx) => {{
    const div = document.createElement('div');
    div.className = 'panel';
    const labelEl = document.createElement('div');
    labelEl.className = 'panel-label';
    labelEl.textContent = scene.name;
    const countEl = document.createElement('div');
    countEl.className = 'panel-count';
    const canvas = document.createElement('canvas');
    div.appendChild(labelEl);
    div.appendChild(countEl);
    div.appendChild(canvas);
    grid.appendChild(div);

    const ctx = canvas.getContext('2d');
    const state = {{
      div, canvas, ctx, countEl, scene,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
    }};
    panels.push(state);

    div.addEventListener('mousedown', e => {{
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      activePanel = state;
      div.classList.add('dragging');
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.panStartX = state.panX;
      state.panStartY = state.panY;
      e.preventDefault();
    }});

    div.addEventListener('wheel', e => {{
      e.preventDefault();
      const rect = div.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
      state.panX = mx - factor * (mx - state.panX);
      state.panY = my - factor * (my - state.panY);
      state.zoom *= factor;
      state.zoom = Math.max(0.01, Math.min(200, state.zoom));
      scheduleRender(state);
    }}, {{passive: false}});
  }});

  window.addEventListener('mousemove', e => {{
    if (!activePanel) return;
    activePanel.panX = activePanel.panStartX + (e.clientX - activePanel.dragStartX);
    activePanel.panY = activePanel.panStartY + (e.clientY - activePanel.dragStartY);
    scheduleRender(activePanel);
  }});
  window.addEventListener('mouseup', () => {{
    if (activePanel) {{
      activePanel.div.classList.remove('dragging');
      activePanel = null;
    }}
  }});
}}

function resizePanels() {{
  const dpr = window.devicePixelRatio || 1;
  panels.forEach(p => {{
    const rect = p.div.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    p.cw = w; p.ch = h;
    p.canvas.width = w * dpr;
    p.canvas.height = h * dpr;
    p.canvas.style.width = w + 'px';
    p.canvas.style.height = h + 'px';
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }});
}}

function fitPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const s = p.scene;
  const dataW = s.xr[1] - s.xr[0];
  const dataH = s.yr[1] - s.yr[0];
  if (dataW <= 0 || dataH <= 0) {{
    p.zoom = 1;
    p.panX = cw / 2 - s.xr[0];
    p.panY = ch / 2 - s.yr[0];
    return;
  }}
  const pad = 0.05;
  p.zoom = Math.min(cw / (dataW * (1 + 2*pad)), ch / (dataH * (1 + 2*pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}}

function renderPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const ctx = p.ctx;

  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);

  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  const r = dotSize / p.zoom;
  const halfR = r / 2;
  let total = 0;
  const scene = p.scene;

  for (const group of scene.groups) {{
    if (hidden.has(group.label)) continue;

    // Draw dots
    ctx.globalAlpha = dotAlpha;
    ctx.fillStyle = group.color;
    const indices = group.indices;
    for (let j = 0; j < indices.length; j++) {{
      const idx = indices[j];
      ctx.fillRect(scene.x[idx] - halfR, scene.y[idx] - halfR, r, r);
    }}
    total += indices.length;

    // Draw spatial cluster hulls
    if (showHulls && group.clusters) {{
      ctx.globalAlpha = 1;
      for (const cl of group.clusters) {{
        if (!cl.hull || cl.hull.length < 3) continue;
        const path = new Path2D();
        path.moveTo(cl.hull[0][0], cl.hull[0][1]);
        for (let i = 1; i < cl.hull.length; i++) {{
          path.lineTo(cl.hull[i][0], cl.hull[i][1]);
        }}
        path.closePath();

        ctx.setLineDash([6/p.zoom, 4/p.zoom]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2.5 / p.zoom;
        ctx.stroke(path);
        ctx.strokeStyle = group.color;
        ctx.lineWidth = 1.2 / p.zoom;
        ctx.stroke(path);
        ctx.setLineDash([]);

        if (showLabels) {{
          const fontSize = 11 / p.zoom;
          ctx.font = fontSize + 'px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const areaStr = cl.areaUm2 >= 1000
            ? (cl.areaUm2 / 1000).toFixed(1) + 'K'
            : Math.round(cl.areaUm2).toString();
          const line1 = cl.n + ' cells';
          const line2 = areaStr + ' um\u00B2';
          const lh = fontSize * 1.2;
          ctx.fillStyle = '#000';
          ctx.fillText(line1, cl.cx + 0.5/p.zoom, cl.cy - lh/2 + 0.5/p.zoom);
          ctx.fillText(line2, cl.cx + 0.5/p.zoom, cl.cy + lh/2 + 0.5/p.zoom);
          ctx.fillStyle = '#fff';
          ctx.fillText(line1, cl.cx, cl.cy - lh/2);
          ctx.fillText(line2, cl.cx, cl.cy + lh/2);
        }}
      }}
    }}
  }}

  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}}

function renderAll() {{
  panels.forEach(renderPanel);
}}

// --- Controls ---
function initControls() {{
  // Appearance clustering sliders
  const uEps = document.getElementById('u-eps');
  const uMin = document.getElementById('u-min');
  uEps.oninput = e => {{
    document.getElementById('u-eps-val').textContent = parseFloat(e.target.value).toFixed(2);
  }};
  uEps.onchange = () => {{ reclusterAll(); renderAll(); }};
  uMin.oninput = e => {{
    document.getElementById('u-min-val').textContent = e.target.value;
  }};
  uMin.onchange = () => {{ reclusterAll(); renderAll(); }};

  // Spatial clustering sliders
  const sEps = document.getElementById('s-eps');
  const sMin = document.getElementById('s-min');
  sEps.oninput = e => {{
    document.getElementById('s-eps-val').textContent = parseFloat(e.target.value).toFixed(2);
  }};
  sEps.onchange = () => {{ reclusterSpatial(); renderAll(); }};
  sMin.oninput = e => {{
    document.getElementById('s-min-val').textContent = e.target.value;
  }};
  sMin.onchange = () => {{ reclusterSpatial(); renderAll(); }};

  // Display controls
  document.getElementById('dot-size').oninput = e => {{
    dotSize = parseFloat(e.target.value);
    document.getElementById('dot-val').textContent = dotSize;
    scheduleRenderAll();
  }};
  document.getElementById('opacity').oninput = e => {{
    dotAlpha = parseFloat(e.target.value);
    document.getElementById('op-val').textContent = dotAlpha.toFixed(2);
    scheduleRenderAll();
  }};
  document.getElementById('show-hulls').onchange = e => {{
    showHulls = e.target.checked;
    renderAll();
  }};
  document.getElementById('show-labels').onchange = e => {{
    showLabels = e.target.checked;
    renderAll();
  }};
  document.getElementById('btn-all').onclick = () => {{
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    renderAll();
  }};
  document.getElementById('btn-none').onclick = () => {{
    for (const scene of SCENES) {{
      for (const g of scene.groups) hidden.add(g.label);
    }}
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    renderAll();
  }};
  document.getElementById('btn-reset').onclick = () => {{
    panels.forEach(fitPanel);
    resizePanels();
    renderAll();
  }};
}}

// --- Init ---
initPanels();
initControls();

function fullInit() {{
  resizePanels();
  panels.forEach(fitPanel);
  reclusterAll();
  renderAll();
}}

setTimeout(fullInit, 50);
window.addEventListener('resize', () => {{
  resizePanels();
  scheduleRenderAll();
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML viewer for grouped spatial detection data')

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir',
                             help='Directory containing sample subdirs (e.g., brain_fish_output)')
    input_group.add_argument('--detections', nargs='+', metavar='FILE',
                             help='Direct detection JSON file(s)')

    parser.add_argument('--subdir', default='zones_expr_spatial_v2',
                        help='Subdirectory within each scene (--input-dir mode, default: zones_expr_spatial_v2)')
    parser.add_argument('--group-field', default='expression_group',
                        help='Field to group by: expression_group, cluster_label, cluster_id, '
                             'zone_label, or any detection field (default: expression_group)')
    parser.add_argument('--top-n', type=int, default=12,
                        help='Keep top N groups by cell count, lump rest into "other" (default: 12)')
    parser.add_argument('--exclude-groups', default='',
                        help='Comma-separated group labels to drop entirely (e.g., "noise,-1")')
    parser.add_argument('--output', default=None,
                        help='Output HTML path (default: auto based on input)')
    parser.add_argument('--title', default=None,
                        help='Page title (default: auto based on group-field)')
    parser.add_argument('--default-min-cells', type=int, default=10,
                        help='Default DBSCAN min_samples (default: 10)')
    parser.add_argument('--min-hull-cells', type=int, default=24,
                        help='Minimum cells in cluster to draw hull outline (default: 24)')
    parser.add_argument('--two-stage', action='store_true',
                        help='Two-stage clustering: appearance (UMAP DBSCAN) then spatial (DBSCAN per group)')
    parser.add_argument('--umap-field', default='umap_x,umap_y',
                        help='Comma-separated UMAP coordinate fields (default: umap_x,umap_y)')

    args = parser.parse_args()
    group_field = args.group_field
    exclude_groups = set(g.strip() for g in args.exclude_groups.split(',') if g.strip())

    # Auto title
    title_map = {
        'expression_group': 'Expression Groups',
        'cluster_label': 'Cluster Labels',
        'cluster_id': 'Cluster IDs',
        'zone_label': 'Zone Labels',
    }
    title = args.title or title_map.get(group_field, group_field.replace('_', ' ').title())

    # --- Two-stage mode: appearance (UMAP) + spatial clustering ---
    if args.two_stage:
        if args.input_dir:
            print("ERROR: --two-stage requires --detections, not --input-dir", file=sys.stderr)
            sys.exit(1)

        umap_fields = tuple(f.strip() for f in args.umap_field.split(','))
        if len(umap_fields) != 2:
            print("ERROR: --umap-field must be 'field_x,field_y' (two comma-separated names)",
                  file=sys.stderr)
            sys.exit(1)

        det_files = [Path(f) for f in args.detections]
        output_path = Path(args.output) if args.output else det_files[0].parent / 'two_stage_viewer.html'

        scenes_data = []
        for fpath in det_files:
            if not fpath.exists():
                print(f"WARNING: File not found, skipping: {fpath}", file=sys.stderr)
                continue

            label = fpath.stem if len(det_files) == 1 else f"{fpath.parent.name}/{fpath.stem}"
            print(f"Loading {label} ({fpath})...", flush=True)
            data = load_file_data_two_stage(fpath, umap_fields=umap_fields)
            if data is None:
                print(f"  Skipped (no valid data or missing UMAP fields)", flush=True)
                continue

            print(f"  {data['n']} cells, autoEpsUmap={data['auto_eps_umap']:.2f}, "
                  f"autoEpsSpatial={data['auto_eps_spatial']:.1f}", flush=True)
            if data['skipped'] > 0:
                print(f"  Skipped {data['skipped']} detections (missing coords/UMAP)", flush=True)
            scenes_data.append((label, data))

        if not scenes_data:
            print("ERROR: No valid data loaded", file=sys.stderr)
            sys.exit(1)

        total_cells = sum(d['n'] for _, d in scenes_data)
        print(f"\nTotal: {total_cells} cells across {len(scenes_data)} panels", flush=True)
        print(f"Generating two-stage HTML...", flush=True)

        generate_html_two_stage(
            scenes_data, output_path,
            title=title,
            default_min_cells=args.default_min_cells,
            default_min_hull=args.min_hull_cells,
        )
        print("Done!", flush=True)
        return

    # --- Mode 1: Multi-scene directory ---
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output) if args.output else input_dir / 'expr_spatial_viewer.html'

        print(f"Scanning {input_dir} for scenes with {args.subdir}...", flush=True)
        scenes = discover_scenes(input_dir, args.subdir)
        print(f"  Found {len(scenes)} scenes", flush=True)

        if not scenes:
            print("ERROR: No scenes found with zone data", file=sys.stderr)
            sys.exit(1)

        scenes_data = []
        for label, scene_dir in scenes:
            print(f"Loading {label}...", flush=True)
            data = load_scene_data(scene_dir, args.subdir, group_field=group_field)
            if data is None:
                print(f"  Skipped (no valid data)", flush=True)
                continue
            print(f"  {data['n_cells']} cells, {len(data['groups'])} groups", flush=True)
            scenes_data.append((label, data))

    # --- Mode 2: Direct detection file(s) ---
    else:
        det_files = [Path(f) for f in args.detections]
        # Default output next to first file
        output_path = Path(args.output) if args.output else det_files[0].parent / 'spatial_viewer.html'

        scenes_data = []
        for fpath in det_files:
            if not fpath.exists():
                print(f"WARNING: File not found, skipping: {fpath}", file=sys.stderr)
                continue

            # Label: grandparent/parent/stem to show full context
            if len(det_files) == 1:
                label = fpath.stem
            else:
                label = f"{fpath.parent.parent.name}/{fpath.parent.name}/{fpath.stem}"

            print(f"Loading {label} ({fpath})...", flush=True)
            data = load_file_data(fpath, group_field)
            if data is None:
                print(f"  Skipped (no valid data)", flush=True)
                continue
            print(f"  {data['n_cells']} cells, {len(data['groups'])} groups", flush=True)
            scenes_data.append((label, data))

    if not scenes_data:
        print("ERROR: No valid data loaded", file=sys.stderr)
        sys.exit(1)

    # Apply top-N grouping and color assignment
    color_map = apply_top_n_colors(scenes_data, args.top_n, exclude_groups, group_field)

    # Recount after exclusions/merging
    total_cells = sum(data['n_cells'] for _, data in scenes_data)
    n_groups = len(set(g['label'] for _, data in scenes_data for g in data['groups']))
    print(f"\nTotal: {total_cells} cells, {n_groups} groups across {len(scenes_data)} panels",
          flush=True)

    if exclude_groups:
        print(f"  Excluded groups: {', '.join(sorted(exclude_groups))}", flush=True)

    # Build legend
    legend_items = build_legend_items(scenes_data, group_field)

    # Print per-group counts
    group_totals = {}
    for _, data in scenes_data:
        for g in data['groups']:
            group_totals[g['label']] = group_totals.get(g['label'], 0) + g['n']
    for lbl, color in legend_items:
        count = group_totals.get(lbl, 0)
        if count > 0:
            print(f"  {lbl}: {count:,} cells", flush=True)

    # Generate HTML
    print(f"Generating HTML...", flush=True)
    generate_html(scenes_data, output_path,
                  legend_items=legend_items,
                  title=title,
                  default_min_cells=args.default_min_cells,
                  default_min_hull=args.min_hull_cells)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
