"""Detection JSON streaming, position/group extraction, and slide discovery."""

import json
import mmap
import re
from pathlib import Path

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def compute_auto_eps(positions, k=10):
    """Compute optimal DBSCAN eps using KNN distance knee/elbow method.

    Builds a KDTree, queries the Kth nearest-neighbor distance for every point,
    sorts ascending, and finds the elbow (max deviation from the diagonal).
    """
    from scipy.spatial import KDTree

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
        return max(float(knn_dists[0]), 1.0)  # floor at 1 um
    y_norm = (knn_dists - knn_dists[0]) / y_range

    # Distance from diagonal (0,0)->(1,1) = (y - x) / sqrt(2), max of that
    diffs = y_norm - x_norm
    elbow_idx = int(np.argmax(diffs))
    return float(knn_dists[elbow_idx])


def extract_position_um(det):
    """Extract (x, y) position in microns from a detection dict.

    Delegates to the canonical ``extract_positions_um`` from
    ``xldvp_seg.utils.detection_utils`` for the actual extraction logic,
    which handles the 3-level fallback: global_center_um → global_center ×
    pixel_size → global_x/y × pixel_size.

    Returns (x, y) tuple or None if position unavailable.
    """
    from xldvp_seg.utils.detection_utils import extract_positions_um

    positions, _ = extract_positions_um([det])
    if len(positions) == 1:
        x, y = float(positions[0, 0]), float(positions[0, 1])
        if np.isfinite(x) and np.isfinite(y):
            return (x, y)
    return None


def extract_group(det, group_field):
    """Extract group label from a detection dict.

    Checks top-level dict first, then features sub-dict, falls back to
    'unknown' if the field is missing everywhere.
    """
    val = det.get(group_field)
    if val is None:
        val = det.get("features", {}).get(group_field)
    if val is None:
        return "unknown"
    return str(val)


def _stream_detections_mmap(filepath):
    """Stream detection dicts one at a time from a JSON array using mmap.

    Uses mmap to avoid reading the entire file into memory, and orjson to
    parse individual objects.  Peak memory is ~size of one detection dict
    (a few KB) + accumulated results, not the entire file.

    Uses re.finditer (C-level regex engine) to scan for structurally
    significant characters ({, }, ", backslash) — skips all other bytes
    at C speed, making this ~10-50x faster than a Python byte loop.
    """
    try:
        import orjson as _json_mod

        _parse = _json_mod.loads
    except ImportError:
        import json as _json_mod

        _parse = _json_mod.loads

    # Two-alternative pattern (order matters — escape sequences consumed first):
    #   \\. = backslash + any byte (2-byte token — handles \n, \t, \", \\ etc.)
    #   [{}"] = structural braces and string delimiters
    # This eliminates the escape_next flag entirely — no cross-chunk state bug.
    _SIG = re.compile(rb'\\.|[{}"]')

    with open(filepath, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = mm.size()

            depth = 0
            in_string = False
            obj_start = -1

            # Process in 4MB chunks — large enough for good regex throughput,
            # small enough to not bloat memory.
            # Overlap by 1 byte so that a backslash at the end of a chunk
            # can pair with the escaped character in the next chunk (the \\. rule).
            CHUNK = 4 * 1024 * 1024
            offset = 0

            while offset < size:
                end = min(offset + CHUNK + 1, size)  # +1 overlap for escape pairs
                chunk = mm[offset:end]

                for m in _SIG.finditer(chunk):
                    tok = chunk[m.start() : m.end()]
                    abs_pos = offset + m.start()

                    if len(tok) == 2:
                        # Escape sequence (\", \\, \n, \t, etc.) — skip entirely
                        continue

                    b = tok[0]
                    if b == 0x22:  # double quote
                        in_string = not in_string
                        continue
                    if in_string:
                        continue

                    if b == 0x7B:  # '{'
                        if depth == 0:
                            obj_start = abs_pos
                        depth += 1
                    elif b == 0x7D:  # '}'
                        depth -= 1
                        if depth == 0 and obj_start >= 0:
                            yield _parse(mm[obj_start : abs_pos + 1])
                            obj_start = -1

                # Advance by CHUNK (not end) to keep 1-byte overlap
                offset = min(offset + CHUNK, size)
        finally:
            mm.close()


def _collect_contour(det, contours_raw, score_threshold):
    """Extract contour from a detection dict and append to contours_raw.

    Supports contour sources (tried in order):
      1. contour_um / contour_dilated_um — already in um (pixel_size=1.0)
      2. outer_contour_global — pixel coords, requires pixel_size_um
      3. contour_px / contour_dilated_px — pixel coords, requires pixel_size_um
    If score_threshold is set, filters by features['score'] >= threshold.
    """
    from xldvp_seg.utils.detection_utils import get_contour_px, get_contour_um

    feat = det.get("features", {})
    if score_threshold is not None:
        score = feat.get("score")
        if score is None:
            score = det.get("score")
        if score is not None and float(score) < score_threshold:
            return

    # Try um contours first (already in coordinate space), then px contours
    contour_um = get_contour_um(det)
    if contour_um is not None and len(contour_um) >= 3:
        # um contours don't need pixel_size conversion — use 1.0 as identity
        contours_raw.append((contour_um, 1.0))
        return

    contour = det.get("outer_contour_global")
    if contour is None:
        contour = feat.get("outer_contour_global")
    if contour is None:
        contour = get_contour_px(det)
    if contour is None or len(contour) < 3:
        return

    pixel_size = det.get("pixel_size_um") or feat.get("pixel_size_um")
    if pixel_size is None or not isinstance(pixel_size, (int, float)):
        return

    contours_raw.append((contour, float(pixel_size)))


def load_slide_data(
    path, group_field, include_contours=False, score_threshold=None, marker_filter=None
):
    """Load a classified detection JSON and extract positions + groups.

    For large files (>500 MB), uses mmap streaming to avoid loading the
    entire JSON into memory.  For smaller files, uses fast_json_load.

    Args:
        path: Path to classified detection JSON.
        group_field: Field name to group by.
        include_contours: If True, also collect outer_contour_global (pixel coords)
            and pixel_size_um for contour rendering.
        score_threshold: If set, only include detections with score >= threshold
            in contours (positions are always included regardless).
        marker_filter: Optional filter string like "MSLN_class==positive".

    Returns:
        Dict with slide data, or None if no valid data.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("File not found, skipping: %s", path)
        return None

    file_size = path.stat().st_size
    use_streaming = file_size > 500_000_000  # >500 MB

    group_cells = {}  # group_label -> list of (x, y)
    contours_raw = []  # list of (outer_contour_global, pixel_size_um) when include_contours

    if use_streaming:
        logger.info("Streaming %s (%.1f GB)...", path.name, file_size / 1e9)
        # Parse marker filter once for streaming path
        _mf_key, _mf_val = None, None
        if marker_filter and "==" in marker_filter:
            _mf_key, _mf_val = [s.strip() for s in marker_filter.split("==", 1)]
        n_parsed = 0
        for det in _stream_detections_mmap(path):
            if _mf_key is not None:
                if det.get(_mf_key) != _mf_val and det.get("features", {}).get(_mf_key) != _mf_val:
                    continue
            pos = extract_position_um(det)
            if pos is None:
                continue
            group = extract_group(det, group_field)
            group_cells.setdefault(group, []).append(pos)
            if include_contours:
                _collect_contour(det, contours_raw, score_threshold)
            n_parsed += 1
            if n_parsed % 100000 == 0:
                logger.info("  %dk parsed...", n_parsed // 1000)
    else:
        try:
            from xldvp_seg.utils.json_utils import fast_json_load

            detections = fast_json_load(path)
        except ImportError:
            with open(path, encoding="utf-8") as f:
                detections = json.load(f)

        if not isinstance(detections, list):
            logger.warning("Not a JSON list, skipping: %s", path)
            return None

        if marker_filter:
            from xldvp_seg.utils.detection_utils import apply_marker_filter

            detections = apply_marker_filter(detections, marker_filter)

        for i in range(len(detections)):
            det = detections[i]
            detections[i] = None  # free memory as we go
            pos = extract_position_um(det)
            if pos is None:
                continue
            group = extract_group(det, group_field)
            group_cells.setdefault(group, []).append(pos)
            if include_contours:
                _collect_contour(det, contours_raw, score_threshold)
        del detections

    if not group_cells:
        return None

    groups_out = []
    for label, cells in sorted(group_cells.items()):
        arr = np.array(cells, dtype=np.float32)
        auto_eps = compute_auto_eps(arr, k=10) if len(cells) >= 11 else None
        groups_out.append(
            {
                "label": label,
                "n": len(cells),
                "x": arr[:, 0],
                "y": arr[:, 1],
                "auto_eps": auto_eps,
            }
        )

    all_x = np.concatenate([g["x"] for g in groups_out])
    all_y = np.concatenate([g["y"] for g in groups_out])

    result = {
        "groups": groups_out,
        "n_cells": sum(g["n"] for g in groups_out),
        "x_range": [float(all_x.min()), float(all_x.max())],
        "y_range": [float(all_y.min()), float(all_y.max())],
    }
    if include_contours and contours_raw:
        result["contours_raw"] = contours_raw
    return result


def discover_slides(input_dir, detection_glob):
    """Discover per-slide detection files in subdirectories.

    Searches recursively with ``**/<detection_glob>`` so that detection files
    nested under timestamp subdirectories are found.  The pipeline produces
    ``output_dir/slide_name/<run_timestamp>/cell_detections_classified.json``,
    so a depth-1 search is insufficient for multi-slide mode.

    For each found file, the slide name is inferred from the deepest ancestor
    directory that is a direct child of *input_dir* (i.e. the slide subdirectory).
    If the file is directly inside *input_dir*, the slide name comes from
    *input_dir* itself.

    Returns list of (slide_name, detection_path) tuples.
    """
    input_dir = Path(input_dir)
    results = []
    seen_paths = set()

    # Recursive search: finds files at any depth under input_dir
    for match in sorted(input_dir.rglob(detection_glob)):
        if not match.is_file():
            continue
        rp = match.resolve()
        if rp in seen_paths:
            continue
        seen_paths.add(rp)

        # Determine slide name: the first directory component relative to
        # input_dir.  E.g. for input_dir/slideA/run_123/det.json -> "slideA".
        # For input_dir/det.json -> input_dir.name.
        try:
            rel = match.relative_to(input_dir)
        except ValueError:
            continue
        parts = rel.parts  # e.g. ('slideA', 'run_123', 'det.json')
        if len(parts) <= 1:
            # File directly in input_dir
            slide_name = input_dir.name
        else:
            # First subdirectory component is the slide name
            slide_name = parts[0]

        results.append((slide_name, match))

    return results


def apply_top_n_filtering(slides_data, top_n, exclude_groups):
    """Apply top-N filtering and group exclusion across all slides.

    Groups in exclude_groups are dropped entirely.  If top_n is set, only the
    top_n most populous groups (by global cell count) are kept; the rest are
    merged into an 'other' group with recomputed auto_eps.
    """
    if exclude_groups:
        exc = set(exclude_groups)
        for _, data in slides_data:
            data["groups"] = [g for g in data["groups"] if g["label"] not in exc]
            data["n_cells"] = sum(g["n"] for g in data["groups"])

    if top_n is None:
        return

    # Count cells per group globally
    global_counts = {}
    for _, data in slides_data:
        for g in data["groups"]:
            global_counts[g["label"]] = global_counts.get(g["label"], 0) + g["n"]

    sorted_groups = sorted(global_counts.items(), key=lambda x: -x[1])
    top_labels = {lbl for i, (lbl, _) in enumerate(sorted_groups) if i < top_n}

    # Merge non-top groups into "other" per slide
    for _, data in slides_data:
        new_groups = []
        other_x = []
        other_y = []
        other_n = 0
        for g in data["groups"]:
            if g["label"] in top_labels:
                new_groups.append(g)
            else:
                other_x.append(g["x"])
                other_y.append(g["y"])
                other_n += g["n"]
        if other_n > 0:
            ox = np.concatenate(other_x)
            oy = np.concatenate(other_y)
            positions = np.column_stack([ox, oy])
            new_groups.append(
                {
                    "label": "other",
                    "n": other_n,
                    "x": ox,
                    "y": oy,
                    "auto_eps": compute_auto_eps(positions, k=10) if other_n >= 11 else None,
                }
            )
        data["groups"] = new_groups
        data["n_cells"] = sum(g["n"] for g in new_groups)
