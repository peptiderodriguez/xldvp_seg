"""Reusable building blocks for HTML spatial viewers.

Extracted from ``scripts/generate_multi_slide_spatial_viewer.py`` to enable
programmatic access and testing.  The script delegates to these functions
for data serialization and index construction.

Public API
----------
- :func:`build_group_index` — consistent group→index mapping across slides
- :func:`serialize_slide_positions` — base64-encoded position/group arrays
- :func:`collect_auto_eps` — per-slide per-group DBSCAN eps values
- :func:`compact_region_data` — region dicts → compact JS-friendly format
"""

from __future__ import annotations

import numpy as np

from xldvp_seg.utils.logging import get_logger
from xldvp_seg.visualization.encoding import encode_float32_base64, encode_uint8_base64

logger = get_logger(__name__)


def build_group_index(
    slides_data: list[tuple[str, dict]],
    color_map: dict[str, str],
    max_groups: int = 255,
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Build a consistent group→index mapping across all slides.

    If more than *max_groups* labels exist, the least frequent are collapsed
    into ``"other"`` so the index fits in a ``uint8``.

    Args:
        slides_data: List of ``(slide_name, data_dict)`` tuples.
        color_map: ``{group_label: hex_color}`` mapping.
        max_groups: Maximum distinct groups (default 255 for uint8).

    Returns:
        ``(group_labels, group_to_idx, color_map)`` — sorted label list,
        label→index dict, and (possibly collapsed) color map.
    """
    group_labels = sorted(color_map.keys())
    if len(group_labels) > max_groups:
        logger.warning(
            "%d groups exceeds Uint8 limit (%d). "
            "Keeping top %d groups, collapsing rest into 'other'.",
            len(group_labels),
            max_groups,
            max_groups - 1,
        )
        all_counts: dict[str, int] = {}
        for _, data in slides_data:
            for g in data["groups"]:
                all_counts[g["label"]] = all_counts.get(g["label"], 0) + g["n"]
        top_labels = [
            lbl for lbl in sorted(all_counts, key=all_counts.get, reverse=True) if lbl != "other"
        ][: max_groups - 1]
        group_labels = sorted(top_labels) + ["other"]
        other_color = "#808080"
        color_map = {lbl: color_map.get(lbl, other_color) for lbl in group_labels}
    group_to_idx = {lbl: i for i, lbl in enumerate(group_labels)}
    return group_labels, group_to_idx, color_map


def serialize_slide_positions(
    slides_data: list[tuple[str, dict]],
    group_to_idx: dict[str, int],
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Encode per-slide cell positions and group labels as base64 binary arrays.

    Positions are interleaved as ``[x0, y0, x1, y1, ...]`` in a flat
    ``Float32Array``.  Group indices are packed as ``Uint8Array``.

    Args:
        slides_data: List of ``(slide_name, data_dict)`` tuples.
        group_to_idx: ``{label: uint8_index}`` from :func:`build_group_index`.

    Returns:
        ``(slides_meta, slides_b64_positions, slides_b64_groups, slide_names_ordered)``
        — metadata dicts, base64-encoded positions, base64-encoded groups,
        and ordered slide names (skipping slides with 0 cells).
    """
    slides_meta: list[dict] = []
    slides_b64_positions: list[str] = []
    slides_b64_groups: list[str] = []

    for name, data in slides_data:
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_gi: list[np.ndarray] = []
        for g in data["groups"]:
            if g["n"] == 0:
                continue
            gi = group_to_idx.get(g["label"], group_to_idx.get("other", 0))
            all_x.append(g["x"])
            all_y.append(g["y"])
            all_gi.append(np.full(g["n"], gi, dtype=np.uint8))

        if not all_x:
            continue  # skip slide with no remaining cells

        all_x_arr = np.concatenate(all_x)
        all_y_arr = np.concatenate(all_y)
        all_gi_arr = np.concatenate(all_gi)
        n = len(all_x_arr)

        positions = np.empty(n * 2, dtype=np.float32)
        positions[0::2] = all_x_arr
        positions[1::2] = all_y_arr

        slides_b64_positions.append(encode_float32_base64(positions))
        slides_b64_groups.append(encode_uint8_base64(all_gi_arr))
        slides_meta.append(
            {
                "name": name,
                "n": int(n),
                "xr": [float(data["x_range"][0]), float(data["x_range"][1])],
                "yr": [float(data["y_range"][0]), float(data["y_range"][1])],
            }
        )

    slide_names_ordered = [m["name"] for m in slides_meta]
    return slides_meta, slides_b64_positions, slides_b64_groups, slide_names_ordered


def collect_auto_eps(
    slides_data: list[tuple[str, dict]],
    group_labels: list[str],
    default_eps: float = 100.0,
) -> list[list[float]]:
    """Collect pre-computed DBSCAN eps values per slide per group.

    The eps values are read from ``g["auto_eps"]`` in each slide's group
    data (computed during :func:`~xldvp_seg.visualization.data_loading.load_slide_data`).

    Args:
        slides_data: List of ``(slide_name, data_dict)`` tuples.
        group_labels: Ordered group labels from :func:`build_group_index`.
        default_eps: Fallback eps in µm when auto_eps is not available.

    Returns:
        List of per-slide eps arrays, each aligned with *group_labels*.
    """
    slides_auto_eps: list[list[float]] = []
    for _, data in slides_data:
        group_eps: dict[str, float] = {}
        for g in data["groups"]:
            eps_val = g.get("auto_eps")
            group_eps[g["label"]] = eps_val if eps_val is not None else default_eps
        eps_arr = [group_eps.get(lbl, default_eps) for lbl in group_labels]
        slides_auto_eps.append(eps_arr)
    return slides_auto_eps


def compact_region_data(
    slides_data: list[tuple[str, dict]],
) -> list[dict]:
    """Convert region dicts to compact JS-friendly format for HTML embedding.

    Args:
        slides_data: List of ``(slide_name, data_dict)`` tuples.

    Returns:
        List of per-slide dicts with ``"regions"`` (and optionally
        ``"regionScales"`` for multi-scale patterns).
    """

    def _compact_regions(reg_list: list[dict]) -> list[dict]:
        compact = []
        for r in reg_list:
            compact.append(
                {
                    "id": int(r["id"]),
                    "type": str(r.get("type", "")),
                    "label": str(r["label"]),
                    "color": str(r["color"]),
                    "pat": str(r.get("pattern", "")),
                    "n": int(r["n_cells"]),
                    "area": float(r["area_um2"]),
                    "elong": float(r["elongation"]),
                    "dfrac": float(r["dominant_frac"]),
                    "comp": {str(k): float(v) for k, v in r["composition"].items()},
                    "bnd": [[float(p["x"]), float(p["y"])] for p in r["boundary"]],
                }
            )
        return compact

    slides_region_data: list[dict] = []
    for _, data in slides_data:
        entry: dict = {"regions": _compact_regions(data.get("regions", []))}
        rs = data.get("region_scales")
        if rs:
            entry["regionScales"] = {k: _compact_regions(v) for k, v in rs.items()}
        slides_region_data.append(entry)
    return slides_region_data
