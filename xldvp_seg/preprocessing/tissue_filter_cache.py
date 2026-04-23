"""JSON cache for tissue-tile filter results.

The tissue filter (calibration + per-tile scan) runs on every detection
invocation and takes ~3-4 min per shard for whole-mouse slides. Output is
``(variance_threshold, tissue_tiles)`` where ``tissue_tiles`` is a list of
``{"x", "y"}`` dicts in global coordinates — keyed entirely by the CZI, the
tile grid, and the tissue channel. Cache it so reruns, resumes, and merge
steps skip the recompute.

Unlike the flat-field cache, this is cheap enough that concurrent recompute
by racing shards on a cold start is tolerable — no advisory lock. The first
shard to finish atomically writes the JSON; subsequent shards will read it
on their next invocation / resume.

Parallel to ``xldvp_seg.preprocessing.flat_field`` in its public shape:

    meta = build_cache_meta(args, tissue_channel=..., modality=..., ...)
    cached = load(cache_path, meta)
    if cached is None:
        threshold, tiles = ...compute...
        save(cache_path, threshold, tiles, meta)
    else:
        threshold, tiles = cached
"""

from __future__ import annotations

import os
from pathlib import Path

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Bump when the tissue-filter algorithm (calibrate_tissue_threshold /
# filter_tissue_tiles / brightfield Otsu path) changes in a way that
# invalidates previously cached outputs.
ALGORITHM_VERSION = "1.0"

CACHE_FILENAME = "tissue_filter.json"


def build_cache_meta(
    args,
    *,
    tissue_channel: int,
    modality: str,
    manual_threshold: float | None,
    n_all_tiles: int,
) -> dict:
    """Assemble the cache-key metadata for the current run.

    Any input that changes the tissue-filter output must appear here — otherwise
    a stale cache from a different configuration could load and produce the
    wrong tile list. ``scene`` is included because multi-scene CZIs can have
    coincidentally-identical slide_shape across scenes (e.g. multi-well plates
    where each well is a scene) but different pixel data.
    """
    czi_path_str = str(getattr(args, "czi_path", ""))
    czi_mtime: float = 0.0
    czi_size: int = 0
    try:
        if czi_path_str:
            st = os.stat(czi_path_str)
            czi_mtime = float(st.st_mtime)
            czi_size = int(st.st_size)
    except OSError:
        pass
    # `or 0` guards against args.scene being explicitly None (hand-built
    # SimpleNamespace in tests); argparse always gives an int default of 0.
    scene = getattr(args, "scene", 0) or 0
    return {
        "czi_path": czi_path_str,
        "czi_mtime": czi_mtime,
        "czi_size": czi_size,
        "scene": int(scene),
        "tile_size": int(args.tile_size),
        "tile_overlap": float(args.tile_overlap),
        "tissue_channel": int(tissue_channel),
        "modality": modality,
        "manual_threshold": (float(manual_threshold) if manual_threshold is not None else None),
        "n_all_tiles": int(n_all_tiles),
        "algorithm_version": ALGORITHM_VERSION,
    }


def load(cache_path: Path | None, expected_meta: dict) -> tuple[float, list[dict]] | None:
    """Return ``(variance_threshold, tissue_tiles)`` from cache, or ``None``.

    Mismatched metadata (different CZI, tile size, channel, etc.) is treated
    as a miss. Corrupt JSON is caught and treated as a miss so a partial write
    from a crashed peer doesn't propagate upward as an exception. Payload
    fields of the wrong type (e.g. hand-edited JSON) are also treated as
    misses rather than raising downstream.
    """
    if cache_path is None or not Path(cache_path).exists():
        return None
    try:
        payload = fast_json_load(cache_path)
    except (ValueError, OSError) as exc:
        logger.info("Tissue cache at %s unreadable (%s) — recomputing.", cache_path, exc)
        return None
    if not isinstance(payload, dict):
        logger.info("Tissue cache at %s has unexpected shape — recomputing.", cache_path)
        return None
    cached_meta = payload.get("__meta__", {})
    for key in (
        "czi_path",
        "czi_size",
        "scene",
        "tile_size",
        "tile_overlap",
        "tissue_channel",
        "modality",
        "manual_threshold",
        "n_all_tiles",
        "algorithm_version",
    ):
        # Back-compat: caches written before `scene` joined the cache key were
        # always implicitly scene=0 (single-scene was the only mode that wrote
        # a cache). Treat missing scene as 0 so a post-upgrade merge / resume
        # against an already-populated cache doesn't re-invalidate. Remove this
        # shim after a few releases when no pre-scene caches remain in the wild.
        cached_val = cached_meta.get(key, 0) if key == "scene" else cached_meta.get(key)
        if cached_val != expected_meta.get(key):
            logger.info(
                "Tissue cache stale: %s differs (cached=%r, current=%r). Recomputing.",
                key,
                cached_val,
                expected_meta.get(key),
            )
            return None
    if abs(cached_meta.get("czi_mtime", 0.0) - expected_meta.get("czi_mtime", 0.0)) > 1.0:
        logger.info("Tissue cache stale: czi_mtime differs. Recomputing.")
        return None
    threshold = payload.get("variance_threshold")
    tiles = payload.get("tissue_tiles")
    if threshold is None or tiles is None:
        logger.info("Tissue cache at %s missing payload fields — recomputing.", cache_path)
        return None
    # Guard against hand-edited JSON where threshold is a list/dict or tiles is
    # a scalar — pure-JSON shapes that pass the None check but crash float()/
    # list iteration downstream.
    try:
        threshold_f = float(threshold)
        tiles_list = list(tiles)
    except (TypeError, ValueError) as exc:
        logger.info("Tissue cache payload has wrong value types (%s) — recomputing.", exc)
        return None
    return threshold_f, tiles_list


def save(
    cache_path: Path,
    variance_threshold: float,
    tissue_tiles: list[dict],
    metadata: dict,
) -> None:
    """Atomically write the tissue filter result to ``cache_path``.

    Uses ``atomic_json_dump`` (tmp file + rename) so a killed process can't
    leave a truncated JSON behind.
    """
    # Persist only x/y: detection_loop adds w/h from args.tile_size at submission
    # time (see detection_loop.py:600 "Worker expects 'x', 'y', 'w', 'h'"), so
    # keeping any extras in the cache would be dead weight and couple the cache
    # schema to the full tile-dict shape.
    payload = {
        "__meta__": metadata,
        "variance_threshold": float(variance_threshold),
        "tissue_tiles": [{"x": int(t["x"]), "y": int(t["y"])} for t in tissue_tiles],
    }
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(payload, Path(cache_path))


__all__ = [
    "ALGORITHM_VERSION",
    "CACHE_FILENAME",
    "build_cache_meta",
    "load",
    "save",
]
