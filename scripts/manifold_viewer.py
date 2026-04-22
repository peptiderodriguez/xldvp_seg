#!/usr/bin/env python
"""Linked 3D-UMAP + slide-thumbnail viewer for manifold-sampling output.

Loads the ``manifold_state_<hash>.npz`` cache written by
``manifold_sample.py`` (keys ``picked_idx``, ``labels``, ``d_to_anchor``,
``outlier_mask``), rebuilds a 3D UMAP on the PCA embedding
(``X_pca_<hash>.npz`` sibling written by
:func:`xldvp_seg.analysis.rare_cell_discovery.load_and_embed`), renders a
composite fluorescence thumbnail from the CZI, and emits a self-contained
HTML via :func:`xldvp_seg.visualization.manifold_viewer.build_linked_viewer_html`.

Typical usage::

    scripts/manifold_viewer.py \\
        --state-npz run/manifold_state_<hash>.npz \\
        --detections run/exemplar_detections.json \\
        --czi-path slide.czi \\
        --display-channels 2,4 \\
        --output run/manifold_viewer.html

Optional filter: pass ``--selected-replicates-json`` (the ``lmd_selected_replicates.json``
emitted by ``manifold_sample.py``) to restrict the UMAP + slide overlay to the
cells in those replicates.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

from xldvp_seg.utils.json_utils import fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402
from xldvp_seg.visualization.fluorescence import read_czi_thumbnail_channels  # noqa: E402
from xldvp_seg.visualization.manifold_viewer import build_linked_viewer_html  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# UMAP (GPU preferred, CPU fallback)
# ---------------------------------------------------------------------------


def _umap_3d(X: np.ndarray, *, seed: int = 42, use_gpu: bool = True) -> np.ndarray:
    """Compute a 3D UMAP embedding. Tries cuML, falls back to umap-learn.

    Raises:
        RuntimeError: When neither backend is installed.
    """
    n_neighbors = max(2, min(30, X.shape[0] - 1))
    if use_gpu:
        try:
            from cuml.manifold import UMAP as cuUMAP  # type: ignore[import-not-found]  # noqa: N811

            logger.info("UMAP (cuML GPU): N=%d d=%d -> 3D", X.shape[0], X.shape[1])
            reducer = cuUMAP(n_components=3, n_neighbors=n_neighbors, random_state=seed)
            return np.asarray(reducer.fit_transform(X), dtype=np.float32)
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            logger.info("cuML UMAP unavailable (%s); falling back to umap-learn", e)

    try:
        import umap  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "Neither cuML nor umap-learn is available. Install one of: "
            "`pip install umap-learn` (CPU) or ensure cuML is importable (GPU)."
        ) from e
    logger.info("UMAP (umap-learn CPU): N=%d d=%d -> 3D", X.shape[0], X.shape[1])
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=seed)
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_pca_cache(state_npz: Path) -> Path:
    """Locate the ``X_pca_<hash>.npz`` cache that pairs with ``manifold_state_<hash>.npz``.

    The two hashes differ (the manifold-state hash folds a superset of
    inputs), so we rely on the ``pca_cache_key`` cross-link baked into
    modern ``manifold_state`` npz files. Silently picking "the only other
    X_pca cache" is banned — it can load a cache from a different run
    with different feature groups / filters and produce misleading viewers.
    """
    parent = state_npz.parent
    try:
        state = np.load(state_npz, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise FileNotFoundError(f"cannot read {state_npz}: {e}") from e
    if "pca_cache_key" in state.files:
        key = str(state["pca_cache_key"]).strip()
        if key:
            candidate = parent / f"X_pca_{key}.npz"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(
                f"manifold_state points at X_pca_{key}.npz but the file is "
                f"missing from {parent}. Rerun manifold_sample.py."
            )
    pca_caches = sorted(parent.glob("X_pca_*.npz"))
    raise FileNotFoundError(
        f"{state_npz.name} has no pca_cache_key (written by an older "
        f"manifold_sample.py run) and cannot be auto-paired. "
        f"Found: {[c.name for c in pca_caches] or 'none'}. "
        "Rerun manifold_sample.py so the cross-link is present."
    )


def _composite_thumbnail(
    channel_arrays: list[np.ndarray | None],
) -> np.ndarray:
    """Combine up to 3 single-channel thumbnails into an (H, W, 3) uint8 RGB image.

    Channel 0 -> R, 1 -> G, 2 -> B. Missing/None channels are zero-filled.
    """
    valid = [a for a in channel_arrays if a is not None]
    if not valid:
        raise RuntimeError("All thumbnail channels failed to read.")
    h, w = valid[0].shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, arr in enumerate(channel_arrays[:3]):
        if arr is None:
            continue
        rgb[:, :, i] = arr
    return rgb


def _restrict_to_replicates(
    uids: list[str],
    replicates_path: Path,
) -> np.ndarray:
    """Return a boolean mask (len == len(uids)) keeping only cells in the listed replicates."""
    reps = fast_json_load(str(replicates_path))
    if not isinstance(reps, list):
        raise ValueError(f"{replicates_path} must decode to a list of replicate dicts")
    keep_uids: set[str] = set()
    for r in reps:
        for u in r.get("cell_uids", []):
            keep_uids.add(str(u))
    mask = np.array([u in keep_uids for u in uids], dtype=bool)
    logger.info(
        "Replicate filter: %d/%d cells match %d replicates from %s",
        int(mask.sum()),
        mask.size,
        len(reps),
        replicates_path.name,
    )
    return mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--state-npz",
        type=Path,
        required=True,
        help="manifold_state_<hash>.npz from manifold_sample.py",
    )
    p.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Kept-detections JSON (exemplar_detections.json from manifold_sample.py) "
        "-- supplies per-cell UIDs + global_center_um.",
    )
    p.add_argument("--czi-path", type=Path, required=True, help="CZI file for slide thumbnail.")
    p.add_argument("--output", type=Path, required=True, help="Output HTML path.")

    p.add_argument(
        "--display-channels",
        type=str,
        default="2,4",
        help="Comma-separated CZI channel indices (up to 3 -> R,G,B). Default: 2,4.",
    )
    p.add_argument(
        "--thumbnail-scale",
        type=float,
        default=0.0625,
        help="CZI mosaic downsample factor. Default: 1/16 = 0.0625.",
    )
    p.add_argument("--scene", type=int, default=0, help="CZI scene index. Default: 0.")
    p.add_argument(
        "--selected-replicates-json",
        type=Path,
        default=None,
        help="If set, restrict cells shown to those listed in this replicate JSON "
        "(lmd_selected_replicates.json from manifold_sample.py).",
    )
    p.add_argument(
        "--selected-only-on-slide",
        action="store_true",
        help="If set, the slide panel shows NO cells by default -- only the "
        "UMAP-selected group's cells light up.",
    )
    p.add_argument(
        "--umap-subsample",
        type=int,
        default=100000,
        help="Cap cells fed to UMAP (random subsample if larger). Default: 100000.",
    )
    p.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cuML UMAP if available; otherwise umap-learn CPU. Default: on.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--title",
        type=str,
        default="Manifold viewer",
        help="HTML title / info-panel header.",
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional kept_groups.json from aggregate_group_annotations.py. "
        "Dropped groups are rendered at 25%% brightness and flagged in the "
        "info panel; kept groups at full color.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Load state + PCA ---------------------------------------------------
    logger.info("Loading manifold state: %s", args.state_npz)
    state = np.load(args.state_npz)
    labels_all = state["labels"].astype(np.int64)
    outlier_mask = state["outlier_mask"].astype(bool)

    pca_path = _find_pca_cache(args.state_npz)
    logger.info("Loading PCA cache: %s", pca_path)
    pca_npz = np.load(pca_path)
    X_pca = np.asarray(pca_npz["X_pca"], dtype=np.float32)
    if X_pca.shape[0] != labels_all.shape[0]:
        raise RuntimeError(
            f"PCA cache has {X_pca.shape[0]} rows but state has {labels_all.shape[0]} labels "
            "-- caches are mismatched (different runs)."
        )

    # --- Load kept detections ----------------------------------------------
    logger.info("Loading detections: %s", args.detections)
    kept = fast_json_load(str(args.detections))
    if len(kept) != X_pca.shape[0]:
        raise RuntimeError(
            f"Detections JSON has {len(kept)} rows but PCA cache has {X_pca.shape[0]} -- "
            "pass the exemplar_detections.json from the SAME manifold_sample run."
        )
    positions_um = np.array([d.get("global_center_um", [0.0, 0.0]) for d in kept], dtype=np.float32)
    uids = [str(d.get("uid", f"cell_{i}")) for i, d in enumerate(kept)]

    # --- Optional replicate-list filter ------------------------------------
    include_mask = ~outlier_mask
    if args.selected_replicates_json is not None:
        include_mask &= _restrict_to_replicates(uids, args.selected_replicates_json)

    n_kept = int(include_mask.sum())
    if n_kept == 0:
        raise RuntimeError("No cells left after applying outlier + replicate filters.")
    logger.info("Cells to render: %d (%.1f%% of %d)", n_kept, 100 * n_kept / len(kept), len(kept))

    # --- UMAP subsample (uniform random) -----------------------------------
    rng = np.random.default_rng(args.seed)
    idx_kept = np.flatnonzero(include_mask)
    if n_kept > args.umap_subsample:
        pick = rng.choice(idx_kept, size=args.umap_subsample, replace=False)
        pick.sort()
    else:
        pick = idx_kept
    logger.info("UMAP input: %d cells", pick.size)

    umap_3d = _umap_3d(X_pca[pick], seed=args.seed, use_gpu=args.use_gpu)

    # --- CZI thumbnail -----------------------------------------------------
    display_channels = [int(c) for c in args.display_channels.split(",") if c.strip()]
    logger.info(
        "Reading CZI thumbnail (channels %s, scale %s)", display_channels, args.thumbnail_scale
    )
    chans, pixel_size_um, mosaic_x, mosaic_y = read_czi_thumbnail_channels(
        args.czi_path,
        display_channels,
        scale_factor=args.thumbnail_scale,
        scene=args.scene,
    )
    thumb_rgb = _composite_thumbnail(chans)
    thumb_h, thumb_w = thumb_rgb.shape[:2]

    # Slide extent in um: thumbnail pixels / scale * pixel_size_um per-full-res px.
    if pixel_size_um is None:
        # Fallback: use positions' extent so the cells still land inside the canvas.
        width_um = float(np.ptp(positions_um[:, 0])) or 1.0
        height_um = float(np.ptp(positions_um[:, 1])) or 1.0
        logger.warning(
            "No pixel size from CZI; using positions extent (w=%.0f h=%.0f)", width_um, height_um
        )
    else:
        width_um = thumb_w / args.thumbnail_scale * pixel_size_um
        height_um = thumb_h / args.thumbnail_scale * pixel_size_um

    # Shift positions by mosaic origin so that (0,0) lines up with thumbnail top-left.
    pos_shifted = positions_um.copy()
    if pixel_size_um is not None:
        pos_shifted[:, 0] -= mosaic_x * pixel_size_um
        pos_shifted[:, 1] -= mosaic_y * pixel_size_um

    # --- Optional: annotation-driven color override + info panel ----------
    group_colors = None
    info_extra_html = ""
    if args.annotations is not None:
        from xldvp_seg.visualization.colors import shuffled_hsv_palette

        payload = fast_json_load(str(args.annotations))
        per_group = {int(g["manifold_group_id"]): g for g in payload.get("groups", [])}
        n_groups = int(labels_all.max()) + 1 if labels_all.size else 0
        # Start from the same palette the viewer uses by default, then dim
        # dropped / un-annotated groups so kept ones stand out visually.
        palette = shuffled_hsv_palette(n_groups, seed=0)
        for gid in range(n_groups):
            g = per_group.get(gid)
            if g is None or not g.get("kept"):
                palette[gid] = (palette[gid] * 0.25).astype(np.uint8)
        group_colors = palette
        # Summary stats for the info panel.
        n_kept_groups = int(payload.get("n_groups_kept", 0))
        n_total_groups = int(payload.get("n_groups_total", n_groups))
        n_ann = int(payload.get("n_annotated_total", 0))
        threshold = float(payload.get("threshold", 0.0))
        info_extra_html = (
            f"<div style='margin-top:12px;padding:8px;background:#1a1a1a;"
            f"border-radius:6px;font-size:12px'>"
            f"<b>Annotation results</b><br>"
            f"Threshold: {threshold:.0%}<br>"
            f"Annotated: {n_ann} cards<br>"
            f"Kept: {n_kept_groups}/{n_total_groups} groups<br>"
            f"<span style='color:#888'>dropped groups dimmed to 25%</span>"
            f"</div>"
        )
        logger.info(
            "Annotations: %d kept / %d total groups (threshold=%.2f)",
            n_kept_groups,
            n_total_groups,
            threshold,
        )

    # --- Build HTML --------------------------------------------------------
    html = build_linked_viewer_html(
        umap_coords_3d=umap_3d,
        positions_um=pos_shifted[pick],
        labels=labels_all[pick],
        thumbnail_rgb=thumb_rgb,
        slide_extent_um=(width_um, height_um),
        selected_only_on_slide=args.selected_only_on_slide,
        group_colors=group_colors,
        title=args.title,
        info_extra_html=info_extra_html,
    )

    # Atomic write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, args.output)
    logger.info(
        "Wrote linked viewer HTML: %s (%.1f KB)", args.output, args.output.stat().st_size / 1024
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
