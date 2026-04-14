"""SAM2-based anatomical region segmentation on fluorescence thumbnails.

Segments tissue into anatomical regions by running SAM2 auto-mask on
CZI fluorescence channel thumbnails. Regions are born in fluorescence
coordinate space — no cross-modal registration needed.

Typical workflow::

    from xldvp_seg.analysis.region_segmentation import (
        build_tissue_mask, segment_regions, clean_labels, fill_labels,
    )

    tissue = build_tissue_mask(hoechst_thumb)
    labels, masks = segment_regions(rgb_composite, points_per_side=64)
    labels = clean_labels(labels, tissue)
    labels = fill_labels(labels, tissue)

Functions are pure — arrays in, arrays out. CZI loading, model management,
and file I/O are handled by the CLI wrapper (``scripts/segment_regions.py``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
)
from scipy.ndimage import (
    label as nd_label,
)
from skimage.filters import threshold_otsu
from skimage.segmentation import expand_labels

from xldvp_seg.utils.image_utils import percentile_normalize  # noqa: F401 — re-export
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def build_tissue_mask(
    channel_data: np.ndarray,
    *,
    threshold_factor: float = 0.3,
    dilate: int = 2,
    erode: int = 12,
) -> np.ndarray:
    """Build a binary tissue mask from a fluorescence channel.

    Uses Otsu thresholding scaled by *threshold_factor*, morphological
    hole filling, dilation, and erosion. The erosion is aggressive by
    default (12 iterations) to remove thin tissue edges that produce
    spurious SAM2 masks.

    Args:
        channel_data: 2D uint8 or uint16 fluorescence channel.
        threshold_factor: Multiply Otsu threshold by this (default: 0.3).
        dilate: Dilation iterations before erosion (default: 2).
        erode: Erosion iterations (default: 12).

    Returns:
        Binary mask (uint8, 0 or 1) with the same shape.
    """
    gray = channel_data
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    if gray.max() == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    thresh = threshold_otsu(gray) * threshold_factor
    mask = gray > thresh
    mask = binary_fill_holes(mask)
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    if erode > 0:
        mask = binary_erosion(mask, iterations=erode)
    return mask.astype(np.uint8)


def segment_regions(
    img_rgb: np.ndarray,
    *,
    points_per_side: int = 64,
    pred_iou_thresh: float = 0.0,
    stability_score_thresh: float = 0.0,
    min_mask_region_area: int = 150,
    crop_n_layers: int = 0,
    device: str = "cpu",
    sigma: float = 5.0,
    model=None,
) -> tuple[np.ndarray, list[dict]]:
    """Run SAM2 auto-mask and build a non-overlapping label map.

    Masks are sorted smallest-first so small structures claim pixels before
    large ones. Each mask is Gaussian-smoothed (sigma) before assignment to
    reduce jagged boundaries.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image.
        points_per_side: Grid density of SAM2 point prompts (default: 64).
        pred_iou_thresh: Minimum predicted IoU to keep a mask (default: 0.0).
        stability_score_thresh: Minimum stability score (default: 0.0).
        min_mask_region_area: Minimum mask area in pixels (default: 150).
        crop_n_layers: SAM2 multi-crop layers (default: 0 — disabled to
            reduce GPU memory for high point counts).
        device: ``"cpu"``, ``"cuda"``, or ``"cuda:N"``. Use ``"cpu"`` for
            points_per_side >= 256 to avoid GPU OOM.
        sigma: Gaussian smoothing sigma for mask boundaries (default: 5.0).
        model: Optional pre-loaded SAM2 model. If ``None``, loads from
            the checkpoint directory. Pass a model to reuse across a
            point-density series without reloading.

    Returns:
        ``(label_map, raw_masks)`` where *label_map* is an int32 array
        and *raw_masks* is the list of SAM2 mask dicts (for diagnostics).
    """
    if model is None:
        model = _load_sam2_model(device)

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    gen = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers,
    )
    raw_masks = gen.generate(img_rgb)
    raw_masks = sorted(raw_masks, key=lambda m: m["area"])
    logger.info("SAM2: %d masks (pts=%d)", len(raw_masks), points_per_side)

    # Build non-overlapping label map (smallest-first)
    label_map = np.zeros(img_rgb.shape[:2], dtype=np.int32)
    for i, m in enumerate(raw_masks):
        seg = gaussian_filter(m["segmentation"].astype(float), sigma=sigma) > 0.5
        label_map[(label_map == 0) & seg] = i + 1

    n = len(np.unique(label_map)) - 1
    logger.info("Label map: %s, %d regions", label_map.shape, n)
    return label_map, raw_masks


def clean_labels(
    label_map: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    min_area: int = 300,
    min_tissue_overlap: float = 0.75,
) -> np.ndarray:
    """Remove spurious regions from a label map.

    Removes regions that are:
    - Smaller than *min_area* pixels.
    - Less than *min_tissue_overlap* fraction inside the tissue mask
      (catches peripheral junk along tissue edges).

    Args:
        label_map: int32 label map from :func:`segment_regions`.
        tissue_mask: Binary tissue mask from :func:`build_tissue_mask`.
        min_area: Minimum region area in pixels (default: 300).
        min_tissue_overlap: Minimum fraction of region pixels inside
            tissue mask (default: 0.75).

    Returns:
        Cleaned label map (int32). Removed regions are set to 0.
    """
    lh, lw = label_map.shape
    th, tw = tissue_mask.shape
    if (lh, lw) != (th, tw):
        tmask = cv2.resize(tissue_mask, (lw, lh), interpolation=cv2.INTER_NEAREST)
    else:
        tmask = tissue_mask

    cleaned = label_map.copy()
    for lid in np.unique(cleaned):
        if lid == 0:
            continue
        region = cleaned == lid
        area = region.sum()
        if area < min_area:
            cleaned[region] = 0
            continue
        overlap = (region & (tmask > 0)).sum()
        if (overlap / area) < min_tissue_overlap:
            cleaned[region] = 0

    n_before = len(np.unique(label_map)) - 1
    n_after = len(np.unique(cleaned)) - 1
    logger.info("Clean: %d → %d regions (removed %d)", n_before, n_after, n_before - n_after)
    return cleaned


def fill_labels(
    label_map: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    fill_interstitial: bool = True,
) -> np.ndarray:
    """Fill gaps in a label map so every tissue pixel belongs to a region.

    Two-phase fill:

    1. ``expand_labels`` grows each region outward until it touches a
       neighbor, filling all inter-region gaps.
    2. If *fill_interstitial*: find enclosed unlabeled connected components
       (not touching image border) and assign each a unique new region ID.
       These are typically intestinal lumens, vessel spaces, or other
       interstitial structures.

    Args:
        label_map: int32 label map (may have gaps = label 0 inside tissue).
        tissue_mask: Binary tissue mask.
        fill_interstitial: If True (default), enclosed holes become their
            own regions instead of being absorbed by neighbors.

    Returns:
        Filled label map (int32). All tissue pixels have a non-zero label.
    """
    lh, lw = label_map.shape
    th, tw = tissue_mask.shape
    if (lh, lw) != (th, tw):
        tmask = cv2.resize(tissue_mask, (lw, lh), interpolation=cv2.INTER_NEAREST)
    else:
        tmask = tissue_mask

    # Detect interstitial holes BEFORE expand_labels (which would fill them)
    if fill_interstitial:
        unlabeled = (label_map == 0).astype(np.uint8)
        hole_labels, n_holes = nd_label(unlabeled)
        if n_holes > 0:
            # Identify border-touching components (background, not enclosed)
            border = np.zeros_like(hole_labels, dtype=bool)
            border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
            border_ids = set(np.unique(hole_labels[border])) - {0}

            # Assign enclosed holes new IDs in the INPUT label map so
            # expand_labels treats them as seeds (not gaps to fill)
            next_id = label_map.max() + 1
            label_with_holes = label_map.copy()
            n_filled_holes = 0
            for hid in range(1, n_holes + 1):
                if hid in border_ids:
                    continue
                hole_mask = hole_labels == hid
                if hole_mask.sum() < 5:
                    continue
                label_with_holes[hole_mask] = next_id
                next_id += 1
                n_filled_holes += 1
            if n_filled_holes > 0:
                logger.info("Assigned %d interstitial holes as own regions", n_filled_holes)
        else:
            label_with_holes = label_map
    else:
        label_with_holes = label_map

    filled = expand_labels(label_with_holes, distance=9999)
    filled[tmask == 0] = 0

    n = len(np.unique(filled)) - 1
    logger.info("Fill: %d regions total", n)
    return filled


def compute_region_nuc_stats(
    detections: list[dict],
    *,
    exclude_zero_nuclei: bool = True,
) -> dict[int, dict]:
    """Compute per-region nuclear count statistics.

    Each detection must have ``organ_id`` set (via
    ``scripts/assign_cells_to_regions.py``) and ``features.n_nuclei``
    set (via pipeline Phase 4 nuclear counting).

    Args:
        detections: List of detection dicts with ``organ_id`` and
            ``features.n_nuclei``.
        exclude_zero_nuclei: If True (default), exclude cells with
            n_nuclei == 0 from all counts and statistics. These are
            typically fragments or edge artifacts, not real cells.

    Returns:
        Dict mapping region_id → ``{count, mean_nuc, median_nuc, nuc_dist}``
        where *nuc_dist* is ``{n: count}`` (e.g., ``{1: 500, 2: 80, 3: 5}``).
        Region 0 (background) is excluded.
    """
    stats: dict[int, list[int]] = defaultdict(list)
    for det in detections:
        oid = det.get("organ_id", 0)
        if oid == 0:
            continue
        nn = det.get("features", {}).get("n_nuclei")
        if nn is None:
            continue
        nn = int(nn)
        if exclude_zero_nuclei and nn == 0:
            continue
        stats[oid].append(nn)

    result = {}
    for oid, vals in stats.items():
        if not vals:
            continue
        arr = np.array(vals)
        dist = dict(Counter(vals))
        result[int(oid)] = {
            "count": len(vals),
            "mean_nuc": round(float(arr.mean()), 2),
            "median_nuc": int(np.median(arr)),
            "nuc_dist": {str(k): v for k, v in sorted(dist.items())},
        }
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_sam2_model(device: str = "cpu"):
    """Load SAM2 model from checkpoint directory.

    Uses the same checkpoint search path as ``ModelManager`` but loads
    directly via ``build_sam2()`` — no singleton, supports CPU mode.
    """
    from sam2.build_sam import build_sam2

    script_dir = Path(__file__).parent.parent.parent.resolve()
    checkpoint = None
    for cp in [
        script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
        script_dir / "checkpoints" / "sam2.1_hiera_l.pt",
        Path.home() / ".cache" / "sam2" / "sam2.1_hiera_large.pt",
    ]:
        if cp.exists():
            checkpoint = cp
            break
    if checkpoint is None:
        raise FileNotFoundError(
            "SAM2 checkpoint not found. Expected in checkpoints/ or ~/.cache/sam2/"
        )

    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    logger.info("Loading SAM2 from %s (device=%s)", checkpoint, device)
    return build_sam2(config, str(checkpoint), device=device)
