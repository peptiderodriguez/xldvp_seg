"""Instance segmentation metrics: IoU, Dice, Panoptic Quality.

Supports both mask-based and contour-based evaluation. Contour-based
is useful for evaluating pipeline output without loading HDF5 masks.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def iou_matrix(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU between predicted and ground-truth masks.

    Args:
        pred_masks: List of binary masks (HxW bool/uint8), one per prediction
        gt_masks: List of binary masks (HxW bool/uint8), one per ground truth

    Returns:
        IoU matrix of shape (n_pred, n_gt)
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    iou = np.zeros((n_pred, n_gt), dtype=np.float64)

    for i in range(n_pred):
        pi = pred_masks[i].astype(bool)
        for j in range(n_gt):
            gj = gt_masks[j].astype(bool)
            intersection = np.logical_and(pi, gj).sum()
            union = np.logical_or(pi, gj).sum()
            if union > 0:
                iou[i, j] = intersection / union

    return iou


def iou_from_contours(
    pred_contours: List[np.ndarray],
    gt_contours: List[np.ndarray],
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Compute pairwise IoU from contour polygons.

    Rasterizes contours to masks, then computes IoU. If image_shape is None,
    infers from the bounding box of all contours.

    Args:
        pred_contours: List of Nx2 contour arrays (x, y coordinates)
        gt_contours: List of Nx2 contour arrays
        image_shape: (height, width) for rasterization canvas. If None, auto-computed.

    Returns:
        IoU matrix of shape (n_pred, n_gt)
    """
    import cv2

    # Auto-compute canvas size from contour bounds
    if image_shape is None:
        all_contours = pred_contours + gt_contours
        if not all_contours or all(len(c) == 0 for c in all_contours):
            return np.zeros((len(pred_contours), len(gt_contours)), dtype=np.float64)
        non_empty = [c for c in all_contours if len(c) > 0]
        max_x = max(c[:, 0].max() for c in non_empty) + 10
        max_y = max(c[:, 1].max() for c in non_empty) + 10
        image_shape = (int(max_y), int(max_x))

    def _rasterize(contour, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        if len(contour) == 0:
            return mask.astype(bool)
        pts = contour.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)

    pred_masks = [_rasterize(c, image_shape) for c in pred_contours]
    gt_masks = [_rasterize(c, image_shape) for c in gt_contours]

    return iou_matrix(pred_masks, gt_masks)


def hungarian_match(
    iou: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Optimal matching between predictions and ground truth using Hungarian algorithm.

    Args:
        iou: IoU matrix of shape (n_pred, n_gt) from iou_matrix()
        threshold: Minimum IoU to consider a match (default: 0.5)

    Returns:
        matched: List of (pred_idx, gt_idx, iou_value) tuples
        unmatched_pred: List of prediction indices with no match
        unmatched_gt: List of ground truth indices with no match
    """
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = iou.shape
    if n_pred == 0 or n_gt == 0:
        return [], list(range(n_pred)), list(range(n_gt))

    # Hungarian algorithm (minimize cost = maximize IoU)
    cost = 1.0 - iou
    pred_indices, gt_indices = linear_sum_assignment(cost)

    matched = []
    matched_pred = set()
    matched_gt = set()

    for pi, gi in zip(pred_indices, gt_indices):
        if iou[pi, gi] >= threshold:
            matched.append((int(pi), int(gi), float(iou[pi, gi])))
            matched_pred.add(pi)
            matched_gt.add(gi)

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]

    return matched, unmatched_pred, unmatched_gt


def panoptic_quality(
    matched: List[Tuple[int, int, float]],
    n_pred: int,
    n_gt: int,
) -> Tuple[float, float, float]:
    """Compute Panoptic Quality = SQ * RQ.

    PQ decomposes into:
    - SQ (Segmentation Quality): average IoU of matched pairs
    - RQ (Recognition Quality): F1 of detection (TP / (TP + 0.5*FP + 0.5*FN))

    Args:
        matched: List of (pred_idx, gt_idx, iou) from hungarian_match()
        n_pred: Total number of predictions
        n_gt: Total number of ground truth instances

    Returns:
        (pq, sq, rq) tuple, each in [0, 1]
    """
    tp = len(matched)
    fp = n_pred - tp
    fn = n_gt - tp

    if tp == 0:
        return 0.0, 0.0, 0.0

    sq = sum(iou_val for _, _, iou_val in matched) / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    pq = sq * rq

    return pq, sq, rq


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks.

    Dice = 2 * |intersection| / (|pred| + |gt|)

    Args:
        pred_mask: Binary prediction mask (HxW)
        gt_mask: Binary ground truth mask (HxW)

    Returns:
        Dice score in [0, 1]
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / total


def detection_f1(
    matched: List[Tuple[int, int, float]],
    n_pred: int,
    n_gt: int,
) -> Dict[str, float]:
    """Compute detection-level precision, recall, and F1.

    Args:
        matched: List of (pred_idx, gt_idx, iou) from hungarian_match()
        n_pred: Total predictions
        n_gt: Total ground truth

    Returns:
        Dict with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'
    """
    tp = len(matched)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_instance_segmentation(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Full instance segmentation evaluation.

    Computes IoU matrix, optimal matching, PQ, SQ, RQ, F1, precision, recall.

    Args:
        pred_masks: List of binary prediction masks
        gt_masks: List of binary ground truth masks
        iou_threshold: Minimum IoU for a match (default: 0.5)

    Returns:
        Dict with all metrics:
            - 'pq', 'sq', 'rq': Panoptic Quality components
            - 'precision', 'recall', 'f1': Detection metrics
            - 'tp', 'fp', 'fn': Counts
            - 'n_pred', 'n_gt': Input sizes
            - 'matched': List of (pred_idx, gt_idx, iou) tuples
            - 'mean_iou': Mean IoU of matched pairs
            - 'iou_threshold': Threshold used
    """
    iou = iou_matrix(pred_masks, gt_masks)
    matched, unmatched_pred, unmatched_gt = hungarian_match(iou, threshold=iou_threshold)
    pq, sq, rq = panoptic_quality(matched, len(pred_masks), len(gt_masks))
    det_metrics = detection_f1(matched, len(pred_masks), len(gt_masks))

    mean_iou = float(np.mean([m[2] for m in matched])) if matched else 0.0

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        **det_metrics,
        "n_pred": len(pred_masks),
        "n_gt": len(gt_masks),
        "matched": matched,
        "mean_iou": mean_iou,
        "iou_threshold": iou_threshold,
    }
