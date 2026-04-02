"""Segmentation quality metrics: IoU, Dice, Panoptic Quality, F1.

For benchmarking segmentation methods (Cellpose vs InstanSeg), comparing
dedup strategies, and evaluating contour accuracy against ground truth.

Usage:
    from xldvp_seg.metrics import iou_matrix, hungarian_match, panoptic_quality

    iou = iou_matrix(pred_masks, gt_masks)
    matched, unmatched_pred, unmatched_gt = hungarian_match(iou, threshold=0.5)
    pq, sq, rq = panoptic_quality(matched, len(pred_masks), len(gt_masks))
"""

from .instance import (
    detection_f1,
    dice_score,
    evaluate_instance_segmentation,
    hungarian_match,
    iou_from_contours,
    iou_matrix,
    panoptic_quality,
)

__all__ = [
    "iou_matrix",
    "iou_from_contours",
    "hungarian_match",
    "panoptic_quality",
    "dice_score",
    "detection_f1",
    "evaluate_instance_segmentation",
]
