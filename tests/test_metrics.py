"""Tests for xldvp_seg.metrics (IoU, Dice, PQ, Hungarian matching)."""

import numpy as np

from xldvp_seg.metrics import (
    detection_f1,
    dice_score,
    evaluate_instance_segmentation,
    hungarian_match,
    iou_matrix,
    panoptic_quality,
)


class TestIoU:

    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=bool)
        iou = iou_matrix([mask], [mask])
        assert iou.shape == (1, 1)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self):
        m1 = np.zeros((10, 10), dtype=bool)
        m1[:5] = True
        m2 = np.zeros((10, 10), dtype=bool)
        m2[5:] = True
        assert iou_matrix([m1], [m2])[0, 0] == 0.0

    def test_partial_overlap(self):
        m1 = np.zeros((10, 10), dtype=bool)
        m1[0:6, 0:6] = True  # 36 px
        m2 = np.zeros((10, 10), dtype=bool)
        m2[3:9, 3:9] = True  # 36 px, intersection = 3x3 = 9
        iou = iou_matrix([m1], [m2])
        assert abs(iou[0, 0] - 9.0 / 63.0) < 1e-6

    def test_empty_inputs(self):
        assert iou_matrix([], []).shape == (0, 0)

    def test_multiple_masks(self):
        m1 = np.zeros((10, 10), dtype=bool)
        m1[:5] = True
        m2 = np.zeros((10, 10), dtype=bool)
        m2[5:] = True
        iou = iou_matrix([m1, m2], [m1, m2])
        assert iou[0, 0] == 1.0
        assert iou[0, 1] == 0.0
        assert iou[1, 1] == 1.0


class TestDice:

    def test_perfect(self):
        mask = np.ones((10, 10), dtype=bool)
        assert dice_score(mask, mask) == 1.0

    def test_no_overlap(self):
        m1 = np.zeros((10, 10), dtype=bool)
        m1[:5] = True
        m2 = np.zeros((10, 10), dtype=bool)
        m2[5:] = True
        assert dice_score(m1, m2) == 0.0

    def test_both_empty(self):
        m = np.zeros((10, 10), dtype=bool)
        assert dice_score(m, m) == 1.0


class TestHungarianMatch:

    def test_perfect_matching(self):
        iou = np.array([[1.0, 0.0], [0.0, 1.0]])
        matched, up, ug = hungarian_match(iou, threshold=0.5)
        assert len(matched) == 2
        assert len(up) == 0
        assert len(ug) == 0

    def test_threshold_filtering(self):
        iou = np.array([[0.3]])
        matched, up, ug = hungarian_match(iou, threshold=0.5)
        assert len(matched) == 0
        assert len(up) == 1

    def test_empty_inputs(self):
        matched, up, ug = hungarian_match(np.zeros((0, 0)))
        assert len(matched) == 0


class TestPanopticQuality:

    def test_perfect(self):
        matched = [(0, 0, 1.0), (1, 1, 1.0)]
        pq, sq, rq = panoptic_quality(matched, n_pred=2, n_gt=2)
        assert pq == 1.0 and sq == 1.0 and rq == 1.0

    def test_no_matches(self):
        pq, sq, rq = panoptic_quality([], n_pred=5, n_gt=3)
        assert pq == 0.0

    def test_partial(self):
        matched = [(0, 0, 0.8)]
        pq, sq, rq = panoptic_quality(matched, n_pred=2, n_gt=2)
        assert abs(sq - 0.8) < 1e-6
        assert abs(rq - 0.5) < 1e-6
        assert abs(pq - 0.4) < 1e-6


class TestDetectionF1:

    def test_perfect(self):
        r = detection_f1([(0, 0, 1.0)], n_pred=1, n_gt=1)
        assert r["f1"] == 1.0

    def test_all_fp(self):
        r = detection_f1([], n_pred=5, n_gt=0)
        assert r["precision"] == 0.0
        assert r["fp"] == 5


class TestEvaluateInstanceSegmentation:

    def test_full_pipeline(self):
        pred = [np.zeros((50, 50), dtype=bool)]
        pred[0][10:30, 10:30] = True
        gt = [np.zeros((50, 50), dtype=bool)]
        gt[0][15:35, 15:35] = True
        r = evaluate_instance_segmentation(pred, gt, iou_threshold=0.3)
        assert r["tp"] == 1
        assert r["pq"] > 0
        assert "sq" in r and "rq" in r and "f1" in r
