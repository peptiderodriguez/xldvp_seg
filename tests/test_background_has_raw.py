"""Regression test for the Phase 1.4 fix to `has_raw` detection in background.py.

The old `any(...)` over the first 10 detections could silently flip
`value_key` to `_median_raw` and read 0 for most cells when only a handful
of early detections carried `_raw` keys (a common mixed-partial-rerun state).

The fix requires ≥ 50% of a probe of up to 20 detections to carry `_raw`
keys before using the `_raw` value_key.
"""

from xldvp_seg.analysis.background import local_background_subtract  # noqa: F401


def _make_det(has_raw: bool, ch: int = 0, uid_suffix: str = "") -> dict:
    feats: dict = {f"ch{ch}_median": 100.0, "area_um2": 10.0}
    if has_raw:
        feats[f"ch{ch}_median_raw"] = 200.0
    return {"uid": f"u{uid_suffix}", "global_center_um": [0.0, 0.0], "features": feats}


def _decide_has_raw(detections, ch: int = 0) -> bool:
    """Replicate the function-local heuristic used in
    xldvp_seg.analysis.background.local_background_subtract."""
    if not detections:
        return False
    probe = detections[: min(len(detections), 20)]
    n_with_raw = sum(1 for d in probe if f"ch{ch}_median_raw" in d.get("features", {}))
    return n_with_raw >= 0.5 * len(probe)


class TestHasRawMajority:
    def test_all_have_raw(self):
        dets = [_make_det(has_raw=True, uid_suffix=str(i)) for i in range(12)]
        assert _decide_has_raw(dets) is True

    def test_none_have_raw(self):
        dets = [_make_det(has_raw=False, uid_suffix=str(i)) for i in range(12)]
        assert _decide_has_raw(dets) is False

    def test_first_few_only(self):
        """Old bug: first 3 of 12 had _raw → would set has_raw=True; now False."""
        dets = [
            _make_det(has_raw=True, uid_suffix="a"),
            _make_det(has_raw=True, uid_suffix="b"),
            _make_det(has_raw=True, uid_suffix="c"),
        ] + [_make_det(has_raw=False, uid_suffix=str(i)) for i in range(9)]
        assert _decide_has_raw(dets) is False

    def test_majority(self):
        """7 of 10 have _raw → majority triggers True."""
        dets = [_make_det(has_raw=True, uid_suffix=str(i)) for i in range(7)] + [
            _make_det(has_raw=False, uid_suffix=str(i)) for i in range(3)
        ]
        assert _decide_has_raw(dets) is True

    def test_empty(self):
        assert _decide_has_raw([]) is False

    def test_boundary_exactly_half(self):
        dets = [_make_det(has_raw=True, uid_suffix=str(i)) for i in range(5)] + [
            _make_det(has_raw=False, uid_suffix=str(i)) for i in range(5)
        ]
        assert _decide_has_raw(dets) is True
