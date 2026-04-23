"""Phase C.1 regression test for cross-tile vessel skip_postdedup routing.

Merged cross-tile vessels (produced by VesselStrategy.merge_cross_tile_vessels)
have no HDF5 mask in any single tile dir. `detection_loop.py` flags them
with ``skip_postdedup=True`` and sets contour_px/contour_um at merge time.
`post_detection.py` must exclude them from the `by_tile` grouping that
feeds Phase 1 + Phase 3 mask reads, AND from Phase 2 background
estimation (their missing _bg_quick_medians would pollute neighbor medians
with sentinel zeros).

This is a behavior-contract test — it locks in the exact rule the
production code must follow without exercising the full detection loop.
"""


def _replicate_by_tile_grouping(detections):
    """Mirror xldvp_seg/pipeline/post_detection.py:846-855 by_tile logic."""
    by_tile: dict[str, list[dict]] = {}
    for det in detections:
        if det.get("skip_postdedup"):
            continue
        origin = det.get("tile_origin", [0, 0])
        key = f"{origin[0]}_{origin[1]}"
        by_tile.setdefault(key, []).append(det)
    return by_tile


def _replicate_phase2_filter(detections):
    """Mirror Phase 2 bg correction's skip_postdedup filter (Phase B.1 fix)."""
    return [i for i, d in enumerate(detections) if not d.get("skip_postdedup")]


class TestCrossTileSkipContract:
    def test_by_tile_excludes_skip_postdedup(self):
        detections = [
            {"uid": "regular", "tile_origin": [0, 0], "features": {}},
            {
                "uid": "merged",
                "tile_origin": [1000, 0],
                "skip_postdedup": True,
                "features": {},
            },
        ]
        by_tile = _replicate_by_tile_grouping(detections)
        assert "0_0" in by_tile
        assert "1000_0" not in by_tile
        assert sum(len(v) for v in by_tile.values()) == 1

    def test_phase2_excludes_skip_postdedup(self):
        detections = [
            {"uid": "a", "features": {}, "_bg_quick_medians": {0: 100}},
            {"uid": "merged", "skip_postdedup": True, "features": {}},
            {"uid": "b", "features": {}, "_bg_quick_medians": {0: 200}},
        ]
        non_merged = _replicate_phase2_filter(detections)
        assert non_merged == [0, 2]
        # Merged vessel is at index 1 and excluded
        assert 1 not in non_merged

    def test_all_merged_skipped_falls_through(self):
        """Edge case: every detection is a merged vessel (pathological)."""
        detections = [
            {"uid": "m1", "skip_postdedup": True, "features": {}},
            {"uid": "m2", "skip_postdedup": True, "features": {}},
        ]
        by_tile = _replicate_by_tile_grouping(detections)
        assert by_tile == {}
        assert _replicate_phase2_filter(detections) == []

    def test_contract_matches_actual_implementation(self):
        """Sanity check: the production code uses this exact contract. If
        this test's replica drifts from `post_detection.py`, fail loudly so
        the contract is re-synced."""
        import inspect

        from xldvp_seg.pipeline import post_detection

        source = inspect.getsource(post_detection)
        # by_tile skip check
        assert (
            'if det.get("skip_postdedup"):' in source or 'if det.get("skip_postdedup"):' in source
        ), "post_detection.py no longer has a skip_postdedup guard on by_tile"
        # Phase 2 skip check
        assert (
            "skip_postdedup" in source and "non_merged_idx" in source
        ), "post_detection.py no longer has a Phase 2 skip_postdedup filter"
