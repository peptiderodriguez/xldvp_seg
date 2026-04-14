"""Tests for the multi-GPU Phase 4 orchestrator's pure-logic helpers.

The orchestrator itself spawns subprocesses with CUDA, which is impractical in
unit tests. We test the merge logic, which is what the orchestrator does once
worker JSONs land — equivalent to verifying detection enrichment is correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xldvp_seg.pipeline.multigpu_phase4 import _merge_phase4_results, run_multigpu_phase4


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    """Per-tile scratch dir with two synthetic worker results."""
    d = tmp_path / "scratch"
    d.mkdir()

    # Tile 100_200: two cells (labels 5 and 7)
    (d / "100_200.json").write_text(
        json.dumps(
            {
                "5": {
                    "n_nuclei": 1,
                    "nuclear_area_um2": 42.5,
                    "nuclear_area_fraction": 0.3,
                    "largest_nucleus_um2": 42.5,
                    "nuclear_solidity": 0.95,
                    "nuclear_eccentricity": 0.4,
                    "nuclei": [{"area_um2": 42.5}],
                },
                "7": {
                    "n_nuclei": 2,
                    "nuclear_area_um2": 80.0,
                    "nuclear_area_fraction": 0.5,
                    "largest_nucleus_um2": 50.0,
                    "nuclear_solidity": 0.9,
                    "nuclear_eccentricity": 0.5,
                    "nuclei": [{"area_um2": 50.0}, {"area_um2": 30.0}],
                },
            }
        )
    )
    # Tile 300_400: one cell (label 3), no per-nucleus list
    (d / "300_400.json").write_text(
        json.dumps(
            {
                "3": {
                    "n_nuclei": 0,
                    "nuclear_area_um2": 0.0,
                    "nuclear_area_fraction": 0.0,
                    "largest_nucleus_um2": 0.0,
                    "nuclear_solidity": 0.0,
                    "nuclear_eccentricity": 0.0,
                    "nuclei": [],
                }
            }
        )
    )
    return d


def test_merge_attaches_summary_features(scratch_dir):
    by_tile = {
        "100_200": [
            {"mask_label": 5},
            {"mask_label": 7},
        ],
        "300_400": [
            {"mask_label": 3},
        ],
    }

    n_enriched = _merge_phase4_results(by_tile, scratch_dir)

    assert n_enriched == 3
    assert by_tile["100_200"][0]["features"]["n_nuclei"] == 1
    assert by_tile["100_200"][0]["features"]["nuclear_area_um2"] == 42.5
    assert by_tile["100_200"][0]["nuclei"] == [{"area_um2": 42.5}]
    assert by_tile["100_200"][1]["features"]["n_nuclei"] == 2
    assert by_tile["100_200"][1]["nuclei"] == [
        {"area_um2": 50.0},
        {"area_um2": 30.0},
    ]
    # n_nuclei=0 cell still gets summary keys but no top-level nuclei list
    assert by_tile["300_400"][0]["features"]["n_nuclei"] == 0
    assert "nuclei" not in by_tile["300_400"][0]


def test_merge_prefers_tile_mask_label(scratch_dir):
    # When both tile_mask_label and mask_label exist, tile_mask_label wins
    # (matches the single-process Phase 4 pattern in post_detection.py)
    by_tile = {
        "100_200": [{"mask_label": 999, "tile_mask_label": 5}],
    }

    n_enriched = _merge_phase4_results(by_tile, scratch_dir)
    assert n_enriched == 1
    assert by_tile["100_200"][0]["features"]["n_nuclei"] == 1


def test_merge_skips_missing_tile(scratch_dir):
    by_tile = {
        "100_200": [{"mask_label": 5}],
        "999_999": [{"mask_label": 1}],  # no JSON file written for this tile
    }

    n_enriched = _merge_phase4_results(by_tile, scratch_dir)
    assert n_enriched == 1
    assert "features" not in by_tile["999_999"][0]


def test_merge_skips_label_not_in_results(scratch_dir):
    by_tile = {
        "100_200": [
            {"mask_label": 5},  # in results
            {"mask_label": 99},  # not in results
        ],
    }

    n_enriched = _merge_phase4_results(by_tile, scratch_dir)
    assert n_enriched == 1
    assert "features" not in by_tile["100_200"][1]


def test_merge_handles_missing_mask_label(scratch_dir):
    by_tile = {
        "100_200": [{"some_other_field": 1}],  # no mask_label at all
    }

    n_enriched = _merge_phase4_results(by_tile, scratch_dir)
    assert n_enriched == 0


def test_merge_handles_corrupt_json(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "100_200.json").write_text("not valid json {")

    by_tile = {"100_200": [{"mask_label": 1}]}
    # Should log warning but not raise
    n_enriched = _merge_phase4_results(by_tile, scratch)
    assert n_enriched == 0


def test_run_multigpu_phase4_rejects_zero_workers():
    with pytest.raises(ValueError, match="num_gpus must be >= 1"):
        run_multigpu_phase4(
            by_tile={},
            detections=[],
            num_gpus=0,
            tiles_dir="/tmp",
            mask_filename="x.h5",
            pixel_size_um=1.0,
            min_nuclear_area=50,
            slide_shm_arr=None,  # never reached
            shm_name="",
            nuc_channel_idx=0,
            ch_to_slot={0: 0},
            x_start=0,
            y_start=0,
        )


def test_run_multigpu_phase4_rejects_missing_nuc_channel():
    import numpy as np

    fake_arr = np.zeros((10, 10, 1), dtype=np.uint16)
    with pytest.raises(ValueError, match="nuc_channel_idx"):
        run_multigpu_phase4(
            by_tile={},
            detections=[],
            num_gpus=1,
            tiles_dir="/tmp",
            mask_filename="x.h5",
            pixel_size_um=1.0,
            min_nuclear_area=50,
            slide_shm_arr=fake_arr,
            shm_name="missing_shm",
            nuc_channel_idx=99,  # not in ch_to_slot
            ch_to_slot={0: 0, 1: 1},
            x_start=0,
            y_start=0,
        )
