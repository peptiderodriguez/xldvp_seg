"""Tests for the generic ``TileProcessor`` and Phase 1/3 MP task functions.

Strategy: the worker task functions are pure — call them directly with a mock
``WorkerContext`` and synthetic in-memory masks/channels. This exercises all
the business logic without spawning subprocesses (which is slow/flaky in CI).

A single smoke test does spawn one worker end-to-end to verify the
``TileProcessor`` plumbing works.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401 — register LZ4 codec
import numpy as np
import pytest

from xldvp_seg.pipeline.post_detection import (
    _apply_tile_updates,
    _build_phase1_tasks,
    _build_phase3_tasks,
    _phase1_mp_task,
    _phase3_mp_task,
)
from xldvp_seg.processing.multiprocess_tiles import (
    SharedSlideConfig,
    TileProcessor,
    WorkerContext,
    _default_num_workers,
    _resolve_qualname,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def synthetic_tile(tmp_path: Path):
    """Build a tile with 2 labels + matching single-channel SHM-like array.

    Labels 1 and 2 occupy disjoint bboxes so find_objects returns both.
    Channel 0 is constant 100 inside label 1, 200 inside label 2, 0 outside.
    """
    tile_x, tile_y = 0, 0
    tile_h, tile_w = 64, 64

    # HDF5 mask file
    tile_dir = tmp_path / "tiles" / f"tile_{tile_x}_{tile_y}"
    tile_dir.mkdir(parents=True)
    masks = np.zeros((tile_h, tile_w), dtype=np.int32)
    masks[5:15, 10:25] = 1
    masks[30:50, 40:60] = 2
    with h5py.File(tile_dir / "cell_masks.h5", "w") as hf:
        hf.create_dataset("masks", data=masks)

    # Slide array (mimics SHM view): (H, W, n_channels)
    slide_arr = np.zeros((tile_h, tile_w, 1), dtype=np.uint16)
    slide_arr[5:15, 10:25, 0] = 100
    slide_arr[30:50, 40:60, 0] = 200

    slide_config = SharedSlideConfig(
        shm_name="not-actually-attached",
        shm_shape=slide_arr.shape,
        shm_dtype=str(slide_arr.dtype),
        ch_to_slot={0: 0},
        x_start=0,
        y_start=0,
        tiles_dir=str(tmp_path / "tiles"),
        mask_filename="cell_masks.h5",
    )
    ctx = WorkerContext(slide_arr=slide_arr, slide_config=slide_config)

    return SimpleNamespace(
        tile_dir=tile_dir,
        masks=masks,
        slide_arr=slide_arr,
        slide_config=slide_config,
        ctx=ctx,
        tile_key=f"{tile_x}_{tile_y}",
    )


# --------------------------------------------------------------------------
# Phase 1 task function
# --------------------------------------------------------------------------


def test_phase1_mp_task_returns_contour_and_medians(synthetic_tile):
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [
            {"idx": 0, "mask_label": 1, "tile_origin": [0, 0]},
            {"idx": 1, "mask_label": 2, "tile_origin": [0, 0]},
        ],
        "contour_processing": True,
        "pixel_size_um": 0.5,
    }
    updates = _phase1_mp_task(task, synthetic_tile.ctx)

    assert set(updates) == {0, 1}
    for idx in (0, 1):
        upd = updates[idx]
        assert "contour_px" in upd
        assert "contour_um" in upd
        assert "_bg_quick_medians" in upd
        assert isinstance(upd["contour_px"], list)
        assert len(upd["contour_px"]) >= 3

    # Label 1 pixel value = 100
    assert updates[0]["_bg_quick_medians"][0] == 100.0
    # Label 2 pixel value = 200
    assert updates[1]["_bg_quick_medians"][0] == 200.0


def test_phase1_mp_task_skips_invalid_labels(synthetic_tile):
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [
            {"idx": 0, "mask_label": 99, "tile_origin": [0, 0]},  # label not in mask
            {"idx": 1, "mask_label": 0, "tile_origin": [0, 0]},  # invalid 0
        ],
        "contour_processing": True,
        "pixel_size_um": 0.5,
    }
    updates = _phase1_mp_task(task, synthetic_tile.ctx)
    assert updates == {}


def test_phase1_mp_task_missing_tile(tmp_path, synthetic_tile):
    slide_config = SharedSlideConfig(
        shm_name="x",
        shm_shape=(10, 10, 1),
        shm_dtype="uint16",
        ch_to_slot={0: 0},
        x_start=0,
        y_start=0,
        tiles_dir=str(tmp_path / "nonexistent"),
        mask_filename="cell_masks.h5",
    )
    ctx = WorkerContext(slide_arr=np.zeros((10, 10, 1), dtype=np.uint16), slide_config=slide_config)
    task = {
        "tile_key": "0_0",
        "dets": [{"idx": 0, "mask_label": 1, "tile_origin": [0, 0]}],
        "contour_processing": True,
        "pixel_size_um": 0.5,
    }
    assert _phase1_mp_task(task, ctx) == {}


def test_phase1_mp_task_contour_processing_off(synthetic_tile):
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [{"idx": 0, "mask_label": 1, "tile_origin": [0, 0]}],
        "contour_processing": False,
        "pixel_size_um": 0.5,
    }
    updates = _phase1_mp_task(task, synthetic_tile.ctx)
    assert 0 in updates
    assert "contour_px" not in updates[0]
    # Quick medians should still be produced
    assert updates[0]["_bg_quick_medians"][0] == 100.0


def test_phase1_contour_coords_respect_tile_origin(synthetic_tile):
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [{"idx": 0, "mask_label": 1, "tile_origin": [1000, 2000]}],
        "contour_processing": True,
        "pixel_size_um": 1.0,
    }
    updates = _phase1_mp_task(task, synthetic_tile.ctx)
    contour = np.array(updates[0]["contour_px"])
    # Label 1 is at rows 5-15, cols 10-25 in tile-local; with origin (1000, 2000)
    # global x should be in [1010, 1025), global y in [2005, 2015)
    assert contour[:, 0].min() >= 1010
    assert contour[:, 0].max() < 1025
    assert contour[:, 1].min() >= 2005
    assert contour[:, 1].max() < 2015


# --------------------------------------------------------------------------
# Phase 3 task function
# --------------------------------------------------------------------------


def test_phase3_mp_task_returns_features_with_bg(synthetic_tile):
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [{"idx": 0, "mask_label": 1}, {"idx": 1, "mask_label": 2}],
        "per_cell_bg": {0: {0: 10.0}, 1: {0: 20.0}},
        "pixel_size_um": 0.5,
    }
    updates = _phase3_mp_task(task, synthetic_tile.ctx)

    assert set(updates) == {0, 1}
    f0 = updates[0]["features"]
    f1 = updates[1]["features"]
    # Raw medians preserved
    assert f0["ch0_median_raw"] == 100.0
    assert f1["ch0_median_raw"] == 200.0
    # Background recorded
    assert f0["ch0_background"] == 10.0
    assert f1["ch0_background"] == 20.0
    # SNR = raw_median / bg
    assert f0["ch0_snr"] == 10.0
    assert f1["ch0_snr"] == 10.0
    # area_um2 present
    assert f0["area_um2"] > 0
    assert f1["area_um2"] > 0


def test_phase3_mp_task_without_bg(synthetic_tile):
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [{"idx": 0, "mask_label": 1}],
        "per_cell_bg": {},
        "pixel_size_um": 0.5,
    }
    updates = _phase3_mp_task(task, synthetic_tile.ctx)
    f0 = updates[0]["features"]
    # No _raw suffix (raw pass not run)
    assert "ch0_median_raw" not in f0
    assert f0["ch0_background"] == 0.0
    assert f0["ch0_snr"] == 0.0


def test_phase3_mp_task_string_keyed_bg(synthetic_tile):
    """per_cell_bg coming in as str-keyed dict (JSON round-trip) should still work."""
    task = {
        "tile_key": synthetic_tile.tile_key,
        "dets": [{"idx": 0, "mask_label": 1}],
        "per_cell_bg": {"0": {"0": 10.0}},
        "pixel_size_um": 0.5,
    }
    updates = _phase3_mp_task(task, synthetic_tile.ctx)
    assert updates[0]["features"]["ch0_background"] == 10.0


# --------------------------------------------------------------------------
# Task builders
# --------------------------------------------------------------------------


def test_build_phase1_tasks_skips_missing_labels():
    by_tile = {
        "0_0": [
            {"_postdedup_idx": 0, "mask_label": 1, "tile_origin": [0, 0]},
            {"_postdedup_idx": 1, "mask_label": None, "tile_origin": [0, 0]},  # skip
            {"_postdedup_idx": 2, "tile_origin": [0, 0]},  # skip (no mask_label)
        ],
    }
    tasks = _build_phase1_tasks(by_tile, contour_processing=True, pixel_size_um=0.5)
    assert len(tasks) == 1
    assert len(tasks[0]["dets"]) == 1
    assert tasks[0]["dets"][0]["mask_label"] == 1


def test_build_phase1_tasks_primitives_only():
    """Task packets must contain only pickle-safe primitives."""
    by_tile = {
        "0_0": [{"_postdedup_idx": 5, "mask_label": 2, "tile_origin": [100, 200]}],
    }
    tasks = _build_phase1_tasks(by_tile, contour_processing=True, pixel_size_um=0.5)
    d = tasks[0]["dets"][0]
    assert isinstance(d["idx"], int)
    assert isinstance(d["mask_label"], int)
    assert isinstance(d["tile_origin"], list)
    assert all(isinstance(x, int) for x in d["tile_origin"])


def test_build_phase3_tasks_slices_bg_per_tile():
    by_tile = {
        "0_0": [
            {"_postdedup_idx": 0, "mask_label": 1},
            {"_postdedup_idx": 1, "mask_label": 2},
        ],
        "100_200": [{"_postdedup_idx": 2, "mask_label": 3}],
    }
    per_cell_bg = {
        0: {0: 10.0, 1: 20.0},
        1: {0: 30.0},
        2: {0: 40.0},
        999: {0: 99.0},  # should NOT appear in any task
    }
    tasks = _build_phase3_tasks(by_tile, per_cell_bg, pixel_size_um=0.5)
    by_tile_key = {t["tile_key"]: t for t in tasks}

    assert set(by_tile_key["0_0"]["per_cell_bg"]) == {0, 1}
    assert set(by_tile_key["100_200"]["per_cell_bg"]) == {2}
    assert 999 not in by_tile_key["0_0"]["per_cell_bg"]
    assert 999 not in by_tile_key["100_200"]["per_cell_bg"]


# --------------------------------------------------------------------------
# Update merger
# --------------------------------------------------------------------------


def test_apply_tile_updates_features_merge():
    dets = [{"features": {"existing": 1.0}}]
    _apply_tile_updates(dets, {0: {"features": {"new": 2.0}}})
    assert dets[0]["features"] == {"existing": 1.0, "new": 2.0}


def test_apply_tile_updates_top_level_replace():
    dets = [{"contour_px": [[1, 2]]}]
    _apply_tile_updates(dets, {0: {"contour_px": [[3, 4]]}})
    assert dets[0]["contour_px"] == [[3, 4]]


def test_apply_tile_updates_returns_count():
    dets = [{}, {}, {}]
    n = _apply_tile_updates(dets, {0: {"x": 1}, 2: {"x": 3}})
    assert n == 2


def test_apply_tile_updates_out_of_range_idx():
    dets = [{}]
    _apply_tile_updates(dets, {99: {"x": 1}})
    assert dets == [{}]


# --------------------------------------------------------------------------
# TileProcessor infrastructure
# --------------------------------------------------------------------------


def test_tile_processor_rejects_zero_workers(synthetic_tile):
    with pytest.raises(ValueError, match="num_workers"):
        TileProcessor(synthetic_tile.slide_config, num_workers=0)


def test_default_num_workers_is_positive():
    n = _default_num_workers()
    assert n >= 1
    assert n <= 64


def test_resolve_qualname_roundtrips():
    q = _resolve_qualname(_phase1_mp_task)
    assert q == "xldvp_seg.pipeline.post_detection:_phase1_mp_task"


def test_tile_processor_empty_tasks_yields_nothing(synthetic_tile):
    proc = TileProcessor(synthetic_tile.slide_config, num_workers=1)
    # No tasks = generator completes without spawning workers
    assert list(proc.run(_phase1_mp_task, [])) == []


# --------------------------------------------------------------------------
# Smoke test that actually spawns a worker (slow; real SHM needed)
# --------------------------------------------------------------------------


def test_tile_processor_spawns_worker_end_to_end(synthetic_tile, tmp_path):
    """One-worker, one-tile spawn test. Validates pickling + plumbing."""
    from multiprocessing.shared_memory import SharedMemory

    # Copy the synthetic slide into real SHM so the worker can attach by name.
    flat = synthetic_tile.slide_arr.tobytes()
    shm = SharedMemory(create=True, size=len(flat))
    try:
        shm.buf[: len(flat)] = flat
        slide_cfg = SharedSlideConfig(
            shm_name=shm.name,
            shm_shape=tuple(synthetic_tile.slide_arr.shape),
            shm_dtype=str(synthetic_tile.slide_arr.dtype),
            ch_to_slot={0: 0},
            x_start=0,
            y_start=0,
            tiles_dir=synthetic_tile.slide_config.tiles_dir,
            mask_filename="cell_masks.h5",
        )
        tasks = [
            {
                "tile_key": synthetic_tile.tile_key,
                "dets": [{"idx": 0, "mask_label": 1, "tile_origin": [0, 0]}],
                "contour_processing": False,
                "pixel_size_um": 1.0,
            }
        ]
        processor = TileProcessor(slide_cfg, num_workers=1)
        results = list(processor.run(_phase1_mp_task, tasks))
        assert len(results) == 1
        task, updates = results[0]
        assert 0 in updates
        assert updates[0]["_bg_quick_medians"][0] == 100.0
    finally:
        shm.close()
        shm.unlink()
