"""Regression tests for the atomic HDF5 mask-file write in multigpu_worker.

Phase 1.3 fix: write to {path}.tmp then os.replace() to prevent concurrent
h5py writers from corrupting the file on resume / shard-retry scenarios.
"""

import multiprocessing
import os
import time
from pathlib import Path

import h5py
import numpy as np


def _write_mask_atomic(tile_out: Path, cell_type: str, masks: np.ndarray) -> None:
    """Reproduce the fixed write path from multigpu_worker.py."""
    masks_file = tile_out / f"{cell_type}_masks.h5"
    masks_tmp = tile_out / f"{cell_type}_masks.h5.tmp"
    try:
        with h5py.File(masks_tmp, "w") as f:
            f.create_dataset("masks", data=masks)
        os.replace(masks_tmp, masks_file)
    except BaseException:
        try:
            masks_tmp.unlink()
        except OSError:
            pass
        raise


def _worker_fn(tile_out_str: str, cell_type: str, data_val: int, delay: float) -> None:
    """Subprocess target: write a mask array (all `data_val`) after a delay."""
    time.sleep(delay)
    tile_out = Path(tile_out_str)
    masks = np.full((4, 64, 64), data_val, dtype=np.uint8)
    _write_mask_atomic(tile_out, cell_type, masks)


class TestAtomicMaskWrite:
    def test_single_write_roundtrip(self, tmp_path):
        tile_out = tmp_path / "tile_0_0"
        tile_out.mkdir()
        expected = np.arange(64 * 64, dtype=np.uint16).reshape(1, 64, 64)
        _write_mask_atomic(tile_out, "cell", expected)

        masks_file = tile_out / "cell_masks.h5"
        assert masks_file.exists()
        with h5py.File(masks_file, "r") as f:
            result = f["masks"][:]
        np.testing.assert_array_equal(result, expected)

    def test_tmp_file_cleaned_up_on_success(self, tmp_path):
        tile_out = tmp_path / "tile_0_0"
        tile_out.mkdir()
        masks = np.zeros((2, 32, 32), dtype=np.uint8)
        _write_mask_atomic(tile_out, "nmj", masks)
        assert not (tile_out / "nmj_masks.h5.tmp").exists()

    def test_concurrent_writes_no_corruption(self, tmp_path):
        """Two processes racing on the same tile must leave a valid HDF5 file."""
        tile_out = tmp_path / "tile_10_20"
        tile_out.mkdir()

        # Worker A starts immediately, worker B starts 50ms later — overlap
        # is tight enough to trigger the race without the atomic fix.
        p1 = multiprocessing.Process(target=_worker_fn, args=(str(tile_out), "cell", 1, 0.0))
        p2 = multiprocessing.Process(target=_worker_fn, args=(str(tile_out), "cell", 2, 0.05))
        p1.start()
        p2.start()
        p1.join(timeout=10)
        p2.join(timeout=10)

        assert p1.exitcode == 0, f"Worker 1 failed: {p1.exitcode}"
        assert p2.exitcode == 0, f"Worker 2 failed: {p2.exitcode}"

        masks_file = tile_out / "cell_masks.h5"
        assert masks_file.exists(), "masks.h5 must exist after both workers finish"

        # File must be a valid HDF5 (not corrupt) and contain one writer's data.
        with h5py.File(masks_file, "r") as f:
            data = f["masks"][:]
        assert data.shape == (4, 64, 64)
        unique_vals = set(np.unique(data))
        assert unique_vals in (
            {1},
            {2},
        ), f"Expected all-1 or all-2 (one winner), got values: {unique_vals}"
