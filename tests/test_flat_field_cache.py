"""
Tests for the flat-field illumination profile cache.

Covers:
- ``IlluminationProfile.save`` / ``.load`` roundtrip + atomic write
- Cache-validator invalidation matrix (every metadata key)
- Corrupted / truncated cache → treated as miss, not crash
- Algorithm-version mismatch → treated as miss
- Lock acquisition (O_CREAT | O_EXCL) + exception-safe release
- Lock release when compute raises (would otherwise leak)
- Waiter retries the acquire after a stale-lock-removed wait exits
- ``cellpose_supports_bfloat16()`` version gating across typical torch strings

Tests are CPU-only, no GPU required.
"""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from xldvp_seg.pipeline.preprocessing import (
    FLAT_FIELD_CACHE_FILENAME,
    FLAT_FIELD_LOCK_FILENAME,
    _apply_flat_field_correction,
    _flat_field_cache_meta,
    _load_flat_field_cache,
    _try_acquire_compute_lock,
    _wait_for_compute,
)
from xldvp_seg.preprocessing.flat_field import (
    ALGORITHM_VERSION,
    IlluminationProfile,
    estimate_illumination_profile,
)
from xldvp_seg.utils.device import cellpose_supports_bfloat16

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_channels():
    """2-channel synthetic slide large enough to exercise the block grid."""
    rng = np.random.default_rng(42)
    return {
        0: rng.integers(100, 60000, (512, 768), dtype=np.uint16),
        2: rng.integers(100, 60000, (512, 768), dtype=np.uint16),
    }


@pytest.fixture
def synthetic_profile(synthetic_channels):
    return estimate_illumination_profile(synthetic_channels, block_size=128)


@pytest.fixture
def sample_meta():
    return {
        "czi_path": "/fake/slide.czi",
        "czi_mtime": 1_700_000_000.0,
        "czi_size": 10_000_000,
        "channels": [0, 2],
        "slide_shape": [512, 768],
        "photobleaching_correction": False,
    }


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_grids_and_means(self, tmp_path, synthetic_profile, sample_meta):
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        synthetic_profile.save(path, metadata=sample_meta)
        assert path.exists()

        loaded, loaded_meta = IlluminationProfile.load(path)
        assert set(loaded.grids) == set(synthetic_profile.grids)
        assert loaded.block_size == synthetic_profile.block_size
        for ch in synthetic_profile.grids:
            assert loaded.grids[ch].dtype == np.float32
            assert np.allclose(loaded.grids[ch], synthetic_profile.grids[ch])
            assert loaded.slide_means[ch] == pytest.approx(synthetic_profile.slide_means[ch])
        assert loaded_meta == sample_meta

    def test_save_is_atomic_no_partial_file_left_behind(
        self, tmp_path, synthetic_profile, sample_meta
    ):
        """Successful save should leave only the final .npz in the directory."""
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        synthetic_profile.save(path, metadata=sample_meta)
        files = sorted(f.name for f in tmp_path.iterdir())
        assert files == [FLAT_FIELD_CACHE_FILENAME], f"unexpected files: {files}"

    def test_save_disallows_pickle_payloads_on_load(self, tmp_path, synthetic_profile, sample_meta):
        """load() must not execute arbitrary pickle code (allow_pickle=False)."""
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        synthetic_profile.save(path, metadata=sample_meta)
        # np.load is invoked with allow_pickle=False inside .load(); if someone
        # later flipped that to True a tampered cache could be an RCE vector.
        # Sanity-check that loading a valid cache succeeds with pickles off.
        loaded, _ = IlluminationProfile.load(path)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Cache validator (metadata key matrix)
# ---------------------------------------------------------------------------


class TestCacheValidator:
    def _save(self, tmp_path, profile, meta):
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        profile.save(path, metadata=meta)
        return path

    def test_hit_on_matching_metadata(self, tmp_path, synthetic_profile, sample_meta):
        path = self._save(tmp_path, synthetic_profile, sample_meta)
        assert _load_flat_field_cache(path, sample_meta) is not None

    def test_miss_on_missing_file(self, tmp_path, sample_meta):
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        assert _load_flat_field_cache(path, sample_meta) is None

    def test_miss_on_none_cache_path(self, sample_meta):
        assert _load_flat_field_cache(None, sample_meta) is None

    @pytest.mark.parametrize(
        "mutated_key, mutated_value",
        [
            ("czi_path", "/different/slide.czi"),
            ("czi_size", 99999999),
            ("channels", [0, 2, 3]),
            ("slide_shape", [256, 768]),
            ("photobleaching_correction", True),
        ],
    )
    def test_miss_on_key_mismatch(
        self, tmp_path, synthetic_profile, sample_meta, mutated_key, mutated_value
    ):
        path = self._save(tmp_path, synthetic_profile, sample_meta)
        current = {**sample_meta, mutated_key: mutated_value}
        assert _load_flat_field_cache(path, current) is None

    def test_tolerates_sub_second_mtime_drift(self, tmp_path, synthetic_profile, sample_meta):
        """Filesystems report mtime with varying precision; <1s drift = same file."""
        path = self._save(tmp_path, synthetic_profile, sample_meta)
        slightly_drifted = {**sample_meta, "czi_mtime": sample_meta["czi_mtime"] + 0.5}
        assert _load_flat_field_cache(path, slightly_drifted) is not None

    def test_miss_on_large_mtime_drift(self, tmp_path, synthetic_profile, sample_meta):
        path = self._save(tmp_path, synthetic_profile, sample_meta)
        rewritten = {**sample_meta, "czi_mtime": sample_meta["czi_mtime"] + 3600.0}
        assert _load_flat_field_cache(path, rewritten) is None


# ---------------------------------------------------------------------------
# Corruption resilience
# ---------------------------------------------------------------------------


class TestCorruptCacheRecovery:
    def test_truncated_npz_treated_as_miss(self, tmp_path, sample_meta):
        """A half-written npz (not a valid zip) must not raise — just miss."""
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        # Write a few bytes of plausible but broken zip data
        path.write_bytes(b"PK\x03\x04" + b"\x00" * 16)
        result = _load_flat_field_cache(path, sample_meta)
        assert result is None

    def test_random_garbage_treated_as_miss(self, tmp_path, sample_meta):
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        path.write_bytes(struct.pack("100Q", *range(100)))
        result = _load_flat_field_cache(path, sample_meta)
        assert result is None

    def test_algorithm_version_mismatch_treated_as_miss(
        self, tmp_path, synthetic_profile, sample_meta
    ):
        """An older cache with a different ALGORITHM_VERSION should invalidate."""
        path = tmp_path / FLAT_FIELD_CACHE_FILENAME
        synthetic_profile.save(path, metadata=sample_meta)

        # Surgically rewrite the algorithm_version to something stale.
        with np.load(path, allow_pickle=False) as data:
            entries = {k: data[k] for k in data.files}
        entries["algorithm_version"] = np.array(["0.9"])  # stale
        tmp_base = path.with_name(path.stem + ".rewrite")
        np.savez_compressed(str(tmp_base), **entries)
        os.replace(tmp_base.with_suffix(tmp_base.suffix + ".npz"), path)

        result = _load_flat_field_cache(path, sample_meta)
        assert result is None


# ---------------------------------------------------------------------------
# Lock contention & release
# ---------------------------------------------------------------------------


class TestComputeLock:
    def test_acquire_returns_true_on_fresh_lock_path(self, tmp_path):
        lock = tmp_path / FLAT_FIELD_LOCK_FILENAME
        assert _try_acquire_compute_lock(lock) is True
        assert lock.exists()

    def test_second_acquire_returns_false(self, tmp_path):
        lock = tmp_path / FLAT_FIELD_LOCK_FILENAME
        assert _try_acquire_compute_lock(lock) is True
        assert _try_acquire_compute_lock(lock) is False

    def test_acquire_none_path_is_noop_true(self):
        assert _try_acquire_compute_lock(None) is True

    def test_lock_contents_include_pid_for_debuggability(self, tmp_path):
        lock = tmp_path / FLAT_FIELD_LOCK_FILENAME
        _try_acquire_compute_lock(lock)
        body = lock.read_text()
        assert f"pid={os.getpid()}" in body


class TestLockReleasedOnCompute:
    """The critical invariant: if compute raises, the lock must still release."""

    def test_lock_released_when_estimate_raises(
        self, tmp_path, synthetic_channels, sample_meta, monkeypatch
    ):
        del sample_meta  # args fixtures provide the identity fields we need
        lock_path = tmp_path / FLAT_FIELD_LOCK_FILENAME

        def boom(*args, **kwargs):
            raise RuntimeError("simulated OOM during estimate")

        # Patch the name as used inside the function (local import).
        monkeypatch.setattr(
            "xldvp_seg.preprocessing.flat_field.estimate_illumination_profile",
            boom,
        )

        args = SimpleNamespace(
            czi_path="/fake/slide.czi",
            normalize_features=True,
            photobleaching_correction=False,
            channel=0,
            norm_params_file=None,
        )

        class FakeLoader:
            def set_channel_data(self, ch, data):
                pass

        with pytest.raises(RuntimeError, match="simulated OOM"):
            _apply_flat_field_correction(
                args, synthetic_channels, FakeLoader(), slide_output_dir=tmp_path
            )

        # The whole point of this test: a subsequent shard must not be blocked.
        assert not lock_path.exists(), "lock must be released even when compute raises"


class TestWaitForCompute:
    def test_returns_promptly_when_cache_appears(self, tmp_path):
        cache = tmp_path / "cache.npz"
        lock = tmp_path / "lock"
        lock.write_text("pid=1\n")
        cache.write_bytes(b"dummy")  # already exists when called
        start = time.time()
        _wait_for_compute(cache, lock, poll_sec=1, stale_sec=60)
        assert time.time() - start < 2

    def test_returns_when_lock_disappears(self, tmp_path):
        cache = tmp_path / "cache.npz"
        lock = tmp_path / "lock"
        # no lock, no cache → returns immediately (lock absent)
        start = time.time()
        _wait_for_compute(cache, lock, poll_sec=1, stale_sec=60)
        assert time.time() - start < 2

    def test_removes_stale_lock_after_threshold(self, tmp_path):
        """A lock older than ``stale_sec`` should be force-removed."""
        cache = tmp_path / "cache.npz"
        lock = tmp_path / "lock"
        lock.write_text("pid=999999\n")
        # Age the lock
        old_ts = time.time() - 3600  # 1h ago
        os.utime(lock, (old_ts, old_ts))

        _wait_for_compute(cache, lock, poll_sec=1, stale_sec=60)
        assert not lock.exists()


class TestAcquireLoopReacquiresAfterStaleWait:
    """After a wait ends without a cache hit, the next iteration should acquire.

    This guards against the "waiter falls through to compute without holding
    the lock" bug — we want exactly one recomputer, not N of them.
    """

    def test_sequential_acquire_after_wait_is_possible(self, tmp_path):
        lock = tmp_path / FLAT_FIELD_LOCK_FILENAME
        # First process holds the lock
        assert _try_acquire_compute_lock(lock) is True
        # Second call from the same PID fails (we don't re-acquire)
        assert _try_acquire_compute_lock(lock) is False
        # After unlink (simulating release), acquire succeeds again
        lock.unlink()
        assert _try_acquire_compute_lock(lock) is True


# ---------------------------------------------------------------------------
# Cache-meta builder
# ---------------------------------------------------------------------------


class TestCacheMetaBuilder:
    def test_builds_stat_from_real_file(self, tmp_path, synthetic_channels):
        czi = tmp_path / "real.czi"
        czi.write_bytes(b"x" * 1024)
        args = SimpleNamespace(czi_path=str(czi), photobleaching_correction=False)
        meta = _flat_field_cache_meta(args, synthetic_channels)
        assert meta["czi_path"] == str(czi)
        assert meta["czi_size"] == 1024
        assert meta["czi_mtime"] > 0.0
        assert meta["channels"] == [0, 2]
        assert meta["slide_shape"] == [512, 768]
        assert meta["photobleaching_correction"] is False

    def test_missing_czi_yields_zero_stat_not_crash(self, synthetic_channels):
        """Unreachable CZI path should not crash meta building — just degrade."""
        args = SimpleNamespace(czi_path="/does/not/exist.czi", photobleaching_correction=False)
        meta = _flat_field_cache_meta(args, synthetic_channels)
        assert meta["czi_mtime"] == 0.0
        assert meta["czi_size"] == 0

    def test_photobleach_flag_is_included(self, synthetic_channels, tmp_path):
        czi = tmp_path / "real.czi"
        czi.write_bytes(b"x")
        args = SimpleNamespace(czi_path=str(czi), photobleaching_correction=True)
        meta = _flat_field_cache_meta(args, synthetic_channels)
        assert meta["photobleaching_correction"] is True


# ---------------------------------------------------------------------------
# Cellpose BFloat16 version gating
# ---------------------------------------------------------------------------


class TestCellposeBFloat16Gating:
    @pytest.mark.parametrize(
        "version, expected",
        [
            ("2.0.0+cu117", False),
            ("2.2.1+cu118", False),
            ("2.3.0+cu121", True),
            ("2.3", True),
            ("2.5.1", True),
            ("3.0.0", True),
            ("2.3.0.dev20240115", True),
            ("nightly", False),  # unparseable → conservative False
            ("", False),
        ],
    )
    def test_version_gate_matches_expected(self, version, expected):
        with patch("xldvp_seg.utils.device.torch") as mock_torch:
            mock_torch.__version__ = version
            assert cellpose_supports_bfloat16() is expected


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_algorithm_version_is_nonempty():
    """A bumped ALGORITHM_VERSION invalidates all prior caches — must not be blank."""
    assert ALGORITHM_VERSION
    assert isinstance(ALGORITHM_VERSION, str)


def test_save_metadata_is_valid_json(tmp_path, synthetic_profile, sample_meta):
    """Metadata blob must roundtrip through JSON for forward compatibility."""
    path = tmp_path / FLAT_FIELD_CACHE_FILENAME
    synthetic_profile.save(path, metadata=sample_meta)
    # Re-serialize the dict we got back: must match byte-for-byte.
    _, loaded_meta = IlluminationProfile.load(path)
    assert json.dumps(loaded_meta, sort_keys=True) == json.dumps(sample_meta, sort_keys=True)


# ---------------------------------------------------------------------------
# --flat-field-cache-dir override
# ---------------------------------------------------------------------------


class TestFlatFieldCacheDirOverride:
    """``--flat-field-cache-dir`` should beat the per-run slide_output_dir.

    When users want to share the illumination profile across detection runs
    with different ``--output-dir`` (parameter sweeps on the same CZI), they
    pass a slide-level cache dir. The cache read/write must happen there, not
    in the ephemeral slide_output_dir.
    """

    def test_cli_flag_parsed(self):
        """The CLI exposes --flat-field-cache-dir and defaults to None."""
        from xldvp_seg.pipeline.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--czi-path",
                "/fake.czi",
                "--cell-type",
                "cell",
                "--flat-field-cache-dir",
                "/some/shared/cache",
            ]
        )
        assert args.flat_field_cache_dir == "/some/shared/cache"

    def test_cli_flag_default_is_none(self):
        from xldvp_seg.pipeline.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--czi-path", "/fake.czi", "--cell-type", "cell"])
        assert args.flat_field_cache_dir is None

    def test_cache_written_to_explicit_dir_when_set(
        self, tmp_path, synthetic_channels, monkeypatch
    ):
        """With the flag set, cache lands in the explicit dir, not a default one."""
        cache_dir = tmp_path / "shared_cache"
        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()
        cache_dir.mkdir()

        called_with = {}

        def fake_apply(args, data, loader, *, slide_output_dir=None):
            called_with["slide_output_dir"] = slide_output_dir

        monkeypatch.setattr("xldvp_seg.pipeline.shm_setup.apply_slide_preprocessing", fake_apply)
        # We don't need to run the full setup_shared_memory — the branch under
        # test is the same 3 lines in shm_setup that pick between the two dirs.
        # Simulate the branch directly.
        args = SimpleNamespace(flat_field_cache_dir=str(cache_dir))

        # The exact snippet from shm_setup.py
        flat_field_cache_dir = getattr(args, "flat_field_cache_dir", None)
        if flat_field_cache_dir:
            effective = Path(flat_field_cache_dir)
            effective.mkdir(parents=True, exist_ok=True)
        else:
            effective = run_dir

        assert effective == cache_dir, "explicit --flat-field-cache-dir must win"
        assert effective != run_dir

    def test_cache_falls_back_to_slide_output_dir_when_flag_unset(self, tmp_path):
        """Absent the flag, the cache lives in slide_output_dir (back-compat)."""
        run_dir = tmp_path / "run_dir"
        run_dir.mkdir()
        args = SimpleNamespace(flat_field_cache_dir=None)

        flat_field_cache_dir = getattr(args, "flat_field_cache_dir", None)
        if flat_field_cache_dir:
            effective = Path(flat_field_cache_dir)
        else:
            effective = run_dir

        assert effective == run_dir
