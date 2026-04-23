"""Unit tests for xldvp_seg.utils.cache_utils — the shared primitives both
preprocessing caches depend on.

Focus is on the functions that aren't exercised indirectly via the two
cache-suite test files: the meta_mismatch reporter (raw vs effective values),
the scene back-compat asymmetry (must hit on scene=0, must miss on scene=1),
get_scene's None-guard, and the lock / wait / release lifecycle.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from types import SimpleNamespace

from xldvp_seg.utils.cache_utils import (
    czi_identity,
    get_scene,
    meta_mismatch,
    release_compute_lock,
    try_acquire_compute_lock,
    wait_for_compute,
)

# ---------------------------------------------------------------------------
# czi_identity
# ---------------------------------------------------------------------------


class TestCziIdentity:
    def test_returns_path_mtime_size_from_real_file(self, tmp_path):
        czi = tmp_path / "slide.czi"
        czi.write_bytes(b"x" * 512)
        args = SimpleNamespace(czi_path=str(czi))
        meta = czi_identity(args)
        assert meta["czi_path"] == str(czi)
        assert meta["czi_size"] == 512
        assert meta["czi_mtime"] > 0.0

    def test_missing_file_degrades_to_zero(self):
        args = SimpleNamespace(czi_path="/does/not/exist.czi")
        meta = czi_identity(args)
        assert meta["czi_mtime"] == 0.0
        assert meta["czi_size"] == 0
        assert meta["czi_path"] == "/does/not/exist.czi"

    def test_missing_czi_path_attr_yields_empty_string(self):
        args = SimpleNamespace()
        meta = czi_identity(args)
        assert meta["czi_path"] == ""
        assert meta["czi_mtime"] == 0.0
        assert meta["czi_size"] == 0


# ---------------------------------------------------------------------------
# get_scene
# ---------------------------------------------------------------------------


class TestGetScene:
    def test_default_zero_when_attr_missing(self):
        assert get_scene(SimpleNamespace()) == 0

    def test_passthrough_positive_int(self):
        assert get_scene(SimpleNamespace(scene=3)) == 3

    def test_none_coerced_to_zero(self):
        """argparse int='scene' can't be None but hand-built namespaces can."""
        assert get_scene(SimpleNamespace(scene=None)) == 0

    def test_string_int_coerced(self):
        assert get_scene(SimpleNamespace(scene="5")) == 5


# ---------------------------------------------------------------------------
# meta_mismatch
# ---------------------------------------------------------------------------


class TestMetaMismatch:
    def test_returns_none_on_full_match(self):
        cached = {"a": 1, "b": "x", "scene": 0}
        expected = {"a": 1, "b": "x", "scene": 0}
        assert meta_mismatch(cached, expected, ["a", "b", "scene"]) is None

    def test_reports_first_mismatch_with_raw_values(self):
        cached = {"a": 1, "b": "x"}
        expected = {"a": 1, "b": "y"}
        result = meta_mismatch(cached, expected, ["a", "b"])
        assert result == ("b", "x", "y")

    def test_missing_nonscene_key_reported_as_missing_sentinel(self):
        cached = {"a": 1}  # b absent
        expected = {"a": 1, "b": "y"}
        result = meta_mismatch(cached, expected, ["a", "b"])
        assert result is not None
        key, stored, current = result
        assert key == "b"
        assert stored == "(missing)"
        assert current == "y"

    def test_scene_backcompat_hits_on_missing_scene_and_zero_expected(self):
        """Pre-scene caches had no 'scene' key. Back-compat treats absent as 0."""
        cached = {"a": 1}  # no scene
        expected = {"a": 1, "scene": 0}
        assert meta_mismatch(cached, expected, ["a", "scene"]) is None

    def test_scene_backcompat_misses_on_missing_scene_and_nonzero_expected(self):
        """A pre-scene cache must NOT false-hit when current run wants scene=1."""
        cached = {"a": 1}  # no scene
        expected = {"a": 1, "scene": 1}
        result = meta_mismatch(cached, expected, ["a", "scene"])
        assert result is not None
        key, stored, current = result
        assert key == "scene"
        # Stored must show the raw truth (missing), not the back-compat default
        assert stored == "(missing)"
        assert current == 1

    def test_scene_backcompat_off_disables_missing_scene_shim(self):
        cached = {"a": 1}
        expected = {"a": 1, "scene": 0}
        result = meta_mismatch(cached, expected, ["a", "scene"], scene_backcompat=False)
        assert result is not None
        assert result[0] == "scene"

    def test_explicit_scene_in_cached_always_compared_literally(self):
        """If cached has scene=0 explicitly and current wants scene=1, still miss."""
        cached = {"scene": 0}
        expected = {"scene": 1}
        result = meta_mismatch(cached, expected, ["scene"])
        assert result is not None
        assert result == ("scene", 0, 1)


# ---------------------------------------------------------------------------
# try_acquire_compute_lock + release_compute_lock
# ---------------------------------------------------------------------------


class TestComputeLockLifecycle:
    def test_acquire_returns_true_on_fresh_path(self, tmp_path):
        lock = tmp_path / "lock"
        assert try_acquire_compute_lock(lock) is True
        assert lock.exists()

    def test_second_acquire_returns_false(self, tmp_path):
        lock = tmp_path / "lock"
        assert try_acquire_compute_lock(lock) is True
        assert try_acquire_compute_lock(lock) is False

    def test_release_removes_lock(self, tmp_path):
        lock = tmp_path / "lock"
        try_acquire_compute_lock(lock)
        release_compute_lock(lock)
        assert not lock.exists()

    def test_release_tolerates_missing_lock(self, tmp_path):
        """release() must be idempotent — called in `finally` when lock may not exist."""
        lock = tmp_path / "lock"
        release_compute_lock(lock)  # no prior acquire
        # should not raise

    def test_acquire_none_path_is_noop_true(self):
        assert try_acquire_compute_lock(None) is True

    def test_release_none_path_is_noop(self):
        release_compute_lock(None)  # no-op

    def test_lock_contents_have_pid_and_host(self, tmp_path):
        lock = tmp_path / "lock"
        try_acquire_compute_lock(lock)
        body = lock.read_text()
        assert f"pid={os.getpid()}" in body
        assert "host=" in body


# ---------------------------------------------------------------------------
# wait_for_compute
# ---------------------------------------------------------------------------


class TestWaitForCompute:
    def test_returns_immediately_when_cache_exists(self, tmp_path):
        cache = tmp_path / "cache.npz"
        lock = tmp_path / "lock"
        cache.write_bytes(b"x")
        lock.write_text("pid=1\n")
        start = time.time()
        wait_for_compute(cache, lock, poll_sec=1, stale_sec=60)
        assert time.time() - start < 2

    def test_returns_immediately_when_lock_gone(self, tmp_path):
        cache = tmp_path / "cache.npz"
        lock = tmp_path / "lock"
        start = time.time()
        wait_for_compute(cache, lock, poll_sec=1, stale_sec=60)
        assert time.time() - start < 2

    def test_force_removes_stale_lock(self, tmp_path):
        cache = tmp_path / "cache.npz"
        lock = tmp_path / "lock"
        lock.write_text("pid=999999\n")
        old_ts = time.time() - 3600
        os.utime(lock, (old_ts, old_ts))
        wait_for_compute(cache, lock, poll_sec=1, stale_sec=60)
        assert not lock.exists()


# ---------------------------------------------------------------------------
# Multiprocess concurrency: two processes racing to acquire the same lock
# ---------------------------------------------------------------------------


def _child_acquire_and_hold(lock_path_str, hold_sec, out_q):
    """Helper executed in a subprocess: acquire the lock and hold it."""
    from xldvp_seg.utils.cache_utils import (
        release_compute_lock,
        try_acquire_compute_lock,
    )

    lp = Path(lock_path_str)
    got = try_acquire_compute_lock(lp)
    out_q.put(("acquired", got))
    if got:
        time.sleep(hold_sec)
        release_compute_lock(lp)
    out_q.put(("done", got))


class TestConcurrentLockAcquisition:
    """Exactly one of N racing processes wins the O_CREAT|O_EXCL lock."""

    def test_only_one_of_two_processes_acquires(self, tmp_path):
        ctx = mp.get_context("spawn")
        lock_path = tmp_path / "race_lock"
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        p1 = ctx.Process(target=_child_acquire_and_hold, args=(str(lock_path), 0.3, q1))
        p2 = ctx.Process(target=_child_acquire_and_hold, args=(str(lock_path), 0.3, q2))
        p1.start()
        p2.start()
        p1.join(timeout=10)
        p2.join(timeout=10)
        results = [q1.get(timeout=5), q2.get(timeout=5)]
        # Each child's first message is ("acquired", got)
        acquired = sum(1 for (tag, got) in results if tag == "acquired" and got)
        denied = sum(1 for (tag, got) in results if tag == "acquired" and not got)
        assert acquired == 1, f"expected exactly one winner, got {acquired}"
        assert denied == 1

    def test_lock_is_re_acquirable_after_release(self, tmp_path):
        lock = tmp_path / "lock"
        assert try_acquire_compute_lock(lock) is True
        release_compute_lock(lock)
        # Next acquire (simulating a subsequent shard after winner finished) must succeed.
        assert try_acquire_compute_lock(lock) is True
