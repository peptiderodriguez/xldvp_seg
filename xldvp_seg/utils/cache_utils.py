"""Shared primitives for the per-slide preprocessing caches.

Both the flat-field illumination cache and the tissue-filter cache need the
same building blocks:

    - CZI identity metadata (path, mtime, size) for cache-key validation
    - A ``scene`` field in the cache key (multi-scene CZI safety) with a
      back-compat default of 0 for caches written before scene joined the key
    - A metadata-mismatch check that reports exactly which key differed, with
      the raw stored value (not the back-compat default) for debuggability
    - An atomic-create advisory lock (``O_CREAT | O_EXCL``) so only one shard
      recomputes an expensive per-slide artifact on cold start

Factoring these here keeps the two cache modules from drifting.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# Sentinel that cannot collide with any JSON-serializable value — lets us
# distinguish "key absent from cached_meta" from "key present with value None".
_MISSING = object()


def czi_identity(args) -> dict:
    """Return the ``{czi_path, czi_mtime, czi_size}`` cache-key triplet.

    Reads ``args.czi_path`` (default ``""``) and best-effort ``os.stat``. On
    FS errors the mtime/size fall back to 0.0/0 — this is the safe degraded
    mode (stale caches self-heal on the next run that can read the file).
    """
    czi_path_str = str(getattr(args, "czi_path", ""))
    czi_mtime: float = 0.0
    czi_size: int = 0
    try:
        if czi_path_str:
            st = os.stat(czi_path_str)
            czi_mtime = float(st.st_mtime)
            czi_size = int(st.st_size)
    except OSError:
        pass
    return {
        "czi_path": czi_path_str,
        "czi_mtime": czi_mtime,
        "czi_size": czi_size,
    }


def get_scene(args) -> int:
    """Resolve ``args.scene`` to an int, defaulting to 0 when missing or None.

    argparse always produces an int default of 0. The extra ``or 0`` guards
    against ``SimpleNamespace(scene=None)`` which some tests construct.
    """
    scene = getattr(args, "scene", 0) or 0
    return int(scene)


def meta_mismatch(
    cached: dict,
    expected: dict,
    keys: Iterable[str],
    *,
    scene_backcompat: bool = True,
) -> tuple[str, Any, Any] | None:
    """Check cache freshness against the expected metadata, key by key.

    Returns ``(key, raw_stored_value, expected_value)`` on the first mismatch,
    where ``raw_stored_value`` is the literal value in ``cached`` (or the
    string ``"(missing)"`` if the key was absent) — distinct from the
    back-compat effective value so a log line reports what's actually on disk.

    Returns ``None`` when every key matches.

    When ``scene_backcompat=True`` (default) a missing ``scene`` key in
    ``cached`` is treated as ``0`` for comparison. Pre-scene caches were
    always implicitly single-scene, so this hits them correctly. A non-zero
    expected scene always misses when ``cached`` lacks the key — so a cache
    from scene 0 can't be mistakenly reused for scene 1.
    """
    for key in keys:
        raw = cached.get(key, _MISSING)
        if key == "scene" and scene_backcompat and raw is _MISSING:
            effective = 0
        else:
            effective = None if raw is _MISSING else raw
        if effective != expected.get(key):
            return key, "(missing)" if raw is _MISSING else raw, expected.get(key)
    return None


def try_acquire_compute_lock(lock_path: Path | None) -> bool:
    """Atomic advisory lock via ``O_CREAT | O_EXCL``.

    Returns ``True`` if the lock was acquired (caller must compute + save +
    release). Returns ``False`` if another process is already computing —
    caller should wait via :func:`wait_for_compute` then retry load.

    ``lock_path=None`` is treated as "no lock coordination requested" and
    returns ``True`` so single-process callers don't need to special-case it.
    """
    if lock_path is None:
        return True
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError as exc:
        logger.warning(
            "Could not create compute lock %s: %s — proceeding without lock",
            lock_path,
            exc,
        )
        return True
    try:
        os.write(fd, f"pid={os.getpid()}\nhost={os.uname().nodename}\n".encode())
    finally:
        os.close(fd)
    return True


def release_compute_lock(lock_path: Path | None) -> None:
    """Best-effort ``unlink`` of an advisory lock. Tolerates already-gone."""
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def wait_for_compute(
    cache_path: Path,
    lock_path: Path,
    *,
    poll_sec: int = 30,
    stale_sec: int = 3 * 3600,
) -> None:
    """Poll until the cache file lands or the lock disappears / goes stale.

    Returns when any of:
      - ``cache_path`` exists (peer wrote it)
      - ``lock_path`` no longer exists (peer released; caller should retry
        load then fall through to acquire-and-compute if still a miss)
      - lock-file mtime exceeds ``stale_sec`` (peer died; lock force-removed)
      - elapsed wait time exceeds ``stale_sec`` (defensive cap against
        filesystem mtime anomalies)
    """
    start = time.time()
    while True:
        if cache_path.exists():
            return
        if not lock_path.exists():
            return
        try:
            lock_age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if lock_age > stale_sec:
            logger.warning(
                "Compute lock %s is %.1f h old — removing and retrying.",
                lock_path,
                lock_age / 3600,
            )
            release_compute_lock(lock_path)
            return
        if time.time() - start > stale_sec:
            logger.warning(
                "Timed out waiting for cache at %s — will recompute locally.", cache_path
            )
            return
        time.sleep(poll_sec)


__all__ = [
    "czi_identity",
    "get_scene",
    "meta_mismatch",
    "try_acquire_compute_lock",
    "release_compute_lock",
    "wait_for_compute",
]
