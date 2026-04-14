"""Shared helper for attaching a slide's SHM segment inside a worker process.

Used by both ``multigpu_phase4.py`` (GPU workers for nuclear counting) and
``multiprocess_tiles.py`` (CPU workers for per-tile post-dedup phases).
Keeping the attach logic in one place ensures the contract — shape, dtype,
buffer view, caller's responsibility to close — stays consistent.
"""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import numpy as np


def attach_slide_shm(
    shm_name: str,
    shm_shape: tuple[int, ...] | list[int],
    shm_dtype: str,
) -> tuple[SharedMemory, np.ndarray]:
    """Attach to an existing SHM segment and return ``(handle, array_view)``.

    The numpy array is a zero-copy view onto the SHM buffer. The caller is
    responsible for calling ``handle.close()`` when done (typically at worker
    exit) — the parent process owns the SHM lifetime and will ``unlink()``.

    Args:
        shm_name: Name of the existing SHM segment.
        shm_shape: Shape tuple of the array.
        shm_dtype: String representation of the numpy dtype.

    Returns:
        ``(SharedMemory handle, np.ndarray view)``.

    Raises:
        FileNotFoundError: If the SHM segment with *shm_name* does not exist.
    """
    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(tuple(shm_shape), dtype=np.dtype(shm_dtype), buffer=shm.buf)
    return shm, arr
