"""Generic multi-process per-tile executor with SHM attachment.

This module provides a reusable abstraction for running per-tile work in
parallel across CPU processes. It solves the ``h5py`` global lock (``phil``)
problem that cripples ThreadPool-based HDF5 I/O: each Python process has its
own ``phil``, so concurrent HDF5 reads truly parallelize when the workload is
split across processes.

**When to use this:**
  - Per-tile post-processing that reads HDF5 masks and/or SHM channel data.
  - Workloads where ``h5py.File.read(...)`` dominates per-tile cost.
  - Anywhere ThreadPool gave you ~1-3× parallelism despite more cores.

**When NOT to use this:**
  - Pure in-memory compute (no HDF5, no SHM) — a ThreadPool is fine.
  - GPU workloads — see ``multigpu_phase4.py`` for the GPU pattern (separate
    because GPU workers have different lifecycle and pinning).

**Design:**
  - Each worker attaches once to the slide's SHM segment (via
    :func:`xldvp_seg.processing.shm_attach.attach_slide_shm`) at init time.
  - Tasks are dispatched one-per-item via
    ``ProcessPoolExecutor.map(chunksize=1)`` so variable-cost tiles load-balance.
  - Results stream back as an iterator; the caller merges them into
    main-process state.
  - Task functions are plain module-level callables, picklable by name.

**Testing:**
  The worker task functions are pure — call them directly with a mock
  ``WorkerContext`` in unit tests. The :class:`TileProcessor` class itself
  requires spawning processes; smoke-test it but keep it out of fast CI.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SharedSlideConfig:
    """Configuration passed to worker processes to attach to a slide's SHM.

    All fields must be picklable (primitives + basic containers). Stored as a
    frozen dataclass so it can be passed as ``initargs`` to the process pool.
    """

    shm_name: str
    shm_shape: tuple[int, ...]
    shm_dtype: str
    ch_to_slot: dict[int, int]  # CZI channel index → SHM slot index
    x_start: int
    y_start: int
    tiles_dir: str
    mask_filename: str


class WorkerContext:
    """Per-worker state accessible to task functions.

    Attributes:
        slide_arr: numpy view onto the slide's SHM (zero-copy).
        slide_config: the :class:`SharedSlideConfig` the worker was initialized with.
    """

    def __init__(self, slide_arr: np.ndarray, slide_config: SharedSlideConfig):
        self.slide_arr = slide_arr
        self.slide_config = slide_config


# Module-level per-worker state. Set by :func:`_worker_init`.
_WORKER_SHM: Any | None = None
_WORKER_CTX: WorkerContext | None = None


def _worker_init(slide_config: SharedSlideConfig) -> None:
    """Initialize a worker process: attach SHM, prime heavy imports.

    Runs once per worker. Imports that matter (h5py + hdf5plugin codec
    registration, cv2) are done here so the first task doesn't pay the cost.
    """
    # Import before h5py so the LZ4 codec registers
    import cv2  # noqa: F401 — warm the import
    import h5py  # noqa: F401 — warm the import
    import hdf5plugin  # noqa: F401

    from xldvp_seg.processing.shm_attach import attach_slide_shm

    global _WORKER_SHM, _WORKER_CTX
    shm, slide_arr = attach_slide_shm(
        slide_config.shm_name, slide_config.shm_shape, slide_config.shm_dtype
    )
    _WORKER_SHM = shm
    _WORKER_CTX = WorkerContext(slide_arr=slide_arr, slide_config=slide_config)


def _worker_run_task(task_fn_qualname: str, task: dict) -> Any:
    """Dispatcher that resolves ``task_fn_qualname`` and invokes it.

    Called by ``ProcessPoolExecutor.submit``. We pass the function's qualname
    (module:name) rather than the callable itself so workers re-import it
    cleanly after spawn (avoids pickling closures or partial bindings).
    """
    if _WORKER_CTX is None:
        raise RuntimeError("Worker not initialized — _worker_init was not called")

    module_name, _, fn_name = task_fn_qualname.rpartition(":")
    if not module_name or not fn_name:
        raise ValueError(
            f"task_fn_qualname must be 'module.path:function_name', got {task_fn_qualname!r}"
        )
    import importlib

    mod = importlib.import_module(module_name)
    fn = getattr(mod, fn_name)
    return fn(task, _WORKER_CTX)


def _resolve_qualname(fn: Callable) -> str:
    """Return the 'module:name' qualname used to re-import *fn* in workers."""
    return f"{fn.__module__}:{fn.__qualname__}"


def _default_num_workers() -> int:
    """Default worker count: ``min(SLURM_CPUS_PER_TASK, 64)``.

    Capped at 64 so memory footprint (each process imports numpy/scipy/h5py —
    a few hundred MB) stays reasonable on typical HPC nodes.
    """
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "") or (os.cpu_count() or 1))
    except (ValueError, TypeError):
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, 64))


class TileProcessor:
    """Generic multi-process per-tile executor.

    Args:
        slide_config: SHM + paths needed by worker processes.
        num_workers: Number of worker processes. Defaults to
            ``min(SLURM_CPUS_PER_TASK, 64)``.

    Usage:
        >>> cfg = SharedSlideConfig(...)
        >>> proc = TileProcessor(cfg, num_workers=32)
        >>> tasks = [{"tile_key": "0_0", ...}, ...]
        >>> for task, result in proc.run(my_task_fn, tasks):
        ...     apply_update(result)
    """

    def __init__(self, slide_config: SharedSlideConfig, num_workers: int | None = None):
        if num_workers is None:
            num_workers = _default_num_workers()
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self.slide_config = slide_config
        self.num_workers = num_workers

    def run(
        self,
        task_fn: Callable[[dict, WorkerContext], Any],
        tasks: list[dict],
        *,
        desc: str = "tile",
        largest_first_key: Callable[[dict], int] | None = None,
    ) -> Iterator[tuple[dict, Any]]:
        """Yield ``(task, result)`` pairs as tasks complete (unordered).

        Args:
            task_fn: Module-level function with signature
                ``(task: dict, ctx: WorkerContext) -> Any``. Must be importable
                by its ``__module__`` / ``__qualname__`` — no lambdas or closures.
            tasks: List of task dicts (each picklable — primitives + basic
                containers only).
            desc: Label for logs.
            largest_first_key: Optional function mapping a task to a size
                metric. If provided, tasks are sorted in descending order
                before dispatch so stragglers start early.

        Yields:
            ``(task, result)`` pairs in completion order. If a task raises,
            the exception propagates.
        """
        if not tasks:
            return

        qualname = _resolve_qualname(task_fn)

        ordered = (
            sorted(tasks, key=largest_first_key, reverse=True)
            if largest_first_key is not None
            else list(tasks)
        )

        ctx = mp.get_context("spawn")
        logger.info(
            "Starting %s processing: %d tasks, %d workers",
            desc,
            len(ordered),
            self.num_workers,
        )

        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.slide_config,),
        ) as pool:
            futures = {pool.submit(_worker_run_task, qualname, t): t for t in ordered}
            for f in as_completed(futures):
                task = futures[f]
                # Let exceptions propagate so the caller decides policy
                result = f.result()
                yield task, result
