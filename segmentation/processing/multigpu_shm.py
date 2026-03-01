"""
Multi-GPU tile processing with shared memory (zero-copy).

This version uses shared memory to avoid serializing tile data through queues.
Slides are loaded into shared memory once, and workers read tiles directly.

Architecture:
- Main process: loads slides into shared memory, sends tile coordinates to queue
- N GPU workers: each pinned to one GPU, reads tiles from shared memory, processes, returns results

Performance improvement: ~27MB per tile no longer serialized, only ~100 bytes of coordinates.

Coordinate System:
    Workers receive tiles with GLOBAL CZI coordinates. The worker converts to
    relative array indices by subtracting the mosaic origin. Shared memory
    contains the loaded mosaic data starting at array index (0, 0).

    Tile dict format:
        tile = {
            'x': 45000,  # Global CZI X coordinate
            'y': 12000,  # Global CZI Y coordinate
            'w': 3000,   # Tile width
            'h': 3000,   # Tile height
        }

    Worker tile extraction (in multigpu_worker.py):
        mosaic_ox, mosaic_oy = slide_info[slide_name].get('mosaic_origin', (0, 0))
        y_rel = tile['y'] - mosaic_oy
        x_rel = tile['x'] - mosaic_ox
        tile_img = slide_arr[y_rel:y_rel+tile['h'], x_rel:x_rel+tile['w']]

    See docs/COORDINATE_SYSTEM.md for the complete specification.
"""

import atexit
import signal
from typing import Dict, Any, Set
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry of shared memory names for cleanup on crash
_shm_registry: Set[str] = set()

def _cleanup_shared_memory_on_exit():
    """Emergency cleanup of shared memory on process exit.

    This ensures shared memory is released even if the main process crashes
    or is killed. Without this, shared memory persists until system reboot.
    """
    for shm_name in list(_shm_registry):
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            logger.info(f"Emergency cleanup: unlinked shared memory {shm_name}")
        except FileNotFoundError:
            pass  # Already cleaned up
        except Exception as e:
            logger.warning(f"Failed to cleanup shared memory {shm_name}: {e}")
    _shm_registry.clear()

# Register cleanup on normal exit
atexit.register(_cleanup_shared_memory_on_exit)

# Register cleanup on SIGTERM (e.g., SLURM job cancellation, timeout)
def _signal_cleanup(signum, frame):
    """Clean up shared memory on SIGTERM/SIGINT before exiting."""
    _cleanup_shared_memory_on_exit()
    raise SystemExit(128 + signum)

if multiprocessing.current_process().name == 'MainProcess':
    signal.signal(signal.SIGTERM, _signal_cleanup)


def register_shm_for_cleanup(name: str):
    """Register a shared memory name for emergency cleanup on process exit."""
    _shm_registry.add(name)


def unregister_shm_for_cleanup(name: str):
    """Remove a shared memory name from the cleanup registry."""
    _shm_registry.discard(name)


class SharedSlideManager:
    """Manages shared memory for slide data."""

    def __init__(self):
        self.shared_memories: Dict[str, SharedMemory] = {}
        self.slide_info: Dict[str, Dict[str, Any]] = {}

    def add_slide(self, name: str, data: np.ndarray) -> Dict[str, Any]:
        """
        Copy slide data into shared memory.

        Args:
            name: Slide name
            data: Numpy array with slide data

        Returns:
            Dict with shared memory info (shm_name, shape, dtype)
        """
        # Create shared memory
        shm = SharedMemory(create=True, size=data.nbytes)

        # Register for emergency cleanup (in case of crash)
        _shm_registry.add(shm.name)

        # Create numpy array backed by shared memory and copy data
        shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared_arr[:] = data[:]

        # Store references
        self.shared_memories[name] = shm
        info = {
            'shm_name': shm.name,
            'shape': data.shape,
            'dtype': str(data.dtype),
        }
        self.slide_info[name] = info

        logger.debug(f"Created shared memory for {name}: {data.nbytes / 1e9:.2f} GB")
        return info

    def create_slide_buffer(self, name: str, shape: tuple, dtype) -> np.ndarray:
        """Create shared memory and return numpy array backed by it for direct loading.

        Args:
            name: Slide name
            shape: Shape of the array to create
            dtype: Data type of the array

        Returns:
            numpy array backed by shared memory (caller can load data directly into it)
        """
        size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        shm = SharedMemory(create=True, size=size)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Register for emergency cleanup (in case of crash)
        _shm_registry.add(shm.name)

        self.shared_memories[name] = shm
        self.slide_info[name] = {'shm_name': shm.name, 'shape': shape, 'dtype': str(dtype)}

        logger.info(f"Created shared memory for {name}: {size/1e9:.2f} GB")
        return arr

    def get_slide_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info dict for all slides (to pass to workers)."""
        return self.slide_info.copy()

    def cleanup_slide(self, name: str):
        """Release shared memory for a single slide."""
        if name in self.shared_memories:
            try:
                shm = self.shared_memories[name]
                # Remove from emergency cleanup registry
                _shm_registry.discard(shm.name)
                shm.close()
                shm.unlink()
                logger.debug(f"Released shared memory for {name}")
            except Exception as e:
                logger.warning(f"Error releasing shared memory for {name}: {e}")
            del self.shared_memories[name]
        if name in self.slide_info:
            del self.slide_info[name]

    def cleanup(self):
        """Release all shared memory."""
        for name, shm in self.shared_memories.items():
            try:
                # Remove from emergency cleanup registry
                _shm_registry.discard(shm.name)
                shm.close()
                shm.unlink()
                logger.debug(f"Released shared memory for {name}")
            except Exception as e:
                logger.warning(f"Error releasing shared memory for {name}: {e}")
        self.shared_memories.clear()
        self.slide_info.clear()



# NOTE: _gpu_worker_shm and MultiGPUTileProcessorSHM were removed (Feb 2026).
# They were deprecated dead code — use multigpu_worker.MultiGPUTileProcessor instead.
# Old code is preserved in archive/ if needed for reference.
