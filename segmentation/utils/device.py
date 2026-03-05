"""Device detection and selection for CUDA, MPS (Apple Silicon), and CPU backends.

Provides a single entry point for choosing the best available torch device,
so the rest of the pipeline doesn't need to scatter torch.cuda checks everywhere.

Usage:
    from segmentation.utils.device import get_default_device, get_device_count, empty_cache

    device = get_default_device()          # 'cuda', 'mps', or 'cpu'
    n = get_device_count()                 # number of accelerator devices
    empty_cache()                          # backend-appropriate cache cleanup
"""

import os

import torch

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def get_default_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_count() -> int:
    """Return the number of accelerator devices (CUDA GPUs, or 1 for MPS, or 0)."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 1  # Apple Silicon has exactly one MPS device
    return 0


def is_gpu_available() -> bool:
    """Return True if any GPU backend (CUDA or MPS) is available."""
    return get_default_device() != "cpu"


def empty_cache() -> None:
    """Free cached memory on the active accelerator backend."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Synchronize before clearing to ensure queued ops finish
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def set_device_for_worker(gpu_id: int) -> torch.device:
    """Configure the process for a specific GPU worker and return the device.

    On CUDA: sets CUDA_VISIBLE_DEVICES so the worker sees only one GPU,
             then returns torch.device('cuda:0').
    On MPS:  ignores gpu_id (only one device), returns torch.device('mps').
    On CPU:  returns torch.device('cpu').

    Must be called in a spawned subprocess BEFORE any CUDA tensor operations.
    Not thread-safe (mutates os.environ).
    """
    # Set CUDA_VISIBLE_DEVICES BEFORE checking availability to ensure
    # the CUDA runtime (if initialized later) sees only this GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device('cuda:0')
    # On non-CUDA systems the env var is harmless
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if gpu_id > 0:
            logger.warning("MPS has only one device; gpu_id=%d ignored", gpu_id)
        return torch.device('mps')
    return torch.device('cpu')


def device_supports_gpu(device) -> bool:
    """Return True if the device is a GPU (CUDA or MPS).

    Accepts torch.device objects or device strings.
    """
    if isinstance(device, str):
        device = torch.device(device)
    return device.type in ('cuda', 'mps')


__all__ = [
    'get_default_device',
    'get_device_count',
    'is_gpu_available',
    'empty_cache',
    'set_device_for_worker',
    'device_supports_gpu',
]
