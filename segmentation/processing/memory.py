"""
Memory validation and management utilities.

Provides functions to check system resources before processing and
automatically adjust worker counts to prevent OOM crashes.

Shared across all detection pipelines.

Usage:
    from segmentation.processing.memory import validate_system_resources, get_safe_worker_count

    # Check resources before starting
    result = validate_system_resources(num_workers=4, tile_size=3000)
    if result['should_abort']:
        sys.exit(1)

    # Auto-adjust worker count
    safe_workers = get_safe_worker_count(requested_workers=8, tile_size=3000)
"""

from typing import Dict, Any
import psutil

from segmentation.utils.logging import get_logger
from segmentation.utils.config import get_memory_threshold

logger = get_logger(__name__)


def validate_system_resources(num_workers: int, tile_size: int) -> Dict[str, Any]:
    """
    Check available RAM and GPU memory before starting processing.

    Returns dict with warnings and recommended settings.

    Each worker uses approximately:
    - 10-15 GB RAM (tile data, SAM2 masks, features)
    - 4-6 GB GPU memory (SAM2 + ResNet models)

    Larger tiles (4096) use ~2x memory of smaller tiles (3000).

    Args:
        num_workers: Number of workers requested
        tile_size: Tile size being used

    Returns:
        Dict with keys:
        - warnings: List of warning messages
        - recommended_workers: Recommended number of workers
        - recommended_tile_size: Recommended tile size
        - should_abort: True if system resources critically low
    """
    result = {
        'warnings': [],
        'recommended_workers': num_workers,
        'recommended_tile_size': tile_size,
        'should_abort': False
    }

    # Get thresholds from config
    min_ram_gb = get_memory_threshold('min_ram_gb')
    min_gpu_gb = get_memory_threshold('min_gpu_gb')
    mem_per_worker_large = get_memory_threshold('mem_per_worker_large_tile')
    mem_per_worker_small = get_memory_threshold('mem_per_worker_small_tile')

    # Check RAM
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    # Estimate memory per worker based on tile size
    # 4096 tiles: ~15GB per worker, 3000 tiles: ~10GB per worker
    mem_per_worker_gb = mem_per_worker_large if tile_size >= 4096 else mem_per_worker_small
    max_safe_workers = max(1, int(available_gb / mem_per_worker_gb))

    logger.info(f"System RAM: {available_gb:.1f} GB available / {total_gb:.1f} GB total")
    logger.info(f"Estimated memory per worker: ~{mem_per_worker_gb:.0f} GB (tile_size={tile_size})")

    if num_workers > max_safe_workers:
        result['warnings'].append(
            f"WARNING: {num_workers} workers requested but only ~{available_gb:.0f} GB RAM available. "
            f"Risk of OOM crash! Recommend max {max_safe_workers} workers."
        )
        result['recommended_workers'] = max_safe_workers

    # Check GPU memory (all available GPUs, not just device 0)
    try:
        import torch
    except ImportError:
        torch = None
        logger.info("PyTorch not available, skipping GPU validation")
    if torch is not None and torch.cuda.is_available():
        try:
            num_gpus_check = min(torch.cuda.device_count(), num_workers if num_workers else 1)
            min_gpu_available = float('inf')
            for gpu_id in range(num_gpus_check):
                gpu_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                gpu_available = gpu_total - gpu_reserved
                logger.info(f"GPU {gpu_id} memory: {gpu_available:.1f} GB available / {gpu_total:.1f} GB total")
                min_gpu_available = min(min_gpu_available, gpu_available)

            # SAM2 + ResNet need ~6-8 GB minimum per GPU
            if min_gpu_available < min_gpu_gb:
                result['warnings'].append(
                    f"WARNING: GPU with only {min_gpu_available:.1f} GB memory available. "
                    f"SAM2 + ResNet need ~{min_gpu_gb:.0f}-8 GB. May cause CUDA OOM errors."
                )

            # Tip for monitoring
            logger.info("TIP: Monitor GPU in another terminal: nvidia-smi -l 1")
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
    else:
        result['warnings'].append("WARNING: No CUDA GPU detected. Processing will be very slow.")

    # Critical memory threshold - abort if less than min_ram_gb available
    if available_gb < min_ram_gb:
        result['warnings'].append(
            f"CRITICAL: Only {available_gb:.1f} GB RAM available. "
            f"This is likely to crash your system. Aborting."
        )
        result['should_abort'] = True

    # Log warnings
    for warning in result['warnings']:
        logger.warning(warning)

    return result


def get_safe_worker_count(
    requested_workers: int,
    tile_size: int,
    auto_adjust: bool = True
) -> int:
    """
    Return a safe number of workers based on available system resources.

    Args:
        requested_workers: Number of workers requested by user
        tile_size: Tile size being used
        auto_adjust: If True, automatically reduce workers if needed

    Returns:
        Safe number of workers to use
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)

    # Get thresholds from config
    mem_per_worker_large = get_memory_threshold('mem_per_worker_large_tile')
    mem_per_worker_small = get_memory_threshold('mem_per_worker_small_tile')

    # Memory per worker estimate
    mem_per_worker_gb = mem_per_worker_large if tile_size >= 4096 else mem_per_worker_small
    max_safe_workers = max(1, int(available_gb / mem_per_worker_gb))

    if auto_adjust and requested_workers > max_safe_workers:
        logger.warning(
            f"Auto-adjusting workers from {requested_workers} to {max_safe_workers} "
            f"based on available RAM ({available_gb:.1f} GB)"
        )
        return max_safe_workers

    return requested_workers


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dict with RAM and GPU memory info
    """
    mem = psutil.virtual_memory()
    result = {
        'ram_available_gb': mem.available / (1024**3),
        'ram_total_gb': mem.total / (1024**3),
        'ram_used_percent': mem.percent,
    }

    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        _has_cuda = False
    if _has_cuda:
        try:
            num_gpus = torch.cuda.device_count()
            # Report per-GPU stats and aggregate totals
            total_gpu_total = 0
            total_gpu_allocated = 0
            total_gpu_reserved = 0
            for gpu_id in range(num_gpus):
                gpu_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                gpu_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
                total_gpu_total += gpu_total
                total_gpu_allocated += gpu_allocated
                total_gpu_reserved += gpu_reserved
                if num_gpus > 1:
                    result[f'gpu{gpu_id}_total_gb'] = gpu_total
                    result[f'gpu{gpu_id}_available_gb'] = gpu_total - gpu_reserved

            result['gpu_total_gb'] = total_gpu_total
            result['gpu_allocated_gb'] = total_gpu_allocated
            result['gpu_reserved_gb'] = total_gpu_reserved
            result['gpu_available_gb'] = total_gpu_total - total_gpu_reserved
            result['gpu_count'] = num_gpus
        except Exception:
            pass

    return result


def log_memory_status(prefix: str = "") -> None:
    """
    Log current memory usage.

    Args:
        prefix: Optional prefix for log message
    """
    usage = get_memory_usage()
    msg_parts = [
        f"RAM: {usage['ram_available_gb']:.1f}/{usage['ram_total_gb']:.1f} GB available"
    ]

    if 'gpu_available_gb' in usage:
        msg_parts.append(
            f"GPU: {usage['gpu_available_gb']:.1f}/{usage['gpu_total_gb']:.1f} GB available"
        )

    msg = " | ".join(msg_parts)
    if prefix:
        msg = f"{prefix}: {msg}"

    logger.info(msg)


__all__ = [
    'validate_system_resources',
    'get_safe_worker_count',
    'get_memory_usage',
    'log_memory_status',
]
