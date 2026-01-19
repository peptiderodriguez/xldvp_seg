"""
Centralized model management for the segmentation pipeline.

Provides:
- ModelManager: Singleton class for lazy-loading and caching ML models
- Checkpoint discovery: Unified checkpoint path management
- Resource cleanup: Proper GPU memory management

Usage:
    from segmentation.models import get_model_manager

    # Get singleton manager instance
    manager = get_model_manager(device="cuda")

    # Lazy-load models
    sam2_predictor, sam2_auto = manager.get_sam2()
    cellpose = manager.get_cellpose()
    resnet, transform = manager.get_resnet()

    # Cleanup when done
    manager.cleanup()

    # Or use as context manager
    with get_model_manager("cuda") as manager:
        sam2_predictor, sam2_auto = manager.get_sam2()
        # ... do work
    # Automatic cleanup on exit
"""

from .manager import (
    ModelManager,
    get_model_manager,
    find_checkpoint,
    CHECKPOINT_PATHS,
)

__all__ = [
    'ModelManager',
    'get_model_manager',
    'find_checkpoint',
    'CHECKPOINT_PATHS',
]
