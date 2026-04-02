"""
Centralized model management for the segmentation pipeline.

Provides:
- ModelManager: Singleton class for lazy-loading and caching ML models
- Checkpoint discovery: Unified checkpoint path management
- Resource cleanup: Proper GPU memory management

Usage:
    from segmentation.models import get_model_manager

    # Get singleton manager instance
    manager = get_model_manager()

    # Lazy-load models
    sam2_predictor, sam2_auto = manager.get_sam2()
    cellpose = manager.get_cellpose()
    resnet, transform = manager.get_resnet()

    # Cleanup when done
    manager.cleanup()

    # Or use as context manager
    with get_model_manager() as manager:
        sam2_predictor, sam2_auto = manager.get_sam2()
        # ... do work
    # Automatic cleanup on exit
"""

from .manager import (
    CHECKPOINT_PATHS,
    ModelManager,
    find_checkpoint,
    get_model_manager,
)
from .registry import (
    ModelMeta,
    ModelRegistry,
    get_model_info,
    list_models,
    register_model,
)

__all__ = [
    "ModelManager",
    "get_model_manager",
    "find_checkpoint",
    "CHECKPOINT_PATHS",
    "ModelRegistry",
    "ModelMeta",
    "list_models",
    "get_model_info",
    "register_model",
]
