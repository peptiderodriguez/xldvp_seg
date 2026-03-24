"""
Strategy factory for creating detection strategies.

Centralizes strategy construction logic so that both `multigpu_worker.py` and
`run_segmentation.py` use the same code path, eliminating duplication risk.

Uses the StrategyRegistry for class lookup (strategies self-register via
@register_strategy decorators at import time). The parameter mapping from
flat strategy_params dict to strategy-specific constructor kwargs is inherently
per-strategy and stays here.

Usage:
    from segmentation.processing.strategy_factory import create_strategy

    strategy = create_strategy(
        cell_type='nmj',
        strategy_params={'intensity_percentile': 98.0},
        extract_deep_features=False,
        extract_sam2_embeddings=True,
        pixel_size_um=0.1725,
        has_classifier=False,
    )
"""

from typing import Any, Dict, Optional

from segmentation.detection.registry import StrategyRegistry
from segmentation.utils.logging import get_logger

import segmentation.detection.strategies  # noqa: F401 — triggers @register_strategy decorators

logger = get_logger(__name__)


def _build_kwargs_nmj(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    has_classifier: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for NMJStrategy."""
    return dict(
        intensity_percentile=strategy_params.get("intensity_percentile", 98.0),
        max_solidity=strategy_params.get("max_solidity", 0.85),
        min_skeleton_length=strategy_params.get("min_skeleton_length", 30),
        min_area_px=strategy_params.get("min_area", 150),
        min_area_um=strategy_params.get("min_area_um", 25.0),
        classifier_threshold=strategy_params.get("classifier_threshold", 0.5),
        use_classifier=has_classifier,
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
    )


def _build_kwargs_mk(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for MKStrategy."""
    return dict(
        min_area_um=strategy_params.get("mk_min_area", 200.0),
        max_area_um=strategy_params.get("mk_max_area", 2000.0),
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
        refine_masks=strategy_params.get("refine_masks", False),
    )


def _build_kwargs_cell(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for CellStrategy."""
    return dict(
        min_area_um=strategy_params.get("min_area_um", 50),
        max_area_um=strategy_params.get("max_area_um", 200),
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
        cellpose_input_channels=strategy_params.get("cellpose_input_channels"),
    )


def _build_kwargs_islet(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for IsletStrategy."""
    return dict(
        membrane_channel=strategy_params.get("membrane_channel", 1),
        nuclear_channel=strategy_params.get("nuclear_channel", 4),
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
        marker_signal_factor=strategy_params.get("marker_signal_factor", 2.0),
        gmm_prefilter_thresholds=strategy_params.get("gmm_prefilter_thresholds"),
    )


def _build_kwargs_tissue_pattern(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for TissuePatternStrategy."""
    return dict(
        detection_channels=strategy_params.get("detection_channels", [0, 3]),
        nuclear_channel=strategy_params.get("nuclear_channel", 4),
        min_area_um=strategy_params.get("min_area_um", 20),
        max_area_um=strategy_params.get("max_area_um", 300),
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
    )


def _build_kwargs_vessel(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for VesselStrategy."""
    return dict(
        min_diameter_um=strategy_params.get("min_vessel_diameter_um", 10),
        max_diameter_um=strategy_params.get("max_vessel_diameter_um", 1000),
        min_wall_thickness_um=strategy_params.get("min_wall_thickness_um", 2),
        max_aspect_ratio=strategy_params.get("max_aspect_ratio", 4.0),
        min_circularity=strategy_params.get("min_circularity", 0.3),
        min_ring_completeness=strategy_params.get("min_ring_completeness", 0.5),
        classify_vessel_types=strategy_params.get("classify_vessel_types", False),
        candidate_mode=strategy_params.get("candidate_mode", False),
        ring_only=strategy_params.get("ring_only", False),
        smooth_contours=strategy_params.get("smooth_contours", True),
        smooth_contours_factor=strategy_params.get("smooth_contours_factor", 3.0),
        parallel_detection=strategy_params.get("parallel_detection", False),
        parallel_workers=strategy_params.get("parallel_workers", 3),
        multi_marker=strategy_params.get("multi_marker", False),
        merge_iou_threshold=strategy_params.get("merge_iou_threshold", 0.5),
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
    )


def _build_kwargs_mesothelium(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    pixel_size_um: Optional[float] = None,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for MesotheliumStrategy."""
    meso_params = {k: v for k, v in strategy_params.items() if k != "pixel_size_um"}
    return dict(
        pixel_size_um=pixel_size_um,
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        **meso_params,
    )


def _build_kwargs_instanseg(
    strategy_params: Dict[str, Any],
    extract_deep_features: bool,
    extract_sam2_embeddings: bool,
    **_ignored,
) -> Dict[str, Any]:
    """Build constructor kwargs for InstanSegStrategy."""
    return dict(
        instanseg_model=strategy_params.get("instanseg_model", "fluorescence_nuclei_and_cells"),
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        resnet_batch_size=strategy_params.get("resnet_batch_size", 32),
    )


# Map cell_type -> kwargs builder function
_KWARGS_BUILDERS = {
    "nmj": _build_kwargs_nmj,
    "mk": _build_kwargs_mk,
    "cell": _build_kwargs_cell,
    "islet": _build_kwargs_islet,
    "tissue_pattern": _build_kwargs_tissue_pattern,
    "vessel": _build_kwargs_vessel,
    "mesothelium": _build_kwargs_mesothelium,
    "instanseg": _build_kwargs_instanseg,
}

# Validate that all registered strategies have kwargs builders
_registered = set(StrategyRegistry.list_strategies())
_missing = _registered - set(_KWARGS_BUILDERS.keys())
if _missing:
    logger.warning(
        "Strategies registered without kwargs builders: %s. "
        "create_strategy() will fail for these types.",
        ", ".join(sorted(_missing)),
    )


def create_strategy(
    cell_type: str,
    strategy_params: Dict[str, Any],
    extract_deep_features: bool = False,
    extract_sam2_embeddings: bool = True,
    pixel_size_um: float = None,
    has_classifier: bool = False,
) -> Any:
    """
    Create the appropriate detection strategy for a cell type.

    Uses StrategyRegistry for class lookup and per-strategy kwargs builders
    for parameter mapping from the flat strategy_params dict.

    Args:
        cell_type: One of 'nmj', 'mk', 'cell', 'vessel', 'mesothelium',
                   'islet', 'tissue_pattern'
        strategy_params: Cell-type specific parameters dict
        extract_deep_features: Whether to extract ResNet+DINOv2 features
        extract_sam2_embeddings: Whether to extract SAM2 embeddings
        pixel_size_um: Pixel size in microns (used by mesothelium strategy)
        has_classifier: Whether a classifier model is loaded (NMJ only)

    Returns:
        DetectionStrategy instance

    Raises:
        ValueError: If cell_type is not recognized
    """
    # Look up strategy class from registry
    try:
        strategy_class = StrategyRegistry.get_strategy_class(cell_type)
    except KeyError:
        available = ", ".join(sorted(StrategyRegistry.list_strategies()))
        raise ValueError(
            f"Unknown cell_type: '{cell_type}'. Supported types: {available}"
        )

    # Build strategy-specific kwargs
    kwargs_builder = _KWARGS_BUILDERS.get(cell_type)
    if kwargs_builder is None:
        raise ValueError(
            f"No parameter builder for cell_type '{cell_type}'. "
            f"Add a _build_kwargs_{cell_type}() function and register it in _KWARGS_BUILDERS."
        )

    kwargs = kwargs_builder(
        strategy_params=strategy_params,
        extract_deep_features=extract_deep_features,
        extract_sam2_embeddings=extract_sam2_embeddings,
        pixel_size_um=pixel_size_um,
        has_classifier=has_classifier,
    )

    return strategy_class(**kwargs)
