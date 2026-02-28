"""
Strategy factory for creating detection strategies.

Centralizes strategy construction logic so that both `multigpu_worker.py` and
`run_segmentation.py` use the same code path, eliminating duplication risk.

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

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


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
    if cell_type == 'nmj':
        from segmentation.detection.strategies.nmj import NMJStrategy
        return NMJStrategy(
            intensity_percentile=strategy_params.get('intensity_percentile', 98.0),
            max_solidity=strategy_params.get('max_solidity', 0.85),
            min_skeleton_length=strategy_params.get('min_skeleton_length', 30),
            min_area_px=strategy_params.get('min_area', 150),
            min_area_um=strategy_params.get('min_area_um', 25.0),
            classifier_threshold=strategy_params.get('classifier_threshold', 0.5),
            use_classifier=has_classifier,
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=strategy_params.get('resnet_batch_size', 32),
        )
    elif cell_type == 'mk':
        from segmentation.detection.strategies.mk import MKStrategy
        return MKStrategy(
            min_area_um=strategy_params.get('mk_min_area', 200.0),
            max_area_um=strategy_params.get('mk_max_area', 2000.0),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=strategy_params.get('resnet_batch_size', 32),
        )
    elif cell_type == 'cell':
        from segmentation.detection.strategies.cell import CellStrategy
        return CellStrategy(
            min_area_um=strategy_params.get('min_area_um', 50),
            max_area_um=strategy_params.get('max_area_um', 200),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=strategy_params.get('resnet_batch_size', 32),
            cellpose_input_channels=strategy_params.get('cellpose_input_channels'),
        )
    elif cell_type == 'islet':
        from segmentation.detection.strategies.islet import IsletStrategy
        return IsletStrategy(
            membrane_channel=strategy_params.get('membrane_channel', 1),
            nuclear_channel=strategy_params.get('nuclear_channel', 4),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=strategy_params.get('resnet_batch_size', 32),
            marker_signal_factor=strategy_params.get('marker_signal_factor', 2.0),
            gmm_prefilter_thresholds=strategy_params.get('gmm_prefilter_thresholds'),
        )
    elif cell_type == 'tissue_pattern':
        from segmentation.detection.strategies.tissue_pattern import TissuePatternStrategy
        return TissuePatternStrategy(
            detection_channels=strategy_params.get('detection_channels', [0, 3]),
            nuclear_channel=strategy_params.get('nuclear_channel', 4),
            min_area_um=strategy_params.get('min_area_um', 20),
            max_area_um=strategy_params.get('max_area_um', 300),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=strategy_params.get('resnet_batch_size', 32),
        )
    elif cell_type == 'vessel':
        from segmentation.detection.strategies.vessel import VesselStrategy
        return VesselStrategy(
            min_diameter_um=strategy_params.get('min_vessel_diameter_um', 10),
            max_diameter_um=strategy_params.get('max_vessel_diameter_um', 1000),
            min_wall_thickness_um=strategy_params.get('min_wall_thickness_um', 2),
            max_aspect_ratio=strategy_params.get('max_aspect_ratio', 4.0),
            min_circularity=strategy_params.get('min_circularity', 0.3),
            min_ring_completeness=strategy_params.get('min_ring_completeness', 0.5),
            classify_vessel_types=strategy_params.get('classify_vessel_types', False),
            candidate_mode=strategy_params.get('candidate_mode', False),
            lumen_first=strategy_params.get('lumen_first', False),
            ring_only=strategy_params.get('ring_only', False),
            smooth_contours=strategy_params.get('smooth_contours', True),
            smooth_contours_factor=strategy_params.get('smooth_contours_factor', 3.0),
            parallel_detection=strategy_params.get('parallel_detection', False),
            parallel_workers=strategy_params.get('parallel_workers', 3),
            multi_marker=strategy_params.get('multi_marker', False),
            merge_iou_threshold=strategy_params.get('merge_iou_threshold', 0.5),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=strategy_params.get('resnet_batch_size', 32),
        )
    elif cell_type == 'mesothelium':
        from segmentation.detection.strategies.mesothelium import MesotheliumStrategy
        meso_params = {k: v for k, v in strategy_params.items() if k != 'pixel_size_um'}
        return MesotheliumStrategy(
            pixel_size_um=pixel_size_um,
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            **meso_params,
        )
    else:
        raise ValueError(
            f"Unknown cell_type: '{cell_type}'. "
            f"Supported types: nmj, mk, cell, vessel, mesothelium, islet, tissue_pattern"
        )
