"""Plotting functions."""

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def umap(slide, color_by=None, **kwargs):
    """Display UMAP plot.

    Use scripts/cluster_by_features.py --methods both directly for now.
    """
    raise NotImplementedError(
        "Use scripts/cluster_by_features.py --detections <det.json> --methods both"
    )


def spatial(slide, **kwargs):
    """Display spatial viewer.

    Use scripts/generate_multi_slide_spatial_viewer.py directly for now.
    """
    raise NotImplementedError(
        "Use scripts/generate_multi_slide_spatial_viewer.py --input-dir <output>"
    )
