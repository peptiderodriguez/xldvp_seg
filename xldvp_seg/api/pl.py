"""Plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from xldvp_seg.utils.logging import get_logger

if TYPE_CHECKING:
    from xldvp_seg.core.slide_analysis import SlideAnalysis

logger = get_logger(__name__)


def umap(slide: SlideAnalysis, color_by: str | None = None, **kwargs: Any) -> NoReturn:
    """Display UMAP plot.

    Use scripts/cluster_by_features.py --methods both directly for now.
    """
    raise NotImplementedError(
        "Use scripts/cluster_by_features.py --detections <det.json> --methods both"
    )


def spatial(slide: SlideAnalysis, **kwargs: Any) -> NoReturn:
    """Display spatial viewer.

    Use scripts/generate_multi_slide_spatial_viewer.py directly for now.
    """
    raise NotImplementedError(
        "Use scripts/generate_multi_slide_spatial_viewer.py --input-dir <output>"
    )
