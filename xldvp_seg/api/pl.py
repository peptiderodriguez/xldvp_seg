"""Plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.utils.logging import get_logger

if TYPE_CHECKING:
    from xldvp_seg.core.slide_analysis import SlideAnalysis

logger = get_logger(__name__)


def umap(
    slide: SlideAnalysis,
    color_by: str | None = None,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> SlideAnalysis:
    """Generate UMAP embedding visualization for detections.

    Delegates to :func:`xldvp_seg.analysis.cluster_features.run_clustering`
    with ``methods='umap'``.

    Args:
        slide: SlideAnalysis object with detections.
        color_by: Not yet used (reserved for future coloring options).
        output_dir: Output directory for UMAP plots (required).
        **kwargs: Forwarded to ``run_clustering`` (e.g., ``n_neighbors``,
            ``min_dist``, ``feature_groups``).

    Returns:
        slide (unchanged -- UMAP output is written to ``output_dir``).
    """
    from xldvp_seg.analysis.cluster_features import run_clustering

    if slide.detections_path is None:
        raise ConfigError("SlideAnalysis has no detections_path. Save detections first.")

    if output_dir is None:
        raise ConfigError(
            "output_dir is required — UMAP plots are written to disk. "
            "Pass a directory path to save the output."
        )
    _out = str(output_dir)
    Path(_out).mkdir(parents=True, exist_ok=True)

    run_clustering(
        detections=str(slide.detections_path),
        output_dir=_out,
        methods="umap",
        **kwargs,
    )
    logger.info("UMAP output written to %s", _out)
    return slide
