"""I/O and export functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from xldvp_seg.utils.logging import get_logger

if TYPE_CHECKING:
    from xldvp_seg.core.slide_analysis import SlideAnalysis

logger = get_logger(__name__)


def export_lmd(
    slide: SlideAnalysis,
    crosses: str | Path,
    output_dir: str | Path | None = None,
    min_score: float = 0.5,
    generate_controls: bool = True,
    erosion_um: float = 0.0,
    **kwargs: Any,
) -> Path:
    """Export detections for laser microdissection.

    Filters by score, extracts contours, assigns wells, generates XML.

    Args:
        slide: SlideAnalysis object.
        crosses: Path to reference crosses JSON (from napari_place_crosses.py).
        output_dir: Output directory for XML files.
        min_score: Minimum rf_prediction to include (default: 0.5).
        generate_controls: Generate spatial control wells (default: True).
        erosion_um: Contour erosion in um (default: 0.0).

    Returns:
        Path to output directory with XML files.
    """
    raise NotImplementedError(
        f"Full LMD export pipeline requires contour extraction from HDF5 masks.\n"
        f"Use: xlseg export-lmd --detections {slide.detections_path} "
        f"--crosses {crosses} --output-dir {output_dir} --min-score {min_score}"
    )


def to_spatialdata(
    slide: SlideAnalysis,
    output_path: str | Path | None = None,
    cell_type: str | None = None,
    **kwargs: Any,
) -> Path | None:
    """Export detections to SpatialData zarr for scverse ecosystem.

    Args:
        slide: SlideAnalysis object.
        output_path: Path for .zarr output.
        cell_type: Cell type string (auto-detected from slide if None).

    Returns:
        Path to the zarr store.
    """
    from xldvp_seg.io.spatialdata_export import build_anndata

    ct = cell_type or slide.cell_type or "cell"
    detections = slide.detections

    if not detections:
        logger.warning("No detections to export")
        return None

    # Build AnnData
    adata = build_anndata(detections, ct)
    logger.info("Built AnnData: %s", adata.shape)

    # Save as zarr via spatialdata
    if output_path is None:
        if slide.output_dir:
            output_path = slide.output_dir / f"{ct}_spatialdata.zarr"
        else:
            output_path = Path(f"{ct}_spatialdata.zarr")

    output_path = Path(output_path)

    try:
        import spatialdata as sd
        from spatialdata.models import TableModel

        table = TableModel.parse(adata)
        sdata = sd.SpatialData(tables={"table": table})
        sdata.write(str(output_path), overwrite=True)
        logger.info("SpatialData written to %s", output_path)
    except ImportError:
        # Fallback: save as h5ad
        h5ad_path = output_path.with_suffix(".h5ad")
        adata.write_h5ad(str(h5ad_path))
        logger.info("spatialdata not available, saved as H5AD: %s", h5ad_path)
        return h5ad_path

    return output_path
