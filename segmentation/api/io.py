"""I/O and export functions."""

from pathlib import Path
from typing import Optional
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def export_lmd(slide, crosses, output_dir=None, min_score=0.5,
               generate_controls=True, erosion_um=0.0, **kwargs):
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
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from segmentation.utils.json_utils import fast_json_load

    if output_dir is None:
        output_dir = str(slide.output_dir / "lmd") if slide.output_dir else "lmd_export"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter by score
    detections = [d for d in slide.detections
                  if d.get("rf_prediction", 0) >= min_score]
    logger.info("LMD export: %d detections above score %.2f", len(detections), min_score)

    if not detections:
        logger.warning("No detections above score threshold")
        return output_dir

    # Load crosses
    crosses_data = fast_json_load(str(crosses))

    # This chains several functions from run_lmd_export.py
    # For the full pipeline (contour extraction, clustering, well assignment, XML),
    # use run_lmd_export.py directly
    raise NotImplementedError(
        f"Full LMD export pipeline requires contour extraction from HDF5 masks.\n"
        f"Use: xlseg export-lmd --detections {slide.detections_path} "
        f"--crosses {crosses} --output-dir {output_dir} --min-score {min_score}"
    )


def to_spatialdata(slide, output_path=None, cell_type=None, **kwargs):
    """Export detections to SpatialData zarr for scverse ecosystem.

    Args:
        slide: SlideAnalysis object.
        output_path: Path for .zarr output.
        cell_type: Cell type string (auto-detected from slide if None).

    Returns:
        Path to the zarr store.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.convert_to_spatialdata import build_anndata

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
