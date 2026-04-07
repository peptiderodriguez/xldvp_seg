"""I/O and export functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from xldvp_seg.utils.logging import get_logger

if TYPE_CHECKING:
    from xldvp_seg.core.slide_analysis import SlideAnalysis

logger = get_logger(__name__)


def read_proteomics(
    path: str | Path,
    search_engine: str | None = None,
    well_column: str = "well_id",
    **kwargs: Any,
):
    """Read proteomics data from CSV or search engine report.

    If *search_engine* is provided, uses `dvp-io <https://github.com/MannLabs/dvp-io>`_
    to parse the report.  Otherwise reads plain CSV (rows=wells, columns=proteins).

    Supported engines: alphadia, alphapept, diann, directlfq, fragpipe,
    maxquant, mztab, spectronaut.

    Args:
        path: Path to CSV or search engine report file.
        search_engine: Engine name (e.g., ``'diann'``, ``'maxquant'``).
            ``None`` for plain CSV.
        well_column: Column identifying wells/samples.
        **kwargs: Forwarded to ``dvpio.read.omics.read_pg_table()``.

    Returns:
        DataFrame with wells as index, proteins as columns.
    """
    import pandas as pd

    if search_engine is None:
        return pd.read_csv(str(path), index_col=well_column)

    from xldvp_seg.analysis.omic_linker import OmicLinker

    linker = OmicLinker()
    linker.load_proteomics_report(path, search_engine, well_column=well_column, **kwargs)
    return linker._proteomics


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
