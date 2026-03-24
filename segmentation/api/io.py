"""I/O and export functions."""

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def export_lmd(slide, crosses, output_dir=None, **kwargs):
    """Export for laser microdissection.

    Use run_lmd_export.py directly for now.
    """
    raise NotImplementedError(
        "Use: xlseg export-lmd --detections <det.json> --crosses <crosses.json>"
    )


def to_spatialdata(slide, output_path=None, **kwargs):
    """Export to SpatialData zarr for scverse.

    Use scripts/convert_to_spatialdata.py directly for now.
    """
    raise NotImplementedError(
        "Use scripts/convert_to_spatialdata.py --detections <det.json> --output <path.zarr>"
    )
