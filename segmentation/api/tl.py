"""Analysis tools operating on SlideAnalysis objects.

V1: Structure in place, complex wrappers raise NotImplementedError with
helpful messages pointing to the underlying scripts.
"""

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def classify(slide, annotations, feature_set="morph", output_dir=None, **kwargs):
    """Train RF classifier and apply to detections.

    Use train_classifier.py + apply_classifier.py directly for now.
    """
    logger.info("classify: %d detections, feature_set=%s", slide.n_detections, feature_set)
    raise NotImplementedError(
        "Full classify wrapper not yet implemented. Use:\n"
        "  xlseg classify --detections <det.json> --annotations <ann.json>\n"
        "  xlseg score --detections <det.json> --classifier <clf.pkl>"
    )


def markers(slide, marker_channel=None, marker_name=None, marker_wavelength=None,
            method="snr", czi_path=None, **kwargs):
    """Classify markers pos/neg per channel.

    Use scripts/classify_markers.py directly for now.
    """
    logger.info("markers: %d detections, method=%s", slide.n_detections, method)
    raise NotImplementedError(
        "Full markers wrapper not yet implemented. Use:\n"
        "  xlseg markers --detections <det.json> --marker-wavelength 647,555 "
        "--marker-name SMA,CD31 --czi-path <czi>"
    )


def cluster(slide, feature_groups="morph", methods="both", resolution=0.1,
            output_dir=None, **kwargs):
    """Feature clustering with UMAP/t-SNE + Leiden.

    Use scripts/cluster_by_features.py directly for now.
    """
    logger.info("cluster: %d detections, methods=%s", slide.n_detections, methods)
    raise NotImplementedError(
        "Full cluster wrapper not yet implemented. Use:\n"
        "  python scripts/cluster_by_features.py --detections <det.json> "
        "--methods both --clustering leiden"
    )


def spatial(slide, output_dir=None, **kwargs):
    """Spatial network analysis.

    Use scripts/spatial_cell_analysis.py directly for now.
    """
    raise NotImplementedError(
        "Use scripts/spatial_cell_analysis.py --detections <det.json>"
    )


def nuclei(slide, czi_path=None, nuclear_channel=None, **kwargs):
    """Count nuclei per cell.

    Use scripts/count_nuclei_per_cell.py directly for now.
    """
    raise NotImplementedError(
        "Use scripts/count_nuclei_per_cell.py --detections <det.json> "
        "--czi-path <czi> --channel-spec 'nuc=Hoechst'"
    )
