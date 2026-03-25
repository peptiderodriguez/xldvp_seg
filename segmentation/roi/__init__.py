"""ROI-restricted cell detection.

Provides utilities for finding, filtering, and processing Regions of Interest
(ROIs) in whole-slide images, so that expensive per-cell detection runs only
on biologically relevant areas.

Submodules:
    common              Shared bbox extraction, spatial numbering, tile/detection filters
    marker_threshold    Find ROIs via summed marker signal + Otsu
    circular_objects    Find ROIs that are roughly circular (e.g. islets)
    from_file           Load ROIs from polygon JSON or label masks
"""

from .circular_objects import find_circular_regions
from .common import (
    detect_in_rois,
    extract_region_bboxes,
    filter_detections_by_roi_mask,
    filter_tiles_by_rois,
    number_rois_spatial,
)
from .from_file import load_rois_from_mask, load_rois_from_polygons
from .marker_threshold import find_regions_by_marker_signal

__all__ = [
    # common
    "extract_region_bboxes",
    "number_rois_spatial",
    "filter_tiles_by_rois",
    "filter_detections_by_roi_mask",
    "detect_in_rois",
    # marker_threshold
    "find_regions_by_marker_signal",
    # circular_objects
    "find_circular_regions",
    # from_file
    "load_rois_from_polygons",
    "load_rois_from_mask",
]
