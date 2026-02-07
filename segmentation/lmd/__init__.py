"""
LMD (Laser Microdissection) export utilities.

Provides clustering, contour processing, and export tools for preparing
detection results for Leica LMD laser microdissection systems.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Usage:
    from segmentation.lmd.clustering import two_stage_clustering
    from segmentation.lmd.contour_processing import process_contour
"""

# Use explicit imports from submodules to avoid circular deps and
# unnecessary heavy imports (cv2, shapely) when only clustering is needed.

__all__ = [
    "clustering",
    "contour_processing",
]
