#!/usr/bin/env python3
"""
Contour post-processing for LMD export.

This module re-exports from segmentation.lmd.contour_processing.
Import from there directly for new code:

    from segmentation.lmd.contour_processing import process_contour
"""

# Re-export everything from the canonical location
from segmentation.lmd.contour_processing import (  # noqa: F401
    rdp_simplify,
    validate_polygon,
    dilate_contour,
    process_contour,
    process_contours_batch,
    process_detection_contours,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_DILATION_UM,
    DEFAULT_RDP_EPSILON,
)


if __name__ == '__main__':
    # Quick test
    print("Contour Processing Module")
    print("=" * 40)

    # Create a simple square contour for testing
    test_contour_px = [
        [100, 100],
        [200, 100],
        [200, 200],
        [100, 200],
    ]

    result, stats = process_contour(test_contour_px, return_stats=True)

    print(f"Input: {len(test_contour_px)} points")
    print(f"Output: {stats['points_after']} points")
    print(f"Area: {stats['area_before_um2']:.2f} -> {stats['area_after_um2']:.2f} um2")
    print(f"Valid: {stats['valid']}")
