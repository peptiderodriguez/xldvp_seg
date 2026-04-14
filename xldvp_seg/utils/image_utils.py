"""Image utility functions shared across the pipeline.

Canonical implementations of common image operations. New code should import
from here rather than duplicating.
"""

from __future__ import annotations

import numpy as np


def percentile_normalize(arr: np.ndarray, p_low: float = 1, p_high: float = 99.5) -> np.ndarray:
    """Percentile-normalize any numeric array to uint8 [0, 255].

    Uses percentiles on nonzero pixels for proper dynamic range.
    Zero-valued pixels (CZI padding, background) stay zero.

    Args:
        arr: 2D numeric array (any dtype).
        p_low: Low percentile for clipping (default: 1).
        p_high: High percentile for clipping (default: 99.5).

    Returns:
        uint8 array with the same shape, normalized to [0, 255].
    """
    if arr.dtype == np.uint8:
        return arr
    nonzero = arr[arr > 0]
    if len(nonzero) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(np.percentile(nonzero, p_low))
    hi = float(np.percentile(nonzero, p_high))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    result = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    return result.astype(np.uint8)
