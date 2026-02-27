"""
Slide-level per-channel flat-field correction for fluorescence microscopy.

Estimates a smooth illumination profile from the full-slide data using block
medians, then corrects each pixel so that uniform structures have uniform
intensity across the entire mosaic.

Algorithm:
    1. Divide each channel into coarse blocks and compute median of non-zero pixels
    2. Fill missing blocks (all-zero / CZI padding) via nearest-neighbor interpolation
    3. Gaussian-smooth the coarse grid to get a continuous illumination field
    4. Apply correction: corrected = raw * (slide_mean / local_illumination)

Zero pixels (CZI padding) are preserved as zero throughout.

Usage:
    from segmentation.preprocessing.flat_field import (
        estimate_illumination_profile, IlluminationProfile
    )

    profile = estimate_illumination_profile(all_channel_data, block_size=512)
    for ch, data in all_channel_data.items():
        profile.correct_channel_inplace(data, ch)
"""

import gc
import logging
from typing import Dict

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

logger = logging.getLogger(__name__)


class IlluminationProfile:
    """Stores a smoothed coarse illumination grid per channel and applies correction.

    Attributes:
        grids: Dict mapping channel index to 2D float64 array of smoothed
               block medians (one value per block_size x block_size region).
        block_size: Side length of each block in pixels.
        slide_means: Dict mapping channel index to the global mean intensity
                     (mean of all valid block medians before NaN-fill).
    """

    def __init__(
        self,
        grids: Dict[int, np.ndarray],
        block_size: int,
        slide_means: Dict[int, float],
    ):
        self.grids = grids
        self.block_size = block_size
        self.slide_means = slide_means

    def correct_channel_inplace(self, data: np.ndarray, channel: int) -> None:
        """Apply flat-field correction to a full-slide channel array in-place.

        For each pixel: corrected = raw * (slide_mean / local_illumination),
        where local_illumination is bilinearly interpolated from the coarse grid
        via ``scipy.ndimage.zoom``.

        Processing is done in horizontal strips of ``block_size`` rows to keep
        temporary memory small. Zero pixels (CZI padding) are preserved.

        Args:
            data: 2D uint16 array (H, W) — modified in-place.
            channel: Channel index into ``self.grids`` / ``self.slide_means``.
        """
        grid = self.grids[channel]
        slide_mean = self.slide_means[channel]
        bs = self.block_size
        H, W = data.shape

        # Floor to avoid extreme correction in very dark regions
        illum_floor = slide_mean * 0.1

        logger.info(
            "Correcting channel %d: grid %s, slide_mean=%.1f, illum_floor=%.1f",
            channel, grid.shape, slide_mean, illum_floor,
        )

        # Upscale coarse grid to full slide resolution via bilinear interpolation.
        # zoom() maps grid centers to pixel centers automatically, so no manual
        # coordinate offset is needed. Memory: one float32 array of (H, W).
        zoom_factors = (H / grid.shape[0], W / grid.shape[1])
        full_illum = zoom(grid, zoom_factors, order=1, mode="nearest").astype(np.float32)

        # Clamp illumination to floor
        np.maximum(full_illum, illum_floor, out=full_illum)

        # Pre-compute correction field: slide_mean / local_illumination
        np.divide(slide_mean, full_illum, out=full_illum)  # reuse buffer
        correction_field = full_illum  # renamed for clarity

        correction_min = float(correction_field.min())
        correction_max = float(correction_field.max())

        # Apply correction in strips to avoid a second full-size temporary
        for row_start in range(0, H, bs):
            row_end = min(row_start + bs, H)
            strip = data[row_start:row_end]

            # Remember zeros so we can restore them
            zero_mask = strip == 0

            # Apply: float32 intermediate to save memory vs float64
            corrected = strip.astype(np.float32) * correction_field[row_start:row_end]
            np.clip(corrected, 0, 65535, out=corrected)

            # Write back and restore zero padding
            result = corrected.astype(np.uint16)
            result[zero_mask] = 0
            data[row_start:row_end] = result

        del correction_field, full_illum
        gc.collect()

        logger.info(
            "Channel %d correction range: [%.3f, %.3f]",
            channel, correction_min, correction_max,
        )


def estimate_illumination_profile(
    all_channel_data: Dict[int, np.ndarray],
    block_size: int = 512,
    smooth_sigma: float = 3.0,
) -> IlluminationProfile:
    """Estimate a smooth illumination profile from full-slide channel data.

    Args:
        all_channel_data: Dict mapping channel index to 2D uint16 arrays
                          of shape (H, W), one per channel.
        block_size: Side length of each averaging block in pixels.
        smooth_sigma: Gaussian sigma (in grid units, i.e. blocks) applied to
                      the coarse median grid before correction.

    Returns:
        An :class:`IlluminationProfile` ready for ``correct_channel_inplace``.
    """
    grids: Dict[int, np.ndarray] = {}
    slide_means: Dict[int, float] = {}

    for ch, data in all_channel_data.items():
        H, W = data.shape
        n_rows = int(np.ceil(H / block_size))
        n_cols = int(np.ceil(W / block_size))
        coarse = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

        for r in range(n_rows):
            y0 = r * block_size
            y1 = min(y0 + block_size, H)
            for c in range(n_cols):
                x0 = c * block_size
                x1 = min(x0 + block_size, W)
                block = data[y0:y1, x0:x1]
                nonzero = block[block > 0]
                if nonzero.size > 0:
                    coarse[r, c] = float(np.median(nonzero))

        # Slide mean from valid (non-NaN) blocks only
        valid_mask = ~np.isnan(coarse)
        n_valid = int(valid_mask.sum())
        n_total = coarse.size
        ch_mean = float(np.nanmean(coarse)) if n_valid > 0 else 1.0

        logger.info(
            "Channel %d: coarse grid %dx%d, %d/%d valid blocks, slide_mean=%.1f",
            ch, n_rows, n_cols, n_valid, n_total, ch_mean,
        )

        # Fill NaN blocks via nearest-neighbor interpolation
        if n_valid < n_total and n_valid > 0:
            nan_mask = ~valid_mask
            _, nearest_indices = distance_transform_edt(
                nan_mask, return_distances=True, return_indices=True
            )
            coarse[nan_mask] = coarse[
                nearest_indices[0][nan_mask], nearest_indices[1][nan_mask]
            ]
        elif n_valid == 0:
            # Degenerate: no signal at all — fill with 1.0 (no correction)
            coarse[:] = 1.0

        # Gaussian smooth
        smoothed = gaussian_filter(coarse, sigma=smooth_sigma)

        grids[ch] = smoothed
        slide_means[ch] = ch_mean

    return IlluminationProfile(grids, block_size, slide_means)
