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
    from xldvp_seg.preprocessing.flat_field import (
        estimate_illumination_profile, IlluminationProfile
    )

    profile = estimate_illumination_profile(all_channel_data, block_size=512)
    for ch, data in all_channel_data.items():
        profile.correct_channel_inplace(data, ch)
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# Bump when the estimate_illumination_profile algorithm changes in a way that
# invalidates cached profiles from earlier runs.
ALGORITHM_VERSION = "1.0"


class IlluminationProfile:
    """Stores a smoothed coarse illumination grid per channel and applies correction.

    Attributes:
        grids: Dict mapping channel index to 2D float32 array of smoothed
               block medians (one value per block_size x block_size region).
        block_size: Side length of each block in pixels.
        slide_means: Dict mapping channel index to the global mean intensity
                     (mean of all valid block medians before NaN-fill).
    """

    def __init__(
        self,
        grids: dict[int, np.ndarray],
        block_size: int,
        slide_means: dict[int, float],
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
            data: 2D array (H, W) of any integer dtype (uint8, uint16) — modified in-place.
            channel: Channel index into ``self.grids`` / ``self.slide_means``.
        """
        grid = self.grids[channel]
        slide_mean = self.slide_means[channel]
        bs = self.block_size
        H, W = data.shape

        # Floor to avoid extreme correction in very dark regions
        illum_floor = slide_mean * 0.1

        # Determine clip range based on input dtype
        input_dtype = data.dtype
        if np.issubdtype(input_dtype, np.integer):
            dtype_max = float(np.iinfo(input_dtype).max)
        else:
            dtype_max = float(np.finfo(input_dtype).max)

        logger.info(
            "Correcting channel %d: grid %s, slide_mean=%.1f, illum_floor=%.1f, dtype=%s",
            channel,
            grid.shape,
            slide_mean,
            illum_floor,
            input_dtype,
        )

        # Upscale coarse grid to full slide resolution via bilinear interpolation.
        # zoom() maps grid centers to pixel centers automatically, so no manual
        # coordinate offset is needed. Memory: one float32 array of (H, W).
        # Grid is float32, so zoom() outputs float32 directly (no float64 intermediate).
        zoom_factors = (H / grid.shape[0], W / grid.shape[1])
        full_illum = zoom(grid, zoom_factors, order=1, mode="nearest")

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
            np.clip(corrected, 0, dtype_max, out=corrected)

            # Write back and restore zero padding
            result = corrected.astype(input_dtype)
            result[zero_mask] = 0
            data[row_start:row_end] = result

        del correction_field, full_illum
        gc.collect()

        logger.info(
            "Channel %d correction range: [%.3f, %.3f]",
            channel,
            correction_min,
            correction_max,
        )

    def save(self, path, *, metadata: dict) -> None:
        """Serialize profile + metadata to ``.npz`` via atomic rename.

        Writes via ``{path}.partial.npz`` (``np.savez_compressed`` auto-appends
        ``.npz`` to its target path) then atomically renames it into place.
        Metadata must be JSON-serializable — typically CZI identity + shape +
        channel list + photobleach flag so cache freshness is validatable.
        """
        path = Path(path)
        # np.savez_compressed appends .npz to its target. Use a name ending in
        # .partial.npz so the eventual rename hits the intended path.
        tmp_base = path.with_name(path.stem + ".partial")
        tmp_file = tmp_base.with_suffix(tmp_base.suffix + ".npz")
        channels = sorted(self.grids.keys())
        save_args: dict = {
            "algorithm_version": np.array([ALGORITHM_VERSION]),
            "block_size": np.array([self.block_size], dtype=np.int32),
            "channels": np.array(channels, dtype=np.int32),
            "slide_means": np.array([self.slide_means[c] for c in channels], dtype=np.float64),
            "metadata": np.frombuffer(
                json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
            ),
        }
        for c in channels:
            save_args[f"grid_{c}"] = self.grids[c]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(tmp_base), **save_args)  # writes tmp_file
        os.replace(tmp_file, path)

    @classmethod
    def load(cls, path) -> "tuple[IlluminationProfile, dict]":
        """Load from ``.npz``. Returns ``(profile, metadata_dict)``.

        Raises ``ValueError`` if the file's ``algorithm_version`` doesn't match
        the current module version — callers should treat that as a cache miss
        and recompute.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        version = str(data["algorithm_version"][0])
        if version != ALGORITHM_VERSION:
            raise ValueError(
                f"Flat-field cache algorithm_version={version!r} does not match "
                f"current {ALGORITHM_VERSION!r}"
            )
        block_size = int(data["block_size"][0])
        channels = [int(c) for c in data["channels"]]
        slide_means = {c: float(v) for c, v in zip(channels, data["slide_means"], strict=True)}
        grids = {c: np.asarray(data[f"grid_{c}"], dtype=np.float32) for c in channels}
        metadata = json.loads(bytes(data["metadata"]).decode("utf-8"))
        return cls(grids, block_size, slide_means), metadata


def estimate_illumination_profile(
    all_channel_data: dict[int, np.ndarray],
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
    grids: dict[int, np.ndarray] = {}
    slide_means: dict[int, float] = {}

    for ch, data in all_channel_data.items():
        H, W = data.shape
        n_rows = int(np.ceil(H / block_size))
        n_cols = int(np.ceil(W / block_size))
        coarse = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

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
            ch,
            n_rows,
            n_cols,
            n_valid,
            n_total,
            ch_mean,
        )

        # Fill NaN blocks via nearest-neighbor interpolation
        if n_valid < n_total and n_valid > 0:
            nan_mask = ~valid_mask
            _, nearest_indices = distance_transform_edt(
                nan_mask, return_distances=True, return_indices=True
            )
            coarse[nan_mask] = coarse[nearest_indices[0][nan_mask], nearest_indices[1][nan_mask]]
        elif n_valid == 0:
            # Degenerate: no signal at all — fill with 1.0 (no correction)
            coarse[:] = 1.0

        # Gaussian smooth
        smoothed = gaussian_filter(coarse, sigma=smooth_sigma)

        grids[ch] = smoothed
        slide_means[ch] = ch_mean

    return IlluminationProfile(grids, block_size, slide_means)
