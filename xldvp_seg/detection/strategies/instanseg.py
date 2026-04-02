"""InstanSeg cell detection strategy -- lightweight alternative to Cellpose.

Uses the InstanSeg JIT model (3.8M params, Apache 2.0) for instance
segmentation. All feature extraction (morph, SAM2, per-channel, deep
features) is inherited from CellStrategy.

Requires: pip install instanseg-torch
"""

from typing import Any

import numpy as np

from xldvp_seg.detection.registry import register_strategy
from xldvp_seg.utils.logging import get_logger

from .cell import CellStrategy

logger = get_logger(__name__)


@register_strategy(
    "instanseg",
    description="Cell detection using InstanSeg (3.8M params, Apache 2.0, alternative to Cellpose)",
    channels=["cyto", "nuc"],
)
class InstanSegStrategy(CellStrategy):
    """Cell detection using InstanSeg instead of Cellpose.

    Overrides mask generation only. All feature extraction (morph + SAM2 +
    per-channel + optional deep features) is inherited from CellStrategy.

    InstanSeg is a lightweight instance segmentation model (3.8M parameters)
    that runs as a JIT-compiled model. It supports fluorescence multichannel
    input natively.
    """

    def __init__(
        self,
        instanseg_model: str = "fluorescence_nuclei_and_cells",
        min_mask_pixels: int = 10,
        pixel_size_um: float = None,
        **kwargs,
    ):
        super().__init__(min_mask_pixels=min_mask_pixels, **kwargs)
        self.instanseg_model_name = instanseg_model
        self._pixel_size_um = pixel_size_um
        self._instanseg = None

    @property
    def name(self) -> str:
        return "instanseg"

    def detect(self, tile, models, pixel_size_um, **kwargs):
        """Override to capture pixel_size_um before CellStrategy.detect() calls segment()."""
        self._pixel_size_um = pixel_size_um
        return super().detect(tile, models, pixel_size_um, **kwargs)

    def _ensure_instanseg(self):
        """Lazy-load the InstanSeg model."""
        if self._instanseg is not None:
            return
        try:
            from instanseg import InstanSeg

            self._instanseg = InstanSeg(self.instanseg_model_name)
            logger.info("InstanSeg loaded: %s", self.instanseg_model_name)
        except ImportError:
            raise RuntimeError(
                "InstanSeg not installed. Install with: pip install instanseg-torch\n"
                "See: https://github.com/instanseg/instanseg"
            )

    def segment(
        self,
        tile: np.ndarray,
        models: dict[str, Any],
        extra_channels: dict[int, np.ndarray] = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Generate cell masks using InstanSeg.

        Args:
            tile: RGB image array (HxWx3)
            models: Dict with model references (InstanSeg loaded internally)
            extra_channels: Additional channel data (used for 2-channel input)

        Returns:
            List of boolean masks (one per detected cell)
        """
        self._ensure_instanseg()

        try:
            import torch

            # Build input: 2-channel (cyto+nuc) or RGB
            if self.cellpose_input_channels and extra_channels:
                cyto_idx, nuc_idx = self.cellpose_input_channels
                cyto_ch = extra_channels.get(cyto_idx)
                nuc_ch = extra_channels.get(nuc_idx)
                if cyto_ch is not None and nuc_ch is not None:
                    cyto_u8 = self._percentile_normalize_single(cyto_ch)
                    nuc_u8 = self._percentile_normalize_single(nuc_ch)
                    # InstanSeg expects (C, H, W) tensor
                    input_tensor = torch.from_numpy(np.stack([cyto_u8, nuc_u8], axis=0)).float()
                else:
                    input_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float()
            else:
                # RGB tile (H, W, 3) -> (3, H, W) tensor
                input_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float()

            # Run InstanSeg: eval_small_image returns (labeled_masks, image_tensor)
            # labeled_masks shape: (1, n_types, H, W) where n_types=2 (nuclei+cells)
            pixel_size = self._pixel_size_um or kwargs.get("pixel_size_um", None)
            if pixel_size is None:
                raise ValueError(
                    "InstanSeg requires pixel_size_um — set in constructor or pass via kwargs"
                )
            result = self._instanseg.eval_small_image(
                input_tensor,
                pixel_size=pixel_size,
            )

            # Extract labeled mask from result tuple: (labeled_masks, image_tensor)
            if isinstance(result, tuple):
                labeled_tensor = result[0]
            else:
                labeled_tensor = result
            # Shape: (1, n_types, H, W). For "fluorescence_nuclei_and_cells":
            #   channel 0 = nuclei instances, channel 1 = cell instances
            # Use cell channel (index 1) for whole-cell segmentation.
            if labeled_tensor.dim() == 4 and labeled_tensor.shape[1] >= 2:
                labeled_mask = labeled_tensor[0, 1].cpu().numpy()  # cell channel
            elif labeled_tensor.dim() == 4:
                labeled_mask = labeled_tensor[0, 0].cpu().numpy()
            else:
                labeled_mask = labeled_tensor.squeeze().cpu().numpy()

            labeled_mask = labeled_mask.astype(np.int32)

            # Convert labeled mask to list of boolean masks
            masks = []
            unique_labels = np.unique(labeled_mask)
            unique_labels = unique_labels[unique_labels > 0]

            for label_id in unique_labels:
                binary_mask = labeled_mask == label_id
                if binary_mask.sum() >= self.min_mask_pixels:
                    masks.append(binary_mask)

            logger.debug("InstanSeg: %d cells detected in tile", len(masks))

        except (ValueError, RuntimeError):
            raise  # propagate config/setup errors (pixel_size, model loading)
        except Exception as e:
            logger.warning("InstanSeg failed on tile: %s", e)
            masks = []

        return masks
