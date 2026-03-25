"""Load ROIs from external polygon annotations or pre-computed label masks.

Supports the JSON format produced by ``annotate_bone_regions.py`` (polygons
with ``vertices_px``) and plain NumPy / image label masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def load_rois_from_polygons(
    polygon_json_path: str | Path,
    image_shape: tuple[int, int],
    pixel_size: float = 1.0,
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Rasterise polygon annotations into a label mask.

    Reads a JSON file containing region polygons (each with a ``vertices_px``
    key holding ``[[x, y], ...]`` vertex lists in full-resolution pixel coords).
    Each polygon is rasterised into a unique label value.

    The JSON may be a flat list of region dicts **or** a nested structure where
    each slide has named regions (as produced by ``annotate_bone_regions.py``).

    Args:
        polygon_json_path: Path to the polygon JSON file.
        image_shape: ``(height, width)`` of the full-resolution image.
        pixel_size: Pixel size in um (stored but not used for rasterisation).
        scale_factor: Multiply vertex coordinates by this factor before
            rasterisation (e.g. if vertices were drawn at a different scale).

    Returns:
        ``(region_labels, 1)`` — label array at full resolution (downsample=1).
    """
    import cv2

    path = Path(polygon_json_path)
    with open(path) as f:
        data = json.load(f)

    # Extract polygon vertex lists from various JSON shapes
    polygons: list[list[list[float]]] = []

    if isinstance(data, list):
        # Flat list: each element is a region dict with "vertices_px"
        for region in data:
            verts = region.get("vertices_px")
            if verts:
                polygons.append(verts)
    elif isinstance(data, dict):
        # Nested: {"slides": {name: {region_name: {"vertices_px": [...]}, ...}}}
        slides = data.get("slides", data)
        for slide_key, slide_val in slides.items():
            if not isinstance(slide_val, dict):
                continue
            for region_key, region_val in slide_val.items():
                if isinstance(region_val, dict):
                    verts = region_val.get("vertices_px")
                    if verts:
                        polygons.append(verts)

    if not polygons:
        logger.warning("No polygons found in %s", path)
        return np.zeros(image_shape, dtype=np.int32), 1

    h, w = image_shape
    labels = np.zeros((h, w), dtype=np.int32)

    for label_id, verts in enumerate(polygons, start=1):
        pts = np.array(
            [[int(round(x * scale_factor)), int(round(y * scale_factor))] for x, y in verts],
            dtype=np.int32,
        )
        cv2.fillPoly(labels, [pts], color=int(label_id))

    logger.info("load_rois_from_polygons: %d regions from %s", len(polygons), path.name)
    return labels, 1


def load_rois_from_mask(mask_path: str | Path) -> tuple[np.ndarray, int]:
    """Load a pre-computed label mask from a NumPy ``.npy`` file or image.

    The file should contain a 2-D integer array where 0 = background and
    positive values are region labels.

    Args:
        mask_path: Path to a ``.npy`` file or an image file readable by
            :func:`PIL.Image.open`.

    Returns:
        ``(region_labels, 1)`` — full-resolution label array (downsample=1).
    """
    path = Path(mask_path)

    if path.suffix == ".npy":
        labels = np.load(path).astype(np.int32)
    else:
        from PIL import Image

        img = np.array(Image.open(path))
        if img.ndim == 3:
            # Convert RGB/RGBA to single channel by taking max across channels
            img = img[..., 0]
        labels = img.astype(np.int32)

    logger.info(
        "load_rois_from_mask: %s, shape=%s, %d regions",
        path.name,
        labels.shape,
        int(labels.max()) if labels.size > 0 else 0,
    )
    return labels, 1
