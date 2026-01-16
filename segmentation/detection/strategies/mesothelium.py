"""
Mesothelium detection strategy for laser microdissection.

Detects mesothelial ribbon structures and divides them into chunks of
approximately target area (default ~1500 µm²) suitable for LMD collection.

Pipeline:
1. Ridge detection using Meijering filter (optimized for thin ribbon structures)
2. Morphological cleanup and fragment filtering
3. Medial axis / skeleton extraction with distance transform
4. Walk along skeleton paths, accumulating area until target reached
5. Convert chunks to polygons for LMD export
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from .base import DetectionStrategy, Detection
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def _trace_skeleton_paths(skeleton: np.ndarray) -> List[np.ndarray]:
    """
    Simple skeleton path tracing (fallback if skan not available).

    Returns list of paths, each path is Nx2 array of (row, col) coordinates.
    """
    from collections import deque

    # Find endpoints and branch points
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
    neighbor_count = neighbor_count * skeleton

    endpoints = (neighbor_count == 1) & skeleton

    paths = []
    visited = np.zeros_like(skeleton, dtype=bool)

    # Start from each endpoint
    endpoint_coords = np.argwhere(endpoints)

    for start in endpoint_coords:
        if visited[start[0], start[1]]:
            continue

        # Trace path from this endpoint
        path = [start]
        visited[start[0], start[1]] = True
        current = start

        while True:
            # Find unvisited neighbors
            r, c = current
            found_next = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < skeleton.shape[0] and
                        0 <= nc < skeleton.shape[1] and
                        skeleton[nr, nc] and
                        not visited[nr, nc]):
                        path.append(np.array([nr, nc]))
                        visited[nr, nc] = True
                        current = np.array([nr, nc])
                        found_next = True
                        break
                if found_next:
                    break

            if not found_next:
                break

        if len(path) >= 3:
            paths.append(np.array(path))

    return paths


def _skeleton_chunk_to_polygon(
    path_points: np.ndarray,
    widths_px: np.ndarray,
    pixel_size: float
) -> Optional[np.ndarray]:
    """
    Convert skeleton path with widths to closed polygon.

    Args:
        path_points: Nx2 array of (row, col) skeleton coordinates
        widths_px: Width at each skeleton point in pixels
        pixel_size: Pixel size in µm (for reference)

    Returns:
        Polygon as Mx2 array of (x, y) coordinates, or None if invalid
    """
    path_points = np.array(path_points)
    widths_px = np.array(widths_px)

    if len(path_points) < 2:
        return None

    left_boundary = []
    right_boundary = []

    for i in range(len(path_points)):
        half_width = widths_px[i] / 2

        # Get tangent direction
        if i == 0:
            tangent = path_points[1] - path_points[0]
        elif i == len(path_points) - 1:
            tangent = path_points[-1] - path_points[-2]
        else:
            tangent = path_points[i+1] - path_points[i-1]

        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            continue
        tangent = tangent / norm

        # Perpendicular
        perp = np.array([-tangent[1], tangent[0]])

        # Boundary points (row, col format)
        left_pt = path_points[i] + perp * half_width
        right_pt = path_points[i] - perp * half_width

        left_boundary.append(left_pt)
        right_boundary.append(right_pt)

    if len(left_boundary) < 2:
        return None

    # Create closed polygon: left forward, then right backward
    # Convert to (col, row) = (x, y) for cv2
    polygon = np.vstack([
        np.array(left_boundary)[:, ::-1],  # (row,col) to (col,row)
        np.array(right_boundary)[::-1, ::-1]
    ])

    return polygon


class MesotheliumStrategy(DetectionStrategy):
    """
    Detection strategy for mesothelial ribbon structures.

    Designed for laser microdissection (LMD) workflows:
    - Detects thin ribbon-like mesothelial structures
    - Divides ribbons into chunks of target area
    - Outputs polygons suitable for LMD XML export

    Args:
        target_chunk_area_um2: Target area for each chunk (default 1500 µm²)
        min_ribbon_width_um: Minimum expected ribbon width (default 5 µm)
        max_ribbon_width_um: Maximum expected ribbon width (default 30 µm)
        min_fragment_area_um2: Skip fragments smaller than this (default 1500 µm²)
        pixel_size_um: Pixel size in microns
    """

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return 'mesothelium'

    def __init__(
        self,
        target_chunk_area_um2: float = 1500.0,
        min_ribbon_width_um: float = 5.0,
        max_ribbon_width_um: float = 30.0,
        min_fragment_area_um2: float = 1500.0,
        pixel_size_um: float = 0.22
    ):
        self.target_chunk_area_um2 = target_chunk_area_um2
        self.min_ribbon_width_um = min_ribbon_width_um
        self.max_ribbon_width_um = max_ribbon_width_um
        self.min_fragment_area_um2 = min_fragment_area_um2
        self.pixel_size_um = pixel_size_um

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segment mesothelial ribbons and divide into chunks.

        Args:
            tile: RGB image array
            models: Dict with loaded models (not used for mesothelium)

        Returns:
            Tuple of (label_mask, raw_detections_list)
        """
        from skimage.morphology import (
            skeletonize, medial_axis, remove_small_objects,
            binary_closing, binary_opening, disk
        )
        from skimage.filters import meijering, threshold_local
        from skimage.measure import label, regionprops

        pixel_size = self.pixel_size_um

        # Convert width parameters from µm to pixels
        min_width_px = self.min_ribbon_width_um / pixel_size
        max_width_px = self.max_ribbon_width_um / pixel_size

        # Convert to grayscale
        if tile.ndim == 3:
            gray = np.mean(tile[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = tile.astype(np.float32)

        # Normalize to 0-1 for ridge detection
        gray_range = gray.max() - gray.min()
        if gray_range < 1e-8:
            # Uniform image, no ridges
            return np.zeros(tile.shape[:2], dtype=np.uint32), []
        gray_norm = (gray - gray.min()) / gray_range

        # Ridge detection using Meijering filter (optimized for neurite/line structures)
        sigmas = np.linspace(min_width_px * 0.5, max_width_px * 0.5, 5)
        ridges = meijering(gray_norm, sigmas=sigmas, black_ridges=False)

        # Threshold ridge response
        ridge_thresh = threshold_local(ridges, block_size=51, offset=-0.01)
        binary = ridges > ridge_thresh

        # Morphological cleanup
        binary = binary_opening(binary, disk(1))
        binary = binary_closing(binary, disk(2))
        min_size_px = int(self.min_fragment_area_um2 / (pixel_size ** 2) * 0.1)
        if min_size_px > 0:
            binary = remove_small_objects(binary, min_size=min_size_px)

        # Label connected components and filter by total area
        labeled = label(binary)
        props = regionprops(labeled)

        # Keep only fragments large enough to chunk
        valid_labels = []
        for prop in props:
            area_um2 = prop.area * (pixel_size ** 2)
            if area_um2 >= self.min_fragment_area_um2:
                valid_labels.append(prop.label)

        if len(valid_labels) == 0:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Create cleaned binary with only valid fragments
        binary_clean = np.isin(labeled, valid_labels)

        # Extract medial axis with distance transform
        skeleton, distance = medial_axis(binary_clean, return_distance=True)
        local_width = distance * 2  # Full width at skeleton points

        # Parse skeleton into paths
        try:
            from skan import Skeleton as SkanSkeleton
            skel_obj = SkanSkeleton(skeleton)
            paths = skel_obj.paths_list()
        except ImportError:
            # Fallback: trace paths manually
            paths = _trace_skeleton_paths(skeleton)

        # Chunk each path by area
        masks = np.zeros(tile.shape[:2], dtype=np.uint32)
        raw_detections = []
        chunk_id = 1

        for path_idx, path_coords in enumerate(paths):
            if len(path_coords) < 3:
                continue

            # Get local width at each path point
            widths_px = []
            for pt in path_coords:
                r, c = int(pt[0]), int(pt[1])
                if 0 <= r < local_width.shape[0] and 0 <= c < local_width.shape[1]:
                    widths_px.append(max(local_width[r, c], 1))
                else:
                    widths_px.append(min_width_px)
            widths_um = np.array(widths_px) * pixel_size

            # Walk along path, accumulating area until target reached
            chunks = []
            accumulated_area = 0
            chunk_start_idx = 0

            for i in range(1, len(path_coords)):
                # Segment length
                dx = (path_coords[i][1] - path_coords[i-1][1]) * pixel_size
                dy = (path_coords[i][0] - path_coords[i-1][0]) * pixel_size
                seg_length = np.sqrt(dx**2 + dy**2)

                # Average width
                avg_width = (widths_um[i] + widths_um[i-1]) / 2

                # Segment area
                accumulated_area += seg_length * avg_width

                # Check if we've reached target
                if accumulated_area >= self.target_chunk_area_um2:
                    chunks.append({
                        'start_idx': chunk_start_idx,
                        'end_idx': i,
                        'path_points': path_coords[chunk_start_idx:i+1],
                        'widths_px': widths_px[chunk_start_idx:i+1],
                        'area_um2': accumulated_area
                    })
                    chunk_start_idx = i
                    accumulated_area = 0

            # Handle remainder - merge with previous if too small
            if chunk_start_idx < len(path_coords) - 1:
                remainder_area = accumulated_area
                if remainder_area < self.target_chunk_area_um2 * 0.5 and len(chunks) > 0:
                    # Merge with previous chunk
                    prev = chunks[-1]
                    prev['end_idx'] = len(path_coords) - 1
                    prev['path_points'] = np.vstack([prev['path_points'], path_coords[chunk_start_idx+1:]])
                    prev['widths_px'] = list(prev['widths_px']) + widths_px[chunk_start_idx+1:]
                    prev['area_um2'] += remainder_area
                elif remainder_area >= self.min_fragment_area_um2 * 0.5:
                    # Keep as separate chunk if not too small
                    chunks.append({
                        'start_idx': chunk_start_idx,
                        'end_idx': len(path_coords) - 1,
                        'path_points': path_coords[chunk_start_idx:],
                        'widths_px': widths_px[chunk_start_idx:],
                        'area_um2': remainder_area
                    })

            # Convert chunks to polygons
            for chunk in chunks:
                polygon = _skeleton_chunk_to_polygon(
                    chunk['path_points'],
                    chunk['widths_px'],
                    pixel_size
                )

                if polygon is None or len(polygon) < 4:
                    continue

                # Create mask for this chunk
                chunk_mask = np.zeros(tile.shape[:2], dtype=np.uint8)
                cv2.fillPoly(chunk_mask, [polygon.astype(np.int32)], 255)
                chunk_mask_bool = chunk_mask > 0

                if chunk_mask_bool.sum() == 0:
                    continue

                masks[chunk_mask_bool] = chunk_id

                # Calculate centroid
                cy, cx = ndimage.center_of_mass(chunk_mask_bool)

                raw_detections.append({
                    'det_id': chunk_id,
                    'mask': chunk_mask_bool,
                    'centroid': (float(cx), float(cy)),
                    'polygon': polygon,
                    'area_um2': float(chunk['area_um2']),
                    'path_points': chunk['path_points'],
                    'widths_px': chunk['widths_px'],
                    'branch_id': path_idx,
                })

                chunk_id += 1

        return masks, raw_detections

    def filter(
        self,
        detections: List[Dict[str, Any]],
        tile: np.ndarray,
        models: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter mesothelium detections (minimal filtering needed).

        Args:
            detections: Raw detections from segment()
            tile: Original tile image
            models: Loaded models (not used)

        Returns:
            Filtered detections (same as input for mesothelium)
        """
        # Mesothelium chunks are already filtered during segmentation
        # No additional filtering needed
        return detections

    def extract_features(
        self,
        detections: List[Dict[str, Any]],
        tile: np.ndarray,
        models: Dict[str, Any]
    ) -> List[Detection]:
        """
        Extract features and create Detection objects.

        Args:
            detections: Filtered detections
            tile: Original tile image
            models: Loaded models (not used for mesothelium)

        Returns:
            List of Detection objects with mesothelium-specific features
        """
        pixel_size = self.pixel_size_um
        results = []

        for det in detections:
            mask = det['mask']
            area_px = int(mask.sum())

            # Basic features
            features = {
                'area': area_px,
                'area_um2': det['area_um2'],
                'path_length_um': float(len(det['path_points']) * pixel_size),
                'mean_width_um': float(np.mean(det['widths_px']) * pixel_size),
                'n_vertices': len(det['polygon']),
                'branch_id': det['branch_id'],
            }

            # Create Detection object
            detection = Detection(
                id=f"meso_{det['det_id']}",
                mask=mask,
                centroid=det['centroid'],
                features=features
            )

            # Store polygon for LMD export (not in standard Detection but useful)
            detection.polygon = det['polygon']

            results.append(detection)

        return results

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Full detection pipeline for mesothelium.

        Args:
            tile: RGB image array
            models: Dict with loaded models
            pixel_size_um: Pixel size (uses self.pixel_size_um if not provided)

        Returns:
            Tuple of (label_mask, List[Detection])
        """
        # Update pixel size if provided
        if pixel_size_um is not None:
            self.pixel_size_um = pixel_size_um

        # Segment
        masks, raw_detections = self.segment(tile, models)

        if not raw_detections:
            return masks, []

        # Filter (minimal for mesothelium)
        filtered = self.filter(raw_detections, tile, models)

        # Extract features
        detections = self.extract_features(filtered, tile, models)

        logger.debug(f"Mesothelium: {len(detections)} chunks detected")

        return masks, detections
