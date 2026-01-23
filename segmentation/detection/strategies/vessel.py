"""
Vessel detection strategy.

Detects blood vessel cross-sections (ring structures) in SMA-stained tissue
using contour hierarchy analysis and ellipse fitting.

Full feature extraction: 22 morphological + 256 SAM2 + 2048 ResNet = 2326 features
plus vessel-specific features (wall thickness, diameters, etc.)

Enhanced features (v2):
- Cross-tile boundary detection for partial vessels at tile edges
- RETR_TREE hierarchical contour analysis for complex nested structures
- Partial vessel matching and merging across adjacent tiles
- Tracking flags for merged vs single-tile vessels
"""

import gc
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import cv2
from PIL import Image

from .base import DetectionStrategy, Detection
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import (
    extract_morphological_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
)
from segmentation.utils.vessel_features import (
    extract_vessel_features,
    extract_all_vessel_features_multichannel,
    extract_multichannel_intensity_features,
    compute_channel_ratios,
    VESSEL_FEATURE_NAMES,
    VESSEL_FEATURE_COUNT,
    DEFAULT_CHANNEL_NAMES,
)

logger = get_logger(__name__)


class BoundaryEdge(Enum):
    """Enumeration of tile boundary edges."""
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class PartialVessel:
    """
    Represents a partially visible vessel at a tile boundary.

    Used for cross-tile merging of vessels that span multiple tiles.

    Attributes:
        tile_x: X coordinate of the source tile origin
        tile_y: Y coordinate of the source tile origin
        contour: The boundary contour points (Nx1x2 array)
        boundary_edges: Set of tile edges this vessel touches
        boundary_points: Points on the tile boundary (for matching)
        orientation: Approximate orientation angle in degrees
        curvature_signature: Curvature values along the boundary for matching
        features: Extracted vessel features (partial)
        is_outer: Whether this is an outer (wall) or inner (lumen) contour
        parent_idx: Index of parent contour if this is a child
    """
    tile_x: int
    tile_y: int
    contour: np.ndarray
    boundary_edges: Set[BoundaryEdge]
    boundary_points: np.ndarray
    orientation: float
    curvature_signature: np.ndarray
    features: Dict[str, Any] = field(default_factory=dict)
    is_outer: bool = True
    parent_idx: Optional[int] = None


@dataclass
class CrossTileMergeConfig:
    """
    Configuration for cross-tile vessel merging.

    Attributes:
        enabled: Whether cross-tile merging is enabled
        position_tolerance_px: Maximum distance between boundary points to match (pixels)
        orientation_tolerance_deg: Maximum orientation difference for matching (degrees)
        curvature_match_threshold: Minimum correlation for curvature matching (0-1)
        min_boundary_overlap: Minimum fraction of boundary that must overlap
        merge_partial_rings: Whether to attempt merging incomplete rings
    """
    enabled: bool = True
    position_tolerance_px: float = 10.0
    orientation_tolerance_deg: float = 30.0
    curvature_match_threshold: float = 0.7
    min_boundary_overlap: float = 0.3
    merge_partial_rings: bool = True


# Issue #7: Local extract_morphological_features removed - now imported from shared module


class VesselStrategy(DetectionStrategy):
    """
    Vessel detection strategy for ring structures.

    Vessels are detected using:
    1. Canny edge detection + Otsu thresholding for SMA+ regions
    2. Contour hierarchy analysis (RETR_CCOMP or RETR_TREE) to find parent-child pairs
    3. Ellipse fitting for outer (adventitia) and inner (lumen) boundaries
    4. Wall thickness measurement via distance transform + skeleton analysis
    5. Optional CD31 validation (endothelial marker at lumen boundary)
    6. Full feature extraction (22 morphological + 256 SAM2 + 2048 ResNet = 2326 features)

    Enhanced features (v2):
    7. Cross-tile boundary detection for partial vessels at tile edges
    8. RETR_TREE hierarchical analysis for complex nested structures
    9. Partial vessel matching and merging across adjacent tiles
    10. Tracking flags for merged vs single-tile vessels

    Ring structures are identified as outer contours that have inner contours
    (holes), representing the vessel wall surrounding the lumen.

    Parameters:
        min_diameter_um: Minimum outer diameter in microns (default: 10)
        max_diameter_um: Maximum outer diameter in microns (default: 1000)
        min_wall_thickness_um: Minimum wall thickness in microns (default: 2)
        max_aspect_ratio: Maximum major/minor axis ratio (default: 4.0)
            Higher values exclude longitudinal vessel sections
        min_circularity: Minimum circularity 0-1 (default: 0.3)
        min_ring_completeness: Minimum fraction of SMA+ perimeter (default: 0.5)
        canny_low: Low threshold for Canny (auto if None)
        canny_high: High threshold for Canny (auto if None)
        classify_vessel_types: Whether to auto-classify by diameter (default: False)
        extract_resnet_features: Whether to extract 2048D ResNet features (default: True)
        extract_sam2_embeddings: Whether to extract 256D SAM2 embeddings (default: True)
        resnet_batch_size: Batch size for ResNet feature extraction (default: 16)
        enable_boundary_detection: Enable detection of partial vessels at tile edges (default: True)
        boundary_margin_px: Margin in pixels to consider as tile boundary (default: 5)
        use_tree_hierarchy: Use RETR_TREE instead of RETR_CCOMP for complex structures (default: False)
        cross_tile_config: Configuration for cross-tile vessel merging (optional)
        candidate_mode: Enable candidate generation mode with relaxed thresholds (default: False)
            When True:
            - Lower min_circularity (0.1 instead of 0.3)
            - Lower min_ring_completeness (0.2 instead of 0.5)
            - Higher max_aspect_ratio (6.0 instead of 4.0)
            - Wider diameter range (5-2000 um instead of 10-1000)
            - Accept partial rings / incomplete contours
            - Detect open vessel structures (arcs, curves)
            - Include detection_confidence score (0-1) for each candidate
            Designed for generating training data for manual annotation + RF classifier
    """

    # Default thresholds for standard detection
    DEFAULT_MIN_CIRCULARITY = 0.3
    DEFAULT_MIN_RING_COMPLETENESS = 0.5
    DEFAULT_MAX_ASPECT_RATIO = 4.0
    DEFAULT_MIN_DIAMETER_UM = 10.0
    DEFAULT_MAX_DIAMETER_UM = 1000.0
    DEFAULT_MIN_WALL_THICKNESS_UM = 2.0

    # Relaxed thresholds for candidate generation mode
    CANDIDATE_MIN_CIRCULARITY = 0.1
    CANDIDATE_MIN_RING_COMPLETENESS = 0.2
    CANDIDATE_MAX_ASPECT_RATIO = 6.0
    CANDIDATE_MIN_DIAMETER_UM = 5.0
    CANDIDATE_MAX_DIAMETER_UM = 2000.0
    CANDIDATE_MIN_WALL_THICKNESS_UM = 1.0

    # Thresholds for open vessel (arc) detection in candidate mode
    ARC_MIN_CURVATURE = 0.01  # Minimum average curvature to be considered vessel-like
    ARC_MIN_LENGTH_UM = 20.0  # Minimum arc length in microns
    ARC_MAX_STRAIGHTNESS = 0.8  # Maximum straightness (1.0 = perfectly straight line)
    ARC_MAX_CONTOURS_TO_PROCESS = 500  # Limit contours to prevent performance issues
    ARC_MIN_CONTOUR_POINTS = 50  # Skip very small contours (more aggressive than 10)

    def __init__(
        self,
        min_diameter_um: float = 10,
        max_diameter_um: float = 1000,
        min_wall_thickness_um: float = 2,
        max_aspect_ratio: float = 4.0,
        min_circularity: float = 0.3,
        min_ring_completeness: float = 0.5,
        canny_low: Optional[int] = None,
        canny_high: Optional[int] = None,
        classify_vessel_types: bool = False,
        extract_resnet_features: bool = True,
        extract_sam2_embeddings: bool = True,
        resnet_batch_size: int = 32,
        # New parameters for boundary detection
        enable_boundary_detection: bool = True,
        boundary_margin_px: int = 5,
        use_tree_hierarchy: bool = False,
        cross_tile_config: Optional[CrossTileMergeConfig] = None,
        # Candidate generation mode - relaxes thresholds to catch more potential vessels
        candidate_mode: bool = False,
        # Parallel detection mode - runs SMA, CD31, LYVE1 detection in parallel
        parallel_detection: bool = False,
        parallel_workers: int = 3,
        # Multi-marker mode - enables full pipeline with candidate merging
        multi_marker: bool = False,
        # IoU threshold for merging overlapping candidates from different markers
        merge_iou_threshold: float = 0.5,
        # Lumen-first detection mode - finds dark lumens first, validates bright wall
        lumen_first: bool = False,
    ):
        self._lumen_first = lumen_first
        self.candidate_mode = candidate_mode

        # Apply relaxed thresholds if candidate_mode is enabled
        if candidate_mode:
            # Use most permissive thresholds between user-provided and candidate defaults
            self.min_diameter_um = min(min_diameter_um, self.CANDIDATE_MIN_DIAMETER_UM)
            self.max_diameter_um = max(max_diameter_um, self.CANDIDATE_MAX_DIAMETER_UM)
            self.max_aspect_ratio = max(max_aspect_ratio, self.CANDIDATE_MAX_ASPECT_RATIO)
            self.min_circularity = min(min_circularity, self.CANDIDATE_MIN_CIRCULARITY)
            self.min_ring_completeness = min(min_ring_completeness, self.CANDIDATE_MIN_RING_COMPLETENESS)
            self.min_wall_thickness_um = min(min_wall_thickness_um, self.CANDIDATE_MIN_WALL_THICKNESS_UM)
            logger.info(
                f"Candidate mode enabled: relaxed thresholds - "
                f"circularity>={self.min_circularity}, "
                f"ring_completeness>={self.min_ring_completeness}, "
                f"aspect_ratio<={self.max_aspect_ratio}, "
                f"diameter={self.min_diameter_um}-{self.max_diameter_um}um, "
                f"wall_thickness>={self.min_wall_thickness_um}um"
            )
        else:
            self.min_diameter_um = min_diameter_um
            self.max_diameter_um = max_diameter_um
            self.max_aspect_ratio = max_aspect_ratio
            self.min_circularity = min_circularity
            self.min_ring_completeness = min_ring_completeness
            self.min_wall_thickness_um = min_wall_thickness_um

        self.canny_low = canny_low
        self.canny_high = canny_high
        self.classify_vessel_types = classify_vessel_types
        self.extract_resnet_features = extract_resnet_features
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.resnet_batch_size = resnet_batch_size

        # Boundary detection parameters
        self.enable_boundary_detection = enable_boundary_detection
        self.boundary_margin_px = boundary_margin_px
        self.use_tree_hierarchy = use_tree_hierarchy
        self.cross_tile_config = cross_tile_config or CrossTileMergeConfig()

        # Storage for partial vessels pending merge (tile coords -> list of PartialVessel)
        self._partial_vessels: Dict[Tuple[int, int], List[PartialVessel]] = {}

        # Parallel detection settings
        self.parallel_detection = parallel_detection
        self.parallel_workers = parallel_workers

        # Multi-marker mode settings
        self.multi_marker = multi_marker
        self.merge_iou_threshold = merge_iou_threshold

        # If multi_marker is enabled, auto-enable parallel_detection
        if multi_marker and not parallel_detection:
            self.parallel_detection = True
            logger.info("Multi-marker mode: auto-enabled parallel_detection")

        # Lumen-first detection mode (default False, uses edge-based detection)
        self.lumen_first = getattr(self, '_lumen_first', False)

    @property
    def name(self) -> str:
        return "vessel"

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
        tile_x: int = 0,
        tile_y: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Segment ring structures using contour hierarchy.

        Uses Canny edge detection to find vessel edges, then contour hierarchy
        (RETR_CCOMP or RETR_TREE) to identify outer contours with inner contours (rings).

        Enhanced: Also detects partial vessels at tile boundaries for cross-tile merging.

        Args:
            tile: RGB or grayscale image (SMA channel)
            models: Dict of models (not used for vessel detection)
            pixel_size_um: Pixel size in microns for filtering
            cd31_channel: Optional CD31 channel for validation
            tile_x: X coordinate of tile origin (for boundary tracking)
            tile_y: Y coordinate of tile origin (for boundary tracking)

        Returns:
            List of ring candidate dicts with 'outer', 'inner', 'all_inner' contours,
            plus boundary information if applicable
        """
        h, w = tile.shape[:2]

        # Convert to grayscale
        if tile.ndim == 3:
            gray = np.mean(tile[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = tile.astype(np.float32)

        # Normalize to 0-255 for OpenCV
        gray_min, gray_max = gray.min(), gray.max()
        if gray_max - gray_min > 1e-8:
            gray_norm = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
        else:
            gray_norm = np.zeros_like(gray, dtype=np.uint8)

        # =====================================================================
        # MULTI-SCALE EDGE DETECTION FOR SIZE-INVARIANT VESSEL DETECTION
        # =====================================================================
        # Uses multiple blur scales and Canny parameters to detect vessels
        # across the full size range (5-500um):
        # - Small scale: Better for capillaries (5-10um) and arterioles (10-50um)
        # - Medium scale: Standard for arterioles and small arteries (50-150um)
        # - Large scale: Better for large arteries (>150um)
        # The results are combined to ensure no size class is missed.

        # Calculate minimum area threshold based on smallest expected vessel
        min_vessel_diameter_px = self.min_diameter_um / pixel_size_um
        min_area_threshold = int(np.pi * (min_vessel_diameter_px / 2) ** 2 * 0.1)  # 10% of min vessel area
        min_area_threshold = max(10, min_area_threshold)  # At least 10 pixels

        # Scale 1: Small vessels (fine details, small blur)
        blurred_small = cv2.GaussianBlur(gray_norm, (3, 3), 0.8)

        # Scale 2: Medium vessels (standard parameters)
        blurred_medium = cv2.GaussianBlur(gray_norm, (5, 5), 1.5)

        # Scale 3: Large vessels (coarser details, larger blur)
        blurred_large = cv2.GaussianBlur(gray_norm, (7, 7), 2.0)

        # Auto-calculate Canny thresholds using Otsu's method on medium scale
        if self.canny_low is None or self.canny_high is None:
            otsu_thresh, _ = cv2.threshold(blurred_medium, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = int(otsu_thresh * 0.5)
            canny_high = int(otsu_thresh * 1.0)
        else:
            canny_low = self.canny_low
            canny_high = self.canny_high

        # Multi-scale Canny edge detection
        # Small scale: Use slightly lower thresholds for faint capillary walls
        edges_small = cv2.Canny(blurred_small, int(canny_low * 0.7), int(canny_high * 0.8))
        # Medium scale: Standard thresholds
        edges_medium = cv2.Canny(blurred_medium, canny_low, canny_high)
        # Large scale: Slightly higher thresholds to reduce noise
        edges_large = cv2.Canny(blurred_large, int(canny_low * 1.1), int(canny_high * 1.2))

        # Combine edges from all scales using OR operation
        edges = edges_small | edges_medium | edges_large

        # Size-adaptive dilation to close gaps
        # Smaller dilation for small vessel detection, larger for big vessels
        kernel_dilate_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_dilate_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Apply smaller dilation first (preserves small vessel details)
        edges_dilated = cv2.dilate(edges, kernel_dilate_small, iterations=1)
        # Then apply medium dilation (helps close gaps in larger vessels)
        edges_dilated = cv2.dilate(edges_dilated, kernel_dilate_medium, iterations=1)

        # Fill detected edges to create binary regions
        binary = np.zeros_like(edges_dilated)

        # Find contours from edges
        edge_contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Fill closed contours - use size-adaptive minimum area
        for cnt in edge_contours:
            if cv2.contourArea(cnt) > min_area_threshold:
                cv2.drawContours(binary, [cnt], 0, 255, -1)

        # Size-adaptive morphological cleanup
        # Smaller kernels preserve small vessel details
        kernel_open_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_close_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Two-pass cleanup: first with small kernels, then with larger for gap closing
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open_small)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_small)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_large)

        # Find contours with hierarchy for ring detection
        # Use RETR_TREE for complex nested structures, RETR_CCOMP for standard detection
        retrieval_mode = cv2.RETR_TREE if self.use_tree_hierarchy else cv2.RETR_CCOMP
        contours, hierarchy = cv2.findContours(
            binary, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(contours) == 0:
            # Even with no complete rings, check for partial vessels at boundaries
            if self.enable_boundary_detection:
                self._detect_partial_vessels_from_binary(
                    binary, tile_x, tile_y, h, w, pixel_size_um
                )
            return []

        hierarchy = hierarchy[0]  # Shape: (N, 4) where 4 = [next, prev, child, parent]

        # Find ring candidates based on hierarchy mode
        ring_candidates = []

        if self.use_tree_hierarchy:
            # RETR_TREE: Full hierarchy - find all valid ring patterns
            ring_candidates = self._find_rings_tree_hierarchy(
                contours, hierarchy, binary, h, w, tile_x, tile_y, pixel_size_um
            )
        else:
            # RETR_CCOMP: 2-level hierarchy - outer contours with direct holes
            ring_candidates = self._find_rings_ccomp_hierarchy(
                contours, hierarchy, binary, h, w, tile_x, tile_y, pixel_size_um
            )

        # In candidate mode, also detect "open" vessel structures (arcs/curves)
        # These are vessel-like curved structures that don't form complete rings
        if self.candidate_mode:
            arc_candidates = self._detect_open_vessel_structures(
                edges_dilated, contours, hierarchy, binary, h, w,
                tile_x, tile_y, pixel_size_um
            )
            ring_candidates.extend(arc_candidates)

        return ring_candidates

    def _find_rings_ccomp_hierarchy(
        self,
        contours: List[np.ndarray],
        hierarchy: np.ndarray,
        binary: np.ndarray,
        h: int,
        w: int,
        tile_x: int,
        tile_y: int,
        pixel_size_um: float,
    ) -> List[Dict[str, Any]]:
        """
        Find ring candidates using RETR_CCOMP (2-level) hierarchy.

        Standard approach: outer contours (parent=-1) with children (holes).
        """
        ring_candidates = []
        processed_contours = set()

        for i, (next_c, prev_c, child, parent) in enumerate(hierarchy):
            if parent == -1 and child != -1:  # Outer contour with at least one hole
                outer_contour = contours[i]
                processed_contours.add(i)

                # Collect all child contours (holes)
                inner_contours = []
                child_idx = child
                while child_idx != -1:
                    inner_contours.append(contours[child_idx])
                    processed_contours.add(child_idx)
                    child_idx = hierarchy[child_idx][0]  # Next sibling

                # Take the largest inner contour as the main lumen
                if inner_contours:
                    inner_contour = max(inner_contours, key=cv2.contourArea)

                    # Check if this vessel touches tile boundaries
                    boundary_info = self._analyze_boundary_contact(
                        outer_contour, inner_contour, h, w
                    )

                    ring_candidates.append({
                        'outer': outer_contour,
                        'inner': inner_contour,
                        'all_inner': inner_contours,
                        'binary': binary,
                        'touches_boundary': boundary_info['touches_boundary'],
                        'boundary_edges': boundary_info['edges'],
                        'is_partial': boundary_info['is_partial'],
                        'is_merged': False,
                        'source_tiles': [(tile_x, tile_y)],
                    })

        # Detect partial vessels at boundaries (contours without complete ring structure)
        if self.enable_boundary_detection:
            partial_candidates = self._detect_boundary_partial_vessels(
                contours, hierarchy, processed_contours, binary,
                h, w, tile_x, tile_y, pixel_size_um
            )
            ring_candidates.extend(partial_candidates)

        return ring_candidates

    def _find_rings_tree_hierarchy(
        self,
        contours: List[np.ndarray],
        hierarchy: np.ndarray,
        binary: np.ndarray,
        h: int,
        w: int,
        tile_x: int,
        tile_y: int,
        pixel_size_um: float,
    ) -> List[Dict[str, Any]]:
        """
        Find ring candidates using RETR_TREE (full) hierarchy.

        Enables detection of nested vessels (vessel within vessel) and more
        complex structures that RETR_CCOMP misses.

        RETR_TREE hierarchy: [next, prev, first_child, parent]
        - Top-level contours have parent=-1
        - Children at any level can have their own children
        """
        ring_candidates = []
        processed_as_inner = set()

        def get_children(idx: int) -> List[int]:
            """Get all direct children of a contour."""
            children = []
            child_idx = hierarchy[idx][2]  # First child
            while child_idx != -1:
                children.append(child_idx)
                child_idx = hierarchy[child_idx][0]  # Next sibling
            return children

        def get_depth(idx: int) -> int:
            """Get the nesting depth of a contour."""
            depth = 0
            current = idx
            while hierarchy[current][3] != -1:  # While has parent
                depth += 1
                current = hierarchy[current][3]
            return depth

        # Process all contours by depth level
        for i in range(len(contours)):
            if i in processed_as_inner:
                continue

            depth = get_depth(i)
            children = get_children(i)

            # A ring is: a contour at even depth with children at odd depth (holes)
            # Or: any contour with child contours representing lumens
            if children and depth % 2 == 0:
                outer_contour = contours[i]

                # Get child contours (potential lumens)
                inner_contours = [contours[c] for c in children]

                # Mark children as processed
                for c in children:
                    processed_as_inner.add(c)

                    # Also check for nested rings (grandchildren)
                    grandchildren = get_children(c)
                    if grandchildren:
                        # This child has its own children - could be a nested vessel
                        nested_outer = contours[c]
                        nested_inners = [contours[gc] for gc in grandchildren]

                        if nested_inners:
                            nested_inner = max(nested_inners, key=cv2.contourArea)
                            boundary_info = self._analyze_boundary_contact(
                                nested_outer, nested_inner, h, w
                            )

                            ring_candidates.append({
                                'outer': nested_outer,
                                'inner': nested_inner,
                                'all_inner': nested_inners,
                                'binary': binary,
                                'touches_boundary': boundary_info['touches_boundary'],
                                'boundary_edges': boundary_info['edges'],
                                'is_partial': boundary_info['is_partial'],
                                'is_merged': False,
                                'is_nested': True,
                                'parent_vessel_idx': len(ring_candidates),
                                'source_tiles': [(tile_x, tile_y)],
                            })

                        for gc in grandchildren:
                            processed_as_inner.add(gc)

                if inner_contours:
                    inner_contour = max(inner_contours, key=cv2.contourArea)
                    boundary_info = self._analyze_boundary_contact(
                        outer_contour, inner_contour, h, w
                    )

                    ring_candidates.append({
                        'outer': outer_contour,
                        'inner': inner_contour,
                        'all_inner': inner_contours,
                        'binary': binary,
                        'touches_boundary': boundary_info['touches_boundary'],
                        'boundary_edges': boundary_info['edges'],
                        'is_partial': boundary_info['is_partial'],
                        'is_merged': False,
                        'is_nested': False,
                        'source_tiles': [(tile_x, tile_y)],
                    })

        # Detect partial vessels at boundaries
        if self.enable_boundary_detection:
            processed_contours = processed_as_inner.copy()
            for rc in ring_candidates:
                # Add outer contour indices
                for i, cnt in enumerate(contours):
                    if np.array_equal(cnt, rc['outer']):
                        processed_contours.add(i)
                        break

            partial_candidates = self._detect_boundary_partial_vessels(
                contours, hierarchy, processed_contours, binary,
                h, w, tile_x, tile_y, pixel_size_um
            )
            ring_candidates.extend(partial_candidates)

        return ring_candidates

    def _detect_lumen_first(
        self,
        tile: np.ndarray,
        pixel_size_um: float = 0.1725,
        min_lumen_area_um2: float = 50,
        max_lumen_area_um2: float = 150000,
        min_ellipse_fit: float = 0.40,
        max_aspect_ratio: float = 5.0,
        min_wall_brightness_ratio: float = 1.15,
        wall_thickness_fraction: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Lumen-first vessel detection: find dark lumens, validate bright SMA+ wall.

        This approach is better for candidate generation as it:
        1. Finds dark regions using Otsu threshold
        2. Fits ellipses to validate shape (permissive)
        3. Checks for bright wall surrounding the lumen
        4. Allows irregular shapes to pass for classifier training

        Args:
            tile: Grayscale SMA image
            pixel_size_um: Pixel size in micrometers
            min_lumen_area_um2: Minimum lumen area in µm²
            max_lumen_area_um2: Maximum lumen area in µm²
            min_ellipse_fit: Minimum ellipse fit quality (IoU, 0-1)
            max_aspect_ratio: Maximum major/minor axis ratio
            min_wall_brightness_ratio: Minimum wall/lumen intensity ratio
            wall_thickness_fraction: Wall region as fraction of lumen size

        Returns:
            List of candidate dicts with 'outer', 'inner', 'lumen_contour' keys
        """
        # Convert area thresholds to pixels
        min_lumen_area_px = min_lumen_area_um2 / (pixel_size_um ** 2)
        max_lumen_area_px = max_lumen_area_um2 / (pixel_size_um ** 2)

        # Normalize to uint8
        if tile.ndim == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) if tile.shape[2] == 3 else tile[:, :, 0]
        else:
            gray = tile

        if gray.dtype != np.uint8:
            img_min, img_max = gray.min(), gray.max()
            if img_max - img_min > 1e-8:
                img_norm = ((gray - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                return []
        else:
            img_norm = gray.copy()

        # Float version for intensity measurements
        img_float = gray.astype(np.float32)
        if img_float.max() > 0:
            img_float = img_float / img_float.max()

        h, w = img_float.shape[:2]

        # Find dark regions using Otsu threshold
        blurred = cv2.GaussianBlur(img_norm, (9, 9), 2.5)
        otsu_thresh, lumen_binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological cleanup
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel_open)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel_small)

        # Find contours
        contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.debug(f"Lumen-first: Otsu={otsu_thresh:.1f}, {len(contours)} dark regions")

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Size filter
            if area < min_lumen_area_px or area > max_lumen_area_px:
                continue
            if len(contour) < 5:
                continue

            # Fit ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
            except:
                continue

            (cx, cy), (minor_axis, major_axis), angle = ellipse

            # Aspect ratio check
            if minor_axis > 0:
                aspect_ratio = major_axis / minor_axis
            else:
                continue

            if aspect_ratio > max_aspect_ratio:
                continue

            # Ellipse fit quality (IoU)
            fit_quality = self._compute_ellipse_fit_quality(contour, ellipse, h, w)
            if fit_quality < min_ellipse_fit:
                continue

            # Check for bright wall around lumen
            lumen_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(lumen_mask, [contour], 0, 255, -1)

            avg_radius = (major_axis + minor_axis) / 4
            wall_thickness = max(3, int(avg_radius * wall_thickness_fraction))

            kernel_dilate = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * wall_thickness + 1, 2 * wall_thickness + 1)
            )
            dilated_mask = cv2.dilate(lumen_mask, kernel_dilate)
            wall_mask = dilated_mask - lumen_mask

            lumen_pixels = img_float[lumen_mask > 0]
            wall_pixels = img_float[wall_mask > 0]

            if len(lumen_pixels) < 10 or len(wall_pixels) < 10:
                continue

            lumen_mean = np.mean(lumen_pixels)
            wall_mean = np.mean(wall_pixels)

            if lumen_mean > 0:
                wall_lumen_ratio = wall_mean / lumen_mean
            else:
                wall_lumen_ratio = wall_mean / 0.01

            if wall_lumen_ratio < min_wall_brightness_ratio:
                continue

            # Get outer contour from dilated mask
            outer_contours, _ = cv2.findContours(
                dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            outer_contour = max(outer_contours, key=cv2.contourArea) if outer_contours else contour

            # Calculate diameters
            inner_diameter_um = (major_axis + minor_axis) / 2 * pixel_size_um
            outer_diameter_um = inner_diameter_um + 2 * wall_thickness * pixel_size_um

            candidates.append({
                'outer': outer_contour,
                'inner': contour,
                'lumen_contour': contour,
                'inner_ellipse': ellipse,
                'centroid': (cx, cy),
                'inner_area_px': area,
                'outer_area_px': cv2.contourArea(outer_contour),
                'inner_diameter_um': inner_diameter_um,
                'outer_diameter_um': outer_diameter_um,
                'aspect_ratio': aspect_ratio,
                'ellipse_fit_quality': fit_quality,
                'wall_lumen_ratio': wall_lumen_ratio,
                'lumen_mean': lumen_mean,
                'wall_mean': wall_mean,
                'detection_method': 'lumen_first',
            })

        logger.info(f"Lumen-first detection: {len(candidates)} vessels from {len(contours)} dark regions")
        return candidates

    def _compute_ellipse_fit_quality(
        self,
        contour: np.ndarray,
        ellipse: Tuple,
        img_h: int,
        img_w: int,
    ) -> float:
        """Compute IoU between contour and its fitted ellipse."""
        if ellipse is None or len(contour) < 5:
            return 0.0

        (cx, cy), (ma, MA), angle = ellipse
        x, y, w, h = cv2.boundingRect(contour)
        margin = 5
        x1, y1 = max(0, x - margin), max(0, y - margin)
        w2, h2 = min(w + 2*margin, img_w - x1), min(h + 2*margin, img_h - y1)

        if w2 <= 0 or h2 <= 0:
            return 0.0

        contour_mask = np.zeros((h2, w2), dtype=np.uint8)
        contour_shifted = contour.copy()
        contour_shifted[:, 0, 0] -= x1
        contour_shifted[:, 0, 1] -= y1
        cv2.drawContours(contour_mask, [contour_shifted], 0, 255, -1)

        ellipse_mask = np.zeros((h2, w2), dtype=np.uint8)
        ellipse_shifted = ((cx - x1, cy - y1), (ma, MA), angle)
        try:
            cv2.ellipse(ellipse_mask, ellipse_shifted, 255, -1)
        except:
            return 0.0

        intersection = np.logical_and(contour_mask > 0, ellipse_mask > 0).sum()
        union = np.logical_or(contour_mask > 0, ellipse_mask > 0).sum()

        if union == 0:
            return 0.0
        return intersection / union

    def _detect_open_vessel_structures(
        self,
        edges: np.ndarray,
        contours: List[np.ndarray],
        hierarchy: np.ndarray,
        binary: np.ndarray,
        h: int,
        w: int,
        tile_x: int,
        tile_y: int,
        pixel_size_um: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect "open" vessel structures (arcs/curves) that don't form complete rings.

        In candidate mode, we want to catch partial vessel cross-sections that may
        appear as curved structures rather than closed rings. This uses curvature
        analysis to identify vessel-like curves.

        Args:
            edges: Edge-detected image (dilated Canny output)
            contours: All contours found in the tile
            hierarchy: Contour hierarchy
            binary: Binary mask of the tile
            h: Tile height
            w: Tile width
            tile_x: X coordinate of tile origin
            tile_y: Y coordinate of tile origin
            pixel_size_um: Pixel size in microns

        Returns:
            List of arc/curve candidate dicts
        """
        arc_candidates = []

        # Find contours from edges that aren't already complete rings
        # We look for curved/arc shapes that might be vessel cross-sections
        edge_contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # Early exit if too many contours (performance protection)
        if len(edge_contours) > 10000:
            logger.debug(f"Skipping arc detection: too many edge contours ({len(edge_contours)})")
            return []

        # Track which contours are already part of ring candidates
        ring_contour_areas = set()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0:
                ring_contour_areas.add(int(area))

        # Sort contours by perimeter (larger first) and limit processing
        edge_contours_with_perim = []
        for cnt in edge_contours:
            if len(cnt) >= self.ARC_MIN_CONTOUR_POINTS:
                perim = cv2.arcLength(cnt, False)
                edge_contours_with_perim.append((perim, cnt))

        # Sort by perimeter descending (larger arcs first, more likely to be vessels)
        edge_contours_with_perim.sort(key=lambda x: -x[0])

        # Limit to top N contours by perimeter
        contours_to_process = edge_contours_with_perim[:self.ARC_MAX_CONTOURS_TO_PROCESS]

        logger.debug(f"Arc detection: processing {len(contours_to_process)}/{len(edge_contours)} contours")

        processed_count = 0
        for perimeter, cnt in contours_to_process:
            # Skip very small contours (redundant but kept for safety)
            if len(cnt) < self.ARC_MIN_CONTOUR_POINTS:
                continue

            # perimeter already calculated during sorting
            arc_length_um = perimeter * pixel_size_um

            # Skip if too short
            if arc_length_um < self.ARC_MIN_LENGTH_UM:
                continue

            # Skip if this contour is already part of a ring
            area = cv2.contourArea(cnt)
            if int(area) in ring_contour_areas:
                continue

            # Calculate curvature along the contour
            curvature_sig = self._compute_curvature_signature(cnt, num_samples=32)
            avg_curvature = np.mean(np.abs(curvature_sig))

            # Skip if too straight (not vessel-like)
            if avg_curvature < self.ARC_MIN_CURVATURE:
                continue

            # Calculate straightness: ratio of endpoint distance to arc length
            points = cnt.reshape(-1, 2)
            endpoint_dist = np.sqrt(np.sum((points[0] - points[-1]) ** 2))
            straightness = endpoint_dist / (perimeter + 1e-8)

            # Skip if too straight (likely not a vessel arc)
            if straightness > self.ARC_MAX_STRAIGHTNESS:
                continue

            # This looks like a vessel arc - create a candidate
            # Try to fit an ellipse for shape estimation
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (minor_ax, major_ax), angle = ellipse
                except cv2.error:
                    cx, cy = np.mean(points, axis=0)
                    major_ax = minor_ax = np.sqrt(area / np.pi) * 2 if area > 0 else 10
                    angle = 0
            else:
                cx, cy = np.mean(points, axis=0)
                major_ax = minor_ax = 10
                angle = 0

            # Estimate potential vessel diameter from arc curvature
            # For a circle, curvature = 1/radius, so diameter = 2/curvature
            if avg_curvature > 0:
                estimated_diameter_px = 2.0 / avg_curvature
                estimated_diameter_um = estimated_diameter_px * pixel_size_um
            else:
                estimated_diameter_um = max(major_ax, minor_ax) * pixel_size_um

            # Check diameter range
            if estimated_diameter_um < self.min_diameter_um:
                continue
            if estimated_diameter_um > self.max_diameter_um:
                continue

            # Calculate arc-specific confidence
            curvature_consistency = 1.0 - np.std(curvature_sig) / (avg_curvature + 1e-8)
            curvature_consistency = max(0.0, min(1.0, curvature_consistency))

            arc_confidence = (
                0.40 * (1.0 - straightness) +          # More curved is better
                0.30 * curvature_consistency +         # Consistent curvature
                0.30 * min(1.0, arc_length_um / 100)   # Longer arcs are more confident
            )
            arc_confidence = max(0.0, min(1.0, arc_confidence))

            # Create candidate dict (similar structure to ring candidates but marked as arc)
            arc_candidates.append({
                'outer': cnt,
                'inner': None,  # No inner contour for arcs
                'all_inner': [],
                'binary': binary,
                'touches_boundary': False,  # Will be updated below
                'boundary_edges': set(),
                'is_partial': True,
                'is_partial_only': True,
                'is_merged': False,
                'source_tiles': [(tile_x, tile_y)],
                'curvature_signature': curvature_sig,
                'orientation': float(angle),
                # Arc-specific fields
                'is_arc': True,
                'arc_length_um': float(arc_length_um),
                'avg_curvature': float(avg_curvature),
                'straightness': float(straightness),
                'estimated_diameter_um': float(estimated_diameter_um),
                'arc_confidence': float(arc_confidence),
            })

        logger.debug(f"Detected {len(arc_candidates)} open vessel structures (arcs) in candidate mode")
        return arc_candidates

    def _detect_cd31_tubular(
        self,
        cd31_channel: np.ndarray,
        sma_channel: np.ndarray,
        pixel_size_um: float = 0.22,
        intensity_percentile: float = 95,
        min_tubularity: float = 3.0,
        min_diameter_um: float = 3,
        max_diameter_um: float = 20,
    ) -> List[Dict]:
        """
        Detect capillaries as CD31+ tubular structures without SMA rings.

        Capillaries are CD31+ (endothelial marker) tubular structures that lack
        the smooth muscle actin (SMA) present in larger vessels. They are small
        (3-15 um diameter typically) and elongated (aspect ratio > 3).

        Algorithm:
        1. Threshold CD31 channel at intensity_percentile
        2. Find connected components
        3. For each component:
           - Fit ellipse to get major/minor axes
           - Calculate tubularity = major/minor (aspect ratio)
           - Filter: tubularity > min_tubularity
           - Filter: diameter in range [min, max]
           - Check SMA intensity is LOW (not a muscular vessel)
        4. Return candidates

        Args:
            cd31_channel: CD31 channel image (2D grayscale, uint8 or uint16)
            sma_channel: SMA channel image (2D grayscale, uint8 or uint16)
            pixel_size_um: Pixel size in microns (default: 0.22)
            intensity_percentile: Percentile for CD31 thresholding (default: 95)
            min_tubularity: Minimum aspect ratio to be considered tubular (default: 3.0)
            min_diameter_um: Minimum minor axis diameter in microns (default: 3)
            max_diameter_um: Maximum minor axis diameter in microns (default: 20)

        Returns:
            List of candidate dicts with:
            - 'outer': contour (Nx1x2 array)
            - 'detected_by': ['cd31']
            - 'vessel_type_hint': 'capillary'
            - 'is_tubular': True
            - 'tubularity': aspect_ratio (float)
            - 'minor_diameter_um': minor axis in microns
            - 'major_diameter_um': major axis in microns
            - 'cd31_mean_intensity': mean CD31 intensity in the region
            - 'sma_mean_intensity': mean SMA intensity in the region (should be low)
            - 'center': (cx, cy) center coordinates
            - 'orientation': angle in degrees
        """
        capillary_candidates = []

        # Normalize channels to uint8 if needed
        if cd31_channel.dtype == np.uint16:
            cd31_norm = (cd31_channel / 256).astype(np.uint8)
        else:
            cd31_norm = cd31_channel.astype(np.uint8)

        if sma_channel.dtype == np.uint16:
            sma_norm = (sma_channel / 256).astype(np.uint8)
        else:
            sma_norm = sma_channel.astype(np.uint8)

        # Step 1: Threshold CD31 channel at intensity_percentile
        cd31_nonzero = cd31_norm[cd31_norm > 0]
        if len(cd31_nonzero) == 0:
            logger.debug("CD31 channel has no signal above 0, skipping capillary detection")
            return []
        threshold_value = np.percentile(cd31_nonzero, intensity_percentile)
        threshold_value = max(threshold_value, 10)  # Minimum threshold to avoid noise
        _, cd31_binary = cv2.threshold(cd31_norm, threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological cleanup - remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cd31_binary = cv2.morphologyEx(cd31_binary, cv2.MORPH_OPEN, kernel)
        cd31_binary = cv2.morphologyEx(cd31_binary, cv2.MORPH_CLOSE, kernel)

        # Step 2: Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cd31_binary, connectivity=8
        )

        # Calculate SMA threshold - capillaries should have LOW SMA
        # Use the median of SMA in CD31+ regions as reference
        sma_in_cd31 = sma_norm[cd31_binary > 0]
        if len(sma_in_cd31) > 0:
            sma_low_threshold = np.percentile(sma_in_cd31, 50)  # Below median = low SMA
        else:
            sma_low_threshold = np.percentile(sma_norm, 50)

        # Convert diameter limits to pixels
        min_diameter_px = min_diameter_um / pixel_size_um
        max_diameter_px = max_diameter_um / pixel_size_um

        # Step 3: Process each connected component (skip background label 0)
        for label_idx in range(1, num_labels):
            # Get component mask
            component_mask = (labels == label_idx).astype(np.uint8)

            # Get component stats
            area = stats[label_idx, cv2.CC_STAT_AREA]
            cx, cy = centroids[label_idx]

            # Skip very small components (noise)
            min_area_px = np.pi * (min_diameter_px / 2) ** 2
            if area < min_area_px:
                continue

            # Find contour for this component
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)

            # Need at least 5 points to fit ellipse
            if len(cnt) < 5:
                continue

            # Fit ellipse to get dimensions and orientation
            try:
                ellipse = cv2.fitEllipse(cnt)
                (ecx, ecy), (minor_ax, major_ax), angle = ellipse
            except cv2.error:
                continue

            # Ensure major > minor (swap if needed)
            if minor_ax > major_ax:
                minor_ax, major_ax = major_ax, minor_ax
                angle = (angle + 90) % 180

            # Calculate tubularity (aspect ratio)
            tubularity = major_ax / max(minor_ax, 1)

            # Filter by tubularity (must be elongated)
            if tubularity < min_tubularity:
                continue

            # Filter by diameter (use minor axis as cross-sectional diameter)
            minor_diameter_um = minor_ax * pixel_size_um
            major_diameter_um = major_ax * pixel_size_um

            if minor_diameter_um < min_diameter_um or minor_diameter_um > max_diameter_um:
                continue

            # Step 4: Check SMA intensity is LOW
            # Get mean SMA intensity within the component
            sma_mean = np.mean(sma_norm[component_mask > 0])
            cd31_mean = np.mean(cd31_norm[component_mask > 0])

            # Capillaries should have low SMA (below threshold)
            if sma_mean > sma_low_threshold:
                # High SMA suggests this is a muscular vessel, not a capillary
                continue

            # Calculate confidence based on:
            # - How tubular (elongated) it is
            # - How low the SMA is
            # - How strong the CD31 signal is
            tubularity_score = min(1.0, (tubularity - min_tubularity) / 5.0)
            sma_score = 1.0 - (sma_mean / max(sma_low_threshold, 1))
            sma_score = max(0.0, min(1.0, sma_score))
            cd31_score = min(1.0, cd31_mean / 128.0)  # Normalize to 0-1

            detection_confidence = 0.4 * tubularity_score + 0.3 * sma_score + 0.3 * cd31_score
            detection_confidence = max(0.0, min(1.0, detection_confidence))

            # Create candidate dict
            capillary_candidates.append({
                'outer': cnt,
                'inner': None,  # Capillaries don't have distinct inner/outer walls
                'all_inner': [],
                'binary': component_mask,
                'detected_by': ['cd31'],
                'vessel_type_hint': 'capillary',
                'is_tubular': True,
                'tubularity': float(tubularity),
                'minor_diameter_um': float(minor_diameter_um),
                'major_diameter_um': float(major_diameter_um),
                'cd31_mean_intensity': float(cd31_mean),
                'sma_mean_intensity': float(sma_mean),
                'center': (float(ecx), float(ecy)),
                'orientation': float(angle),
                'area_px': float(area),
                'area_um2': float(area * pixel_size_um ** 2),
                'detection_confidence': float(detection_confidence),
                # Compatibility fields with ring detection
                'touches_boundary': False,
                'boundary_edges': set(),
                'is_partial': False,
                'is_partial_only': False,
                'is_merged': False,
                'is_arc': False,
            })

        logger.debug(
            f"Detected {len(capillary_candidates)} CD31+ tubular capillary candidates "
            f"(tubularity>{min_tubularity}, diameter {min_diameter_um}-{max_diameter_um}um)"
        )
        return capillary_candidates

    def _detect_all_markers_parallel(
        self,
        tile: np.ndarray,
        extra_channels: Dict[int, np.ndarray],
        pixel_size_um: float,
        channel_names: Dict[int, str],
        models: Dict[str, Any],
        n_workers: int = 3,
        tile_x: int = 0,
        tile_y: int = 0,
    ) -> List[Dict]:
        """
        Run SMA, CD31, and LYVE1 detection in parallel.

        Uses concurrent.futures.ThreadPoolExecutor for CPU parallelism.
        Detection is CPU-bound (thresholding, contours), not GPU-bound,
        so this provides significant speedup on multi-core systems.

        Args:
            tile: Primary tile (SMA channel as RGB or grayscale)
            extra_channels: Dict mapping channel index to channel data
                e.g., {0: nuclear, 1: sma, 2: cd31, 3: lyve1}
            pixel_size_um: Pixel size in microns
            channel_names: Dict mapping channel index to marker name
                e.g., {0: 'nuclear', 1: 'sma', 2: 'cd31', 3: 'lyve1'}
            models: Model dict (passed to segment() for SMA detection)
            n_workers: Number of parallel workers (default: 3, one per marker)

        Returns:
            Combined list of candidates from all markers, each with
            'detected_by' field indicating source marker(s)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        candidates = []

        # Build reverse lookup: marker name -> channel index
        name_to_idx = {v.lower(): k for k, v in channel_names.items()}

        # Get SMA channel (primary detection channel)
        sma_idx = name_to_idx.get('sma', 1)
        sma_channel = extra_channels.get(sma_idx) if sma_idx in extra_channels else tile

        # Get CD31 channel if available
        cd31_idx = name_to_idx.get('cd31')
        cd31_channel = extra_channels.get(cd31_idx) if cd31_idx is not None else None

        # Get LYVE1 channel if available
        lyve1_idx = name_to_idx.get('lyve1')
        lyve1_channel = extra_channels.get(lyve1_idx) if lyve1_idx is not None else None

        # Use ThreadPoolExecutor (not ProcessPoolExecutor) since:
        # 1. We're sharing numpy arrays (avoid serialization overhead)
        # 2. Detection is CPU-bound OpenCV operations (releases GIL)
        # 3. Shared memory access is thread-safe for read-only numpy arrays
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            # SMA ring detection (always run)
            # Note: segment() is the main SMA detection method
            futures[executor.submit(
                self.segment,
                tile,
                models,
                pixel_size_um,
                cd31_channel,  # CD31 for validation within SMA detection
                tile_x,
                tile_y,
            )] = 'sma'

            # CD31 tubular detection (for capillaries without SMA)
            if cd31_channel is not None:
                futures[executor.submit(
                    self._detect_cd31_tubular,
                    cd31_channel,
                    sma_channel if isinstance(sma_channel, np.ndarray) and sma_channel.ndim == 2 else (
                        np.mean(tile[:, :, :3], axis=2).astype(np.uint8) if tile.ndim == 3 else tile
                    ),
                    pixel_size_um,
                )] = 'cd31'

            # LYVE1 detection (for lymphatics)
            if lyve1_channel is not None:
                # Get 2D SMA for comparison
                sma_2d = sma_channel if isinstance(sma_channel, np.ndarray) and sma_channel.ndim == 2 else (
                    np.mean(tile[:, :, :3], axis=2).astype(np.uint8) if tile.ndim == 3 else tile
                )
                futures[executor.submit(
                    self._detect_lyve1_structures,
                    lyve1_channel,
                    cd31_channel,
                    sma_2d,
                    pixel_size_um,
                )] = 'lyve1'

            # Collect results as they complete
            for future in as_completed(futures):
                marker = futures[future]
                try:
                    result = future.result()
                    if result:
                        logger.debug(f"Parallel detection: {marker} returned {len(result)} candidates")
                        candidates.extend(result)
                except Exception as e:
                    logger.warning(f"{marker} detection failed in parallel mode: {e}")

        logger.info(
            f"Parallel multi-marker detection complete: {len(candidates)} total candidates "
            f"(workers={n_workers})"
        )
        return candidates

    def _analyze_boundary_contact(
        self,
        outer_contour: np.ndarray,
        inner_contour: np.ndarray,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """
        Analyze whether a vessel touches tile boundaries.

        Args:
            outer_contour: Outer vessel wall contour
            inner_contour: Inner lumen contour
            h: Tile height
            w: Tile width

        Returns:
            Dict with 'touches_boundary', 'edges', and 'is_partial' flags
        """
        margin = self.boundary_margin_px
        edges: Set[BoundaryEdge] = set()

        # Check outer contour points against boundaries
        points = outer_contour.reshape(-1, 2)

        if np.any(points[:, 1] <= margin):  # y near top
            edges.add(BoundaryEdge.TOP)
        if np.any(points[:, 1] >= h - margin):  # y near bottom
            edges.add(BoundaryEdge.BOTTOM)
        if np.any(points[:, 0] <= margin):  # x near left
            edges.add(BoundaryEdge.LEFT)
        if np.any(points[:, 0] >= w - margin):  # x near right
            edges.add(BoundaryEdge.RIGHT)

        # Also check inner contour for partial detection
        inner_points = inner_contour.reshape(-1, 2)
        if np.any(inner_points[:, 1] <= margin):
            edges.add(BoundaryEdge.TOP)
        if np.any(inner_points[:, 1] >= h - margin):
            edges.add(BoundaryEdge.BOTTOM)
        if np.any(inner_points[:, 0] <= margin):
            edges.add(BoundaryEdge.LEFT)
        if np.any(inner_points[:, 0] >= w - margin):
            edges.add(BoundaryEdge.RIGHT)

        touches_boundary = len(edges) > 0

        # Determine if this is a partial vessel (ring not complete)
        # A vessel is partial if significant portion of contour is at boundary
        boundary_point_count = 0
        total_points = len(points)

        for pt in points:
            x, y = pt
            if y <= margin or y >= h - margin or x <= margin or x >= w - margin:
                boundary_point_count += 1

        # More than 20% of points at boundary suggests partial vessel
        is_partial = boundary_point_count / max(total_points, 1) > 0.2

        return {
            'touches_boundary': touches_boundary,
            'edges': edges,
            'is_partial': is_partial,
        }

    def _detect_boundary_partial_vessels(
        self,
        contours: List[np.ndarray],
        hierarchy: np.ndarray,
        processed_contours: Set[int],
        binary: np.ndarray,
        h: int,
        w: int,
        tile_x: int,
        tile_y: int,
        pixel_size_um: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect partial vessels at tile boundaries that don't form complete rings.

        These are contours that touch the boundary and could be part of a vessel
        that continues in an adjacent tile.

        Args:
            contours: All contours found in the tile
            hierarchy: Contour hierarchy
            processed_contours: Set of contour indices already processed as rings
            binary: Binary mask of the tile
            h: Tile height
            w: Tile width
            tile_x: X coordinate of tile origin
            tile_y: Y coordinate of tile origin
            pixel_size_um: Pixel size in microns

        Returns:
            List of partial vessel candidate dicts
        """
        partial_candidates = []
        margin = self.boundary_margin_px

        for i, cnt in enumerate(contours):
            if i in processed_contours:
                continue

            # Check if this contour touches the boundary
            points = cnt.reshape(-1, 2)
            edges: Set[BoundaryEdge] = set()

            at_top = np.any(points[:, 1] <= margin)
            at_bottom = np.any(points[:, 1] >= h - margin)
            at_left = np.any(points[:, 0] <= margin)
            at_right = np.any(points[:, 0] >= w - margin)

            if at_top:
                edges.add(BoundaryEdge.TOP)
            if at_bottom:
                edges.add(BoundaryEdge.BOTTOM)
            if at_left:
                edges.add(BoundaryEdge.LEFT)
            if at_right:
                edges.add(BoundaryEdge.RIGHT)

            if not edges:
                continue  # Not at boundary

            # Check if contour is large enough to be a vessel candidate
            area = cv2.contourArea(cnt)
            min_area_px = (self.min_diameter_um / pixel_size_um) ** 2 * 0.1  # 10% of min diameter area
            if area < min_area_px:
                continue

            # Check shape - should be arc-like or curved (not straight line)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 10:
                continue

            # Calculate curvature signature for matching
            curvature_sig = self._compute_curvature_signature(cnt)

            # Estimate orientation from contour
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    orientation = ellipse[2]
                except cv2.error:
                    # Fall back to PCA-based orientation
                    orientation = self._estimate_orientation_pca(cnt)
            else:
                orientation = self._estimate_orientation_pca(cnt)

            # Get boundary points for matching
            boundary_points = self._extract_boundary_points(cnt, h, w, margin)

            # Store as partial vessel for potential merging
            partial_vessel = PartialVessel(
                tile_x=tile_x,
                tile_y=tile_y,
                contour=cnt,
                boundary_edges=edges,
                boundary_points=boundary_points,
                orientation=orientation,
                curvature_signature=curvature_sig,
                features={'area_px': float(area), 'perimeter_px': float(perimeter)},
                is_outer=True,  # Assume outer until matched
            )

            # Store in partial vessels cache
            tile_key = (tile_x, tile_y)
            if tile_key not in self._partial_vessels:
                self._partial_vessels[tile_key] = []
            self._partial_vessels[tile_key].append(partial_vessel)

            # Create candidate dict for this partial vessel
            partial_candidates.append({
                'outer': cnt,
                'inner': None,  # No inner contour for partial
                'all_inner': [],
                'binary': binary,
                'touches_boundary': True,
                'boundary_edges': edges,
                'is_partial': True,
                'is_partial_only': True,  # No complete ring structure
                'is_merged': False,
                'source_tiles': [(tile_x, tile_y)],
                'curvature_signature': curvature_sig,
                'orientation': orientation,
            })

        return partial_candidates

    def _detect_partial_vessels_from_binary(
        self,
        binary: np.ndarray,
        tile_x: int,
        tile_y: int,
        h: int,
        w: int,
        pixel_size_um: float,
    ) -> None:
        """
        Detect partial vessels from binary mask when no complete rings found.

        This handles edge cases where the tile only contains boundary-touching
        vessel segments without any complete ring structures.
        """
        margin = self.boundary_margin_px

        # Find all contours in binary
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            points = cnt.reshape(-1, 2)
            edges: Set[BoundaryEdge] = set()

            if np.any(points[:, 1] <= margin):
                edges.add(BoundaryEdge.TOP)
            if np.any(points[:, 1] >= h - margin):
                edges.add(BoundaryEdge.BOTTOM)
            if np.any(points[:, 0] <= margin):
                edges.add(BoundaryEdge.LEFT)
            if np.any(points[:, 0] >= w - margin):
                edges.add(BoundaryEdge.RIGHT)

            if not edges:
                continue

            area = cv2.contourArea(cnt)
            min_area_px = (self.min_diameter_um / pixel_size_um) ** 2 * 0.1
            if area < min_area_px:
                continue

            curvature_sig = self._compute_curvature_signature(cnt)
            orientation = self._estimate_orientation_pca(cnt)
            boundary_points = self._extract_boundary_points(cnt, h, w, margin)

            partial_vessel = PartialVessel(
                tile_x=tile_x,
                tile_y=tile_y,
                contour=cnt,
                boundary_edges=edges,
                boundary_points=boundary_points,
                orientation=orientation,
                curvature_signature=curvature_sig,
                features={'area_px': float(area)},
            )

            tile_key = (tile_x, tile_y)
            if tile_key not in self._partial_vessels:
                self._partial_vessels[tile_key] = []
            self._partial_vessels[tile_key].append(partial_vessel)

    def _detect_lyve1_structures(
        self,
        lyve1_channel: np.ndarray,
        cd31_channel: Optional[np.ndarray],
        sma_channel: np.ndarray,
        pixel_size_um: float = 0.22,
        intensity_percentile: float = 92,
        min_diameter_um: float = 10,
        max_diameter_um: float = 500,
    ) -> List[Dict]:
        """
        Detect lymphatics as LYVE1+ structures.

        Lymphatics are characterized by:
        - High LYVE1 expression (lymphatic endothelial marker)
        - Low or absent CD31 expression (distinguishes from blood vessels)
        - Irregular shapes (not perfectly circular like arteries)
        - Larger lumens relative to wall thickness
        - Collecting lymphatics may also be SMA+ (have smooth muscle)

        Algorithm:
        1. Threshold LYVE1 channel at intensity_percentile
        2. Morphological cleanup (open/close)
        3. Find connected components
        4. For each component:
           - Check size is in range
           - Calculate LYVE1 intensity score
           - Calculate CD31 intensity score (should be low for lymphatics)
           - Check SMA intensity - if high, it's a collecting lymphatic
           - Try to find inner contour (lumen) using contour hierarchy

        Args:
            lyve1_channel: LYVE1 channel image (grayscale or normalized)
            cd31_channel: Optional CD31 channel for blood vessel exclusion
            sma_channel: SMA channel for detecting collecting lymphatics
            pixel_size_um: Pixel size in microns for diameter calculation
            intensity_percentile: Percentile threshold for LYVE1 detection (default: 92)
            min_diameter_um: Minimum structure diameter in microns (default: 10)
            max_diameter_um: Maximum structure diameter in microns (default: 500)

        Returns:
            List of candidate dicts with:
            - 'outer': contour (outer boundary)
            - 'inner': contour (if lumen detected), else None
            - 'detected_by': ['lyve1']
            - 'vessel_type_hint': 'lymphatic' or 'collecting_lymphatic' (if SMA+)
            - 'lyve1_score': intensity score (0-1)
            - 'cd31_score': intensity score (0-1, should be low for lymphatics)
            - 'sma_score': intensity score (0-1, high indicates collecting lymphatic)
        """
        candidates = []

        # Convert LYVE1 channel to grayscale if needed
        if lyve1_channel.ndim == 3:
            lyve1_gray = np.mean(lyve1_channel[:, :, :3], axis=2).astype(np.float32)
        else:
            lyve1_gray = lyve1_channel.astype(np.float32)

        # Convert SMA channel to grayscale if needed
        if sma_channel.ndim == 3:
            sma_gray = np.mean(sma_channel[:, :, :3], axis=2).astype(np.float32)
        else:
            sma_gray = sma_channel.astype(np.float32)

        # Convert CD31 channel to grayscale if provided
        cd31_gray = None
        if cd31_channel is not None:
            if cd31_channel.ndim == 3:
                cd31_gray = np.mean(cd31_channel[:, :, :3], axis=2).astype(np.float32)
            else:
                cd31_gray = cd31_channel.astype(np.float32)

        h, w = lyve1_gray.shape[:2]

        # Calculate size thresholds in pixels
        min_diameter_px = min_diameter_um / pixel_size_um
        max_diameter_px = max_diameter_um / pixel_size_um
        min_area_px = np.pi * (min_diameter_px / 2) ** 2 * 0.3  # Allow some tolerance
        max_area_px = np.pi * (max_diameter_px / 2) ** 2 * 1.5  # Allow some tolerance

        # Step 1: Threshold LYVE1 channel at intensity percentile
        lyve1_nonzero = lyve1_gray[lyve1_gray > 0]
        if len(lyve1_nonzero) == 0:
            logger.debug("LYVE1 channel has no signal above 0, skipping lymphatic detection")
            return candidates

        threshold_value = np.percentile(lyve1_nonzero, intensity_percentile)
        if threshold_value <= 0:
            logger.debug("LYVE1 threshold value is 0, skipping lymphatic detection")
            return candidates

        binary = (lyve1_gray >= threshold_value).astype(np.uint8) * 255

        # Step 2: Morphological cleanup
        # Opening removes small noise (disconnected pixels)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # Closing fills small gaps within structures
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        # Step 3: Find connected components with hierarchy for inner/outer detection
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(contours) == 0:
            logger.debug("No contours found in LYVE1 channel")
            return candidates

        hierarchy = hierarchy[0]  # Shape: (N, 4) - [next, prev, child, parent]

        # Process contours - look for outer contours (parent = -1)
        for i, cnt in enumerate(contours):
            # Skip if this is an inner (child) contour - parent != -1
            if hierarchy[i][3] != -1:
                continue

            area = cv2.contourArea(cnt)

            # Check area bounds
            if area < min_area_px or area > max_area_px:
                continue

            # Calculate equivalent diameter
            equivalent_diameter_px = np.sqrt(4 * area / np.pi)
            equivalent_diameter_um = equivalent_diameter_px * pixel_size_um

            if equivalent_diameter_um < min_diameter_um or equivalent_diameter_um > max_diameter_um:
                continue

            # Create mask for this contour to calculate intensity scores
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mask_bool = mask > 0
            mask_pixel_count = np.sum(mask_bool)

            if mask_pixel_count == 0:
                continue

            # Step 4: Calculate intensity scores

            # LYVE1 score: mean intensity within the contour normalized to image max
            lyve1_max = lyve1_gray.max() if lyve1_gray.max() > 0 else 1.0
            lyve1_mean = np.mean(lyve1_gray[mask_bool])
            lyve1_score = float(lyve1_mean / lyve1_max)

            # CD31 score: should be low for lymphatics
            cd31_score = 0.0
            if cd31_gray is not None:
                cd31_max = cd31_gray.max() if cd31_gray.max() > 0 else 1.0
                cd31_mean = np.mean(cd31_gray[mask_bool])
                cd31_score = float(cd31_mean / cd31_max)

            # SMA score: high indicates collecting lymphatic with smooth muscle
            sma_max = sma_gray.max() if sma_gray.max() > 0 else 1.0
            sma_mean = np.mean(sma_gray[mask_bool])
            sma_score = float(sma_mean / sma_max)

            # Filter out likely blood vessels: high CD31 suggests blood vessel, not lymphatic
            # Lymphatics typically have CD31 score < 0.3
            if cd31_gray is not None and cd31_score > 0.5:
                logger.debug(
                    f"Skipping LYVE1+ structure with high CD31 score ({cd31_score:.2f}) - likely blood vessel"
                )
                continue

            # Step 5: Try to find inner contour (lumen)
            inner_contour = None
            child_idx = hierarchy[i][2]  # First child index

            if child_idx != -1:
                # Find the largest inner contour (likely the lumen)
                inner_candidates = []
                idx = child_idx
                while idx != -1:
                    inner_area = cv2.contourArea(contours[idx])
                    if inner_area > 0:
                        inner_candidates.append((idx, inner_area))
                    idx = hierarchy[idx][0]  # Next sibling

                if inner_candidates:
                    # Take the largest inner contour as the lumen
                    largest_inner_idx = max(inner_candidates, key=lambda x: x[1])[0]
                    inner_contour = contours[largest_inner_idx]

            # Determine vessel type hint
            # Collecting lymphatics have smooth muscle (SMA+) and are LYVE1+
            if sma_score > 0.3:
                vessel_type_hint = "collecting_lymphatic"
            else:
                vessel_type_hint = "lymphatic"

            # Calculate morphological features
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Fit ellipse for shape analysis (if enough points)
            orientation = 0.0
            aspect_ratio = 1.0
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (minor_ax, major_ax), orientation = ellipse
                    aspect_ratio = major_ax / minor_ax if minor_ax > 0 else 1.0
                except cv2.error:
                    pass

            # Get centroid
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = np.mean(cnt.reshape(-1, 2), axis=0)

            # Build candidate dict
            candidate = {
                'outer': cnt,
                'inner': inner_contour,
                'all_inner': [inner_contour] if inner_contour is not None else [],
                'detected_by': ['lyve1'],
                'vessel_type_hint': vessel_type_hint,
                'lyve1_score': lyve1_score,
                'cd31_score': cd31_score,
                'sma_score': sma_score,
                # Geometric features
                'area_px': float(area),
                'area_um2': float(area * pixel_size_um ** 2),
                'equivalent_diameter_um': float(equivalent_diameter_um),
                'perimeter_px': float(perimeter),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'orientation': float(orientation),
                'centroid': (float(cx), float(cy)),
                # Detection metadata
                'is_arc': False,
                'is_partial': False,
                'touches_boundary': False,
                'boundary_edges': set(),
                'binary': binary,
            }

            # If inner contour found, calculate lumen features
            if inner_contour is not None:
                inner_area = cv2.contourArea(inner_contour)
                inner_perimeter = cv2.arcLength(inner_contour, True)
                inner_circularity = 4 * np.pi * inner_area / (inner_perimeter ** 2) if inner_perimeter > 0 else 0

                candidate['lumen_area_px'] = float(inner_area)
                candidate['lumen_area_um2'] = float(inner_area * pixel_size_um ** 2)
                candidate['lumen_circularity'] = float(inner_circularity)
                candidate['wall_area_px'] = float(area - inner_area)
                candidate['wall_area_um2'] = float((area - inner_area) * pixel_size_um ** 2)

                # Lumen/wall ratio - lymphatics typically have larger lumens
                if area > 0:
                    candidate['lumen_wall_ratio'] = float(inner_area / area)
                else:
                    candidate['lumen_wall_ratio'] = 0.0

            candidates.append(candidate)

        logger.info(
            f"Detected {len(candidates)} LYVE1+ lymphatic structures "
            f"({sum(1 for c in candidates if c['vessel_type_hint'] == 'collecting_lymphatic')} collecting)"
        )

        return candidates

    def _compute_curvature_signature(self, contour: np.ndarray, num_samples: int = 32) -> np.ndarray:
        """
        Compute a curvature signature along the contour for matching.

        The curvature signature captures the local curvature at evenly spaced
        points along the contour, enabling matching of partial contours.

        Args:
            contour: Contour points (Nx1x2 array)
            num_samples: Number of curvature samples

        Returns:
            Array of curvature values
        """
        points = contour.reshape(-1, 2).astype(np.float64)
        n = len(points)

        if n < 5:
            return np.zeros(num_samples)

        # Resample contour to fixed number of points
        if n >= num_samples:
            indices = np.linspace(0, n - 1, num_samples, dtype=int)
            sampled = points[indices]
        else:
            # Interpolate to get more points
            t = np.linspace(0, 1, n)
            t_new = np.linspace(0, 1, num_samples)
            sampled = np.column_stack([
                np.interp(t_new, t, points[:, 0]),
                np.interp(t_new, t, points[:, 1]),
            ])

        # Compute curvature using finite differences
        # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.zeros(num_samples)

        for i in range(1, num_samples - 1):
            p_prev = sampled[i - 1]
            p_curr = sampled[i]
            p_next = sampled[i + 1]

            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]

            # Second derivatives (approximation)
            ddx = dx2 - dx1
            ddy = dy2 - dy1

            # Average first derivative
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

            denom = (dx ** 2 + dy ** 2) ** 1.5
            if denom > 1e-8:
                curvature[i] = abs(dx * ddy - dy * ddx) / denom

        # Fill edge values
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        return curvature

    def _estimate_orientation_pca(self, contour: np.ndarray) -> float:
        """
        Estimate contour orientation using PCA.

        Args:
            contour: Contour points

        Returns:
            Orientation angle in degrees (0-180)
        """
        points = contour.reshape(-1, 2).astype(np.float64)

        if len(points) < 2:
            return 0.0

        # Center the points
        mean = np.mean(points, axis=0)
        centered = points - mean

        # Compute covariance matrix
        cov = np.cov(centered.T)

        if cov.size == 1:
            return 0.0

        # Get principal direction
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal_idx = np.argmax(eigenvalues)
        principal_vec = eigenvectors[:, principal_idx]

        # Convert to angle
        angle = np.degrees(np.arctan2(principal_vec[1], principal_vec[0]))
        return float(angle % 180)  # Normalize to 0-180

    def _extract_boundary_points(
        self,
        contour: np.ndarray,
        h: int,
        w: int,
        margin: int,
    ) -> np.ndarray:
        """
        Extract points that lie on tile boundaries.

        Args:
            contour: Contour points
            h: Tile height
            w: Tile width
            margin: Boundary margin in pixels

        Returns:
            Array of boundary points
        """
        points = contour.reshape(-1, 2)
        boundary_mask = (
            (points[:, 0] <= margin) |
            (points[:, 0] >= w - margin) |
            (points[:, 1] <= margin) |
            (points[:, 1] >= h - margin)
        )
        return points[boundary_mask]

    # =====================================================================
    # Cross-Tile Vessel Merging Methods
    # =====================================================================

    def attempt_cross_tile_merge(
        self,
        current_tile_x: int,
        current_tile_y: int,
        tile_size: int,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Attempt to merge partial vessels from adjacent tiles.

        This should be called after processing all tiles that border the current
        tile to enable matching of partial vessels across boundaries.

        Args:
            current_tile_x: X coordinate of current tile origin
            current_tile_y: Y coordinate of current tile origin
            tile_size: Size of tiles in pixels
            candidates: Ring candidates from current tile

        Returns:
            Updated list of candidates with merged vessels added
        """
        if not self.cross_tile_config.enabled:
            return candidates

        merged_candidates = list(candidates)

        # Find adjacent tiles that have been processed
        adjacent_tiles = [
            (current_tile_x - tile_size, current_tile_y),  # Left
            (current_tile_x + tile_size, current_tile_y),  # Right
            (current_tile_x, current_tile_y - tile_size),  # Top
            (current_tile_x, current_tile_y + tile_size),  # Bottom
        ]

        # Get partial vessels from current tile
        current_partials = self._partial_vessels.get((current_tile_x, current_tile_y), [])

        for adj_tile in adjacent_tiles:
            if adj_tile not in self._partial_vessels:
                continue

            adj_partials = self._partial_vessels[adj_tile]

            # Determine which edge connects these tiles
            dx = adj_tile[0] - current_tile_x
            dy = adj_tile[1] - current_tile_y

            if dx < 0:  # Adjacent tile is to the left
                current_edge = BoundaryEdge.LEFT
                adj_edge = BoundaryEdge.RIGHT
            elif dx > 0:  # Adjacent tile is to the right
                current_edge = BoundaryEdge.RIGHT
                adj_edge = BoundaryEdge.LEFT
            elif dy < 0:  # Adjacent tile is above
                current_edge = BoundaryEdge.TOP
                adj_edge = BoundaryEdge.BOTTOM
            else:  # Adjacent tile is below
                current_edge = BoundaryEdge.BOTTOM
                adj_edge = BoundaryEdge.TOP

            # Try to match partial vessels
            for curr_partial in current_partials:
                if current_edge not in curr_partial.boundary_edges:
                    continue

                best_match = None
                best_score = 0.0

                for adj_partial in adj_partials:
                    if adj_edge not in adj_partial.boundary_edges:
                        continue

                    score = self._compute_match_score(
                        curr_partial, adj_partial,
                        current_tile_x, current_tile_y,
                        adj_tile[0], adj_tile[1],
                        tile_size
                    )

                    if score > best_score and score >= self.cross_tile_config.curvature_match_threshold:
                        best_score = score
                        best_match = adj_partial

                if best_match is not None:
                    # Merge the partial vessels
                    merged = self._merge_partial_vessels(
                        curr_partial, best_match,
                        current_tile_x, current_tile_y,
                        adj_tile[0], adj_tile[1],
                        tile_size
                    )

                    if merged is not None:
                        merged_candidates.append(merged)
                        logger.debug(
                            f"Merged partial vessels from tiles "
                            f"({current_tile_x}, {current_tile_y}) and {adj_tile} "
                            f"with score {best_score:.3f}"
                        )

        return merged_candidates

    def _compute_match_score(
        self,
        partial1: PartialVessel,
        partial2: PartialVessel,
        tile1_x: int,
        tile1_y: int,
        tile2_x: int,
        tile2_y: int,
        tile_size: int,
    ) -> float:
        """
        Compute a matching score between two partial vessels.

        The score considers:
        1. Position continuity at the boundary
        2. Orientation alignment
        3. Curvature signature correlation

        Args:
            partial1: First partial vessel
            partial2: Second partial vessel
            tile1_x, tile1_y: Origin of first tile
            tile2_x, tile2_y: Origin of second tile
            tile_size: Size of tiles

        Returns:
            Match score between 0 and 1 (higher is better match)
        """
        config = self.cross_tile_config

        # Convert boundary points to global coordinates
        bp1 = partial1.boundary_points.copy()
        bp1[:, 0] += tile1_x
        bp1[:, 1] += tile1_y

        bp2 = partial2.boundary_points.copy()
        bp2[:, 0] += tile2_x
        bp2[:, 1] += tile2_y

        if len(bp1) == 0 or len(bp2) == 0:
            return 0.0

        # 1. Position score: How close are the boundary points?
        min_distances = []
        for pt1 in bp1:
            distances = np.sqrt(np.sum((bp2 - pt1) ** 2, axis=1))
            min_distances.append(np.min(distances))

        mean_min_dist = np.mean(min_distances)
        position_score = max(0, 1 - mean_min_dist / config.position_tolerance_px)

        # 2. Orientation score: How well aligned are the orientations?
        angle_diff = abs(partial1.orientation - partial2.orientation)
        # Handle wrap-around (e.g., 170 and 10 are close)
        angle_diff = min(angle_diff, 180 - angle_diff)
        orientation_score = max(0, 1 - angle_diff / config.orientation_tolerance_deg)

        # 3. Curvature signature correlation
        sig1 = partial1.curvature_signature
        sig2 = partial2.curvature_signature

        if len(sig1) == len(sig2) and np.std(sig1) > 0 and np.std(sig2) > 0:
            # Try both forward and reverse correlation
            corr_fwd = np.corrcoef(sig1, sig2)[0, 1]
            corr_rev = np.corrcoef(sig1, sig2[::-1])[0, 1]
            curvature_score = max(corr_fwd, corr_rev)
            curvature_score = max(0, curvature_score)  # Clip negative correlations
        else:
            curvature_score = 0.5  # Neutral score if can't compute

        # Combine scores (weighted average)
        total_score = (
            0.4 * position_score +
            0.3 * orientation_score +
            0.3 * curvature_score
        )

        return total_score

    def _merge_partial_vessels(
        self,
        partial1: PartialVessel,
        partial2: PartialVessel,
        tile1_x: int,
        tile1_y: int,
        tile2_x: int,
        tile2_y: int,
        tile_size: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Merge two partial vessels into a combined ring candidate.

        Args:
            partial1: First partial vessel
            partial2: Second partial vessel
            tile1_x, tile1_y: Origin of first tile
            tile2_x, tile2_y: Origin of second tile
            tile_size: Size of tiles

        Returns:
            Merged ring candidate dict, or None if merge fails
        """
        # Convert contours to global coordinates
        contour1 = partial1.contour.copy().reshape(-1, 2)
        contour1[:, 0] += tile1_x
        contour1[:, 1] += tile1_y

        contour2 = partial2.contour.copy().reshape(-1, 2)
        contour2[:, 0] += tile2_x
        contour2[:, 1] += tile2_y

        # Concatenate contours
        # Find the closest endpoints to connect them properly
        merged_points = self._connect_contours(contour1, contour2)

        if merged_points is None or len(merged_points) < 5:
            return None

        # Reshape to OpenCV contour format
        merged_contour = merged_points.reshape(-1, 1, 2).astype(np.int32)

        # Try to fit an ellipse to validate it's a reasonable vessel shape
        try:
            ellipse = cv2.fitEllipse(merged_contour)
            center, axes, angle = ellipse
        except cv2.error:
            return None

        # Check if the merged shape is plausible
        major, minor = max(axes), min(axes)
        aspect_ratio = major / (minor + 1e-8)

        if aspect_ratio > self.max_aspect_ratio * 1.5:  # Allow slightly higher for merged
            return None

        # Create merged candidate
        # For merged vessels, we don't have separate inner/outer initially
        # The inner contour needs to be estimated or left as None
        merged_candidate = {
            'outer': merged_contour,
            'inner': None,  # Will need to be detected in the merged mask
            'all_inner': [],
            'binary': None,  # No binary mask for merged
            'touches_boundary': False,  # Merged vessel is now complete
            'boundary_edges': set(),
            'is_partial': False,
            'is_merged': True,
            'source_tiles': [(tile1_x, tile1_y), (tile2_x, tile2_y)],
            'merge_score': self._compute_match_score(
                partial1, partial2, tile1_x, tile1_y, tile2_x, tile2_y, tile_size
            ),
        }

        return merged_candidate

    def _connect_contours(
        self,
        contour1: np.ndarray,
        contour2: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Connect two contours at their closest points.

        Args:
            contour1: First contour points (Nx2)
            contour2: Second contour points (Mx2)

        Returns:
            Connected contour points, or None if connection fails
        """
        if len(contour1) < 2 or len(contour2) < 2:
            return None

        # Find the closest pair of points between contours
        min_dist = float('inf')
        best_i, best_j = 0, 0

        # Subsample for efficiency if contours are large
        step1 = max(1, len(contour1) // 50)
        step2 = max(1, len(contour2) // 50)

        for i in range(0, len(contour1), step1):
            for j in range(0, len(contour2), step2):
                dist = np.sum((contour1[i] - contour2[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_i, best_j = i, j

        # Reorder contours to connect at the closest points
        # Rotate contour1 so best_i is at the end
        rotated1 = np.concatenate([contour1[best_i+1:], contour1[:best_i+1]])

        # Rotate contour2 so best_j is at the start
        rotated2 = np.concatenate([contour2[best_j:], contour2[:best_j]])

        # Concatenate
        merged = np.concatenate([rotated1, rotated2])

        # Close the contour if endpoints are close
        start_to_end_dist = np.sqrt(np.sum((merged[0] - merged[-1]) ** 2))
        if start_to_end_dist < 20:  # Close enough to form a closed contour
            merged = np.concatenate([merged, merged[0:1]])

        return merged

    def get_partial_vessels(
        self,
        tile_x: Optional[int] = None,
        tile_y: Optional[int] = None,
    ) -> Dict[Tuple[int, int], List[PartialVessel]]:
        """
        Get stored partial vessels, optionally filtered by tile.

        Args:
            tile_x: Optional X coordinate to filter by
            tile_y: Optional Y coordinate to filter by

        Returns:
            Dict mapping tile coordinates to lists of PartialVessel objects
        """
        if tile_x is not None and tile_y is not None:
            key = (tile_x, tile_y)
            if key in self._partial_vessels:
                return {key: self._partial_vessels[key]}
            return {}
        return self._partial_vessels.copy()

    def clear_partial_vessels(
        self,
        tile_x: Optional[int] = None,
        tile_y: Optional[int] = None,
    ) -> None:
        """
        Clear stored partial vessels, optionally for specific tile only.

        Args:
            tile_x: Optional X coordinate to clear
            tile_y: Optional Y coordinate to clear
        """
        if tile_x is not None and tile_y is not None:
            key = (tile_x, tile_y)
            if key in self._partial_vessels:
                del self._partial_vessels[key]
        else:
            self._partial_vessels.clear()

    # =====================================================================
    # Multi-Marker Candidate Merging Methods
    # =====================================================================

    def _compute_iou(
        self,
        contour1: np.ndarray,
        contour2: np.ndarray,
        shape: Tuple[int, int],
    ) -> float:
        """
        Compute Intersection over Union (IoU) between two contours.

        Creates binary masks from the contours and computes the standard
        IoU metric: intersection_area / union_area.

        Args:
            contour1: First contour as Nx1x2 or Nx2 array
            contour2: Second contour as Mx1x2 or Mx2 array
            shape: Image shape (height, width) for mask creation

        Returns:
            IoU value between 0 and 1
        """
        h, w = shape

        # Ensure contours are in the correct format for cv2.drawContours
        if contour1.ndim == 2:
            contour1 = contour1.reshape(-1, 1, 2)
        if contour2.ndim == 2:
            contour2 = contour2.reshape(-1, 1, 2)

        # Ensure integer type
        contour1 = contour1.astype(np.int32)
        contour2 = contour2.astype(np.int32)

        # Create binary masks
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)

        cv2.drawContours(mask1, [contour1], 0, 1, -1)
        cv2.drawContours(mask2, [contour2], 0, 1, -1)

        # Compute intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection) / float(union)

    def _merge_candidates(
        self,
        candidates: List[Dict],
        iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Merge overlapping candidates from different detection channels.

        When detecting vessels with multiple markers (SMA, CD31, LYVE1), the same
        vessel may be detected by multiple channels. This method identifies
        overlapping detections using IoU and merges them into single candidates
        with combined marker information.

        Uses Union-Find (Disjoint Set Union) algorithm to group candidates that
        transitively overlap (e.g., if A overlaps B and B overlaps C, all three
        are merged even if A doesn't directly overlap C).

        Args:
            candidates: List of candidate dicts, each with:
                - 'outer': Outer contour (required)
                - 'detected_by': List of marker names that detected this candidate
                - Other standard candidate fields (inner, features, etc.)
            iou_threshold: Minimum IoU to consider candidates as same vessel

        Returns:
            Merged candidates list with:
                - Combined 'detected_by' lists (unique markers)
                - Best candidate contour selected (largest area)
                - Merged marker scores
                - 'merged_from_count': Number of original candidates merged
        """
        if len(candidates) <= 1:
            return candidates

        n = len(candidates)

        # Get image shape from first candidate with a contour
        shape = None
        for cand in candidates:
            outer = cand.get('outer')
            if outer is not None and len(outer) > 0:
                points = outer.reshape(-1, 2)
                max_y = int(np.max(points[:, 1])) + 100
                max_x = int(np.max(points[:, 0])) + 100
                # Ensure reasonable minimum size
                shape = (max(max_y, 1000), max(max_x, 1000))
                break

        if shape is None:
            logger.warning("No valid contours found for IoU computation")
            return candidates

        # Precompute areas for selecting best candidate
        areas = []
        for cand in candidates:
            outer = cand.get('outer')
            if outer is not None and len(outer) >= 3:
                areas.append(cv2.contourArea(outer.reshape(-1, 1, 2).astype(np.int32)))
            else:
                areas.append(0.0)

        # Build adjacency based on IoU > threshold
        # Use Union-Find for efficient grouping
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            """Find with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            """Union by rank."""
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Compute pairwise IoU and union overlapping candidates
        logger.debug(f"Computing pairwise IoU for {n} candidates...")
        for i in range(n):
            outer_i = candidates[i].get('outer')
            if outer_i is None or len(outer_i) < 3:
                continue

            for j in range(i + 1, n):
                outer_j = candidates[j].get('outer')
                if outer_j is None or len(outer_j) < 3:
                    continue

                # Quick bounding box check to skip distant candidates
                pts_i = outer_i.reshape(-1, 2)
                pts_j = outer_j.reshape(-1, 2)

                bbox_i = (pts_i[:, 0].min(), pts_i[:, 1].min(),
                          pts_i[:, 0].max(), pts_i[:, 1].max())
                bbox_j = (pts_j[:, 0].min(), pts_j[:, 1].min(),
                          pts_j[:, 0].max(), pts_j[:, 1].max())

                # Check if bounding boxes overlap
                if (bbox_i[2] < bbox_j[0] or bbox_j[2] < bbox_i[0] or
                    bbox_i[3] < bbox_j[1] or bbox_j[3] < bbox_i[1]):
                    continue  # No overlap possible

                # Compute IoU
                iou = self._compute_iou(outer_i, outer_j, shape)

                if iou >= iou_threshold:
                    union(i, j)
                    logger.debug(f"Candidates {i} and {j} merged with IoU={iou:.3f}")

        # Group candidates by their root parent
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        # Create merged candidates
        merged_candidates = []
        for root, group_indices in groups.items():
            if len(group_indices) == 1:
                # Single candidate, no merging needed
                cand = candidates[group_indices[0]].copy()
                cand['merged_from_count'] = 1
                if 'detected_by' not in cand:
                    cand['detected_by'] = ['unknown']
                merged_candidates.append(cand)
            else:
                # Multiple candidates to merge
                merged = self._merge_candidate_group(
                    [candidates[i] for i in group_indices],
                    [areas[i] for i in group_indices]
                )
                merged['merged_from_count'] = len(group_indices)
                merged_candidates.append(merged)

                logger.debug(
                    f"Merged {len(group_indices)} candidates into one "
                    f"(detected_by: {merged.get('detected_by', [])})"
                )

        logger.info(
            f"Candidate merging: {n} -> {len(merged_candidates)} "
            f"(merged {n - len(merged_candidates)} duplicates)"
        )

        return merged_candidates

    def _merge_candidate_group(
        self,
        group: List[Dict],
        areas: List[float],
    ) -> Dict:
        """
        Merge a group of overlapping candidates into a single candidate.

        Selects the best candidate based on area and combines metadata from
        all candidates in the group.

        Args:
            group: List of candidate dicts to merge
            areas: Corresponding areas for each candidate

        Returns:
            Merged candidate dict with combined metadata
        """
        # Select the candidate with the largest area as the base
        best_idx = np.argmax(areas)
        merged = group[best_idx].copy()

        # Combine 'detected_by' lists from all candidates
        all_markers: Set[str] = set()
        for cand in group:
            detected_by = cand.get('detected_by', [])
            if isinstance(detected_by, str):
                all_markers.add(detected_by)
            elif isinstance(detected_by, (list, tuple)):
                all_markers.update(detected_by)

        merged['detected_by'] = sorted(list(all_markers))

        # Combine marker scores if present
        combined_scores: Dict[str, float] = {}
        for cand in group:
            marker_scores = cand.get('marker_scores', {})
            for marker, score in marker_scores.items():
                if marker not in combined_scores:
                    combined_scores[marker] = score
                else:
                    # Take maximum score for each marker
                    combined_scores[marker] = max(combined_scores[marker], score)

        if combined_scores:
            merged['marker_scores'] = combined_scores

        # Track source candidates for debugging
        source_indices = []
        for cand in group:
            if 'source_index' in cand:
                source_indices.append(cand['source_index'])
        if source_indices:
            merged['merged_source_indices'] = source_indices

        # Keep track of all inner contours from all candidates
        all_inner: List[np.ndarray] = []
        for cand in group:
            inner = cand.get('inner')
            if inner is not None and len(inner) >= 3:
                all_inner.append(inner)
            cand_all_inner = cand.get('all_inner', [])
            for inner_cnt in cand_all_inner:
                if inner_cnt is not None and len(inner_cnt) >= 3:
                    all_inner.append(inner_cnt)

        if all_inner:
            # Select the inner contour with the largest area
            inner_areas = [cv2.contourArea(cnt.reshape(-1, 1, 2).astype(np.int32))
                          for cnt in all_inner]
            best_inner_idx = np.argmax(inner_areas)
            merged['inner'] = all_inner[best_inner_idx]
            merged['all_inner'] = all_inner

        # Mark as multi-marker merged
        merged['is_multi_marker_merged'] = True

        # Combine confidence scores (take maximum)
        confidences = [cand.get('detection_confidence', 0.0) for cand in group]
        if any(c > 0 for c in confidences):
            merged['detection_confidence'] = max(confidences)

        return merged

    def extract_features(
        self,
        ring_candidate: Dict[str, Any],
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract vessel-specific features from a ring candidate.

        Fits ellipses to outer and inner contours, measures wall thickness
        using distance transform and skeleton analysis.

        Enhanced: Also extracts boundary detection flags and merge tracking info.

        Args:
            ring_candidate: Dict with 'outer', 'inner', 'all_inner' contours
                Enhanced fields (optional):
                - 'touches_boundary': Whether vessel touches tile edge
                - 'boundary_edges': Set of BoundaryEdge values
                - 'is_partial': Whether vessel is partially visible
                - 'is_merged': Whether vessel was merged from multiple tiles
                - 'source_tiles': List of source tile coordinates
            tile: Original image for intensity measurements
            models: Dict of models (not used)
            pixel_size_um: Pixel size in microns
            cd31_channel: Optional CD31 channel for validation

        Returns:
            Dict of features, or None if candidate fails validation
            Enhanced output includes:
            - 'touches_boundary': bool
            - 'boundary_edges': list of edge names
            - 'is_partial': bool
            - 'is_merged': bool
            - 'source_tiles': list of [x, y] coordinates
            - 'detection_type': 'complete' | 'partial' | 'merged'
        """
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize

        outer = ring_candidate['outer']
        inner = ring_candidate['inner']
        binary = ring_candidate.get('binary')

        # Extract boundary tracking info (new fields)
        touches_boundary = ring_candidate.get('touches_boundary', False)
        boundary_edges = ring_candidate.get('boundary_edges', set())
        is_partial = ring_candidate.get('is_partial', False)
        is_merged = ring_candidate.get('is_merged', False)
        source_tiles = ring_candidate.get('source_tiles', [])
        is_partial_only = ring_candidate.get('is_partial_only', False)
        is_arc = ring_candidate.get('is_arc', False)

        # Handle arc candidates (open vessel structures detected in candidate mode)
        if is_arc:
            return self._extract_arc_vessel_features(
                ring_candidate, tile, pixel_size_um,
                touches_boundary, boundary_edges, source_tiles
            )

        # Handle partial-only vessels (no inner contour)
        if is_partial_only or inner is None:
            return self._extract_partial_vessel_features(
                ring_candidate, tile, pixel_size_um,
                touches_boundary, boundary_edges, is_partial, is_merged, source_tiles
            )

        # Need at least 5 points for ellipse fitting
        if len(outer) < 5 or len(inner) < 5:
            return None

        # Fit ellipses
        try:
            outer_ellipse = cv2.fitEllipse(outer)
            inner_ellipse = cv2.fitEllipse(inner)
        except cv2.error:
            return None

        # Extract ellipse parameters
        # fitEllipse returns: ((cx, cy), (minor_axis, major_axis), angle)
        (cx_out, cy_out), (minor_out, major_out), angle_out = outer_ellipse
        (cx_in, cy_in), (minor_in, major_in), angle_in = inner_ellipse

        # Calculate areas
        outer_area = cv2.contourArea(outer)
        inner_area = cv2.contourArea(inner)
        wall_area = outer_area - inner_area

        if wall_area <= 0 or inner_area <= 0:
            return None

        # Convert to diameters in microns
        outer_diameter_um = max(major_out, minor_out) * pixel_size_um
        inner_diameter_um = max(major_in, minor_in) * pixel_size_um

        # Size filtering
        if outer_diameter_um < self.min_diameter_um:
            return None
        if outer_diameter_um > self.max_diameter_um:
            return None

        # Aspect ratio filtering (exclude longitudinal sections)
        aspect_ratio_out = max(major_out, minor_out) / (min(major_out, minor_out) + 1e-8)
        if aspect_ratio_out > self.max_aspect_ratio:
            return None

        # Circularity filtering
        perimeter_out = cv2.arcLength(outer, True)
        circularity = 4 * np.pi * outer_area / (perimeter_out ** 2 + 1e-8)
        if circularity < self.min_circularity:
            return None

        # Create wall mask for measurements
        h, w = tile.shape[:2]
        wall_mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(wall_mask_temp, [outer], 0, 255, -1)
        cv2.drawContours(wall_mask_temp, [inner], 0, 0, -1)
        wall_region = wall_mask_temp > 0

        if wall_region.sum() == 0:
            return None

        # Calculate wall thickness using distance transform
        lumen_mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lumen_mask_temp, [inner], 0, 255, -1)

        # Distance from lumen boundary into wall
        dist_from_lumen = distance_transform_edt(~(lumen_mask_temp > 0))

        # Sample thickness at points along inner contour
        wall_thickness_values = []
        for pt in inner[::max(1, len(inner) // 36)]:  # Sample ~36 points
            px, py = pt[0]
            if 0 <= py < h and 0 <= px < w:
                if wall_region[py, px] or (lumen_mask_temp[py, px] > 0):
                    ray_dist = dist_from_lumen[py, px]
                    if ray_dist > 0:
                        wall_thickness_values.append(ray_dist * pixel_size_um)

        # Also measure using skeleton/medial axis approach
        try:
            skeleton = skeletonize(wall_region)
            skeleton_distances = dist_from_lumen[skeleton]
            if len(skeleton_distances) > 0:
                # Thickness is roughly 2x the distance to medial axis
                medial_thicknesses = skeleton_distances * 2 * pixel_size_um
                wall_thickness_values.extend(medial_thicknesses.tolist())
        except Exception as e:
            logger.debug(f"Skeleton analysis failed for wall thickness: {e}")

        if len(wall_thickness_values) < 5:
            return None

        wall_thicknesses = np.array(wall_thickness_values)
        wall_thickness_mean = float(np.mean(wall_thicknesses))
        wall_thickness_std = float(np.std(wall_thicknesses))
        wall_thickness_min = float(np.min(wall_thicknesses))
        wall_thickness_max = float(np.max(wall_thicknesses))
        wall_thickness_median = float(np.median(wall_thicknesses))

        # Wall thickness filtering
        if wall_thickness_mean < self.min_wall_thickness_um:
            return None

        # Calculate ring completeness (fraction of perimeter with SMA signal)
        # SIZE-ADAPTIVE SAMPLING: Adapt sample count based on vessel perimeter
        # This prevents size bias where small vessels get oversampled and large
        # vessels get undersampled with a fixed 72-point approach
        ring_points = 0
        ring_positive = 0

        if binary is not None:
            # Calculate adaptive sample count based on mid-wall perimeter
            avg_diameter = (max(major_out, minor_out) + max(major_in, minor_in)) / 2
            approx_perimeter_px = np.pi * avg_diameter

            # Target: 1 sample per 2 pixels of perimeter
            adaptive_samples = int(approx_perimeter_px / 2)
            # Clamp: min 24 (small capillaries), max 360 (large arteries)
            adaptive_samples = max(24, min(360, adaptive_samples))

            for theta in np.linspace(0, 2 * np.pi, adaptive_samples):
                # Mid-wall radius
                a_out, b_out = major_out / 2, minor_out / 2
                a_in, b_in = major_in / 2, minor_in / 2
                angle_out_rad = np.radians(angle_out)

                cos_t = np.cos(theta - angle_out_rad)
                sin_t = np.sin(theta - angle_out_rad)
                r_out = (a_out * b_out) / np.sqrt((b_out * cos_t) ** 2 + (a_out * sin_t) ** 2 + 1e-8)
                r_in = (a_in * b_in) / np.sqrt((b_in * cos_t) ** 2 + (a_in * sin_t) ** 2 + 1e-8)
                r_mid = (r_out + r_in) / 2

                # Point at mid-wall
                px = int(cx_out + r_mid * np.cos(theta))
                py = int(cy_out + r_mid * np.sin(theta))

                if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                    ring_points += 1
                    if binary[py, px] > 0:
                        ring_positive += 1

        ring_completeness = ring_positive / (ring_points + 1e-8)
        if ring_completeness < self.min_ring_completeness:
            return None

        # CD31 validation (if channel provided)
        cd31_validated = True
        cd31_score = 0.0
        if cd31_channel is not None:
            lumen_mask = np.zeros((h, w), dtype=np.uint8)
            wall_mask = np.zeros((h, w), dtype=np.uint8)

            cv2.drawContours(lumen_mask, [inner], 0, 255, -1)
            cv2.drawContours(wall_mask, [outer], 0, 255, -1)
            cv2.drawContours(wall_mask, [inner], 0, 0, -1)

            cd31_in_lumen = cd31_channel[lumen_mask > 0].mean() if (lumen_mask > 0).any() else 0
            cd31_in_wall = cd31_channel[wall_mask > 0].mean() if (wall_mask > 0).any() else 0

            # CD31 should be at lumen boundary, not in wall
            cd31_score = float(cd31_in_lumen / (cd31_in_wall + 1e-8))
            cd31_validated = cd31_in_lumen > cd31_in_wall * 0.8  # Some tolerance

        # Auto-classify vessel type by size (rule-based)
        # This can be overridden by ML classification in run_segmentation.py
        vessel_type = 'unknown'
        vessel_type_confidence = 0.0
        classification_method = 'none'
        if self.classify_vessel_types:
            if outer_diameter_um < 10:
                vessel_type = 'capillary'
                vessel_type_confidence = 0.8
            elif outer_diameter_um < 100:
                vessel_type = 'arteriole'
                vessel_type_confidence = 0.7
            else:
                vessel_type = 'artery'
                vessel_type_confidence = 0.6
            classification_method = 'rule_based'

        # Determine confidence level (categorical)
        if ring_completeness > 0.8 and circularity > 0.6 and aspect_ratio_out < 2.0:
            confidence = 'high'
        elif ring_completeness > 0.6 and circularity > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Calculate detection_confidence (continuous 0-1 score)
        # Based on: ring_completeness, circularity, wall_uniformity, aspect_ratio proximity to 1.0
        wall_uniformity = 1.0 - (wall_thickness_std / (wall_thickness_mean + 1e-8))
        wall_uniformity = max(0.0, min(1.0, wall_uniformity))  # Clamp to [0, 1]

        aspect_ratio_score = 1.0 - min(1.0, (aspect_ratio_out - 1.0) / 5.0)  # Closer to 1.0 is better

        detection_confidence = (
            0.30 * ring_completeness +           # Ring completeness (0-1)
            0.25 * circularity +                 # Circularity (0-1)
            0.25 * wall_uniformity +             # Wall thickness uniformity (0-1)
            0.20 * aspect_ratio_score            # Aspect ratio score (0-1)
        )
        detection_confidence = max(0.0, min(1.0, detection_confidence))  # Clamp to [0, 1]

        # Determine detection type
        if is_merged:
            detection_type = 'merged'
        elif is_partial:
            detection_type = 'partial'
        else:
            detection_type = 'complete'

        return {
            # Contours (needed for IoU-based merge in multi-scale detection)
            'outer': outer,
            'inner': inner,
            # Diameters
            'outer_diameter_um': float(outer_diameter_um),
            'inner_diameter_um': float(inner_diameter_um),
            'major_axis_um': float(max(major_out, minor_out) * pixel_size_um),
            'minor_axis_um': float(min(major_out, minor_out) * pixel_size_um),
            # Wall thickness measurements
            'wall_thickness_mean_um': wall_thickness_mean,
            'wall_thickness_median_um': wall_thickness_median,
            'wall_thickness_std_um': wall_thickness_std,
            'wall_thickness_min_um': wall_thickness_min,
            'wall_thickness_max_um': wall_thickness_max,
            # Areas
            'lumen_area_um2': float(inner_area * pixel_size_um ** 2),
            'wall_area_um2': float(wall_area * pixel_size_um ** 2),
            'outer_area_um2': float(outer_area * pixel_size_um ** 2),
            # Shape metrics
            'orientation_deg': float(angle_out),
            'aspect_ratio': float(aspect_ratio_out),
            'circularity': float(circularity),
            'ring_completeness': float(ring_completeness),
            # Validation
            'cd31_validated': cd31_validated,
            'cd31_score': cd31_score,
            # Classification
            'vessel_type': vessel_type,
            'vessel_type_confidence': vessel_type_confidence,
            'classification_method': classification_method,
            'confidence': confidence,
            # Detection confidence score (0-1) based on multiple metrics
            'detection_confidence': float(detection_confidence),
            'wall_uniformity': float(wall_uniformity),
            # Centers
            'outer_center': [float(cx_out), float(cy_out)],
            'inner_center': [float(cx_in), float(cy_in)],
            # Boundary detection tracking (new fields)
            'touches_boundary': touches_boundary,
            'boundary_edges': [e.value for e in boundary_edges] if boundary_edges else [],
            'is_partial': is_partial,
            'is_merged': is_merged,
            'source_tiles': [[t[0], t[1]] for t in source_tiles] if source_tiles else [],
            'detection_type': detection_type,
            # Candidate mode flag
            'candidate_mode': self.candidate_mode,
        }

    def _extract_partial_vessel_features(
        self,
        ring_candidate: Dict[str, Any],
        tile: np.ndarray,
        pixel_size_um: float,
        touches_boundary: bool,
        boundary_edges: Set[BoundaryEdge],
        is_partial: bool,
        is_merged: bool,
        source_tiles: List[Tuple[int, int]],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for partial vessels that lack complete ring structure.

        Partial vessels are detected at tile boundaries where only part of
        the vessel wall is visible. This method extracts limited features
        based on the available contour.

        Args:
            ring_candidate: Dict with 'outer' contour (no inner)
            tile: Original image
            pixel_size_um: Pixel size in microns
            touches_boundary: Whether vessel touches tile edge
            boundary_edges: Set of boundary edges touched
            is_partial: Whether vessel is partial
            is_merged: Whether vessel was merged
            source_tiles: Source tile coordinates

        Returns:
            Dict of partial vessel features, or None if invalid
        """
        outer = ring_candidate['outer']

        if len(outer) < 5:
            return None

        # Basic contour measurements
        area = cv2.contourArea(outer)
        perimeter = cv2.arcLength(outer, True)

        if area < 50:  # Too small
            return None

        # Try to fit ellipse for shape estimation
        try:
            ellipse = cv2.fitEllipse(outer)
            (cx, cy), (minor_ax, major_ax), angle = ellipse
            aspect_ratio = max(major_ax, minor_ax) / (min(major_ax, minor_ax) + 1e-8)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        except cv2.error:
            # Fall back to moments-based center
            M = cv2.moments(outer)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                pts = outer.reshape(-1, 2)
                cx, cy = np.mean(pts, axis=0)
            major_ax = minor_ax = np.sqrt(area / np.pi) * 2
            angle = 0
            aspect_ratio = 1.0
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)

        # Estimate outer diameter from area (assuming roughly circular)
        estimated_diameter_um = 2 * np.sqrt(area / np.pi) * pixel_size_um

        # Partial vessels have lower confidence
        confidence = 'low'

        # Curvature info for matching
        curvature_sig = ring_candidate.get('curvature_signature')
        orientation = ring_candidate.get('orientation', angle)

        # Calculate detection_confidence for partial vessels
        # Based only on available metrics (no ring completeness or wall uniformity)
        aspect_ratio_score = 1.0 - min(1.0, (aspect_ratio - 1.0) / 5.0)
        detection_confidence = (
            0.40 * circularity +                 # Circularity is the main indicator
            0.30 * aspect_ratio_score +          # Aspect ratio score
            0.30 * 0.3                           # Partial penalty (assume 30% completeness)
        )
        detection_confidence = max(0.0, min(1.0, detection_confidence))

        return {
            # Contours (needed for IoU-based merge in multi-scale detection)
            'outer': outer,
            'inner': None,
            # Estimated measurements (marked as estimates)
            'outer_diameter_um': float(estimated_diameter_um),
            'inner_diameter_um': None,  # Unknown for partial
            'major_axis_um': float(max(major_ax, minor_ax) * pixel_size_um),
            'minor_axis_um': float(min(major_ax, minor_ax) * pixel_size_um),
            # Wall thickness not measurable for partial
            'wall_thickness_mean_um': None,
            'wall_thickness_median_um': None,
            'wall_thickness_std_um': None,
            'wall_thickness_min_um': None,
            'wall_thickness_max_um': None,
            # Areas
            'lumen_area_um2': None,  # Unknown
            'wall_area_um2': float(area * pixel_size_um ** 2),  # Visible wall area
            'outer_area_um2': float(area * pixel_size_um ** 2),
            # Shape metrics
            'orientation_deg': float(orientation if orientation else angle),
            'aspect_ratio': float(aspect_ratio),
            'circularity': float(circularity),
            'ring_completeness': 0.0,  # Incomplete ring
            # Validation
            'cd31_validated': None,  # Cannot validate partial
            'cd31_score': None,
            # Classification
            'vessel_type': 'unknown',
            'vessel_type_confidence': 0.0,
            'classification_method': 'none',
            'confidence': confidence,
            # Detection confidence score (0-1)
            'detection_confidence': float(detection_confidence),
            'wall_uniformity': None,  # Cannot measure for partial
            # Centers
            'outer_center': [float(cx), float(cy)],
            'inner_center': None,  # Unknown
            # Boundary detection tracking
            'touches_boundary': touches_boundary,
            'boundary_edges': [e.value for e in boundary_edges] if boundary_edges else [],
            'is_partial': True,
            'is_merged': is_merged,
            'source_tiles': [[t[0], t[1]] for t in source_tiles] if source_tiles else [],
            'detection_type': 'partial',
            # Partial-specific fields
            'is_partial_only': True,
            'curvature_signature': curvature_sig.tolist() if curvature_sig is not None else None,
            'awaiting_merge': True,  # Flag for downstream processing
            # Candidate mode flag
            'candidate_mode': self.candidate_mode,
        }

    def _extract_arc_vessel_features(
        self,
        ring_candidate: Dict[str, Any],
        tile: np.ndarray,
        pixel_size_um: float,
        touches_boundary: bool,
        boundary_edges: Set[BoundaryEdge],
        source_tiles: List[Tuple[int, int]],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for arc/curve vessel candidates (open structures).

        Arc candidates are detected in candidate mode and represent vessel
        cross-sections that don't form complete rings but have vessel-like
        curvature characteristics.

        Args:
            ring_candidate: Dict with arc-specific fields
            tile: Original image
            pixel_size_um: Pixel size in microns
            touches_boundary: Whether arc touches tile edge
            boundary_edges: Set of boundary edges touched
            source_tiles: Source tile coordinates

        Returns:
            Dict of arc vessel features, or None if invalid
        """
        outer = ring_candidate['outer']

        if len(outer) < 5:
            return None

        # Get arc-specific fields
        arc_length_um = ring_candidate.get('arc_length_um', 0)
        avg_curvature = ring_candidate.get('avg_curvature', 0)
        straightness = ring_candidate.get('straightness', 1.0)
        estimated_diameter_um = ring_candidate.get('estimated_diameter_um', 0)
        arc_confidence = ring_candidate.get('arc_confidence', 0)
        curvature_sig = ring_candidate.get('curvature_signature')
        orientation = ring_candidate.get('orientation', 0)

        # Basic contour measurements
        area = cv2.contourArea(outer)
        perimeter = cv2.arcLength(outer, False)  # Open contour

        # Try to fit ellipse for shape estimation
        try:
            ellipse = cv2.fitEllipse(outer)
            (cx, cy), (minor_ax, major_ax), angle = ellipse
            aspect_ratio = max(major_ax, minor_ax) / (min(major_ax, minor_ax) + 1e-8)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        except cv2.error:
            # Fall back to moments-based center
            M = cv2.moments(outer)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                pts = outer.reshape(-1, 2)
                cx, cy = np.mean(pts, axis=0)
            major_ax = minor_ax = np.sqrt(area / np.pi) * 2 if area > 0 else 10
            angle = orientation
            aspect_ratio = 1.0
            circularity = 0.5  # Default for open contours

        # Arc vessels have low categorical confidence but may have good detection_confidence
        confidence = 'low'

        # Detection confidence for arcs is based on the arc_confidence calculated during detection
        detection_confidence = arc_confidence

        return {
            # Contours (needed for IoU-based merge in multi-scale detection)
            'outer': outer,
            'inner': None,
            # Estimated measurements (marked as estimates for arcs)
            'outer_diameter_um': float(estimated_diameter_um),
            'inner_diameter_um': None,  # Unknown for arcs
            'major_axis_um': float(max(major_ax, minor_ax) * pixel_size_um),
            'minor_axis_um': float(min(major_ax, minor_ax) * pixel_size_um),
            # Wall thickness not measurable for arcs
            'wall_thickness_mean_um': None,
            'wall_thickness_median_um': None,
            'wall_thickness_std_um': None,
            'wall_thickness_min_um': None,
            'wall_thickness_max_um': None,
            # Areas
            'lumen_area_um2': None,  # Unknown
            'wall_area_um2': float(area * pixel_size_um ** 2),  # Visible area
            'outer_area_um2': float(area * pixel_size_um ** 2),
            # Shape metrics
            'orientation_deg': float(angle),
            'aspect_ratio': float(aspect_ratio),
            'circularity': float(circularity),
            'ring_completeness': 0.0,  # Not a complete ring
            # Validation
            'cd31_validated': None,  # Cannot validate arcs
            'cd31_score': None,
            # Classification
            'vessel_type': 'unknown',
            'vessel_type_confidence': 0.0,
            'classification_method': 'none',
            'confidence': confidence,
            # Detection confidence score (0-1)
            'detection_confidence': float(detection_confidence),
            'wall_uniformity': None,  # Cannot measure for arcs
            # Centers
            'outer_center': [float(cx), float(cy)],
            'inner_center': None,  # Unknown
            # Boundary detection tracking
            'touches_boundary': touches_boundary,
            'boundary_edges': [e.value for e in boundary_edges] if boundary_edges else [],
            'is_partial': True,
            'is_merged': False,
            'source_tiles': [[t[0], t[1]] for t in source_tiles] if source_tiles else [],
            'detection_type': 'arc',
            # Arc-specific fields
            'is_arc': True,
            'arc_length_um': float(arc_length_um),
            'avg_curvature': float(avg_curvature),
            'straightness': float(straightness),
            'curvature_signature': curvature_sig.tolist() if curvature_sig is not None else None,
            # Candidate mode flag
            'candidate_mode': self.candidate_mode,
        }

    def extract_candidate_features(
        self,
        ring_candidate: Dict[str, Any],
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract full vessel features from a ring candidate WITHOUT filtering.

        This is the candidate mode version of extract_features(). It extracts
        ALL features for every candidate regardless of whether it would pass
        filtering criteria. Each candidate is marked with:
        - 'is_vessel': True/False (whether it would pass filtering)
        - 'rejection_reasons': List of reasons why filtering would reject it

        This is essential for RF training where the classifier needs to learn
        from both positive (vessel) and negative (non-vessel) examples.

        The full 2326 features (22 morph + 256 SAM2 + 2048 ResNet) are added by detect().

        Args:
            ring_candidate: Dict with 'outer', 'inner', 'all_inner' contours
            tile: Original image for intensity measurements
            models: Dict of models (not used)
            pixel_size_um: Pixel size in microns
            cd31_channel: Optional CD31 channel for validation

        Returns:
            Dict of features with 'is_vessel' and 'rejection_reasons' fields,
            or None only if ellipse fitting completely fails
        """
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize

        outer = ring_candidate['outer']
        inner = ring_candidate['inner']
        binary = ring_candidate.get('binary')

        rejection_reasons = []
        is_vessel = True

        touches_boundary = ring_candidate.get('touches_boundary', False)
        boundary_edges = ring_candidate.get('boundary_edges', set())
        is_partial = ring_candidate.get('is_partial', False)
        is_merged = ring_candidate.get('is_merged', False)
        source_tiles = ring_candidate.get('source_tiles', [])
        is_partial_only = ring_candidate.get('is_partial_only', False)

        if is_partial_only or inner is None:
            partial_feats = self._extract_partial_vessel_features(
                ring_candidate, tile, pixel_size_um,
                touches_boundary, boundary_edges, is_partial, is_merged, source_tiles
            )
            if partial_feats is not None:
                partial_feats['is_vessel'] = False
                partial_feats['rejection_reasons'] = ['partial_only_no_lumen']
            return partial_feats

        if len(outer) < 5 or len(inner) < 5:
            return None

        try:
            outer_ellipse = cv2.fitEllipse(outer)
            inner_ellipse = cv2.fitEllipse(inner)
        except cv2.error:
            return None

        (cx_out, cy_out), (minor_out, major_out), angle_out = outer_ellipse
        (cx_in, cy_in), (minor_in, major_in), angle_in = inner_ellipse

        outer_area = cv2.contourArea(outer)
        inner_area = cv2.contourArea(inner)
        wall_area = outer_area - inner_area

        if wall_area <= 0 or inner_area <= 0:
            rejection_reasons.append('invalid_wall_area')
            is_vessel = False
            wall_area = max(wall_area, 1)
            inner_area = max(inner_area, 1)

        outer_diameter_um = max(major_out, minor_out) * pixel_size_um
        inner_diameter_um = max(major_in, minor_in) * pixel_size_um

        # Check against original strict thresholds (default values)
        if outer_diameter_um < 10:
            rejection_reasons.append(f'too_small:{outer_diameter_um:.1f}um')
            is_vessel = False
        if outer_diameter_um > 1000:
            rejection_reasons.append(f'too_large:{outer_diameter_um:.1f}um')
            is_vessel = False

        aspect_ratio_out = max(major_out, minor_out) / (min(major_out, minor_out) + 1e-8)
        if aspect_ratio_out > 4.0:
            rejection_reasons.append(f'aspect_ratio:{aspect_ratio_out:.2f}')
            is_vessel = False

        perimeter_out = cv2.arcLength(outer, True)
        circularity = 4 * np.pi * outer_area / (perimeter_out ** 2 + 1e-8)
        if circularity < 0.3:
            rejection_reasons.append(f'circularity:{circularity:.2f}')
            is_vessel = False

        h, w = tile.shape[:2]
        wall_mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(wall_mask_temp, [outer], 0, 255, -1)
        cv2.drawContours(wall_mask_temp, [inner], 0, 0, -1)
        wall_region = wall_mask_temp > 0

        if wall_region.sum() == 0:
            rejection_reasons.append('empty_wall_region')
            is_vessel = False

        lumen_mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lumen_mask_temp, [inner], 0, 255, -1)
        dist_from_lumen = distance_transform_edt(~(lumen_mask_temp > 0))

        wall_thickness_values = []
        for pt in inner[::max(1, len(inner) // 36)]:
            px, py = pt[0]
            if 0 <= py < h and 0 <= px < w:
                if wall_region[py, px] or (lumen_mask_temp[py, px] > 0):
                    ray_dist = dist_from_lumen[py, px]
                    if ray_dist > 0:
                        wall_thickness_values.append(ray_dist * pixel_size_um)

        try:
            if wall_region.sum() > 0:
                skeleton = skeletonize(wall_region)
                skeleton_distances = dist_from_lumen[skeleton]
                if len(skeleton_distances) > 0:
                    medial_thicknesses = skeleton_distances * 2 * pixel_size_um
                    wall_thickness_values.extend(medial_thicknesses.tolist())
        except Exception:
            pass

        if len(wall_thickness_values) >= 5:
            wall_thicknesses = np.array(wall_thickness_values)
            wall_thickness_mean = float(np.mean(wall_thicknesses))
            wall_thickness_std = float(np.std(wall_thicknesses))
            wall_thickness_min = float(np.min(wall_thicknesses))
            wall_thickness_max = float(np.max(wall_thicknesses))
            wall_thickness_median = float(np.median(wall_thicknesses))
        else:
            rejection_reasons.append('insufficient_wall_samples')
            is_vessel = False
            estimated_thickness = (outer_diameter_um - inner_diameter_um) / 2
            wall_thickness_mean = estimated_thickness
            wall_thickness_std = 0.0
            wall_thickness_min = estimated_thickness
            wall_thickness_max = estimated_thickness
            wall_thickness_median = estimated_thickness

        if wall_thickness_mean < 2.0:
            rejection_reasons.append(f'thin_wall:{wall_thickness_mean:.2f}um')
            is_vessel = False

        ring_points = 0
        ring_positive = 0
        if binary is not None:
            for theta in np.linspace(0, 2 * np.pi, 72):
                a_out, b_out = major_out / 2, minor_out / 2
                a_in, b_in = major_in / 2, minor_in / 2
                angle_out_rad = np.radians(angle_out)
                cos_t = np.cos(theta - angle_out_rad)
                sin_t = np.sin(theta - angle_out_rad)
                r_out = (a_out * b_out) / np.sqrt((b_out * cos_t) ** 2 + (a_out * sin_t) ** 2 + 1e-8)
                r_in = (a_in * b_in) / np.sqrt((b_in * cos_t) ** 2 + (a_in * sin_t) ** 2 + 1e-8)
                r_mid = (r_out + r_in) / 2
                px = int(cx_out + r_mid * np.cos(theta))
                py = int(cy_out + r_mid * np.sin(theta))
                if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                    ring_points += 1
                    if binary[py, px] > 0:
                        ring_positive += 1

        ring_completeness = ring_positive / (ring_points + 1e-8)
        if ring_completeness < 0.5:
            rejection_reasons.append(f'incomplete_ring:{ring_completeness:.2f}')
            is_vessel = False

        cd31_validated = True
        cd31_score = 0.0
        if cd31_channel is not None:
            lumen_mask = np.zeros((h, w), dtype=np.uint8)
            wall_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(lumen_mask, [inner], 0, 255, -1)
            cv2.drawContours(wall_mask, [outer], 0, 255, -1)
            cv2.drawContours(wall_mask, [inner], 0, 0, -1)
            cd31_in_lumen = cd31_channel[lumen_mask > 0].mean() if (lumen_mask > 0).any() else 0
            cd31_in_wall = cd31_channel[wall_mask > 0].mean() if (wall_mask > 0).any() else 0
            cd31_score = float(cd31_in_lumen / (cd31_in_wall + 1e-8))
            cd31_validated = cd31_in_lumen > cd31_in_wall * 0.8

        vessel_type = 'unknown'
        vessel_type_confidence = 0.0
        classification_method = 'none'
        if self.classify_vessel_types:
            if outer_diameter_um < 10:
                vessel_type = 'capillary'
                vessel_type_confidence = 0.8
            elif outer_diameter_um < 100:
                vessel_type = 'arteriole'
                vessel_type_confidence = 0.7
            else:
                vessel_type = 'artery'
                vessel_type_confidence = 0.6
            classification_method = 'rule_based'

        if ring_completeness > 0.8 and circularity > 0.6 and aspect_ratio_out < 2.0:
            confidence = 'high'
        elif ring_completeness > 0.6 and circularity > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'

        if is_merged:
            detection_type = 'merged'
        elif is_partial:
            detection_type = 'partial'
        else:
            detection_type = 'complete'

        return {
            # Contours (needed for IoU-based merge in multi-scale detection)
            'outer': outer,
            'inner': inner,
            # Measurements
            'outer_diameter_um': float(outer_diameter_um),
            'inner_diameter_um': float(inner_diameter_um),
            'major_axis_um': float(max(major_out, minor_out) * pixel_size_um),
            'minor_axis_um': float(min(major_out, minor_out) * pixel_size_um),
            'wall_thickness_mean_um': wall_thickness_mean,
            'wall_thickness_median_um': wall_thickness_median,
            'wall_thickness_std_um': wall_thickness_std,
            'wall_thickness_min_um': wall_thickness_min,
            'wall_thickness_max_um': wall_thickness_max,
            'lumen_area_um2': float(inner_area * pixel_size_um ** 2),
            'wall_area_um2': float(wall_area * pixel_size_um ** 2),
            'outer_area_um2': float(outer_area * pixel_size_um ** 2),
            'orientation_deg': float(angle_out),
            'aspect_ratio': float(aspect_ratio_out),
            'circularity': float(circularity),
            'ring_completeness': float(ring_completeness),
            'cd31_validated': cd31_validated,
            'cd31_score': cd31_score,
            'vessel_type': vessel_type,
            'vessel_type_confidence': vessel_type_confidence,
            'classification_method': classification_method,
            'confidence': confidence,
            'outer_center': [float(cx_out), float(cy_out)],
            'inner_center': [float(cx_in), float(cy_in)],
            'touches_boundary': touches_boundary,
            'boundary_edges': [e.value for e in boundary_edges] if boundary_edges else [],
            'is_partial': is_partial,
            'is_merged': is_merged,
            'source_tiles': [[t[0], t[1]] for t in source_tiles] if source_tiles else [],
            'detection_type': detection_type,
            # Candidate mode fields - essential for RF training
            'is_vessel': is_vessel,
            'rejection_reasons': rejection_reasons,
            'candidate_mode': True,
        }

    def filter(
        self,
        masks: List[np.ndarray],
        features: List[Dict[str, Any]],
        pixel_size_um: float,
    ) -> List[Detection]:
        """
        Filter candidates based on extracted features.

        Note: For VesselStrategy, most filtering happens during extract_features().
        This method creates Detection objects from valid candidates.

        Args:
            masks: List of wall masks
            features: List of feature dicts from extract_features()
            pixel_size_um: Pixel size in microns

        Returns:
            List of Detection objects
        """
        detections = []

        for i, (mask, feat) in enumerate(zip(masks, features)):
            if feat is None:
                continue

            # Get centroid from outer center
            center = feat.get('outer_center', [0, 0])

            det = Detection(
                mask=mask,
                centroid=center,
                features=feat,
                id=f"vessel_{i + 1}",
                score=1.0 if feat.get('confidence') == 'high' else (
                    0.7 if feat.get('confidence') == 'medium' else 0.4
                ),
            )
            detections.append(det)

        return detections

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
        extract_full_features: bool = True,
        tile_x: int = 0,
        tile_y: int = 0,
        tile_size: Optional[int] = None,
        attempt_merge: bool = False,
        extra_channels: Optional[Dict[int, np.ndarray]] = None,
        channel_names: Optional[Dict[int, str]] = None,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Complete vessel detection pipeline with full 2326 feature extraction.

        Enhanced: Supports tile boundary detection and cross-tile merging.

        Pipeline:
        1. Segment ring candidates using contour hierarchy analysis
        2. Detect partial vessels at tile boundaries (stored in self._partial_vessels)
        3. Attempt cross-tile merging if enabled (during processing)
        4. Extract vessel-specific features (wall thickness, diameters, etc.)
        5. Extract full features (22 morphological + 256 SAM2 + 2048 ResNet)
        6. Filter by size and create Detection objects

        Cross-Tile Merging Options:
        ---------------------------
        There are two approaches for cross-tile vessel merging:

        1. **During processing** (attempt_merge=True):
           Merges with adjacent tiles as each tile is processed.
           Requires tiles to be processed in order (left-to-right, top-to-bottom).
           Good for streaming processing where memory is limited.

        2. **After all tiles processed** (recommended):
           Process all tiles first with attempt_merge=False, then call
           merge_cross_tile_vessels() once at the end. This approach:
           - Handles any tile processing order
           - Allows batch merging optimization
           - Provides merge statistics via get_merge_statistics()

           Example:
           ```python
           strategy = VesselStrategy(enable_boundary_detection=True)

           # Phase 1: Process all tiles (partials stored automatically)
           all_detections = []
           for tile_x, tile_y, tile_data in tiles:
               masks, detections = strategy.detect(
                   tile_data, models, pixel_size_um,
                   tile_x=tile_x, tile_y=tile_y
               )
               all_detections.extend(detections)

           # Phase 2: Merge cross-tile vessels
           merged_vessels = strategy.merge_cross_tile_vessels(
               tile_size=4000, overlap=0, match_threshold=0.6
           )

           # Phase 3: Process merged vessels (extract features, add to results)
           # ... handle merged_vessels ...

           # Phase 4: Cleanup
           strategy.clear_partial_vessels()
           ```

        Args:
            tile: RGB or grayscale image (SMA channel)
            models: Dict with optional keys:
                - 'sam2_predictor': SAM2ImagePredictor (for embeddings)
                - 'resnet': ResNet model (for features)
                - 'resnet_transform': torchvision transform
                - 'device': torch device
            pixel_size_um: Pixel size in microns
            cd31_channel: Optional CD31 channel for validation
            extract_full_features: Whether to extract all 2326 features (default True)
            tile_x: X coordinate of tile origin (for boundary tracking)
            tile_y: Y coordinate of tile origin (for boundary tracking)
            tile_size: Size of tiles (for cross-tile merging)
            attempt_merge: Whether to attempt merging with adjacent tiles (default False)
                Set to False and use merge_cross_tile_vessels() after all tiles
                for more robust merging.

        Returns:
            Tuple of (combined mask array, list of Detection objects)
            Detection features include:
            - 'detection_type': 'complete' | 'partial' | 'merged'
            - 'touches_boundary': bool
            - 'is_merged': bool
            - 'source_tiles': list of tile coordinates
        """
        import torch

        # Get ring candidates - either parallel multi-marker or sequential SMA-only
        if (self.parallel_detection and
            extra_channels is not None and
            channel_names is not None and
            len(extra_channels) > 0):
            # Parallel multi-marker detection (SMA + CD31 + LYVE1)
            # Uses ThreadPoolExecutor to run CPU-bound detection in parallel
            logger.info(
                f"Running parallel multi-marker detection with {self.parallel_workers} workers "
                f"(channels: {list(channel_names.values())})"
            )
            ring_candidates = self._detect_all_markers_parallel(
                tile=tile,
                extra_channels=extra_channels,
                pixel_size_um=pixel_size_um,
                channel_names=channel_names,
                models=models,
                n_workers=self.parallel_workers,
                tile_x=tile_x,
                tile_y=tile_y,
            )

            # Merge overlapping candidates from different markers (multi-marker mode)
            if self.multi_marker and len(ring_candidates) > 1:
                pre_merge_count = len(ring_candidates)
                ring_candidates = self._merge_candidates(
                    ring_candidates,
                    iou_threshold=self.merge_iou_threshold
                )
                logger.info(
                    f"Multi-marker merge: {pre_merge_count} candidates -> "
                    f"{len(ring_candidates)} after IoU deduplication (threshold={self.merge_iou_threshold})"
                )
        elif self.lumen_first:
            # Lumen-first detection mode
            ring_candidates = self._detect_lumen_first(
                tile, pixel_size_um,
                min_lumen_area_um2=50 if self.candidate_mode else 75,
                min_ellipse_fit=0.40 if self.candidate_mode else 0.55,
                max_aspect_ratio=5.0 if self.candidate_mode else 4.0,
                min_wall_brightness_ratio=1.15,
            )
        else:
            # Standard sequential SMA ring detection
            ring_candidates = self.segment(
                tile, models, pixel_size_um, cd31_channel,
                tile_x=tile_x, tile_y=tile_y
            )

        # Attempt cross-tile merging if enabled and adjacent tiles have been processed
        if attempt_merge and tile_size is not None and self.cross_tile_config.enabled:
            ring_candidates = self.attempt_cross_tile_merge(
                tile_x, tile_y, tile_size, ring_candidates
            )

        if not ring_candidates:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        h, w = tile.shape[:2]

        # Prepare image for SAM2/ResNet
        if tile.ndim == 2:
            tile_rgb = np.stack([tile] * 3, axis=-1)
        elif tile.shape[2] == 1:
            tile_rgb = np.concatenate([tile, tile, tile], axis=-1)
        else:
            tile_rgb = tile[:, :, :3]

        # Ensure uint8 format
        if tile_rgb.dtype != np.uint8:
            if tile_rgb.dtype == np.uint16:
                tile_rgb = (tile_rgb / 256).astype(np.uint8)
            else:
                tile_rgb = tile_rgb.astype(np.uint8)

        # Get models
        sam2_predictor = models.get('sam2_predictor')
        resnet = models.get('resnet')
        resnet_transform = models.get('resnet_transform')
        device = models.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Set image for SAM2 embeddings if available
        if sam2_predictor is not None and self.extract_sam2_embeddings and extract_full_features:
            sam2_predictor.set_image(tile_rgb)

        # Extract features and create masks for valid candidates
        valid_candidates = []
        crops_for_resnet = []
        crop_indices = []

        for cand_idx, cand in enumerate(ring_candidates):
            # Extract vessel-specific features
            # In candidate_mode, use extract_candidate_features to get ALL candidates
            # without filtering (for RF training)
            if self.candidate_mode:
                vessel_feat = self.extract_candidate_features(
                    cand, tile, models, pixel_size_um, cd31_channel
                )
            else:
                vessel_feat = self.extract_features(
                    cand, tile, models, pixel_size_um, cd31_channel
                )

            if vessel_feat is None:
                continue

            # Create wall mask - handle partial vessels (no inner contour)
            temp = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp, [cand['outer']], 0, 1, -1)

            # Only subtract inner contour if it exists (not for partial vessels)
            if cand.get('inner') is not None:
                cv2.drawContours(temp, [cand['inner']], 0, 0, -1)

            wall_mask = temp.astype(bool)

            if wall_mask.sum() == 0:
                continue

            # Extract 22 morphological features
            morph_feat = extract_morphological_features(wall_mask, tile_rgb)
            if not morph_feat:
                continue

            # Create lumen mask if inner contour exists
            lumen_mask = None
            if cand.get('inner') is not None:
                lumen_mask = np.zeros((h, w), dtype=bool)
                lumen_temp = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(lumen_temp, [cand['inner']], 0, 255, -1)
                lumen_mask = lumen_temp > 0

            # Extract vessel-specific features (~28 features + multi-channel if available)
            # These include: ring/wall, diameter, shape, intensity, context features
            # Plus: per-channel intensity (wall/lumen) and cross-channel ratios
            try:
                if extra_channels is not None and len(extra_channels) > 0:
                    # Multi-channel feature extraction (standard + per-channel + ratios)
                    # Use provided channel_names or fall back to defaults
                    effective_channel_names = channel_names if channel_names is not None else DEFAULT_CHANNEL_NAMES
                    vessel_specific_feat = extract_all_vessel_features_multichannel(
                        wall_mask=wall_mask,
                        lumen_mask=lumen_mask,
                        sma_channel=tile_rgb,
                        outer_contour=cand['outer'],
                        inner_contour=cand.get('inner'),
                        pixel_size_um=pixel_size_um,
                        channels_data=extra_channels,
                        channel_names=effective_channel_names,
                        binary_mask=cand.get('binary'),
                    )
                else:
                    # Single-channel (SMA only) feature extraction
                    vessel_specific_feat = extract_vessel_features(
                        wall_mask=wall_mask,
                        lumen_mask=lumen_mask,
                        sma_channel=tile_rgb,
                        outer_contour=cand['outer'],
                        inner_contour=cand.get('inner'),
                        pixel_size_um=pixel_size_um,
                        binary_mask=cand.get('binary'),
                    )
            except Exception as e:
                logger.debug(f"Vessel-specific feature extraction failed: {e}")
                vessel_specific_feat = {}

            # Merge all features: morphological + vessel-specific from extract_features + new vessel features
            # vessel_feat comes from self.extract_features (basic vessel metrics)
            # vessel_specific_feat comes from extract_vessel_features (advanced discriminative features)
            all_features = {**morph_feat, **vessel_feat, **vessel_specific_feat}

            # Get centroid
            center = vessel_feat.get('outer_center', [0, 0])
            if center is None:
                # Fall back to mask centroid
                ys, xs = np.where(wall_mask)
                center = [float(np.mean(xs)), float(np.mean(ys))]

            cx, cy = center[0], center[1]

            # Extract SAM2 embeddings (256D)
            if sam2_predictor is not None and self.extract_sam2_embeddings and extract_full_features:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy, cx)
                for i, v in enumerate(sam2_emb):
                    all_features[f'sam2_emb_{i}'] = float(v)
            elif extract_full_features:
                # Fill with zeros if SAM2 not available
                for i in range(256):
                    all_features[f'sam2_emb_{i}'] = 0.0

            # Prepare crop for batch ResNet processing
            if self.extract_resnet_features and extract_full_features:
                ys, xs = np.where(wall_mask)
                if len(ys) > 0:
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    crop = tile_rgb[y1:y2+1, x1:x2+1].copy()
                    crop_mask = wall_mask[y1:y2+1, x1:x2+1]
                    crop[~crop_mask] = 0  # Zero out background
                    crops_for_resnet.append(crop)
                    crop_indices.append(len(valid_candidates))

            # Store inner contour only if it exists
            inner_contour_data = None
            if cand.get('inner') is not None:
                inner_contour_data = cand['inner'].tolist()

            valid_candidates.append({
                'mask': wall_mask,
                'features': all_features,
                'centroid': center,
                'outer_contour': cand['outer'].tolist(),
                'inner_contour': inner_contour_data,
                'is_partial': vessel_feat.get('is_partial', False),
                'is_merged': vessel_feat.get('is_merged', False),
                'detection_type': vessel_feat.get('detection_type', 'complete'),
            })

        # Batch ResNet feature extraction
        if crops_for_resnet and resnet is not None and resnet_transform is not None and extract_full_features:
            resnet_features_list = self._extract_resnet_features_batch(
                crops_for_resnet, resnet, resnet_transform, device
            )

            # Assign ResNet features to correct candidates
            for crop_idx, resnet_feats in zip(crop_indices, resnet_features_list):
                for i, v in enumerate(resnet_feats):
                    valid_candidates[crop_idx]['features'][f'resnet_{i}'] = float(v)

        # Fill zeros for candidates without ResNet features
        if extract_full_features:
            for cand in valid_candidates:
                if 'resnet_0' not in cand['features']:
                    for i in range(2048):
                        cand['features'][f'resnet_{i}'] = 0.0

        # Reset SAM2 predictor
        if sam2_predictor is not None and self.extract_sam2_embeddings:
            try:
                sam2_predictor.reset_predictor()
            except Exception as e:
                logger.debug(f"Failed to reset SAM2 predictor: {e}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        # Create Detection objects
        masks_list = [cand['mask'] for cand in valid_candidates]
        features_list = [cand['features'] for cand in valid_candidates]

        detections = self.filter(masks_list, features_list, pixel_size_um)

        # Defensive check: ensure detections is a list
        if detections is None:
            detections = []

        # Add contour data and tracking info to detections
        for i, det in enumerate(detections):
            if i < len(valid_candidates):
                det.features['outer_contour'] = valid_candidates[i]['outer_contour']
                det.features['inner_contour'] = valid_candidates[i]['inner_contour']  # May be None for partial

        # Build combined mask with overlap checking
        combined_mask = np.zeros((h, w), dtype=np.uint32)
        final_detections = []
        det_id = 1
        complete_count = 0
        partial_count = 0
        merged_count = 0

        for det in detections:
            mask = det.mask

            # Check overlap with existing detections
            if combined_mask.max() > 0:
                overlap = (mask & (combined_mask > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            combined_mask[mask] = det_id

            # Generate ID based on detection type
            detection_type = det.features.get('detection_type', 'complete')
            if detection_type == 'merged':
                merged_count += 1
                det.id = f"vessel_merged_{merged_count}"
            elif detection_type == 'partial':
                partial_count += 1
                det.id = f"vessel_partial_{partial_count}"
            else:
                complete_count += 1
                det.id = f"vessel_{complete_count}"

            final_detections.append(det)
            det_id += 1

        # Log detection summary
        if final_detections:
            logger.debug(
                f"Vessel detection at ({tile_x}, {tile_y}): "
                f"{complete_count} complete, {partial_count} partial, {merged_count} merged"
            )

        return combined_mask, final_detections

    def detect_multiscale(
        self,
        tile_getter: callable,
        models: Dict[str, Any],
        mosaic_width: int,
        mosaic_height: int,
        tile_size: int = 4000,
        scales: List[int] = None,
        pixel_size_um: float = 0.17,
        channel: int = 0,
        iou_threshold: float = 0.3,
        sample_fraction: float = 1.0,
        progress_callback: Optional[callable] = None,
        **detect_kwargs,
    ) -> Tuple[List[np.ndarray], List[Detection]]:
        """
        Multi-scale vessel detection with IoU-based deduplication.

        Detects vessels at multiple scales (coarse to fine) to capture
        vessels of all sizes while avoiding cross-tile fragmentation.

        Scale levels:
        - 1/8x (coarse): Large vessels >100 µm, detects whole vessels
        - 1/4x (medium): Medium vessels 30-200 µm
        - 1x (fine): Small vessels and capillaries 3-75 µm

        Args:
            tile_getter: Function(x, y, size, channel, scale_factor) -> ndarray
            models: Dict with SAM2/ResNet models
            mosaic_width: Full-resolution mosaic width
            mosaic_height: Full-resolution mosaic height
            tile_size: Tile size in pixels (same at all scales)
            scales: List of scale factors [8, 4, 1] = coarse to fine
            pixel_size_um: Pixel size at full resolution in µm
            channel: Channel to process
            iou_threshold: IoU threshold for deduplication
            sample_fraction: Fraction of tiles to process (0-1)
            progress_callback: Optional callback(scale, tiles_done, total_tiles)
            **detect_kwargs: Additional kwargs passed to detect()

        Returns:
            Tuple of (list of masks, list of Detection objects)

        Example:
            masks, detections = strategy.detect_multiscale(
                tile_getter=loader.get_tile,
                models=models,
                mosaic_width=loader.width,
                mosaic_height=loader.height,
                scales=[8, 4, 1],
                pixel_size_um=0.17,
            )
        """
        from segmentation.utils.multiscale import (
            get_scale_params,
            generate_tile_grid_at_scale,
            convert_detection_to_full_res,
            merge_detections_across_scales,
        )
        import random

        if scales is None:
            scales = [8, 4, 1]  # Default: coarse, medium, fine

        all_detections = []
        all_masks = []

        for scale in scales:
            scale_pixel_size = pixel_size_um * scale
            scale_params = get_scale_params(scale)

            # Generate tile grid at this scale
            tiles = generate_tile_grid_at_scale(
                mosaic_width, mosaic_height, tile_size, scale
            )

            # Sample tiles if sample_fraction < 1.0
            if sample_fraction < 1.0:
                n_sample = max(1, int(len(tiles) * sample_fraction))
                tiles = random.sample(tiles, n_sample)

            logger.info(
                f"Scale 1/{scale}x: Processing {len(tiles)} tiles, "
                f"pixel_size={scale_pixel_size:.3f} µm, "
                f"target: {scale_params['description']}"
            )

            scale_detections = 0
            import time as _time
            for i, (tile_x, tile_y) in enumerate(tiles):
                tile_start = _time.time()
                logger.info(f"  Scale 1/{scale}x: Tile {i+1}/{len(tiles)} at ({tile_x}, {tile_y})...")

                # Get tile at this scale
                tile = tile_getter(tile_x, tile_y, tile_size, channel, scale)
                if tile is None:
                    logger.info(f"  Scale 1/{scale}x: Tile {i+1}/{len(tiles)} - skipped (no data)")
                    continue

                # Temporarily adjust detection parameters for this scale
                original_min_diam = self.min_diameter_um
                original_max_diam = self.max_diameter_um

                self.min_diameter_um = scale_params['min_diameter_um']
                self.max_diameter_um = scale_params['max_diameter_um']

                try:
                    # Run detection
                    masks, detections = self.detect(
                        tile=tile,
                        models=models,
                        pixel_size_um=scale_pixel_size,
                        tile_x=tile_x * scale,  # Convert to full-res coords
                        tile_y=tile_y * scale,
                        **detect_kwargs
                    )

                    # Convert detections to full resolution coordinates
                    for det in detections:
                        det_dict = det.to_dict() if hasattr(det, 'to_dict') else det
                        # Preserve contour for IoU-based merge (to_dict() doesn't include it)
                        # Contour is stored in det.features['outer'], need it at top level for merge
                        if hasattr(det, 'features') and det.features.get('outer') is not None:
                            det_dict['outer'] = det.features['outer']
                        elif isinstance(det_dict.get('features'), dict) and det_dict['features'].get('outer') is not None:
                            det_dict['outer'] = det_dict['features']['outer']
                        det_fullres = convert_detection_to_full_res(
                            det_dict, scale, tile_x, tile_y
                        )
                        all_detections.append(det_fullres)
                        scale_detections += 1

                    if masks is not None and masks.size > 0:
                        all_masks.append((tile_x * scale, tile_y * scale, scale, masks))

                finally:
                    # Restore original parameters
                    self.min_diameter_um = original_min_diam
                    self.max_diameter_um = original_max_diam

                tile_elapsed = _time.time() - tile_start
                logger.info(
                    f"  Scale 1/{scale}x: Tile {i+1}/{len(tiles)} done - "
                    f"found {len(detections)} vessels in {tile_elapsed:.1f}s"
                )

                if progress_callback:
                    progress_callback(scale, i + 1, len(tiles))

            logger.info(f"Scale 1/{scale}x: Found {scale_detections} detections")

        # Merge detections across scales (finer scale takes precedence)
        logger.info(f"Merging {len(all_detections)} detections across scales...")
        merged_detections = merge_detections_across_scales(
            all_detections,
            iou_threshold=iou_threshold,
            prefer_finer_scale=True
        )

        # Convert back to Detection objects if needed
        final_detections = []
        for det_dict in merged_detections:
            if isinstance(det_dict, Detection):
                final_detections.append(det_dict)
            else:
                # Create Detection from dict
                # Detection dataclass expects: mask, centroid, features, id, score
                features = det_dict.get('features', {}).copy()
                # ALWAYS use the scaled 'outer' contour from det_dict (not the original in features)
                # convert_detection_to_full_res() already scaled det_dict['outer'] to full resolution
                if det_dict.get('outer') is not None:
                    features['outer'] = det_dict.get('outer')
                if det_dict.get('inner') is not None:
                    features['inner'] = det_dict.get('inner')

                final_detections.append(Detection(
                    mask=det_dict.get('mask', np.zeros((1, 1), dtype=bool)),
                    centroid=det_dict.get('center', det_dict.get('centroid', [0, 0])),
                    features=features,
                    id=det_dict.get('id', det_dict.get('mask_id')),
                    score=det_dict.get('score', det_dict.get('confidence', 1.0)),
                ))

        logger.info(
            f"Multi-scale detection complete: {len(final_detections)} vessels "
            f"(from {len(all_detections)} raw detections)"
        )

        return all_masks, final_detections

    # _extract_sam2_embedding inherited from DetectionStrategy base class
    # _extract_resnet_features_batch inherited from DetectionStrategy base class

    def detect_medsam_multiscale(
        self,
        tile_getter: callable,
        mosaic_width: int,
        mosaic_height: int,
        tile_size: int = 4000,
        scales: List[int] = None,
        pixel_size_um: float = 0.17,
        channel: int = 0,
        iou_threshold: float = 0.3,
        sample_fraction: float = 1.0,
        medsam_checkpoint: str = None,
        progress_callback: Optional[callable] = None,
        **detect_kwargs,
    ) -> Tuple[List[Dict], List[Detection]]:
        """
        Multi-scale vessel detection using MedSAM with ring structure filtering.

        MedSAM (Medical SAM) is SAM fine-tuned on 1.5M medical image-mask pairs,
        providing better boundary detection for medical/biological structures.

        Approach:
        1. At each scale, run MedSAM automatic mask generation
        2. Filter masks that have "holes" (lumen inside wall = vessel cross-section)
        3. Extract vessel features from filtered masks
        4. Convert coordinates to full resolution
        5. Merge across scales using IoU deduplication

        Args:
            tile_getter: Function(x, y, size, channel, scale_factor) -> ndarray
            mosaic_width: Full-resolution mosaic width
            mosaic_height: Full-resolution mosaic height
            tile_size: Tile size in pixels (same at all scales)
            scales: List of scale factors [16, 8, 4] = coarse to fine
            pixel_size_um: Pixel size at full resolution in µm
            channel: Channel to process
            iou_threshold: IoU threshold for deduplication
            sample_fraction: Fraction of tiles to process (0-1)
            medsam_checkpoint: Path to MedSAM checkpoint (auto-detected if None)
            progress_callback: Optional callback(scale, tiles_done, total_tiles)
            **detect_kwargs: Additional kwargs (min_area_um2, etc.)

        Returns:
            Tuple of (list of mask info dicts, list of Detection objects)
        """
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from segmentation.utils.multiscale import (
            get_scale_params,
            generate_tile_grid_at_scale,
            convert_detection_to_full_res,
            merge_detections_across_scales,
        )
        import random
        import time as _time

        if scales is None:
            scales = [16, 8, 4]  # Default: very coarse to medium

        # Find MedSAM checkpoint
        if medsam_checkpoint is None:
            import os
            possible_paths = [
                "/home/dude/code/vessel_seg/checkpoints/medsam_vit_b.pth",
                "checkpoints/medsam_vit_b.pth",
                os.path.expanduser("~/checkpoints/medsam_vit_b.pth"),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    medsam_checkpoint = p
                    break
            if medsam_checkpoint is None:
                raise FileNotFoundError("MedSAM checkpoint not found. Please provide medsam_checkpoint path.")

        # Load MedSAM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading MedSAM from {medsam_checkpoint} on {device}...")
        sam = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
        sam.to(device)
        sam.eval()

        # Create automatic mask generator with settings tuned for vessels
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,          # Dense point grid for finding all structures
            pred_iou_thresh=0.86,        # Slightly lower for more candidates
            stability_score_thresh=0.92, # Slightly lower for partial rings
            min_mask_region_area=50,     # Small area to catch capillaries
            crop_n_layers=1,             # Multi-crop for better coverage
            crop_n_points_downscale_factor=2,
        )
        logger.info("MedSAM mask generator created")

        all_detections = []
        all_mask_info = []

        for scale in scales:
            scale_pixel_size = pixel_size_um * scale
            scale_params = get_scale_params(scale)

            # Generate tile grid at this scale
            tiles = generate_tile_grid_at_scale(
                mosaic_width, mosaic_height, tile_size, scale
            )

            # Sample tiles if sample_fraction < 1.0
            if sample_fraction < 1.0:
                n_sample = max(1, int(len(tiles) * sample_fraction))
                tiles = random.sample(tiles, n_sample)

            logger.info(
                f"MedSAM Scale 1/{scale}x: Processing {len(tiles)} tiles, "
                f"pixel_size={scale_pixel_size:.3f} µm, "
                f"target: {scale_params['description']}"
            )

            scale_detections = 0
            for i, (tile_x, tile_y) in enumerate(tiles):
                tile_start = _time.time()
                logger.info(f"  MedSAM Scale 1/{scale}x: Tile {i+1}/{len(tiles)} at ({tile_x}, {tile_y})...")

                # Get tile at this scale
                tile = tile_getter(tile_x, tile_y, tile_size, channel, scale)
                if tile is None:
                    logger.info(f"  MedSAM Scale 1/{scale}x: Tile {i+1}/{len(tiles)} - skipped (no data)")
                    continue

                # Convert to RGB for MedSAM (expects 3-channel)
                if tile.ndim == 2:
                    tile_rgb = np.stack([tile, tile, tile], axis=-1)
                elif tile.shape[-1] == 1:
                    tile_rgb = np.concatenate([tile, tile, tile], axis=-1)
                else:
                    tile_rgb = tile

                # Normalize to uint8 if needed
                if tile_rgb.dtype != np.uint8:
                    if tile_rgb.max() > 255:
                        tile_rgb = ((tile_rgb - tile_rgb.min()) / (tile_rgb.max() - tile_rgb.min() + 1e-8) * 255).astype(np.uint8)
                    else:
                        tile_rgb = tile_rgb.astype(np.uint8)

                # Run MedSAM automatic mask generation
                try:
                    masks = mask_generator.generate(tile_rgb)
                except Exception as e:
                    logger.warning(f"  MedSAM error on tile ({tile_x}, {tile_y}): {e}")
                    continue

                logger.info(f"  MedSAM found {len(masks)} masks, filtering for vessel rings...")

                # Filter masks for ring structures (vessel cross-sections)
                vessel_detections = self._filter_medsam_masks_for_vessels(
                    masks,
                    tile_rgb,
                    tile_x, tile_y,
                    scale,
                    scale_pixel_size,
                    scale_params,
                )

                # Convert to full resolution coordinates
                for det in vessel_detections:
                    det_fullres = convert_detection_to_full_res(
                        det, scale, tile_x, tile_y
                    )
                    all_detections.append(det_fullres)
                    scale_detections += 1

                tile_elapsed = _time.time() - tile_start
                logger.info(
                    f"  MedSAM Scale 1/{scale}x: Tile {i+1}/{len(tiles)} done - "
                    f"found {len(vessel_detections)} vessels in {tile_elapsed:.1f}s"
                )

                if progress_callback:
                    progress_callback(scale, i + 1, len(tiles))

            logger.info(f"MedSAM Scale 1/{scale}x: Found {scale_detections} vessel detections")

        # Cleanup MedSAM model
        del sam, mask_generator
        torch.cuda.empty_cache()
        gc.collect()

        # Merge detections across scales (finer scale takes precedence)
        logger.info(f"Merging {len(all_detections)} detections across scales...")
        merged_detections = merge_detections_across_scales(
            all_detections,
            iou_threshold=iou_threshold,
            prefer_finer_scale=True
        )

        # Convert to Detection objects
        final_detections = []
        for det_dict in merged_detections:
            if isinstance(det_dict, Detection):
                final_detections.append(det_dict)
            else:
                features = det_dict.get('features', {}).copy()
                if det_dict.get('outer') is not None:
                    features['outer'] = det_dict.get('outer')
                if det_dict.get('inner') is not None:
                    features['inner'] = det_dict.get('inner')

                final_detections.append(Detection(
                    mask=det_dict.get('mask', np.zeros((1, 1), dtype=bool)),
                    centroid=det_dict.get('center', det_dict.get('centroid', [0, 0])),
                    features=features,
                    id=det_dict.get('id', det_dict.get('mask_id')),
                    score=det_dict.get('score', det_dict.get('confidence', 1.0)),
                ))

        logger.info(
            f"MedSAM multi-scale detection complete: {len(final_detections)} vessels "
            f"(from {len(all_detections)} raw detections)"
        )

        return all_mask_info, final_detections

    def _filter_medsam_masks_for_vessels(
        self,
        masks: List[Dict],
        tile_rgb: np.ndarray,
        tile_x: int,
        tile_y: int,
        scale: int,
        pixel_size_um: float,
        scale_params: Dict,
    ) -> List[Dict]:
        """
        Filter MedSAM masks for vessel ring structures.

        A vessel cross-section has:
        - Ring structure: wall surrounding a lumen (hole in mask)
        - Appropriate size for the scale
        - Circularity (not too elongated)

        Args:
            masks: List of MedSAM mask dicts with 'segmentation', 'area', etc.
            tile_rgb: The RGB tile image
            tile_x, tile_y: Tile position in scaled coordinates
            scale: Scale factor
            pixel_size_um: Pixel size at this scale
            scale_params: Scale-specific parameters

        Returns:
            List of vessel detection dicts
        """
        vessel_detections = []
        min_diam_px = scale_params['min_diameter_um'] / pixel_size_um
        max_diam_px = scale_params['max_diameter_um'] / pixel_size_um
        min_area_px = np.pi * (min_diam_px / 2) ** 2 * 0.5  # Allow some margin
        max_area_px = np.pi * (max_diam_px / 2) ** 2 * 2.0

        for mask_info in masks:
            mask = mask_info['segmentation'].astype(np.uint8)
            area = mask_info['area']

            # Size filter
            if area < min_area_px or area > max_area_px:
                continue

            # Find contours to check for ring structure
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            # Check for ring structure: outer contour with inner hole(s)
            # In RETR_CCOMP hierarchy: [Next, Previous, First_Child, Parent]
            # A ring has an outer contour (parent=-1) with a child (lumen)
            outer_contour = None
            inner_contour = None
            has_ring = False

            if hierarchy is not None:
                hierarchy = hierarchy[0]
                for idx, (cnt, h) in enumerate(zip(contours, hierarchy)):
                    # h = [Next, Previous, First_Child, Parent]
                    if h[3] == -1 and h[2] != -1:  # No parent, has child = ring
                        outer_contour = cnt
                        child_idx = h[2]
                        if child_idx < len(contours):
                            inner_contour = contours[child_idx]
                            has_ring = True
                            break

            # If no ring structure found, check if it's a solid vessel-like shape
            if not has_ring:
                # Use largest contour as outer
                outer_contour = max(contours, key=cv2.contourArea)
                # Check if mask has internal holes by comparing mask area to contour area
                contour_area = cv2.contourArea(outer_contour)
                mask_area = np.sum(mask > 0)
                if contour_area > 0 and mask_area < contour_area * 0.9:
                    # Mask has holes - it's a ring!
                    has_ring = True
                    # Find the largest internal contour as lumen
                    inverted = 255 - mask
                    inner_contours, _ = cv2.findContours(
                        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    # Filter to internal holes only
                    internal_holes = []
                    for ic in inner_contours:
                        # Check if this contour is inside the outer contour
                        M = cv2.moments(ic)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            if cv2.pointPolygonTest(outer_contour, (cx, cy), False) > 0:
                                internal_holes.append(ic)
                    if internal_holes:
                        inner_contour = max(internal_holes, key=cv2.contourArea)

            if outer_contour is None:
                continue

            # Compute vessel metrics
            outer_area = cv2.contourArea(outer_contour)
            if outer_area < 10:
                continue

            # Fit ellipse if enough points
            if len(outer_contour) >= 5:
                ellipse = cv2.fitEllipse(outer_contour)
                center, axes, angle = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                aspect_ratio = major_axis / (minor_axis + 1e-6)
                outer_diameter_px = (major_axis + minor_axis) / 2
            else:
                # Fallback to bounding rect
                x, y, w, h = cv2.boundingRect(outer_contour)
                center = (x + w/2, y + h/2)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                outer_diameter_px = (w + h) / 2

            outer_diameter_um = outer_diameter_px * pixel_size_um

            # Filter by aspect ratio (reject elongated structures)
            if aspect_ratio > self.max_aspect_ratio:
                continue

            # Compute circularity
            perimeter = cv2.arcLength(outer_contour, True)
            circularity = 4 * np.pi * outer_area / (perimeter ** 2 + 1e-6)

            if circularity < self.min_circularity:
                continue

            # Compute ring completeness (for ring structures)
            ring_completeness = 0.0
            wall_thickness_um = 0.0
            inner_diameter_um = 0.0

            if has_ring and inner_contour is not None:
                inner_area = cv2.contourArea(inner_contour)
                wall_area = outer_area - inner_area
                ring_completeness = wall_area / (outer_area + 1e-6)

                # Estimate wall thickness
                inner_diameter_px = np.sqrt(4 * inner_area / np.pi)
                inner_diameter_um = inner_diameter_px * pixel_size_um
                wall_thickness_px = (outer_diameter_px - inner_diameter_px) / 2
                wall_thickness_um = wall_thickness_px * pixel_size_um

            # Build detection dict
            detection = {
                'id': f"medsam_{tile_x}_{tile_y}_{len(vessel_detections)}",
                'mask_id': f"medsam_{tile_x}_{tile_y}_{len(vessel_detections)}",
                'center': [float(center[0]), float(center[1])],
                'centroid': [float(center[0]), float(center[1])],
                'outer': outer_contour,
                'inner': inner_contour,
                'mask': mask.astype(bool),
                'score': float(mask_info.get('predicted_iou', 0.9)),
                'confidence': float(mask_info.get('stability_score', 0.9)),
                'features': {
                    'outer': outer_contour,
                    'inner': inner_contour,
                    'outer_diameter_um': outer_diameter_um,
                    'inner_diameter_um': inner_diameter_um,
                    'wall_thickness_mean_um': wall_thickness_um,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'ring_completeness': ring_completeness,
                    'has_ring': has_ring,
                    'area_px': outer_area,
                    'area_um2': outer_area * (pixel_size_um ** 2),
                    'scale_detected': scale,
                    'detection_method': 'medsam',
                },
            }
            vessel_detections.append(detection)

        return vessel_detections

    def create_vessel_mask(
        self,
        outer_contour: np.ndarray,
        inner_contour: Optional[np.ndarray],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a wall mask from outer and inner contours.

        Enhanced: Handles partial vessels where inner_contour may be None.

        Args:
            outer_contour: Outer boundary contour
            inner_contour: Inner (lumen) boundary contour, or None for partial vessels
            shape: (height, width) of output mask

        Returns:
            Boolean mask of vessel wall (or full contour for partial vessels)
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [outer_contour], 0, 255, -1)

        # Only subtract inner contour if it exists
        if inner_contour is not None:
            cv2.drawContours(mask, [inner_contour], 0, 0, -1)

        return mask > 0

    def get_config(self) -> Dict[str, Any]:
        """
        Return strategy configuration including boundary detection settings.

        Returns:
            Dict with all configuration parameters
        """
        base_config = {
            'strategy': self.name,
            'min_diameter_um': self.min_diameter_um,
            'max_diameter_um': self.max_diameter_um,
            'min_wall_thickness_um': self.min_wall_thickness_um,
            'max_aspect_ratio': self.max_aspect_ratio,
            'min_circularity': self.min_circularity,
            'min_ring_completeness': self.min_ring_completeness,
            'canny_low': self.canny_low,
            'canny_high': self.canny_high,
            'classify_vessel_types': self.classify_vessel_types,
            'extract_resnet_features': self.extract_resnet_features,
            'extract_sam2_embeddings': self.extract_sam2_embeddings,
            'resnet_batch_size': self.resnet_batch_size,
            # Boundary detection settings
            'enable_boundary_detection': self.enable_boundary_detection,
            'boundary_margin_px': self.boundary_margin_px,
            'use_tree_hierarchy': self.use_tree_hierarchy,
            # Cross-tile merge settings
            'cross_tile_merge_enabled': self.cross_tile_config.enabled,
            'cross_tile_position_tolerance_px': self.cross_tile_config.position_tolerance_px,
            'cross_tile_orientation_tolerance_deg': self.cross_tile_config.orientation_tolerance_deg,
            'cross_tile_curvature_threshold': self.cross_tile_config.curvature_match_threshold,
        }
        return base_config

    def merge_cross_tile_vessels(
        self,
        tile_size: int,
        overlap: int = 0,
        match_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Orchestrate cross-tile vessel merging AFTER all tiles have been processed.

        This method should be called once after processing all tiles in the slide.
        It iterates through all stored partial vessels and merges matching partials
        across adjacent tile boundaries.

        The method handles:
        - Horizontal boundaries (LEFT <-> RIGHT edges)
        - Vertical boundaries (TOP <-> BOTTOM edges)
        - Avoids duplicate merges by tracking which partials have been merged

        Args:
            tile_size: Size of each tile in pixels (assumes square tiles)
            overlap: Overlap between tiles in pixels (default: 0)
            match_threshold: Minimum match score to accept a merge (default: 0.6)
                Uses _compute_match_score() which considers position, orientation,
                and curvature signature. Range 0-1, higher is better.

        Returns:
            List of merged vessel candidate dicts. Each dict contains:
                - 'outer': merged outer contour in global coordinates
                - 'inner': None (inner contour not preserved in merge)
                - 'all_inner': []
                - 'binary': None
                - 'touches_boundary': False (merged vessel is complete)
                - 'boundary_edges': set()
                - 'is_partial': False
                - 'is_merged': True
                - 'source_tiles': list of (tile_x, tile_y) tuples
                - 'merge_score': float (0-1)

        Example usage:
            ```python
            strategy = VesselStrategy(enable_boundary_detection=True)

            # Process all tiles
            for tile_x, tile_y, tile_data in tile_iterator:
                masks, detections = strategy.detect(
                    tile_data, models, pixel_size_um,
                    tile_x=tile_x, tile_y=tile_y
                )
                # ... save results ...

            # After all tiles processed, merge cross-tile vessels
            merged_vessels = strategy.merge_cross_tile_vessels(
                tile_size=4000, overlap=0, match_threshold=0.6
            )

            # Process merged vessels (e.g., extract features, add to final results)
            for merged in merged_vessels:
                # merged['outer'] is in global coordinates
                # merged['source_tiles'] lists contributing tiles
                pass

            # Clean up partial vessel cache
            strategy.clear_partial_vessels()
            ```

        Notes:
            - Call this method AFTER all tiles have been processed via detect()
            - Partial vessels are stored in self._partial_vessels during detect()
            - Call clear_partial_vessels() after using this method to free memory
            - The effective_tile_size accounts for overlap between tiles
            - Merged vessels have contours in GLOBAL coordinates (not tile-local)
        """
        if not self.cross_tile_config.enabled:
            logger.info("Cross-tile merging disabled in config")
            return []

        if not self._partial_vessels:
            logger.info("No partial vessels stored for merging")
            return []

        # Effective tile spacing accounts for overlap
        effective_tile_step = tile_size - overlap

        merged_vessels: List[Dict[str, Any]] = []
        merged_partial_ids: Set[Tuple[Tuple[int, int], int]] = set()  # (tile_coords, idx)

        # Collect all tile coordinates that have partial vessels
        tile_coords = list(self._partial_vessels.keys())
        logger.info(
            f"Starting cross-tile merge: {len(tile_coords)} tiles with partial vessels, "
            f"tile_size={tile_size}, overlap={overlap}, threshold={match_threshold}"
        )

        # Build a lookup map of partials by their boundary edges for efficient matching
        # Key: (tile_x, tile_y, edge) -> List[(partial_idx, PartialVessel)]
        edge_lookup: Dict[Tuple[int, int, BoundaryEdge], List[Tuple[int, PartialVessel]]] = {}

        for tile_key, partials in self._partial_vessels.items():
            tile_x, tile_y = tile_key
            for idx, partial in enumerate(partials):
                for edge in partial.boundary_edges:
                    lookup_key = (tile_x, tile_y, edge)
                    if lookup_key not in edge_lookup:
                        edge_lookup[lookup_key] = []
                    edge_lookup[lookup_key].append((idx, partial))

        # Define complementary edge pairs and their tile offsets
        # (edge_a, edge_b, dx, dy) means: if tile A has edge_a, look for edge_b in tile at (A.x+dx, A.y+dy)
        edge_pairs = [
            (BoundaryEdge.RIGHT, BoundaryEdge.LEFT, effective_tile_step, 0),   # Right edge -> Left of tile to the right
            (BoundaryEdge.BOTTOM, BoundaryEdge.TOP, 0, effective_tile_step),   # Bottom edge -> Top of tile below
        ]

        total_matches_found = 0
        total_merges_successful = 0

        # Iterate through all tiles and their partial vessels
        for tile1_key in tile_coords:
            tile1_x, tile1_y = tile1_key
            partials1 = self._partial_vessels[tile1_key]

            for idx1, partial1 in enumerate(partials1):
                partial1_id = (tile1_key, idx1)

                # Skip if already merged
                if partial1_id in merged_partial_ids:
                    continue

                # Check each edge pair
                for edge1, edge2, dx, dy in edge_pairs:
                    if edge1 not in partial1.boundary_edges:
                        continue

                    # Look for adjacent tile
                    tile2_x = tile1_x + dx
                    tile2_y = tile1_y + dy
                    tile2_key = (tile2_x, tile2_y)

                    # Check if adjacent tile has partials at the complementary edge
                    lookup_key = (tile2_x, tile2_y, edge2)
                    if lookup_key not in edge_lookup:
                        continue

                    # Find best match among candidates at complementary edge
                    best_match_idx = None
                    best_match_partial = None
                    best_score = 0.0

                    for idx2, partial2 in edge_lookup[lookup_key]:
                        partial2_id = (tile2_key, idx2)

                        # Skip if already merged
                        if partial2_id in merged_partial_ids:
                            continue

                        # Compute match score
                        score = self._compute_match_score(
                            partial1, partial2,
                            tile1_x, tile1_y,
                            tile2_x, tile2_y,
                            tile_size  # Use original tile_size for score computation
                        )

                        if score > best_score and score >= match_threshold:
                            best_score = score
                            best_match_idx = idx2
                            best_match_partial = partial2

                    if best_match_partial is not None:
                        total_matches_found += 1

                        # Attempt to merge
                        merged = self._merge_partial_vessels(
                            partial1, best_match_partial,
                            tile1_x, tile1_y,
                            tile2_x, tile2_y,
                            tile_size
                        )

                        if merged is not None:
                            # Update merge score with actual computed score
                            merged['merge_score'] = best_score
                            merged_vessels.append(merged)
                            total_merges_successful += 1

                            # Mark both partials as merged
                            merged_partial_ids.add(partial1_id)
                            merged_partial_ids.add((tile2_key, best_match_idx))

                            logger.debug(
                                f"Merged partial vessels: "
                                f"tile ({tile1_x}, {tile1_y}) {edge1.value} <-> "
                                f"tile ({tile2_x}, {tile2_y}) {edge2.value}, "
                                f"score={best_score:.3f}"
                            )

        logger.info(
            f"Cross-tile merge complete: "
            f"{total_matches_found} matches found, "
            f"{total_merges_successful} successful merges, "
            f"{len(merged_partial_ids)} partials consumed"
        )

        return merged_vessels

    def get_unmerged_partials(self) -> List[Dict[str, Any]]:
        """
        Get partial vessels that were not merged with any adjacent tile.

        These represent vessels at the slide boundary or vessels where no
        matching partial was found in adjacent tiles.

        Returns:
            List of partial vessel dicts with contours in global coordinates.
            Each dict contains:
                - 'outer': contour in global coordinates
                - 'inner': None
                - 'tile_origin': (tile_x, tile_y)
                - 'boundary_edges': set of BoundaryEdge
                - 'is_partial': True
                - 'is_merged': False
                - 'features': dict of extracted features
        """
        unmerged = []
        for tile_key, partials in self._partial_vessels.items():
            tile_x, tile_y = tile_key
            for partial in partials:
                # Convert contour to global coordinates
                contour = partial.contour.copy().reshape(-1, 2)
                contour[:, 0] += tile_x
                contour[:, 1] += tile_y
                contour = contour.reshape(-1, 1, 2).astype(np.int32)

                unmerged.append({
                    'outer': contour,
                    'inner': None,
                    'all_inner': [],
                    'tile_origin': (tile_x, tile_y),
                    'boundary_edges': partial.boundary_edges,
                    'is_partial': True,
                    'is_merged': False,
                    'features': partial.features,
                    'orientation': partial.orientation,
                })

        return unmerged

    def get_merge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored partial vessels for debugging.

        Returns:
            Dict with statistics:
                - 'total_tiles': number of tiles with partial vessels
                - 'total_partials': total number of partial vessels
                - 'partials_by_edge': count of partials touching each edge
                - 'tiles_with_partials': list of (tile_x, tile_y) tuples
        """
        stats = {
            'total_tiles': len(self._partial_vessels),
            'total_partials': 0,
            'partials_by_edge': {
                'top': 0,
                'bottom': 0,
                'left': 0,
                'right': 0,
            },
            'tiles_with_partials': list(self._partial_vessels.keys()),
        }

        for partials in self._partial_vessels.values():
            stats['total_partials'] += len(partials)
            for partial in partials:
                for edge in partial.boundary_edges:
                    stats['partials_by_edge'][edge.value] += 1

        return stats
