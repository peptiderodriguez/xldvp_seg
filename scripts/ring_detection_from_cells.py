#!/usr/bin/env python3
"""
Ring Detection from Cellpose-SAM Detected Cells

This script:
1. Loads a tile from a CZI file (SMA + Nuclear channels)
2. Runs Cellpose-SAM to detect individual cells
3. Clusters cell centroids using DBSCAN
4. For each cluster, checks if cells are arranged in a ring pattern
5. Each detected ring = one vessel

Ring detection criteria:
- Cells at similar distance from cluster centroid
- Angular coverage around the center (>50% of angles covered)
- Minimum number of cells in cluster (>= 5)
"""

import sys
import os

# Use non-interactive backend before importing matplotlib
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '/home/dude/code/vessel_seg')

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
from scipy import ndimage
from scipy.stats import circvar
from typing import List, Dict, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tile_from_czi(
    czi_path: str,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    channels: List[int]
) -> Dict[int, np.ndarray]:
    """Load a tile from CZI for multiple channels."""
    from segmentation.io.czi_loader import CZILoader

    loader = CZILoader(czi_path)
    logger.info(f"Loaded CZI: {loader.width}x{loader.height} px")
    logger.info(f"Mosaic origin: ({loader.x_start}, {loader.y_start})")

    tiles = {}
    for ch in channels:
        tile = loader.get_tile(tile_x, tile_y, tile_size, channel=ch)
        if tile is not None:
            tiles[ch] = tile
            logger.info(f"Channel {ch} tile shape: {tile.shape}, dtype: {tile.dtype}")

    loader.close()
    return tiles


def run_cellpose_detection(
    image: np.ndarray,
    model_type: str = 'cpsam',
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Run Cellpose detection on an image.

    Args:
        image: 2D grayscale image (uint8 or uint16)
        model_type: Cellpose model type ('cpsam' for SAM-based)
        diameter: Expected cell diameter (None for auto)
        flow_threshold: Flow error threshold
        cellprob_threshold: Cell probability threshold

    Returns:
        masks: Label image where each cell has unique ID
        n_cells: Number of detected cells
    """
    from cellpose.models import CellposeModel

    logger.info(f"Running Cellpose ({model_type}) detection...")

    # Initialize model
    model = CellposeModel(model_type=model_type, gpu=True)

    # Normalize image to uint8 if needed
    if image.dtype == np.uint16:
        # Percentile normalization
        p_low, p_high = np.percentile(image, [1, 99])
        img_norm = np.clip((image - p_low) / (p_high - p_low + 1e-8) * 255, 0, 255).astype(np.uint8)
    else:
        img_norm = image

    # Run detection
    masks, flows, styles = model.eval(
        img_norm,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=[0, 0]  # Grayscale
    )

    n_cells = masks.max()
    logger.info(f"Detected {n_cells} cells")

    return masks, n_cells


def extract_cell_centroids(masks: np.ndarray) -> np.ndarray:
    """
    Extract centroids from cell masks.

    Args:
        masks: Label image from Cellpose

    Returns:
        centroids: Nx2 array of (x, y) centroid coordinates
    """
    n_cells = masks.max()
    centroids = []

    for cell_id in range(1, n_cells + 1):
        cell_mask = masks == cell_id
        if cell_mask.sum() == 0:
            continue

        # Calculate centroid
        y_coords, x_coords = np.where(cell_mask)
        cx = x_coords.mean()
        cy = y_coords.mean()
        centroids.append([cx, cy])

    return np.array(centroids) if centroids else np.array([]).reshape(0, 2)


def cluster_cells(
    centroids: np.ndarray,
    eps: float = 100,  # Max distance between cells in same cluster
    min_samples: int = 5  # Min cells to form a cluster
) -> Tuple[np.ndarray, int]:
    """
    Cluster cell centroids using DBSCAN.

    Args:
        centroids: Nx2 array of centroid coordinates
        eps: Maximum distance between two cells in same cluster
        min_samples: Minimum cells to form a cluster

    Returns:
        labels: Cluster label for each centroid (-1 = noise)
        n_clusters: Number of clusters found
    """
    if len(centroids) < min_samples:
        return np.array([]), 0

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    logger.info(f"DBSCAN found {n_clusters} clusters from {len(centroids)} cells")

    return labels, n_clusters


def check_ring_pattern(
    centroids: np.ndarray,
    cluster_label: int,
    labels: np.ndarray,
    min_angular_coverage: float = 0.5,
    distance_cv_threshold: float = 0.4
) -> Optional[Dict[str, Any]]:
    """
    Check if cells in a cluster form a ring pattern.

    Ring criteria:
    1. Cells are at similar distance from cluster center (low CV)
    2. Cells have good angular coverage around the center

    Args:
        centroids: All cell centroids
        cluster_label: Label of cluster to check
        labels: Cluster labels for all cells
        min_angular_coverage: Minimum fraction of angles covered (0-1)
        distance_cv_threshold: Maximum coefficient of variation for radial distances

    Returns:
        Ring info dict if ring detected, None otherwise
    """
    # Get cells in this cluster
    cluster_mask = labels == cluster_label
    cluster_centroids = centroids[cluster_mask]
    n_cells = len(cluster_centroids)

    if n_cells < 5:
        return None

    # Calculate cluster center
    center_x = cluster_centroids[:, 0].mean()
    center_y = cluster_centroids[:, 1].mean()

    # Calculate radial distances from center
    dx = cluster_centroids[:, 0] - center_x
    dy = cluster_centroids[:, 1] - center_y
    distances = np.sqrt(dx**2 + dy**2)

    # Check distance uniformity (cells at similar radius)
    mean_radius = distances.mean()
    std_radius = distances.std()
    cv_distance = std_radius / (mean_radius + 1e-8)

    # Calculate angular positions
    angles = np.arctan2(dy, dx)  # -pi to pi

    # Check angular coverage - divide circle into bins and count coverage
    n_bins = 12  # 30 degree bins
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    angular_coverage = (hist > 0).sum() / n_bins

    # Ring detection criteria
    is_ring = (
        cv_distance < distance_cv_threshold and
        angular_coverage >= min_angular_coverage and
        mean_radius > 20  # Minimum vessel radius in pixels
    )

    if is_ring:
        # Estimate vessel properties
        # Inner radius = mean - std, outer radius = mean + std
        inner_radius = max(0, mean_radius - std_radius)
        outer_radius = mean_radius + std_radius

        return {
            'center': (center_x, center_y),
            'mean_radius': mean_radius,
            'std_radius': std_radius,
            'inner_radius': inner_radius,
            'outer_radius': outer_radius,
            'cv_distance': cv_distance,
            'angular_coverage': angular_coverage,
            'n_cells': n_cells,
            'cell_indices': np.where(cluster_mask)[0],
            'cluster_label': cluster_label
        }

    return None


def fit_ellipse_to_ring(centroids: np.ndarray, cell_indices: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Fit an ellipse to ring cells for better vessel shape estimation.

    Args:
        centroids: All centroids
        cell_indices: Indices of cells in this ring

    Returns:
        Ellipse parameters if successful
    """
    ring_points = centroids[cell_indices].astype(np.float32)

    if len(ring_points) < 5:
        return None

    try:
        # Fit ellipse using OpenCV
        ellipse = cv2.fitEllipse(ring_points)
        center, axes, angle = ellipse

        return {
            'center': center,
            'major_axis': max(axes),
            'minor_axis': min(axes),
            'angle': angle,
            'aspect_ratio': max(axes) / (min(axes) + 1e-8)
        }
    except cv2.error:
        return None


def detect_vessel_rings(
    centroids: np.ndarray,
    eps_values: List[float] = [50, 100, 150, 200],
    min_samples: int = 5
) -> List[Dict[str, Any]]:
    """
    Detect vessel rings using multi-scale clustering.

    Tries multiple eps values to capture vessels of different sizes.

    Args:
        centroids: Cell centroid coordinates
        eps_values: List of DBSCAN eps values to try
        min_samples: Minimum cells per cluster

    Returns:
        List of detected rings (deduplicated)
    """
    all_rings = []
    used_cells = set()

    for eps in eps_values:
        labels, n_clusters = cluster_cells(centroids, eps=eps, min_samples=min_samples)

        if n_clusters == 0:
            continue

        for cluster_id in range(n_clusters):
            ring = check_ring_pattern(centroids, cluster_id, labels)

            if ring is not None:
                # Check if this ring's cells are already used
                ring_cells = set(ring['cell_indices'])
                overlap = len(ring_cells & used_cells) / len(ring_cells)

                if overlap < 0.5:  # Less than 50% overlap with existing rings
                    # Fit ellipse for better shape
                    ellipse = fit_ellipse_to_ring(centroids, ring['cell_indices'])
                    if ellipse:
                        ring['ellipse'] = ellipse

                    all_rings.append(ring)
                    used_cells.update(ring_cells)

    logger.info(f"Detected {len(all_rings)} vessel rings")
    return all_rings


def create_visualization(
    sma_image: np.ndarray,
    masks: np.ndarray,
    centroids: np.ndarray,
    rings: List[Dict[str, Any]],
    output_path: str,
    pixel_size_um: float = 0.22
):
    """
    Create visualization showing detection results.

    Args:
        sma_image: Original SMA channel image
        masks: Cellpose masks
        centroids: Cell centroids
        rings: Detected vessel rings
        output_path: Path to save visualization
        pixel_size_um: Pixel size in microns
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Normalize SMA for display
    sma_display = sma_image.astype(float)
    p1, p99 = np.percentile(sma_display, [1, 99])
    sma_display = np.clip((sma_display - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Panel 1: Original SMA image
    axes[0].imshow(sma_display, cmap='gray')
    axes[0].set_title('SMA Channel', fontsize=14)
    axes[0].axis('off')

    # Panel 2: Cellpose masks
    # Create colored mask overlay
    mask_colored = np.zeros((*masks.shape, 3), dtype=np.float32)
    n_cells = masks.max()
    np.random.seed(42)
    colors = np.random.rand(n_cells + 1, 3)
    colors[0] = [0, 0, 0]  # Background

    for cell_id in range(1, n_cells + 1):
        mask_colored[masks == cell_id] = colors[cell_id]

    # Overlay on SMA
    overlay = sma_display[:, :, np.newaxis] * 0.5 + mask_colored * 0.5
    axes[1].imshow(overlay)
    axes[1].set_title(f'Cellpose Masks ({n_cells} cells)', fontsize=14)
    axes[1].axis('off')

    # Panel 3: Detected rings
    axes[2].imshow(sma_display, cmap='gray')

    # Plot all centroids
    if len(centroids) > 0:
        axes[2].scatter(centroids[:, 0], centroids[:, 1],
                       c='cyan', s=5, alpha=0.3, label='All cells')

    # Plot detected rings
    ring_colors = plt.cm.tab10(np.linspace(0, 1, max(len(rings), 1)))

    for i, ring in enumerate(rings):
        color = ring_colors[i % len(ring_colors)]
        center = ring['center']

        # Plot ring cells
        ring_centroids = centroids[ring['cell_indices']]
        axes[2].scatter(ring_centroids[:, 0], ring_centroids[:, 1],
                       c=[color], s=30, edgecolors='white', linewidth=0.5)

        # Draw fitted ellipse if available
        if 'ellipse' in ring:
            ellipse_params = ring['ellipse']
            ellipse = Ellipse(
                ellipse_params['center'],
                ellipse_params['major_axis'],
                ellipse_params['minor_axis'],
                angle=ellipse_params['angle'],
                fill=False,
                edgecolor=color,
                linewidth=2,
                linestyle='--'
            )
            axes[2].add_patch(ellipse)
        else:
            # Draw simple circle
            circle = Circle(center, ring['mean_radius'],
                          fill=False, edgecolor=color, linewidth=2, linestyle='--')
            axes[2].add_patch(circle)

        # Mark center
        axes[2].plot(center[0], center[1], 'x', color=color, markersize=10, markeredgewidth=2)

        # Add diameter annotation
        diameter_um = ring['mean_radius'] * 2 * pixel_size_um
        axes[2].annotate(f'{diameter_um:.0f}um',
                        xy=center, xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')

    axes[2].set_title(f'Detected Vessels ({len(rings)} rings)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved visualization to {output_path}")


def print_summary(rings: List[Dict[str, Any]], pixel_size_um: float = 0.22):
    """Print summary of detected vessels."""
    print("\n" + "="*60)
    print("VESSEL DETECTION SUMMARY")
    print("="*60)
    print(f"Total vessels detected: {len(rings)}")
    print()

    if len(rings) == 0:
        print("No vessels found.")
        return

    diameters = []
    for i, ring in enumerate(rings):
        diameter_um = ring['mean_radius'] * 2 * pixel_size_um
        diameters.append(diameter_um)

        print(f"Vessel {i+1}:")
        print(f"  Center: ({ring['center'][0]:.0f}, {ring['center'][1]:.0f}) px")
        print(f"  Diameter: {diameter_um:.1f} um")
        print(f"  Wall cells: {ring['n_cells']}")
        print(f"  Angular coverage: {ring['angular_coverage']*100:.0f}%")
        print(f"  Distance CV: {ring['cv_distance']:.2f}")

        if 'ellipse' in ring:
            e = ring['ellipse']
            print(f"  Ellipse aspect ratio: {e['aspect_ratio']:.2f}")
        print()

    diameters = np.array(diameters)
    print("-"*60)
    print(f"Diameter statistics:")
    print(f"  Mean: {diameters.mean():.1f} um")
    print(f"  Std:  {diameters.std():.1f} um")
    print(f"  Min:  {diameters.min():.1f} um")
    print(f"  Max:  {diameters.max():.1f} um")
    print("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Ring detection from Cellpose-SAM cells')
    parser.add_argument('--tile-x', type=int, default=None, help='Tile X coordinate (default: center)')
    parser.add_argument('--tile-y', type=int, default=None, help='Tile Y coordinate (default: center)')
    parser.add_argument('--tile-size', type=int, default=2000, help='Tile size in pixels')
    parser.add_argument('--output', type=str, default='/tmp/ring_detection_result.png', help='Output path')
    args = parser.parse_args()

    # Configuration
    czi_path = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
    output_path = args.output

    # Channels
    nuclear_channel = 0  # Nuclear (488)
    sma_channel = 2      # SMA (647)

    tile_size = args.tile_size

    # Load CZI to get mosaic dimensions
    from segmentation.io.czi_loader import CZILoader
    loader = CZILoader(czi_path)
    width, height = loader.width, loader.height
    x_start, y_start = loader.x_start, loader.y_start
    pixel_size = loader.get_pixel_size()
    logger.info(f"Pixel size: {pixel_size:.4f} um/px")
    loader.close()

    # Get tile coordinates
    if args.tile_x is not None and args.tile_y is not None:
        tile_x = args.tile_x
        tile_y = args.tile_y
    else:
        # Default: middle of mosaic
        tile_x = x_start + (width // 2) - (tile_size // 2)
        tile_y = y_start + (height // 2) - (tile_size // 2)

    logger.info(f"Extracting {tile_size}x{tile_size} tile from ({tile_x}, {tile_y})")

    # Load tile data
    tiles = load_tile_from_czi(czi_path, tile_x, tile_y, tile_size,
                               channels=[nuclear_channel, sma_channel])

    if sma_channel not in tiles:
        logger.error("Failed to load SMA channel")
        return

    sma_tile = tiles[sma_channel]
    logger.info(f"SMA tile: shape={sma_tile.shape}, dtype={sma_tile.dtype}")
    logger.info(f"SMA intensity range: [{sma_tile.min()}, {sma_tile.max()}]")

    # Run Cellpose-SAM detection on SMA channel
    masks, n_cells = run_cellpose_detection(
        sma_tile,
        model_type='cpsam',
        diameter=None,  # Auto-detect
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )

    if n_cells == 0:
        logger.warning("No cells detected!")
        return

    # Extract centroids
    centroids = extract_cell_centroids(masks)
    logger.info(f"Extracted {len(centroids)} cell centroids")

    # Detect vessel rings using multi-scale clustering
    rings = detect_vessel_rings(
        centroids,
        eps_values=[30, 50, 75, 100, 150, 200],  # Different scales
        min_samples=5
    )

    # Create visualization
    create_visualization(
        sma_tile,
        masks,
        centroids,
        rings,
        output_path,
        pixel_size_um=pixel_size
    )

    # Print summary
    print_summary(rings, pixel_size_um=pixel_size)


if __name__ == "__main__":
    main()
