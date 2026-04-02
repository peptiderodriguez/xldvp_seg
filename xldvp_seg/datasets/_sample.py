"""Generate synthetic detection data for testing and demonstration.

Usage:
    from segmentation.datasets import sample
    data = sample()
    detections = data["detections"]  # list of detection dicts
    print(f"{len(detections)} synthetic detections across {data['n_clusters']} clusters")
"""

from typing import Any

import numpy as np


def sample(
    n_cells: int = 500,
    n_clusters: int = 5,
    n_channels: int = 4,
    pixel_size_um: float = 0.325,
    image_size: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic detection data with known cluster structure.

    Creates detections with:
    - Morphological features (area, solidity, eccentricity, etc.)
    - Per-channel intensity features (ch0_mean, ch0_median, etc.)
    - SAM2-style embeddings (256D random)
    - Global coordinates spread across a synthetic slide
    - Known cluster labels for validation

    Args:
        n_cells: Number of synthetic detections
        n_clusters: Number of distinct cell populations
        n_channels: Number of fluorescence channels
        pixel_size_um: Pixel size in microns
        image_size: Synthetic image dimension in pixels
        seed: Random seed for reproducibility

    Returns:
        Dict with keys:
            - 'detections': List of detection dicts (same format as pipeline output)
            - 'n_clusters': Number of ground-truth clusters
            - 'cluster_labels': np.array of cluster assignments
            - 'pixel_size_um': Pixel size used
            - 'channel_names': List of synthetic channel names
    """
    rng = np.random.RandomState(seed)

    channel_names = [f"ch{i}" for i in range(n_channels)]

    # Generate cluster centers in position space
    cluster_centers_px = rng.uniform(500, image_size - 500, size=(n_clusters, 2))

    # Assign cells to clusters
    labels = rng.choice(n_clusters, size=n_cells)

    # Generate positions around cluster centers (Gaussian spread)
    positions = np.zeros((n_cells, 2))
    for i in range(n_cells):
        spread = rng.uniform(100, 500)  # variable cluster tightness
        positions[i] = cluster_centers_px[labels[i]] + rng.randn(2) * spread
    positions = np.clip(positions, 0, image_size)

    # Generate per-cluster feature profiles
    # Each cluster has distinct morphology and channel intensities
    cluster_morph = {
        "area": rng.uniform(200, 2000, n_clusters),
        "solidity": rng.uniform(0.7, 0.99, n_clusters),
        "eccentricity": rng.uniform(0.1, 0.8, n_clusters),
        "perimeter": rng.uniform(50, 200, n_clusters),
        "major_axis_length": rng.uniform(15, 60, n_clusters),
        "minor_axis_length": rng.uniform(10, 40, n_clusters),
    }
    cluster_channel_means = rng.uniform(100, 5000, size=(n_clusters, n_channels))

    detections: list[dict[str, Any]] = []
    for i in range(n_cells):
        c = labels[i]
        x, y = positions[i]

        features: dict[str, Any] = {}
        # Morphological features (with noise around cluster center)
        for key, centers in cluster_morph.items():
            noise_scale = centers[c] * 0.15
            val = max(0.01, centers[c] + rng.randn() * noise_scale)
            # Clamp eccentricity to valid range [0, 1]
            if key == "eccentricity":
                val = min(1.0, max(0.0, val))
            features[key] = val

        features["area_um2"] = features["area"] * pixel_size_um**2

        # Per-channel intensity features
        for ch_idx in range(n_channels):
            base = cluster_channel_means[c, ch_idx]
            noise = base * 0.2
            mean_val = max(0, base + rng.randn() * noise)
            features[f"ch{ch_idx}_mean"] = mean_val
            features[f"ch{ch_idx}_median"] = mean_val * rng.uniform(0.8, 1.2)
            features[f"ch{ch_idx}_std"] = mean_val * rng.uniform(0.1, 0.5)
            features[f"ch{ch_idx}_max"] = mean_val * rng.uniform(1.5, 3.0)
            features[f"ch{ch_idx}_min"] = mean_val * rng.uniform(0.0, 0.3)
            features[f"ch{ch_idx}_snr"] = rng.uniform(0.5, 5.0)
            features[f"ch{ch_idx}_background"] = rng.uniform(50, 500)
            features[f"ch{ch_idx}_median_raw"] = mean_val + features[f"ch{ch_idx}_background"]

        # SAM2-style embeddings (256D, cluster-structured)
        sam2_base = rng.randn(256) * 0.1  # per-cell noise
        # Add cluster-specific component
        cluster_embedding = np.random.RandomState(seed + c).randn(256)
        sam2 = sam2_base + cluster_embedding * 0.5
        for j in range(256):
            features[f"sam2_{j}"] = float(sam2[j])

        det: dict[str, Any] = {
            "id": f"sample_cell_{i}",
            "uid": f"sample_cell_{i}",
            "global_center": [float(x), float(y)],
            "global_center_um": [float(x * pixel_size_um), float(y * pixel_size_um)],
            "features": features,
            "cell_type": "cell",
            "cluster_label": int(c),  # ground truth
        }
        detections.append(det)

    return {
        "detections": detections,
        "n_clusters": n_clusters,
        "cluster_labels": labels,
        "pixel_size_um": pixel_size_um,
        "channel_names": channel_names,
        "image_size_px": image_size,
    }
