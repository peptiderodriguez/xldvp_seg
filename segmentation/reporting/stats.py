"""
Statistical analysis functions for vessel segmentation results.

Provides summary statistics, distribution analysis, and batch comparisons
for vessel detection data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class VesselStatistics:
    """
    Container for computed vessel statistics.

    Attributes:
        vessel_count: Total number of vessels
        diameter_stats: Dict with mean, std, min, max, median, q25, q75
        wall_thickness_stats: Dict with mean, std, min, max, quantiles
        area_stats: Dict with lumen and wall area statistics
        vessel_types: Dict mapping vessel type to count
        quality_stats: Dict with confidence and completeness metrics
        slide_name: Optional slide identifier
    """

    vessel_count: int = 0
    diameter_stats: Dict[str, float] = field(default_factory=dict)
    wall_thickness_stats: Dict[str, float] = field(default_factory=dict)
    area_stats: Dict[str, float] = field(default_factory=dict)
    vessel_types: Dict[str, int] = field(default_factory=dict)
    quality_stats: Dict[str, float] = field(default_factory=dict)
    slide_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for JSON serialization."""
        return {
            "vessel_count": self.vessel_count,
            "diameter_stats": self.diameter_stats,
            "wall_thickness_stats": self.wall_thickness_stats,
            "area_stats": self.area_stats,
            "vessel_types": self.vessel_types,
            "quality_stats": self.quality_stats,
            "slide_name": self.slide_name,
        }


def compute_summary_statistics(
    detections: List[Dict[str, Any]],
    slide_name: Optional[str] = None,
) -> VesselStatistics:
    """
    Compute comprehensive summary statistics from vessel detections.

    Args:
        detections: List of detection dicts from vessel_detections.json
        slide_name: Optional slide identifier for the statistics

    Returns:
        VesselStatistics object with all computed metrics
    """
    if not detections:
        return VesselStatistics(slide_name=slide_name)

    # Extract features from detections
    features_list = [d.get("features", d) for d in detections]

    # Diameter statistics
    diameters = _extract_values(features_list, "outer_diameter_um")
    diameter_stats = _compute_distribution_stats(diameters) if diameters else {}

    # Wall thickness statistics
    wall_thicknesses = _extract_values(features_list, "wall_thickness_mean_um")
    wall_stats = compute_wall_thickness_quantiles(features_list)

    # Area statistics
    lumen_areas = _extract_values(features_list, "lumen_area_um2")
    wall_areas = _extract_values(features_list, "wall_area_um2")
    area_stats = {
        "lumen_area_mean": float(np.mean(lumen_areas)) if lumen_areas else 0,
        "lumen_area_std": float(np.std(lumen_areas)) if lumen_areas else 0,
        "wall_area_mean": float(np.mean(wall_areas)) if wall_areas else 0,
        "wall_area_std": float(np.std(wall_areas)) if wall_areas else 0,
    }

    # Vessel type breakdown
    vessel_types = compute_vessel_type_breakdown(features_list)

    # Quality metrics
    quality_stats = compute_quality_metrics(features_list)

    return VesselStatistics(
        vessel_count=len(detections),
        diameter_stats=diameter_stats,
        wall_thickness_stats=wall_stats,
        area_stats=area_stats,
        vessel_types=vessel_types,
        quality_stats=quality_stats,
        slide_name=slide_name,
    )


def compute_diameter_distribution(
    features_list: List[Dict[str, Any]],
    bins: int = 30,
    range_um: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Compute diameter distribution histogram data.

    Args:
        features_list: List of feature dicts from detections
        bins: Number of histogram bins
        range_um: Optional (min, max) range in microns

    Returns:
        Dict with 'edges', 'counts', 'bin_centers', and 'stats' keys
    """
    diameters = _extract_values(features_list, "outer_diameter_um")
    if not diameters:
        return {"edges": [], "counts": [], "bin_centers": [], "stats": {}}

    diameters = np.array(diameters)

    if range_um is None:
        range_um = (diameters.min(), diameters.max())

    counts, edges = np.histogram(diameters, bins=bins, range=range_um)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    return {
        "edges": edges.tolist(),
        "counts": counts.tolist(),
        "bin_centers": bin_centers.tolist(),
        "stats": _compute_distribution_stats(diameters.tolist()),
    }


def compute_wall_thickness_quantiles(
    features_list: List[Dict[str, Any]],
    quantiles: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute wall thickness quantile statistics.

    Args:
        features_list: List of feature dicts from detections
        quantiles: List of quantile values (default: [0.1, 0.25, 0.5, 0.75, 0.9])

    Returns:
        Dict with quantile keys (e.g., 'q10', 'q25', etc.) and values in microns
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    wall_thicknesses = _extract_values(features_list, "wall_thickness_mean_um")
    if not wall_thicknesses:
        return {}

    wall_arr = np.array(wall_thicknesses)
    result = {
        "mean": float(np.mean(wall_arr)),
        "std": float(np.std(wall_arr)),
        "min": float(np.min(wall_arr)),
        "max": float(np.max(wall_arr)),
    }

    for q in quantiles:
        key = f"q{int(q * 100)}"
        result[key] = float(np.percentile(wall_arr, q * 100))

    return result


def compute_vessel_type_breakdown(
    features_list: List[Dict[str, Any]],
    diameter_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, int]:
    """
    Compute vessel type counts and percentages.

    If vessel_type is not present in features, classifies by diameter:
    - capillary: < 10 um
    - arteriole: 10-100 um
    - artery: > 100 um

    Args:
        features_list: List of feature dicts from detections
        diameter_thresholds: Optional custom thresholds dict
            e.g., {"capillary": (0, 10), "arteriole": (10, 100), "artery": (100, inf)}

    Returns:
        Dict mapping vessel type to count
    """
    if diameter_thresholds is None:
        diameter_thresholds = {
            "capillary": (0, 10),
            "arteriole": (10, 100),
            "artery": (100, float("inf")),
        }

    counts: Dict[str, int] = {vtype: 0 for vtype in diameter_thresholds}
    counts["unknown"] = 0

    for feat in features_list:
        # Check if vessel_type already assigned
        vessel_type = feat.get("vessel_type", "unknown")

        if vessel_type == "unknown":
            # Classify by diameter
            diameter = feat.get("outer_diameter_um", 0)
            classified = False
            for vtype, (min_d, max_d) in diameter_thresholds.items():
                if min_d <= diameter < max_d:
                    vessel_type = vtype
                    classified = True
                    break
            if not classified:
                vessel_type = "unknown"

        if vessel_type in counts:
            counts[vessel_type] += 1
        else:
            counts[vessel_type] = 1

    return counts


def compute_quality_metrics(
    features_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute quality-related statistics from detections.

    Includes:
    - Confidence distribution (high/medium/low counts)
    - Ring completeness statistics
    - CD31 validation rates (if available)

    Args:
        features_list: List of feature dicts from detections

    Returns:
        Dict with quality metric statistics
    """
    if not features_list:
        return {}

    # Confidence distribution
    confidence_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for feat in features_list:
        conf = feat.get("confidence", "unknown")
        if conf in confidence_counts:
            confidence_counts[conf] += 1
        else:
            confidence_counts["unknown"] += 1

    # Ring completeness statistics
    ring_completeness = _extract_values(features_list, "ring_completeness")
    completeness_stats = _compute_distribution_stats(ring_completeness) if ring_completeness else {}

    # CD31 validation (if available)
    cd31_validated = [feat.get("cd31_validated") for feat in features_list if "cd31_validated" in feat]
    cd31_stats = {}
    if cd31_validated:
        cd31_stats = {
            "validated_count": sum(1 for v in cd31_validated if v is True),
            "not_validated_count": sum(1 for v in cd31_validated if v is False),
            "validation_rate": sum(1 for v in cd31_validated if v is True) / len(cd31_validated),
        }

    # Circularity statistics
    circularity = _extract_values(features_list, "circularity")
    circularity_stats = _compute_distribution_stats(circularity) if circularity else {}

    # Aspect ratio statistics
    aspect_ratios = _extract_values(features_list, "aspect_ratio")
    aspect_stats = _compute_distribution_stats(aspect_ratios) if aspect_ratios else {}

    return {
        "confidence_distribution": confidence_counts,
        "ring_completeness": completeness_stats,
        "cd31_validation": cd31_stats,
        "circularity": circularity_stats,
        "aspect_ratio": aspect_stats,
    }


def compute_batch_comparison(
    slide_statistics: List[VesselStatistics],
) -> Dict[str, Any]:
    """
    Compute comparison statistics across multiple slides.

    Args:
        slide_statistics: List of VesselStatistics objects for each slide

    Returns:
        Dict with batch-level comparison metrics
    """
    if not slide_statistics:
        return {}

    slide_names = [s.slide_name or f"Slide_{i}" for i, s in enumerate(slide_statistics)]

    # Vessel counts per slide
    vessel_counts = [s.vessel_count for s in slide_statistics]

    # Mean diameters per slide
    mean_diameters = [
        s.diameter_stats.get("mean", 0) for s in slide_statistics
    ]

    # Mean wall thickness per slide
    mean_wall_thickness = [
        s.wall_thickness_stats.get("mean", 0) for s in slide_statistics
    ]

    # Vessel type breakdown per slide
    vessel_type_data: Dict[str, List[int]] = {
        "capillary": [],
        "arteriole": [],
        "artery": [],
        "unknown": [],
    }
    for stats in slide_statistics:
        for vtype in vessel_type_data:
            vessel_type_data[vtype].append(stats.vessel_types.get(vtype, 0))

    # Quality metrics per slide
    high_confidence_counts = []
    for stats in slide_statistics:
        conf_dist = stats.quality_stats.get("confidence_distribution", {})
        high_confidence_counts.append(conf_dist.get("high", 0))

    return {
        "slide_names": slide_names,
        "vessel_counts": vessel_counts,
        "mean_diameters": mean_diameters,
        "mean_wall_thickness": mean_wall_thickness,
        "vessel_type_breakdown": vessel_type_data,
        "high_confidence_counts": high_confidence_counts,
        "total_vessels": sum(vessel_counts),
        "avg_vessels_per_slide": float(np.mean(vessel_counts)) if vessel_counts else 0,
        "std_vessels_per_slide": float(np.std(vessel_counts)) if vessel_counts else 0,
    }


def _extract_values(
    features_list: List[Dict[str, Any]],
    key: str,
) -> List[float]:
    """
    Extract numeric values for a specific key from feature dicts.

    Args:
        features_list: List of feature dicts
        key: Key to extract values for

    Returns:
        List of numeric values (excludes None and non-numeric values)
    """
    values = []
    for feat in features_list:
        val = feat.get(key)
        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
            values.append(float(val))
    return values


def _compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    """
    Compute standard distribution statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dict with mean, std, min, max, median, q25, q75
    """
    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "count": len(values),
    }
