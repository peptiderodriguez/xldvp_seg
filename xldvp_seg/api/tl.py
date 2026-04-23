"""Analysis tools operating on SlideAnalysis objects.

Each function calls the underlying script's internal work function directly
(not through argparse). Functions mutate the slide's detections in-place
and return the slide for chaining.

Usage:
    from xldvp_seg.core import SlideAnalysis
    from xldvp_seg.api import tl

    slide = SlideAnalysis.load("output/...")
    tl.markers(slide, marker_channels=[1, 2], marker_names=["NeuN", "tdTomato"])
    tl.score(slide, classifier="classifiers/rf_morph.pkl")
    tl.spatial(slide, output_dir="results/spatial/")
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.utils.logging import get_logger
from xldvp_seg.utils.seeding import resolve_seed

if TYPE_CHECKING:
    from xldvp_seg.core.slide_analysis import SlideAnalysis

logger = get_logger(__name__)


def markers(
    slide: SlideAnalysis,
    marker_channels: list[int],
    marker_names: list[str],
    method: str = "snr",
    snr_threshold: float = 1.5,
    czi_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> SlideAnalysis:
    """Classify markers as positive/negative per channel.

    Calls classify_single_marker() for each marker. Mutates detections
    in-place with {marker}_class, {marker}_value, and marker_profile
    fields (stored in each detection's ``features`` sub-dict).

    Note: defaults to ``intensity_feature='snr'`` (recommended for most
    use cases). The lower-level ``classify_single_marker`` defaults to
    ``'mean'``.

    Args:
        slide: SlideAnalysis object.
        marker_channels: List of channel indices (e.g., [1, 2]).
        marker_names: List of marker names (e.g., ["NeuN", "tdTomato"]).
        method: Classification method ("snr", "otsu", "otsu_half", "gmm").
        snr_threshold: SNR threshold for 'snr' method (default: 1.5).
        czi_path: Path to CZI for channel metadata resolution.
        output_dir: Directory for summary output (optional).

    Returns:
        slide (mutated with marker classifications).
    """
    from xldvp_seg.analysis.marker_classification import classify_single_marker

    detections = slide.detections
    if not detections:
        logger.warning("No detections to classify")
        return slide

    _tmp_ctx = None
    if output_dir:
        out_dir = Path(output_dir)
    else:
        _tmp_ctx = tempfile.TemporaryDirectory()
        out_dir = Path(_tmp_ctx.name)

    try:
        summaries = []
        for ch, name in zip(marker_channels, marker_names):
            logger.info("Classifying marker %s (channel %d, method=%s)", name, ch, method)
            summary = classify_single_marker(
                detections=detections,
                channel=ch,
                marker_name=name,
                method=method,
                output_dir=out_dir,
                snr_threshold=snr_threshold,
                intensity_feature=kwargs.get("intensity_feature", "snr"),
                **{k: v for k, v in kwargs.items() if k != "intensity_feature"},
            )
            summaries.append(summary)
            logger.info(
                "  %s: %d+ / %d- (threshold=%.2f)",
                name,
                summary.get("n_positive", 0),
                summary.get("n_negative", 0),
                summary.get("threshold", 0),
            )
    finally:
        if _tmp_ctx is not None:
            _tmp_ctx.cleanup()

    # Build marker_profile from all markers (even for single marker)
    for det in detections:
        feat = det.setdefault("features", {})
        parts = []
        for name in marker_names:
            cls = feat.get(f"{name}_class", "negative")
            parts.append(f"{name}+" if cls == "positive" else f"{name}-")
        feat["marker_profile"] = "/".join(parts)

    # Invalidate cached features_df since detections changed
    slide._features_df = None
    logger.info("Marker classification complete: %d detections", len(detections))
    return slide


def score(
    slide: SlideAnalysis,
    classifier: str | Path,
    score_field: str = "rf_prediction",
    **kwargs: Any,
) -> SlideAnalysis:
    """Score detections with a trained RF classifier.

    Calls extract_feature_matrix + pipeline.predict_proba directly.
    Mutates detections in-place with rf_prediction field.

    Missing feature values are defaulted to 0.0 so that all detections
    are scored (matching ``extract_feature_matrix`` behavior).

    Args:
        slide: SlideAnalysis object.
        classifier: Path to .pkl classifier file.
        score_field: Field name for the score (default: "rf_prediction").

    Returns:
        slide (mutated with scores).
    """
    from xldvp_seg.utils.detection_utils import load_rf_classifier

    clf_data = load_rf_classifier(str(classifier))
    pipeline = clf_data["pipeline"]
    feature_names = clf_data["feature_names"]

    detections = slide.detections
    if not detections:
        logger.warning("No detections to score")
        return slide

    # Extract feature matrix
    import numpy as np

    X_rows = []
    valid_indices = []
    n_zeroed = 0
    zeroed_names: list[str] = []
    for i, det in enumerate(detections):
        features = det.get("features", {})
        row = []
        det_had_missing = False
        for fn in feature_names:
            val = features.get(fn)
            if val is None:
                val = 0.0
                if not det_had_missing:
                    det_had_missing = True
                    n_zeroed += 1
                if fn not in zeroed_names:
                    zeroed_names.append(fn)
            row.append(float(val))
        X_rows.append(row)
        valid_indices.append(i)

    if n_zeroed > 0:
        logger.warning(
            "%d detections had missing features defaulted to 0.0 (%s...)",
            n_zeroed,
            ", ".join(zeroed_names[:3]),
        )

    if not X_rows:
        logger.warning("No detections have all required features")
        return slide

    X = np.array(X_rows)
    proba = pipeline.predict_proba(X)
    if proba.shape[1] == 1:
        # Single class in training data — no positive class probability
        scores = np.zeros(len(X))
    else:
        scores = proba[:, 1]

    for idx, s in zip(valid_indices, scores):
        detections[idx][score_field] = float(s)

    n_above_50 = sum(1 for s in scores if s >= 0.5)
    logger.info(
        "Scored %d/%d detections (%.1f%% above 0.5)",
        len(valid_indices),
        len(detections),
        100 * n_above_50 / len(scores) if scores.size else 0,
    )

    slide._features_df = None
    return slide


def train(
    slide: SlideAnalysis,
    annotations: str | Path,
    feature_set: str = "morph",
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train RF classifier from annotations.

    Args:
        slide: SlideAnalysis with detections.
        annotations: Path to annotations JSON (from HTML viewer export).
        feature_set: Feature set to use ("morph", "morph_sam2", "all").
        output_path: Where to save the .pkl classifier.

    Returns:
        Dict with training metrics (f1, precision, recall, etc.)
    """
    import joblib

    from xldvp_seg.training.feature_loader import load_features_and_annotations

    det_path = slide.detections_path
    if det_path is None:
        raise ConfigError("SlideAnalysis has no detections_path. Save detections first.")

    X, y, feature_names = load_features_and_annotations(
        str(det_path), str(annotations), feature_set=feature_set
    )

    if len(X) == 0:
        raise ConfigError("No annotated detections found matching the detections file.")

    # Train RF directly — no scaler needed (RF is invariant to monotonic transforms)

    _rs = resolve_seed(kwargs.get("seed"), caller="tl.train")
    rf = RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 20),
        random_state=_rs,
        n_jobs=-1,
    )

    # CV on the bare RF — use StratifiedKFold with shuffle so seed matters.
    from sklearn.model_selection import StratifiedKFold

    _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_rs)
    cv_scores = cross_val_score(rf, X, y, cv=_cv, scoring="f1")
    logger.info("5-fold CV F1: %.3f +/- %.3f", cv_scores.mean(), cv_scores.std())

    # Fit on all data
    rf.fit(X, y)

    # Save
    if output_path is None:
        if slide.output_dir:
            output_path = Path(slide.output_dir) / f"rf_{feature_set}.pkl"
        else:
            output_path = Path(tempfile.mkdtemp()) / f"rf_{feature_set}.pkl"
            logger.warning(
                "No output_dir on slide — saving classifier to temp dir: %s",
                output_path.parent,
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": rf,
            "classifier": rf,  # legacy compat key for load_rf_classifier
            "feature_names": feature_names,
            "feature_set": feature_set,
            "feature_extraction": "original_mask",
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        },
        str(output_path),
    )

    logger.info("Classifier saved to %s", output_path)
    return {
        "classifier_path": str(output_path),
        "feature_set": feature_set,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
    }


def cluster(
    slide: SlideAnalysis,
    feature_groups: str = "morph,sam2,channel",
    methods: str = "both",
    resolution: float = 1.0,
    output_dir: str | Path | None = None,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    clustering: str = "leiden",
    **kwargs: Any,
) -> SlideAnalysis:
    """Feature clustering with UMAP/t-SNE + Leiden/HDBSCAN.

    Args:
        slide: SlideAnalysis object.
        feature_groups: Comma-separated feature groups to include. Valid values:
            "morph" (shape + color), "shape", "color", "sam2", "channel"
            (per-channel intensities), "deep" (ResNet + DINOv2). Example:
            ``"morph,sam2,channel"``. Default: ``"morph,sam2,channel"``.
        methods: Dim reduction methods ("umap", "tsne", "both").
        resolution: Leiden resolution (default: 1.0).
        output_dir: Output directory for plots and clustered JSON.
        n_neighbors: UMAP n_neighbors (default: 30).
        min_dist: UMAP min_dist (default: 0.1).
        clustering: Clustering method ("leiden", "hdbscan").

    Returns:
        slide (mutated with cluster labels).
    """
    from xldvp_seg.analysis.cluster_features import run_clustering

    if slide.detections_path is None:
        raise ConfigError("SlideAnalysis has no detections_path. Save detections first.")

    if output_dir is None:
        output_dir = (
            str(slide.output_dir / "clustering") if slide.output_dir else tempfile.mkdtemp()
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_clustering(
        detections=str(slide.detections_path),
        output_dir=str(output_dir),
        feature_groups=feature_groups,
        methods=methods,
        clustering=clustering,
        resolution=resolution,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        threshold=kwargs.get("threshold", 0.0),
        min_cluster_size=kwargs.get("min_cluster_size", 15),
        marker_channels=kwargs.get("marker_channels", ""),
        exclude_channels=kwargs.get("exclude_channels", ""),
        marker_rings=not kwargs.get("no_marker_rings", False),
        trajectory=kwargs.get("trajectory", False),
        root_cluster=kwargs.get("root_cluster", None),
        spatial_smooth=kwargs.get("spatial_smooth", False),
        smooth_k=kwargs.get("smooth_k", 15),
        smooth_sim_threshold=kwargs.get("smooth_sim_threshold", 0.5),
        marker_only=kwargs.get("marker_only", False),
        gate_channel=kwargs.get("gate_channel", None),
        gate_percentile=kwargs.get("gate_percentile", 90),
        perplexity=kwargs.get("perplexity", 30),
        tsne_n_iter=kwargs.get("tsne_n_iter", 1000),
        min_samples=kwargs.get("min_samples", None),
        subcluster=kwargs.get("subcluster", False),
        subcluster_features=kwargs.get("subcluster_features", "shape,sam2"),
        subcluster_min_size=kwargs.get("subcluster_min_size", 50),
    )
    logger.info("Clustering complete. Output: %s", output_dir)

    # Reload detections with cluster labels
    from xldvp_seg.utils.json_utils import fast_json_load

    clustered_files = sorted(Path(output_dir).glob("*_clustered.json"))
    if clustered_files:
        slide._detections = fast_json_load(clustered_files[0])
        slide._features_df = None
        logger.info("Loaded clustered detections: %d", len(slide._detections))

    return slide


def spatial(
    slide: SlideAnalysis,
    output_dir: str | Path | None = None,
    pixel_size: float | None = None,
    marker_filter: str | None = None,
    max_edge_distance: float = 50.0,
    **kwargs: Any,
) -> SlideAnalysis:
    """Spatial network analysis (Delaunay graph + communities).

    Args:
        slide: SlideAnalysis object.
        output_dir: Output directory.
        pixel_size: Pixel size in um (auto-detected from slide if None).
        marker_filter: Filter string (e.g., "NeuN_class==positive").
        max_edge_distance: Max edge distance in um (default: 50).

    Returns:
        slide (mutated with spatial features).
    """
    from xldvp_seg.analysis.spatial_network import run_spatial_network

    if output_dir is None:
        output_dir = str(slide.output_dir / "spatial") if slide.output_dir else tempfile.mkdtemp()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    detections = slide.detections
    px = pixel_size if pixel_size is not None else (slide.pixel_size_um or None)

    result_detections = run_spatial_network(
        detections,
        output_dir,
        pixel_size=px,
        marker_filter=marker_filter,
        max_edge_distance=max_edge_distance,
        min_component_cells=kwargs.get("min_component_cells", 3),
    )

    if result_detections is not None:
        slide._detections = result_detections
        slide._features_df = None

    logger.info("Spatial analysis complete. Output: %s", output_dir)
    return slide
