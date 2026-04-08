"""Slide-level feature aggregation for cohort studies.

Aggregates per-cell features to slide-level summaries for cohort analysis
(e.g., treatment vs control across 16 slides).

Usage:
    from xldvp_seg.analysis.aggregation import aggregate_slide, aggregate_cohort
    from xldvp_seg.core import SlideAnalysis

    slides = [SlideAnalysis.load(d) for d in slide_dirs]
    cohort = aggregate_cohort(slides, group_by="marker_profile")
    adata = cohort_to_anndata(cohort)
"""

import numpy as np
import pandas as pd

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_slide(slide, group_by=None):
    """Aggregate per-cell features to slide-level summary.

    Args:
        slide: SlideAnalysis object.
        group_by: Column to group by (e.g., 'marker_profile'). If None,
                  aggregates all cells into one row.

    Returns:
        DataFrame with aggregated stats (mean, median, std, count per feature).
    """
    df = slide.features_df
    if df.empty:
        return pd.DataFrame()

    # Use morph + channel features, skip embeddings (too many columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_cols = [
        c
        for c in numeric_cols
        if not c.startswith("sam2_") and not c.startswith("resnet_") and not c.startswith("dinov2_")
    ]

    if group_by and group_by in df.columns:
        agg = df.groupby(group_by)[summary_cols].agg(["mean", "median", "std", "count"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        return agg
    else:
        stats = {}
        for col in summary_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                stats[f"{col}_mean"] = vals.mean()
                stats[f"{col}_median"] = vals.median()
                stats[f"{col}_std"] = vals.std()
                stats[f"{col}_count"] = len(vals)
        result = pd.DataFrame([stats])
        result.index = [slide.slide_name]
        return result


def aggregate_cohort(slides, group_by=None):
    """Aggregate multiple slides into cohort-level feature matrix.

    Args:
        slides: List of SlideAnalysis objects.
        group_by: Optional column to group by within each slide.

    Returns:
        DataFrame: rows=slides (or slide x group), columns=aggregated features.
    """
    rows = []
    for slide in slides:
        slide_agg = aggregate_slide(slide, group_by=group_by)
        if slide_agg.empty:
            continue
        if group_by:
            for idx, row in slide_agg.iterrows():
                row_dict = row.to_dict()
                row_dict["slide_name"] = slide.slide_name
                row_dict["group"] = idx
                rows.append(row_dict)
        else:
            row_dict = slide_agg.mean(numeric_only=True).to_dict()
            row_dict["slide_name"] = slide.slide_name
            row_dict["n_cells"] = slide.n_detections
            rows.append(row_dict)

    result = pd.DataFrame(rows)
    if "slide_name" in result.columns:
        result = result.set_index("slide_name")
    logger.info("Cohort aggregation: %d slides -> %s", len(slides), result.shape)
    return result


def cohort_to_anndata(cohort_df, metadata=None):
    """Convert cohort DataFrame to AnnData for slide-level analysis.

    Args:
        cohort_df: Output from aggregate_cohort().
        metadata: Optional per-slide metadata DataFrame (condition, treatment, etc.)

    Returns:
        AnnData with X=aggregated features, obs=metadata.
    """
    import anndata

    # Exclude metadata columns from X (they belong in obs, not feature matrix)
    count_cols = {c for c in cohort_df.columns if c.endswith("_count")}
    obs_numeric = {"n_cells", "pool_n_cells", "pool_total_area_um2"} | count_cols
    numeric_cols = [
        c for c in cohort_df.select_dtypes(include=[np.number]).columns if c not in obs_numeric
    ]
    X = cohort_df[numeric_cols].values.astype(np.float32)
    missing_mask = np.isnan(X)
    nan_counts = missing_mask.sum(axis=0)
    if nan_counts.any():
        nan_features = [numeric_cols[i] for i in range(len(nan_counts)) if nan_counts[i] > 0]
        logger.warning(
            "NaN values in %d features replaced with 0: %s",
            len(nan_features),
            nan_features[:5],
        )
    X = np.nan_to_num(X, nan=0.0)

    obs = pd.DataFrame(index=cohort_df.index)
    if "group" in cohort_df.columns:
        obs["group"] = cohort_df["group"].values
    if "n_cells" in cohort_df.columns:
        obs["n_cells"] = cohort_df["n_cells"].values
    if metadata is not None:
        obs = obs.join(metadata, how="left")

    adata = anndata.AnnData(X=X, obs=obs)
    adata.var_names = list(numeric_cols)

    if missing_mask.any():
        adata.layers["missing"] = missing_mask

    logger.info("Cohort AnnData: %d slides x %d features", adata.n_obs, adata.n_vars)
    return adata
