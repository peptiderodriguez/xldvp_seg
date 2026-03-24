"""Multi-omic linker: bridge morphological features to proteomics.

For DVP (Deep Visual Proteomics): links cell morphology from the pipeline
to mass-spec proteomic profiles from LMD-cut cells.

Usage:
    from segmentation.analysis.omic_linker import OmicLinker
    from segmentation.core import SlideAnalysis

    slide = SlideAnalysis.load("output/...")
    linker = OmicLinker.from_slide(slide)
    linker.load_proteomics("proteomics.csv")
    linker.load_well_mapping("lmd_export/")
    linked = linker.link()
    diff = linker.differential_features("marker_profile", "NeuN+", "NeuN-")
    corr = linker.correlate(method="spearman")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


class OmicLinker:
    """Links morphological features to mass-spec proteomics data."""

    def __init__(self, features_df=None, detections=None):
        self._features_df = features_df
        self._detections = detections
        self._proteomics = None
        self._well_mapping = None
        self._linked = None

    @classmethod
    def from_slide(cls, slide):
        """Create from a SlideAnalysis object."""
        return cls(features_df=slide.features_df, detections=slide.detections)

    @classmethod
    def from_detections(cls, detections):
        """Create from a detections list."""
        return cls(detections=detections)

    def load_proteomics(self, path, well_column="well_id"):
        """Load proteomics CSV. Rows=wells, columns=proteins."""
        self._proteomics = pd.read_csv(path, index_col=well_column)
        logger.info("Loaded proteomics: %d wells, %d proteins", *self._proteomics.shape)

    def load_well_mapping(self, lmd_dir):
        """Load detection -> well mapping from LMD export."""
        lmd_dir = Path(lmd_dir)
        mapping_files = (
            list(lmd_dir.glob("*well_mapping*.json"))
            + list(lmd_dir.glob("*well_assignment*.json"))
        )
        if not mapping_files:
            if self._detections:
                self._well_mapping = {
                    d.get("uid", d.get("id", "")): d.get("well")
                    for d in self._detections if d.get("well")
                }
                logger.info("Built well mapping from detections: %d entries",
                            len(self._well_mapping))
                return
            raise FileNotFoundError(f"No well mapping files found in {lmd_dir}")
        from segmentation.utils.json_utils import fast_json_load
        mapping = fast_json_load(mapping_files[0])
        if isinstance(mapping, list):
            self._well_mapping = {
                m["uid"]: m["well"] for m in mapping
                if "uid" in m and "well" in m
            }
        elif isinstance(mapping, dict):
            self._well_mapping = mapping
        logger.info("Loaded well mapping: %d entries from %s",
                    len(self._well_mapping), mapping_files[0].name)

    def link(self):
        """Join morphological features to proteomics by well.

        Returns:
            DataFrame with rows=wells, columns=[morph features + proteins].
        """
        if self._features_df is None:
            raise ValueError("No features loaded. Use from_slide() or from_detections().")
        if self._proteomics is None:
            raise ValueError("No proteomics loaded. Call load_proteomics() first.")
        if self._well_mapping is None:
            raise ValueError("No well mapping. Call load_well_mapping() first.")

        df = self._features_df.copy()
        df["well"] = df.index.map(self._well_mapping)
        df = df.dropna(subset=["well"])

        morph_cols = [c for c in df.columns
                      if c != "well" and not c.endswith("_class")]
        numeric_morph = df[morph_cols].select_dtypes(include=[np.number]).columns
        well_morph = df.groupby("well")[numeric_morph].mean()

        self._linked = well_morph.join(self._proteomics, how="inner")
        logger.info("Linked: %d wells with both morph and proteomics",
                    len(self._linked))
        return self._linked

    def differential_features(self, group_col, group_a, group_b,
                              test="mannwhitneyu"):
        """Differential feature analysis between two groups.

        Args:
            group_col: Column name for grouping (e.g., 'marker_profile').
            group_a: Value for group A (e.g., 'NeuN+').
            group_b: Value for group B (e.g., 'NeuN-').
            test: Statistical test ('mannwhitneyu' or 'ttest').

        Returns:
            DataFrame with feature, statistic, p_value, effect_size, p_adjusted.
        """
        from scipy import stats

        if self._features_df is None:
            raise ValueError("No features loaded.")

        df = self._features_df.copy()
        if group_col not in df.columns and self._detections:
            df[group_col] = [d.get(group_col, "") for d in self._detections]

        mask_a = df[group_col] == group_a
        mask_b = df[group_col] == group_b
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        results = []
        for col in numeric_cols:
            vals_a = df.loc[mask_a, col].dropna()
            vals_b = df.loc[mask_b, col].dropna()
            if len(vals_a) < 3 or len(vals_b) < 3:
                continue
            if test == "mannwhitneyu":
                stat, pval = stats.mannwhitneyu(
                    vals_a, vals_b, alternative="two-sided"
                )
            else:
                stat, pval = stats.ttest_ind(vals_a, vals_b)
            pooled_std = vals_a.std() + vals_b.std() + 1e-10
            effect = (vals_a.mean() - vals_b.mean()) / pooled_std * 2
            results.append({
                "feature": col, "statistic": stat, "p_value": pval,
                "effect_size": effect,
                "mean_a": vals_a.mean(), "mean_b": vals_b.mean(),
            })

        result_df = pd.DataFrame(results).sort_values("p_value")
        if len(result_df) > 0:
            try:
                from statsmodels.stats.multitest import multipletests
                _, result_df["p_adjusted"], _, _ = multipletests(
                    result_df["p_value"], method="fdr_bh"
                )
            except ImportError:
                result_df["p_adjusted"] = result_df["p_value"]
        return result_df

    def correlate(self, morph_features=None, proteins=None, method="spearman"):
        """Correlate morphological features with protein abundances.

        Returns:
            DataFrame: rows=morph_features, columns=proteins, values=correlation.
        """
        if self._linked is None:
            raise ValueError("Call link() first.")

        prot_cols = set(self._proteomics.columns) if self._proteomics is not None else set()
        if morph_features is None:
            morph_features = [c for c in self._linked.columns if c not in prot_cols]
        if proteins is None:
            proteins = [c for c in self._linked.columns if c in prot_cols]

        corr_matrix = pd.DataFrame(
            index=morph_features, columns=proteins, dtype=float
        )
        for mf in morph_features:
            for prot in proteins:
                vals = self._linked[[mf, prot]].dropna()
                if len(vals) < 5:
                    corr_matrix.loc[mf, prot] = np.nan
                    continue
                if method == "spearman":
                    from scipy.stats import spearmanr
                    r, _ = spearmanr(vals[mf], vals[prot])
                else:
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(vals[mf], vals[prot])
                corr_matrix.loc[mf, prot] = r
        return corr_matrix

    def rank_proteins(self, morph_feature, top_n=50):
        """Rank proteins by correlation with a morphological feature."""
        if self._linked is None:
            raise ValueError("Call link() first.")
        prot_cols = set(self._proteomics.columns) if self._proteomics is not None else set()
        proteins = [c for c in self._linked.columns if c in prot_cols]
        correlations = []
        for prot in proteins:
            vals = self._linked[[morph_feature, prot]].dropna()
            if len(vals) < 5:
                continue
            from scipy.stats import spearmanr
            r, p = spearmanr(vals[morph_feature], vals[prot])
            correlations.append({"protein": prot, "correlation": r, "p_value": p})
        return pd.DataFrame(correlations).sort_values(
            "correlation", ascending=False
        ).head(top_n)
