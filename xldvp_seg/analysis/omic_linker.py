"""Multi-omic linker: bridge morphological features to proteomics.

For DVP (Deep Visual Proteomics): links cell morphology from the pipeline
to mass-spec proteomic profiles from LMD-cut cells.

Usage:
    from xldvp_seg.analysis.omic_linker import OmicLinker
    from xldvp_seg.core import SlideAnalysis

    slide = SlideAnalysis.load("output/...")
    linker = OmicLinker.from_slide(slide)
    linker.load_proteomics("proteomics.csv")
    linker.load_well_mapping("lmd_export/")
    linked = linker.link()
    diff = linker.differential_features("marker_profile", "NeuN+", "NeuN-")
    corr = linker.correlate(method="spearman")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.utils.logging import get_logger

if TYPE_CHECKING:
    import anndata as ad

logger = get_logger(__name__)


class OmicLinker:
    """Links morphological features to mass-spec proteomics data."""

    def __init__(
        self,
        features_df: pd.DataFrame | None = None,
        detections: list[dict] | None = None,
    ) -> None:
        self.__features_df = features_df
        self._detections = detections
        self._proteomics = None
        self._proteomics_adata = None
        self._well_mapping = None
        self._linked = None

    @property
    def _features_df(self):
        """Lazily build features DataFrame from detections if needed."""
        if self.__features_df is None and self._detections is not None:
            rows = []
            for det in self._detections:
                row = dict(det.get("features", {}))
                row["uid"] = det.get("uid") or det.get("id", "")
                rows.append(row)
            self.__features_df = pd.DataFrame(rows).set_index("uid")
        return self.__features_df

    @_features_df.setter
    def _features_df(self, value):
        self.__features_df = value

    @property
    def proteomics_adata(self):
        """Raw AnnData from dvp-io, available after ``load_proteomics_report()``."""
        return self._proteomics_adata

    @classmethod
    def from_slide(cls, slide):
        """Create from a SlideAnalysis object."""
        return cls(features_df=slide.features_df, detections=slide.detections)

    @classmethod
    def from_detections(cls, detections):
        """Create from a detections list."""
        return cls(detections=detections)

    def load_proteomics(self, path: str | Path, well_column: str = "well_id") -> None:
        """Load proteomics CSV. Rows=wells, columns=proteins."""
        self._proteomics = pd.read_csv(path, index_col=well_column)
        logger.info("Loaded proteomics: %d wells, %d proteins", *self._proteomics.shape)

    def load_proteomics_report(
        self,
        path: str | Path,
        search_engine: str,
        well_column: str = "well_id",
        **kwargs,
    ) -> ad.AnnData:
        """Load proteomics from a search engine report via dvp-io.

        Supports: alphadia, alphapept, diann, directlfq, fragpipe,
        maxquant, mztab, spectronaut. See :meth:`available_engines` for
        the full list.

        Args:
            path: Path to search engine output file.
            search_engine: Engine name (e.g., ``'diann'``, ``'maxquant'``).
            well_column: Column in ``adata.obs`` that identifies wells/samples.
                dvp-io uses ``'sample_id'`` by default.  If your LMD well IDs
                differ, set this to the column that maps to your well plate.
            **kwargs: Forwarded to ``dvpio.read.omics.read_pg_table()``.

        Returns:
            AnnData from dvp-io (also stored internally as DataFrame for linking).
        """
        from dvpio.read.omics.report_reader import read_pg_table

        adata = read_pg_table(str(path), search_engine, **kwargs)
        # adata.to_df() handles both sparse and dense X matrices
        df = adata.to_df()
        if well_column in adata.obs.columns:
            df.index = adata.obs[well_column].values
        else:
            logger.warning(
                "Column '%s' not found in adata.obs (available: %s). " "Using obs_names as index.",
                well_column,
                list(adata.obs.columns),
            )
        df.index.name = well_column
        self._proteomics = df
        # Keep raw AnnData for scverse downstream (squidpy, scanpy)
        self._proteomics_adata = adata
        logger.info(
            "Loaded proteomics report (%s): %d samples, %d proteins",
            search_engine,
            *df.shape,
        )
        return adata

    @staticmethod
    def available_engines() -> list[str]:
        """List supported proteomics search engines (from dvp-io)."""
        from dvpio.read.omics.report_reader import available_reader

        return available_reader("pg_reader")

    def load_well_mapping(self, lmd_dir):
        """Load detection -> well mapping from LMD export."""
        lmd_dir = Path(lmd_dir)
        mapping_files = list(lmd_dir.glob("*well_mapping*.json")) + list(
            lmd_dir.glob("*well_assignment*.json")
        )
        if not mapping_files:
            if self._detections:
                self._well_mapping = {
                    d.get("uid", d.get("id", "")): d.get("well")
                    for d in self._detections
                    if d.get("well")
                }
                logger.info(
                    "Built well mapping from detections: %d entries", len(self._well_mapping)
                )
                return
            raise FileNotFoundError(f"No well mapping files found in {lmd_dir}")
        from xldvp_seg.utils.json_utils import fast_json_load

        mapping = fast_json_load(mapping_files[0])
        if isinstance(mapping, list):
            self._well_mapping = {
                m["uid"]: m["well"] for m in mapping if "uid" in m and "well" in m
            }
        elif isinstance(mapping, dict):
            self._well_mapping = mapping
        logger.info(
            "Loaded well mapping: %d entries from %s",
            len(self._well_mapping),
            mapping_files[0].name,
        )

    def link(self) -> pd.DataFrame:
        """Join morphological features to proteomics by well.

        Aggregates per-cell features to well level (mean of numeric features),
        adds pool spatial metadata (centroid, spread, cell count), then joins
        with well-level proteomics.

        Pool spatial columns added:
            - ``pool_x_um``, ``pool_y_um``: centroid of pooled cells (mean position)
            - ``pool_spread_um``: spatial spread (std of distances from centroid)
            - ``pool_n_cells``: number of cells pooled into the well

        Returns:
            DataFrame with rows=wells, columns=[morph features + spatial + proteins].
        """
        if self._features_df is None:
            raise ConfigError("No features loaded. Use from_slide() or from_detections().")
        if self._proteomics is None:
            raise ConfigError("No proteomics loaded. Call load_proteomics() first.")
        if self._well_mapping is None:
            raise ConfigError("No well mapping. Call load_well_mapping() first.")

        df = self._features_df.copy()
        df["well"] = df.index.map(self._well_mapping)
        df = df.dropna(subset=["well"])
        if len(df) == 0:
            logger.warning(
                "No detections matched well mapping keys. "
                "Check UID format consistency between detections and well mapping."
            )
            return pd.DataFrame()

        # Aggregate numeric features per well
        # Embeddings (sam2_, resnet_, dinov2_) use mean — preserves centroid
        # in representation space. All other features use median — robust to outliers.
        morph_cols = [c for c in df.columns if c != "well" and not c.endswith("_class")]
        numeric_morph = df[morph_cols].select_dtypes(include=[np.number]).columns
        embedding_prefixes = ("sam2_", "resnet_", "dinov2_", "resnet_ctx_", "dinov2_ctx_")
        embedding_cols = [
            c for c in numeric_morph if any(c.startswith(p) for p in embedding_prefixes)
        ]
        scalar_cols = [c for c in numeric_morph if c not in embedding_cols]

        grouped = df.groupby("well")
        well_morph = (
            grouped[scalar_cols].median()
            if scalar_cols
            else pd.DataFrame(index=grouped.size().index)
        )
        if embedding_cols:
            well_morph = well_morph.join(grouped[embedding_cols].mean())

        # Within-well variability (std per feature)
        if scalar_cols:
            well_std = grouped[scalar_cols].std()
            well_std.columns = [f"pool_std_{c}" for c in well_std.columns]
            well_morph = well_morph.join(well_std)

        # Pool cell count + total area
        well_morph["pool_n_cells"] = df.groupby("well").size()
        if "area_um2" in df.columns:
            well_morph["pool_total_area_um2"] = grouped["area_um2"].sum()

        # Pool spatial position (centroid + spread)
        if self._detections:
            uid_to_pos = {}
            for det in self._detections:
                uid = det.get("uid") or det.get("id", "")
                pos = det.get("global_center_um")
                if pos is None:
                    gc = det.get("global_center")
                    px = det.get("pixel_size_um") or det.get("features", {}).get("pixel_size_um")
                    if gc and px:
                        pos = [gc[0] * px, gc[1] * px]
                if pos and len(pos) == 2:
                    uid_to_pos[uid] = pos

            if uid_to_pos:
                # Build per-detection position series aligned to df
                pos_x = df.index.map(lambda u: uid_to_pos.get(u, [np.nan, np.nan])[0])
                pos_y = df.index.map(lambda u: uid_to_pos.get(u, [np.nan, np.nan])[1])
                df["_pos_x"] = pos_x.astype(float)
                df["_pos_y"] = pos_y.astype(float)

                # Centroid per well
                well_morph["pool_x_um"] = df.groupby("well")["_pos_x"].mean()
                well_morph["pool_y_um"] = df.groupby("well")["_pos_y"].mean()

                # Spatial spread per well (std of distances from centroid)
                spreads = {}
                for well, grp in df.groupby("well"):
                    xs = grp["_pos_x"].dropna().values
                    ys = grp["_pos_y"].dropna().values
                    if len(xs) >= 2:
                        cx, cy = xs.mean(), ys.mean()
                        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
                        spreads[well] = float(dists.std())
                    else:
                        spreads[well] = 0.0
                well_morph["pool_spread_um"] = pd.Series(spreads)

                logger.info(
                    "Pool spatial: %d/%d wells have positions",
                    well_morph["pool_x_um"].notna().sum(),
                    len(well_morph),
                )

        self._linked = well_morph.join(self._proteomics, how="inner")
        logger.info("Linked: %d wells with both morph and proteomics", len(self._linked))
        return self._linked

    def differential_features(self, group_col, group_a, group_b, test="mannwhitneyu"):
        """Differential feature analysis between two groups.

        Uses Cohen's d (difference of means / pooled std) as the primary
        effect size metric. This is valid for both positive-only features
        (e.g., intensities, areas) and signed features (e.g., PCA components,
        z-scored values) where log2 fold-change would be meaningless.

        Args:
            group_col: Column name for grouping (e.g., 'marker_profile').
            group_a: Value for group A (e.g., 'NeuN+').
            group_b: Value for group B (e.g., 'NeuN-').
            test: Statistical test ('mannwhitneyu' or 'ttest').

        Returns:
            DataFrame with feature, statistic, p_value, effect_size (Cohen's d),
            mean_diff, mean_a, mean_b, p_adjusted.
        """
        from scipy import stats

        if self._features_df is None:
            raise ConfigError("No features loaded.")

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
                stat, pval = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            else:
                stat, pval = stats.ttest_ind(vals_a, vals_b)
            n_a, n_b = len(vals_a), len(vals_b)
            mean_a = vals_a.mean()
            mean_b = vals_b.mean()
            mean_diff = mean_a - mean_b
            pooled_var = ((n_a - 1) * vals_a.var(ddof=1) + (n_b - 1) * vals_b.var(ddof=1)) / max(
                n_a + n_b - 2, 1
            )
            if pooled_var < 1e-20:
                cohens_d = 0.0
            else:
                cohens_d = mean_diff / np.sqrt(pooled_var)
            results.append(
                {
                    "feature": col,
                    "statistic": stat,
                    "p_value": pval,
                    "effect_size": cohens_d,
                    "mean_diff": mean_diff,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                }
            )

        result_df = pd.DataFrame(results)
        if len(result_df) > 0:
            result_df = result_df.sort_values("p_value")
        if len(result_df) > 0:
            from statsmodels.stats.multitest import multipletests

            _, result_df["p_adjusted"], _, _ = multipletests(result_df["p_value"], method="fdr_bh")
        return result_df

    def correlate(
        self,
        morph_features: list[str] | None = None,
        proteins: list[str] | None = None,
        method: str = "spearman",
        return_pvalues: bool = False,
        fdr_correct: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Correlate morphological features with protein abundances.

        Args:
            morph_features: List of morphological feature column names (default: all non-protein
                columns).
            proteins: List of protein column names (default: all protein columns).
            method: Correlation method ('spearman' or 'pearson').
            return_pvalues: If True, return (corr_df, pval_df) tuple.
            fdr_correct: If True (default) and return_pvalues=True, apply
                Benjamini-Hochberg FDR correction to the p-value matrix.
                With many features x proteins (e.g., 78 x 5000 = 390K tests),
                uncorrected p-values are misleading.

        Returns:
            DataFrame (rows=morph_features, columns=proteins, values=correlation).
            If return_pvalues=True, returns (corr_df, pval_df) tuple where
            pval_df contains FDR-adjusted p-values when fdr_correct=True.
        """
        if self._linked is None:
            raise ConfigError("Call link() first.")

        prot_cols = set(self._proteomics.columns) if self._proteomics is not None else set()
        if morph_features is None:
            morph_features = [c for c in self._linked.columns if c not in prot_cols]
        if proteins is None:
            proteins = [c for c in self._linked.columns if c in prot_cols]

        all_cols = morph_features + proteins
        corr_full = self._linked[all_cols].corr(method=method, min_periods=5)
        corr_df = corr_full.loc[morph_features, proteins]

        if not return_pvalues:
            return corr_df

        # Compute p-values pairwise (matching corr()'s pairwise-complete behavior)
        from scipy.stats import pearsonr, spearmanr

        corr_func = spearmanr if method == "spearman" else pearsonr
        pval_df = pd.DataFrame(index=morph_features, columns=proteins, dtype=float)
        for mf in morph_features:
            for prot in proteins:
                vals = self._linked[[mf, prot]].dropna()
                if len(vals) < 5:
                    pval_df.loc[mf, prot] = np.nan
                else:
                    _, p = corr_func(vals[mf], vals[prot])
                    pval_df.loc[mf, prot] = p

        if fdr_correct:
            from statsmodels.stats.multitest import multipletests

            pvals_flat = pval_df.values.flatten().astype(float)
            valid = ~np.isnan(pvals_flat)
            if valid.sum() > 0:
                _, adjusted, _, _ = multipletests(pvals_flat[valid], method="fdr_bh")
                pvals_flat[valid] = adjusted
                pval_df = pd.DataFrame(
                    pvals_flat.reshape(pval_df.shape),
                    index=pval_df.index,
                    columns=pval_df.columns,
                )

        return corr_df, pval_df

    def rank_proteins(
        self, morph_feature: str, top_n: int = 50, sort_by: str = "correlation"
    ) -> pd.DataFrame:
        """Rank proteins by correlation with a morphological feature.

        Args:
            morph_feature: Name of the morphological feature column to correlate against.
            top_n: Number of top-ranked proteins to return.
            sort_by: Sort criterion — ``"correlation"`` (descending, default) or
                ``"p_adjusted"`` (ascending, most significant first).
        """
        if self._linked is None:
            raise ConfigError("Call link() first.")
        if morph_feature not in self._linked.columns:
            raise ConfigError(
                f"Feature '{morph_feature}' not found in linked data. "
                f"Available: {sorted(self._linked.columns[:20].tolist())}"
            )
        prot_cols = set(self._proteomics.columns) if self._proteomics is not None else set()
        proteins = [c for c in self._linked.columns if c in prot_cols]
        from scipy.stats import spearmanr

        correlations = []
        for prot in proteins:
            vals = self._linked[[morph_feature, prot]].dropna()
            if len(vals) < 5:
                continue
            r, p = spearmanr(vals[morph_feature], vals[prot])
            correlations.append({"protein": prot, "correlation": r, "p_value": p})
        result_df = pd.DataFrame(correlations)
        if len(result_df) == 0:
            return result_df
        from statsmodels.stats.multitest import multipletests

        _, result_df["p_adjusted"], _, _ = multipletests(result_df["p_value"], method="fdr_bh")
        if sort_by == "p_adjusted":
            result_df = result_df.sort_values("p_adjusted", ascending=True)
        else:
            result_df = result_df.sort_values("correlation", ascending=False)
        return result_df.head(top_n)
