"""Regression test for Phase 1.5 fix to OmicLinker.correlate FDR handling.

Old bug: synthetic p=1.0 values for prefiltered weak correlations (|r| < 0.2)
were passed to multipletests along with real p-values. BH adjusted_p =
p * m / rank; inflating m with synthetic 1.0s made the correction more
conservative than a principled "test only strong pairs" plan.

Fix: keep synthetic 1.0 visible in the returned pval_df, but feed only
|r| >= 0.2 entries to multipletests.
"""

import numpy as np
import pandas as pd
import pytest

from xldvp_seg.analysis.omic_linker import OmicLinker


def _build_linker_with_mock_data(n_wells: int = 60, n_morph: int = 4, n_prot: int = 8):
    rng = np.random.default_rng(0)
    wells = [f"W{i:03d}" for i in range(n_wells)]

    # Morphology: strong-correlated pairs, weak pairs, and noise.
    morph = pd.DataFrame(
        rng.standard_normal((n_wells, n_morph)),
        index=wells,
        columns=[f"morph{i}" for i in range(n_morph)],
    )

    # Proteins: first 2 highly correlated with morph0, rest are weakly correlated.
    prot_data = {}
    prot_data["P_strong_0"] = morph["morph0"] * 2.0 + rng.standard_normal(n_wells) * 0.2
    prot_data["P_strong_1"] = morph["morph0"] * 1.5 + rng.standard_normal(n_wells) * 0.3
    for i in range(n_prot - 2):
        prot_data[f"P_weak_{i}"] = rng.standard_normal(n_wells) * 1.0
    prot = pd.DataFrame(prot_data, index=wells)

    linker = OmicLinker.__new__(OmicLinker)
    linker._linked = morph.join(prot)
    linker._morph_cols = list(morph.columns)
    linker._proteomics_cols = list(prot.columns)
    linker._proteomics = prot
    return linker


class TestFDRPrefilterFix:
    def test_prefiltered_weak_cells_stay_at_one(self):
        """Cells with |r| < 0.2 should show p=1.0 in returned pval_df."""
        linker = _build_linker_with_mock_data()
        _, pval = linker.correlate(method="pearson", return_pvalues=True, fdr_correct=True)
        corr, _ = linker.correlate(method="pearson", return_pvalues=True, fdr_correct=False)

        weak_mask = corr.abs() < 0.2
        assert weak_mask.any().any(), "fixture should include weak correlations"
        # Weak cells should carry 1.0 sentinel (not NaN, not adjusted downward).
        weak_vals = pval.where(weak_mask).values
        weak_vals = weak_vals[~np.isnan(weak_vals)]
        assert np.allclose(
            weak_vals, 1.0
        ), f"Weak-correlation cells should retain 1.0 sentinel, got min={weak_vals.min()}"

    def test_strong_adjustment_less_conservative_than_buggy(self):
        """Strong-r adjusted p-values should be SMALLER (less conservative)
        than what the buggy all-cells-feed version would produce."""
        from statsmodels.stats.multitest import multipletests

        linker = _build_linker_with_mock_data()
        corr, pval_fixed = linker.correlate(method="pearson", return_pvalues=True, fdr_correct=True)

        # Reproduce what the buggy version did: feed ALL p-values (including
        # synthetic 1.0s) to BH.
        corr_no_fdr, pval_no_fdr = linker.correlate(
            method="pearson", return_pvalues=True, fdr_correct=False
        )
        pvals_flat = pval_no_fdr.values.flatten().astype(float)
        valid = ~np.isnan(pvals_flat)
        _, buggy_adj, _, _ = multipletests(pvals_flat[valid], method="fdr_bh")
        buggy_flat = pvals_flat.copy()
        buggy_flat[valid] = buggy_adj
        pval_buggy = pd.DataFrame(
            buggy_flat.reshape(pval_no_fdr.shape),
            index=pval_no_fdr.index,
            columns=pval_no_fdr.columns,
        )

        strong_mask = corr.abs() >= 0.2
        fixed_strong = pval_fixed.where(strong_mask).values
        fixed_strong = fixed_strong[~np.isnan(fixed_strong)]
        buggy_strong = pval_buggy.where(strong_mask).values
        buggy_strong = buggy_strong[~np.isnan(buggy_strong)]

        # Fixed version must be AT LEAST as optimistic as buggy (smaller m).
        assert (fixed_strong <= buggy_strong + 1e-12).all(), (
            "After the fix, strong-r adjusted p-values should be "
            "≤ those from the buggy version (smaller m in BH)"
        )
        # And at least one should be STRICTLY smaller (otherwise no real fix).
        assert (
            fixed_strong < buggy_strong - 1e-9
        ).any(), "Fix should produce at least one strictly-smaller adjusted p"

    def test_spearman_fast_path_all_cells_adjusted(self):
        """Phase A.2 regression: Spearman fast path has real p-values for
        weak correlations; those must still feed BH (flag-based distinction
        from the slow path). A previous fix excluded them incorrectly."""
        linker = _build_linker_with_mock_data()
        corr, pval = linker.correlate(method="spearman", return_pvalues=True, fdr_correct=True)
        # No cells should be stuck at synthetic 1.0 in the fast path —
        # the prefilter never ran.
        weak_mask = corr.abs() < 0.2
        if weak_mask.any().any():
            weak_vals = pval.where(weak_mask).values
            weak_vals = weak_vals[~np.isnan(weak_vals)]
            # Spearman fast path: weak cells carry real (possibly adjusted)
            # p-values, not synthetic 1.0 — at least some should be < 1.0.
            if len(weak_vals) > 0:
                assert (weak_vals < 1.0).any(), (
                    "Spearman fast path should keep real p-values (not 1.0) "
                    "for weak correlations"
                )

    def test_fdr_disabled_raw_pvalues_preserved(self):
        linker = _build_linker_with_mock_data()
        corr, pval = linker.correlate(method="pearson", return_pvalues=True, fdr_correct=False)
        # Weak cells show synthetic 1.0; strong cells show real (<1.0) p-values.
        weak_mask = corr.abs() < 0.2
        strong_mask = corr.abs() >= 0.2
        assert np.allclose(
            pval.where(weak_mask).values[~np.isnan(pval.where(weak_mask).values)], 1.0
        )
        strong_vals = pval.where(strong_mask).values
        strong_vals = strong_vals[~np.isnan(strong_vals)]
        if len(strong_vals) > 0:
            # At least some strong correlations should have p < 1.
            assert strong_vals.min() < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
