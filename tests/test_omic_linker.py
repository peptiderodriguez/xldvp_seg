"""Tests for xldvp_seg.analysis.omic_linker.OmicLinker.

Covers:
- link() with known pools (median aggregation by default)
- link() with mean method (embedding columns)
- Mismatched pool IDs (no overlap)
- differential_features() with two groups
- load_proteomics_report() with dvp-io integration
- available_engines() helper
"""

from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from xldvp_seg.analysis.omic_linker import OmicLinker


def _make_detections(n_wells=3, n_cells_per_well=10, seed=42):
    """Build synthetic detections with well assignments and features."""
    rng = np.random.default_rng(seed)
    dets = []
    for w in range(n_wells):
        well_id = f"W{w+1}"
        for c in range(n_cells_per_well):
            uid = f"cell_{w}_{c}"
            dets.append(
                {
                    "uid": uid,
                    "well": well_id,
                    "global_center_um": [
                        float(rng.uniform(0, 1000)),
                        float(rng.uniform(0, 1000)),
                    ],
                    "features": {
                        "area": float(rng.uniform(50, 200)),
                        "area_um2": float(rng.uniform(10, 50)),
                        "circularity": float(rng.uniform(0.5, 1.0)),
                        "ch0_mean": float(rng.uniform(10, 100)),
                    },
                }
            )
    return dets


def _make_proteomics(n_wells=3, n_proteins=5, seed=42):
    """Build synthetic proteomics DataFrame."""
    rng = np.random.default_rng(seed)
    data = {}
    data["well_id"] = [f"W{w+1}" for w in range(n_wells)]
    for p in range(n_proteins):
        data[f"PROT{p+1}"] = rng.uniform(0, 100, n_wells).tolist()
    return pd.DataFrame(data)


class TestOmicLinkerLink:
    def test_link_median_aggregation(self, tmp_path):
        """link() with default method should aggregate numeric features by median."""
        dets = _make_detections(n_wells=3, n_cells_per_well=10)
        prot_df = _make_proteomics(n_wells=3)
        prot_csv = tmp_path / "proteomics.csv"
        prot_df.to_csv(prot_csv, index=False)

        linker = OmicLinker.from_detections(dets)
        linker.load_proteomics(str(prot_csv))
        # Build well mapping from detections
        linker._well_mapping = {d["uid"]: d["well"] for d in dets}

        linked = linker.link()
        assert len(linked) == 3
        assert "area" in linked.columns
        assert "PROT1" in linked.columns
        assert "pool_n_cells" in linked.columns
        # Each well has 10 cells
        assert all(linked["pool_n_cells"] == 10)

    def test_link_spatial_metadata(self, tmp_path):
        """link() should add pool_x_um, pool_y_um, pool_spread_um."""
        dets = _make_detections(n_wells=2, n_cells_per_well=5)
        prot_df = _make_proteomics(n_wells=2)
        prot_csv = tmp_path / "proteomics.csv"
        prot_df.to_csv(prot_csv, index=False)

        linker = OmicLinker.from_detections(dets)
        linker.load_proteomics(str(prot_csv))
        linker._well_mapping = {d["uid"]: d["well"] for d in dets}

        linked = linker.link()
        assert "pool_x_um" in linked.columns
        assert "pool_y_um" in linked.columns
        assert "pool_spread_um" in linked.columns
        assert linked["pool_x_um"].notna().all()

    def test_link_no_overlap(self, tmp_path):
        """Mismatched pool IDs should return empty DataFrame."""
        dets = _make_detections(n_wells=2)
        prot_df = _make_proteomics(n_wells=2)
        prot_csv = tmp_path / "proteomics.csv"
        prot_df.to_csv(prot_csv, index=False)

        linker = OmicLinker.from_detections(dets)
        linker.load_proteomics(str(prot_csv))
        # Mismatched well mapping
        linker._well_mapping = {d["uid"]: "NONEXISTENT_WELL" for d in dets}

        linked = linker.link()
        # inner join with proteomics where wells don't match -> empty
        assert len(linked) == 0

    def test_link_raises_without_proteomics(self):
        """link() should raise when proteomics not loaded."""
        dets = _make_detections(n_wells=1)
        linker = OmicLinker.from_detections(dets)
        linker._well_mapping = {}
        with pytest.raises(ValueError, match="No proteomics"):
            linker.link()

    def test_link_raises_without_well_mapping(self, tmp_path):
        """link() should raise when well mapping not loaded."""
        dets = _make_detections(n_wells=1)
        prot_df = _make_proteomics(n_wells=1)
        prot_csv = tmp_path / "proteomics.csv"
        prot_df.to_csv(prot_csv, index=False)

        linker = OmicLinker.from_detections(dets)
        linker.load_proteomics(str(prot_csv))
        with pytest.raises(ValueError, match="No well mapping"):
            linker.link()


class TestOmicLinkerDifferentialFeatures:
    def test_two_groups(self):
        """differential_features() should return p-values for two groups."""
        # Create detections with two groups
        rng = np.random.default_rng(42)
        dets = []
        for i in range(60):
            group = "A" if i < 30 else "B"
            dets.append(
                {
                    "uid": f"cell_{i}",
                    "marker_profile": group,
                    "features": {
                        "area": float(rng.normal(100, 10) if group == "A" else rng.normal(150, 10)),
                        "circularity": float(rng.normal(0.7, 0.05)),
                    },
                }
            )

        linker = OmicLinker.from_detections(dets)
        result = linker.differential_features("marker_profile", "A", "B")
        assert len(result) > 0
        assert "feature" in result.columns
        assert "p_value" in result.columns
        assert "effect_size" in result.columns
        assert "p_adjusted" in result.columns
        # Area should show significant difference
        area_row = result[result["feature"] == "area"]
        assert len(area_row) == 1
        assert area_row.iloc[0]["p_value"] < 0.05

    def test_no_features_raises(self):
        """differential_features() should raise with no features loaded."""
        linker = OmicLinker()
        with pytest.raises(ValueError, match="No features"):
            linker.differential_features("group", "A", "B")

    def test_insufficient_samples_skips(self):
        """Features with < 3 samples per group should be skipped."""
        # Need 3+ samples per group for a test to run; give 2 vs 1
        dets = [
            {"uid": "c1", "group": "A", "features": {"area": 100.0}},
            {"uid": "c2", "group": "A", "features": {"area": 110.0}},
            {"uid": "c3", "group": "B", "features": {"area": 200.0}},
        ]
        linker = OmicLinker.from_detections(dets)
        result = linker.differential_features("group", "A", "B")
        # Too few samples (2 in A, 1 in B -- both < 3), so no features tested
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


def _make_mock_adata(n_samples=3, n_proteins=5, well_ids=None, sparse=False):
    """Build a mock AnnData matching dvp-io read_pg_table output."""
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_proteins)).astype(np.float32)
    if sparse:
        import scipy.sparse

        X = scipy.sparse.csr_matrix(X)
    if well_ids is None:
        well_ids = [f"W{i + 1}" for i in range(n_samples)]
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame({"well_id": well_ids}, index=[f"s{i}" for i in range(n_samples)]),
        var=pd.DataFrame(index=[f"PROT{i + 1}" for i in range(n_proteins)]),
    )


class TestLoadProteomicsReport:
    """Tests for load_proteomics_report() with dvp-io integration."""

    def test_load_report_success(self):
        """Should parse AnnData -> DataFrame and store both."""
        mock_adata = _make_mock_adata(n_samples=3, n_proteins=5)
        with patch("dvpio.read.omics.report_reader.read_pg_table", return_value=mock_adata):
            linker = OmicLinker()
            ret = linker.load_proteomics_report("report.tsv", "diann")
            assert ret is mock_adata
            assert linker._proteomics.shape == (3, 5)
            assert list(linker._proteomics.index) == ["W1", "W2", "W3"]
            assert linker._proteomics_adata is mock_adata

    def test_load_report_sparse_matrix(self):
        """Should handle sparse X matrix correctly."""
        mock_adata = _make_mock_adata(n_samples=3, n_proteins=5, sparse=True)
        with patch("dvpio.read.omics.report_reader.read_pg_table", return_value=mock_adata):
            linker = OmicLinker()
            linker.load_proteomics_report("report.tsv", "maxquant")
            assert linker._proteomics.shape == (3, 5)
            # Verify it was converted to dense
            assert not hasattr(linker._proteomics.values, "toarray")

    def test_load_report_no_well_column(self):
        """When well_column not in obs, should keep obs_names as index."""
        mock_adata = ad.AnnData(
            X=np.random.rand(2, 3).astype(np.float32),
            obs=pd.DataFrame({"other_col": ["a", "b"]}, index=["sample_0", "sample_1"]),
            var=pd.DataFrame(index=["P1", "P2", "P3"]),
        )
        with patch("dvpio.read.omics.report_reader.read_pg_table", return_value=mock_adata):
            linker = OmicLinker()
            linker.load_proteomics_report("report.tsv", "diann")
            # Falls back to obs_names since "well_id" not in obs
            assert list(linker._proteomics.index) == ["sample_0", "sample_1"]

    def test_available_engines(self):
        """Should return list of engine names."""
        engines = OmicLinker.available_engines()
        assert isinstance(engines, list)
        assert len(engines) >= 5
        assert "diann" in engines

    def test_link_with_report(self):
        """End-to-end: load report -> link -> verify joined output."""
        dets = _make_detections(n_wells=2, n_cells_per_well=5)
        mock_adata = _make_mock_adata(n_samples=2, n_proteins=3, well_ids=["W1", "W2"])
        with patch("dvpio.read.omics.report_reader.read_pg_table", return_value=mock_adata):
            linker = OmicLinker.from_detections(dets)
            linker.load_proteomics_report("report.tsv", "diann")
            linker._well_mapping = {d["uid"]: d["well"] for d in dets}
            linked = linker.link()
            assert "PROT1" in linked.columns
            assert "area" in linked.columns
            assert len(linked) == 2
