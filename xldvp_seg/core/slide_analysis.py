"""Central state object wrapping pipeline output.

Usage:
    from segmentation.core import SlideAnalysis

    slide = SlideAnalysis.load("/path/to/output/slide_name/run_timestamp/")
    print(f"{slide.n_detections} detections, {slide.cell_type}")

    # Lazy-loaded properties
    df = slide.features_df  # pandas DataFrame of all features
    pos = slide.positions_um  # Nx2 array

    # Filtering
    good = slide.filter(score_threshold=0.5)
    neurons = slide.filter(marker="NeuN", positive=True)

    # Export
    slide.save("filtered_detections.json")
    adata = slide.to_anndata()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from segmentation.utils.detection_utils import extract_positions_um

if TYPE_CHECKING:
    import anndata
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


class SlideAnalysis:
    """Rich wrapper around a pipeline output directory.

    Provides lazy-loaded access to detections, features, positions, and
    contours.  Filtering returns a new instance (immutable pattern).
    Uses existing pipeline utilities for all I/O and coordinate extraction.
    """

    def __init__(
        self, output_dir: str | Path | None = None, detections: list[dict] | None = None
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir else None
        self._detections_override = detections  # for filtered instances
        self._detections = None
        self._summary = None
        self._config = None
        self._features_df = None
        self._positions_um = None
        self._contours = None
        self._pixel_size_um_cached = None
        # Discovered file paths
        self._detections_path = None
        self._summary_path = None
        self._config_path = None
        self._tiles_dir = None
        self._html_dir = None
        self._spatialdata_path = None

    @classmethod
    def load(cls, output_dir: str | Path) -> SlideAnalysis:
        """Factory: auto-discovers files in the output directory."""
        sa = cls(output_dir)
        sa._discover_files()
        return sa

    @classmethod
    def from_detections(
        cls, detections: list[dict], output_dir: str | Path | None = None
    ) -> SlideAnalysis:
        """Create from a pre-loaded detections list."""
        sa = cls(output_dir, detections=detections)
        if output_dir:
            sa._discover_files()
        return sa

    def _discover_files(self):
        """Glob for pipeline output files."""
        d = self._output_dir
        if d is None or not d.exists():
            return

        # Detections JSON (primary output)
        # Exclude intermediate files like *_merged.json, *_postdedup.json
        det_files = sorted(d.glob("*_detections.json"))
        # Prefer the canonical name (no extra suffix before _detections)
        # e.g. cell_detections.json over cell_detections_merged.json
        canonical = [f for f in det_files if "_merged" not in f.stem and "_postdedup" not in f.stem]
        if canonical:
            self._detections_path = canonical[0]
        elif det_files:
            self._detections_path = det_files[0]

        # Summary
        summary_path = d / "summary.json"
        if summary_path.exists():
            self._summary_path = summary_path

        # Config
        config_path = d / "pipeline_config.json"
        if config_path.exists():
            self._config_path = config_path

        # Tiles directory
        tiles_dir = d / "tiles"
        if tiles_dir.is_dir():
            self._tiles_dir = tiles_dir

        # HTML directory
        html_dir = d / "html"
        if html_dir.is_dir():
            self._html_dir = html_dir

        # SpatialData zarr
        zarr_files = sorted(d.glob("*_spatialdata.zarr"))
        if zarr_files:
            self._spatialdata_path = zarr_files[0]

    # --- Lazy-loaded properties ---

    @property
    def detections(self) -> list[dict]:
        """List of detection dicts (loaded lazily from disk)."""
        if self._detections is None:
            if self._detections_override is not None:
                self._detections = self._detections_override
            elif self._detections_path and self._detections_path.exists():
                logger.info("Loading detections from %s", self._detections_path)
                self._detections = fast_json_load(self._detections_path)
            else:
                self._detections = []
        return self._detections

    @property
    def summary(self) -> dict:
        """Pipeline summary dict (from summary.json)."""
        if self._summary is None:
            if self._summary_path and self._summary_path.exists():
                self._summary = fast_json_load(self._summary_path)
            else:
                self._summary = {}
        return self._summary

    @property
    def config(self) -> dict:
        """Pipeline config dict (from pipeline_config.json)."""
        if self._config is None:
            if self._config_path and self._config_path.exists():
                self._config = fast_json_load(self._config_path)
            else:
                self._config = {}
        return self._config

    @property
    def features_df(self) -> pd.DataFrame:
        """DataFrame of all detection features, indexed by uid."""
        if self._features_df is None:
            dets = self.detections
            if not dets:
                self._features_df = pd.DataFrame()
            else:
                rows = []
                for det in dets:
                    row = dict(det.get("features", {}))
                    row["uid"] = det.get("uid") or det.get("id", "")
                    row["rf_prediction"] = det.get("rf_prediction")
                    # Add marker classes
                    for key in det:
                        if key.endswith("_class") and key != "classifier_info":
                            row[key] = det[key]
                    if "marker_profile" in det:
                        row["marker_profile"] = det["marker_profile"]
                    rows.append(row)
                self._features_df = pd.DataFrame(rows)
                if "uid" in self._features_df.columns:
                    self._features_df = self._features_df.set_index("uid")
        return self._features_df

    @property
    def positions_um(self) -> np.ndarray:
        """Nx2 array of [x, y] positions in microns."""
        if self._positions_um is None:
            positions, inferred_px = extract_positions_um(
                self.detections, self.pixel_size_um or None
            )
            self._positions_um = positions
            # Cache inferred pixel size if we didn't have one
            if inferred_px and not self._pixel_size_um_cached:
                self._pixel_size_um_cached = inferred_px
        return self._positions_um

    @property
    def contours(self) -> list[np.ndarray | None]:
        """List of contour arrays (contour_px or contour_um per detection)."""
        from segmentation.utils.detection_utils import get_contour_px, get_contour_um

        if self._contours is None:
            self._contours = []
            for det in self.detections:
                c = get_contour_px(det)
                if c is None:
                    c = get_contour_um(det)
                self._contours.append(np.array(c) if c is not None else None)
        return self._contours

    # --- Metadata properties ---

    @property
    def cell_type(self) -> str:
        """Cell type string (from summary or config)."""
        return self.summary.get("cell_type", self.config.get("cell_type", "unknown"))

    @property
    def pixel_size_um(self) -> float:
        """Pixel size in microns (from summary, config, or inferred)."""
        if self._pixel_size_um_cached:
            return self._pixel_size_um_cached
        val = self.summary.get("pixel_size_um", self.config.get("pixel_size_um", 0.0))
        if val:
            self._pixel_size_um_cached = val
        return val

    @property
    def n_detections(self) -> int:
        """Number of detections."""
        return len(self.detections)

    @property
    def slide_name(self) -> str:
        """Slide name (from summary or output directory name)."""
        return self.summary.get(
            "slide_name", self._output_dir.name if self._output_dir else "unknown"
        )

    # --- File path properties ---

    @property
    def detections_path(self) -> Path | None:
        """Path to the detections JSON file (if discovered)."""
        return self._detections_path

    @property
    def tiles_dir(self) -> Path | None:
        """Path to the tiles directory (if it exists)."""
        return self._tiles_dir

    @property
    def html_dir(self) -> Path | None:
        """Path to the HTML viewer directory (if it exists)."""
        return self._html_dir

    @property
    def spatialdata_path(self) -> Path | None:
        """Path to the SpatialData zarr store (if it exists)."""
        return self._spatialdata_path

    @property
    def output_dir(self) -> Path | None:
        """Root output directory."""
        return self._output_dir

    # --- Filtering ---

    def filter(
        self,
        *,
        score_threshold: float | None = None,
        marker: str | None = None,
        positive: bool = True,
    ) -> SlideAnalysis:
        """Return new SlideAnalysis with filtered detections.

        Args:
            score_threshold: Keep detections with rf_prediction >= this value.
            marker: Marker name to filter on (e.g. "NeuN").
            positive: If True, keep marker-positive detections; if False, negative.

        Returns:
            New SlideAnalysis instance with filtered detections.
        """
        filtered = list(self.detections)

        if score_threshold is not None:
            filtered = [d for d in filtered if d.get("rf_prediction", 0) >= score_threshold]

        if marker is not None:
            marker_key = f"{marker}_class"
            target = "positive" if positive else "negative"
            # Check both top-level and features dict (classify_markers stores in features)
            filtered = [
                d
                for d in filtered
                if d.get(marker_key) == target or d.get("features", {}).get(marker_key) == target
            ]

        logger.info("Filtered: %d -> %d detections", len(self.detections), len(filtered))
        return SlideAnalysis.from_detections(filtered, self._output_dir)

    # --- I/O ---

    def save(self, path: str | Path) -> None:
        """Write detections JSON via atomic_json_dump."""
        atomic_json_dump(self.detections, Path(path))
        logger.info("Saved %d detections to %s", len(self.detections), path)

    def to_anndata(self) -> anndata.AnnData:
        """Export as AnnData for scanpy/scverse workflows.

        Layout:
            **X** — morphological + per-channel intensity features (float32).
            **obs** — per-cell metadata: uid, slide_name, cell_type,
                rf_prediction, marker classes, marker_profile, n_nuclei,
                nuclear_area_fraction, area_um2.
            **var** — per-feature metadata: feature_group (morph / channel /
                ratio / nuclear).
            **obsm** — embeddings and coordinates:
                ``spatial`` (N, 2) um positions,
                ``X_sam2`` (256D), ``X_resnet`` (2048D),
                ``X_dinov2`` (1024D), plus ``_ctx`` variants.
            **uns** — pipeline provenance: slide_name, cell_type,
                pixel_size_um, n_detections, pipeline_version, channel info.
        """
        import anndata

        from segmentation import __version__

        df = self.features_df
        if df.empty:
            return anndata.AnnData()

        # --- Classify columns ---
        obs_meta_cols = [
            c
            for c in df.columns
            if c.endswith("_class") or c in ("rf_prediction", "marker_profile")
        ]
        sam2_cols = sorted([c for c in df.columns if c.startswith("sam2_")])
        embedding_prefixes = ("resnet_", "dinov2_", "resnet_ctx_", "dinov2_ctx_")
        embedding_cols: set[str] = set()
        for prefix in embedding_prefixes:
            embedding_cols |= {c for c in df.columns if c.startswith(prefix)}
        # Metadata columns that belong in obs, not X
        obs_feature_cols = {"area_um2", "n_nuclei", "nuclear_area_fraction"}
        exclude = set(obs_meta_cols) | set(sam2_cols) | embedding_cols | obs_feature_cols
        x_cols = [c for c in df.columns if c not in exclude]

        # --- Build X (morph + channel features) ---
        X = df[x_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Build obs (per-cell metadata) ---
        obs = df[obs_meta_cols].copy()
        # Enrich obs with top-level detection metadata
        dets = self.detections
        uids = [d.get("uid") or d.get("id", "") for d in dets]
        # Ensure unique obs_names (AnnData requirement)
        seen: dict[str, int] = {}
        unique_uids = []
        for u in uids:
            if u in seen:
                seen[u] += 1
                unique_uids.append(f"{u}_{seen[u]}")
            else:
                seen[u] = 0
                unique_uids.append(u)
        obs.index = pd.Index(unique_uids, name="uid")
        obs["slide_name"] = [d.get("slide_name", self.slide_name) for d in dets]
        obs["cell_type"] = [d.get("cell_type", self.cell_type) for d in dets]
        obs["pixel_size_um"] = self.pixel_size_um or np.nan
        obs["area_um2"] = [d.get("features", {}).get("area_um2", np.nan) for d in dets]
        # Nuclear counting fields (if available)
        if any("n_nuclei" in d.get("features", {}) for d in dets[:10]):
            obs["n_nuclei"] = [d.get("features", {}).get("n_nuclei", np.nan) for d in dets]
            obs["nuclear_area_fraction"] = [
                d.get("features", {}).get("nuclear_area_fraction", np.nan) for d in dets
            ]

        # --- Build var (per-feature metadata) ---
        feature_groups = []
        for col in x_cols:
            if col.startswith("ch") and "_" in col[2:5]:
                # ch0_mean, ch1_snr, etc.
                if "_ratio" in col or "_diff" in col or "_specificity" in col:
                    feature_groups.append("ratio")
                else:
                    feature_groups.append("channel")
            elif col.startswith("n_nuclei") or col.startswith("nuclear_"):
                feature_groups.append("nuclear")
            else:
                feature_groups.append("morph")
        var = pd.DataFrame(
            {"feature_group": feature_groups}, index=pd.Index(x_cols, name="feature")
        )

        # --- Assemble AnnData ---
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # --- Embeddings in obsm ---
        if sam2_cols:
            sam2_vals = df[sam2_cols].values.astype(np.float32)
            adata.obsm["X_sam2"] = np.nan_to_num(sam2_vals, nan=0.0, posinf=0.0, neginf=0.0)

        resnet_ctx = sorted([c for c in df.columns if c.startswith("resnet_ctx_")])
        resnet = sorted(
            [c for c in df.columns if c.startswith("resnet_") and not c.startswith("resnet_ctx_")]
        )
        dinov2_ctx = sorted([c for c in df.columns if c.startswith("dinov2_ctx_")])
        dinov2 = sorted(
            [c for c in df.columns if c.startswith("dinov2_") and not c.startswith("dinov2_ctx_")]
        )
        for cols, key in [
            (resnet, "X_resnet"),
            (resnet_ctx, "X_resnet_ctx"),
            (dinov2, "X_dinov2"),
            (dinov2_ctx, "X_dinov2_ctx"),
        ]:
            if cols:
                vals = df[cols].values.astype(np.float32)
                adata.obsm[key] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Spatial coordinates ---
        pos = self.positions_um
        if pos is not None and len(pos) == adata.n_obs:
            adata.obsm["spatial"] = pos
        elif pos is not None and len(pos) != adata.n_obs:
            logger.warning(
                "Spatial coordinates length (%d) != n_obs (%d); obsm['spatial'] not set.",
                len(pos),
                adata.n_obs,
            )

        # --- Pipeline provenance in uns ---
        adata.uns["pipeline"] = {
            "package": "xldvp_seg",
            "version": __version__,
            "slide_name": self.slide_name,
            "cell_type": self.cell_type,
            "pixel_size_um": self.pixel_size_um,
            "n_detections": len(dets),
        }
        # Channel info from config if available
        config = self.config
        if config.get("channel_map"):
            adata.uns["channel_map"] = config["channel_map"]
        if config.get("channels"):
            adata.uns["channels"] = config["channels"]

        return adata

    # --- Repr ---

    def __repr__(self) -> str:
        # Use cached data to avoid triggering lazy loading from repr
        n = len(self._detections) if self._detections is not None else "?"
        summary = self._summary or {}
        config = self._config or {}
        name = summary.get("slide_name", self._output_dir.name if self._output_dir else "unknown")
        ct = summary.get("cell_type", config.get("cell_type", "unknown"))
        px = self._pixel_size_um_cached or summary.get(
            "pixel_size_um", config.get("pixel_size_um", 0.0)
        )
        px_str = f"{px:.4f}" if px else "?"
        return f"SlideAnalysis(slide='{name}', cell_type='{ct}', " f"n={n}, px={px_str} um)"

    def __len__(self) -> int:
        return self.n_detections
