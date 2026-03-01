#!/usr/bin/env python3
"""Convert pipeline detections to SpatialData format for scverse ecosystem analysis.

Standalone converter that processes any completed pipeline run into a SpatialData
zarr store with AnnData tables, polygon shape layers, optional image references,
and optional squidpy spatial analyses.

Works with all cell types (NMJ, MK, vessel, cell, islet, mesothelium, tissue_pattern).

Usage:
    # Basic conversion
    python scripts/convert_to_spatialdata.py \\
        --detections /path/to/cell_detections.json \\
        --output /path/to/output.zarr

    # With contour extraction from HDF5 masks
    python scripts/convert_to_spatialdata.py \\
        --detections /path/to/cell_detections.json \\
        --output /path/to/output.zarr \\
        --tiles-dir /path/to/tiles/

    # With OME-Zarr image and squidpy analysis
    python scripts/convert_to_spatialdata.py \\
        --detections /path/to/cell_detections.json \\
        --output /path/to/output.zarr \\
        --zarr-image /path/to/slide.ome.zarr \\
        --run-squidpy --squidpy-cluster-key tdTomato_class

    # Vessel pipeline (contours from JSON, not HDF5)
    python scripts/convert_to_spatialdata.py \\
        --detections /path/to/vessel_detections.json \\
        --output /path/to/output.zarr \\
        --cell-type vessel
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np

# Ensure repo root is on PYTHONPATH
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.json_utils import fast_json_load
from segmentation.utils.detection_utils import load_detections

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Morphological features that go into X matrix
MORPH_FEATURES = [
    'area', 'perimeter', 'circularity', 'eccentricity', 'solidity',
    'extent', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length',
    'aspect_ratio', 'compactness', 'convexity', 'roughness',
    'mean_intensity', 'std_intensity', 'max_intensity', 'min_intensity',
    'intensity_range', 'median_intensity', 'p25_intensity', 'p75_intensity',
    'p90_intensity', 'p95_intensity', 'p99_intensity', 'iqr_intensity',
    'skewness', 'kurtosis', 'entropy', 'energy', 'contrast',
    'homogeneity', 'correlation', 'dissimilarity',
]

# Vessel-specific morphological features
VESSEL_FEATURES = [
    'outer_diameter_um', 'inner_diameter_um', 'wall_thickness_mean_um',
    'wall_thickness_std_um', 'wall_thickness_cv', 'lumen_area_um2',
    'wall_area_um2', 'sma_ring_area_um2', 'has_sma_ring', 'confidence',
]

# Embedding prefixes -> obsm key
EMBEDDING_PREFIXES = {
    'sam2_': ('X_sam2', 256),
    'resnet_masked_': ('X_resnet_masked', 2048),
    'resnet_context_': ('X_resnet_context', 2048),
    'dinov2_masked_': ('X_dinov2_masked', 1024),
    'dinov2_context_': ('X_dinov2_context', 1024),
}

# Metadata fields that go into obs (not X)
OBS_FIELDS = [
    'uid', 'slide_name', 'cell_type', 'tile_origin', 'global_center',
    'global_center_um', 'rf_prediction', 'community_id', 'cluster_id',
    'mask_label', 'tile_mask_label', 'global_id',
]

# Vessel contour keys -> shape layer names
VESSEL_CONTOUR_KEYS = {
    'outer_contour_global': 'vessel_outer',
    'inner_contour_global': 'vessel_lumen',
    'sma_contour_global': 'vessel_sma',
}


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def _discover_features(detections):
    """Scan detections to discover all available feature names and embeddings.

    Returns:
        (morph_names, channel_stat_names, embedding_map)
        where embedding_map = {obsm_key: (prefix, dim, actual_count)}
    """
    morph_names = set()
    channel_stat_names = set()
    embedding_counts = {}  # prefix -> max index seen

    for det in detections:
        feats = det.get('features', {})
        for key in feats:
            val = feats[key]
            # Skip non-numeric values
            if isinstance(val, (list, tuple, dict)):
                continue
            if val is None:
                continue

            # Check if it's an embedding dimension
            matched_embed = False
            for prefix, (obsm_key, expected_dim) in EMBEDDING_PREFIXES.items():
                if key.startswith(prefix):
                    try:
                        idx = int(key[len(prefix):])
                        embedding_counts.setdefault(prefix, 0)
                        embedding_counts[prefix] = max(embedding_counts[prefix], idx + 1)
                        matched_embed = True
                    except ValueError:
                        pass
                    break
            if matched_embed:
                continue

            # Check if it's a channel stat (e.g., ch0_mean, ch2_p95)
            if key[:3] == 'ch' and key[3:4].isdigit() and '_' in key:
                channel_stat_names.add(key)
            elif key in MORPH_FEATURES or key in VESSEL_FEATURES:
                morph_names.add(key)
            else:
                # Include other scalar features as morph
                try:
                    float(val)
                    morph_names.add(key)
                except (TypeError, ValueError):
                    pass

    # Build embedding map
    embedding_map = {}
    for prefix, (obsm_key, expected_dim) in EMBEDDING_PREFIXES.items():
        if prefix in embedding_counts:
            actual = embedding_counts[prefix]
            embedding_map[obsm_key] = (prefix, expected_dim, actual)

    # Sort for reproducibility
    morph_names = sorted(morph_names)
    channel_stat_names = sorted(channel_stat_names)

    return morph_names, channel_stat_names, embedding_map


def _discover_obs_classes(detections):
    """Discover classification columns (e.g., tdTomato_class, GFP_class) from detections."""
    class_cols = set()
    for det in detections:
        feats = det.get('features', {})
        for key, val in feats.items():
            if key.endswith('_class') and isinstance(val, str):
                class_cols.add(key)
        # Also check top-level keys
        for key, val in det.items():
            if key.endswith('_class') and isinstance(val, str):
                class_cols.add(key)
    return sorted(class_cols)


def build_anndata(detections, cell_type):
    """Build AnnData object from detections list.

    Args:
        detections: List of detection dicts with 'features' sub-dicts.
        cell_type: Cell type string for obs metadata.

    Returns:
        anndata.AnnData with X, obs, obsm, var populated.
    """
    import anndata as ad
    import pandas as pd

    n = len(detections)
    if n == 0:
        raise ValueError("No detections to convert")

    logger.info("Discovering features from %s detections...", f"{n:,}")
    morph_names, channel_stat_names, embedding_map = _discover_features(detections)
    class_cols = _discover_obs_classes(detections)

    # X matrix: morph + channel stats
    x_names = morph_names + channel_stat_names
    logger.info("  Morphological features: %d", len(morph_names))
    logger.info("  Channel stat features: %d", len(channel_stat_names))
    logger.info("  Embedding layers: %s", ', '.join(f"{k} ({v[2]}d)" for k, v in embedding_map.items()) or 'none')
    logger.info("  Classification columns: %s", ', '.join(class_cols) or 'none')

    # Pre-allocate arrays
    X = np.zeros((n, len(x_names)), dtype=np.float32)
    embeddings = {key: np.zeros((n, info[2]), dtype=np.float32) for key, info in embedding_map.items()}

    # Obs columns
    obs_data = {field: [] for field in OBS_FIELDS}
    for col in class_cols:
        obs_data[col] = []

    spatial_px = np.zeros((n, 2), dtype=np.float64)
    spatial_um = np.zeros((n, 2), dtype=np.float64)

    # Fill arrays
    for i, det in enumerate(detections):
        feats = det.get('features', {})

        # X matrix
        for j, name in enumerate(x_names):
            val = feats.get(name, 0)
            if isinstance(val, (list, tuple)):
                val = 0
            X[i, j] = float(val) if val is not None else 0.0

        # Embeddings
        for obsm_key, (prefix, expected_dim, actual) in embedding_map.items():
            for d in range(actual):
                val = feats.get(f"{prefix}{d}", 0.0)
                embeddings[obsm_key][i, d] = float(val) if val is not None else 0.0

        # Spatial coordinates
        gc = det.get('global_center', [0, 0])
        spatial_px[i] = [gc[0] if gc[0] is not None else 0, gc[1] if gc[1] is not None else 0]
        gc_um = det.get('global_center_um', [0, 0])
        spatial_um[i] = [gc_um[0] if gc_um[0] is not None else 0, gc_um[1] if gc_um[1] is not None else 0]

        # Obs metadata
        for field in OBS_FIELDS:
            val = det.get(field, feats.get(field))
            if isinstance(val, (list, tuple)):
                val = str(val)
            obs_data[field].append(val)

        # Classification columns
        for col in class_cols:
            val = det.get(col, feats.get(col))
            obs_data[col].append(val if val is not None else 'unknown')

    # Clean NaN/Inf in X
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    for key in embeddings:
        embeddings[key] = np.nan_to_num(embeddings[key], nan=0.0, posinf=0.0, neginf=0.0)

    # Build obs DataFrame
    obs_df = pd.DataFrame(obs_data)
    if 'uid' in obs_df.columns:
        obs_df.index = obs_df['uid'].astype(str)
        obs_df.index.name = None
        if obs_df.index.duplicated().any():
            n_dup = obs_df.index.duplicated().sum()
            logger.warning("Making %d duplicate UIDs unique", n_dup)
            obs_df.index = ad.utils.make_index_unique(obs_df.index)
    # Ensure cell_type column
    if 'cell_type' not in obs_df.columns or obs_df['cell_type'].isna().all():
        obs_df['cell_type'] = cell_type

    # Convert classification columns to categorical
    for col in class_cols:
        obs_df[col] = pd.Categorical(obs_df[col])

    # Build var DataFrame
    var_df = pd.DataFrame(index=x_names)
    var_df.index.name = 'feature'
    var_df['feature_type'] = ['morphological'] * len(morph_names) + ['channel_stat'] * len(channel_stat_names)

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

    # Add spatial coordinates to obsm
    adata.obsm['spatial'] = spatial_um  # um coords for squidpy
    adata.obsm['spatial_pixel'] = spatial_px

    # Add embeddings to obsm
    for key, arr in embeddings.items():
        adata.obsm[key] = arr

    logger.info("Built AnnData: %d obs x %d var, %d obsm layers",
                adata.n_obs, adata.n_vars, len(adata.obsm))

    return adata


# ---------------------------------------------------------------------------
# Shape extraction
# ---------------------------------------------------------------------------

def _extract_shapes_from_hdf5(detections, tiles_dir, cell_type):
    """Extract polygon contours from HDF5 mask files, tile by tile.

    Returns:
        List of shapely Polygons (or None for missing contours), parallel to detections.
    """
    import hdf5plugin  # noqa: F401 - must import before h5py for LZ4
    import h5py
    import cv2
    from shapely.geometry import Polygon

    tiles_dir = Path(tiles_dir)

    # Determine mask filename
    mask_filenames = {
        'nmj': 'nmj_masks.h5',
        'mk': 'mk_masks.h5',
        'cell': 'cell_masks.h5',
        'islet': 'islet_masks.h5',
        'mesothelium': 'mesothelium_masks.h5',
        'tissue_pattern': 'tissue_pattern_masks.h5',
    }
    mask_filename = mask_filenames.get(cell_type, f'{cell_type}_masks.h5')

    # Group detections by tile
    by_tile = {}
    for idx, det in enumerate(detections):
        tile_origin = det.get('tile_origin', [0, 0])
        tile_key = f"tile_{int(tile_origin[0])}_{int(tile_origin[1])}"
        by_tile.setdefault(tile_key, []).append((idx, det))

    polygons = [None] * len(detections)
    tiles_found = 0
    contours_extracted = 0

    for tile_idx, (tile_key, tile_dets) in enumerate(by_tile.items()):
        if (tile_idx + 1) % 50 == 0:
            logger.info("  Shape extraction: tile %d/%d...", tile_idx + 1, len(by_tile))

        mask_path = tiles_dir / tile_key / mask_filename
        if not mask_path.exists():
            continue
        tiles_found += 1

        with h5py.File(mask_path, 'r') as hf:
            masks = hf['masks'][:]

        for det_idx, det in tile_dets:
            label = det.get('mask_label')
            if label is None:
                det_id = det.get('id', '')
                try:
                    label = int(det_id.split('_')[-1])
                except (ValueError, IndexError):
                    continue

            binary = (masks == label).astype(np.uint8)
            if binary.sum() == 0:
                continue

            contours_cv, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours_cv:
                continue

            largest = max(contours_cv, key=cv2.contourArea)
            pts = largest.reshape(-1, 2).astype(float)

            # Convert to global coordinates
            tile_origin = det.get('tile_origin', [0, 0])
            pts[:, 0] += tile_origin[0]
            pts[:, 1] += tile_origin[1]

            if len(pts) >= 3:
                try:
                    poly = Polygon(pts)
                    if poly.is_valid:
                        polygons[det_idx] = poly
                        contours_extracted += 1
                except Exception:  # noqa: E722
                    pass

        del masks  # Free memory before next tile

    logger.info("  Extracted %d polygons from %d tiles (%d tiles with masks)",
                contours_extracted, len(by_tile), tiles_found)
    return polygons


def _extract_vessel_shapes(detections):
    """Extract vessel multi-contour shape layers from JSON contour fields.

    Returns:
        Dict of {layer_name: list_of_polygons_or_None} parallel to detections.
    """
    from shapely.geometry import Polygon

    layers = {name: [None] * len(detections) for name in VESSEL_CONTOUR_KEYS.values()}
    counts = {name: 0 for name in VESSEL_CONTOUR_KEYS.values()}

    for i, det in enumerate(detections):
        for json_key, layer_name in VESSEL_CONTOUR_KEYS.items():
            contour = det.get(json_key)
            if contour is None:
                continue
            pts = np.array(contour, dtype=float)
            if len(pts) < 3:
                continue
            try:
                poly = Polygon(pts)
                if poly.is_valid:
                    layers[layer_name][i] = poly
                    counts[layer_name] += 1
            except Exception:
                pass

    for name, count in counts.items():
        logger.info("  %s: %d polygons", name, count)

    return layers


def _make_circle_fallback(detections, pixel_size_um):
    """Create circle polygons as fallback when no masks/contours available.

    Uses detection area to compute radius.

    Returns:
        List of shapely Polygons, parallel to detections.
    """
    from shapely.geometry import Point

    polygons = [None] * len(detections)
    for i, det in enumerate(detections):
        gc = det.get('global_center')
        if gc is None or gc[0] is None or gc[1] is None:
            continue
        feats = det.get('features', {})
        area_px = feats.get('area', 100)
        radius_px = max(np.sqrt(area_px / np.pi), 3)
        try:
            circle = Point(gc[0], gc[1]).buffer(radius_px, resolution=16)
            polygons[i] = circle
        except Exception:
            pass

    n_valid = sum(1 for p in polygons if p is not None)
    logger.info("  Circle fallback: %d/%d polygons", n_valid, len(detections))
    return polygons


def build_shapes(detections, cell_type, tiles_dir=None, pixel_size_um=1.0):
    """Build GeoDataFrame shape layers from detections.

    Args:
        detections: List of detection dicts.
        cell_type: Cell type string.
        tiles_dir: Path to tiles directory with HDF5 masks (optional).
        pixel_size_um: Pixel size in microns for circle fallback.

    Returns:
        Dict of {layer_name: geopandas.GeoDataFrame} with polygon geometries.
    """
    import geopandas as gpd

    if cell_type == 'vessel':
        logger.info("Extracting vessel multi-contour shapes from JSON...")
        layers_dict = _extract_vessel_shapes(detections)

        result = {}
        for layer_name, polys in layers_dict.items():
            # Filter to non-None
            valid = [(i, p) for i, p in enumerate(polys) if p is not None]
            if not valid:
                continue
            indices, geoms = zip(*valid)
            uids = [detections[i].get('uid', f'det_{i}') for i in indices]
            gdf = gpd.GeoDataFrame(
                {'uid': uids, 'det_index': list(indices)},
                geometry=list(geoms),
            )
            gdf.index = gdf['uid']
            result[layer_name] = gdf

        return result

    # Non-vessel: single shape layer
    logger.info("Extracting shapes for %s...", cell_type)

    polygons = None
    if tiles_dir:
        polygons = _extract_shapes_from_hdf5(detections, tiles_dir, cell_type)
        n_valid = sum(1 for p in polygons if p is not None)
        if n_valid < len(detections) * 0.5:
            logger.warning("Only %d/%d detections had mask contours, using circle fallback for remainder",
                          n_valid, len(detections))
            # Fill gaps with circles
            circles = _make_circle_fallback(detections, pixel_size_um)
            for i in range(len(polygons)):
                if polygons[i] is None:
                    polygons[i] = circles[i]
    else:
        logger.info("No tiles-dir provided, using circle fallback for all shapes")
        polygons = _make_circle_fallback(detections, pixel_size_um)

    # Build single GeoDataFrame
    valid = [(i, p) for i, p in enumerate(polygons) if p is not None]
    if not valid:
        logger.warning("No valid shapes extracted")
        return {}

    indices, geoms = zip(*valid)
    uids = [detections[i].get('uid', f'det_{i}') for i in indices]
    layer_name = f'{cell_type}_cells'
    gdf = gpd.GeoDataFrame(
        {'uid': uids, 'det_index': list(indices)},
        geometry=list(geoms),
    )
    gdf.index = gdf['uid']

    logger.info("Built shape layer '%s': %d polygons", layer_name, len(gdf))
    return {layer_name: gdf}


# ---------------------------------------------------------------------------
# Image linking
# ---------------------------------------------------------------------------

def link_zarr_image(zarr_path):
    """Create a lazy dask-backed reference to an OME-Zarr image.

    Returns:
        Dict suitable for SpatialData images parameter, or None on failure.
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        logger.warning("Zarr image not found: %s", zarr_path)
        return None

    try:
        from spatialdata.models import Image2DModel
        import dask.array as da
        import zarr

        store = zarr.open(str(zarr_path), mode='r')

        # Find the highest-resolution array (level 0)
        # OME-Zarr stores pyramids as '0', '1', '2', ... under root
        if '0' in store:
            arr = da.from_zarr(str(zarr_path / '0'))
        elif 'data' in store:
            arr = da.from_zarr(str(zarr_path), component='data')
        else:
            # Try root
            arr = da.from_zarr(str(zarr_path))

        # Ensure (C, Y, X) ordering for SpatialData
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # (Y, X) -> (1, Y, X)
        elif arr.ndim == 3:
            # Could be (C, Y, X) or (Y, X, C)
            if arr.shape[-1] <= 10 and arr.shape[0] > 10:
                # Likely (Y, X, C)
                arr = da.moveaxis(arr, -1, 0)

        image = Image2DModel.parse(arr, dims=("c", "y", "x"))
        logger.info("Linked image: shape=%s, dtype=%s (lazy/dask)", arr.shape, arr.dtype)
        return {'slide': image}

    except Exception as e:
        logger.warning("Failed to link zarr image: %s", e)
        return None


# ---------------------------------------------------------------------------
# Squidpy analyses
# ---------------------------------------------------------------------------

def run_squidpy_analyses(adata, cluster_key=None, output_dir=None):
    """Run spatial analyses using squidpy on the AnnData object.

    Args:
        adata: AnnData with obsm['spatial'] populated.
        cluster_key: obs column to use for neighborhood analyses (e.g., 'tdTomato_class').
        output_dir: Directory to save plots and CSVs.

    Returns:
        Modified adata with results in .obsp and .uns.
    """
    import squidpy as sq

    # Set non-interactive backend before any pyplot import
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spatial neighbors (k-nearest neighbors graph)
    logger.info("Computing spatial neighbors (k=15, kNN)...")
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=15)
    logger.info("  Stored: adata.obsp['spatial_connectivities'], adata.obsp['spatial_distances']")

    # 2. Neighborhood enrichment (requires categorical cluster_key)
    if cluster_key and cluster_key in adata.obs.columns:
        import pandas as pd
        if not isinstance(adata.obs[cluster_key].dtype, pd.CategoricalDtype):
            adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key])

        n_cats = len(adata.obs[cluster_key].cat.categories)
        if n_cats >= 2:
            logger.info("Neighborhood enrichment on '%s' (%d categories)...", cluster_key, n_cats)
            try:
                sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
                logger.info("  Stored: adata.uns['%s_nhood_enrichment']", cluster_key)

                if output_dir:
                    try:
                        sq.pl.nhood_enrichment(adata, cluster_key=cluster_key)
                        plt.savefig(output_dir / 'nhood_enrichment.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        logger.info("  Saved: nhood_enrichment.png")
                    except Exception as e:
                        logger.warning("  Plot failed: %s", e)
            except Exception as e:
                logger.warning("  Neighborhood enrichment failed: %s", e)

            # 3. Co-occurrence
            logger.info("Co-occurrence analysis on '%s'...", cluster_key)
            try:
                sq.gr.co_occurrence(adata, cluster_key=cluster_key)
                logger.info("  Stored: adata.uns['%s_co_occurrence']", cluster_key)

                if output_dir:
                    try:
                        sq.pl.co_occurrence(adata, cluster_key=cluster_key)
                        plt.savefig(output_dir / 'co_occurrence.png', dpi=150, bbox_inches='tight')
                        plt.close()
                        logger.info("  Saved: co_occurrence.png")
                    except Exception as e:
                        logger.warning("  Plot failed: %s", e)
            except Exception as e:
                logger.warning("  Co-occurrence failed: %s", e)

            # 4. Ripley's function
            logger.info("Ripley's L function on '%s'...", cluster_key)
            try:
                sq.gr.ripley(adata, cluster_key=cluster_key, mode='L')
                logger.info("  Stored: adata.uns['%s_ripley']", cluster_key)
            except Exception as e:
                logger.warning("  Ripley failed: %s", e)
        else:
            logger.warning("Skipping cluster analyses: '%s' has only %d category", cluster_key, n_cats)
    else:
        if cluster_key:
            logger.warning("Cluster key '%s' not found in obs columns: %s",
                          cluster_key, list(adata.obs.columns))

    # 5. Spatial autocorrelation (Moran's I) on numeric features
    logger.info("Spatial autocorrelation (Moran's I) on X features...")
    try:
        sq.gr.spatial_autocorr(adata, mode='moran')
        moranI = adata.uns.get('moranI')
        if moranI is not None and output_dir:
            moranI.to_csv(output_dir / 'morans_i.csv')
            logger.info("  Saved: morans_i.csv (%d features)", len(moranI))

            # Log top spatially autocorrelated features
            top = moranI.sort_values('I', ascending=False).head(10)
            logger.info("  Top spatially autocorrelated features:")
            for feat, row in top.iterrows():
                logger.info("    %s: I=%.3f (p=%.2e)", feat, row['I'], row.get('pval_norm', float('nan')))
    except Exception as e:
        logger.warning("  Moran's I failed: %s", e)

    return adata


# ---------------------------------------------------------------------------
# SpatialData assembly
# ---------------------------------------------------------------------------

def assemble_spatialdata(adata, shapes=None, images=None):
    """Assemble SpatialData object from components.

    Args:
        adata: AnnData table.
        shapes: Dict of {name: GeoDataFrame} shape layers.
        images: Dict of {name: DataArray} image layers.

    Returns:
        spatialdata.SpatialData object.
    """
    from spatialdata import SpatialData
    from spatialdata.models import ShapesModel, TableModel

    # Wrap table with SpatialData table model
    # Link table to the primary (first) shape layer only.
    # For vessel, this is vessel_outer; other layers are standalone shapes.
    # SpatialData requires the set of obs['region'] values to exactly match
    # the region parameter, so we use a single primary region.
    region_names = list(shapes.keys()) if shapes else []
    if region_names:
        primary_region = region_names[0]
        adata.obs['region'] = primary_region
        adata.obs['instance_id'] = adata.obs.index
        adata = TableModel.parse(
            adata,
            region=primary_region,
            region_key='region',
            instance_key='instance_id',
        )

    # Parse shapes through ShapesModel
    parsed_shapes = {}
    if shapes:
        for name, gdf in shapes.items():
            try:
                parsed_shapes[name] = ShapesModel.parse(gdf)
            except Exception as e:
                logger.warning("Failed to parse shape layer '%s': %s", name, e)

    sdata = SpatialData(
        images=images or {},
        shapes=parsed_shapes,
        tables={'table': adata},
    )

    return sdata


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(args):
    """Main conversion logic."""
    detections_path = Path(args.detections)
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        logger.error("Output already exists: %s (use --overwrite to replace)", output_path)
        sys.exit(1)

    # 1. Load detections
    logger.info("Loading detections from %s...", detections_path)
    detections = load_detections(str(detections_path), score_threshold=args.score_threshold)
    if not detections:
        logger.error("No detections loaded")
        sys.exit(1)
    logger.info("Loaded %s detections", f"{len(detections):,}")

    # Detect cell type from detections if not specified
    cell_type = args.cell_type
    if not cell_type:
        # Try to infer from first detection or filename
        sample = detections[0] if detections else {}
        cell_type = sample.get('cell_type', '')
        if not cell_type:
            stem = detections_path.stem.lower()
            for ct in ('vessel', 'nmj', 'mk', 'cell', 'islet', 'mesothelium', 'tissue_pattern'):
                if ct in stem:
                    cell_type = ct
                    break
        if not cell_type:
            cell_type = 'cell'
        logger.info("Inferred cell type: %s", cell_type)

    # Get pixel size from first detection
    pixel_size_um = 1.0
    for det in detections[:10]:
        ps = det.get('pixel_size_um') or det.get('features', {}).get('pixel_size_um')
        if ps and ps > 0:
            pixel_size_um = float(ps)
            break
    logger.info("Pixel size: %.4f um/px", pixel_size_um)

    # 2. Build AnnData
    adata = build_anndata(detections, cell_type)

    # 3. Extract shapes
    shapes = {}
    if not args.no_shapes:
        tiles_dir = args.tiles_dir
        shapes = build_shapes(detections, cell_type, tiles_dir=tiles_dir, pixel_size_um=pixel_size_um)
    else:
        logger.info("Shape extraction disabled (--no-shapes)")

    # 4. Link image
    images = {}
    if args.zarr_image:
        logger.info("Linking OME-Zarr image: %s", args.zarr_image)
        img_dict = link_zarr_image(args.zarr_image)
        if img_dict:
            images = img_dict

    # 5. Run squidpy analyses
    if args.run_squidpy:
        squidpy_out = output_path.parent / f"{output_path.stem}_squidpy"
        adata = run_squidpy_analyses(adata, cluster_key=args.squidpy_cluster_key, output_dir=squidpy_out)

    # 6. Assemble and write
    logger.info("Assembling SpatialData...")
    sdata = assemble_spatialdata(adata, shapes=shapes, images=images)
    logger.info("SpatialData: %s", sdata)

    # Write atomically: write to tmp dir, then rename
    import shutil
    tmp_path = output_path.with_suffix('.zarr.tmp')
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    logger.info("Writing to %s...", output_path)
    sdata.write(tmp_path)

    if output_path.exists() and args.overwrite:
        shutil.rmtree(output_path)
    tmp_path.rename(output_path)
    logger.info("Done! SpatialData written to %s", output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SpatialData Export Summary")
    print("=" * 60)
    print(f"  Detections: {len(detections):,}")
    print(f"  Cell type:  {cell_type}")
    print(f"  Features:   {adata.n_vars} in X, {len(adata.obsm)} obsm layers")
    print(f"  Shapes:     {len(shapes)} layers ({', '.join(f'{k}: {len(v)}' for k, v in shapes.items())})")
    print(f"  Images:     {len(images)} layers")
    print(f"  Output:     {output_path}")
    if args.run_squidpy:
        print(f"  Squidpy:    results in {squidpy_out}/")
    print("=" * 60)
    print("\nLoad in Python:")
    print(f"  import spatialdata as sd")
    print(f"  sdata = sd.read_zarr('{output_path}')")
    print(f"  adata = sdata['table']")
    print()


# ---------------------------------------------------------------------------
# Lightweight API for pipeline integration (no argparse)
# ---------------------------------------------------------------------------

def export_spatialdata(detections, output_path, cell_type='cell',
                       tiles_dir=None, zarr_image=None, pixel_size_um=1.0,
                       run_squidpy=False, squidpy_cluster_key=None, overwrite=True):
    """Programmatic API for converting detections to SpatialData.

    Called from segmentation/pipeline/finalize.py for automatic export.

    Args:
        detections: List of detection dicts.
        output_path: Path for output .zarr store.
        cell_type: Cell type string.
        tiles_dir: Optional path to tiles directory with HDF5 masks.
        zarr_image: Optional path to OME-Zarr image.
        pixel_size_um: Pixel size in microns.
        run_squidpy: Whether to run squidpy analyses.
        squidpy_cluster_key: obs column for squidpy cluster analyses.
        overwrite: Whether to overwrite existing output.

    Returns:
        Path to written zarr store, or None on failure.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info("SpatialData already exists: %s (skipping)", output_path)
        return output_path

    if not detections:
        logger.warning("No detections for SpatialData export")
        return None

    try:
        # Build AnnData
        adata = build_anndata(detections, cell_type)

        # Extract shapes
        shapes = build_shapes(detections, cell_type, tiles_dir=tiles_dir, pixel_size_um=pixel_size_um)

        # Link image
        images = {}
        if zarr_image:
            img_dict = link_zarr_image(zarr_image)
            if img_dict:
                images = img_dict

        # Squidpy analyses
        if run_squidpy:
            squidpy_out = output_path.parent / f"{output_path.stem}_squidpy"
            adata = run_squidpy_analyses(adata, cluster_key=squidpy_cluster_key, output_dir=squidpy_out)

        # Assemble and write atomically
        import shutil
        sdata = assemble_spatialdata(adata, shapes=shapes, images=images)

        tmp_path = output_path.with_suffix('.zarr.tmp')
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        sdata.write(tmp_path)

        if output_path.exists():
            shutil.rmtree(output_path)
        tmp_path.rename(output_path)

        logger.info("SpatialData written: %s (%d obs, %d shapes)",
                    output_path, adata.n_obs, len(shapes))
        return output_path

    except Exception as e:
        logger.error("SpatialData export failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert pipeline detections to SpatialData format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--detections', required=True,
                        help='Path to detections JSON file')
    parser.add_argument('--output', required=True,
                        help='Output path for .zarr store')
    parser.add_argument('--cell-type', default=None,
                        help='Cell type (auto-detected from detections if not specified)')
    parser.add_argument('--tiles-dir', default=None,
                        help='Path to tiles directory for HDF5 mask contour extraction')
    parser.add_argument('--zarr-image', default=None,
                        help='Path to OME-Zarr image to link (lazy/dask, no RAM)')
    parser.add_argument('--score-threshold', type=float, default=None,
                        help='Filter detections by rf_prediction >= threshold')
    parser.add_argument('--no-shapes', action='store_true',
                        help='Skip shape extraction (table only)')
    parser.add_argument('--run-squidpy', action='store_true',
                        help='Run squidpy spatial analyses')
    parser.add_argument('--squidpy-cluster-key', default=None,
                        help='obs column for squidpy cluster analyses (e.g., tdTomato_class)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level='INFO')

    # Check dependencies
    missing = []
    for pkg in ('spatialdata', 'anndata', 'geopandas', 'scanpy'):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if args.run_squidpy:
        try:
            __import__('squidpy')
        except ImportError:
            missing.append('squidpy')
    if missing:
        logger.error("Missing dependencies: %s", ', '.join(missing))
        logger.error("Install with: pip install %s", ' '.join(missing))
        sys.exit(1)

    # Suppress noisy warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    convert(args)


if __name__ == '__main__':
    main()
