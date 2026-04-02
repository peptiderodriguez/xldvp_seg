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

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import sys
import warnings
from pathlib import Path

from xldvp_seg.io.spatialdata_export import (
    assemble_spatialdata,
    build_anndata,
    build_shapes,
    link_zarr_image,
    run_squidpy_analyses,
)
from xldvp_seg.utils.detection_utils import load_detections
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


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
        cell_type = sample.get("cell_type", "")
        if not cell_type:
            stem = detections_path.stem.lower()
            for ct in ("vessel", "nmj", "mk", "cell", "islet", "mesothelium", "tissue_pattern"):
                if ct in stem:
                    cell_type = ct
                    break
        if not cell_type:
            cell_type = "cell"
        logger.info("Inferred cell type: %s", cell_type)

    # Get pixel size from first detection
    pixel_size_um = 1.0
    for det in detections[:10]:
        ps = det.get("pixel_size_um") or det.get("features", {}).get("pixel_size_um")
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
        shapes = build_shapes(
            detections, cell_type, tiles_dir=tiles_dir, pixel_size_um=pixel_size_um
        )
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
        adata = run_squidpy_analyses(
            adata, cluster_key=args.squidpy_cluster_key, output_dir=squidpy_out
        )

    # 6. Assemble and write
    logger.info("Assembling SpatialData...")
    sdata = assemble_spatialdata(adata, shapes=shapes, images=images)
    logger.info("SpatialData: %s", sdata)

    # Write atomically: write to tmp dir, then rename
    import shutil

    tmp_path = output_path.with_suffix(".zarr.tmp")
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
    print(
        f"  Shapes:     {len(shapes)} layers ({', '.join(f'{k}: {len(v)}' for k, v in shapes.items())})"
    )
    print(f"  Images:     {len(images)} layers")
    print(f"  Output:     {output_path}")
    if args.run_squidpy:
        print(f"  Squidpy:    results in {squidpy_out}/")
    print("=" * 60)
    print("\nLoad in Python:")
    print("  import spatialdata as sd")
    print(f"  sdata = sd.read_zarr('{output_path}')")
    print("  adata = sdata['table']")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert pipeline detections to SpatialData format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--detections", required=True, help="Path to detections JSON file")
    parser.add_argument("--output", required=True, help="Output path for .zarr store")
    parser.add_argument(
        "--cell-type",
        default=None,
        help="Cell type (auto-detected from detections if not specified)",
    )
    parser.add_argument(
        "--tiles-dir", default=None, help="Path to tiles directory for HDF5 mask contour extraction"
    )
    parser.add_argument(
        "--zarr-image", default=None, help="Path to OME-Zarr image to link (lazy/dask, no RAM)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Filter detections by rf_prediction >= threshold",
    )
    parser.add_argument(
        "--no-shapes", action="store_true", help="Skip shape extraction (table only)"
    )
    parser.add_argument("--run-squidpy", action="store_true", help="Run squidpy spatial analyses")
    parser.add_argument(
        "--squidpy-cluster-key",
        default=None,
        help="obs column for squidpy cluster analyses (e.g., tdTomato_class)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level="INFO")

    # Check dependencies
    missing = []
    for pkg in ("spatialdata", "anndata", "geopandas", "scanpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if args.run_squidpy:
        try:
            __import__("squidpy")
        except ImportError:
            missing.append("squidpy")
    if missing:
        logger.error("Missing dependencies: %s", ", ".join(missing))
        logger.error("Install with: pip install %s", " ".join(missing))
        sys.exit(1)

    # Suppress noisy warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    convert(args)


if __name__ == "__main__":
    main()
