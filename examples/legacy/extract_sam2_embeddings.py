#!/usr/bin/env python3
"""Extract SAM2 embeddings for MK detections missing them.

Two modes:
  1. FAST (no GPU): If tile features.json already has sam2_0..sam2_255,
     just reads them. Use when the pipeline saved SAM2 but the export dropped them.
  2. FULL (GPU): If tile features.json lacks SAM2, loads CZI tiles and runs
     SAM2 encoder to extract embeddings at each cell centroid.

Outputs a JSON mapping uid -> [sam2_0, ..., sam2_255] that can be merged
into the training data or full detection JSON.

Usage:
    # Fast mode (read from existing tile features)
    python examples/legacy/extract_sam2_embeddings.py \
        --base-dir /path/to/100pct_run \
        --output sam2_embeddings.json

    # Full mode (extract from CZI with SAM2 model)
    python examples/legacy/extract_sam2_embeddings.py \
        --base-dir /path/to/100pct_run \
        --czi-dir /path/to/czi/files \
        --output sam2_embeddings.json

    # Merge into existing full JSON
    python examples/legacy/extract_sam2_embeddings.py \
        --merge-into all_mks_with_rejected3_full.json \
        --embeddings sam2_embeddings.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SAM2_DIM = 256


def extract_from_tile_features(base_dir, slides=None):
    """Read SAM2 embeddings from existing tile features.json files."""
    base_dir = Path(base_dir)
    embeddings = {}
    n_missing = 0
    n_found = 0

    # Find all slide directories
    slide_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if slides:
        slide_dirs = [d for d in slide_dirs if d.name in slides]

    for slide_dir in slide_dirs:
        tile_base = slide_dir / "mk" / "tiles"
        if not tile_base.exists():
            continue

        tile_dirs = sorted([d for d in tile_base.iterdir() if d.is_dir()])
        slide_found = 0

        for tile_dir in tile_dirs:
            feat_file = tile_dir / "features.json"
            if not feat_file.exists():
                continue

            with open(feat_file) as f:
                tile_feats = json.load(f)

            for det in tile_feats:
                uid = det.get("uid", "")
                feats = det.get("features", {})

                # Check if SAM2 features exist
                if "sam2_0" in feats:
                    emb = [float(feats.get(f"sam2_{i}", 0.0)) for i in range(SAM2_DIM)]
                    if any(v != 0 for v in emb):
                        embeddings[uid] = emb
                        slide_found += 1
                    else:
                        n_missing += 1
                else:
                    n_missing += 1

        n_found += slide_found
        print(f"  {slide_dir.name}: {slide_found} embeddings read")

    print(f"\nTotal: {n_found} embeddings found, {n_missing} missing")
    return embeddings


def extract_from_czi(base_dir, czi_dir, slides=None):
    """Extract SAM2 embeddings by running SAM2 on CZI tiles."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    from segmentation.utils.device import get_default_device

    base_dir = Path(base_dir)
    czi_dir = Path(czi_dir)
    device = get_default_device()
    print(f"Using device: {device}")

    # Find SAM2 checkpoint
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint_candidates = [
        repo_root / "checkpoints" / "sam2.1_hiera_large.pt",
        repo_root / "checkpoints" / "sam2.1_hiera_l.pt",
        Path("/path/to/checkpoints/sam2.1_hiera_large.pt"),
    ]
    checkpoint = None
    for cp in checkpoint_candidates:
        if cp.exists():
            checkpoint = cp
            break
    if checkpoint is None:
        print("ERROR: SAM2 checkpoint not found. Searched:")
        for cp in checkpoint_candidates:
            print(f"  {cp}")
        sys.exit(1)

    print(f"Loading SAM2 from {checkpoint}...")
    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(checkpoint),
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    # Find all slide directories
    slide_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if slides:
        slide_dirs = [d for d in slide_dirs if d.name in slides]

    embeddings = {}

    for slide_dir in slide_dirs:
        slide_name = slide_dir.name
        tile_base = slide_dir / "mk" / "tiles"
        if not tile_base.exists():
            continue

        # Find CZI file
        czi_file = czi_dir / f"{slide_name}.czi"
        if not czi_file.exists():
            # Try without date prefix
            czi_candidates = list(czi_dir.glob(f"*{slide_name}*.czi"))
            if czi_candidates:
                czi_file = czi_candidates[0]
            else:
                print(f"  WARNING: No CZI found for {slide_name}, skipping")
                continue

        # Load CZI for tile extraction
        from aicspylibczi import CziFile

        czi = CziFile(str(czi_file))
        print(f"\n=== {slide_name} (CZI: {czi_file.name}) ===")

        tile_dirs = sorted([d for d in tile_base.iterdir() if d.is_dir()])
        slide_count = 0

        for tile_dir in tile_dirs:
            feat_file = tile_dir / "features.json"
            window_file = tile_dir / "window.csv"
            if not feat_file.exists() or not window_file.exists():
                continue

            with open(feat_file) as f:
                tile_feats = json.load(f)

            if not tile_feats:
                continue

            # Parse window.csv: "(slice(y0, y1, None), slice(x0, x1, None))"
            with open(window_file) as f:
                window_str = f.read().strip()
            slices = re.findall(r"slice\((\d+),\s*(\d+)", window_str)
            if len(slices) != 2:
                print(f"    WARNING: Cannot parse window for {tile_dir.name}")
                continue
            y0, y1 = int(slices[0][0]), int(slices[0][1])
            x0, x1 = int(slices[1][0]), int(slices[1][1])

            # Load tile from CZI (channel 0, grayscale)
            try:
                tile_data = czi.read_mosaic(
                    region=(x0, y0, x1 - x0, y1 - y0),
                    C=0,
                    scale_factor=1.0,
                )
                # Remove singleton dims
                tile_img = np.squeeze(tile_data)
                if tile_img.ndim == 2:
                    # Convert grayscale to RGB for SAM2
                    tile_img = np.stack([tile_img] * 3, axis=-1)
            except Exception as e:
                print(f"    WARNING: Failed to read tile {tile_dir.name}: {e}")
                continue

            # Run SAM2 encoder
            predictor.set_image(tile_img)

            # Extract embedding for each detection
            shape = predictor._features["image_embed"].shape
            emb_h, emb_w = shape[2], shape[3]
            img_h, img_w = predictor._orig_hw[0]

            for det in tile_feats:
                uid = det.get("uid", "")
                center = det.get("center", [0, 0])
                # Convert global coords to tile-local
                local_x = float(center[0]) - x0
                local_y = float(center[1]) - y0

                # Map to embedding space
                if img_h > 0 and img_w > 0:
                    emb_y = int(local_y * emb_h / img_h)
                    emb_x = int(local_x * emb_w / img_w)
                    emb_y = min(max(emb_y, 0), emb_h - 1)
                    emb_x = min(max(emb_x, 0), emb_w - 1)

                    emb = predictor._features["image_embed"][0, :, emb_y, emb_x].cpu().numpy()
                    embeddings[uid] = [float(v) for v in emb]
                    slide_count += 1

            # Free GPU memory
            predictor.reset_predictor()

        print(f"  {slide_name}: {slide_count} embeddings extracted")

    print(f"\nTotal: {len(embeddings)} embeddings extracted")
    return embeddings


def merge_embeddings(target_json, embeddings_json, output_json=None):
    """Merge SAM2 embeddings into a full detection JSON."""
    print(f"Loading embeddings from {embeddings_json}...")
    with open(embeddings_json) as f:
        embeddings = json.load(f)
    print(f"  {len(embeddings)} embeddings")

    print(f"Loading detections from {target_json}...")
    with open(target_json) as f:
        detections = json.load(f)
    print(f"  {len(detections)} detections")

    n_updated = 0
    n_already = 0
    n_missing = 0

    for det in detections:
        uid = det.get("uid", "")
        feats = det.get("features", {})

        if uid in embeddings:
            emb = embeddings[uid]
            for i, v in enumerate(emb):
                feats[f"sam2_{i}"] = v
            det["features"] = feats
            n_updated += 1
        elif "sam2_0" in feats and feats["sam2_0"] != 0:
            n_already += 1
        else:
            n_missing += 1

    print(f"\n  Updated: {n_updated}")
    print(f"  Already had SAM2: {n_already}")
    print(f"  Still missing: {n_missing}")

    if output_json is None:
        output_json = target_json
    with open(output_json, "w") as f:
        json.dump(detections, f)
    print(f"  Wrote {len(detections)} detections to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")

    # Extract command
    ext = sub.add_parser("extract", help="Extract SAM2 embeddings from tile features or CZI")
    ext.add_argument(
        "--base-dir", type=Path, required=True, help="Base directory with slide/mk/tiles/ structure"
    )
    ext.add_argument(
        "--czi-dir",
        type=Path,
        default=None,
        help="Directory containing CZI files (enables GPU extraction)",
    )
    ext.add_argument(
        "--slides", nargs="*", default=None, help="Specific slide names to process (default: all)"
    )
    ext.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON with uid -> [sam2_0..sam2_255] mapping",
    )

    # Merge command
    mrg = sub.add_parser("merge", help="Merge SAM2 embeddings into detection JSON")
    mrg.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Detection JSON to update (e.g. all_mks_with_rejected3_full.json)",
    )
    mrg.add_argument(
        "--embeddings", type=Path, required=True, help="SAM2 embeddings JSON from extract command"
    )
    mrg.add_argument(
        "--output", type=Path, default=None, help="Output path (default: overwrite target)"
    )

    args = parser.parse_args()

    if args.command == "extract":
        if args.czi_dir:
            print("Full extraction mode (GPU required)")
            embeddings = extract_from_czi(args.base_dir, args.czi_dir, slides=args.slides)
        else:
            print("Fast extraction mode (reading existing tile features)")
            embeddings = extract_from_tile_features(args.base_dir, slides=args.slides)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(embeddings, f)
        print(f"\nSaved {len(embeddings)} embeddings to {args.output}")

    elif args.command == "merge":
        merge_embeddings(
            args.target, args.embeddings, output_json=str(args.output) if args.output else None
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
