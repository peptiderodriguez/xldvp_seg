"""
Convert HTML annotation export to training data format for RF classifier.

This script:
1. Loads annotation file (all_labels_combined-2.json)
2. Recreates the cell list in the same order as HTML export
3. Maps annotations by global index to actual cells
4. Loads features for annotated cells
5. Creates training_data.json for RF training

Usage:
    python convert_annotations_to_training.py \
        --annotations /viper/ptmp2/edrod/MKsegmentation/annotations/all_labels_combined-2.json \
        --base-dir /ptmp/edrod/unified_10pct \
        --mk-min-area-um 100 \
        --mk-max-area-um 2100 \
        --output /viper/ptmp2/edrod/MKsegmentation/annotations/training_data.json
"""

import json
import numpy as np
from pathlib import Path
import argparse


def load_cell_samples(base_dir, cell_type, size_filter=None):
    """
    Load all cells into a dictionary keyed by spatial UID from features.json.

    Returns: dict {uid: sample_dict}
    UID format: {slide}_{celltype}_{round(global_x)}_{round(global_y)}
    Falls back to legacy {slide}_{tile_id}_{det_id} if 'uid' field missing.
    """
    base_dir = Path(base_dir)
    pixel_size_um = 0.1725

    inventory = {}

    # Get all slides
    slides = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    for slide_dir in slides:
        slide_name = slide_dir.name
        safe_slide_name = slide_name.replace('.', '-')

        cell_type_dir = slide_dir / cell_type / "tiles"

        if not cell_type_dir.exists():
            continue

        # Get all tile directories
        tile_dirs = sorted([d for d in cell_type_dir.iterdir() if d.is_dir()],
                          key=lambda x: x.name)

        for tile_dir in tile_dirs:
            tile_id = tile_dir.name
            features_file = tile_dir / "features.json"

            if not features_file.exists():
                continue

            # Load features
            with open(features_file, 'r') as f:
                tile_features = json.load(f)

            # Add each detection
            for feat_dict in tile_features:
                det_id = feat_dict['id']
                features = feat_dict['features']
                area_px = features.get('area', 0)
                area_um2 = area_px * (pixel_size_um ** 2)

                # Check size filter
                if size_filter:
                    min_px, max_px = size_filter
                    if not (min_px <= area_px <= max_px):
                        continue

                # Use spatial UID from features.json if present, else legacy format
                uid = feat_dict.get('uid', f"{safe_slide_name}_{tile_id}_{det_id}")

                sample = {
                    'slide': slide_name,
                    'tile_id': tile_id,
                    'det_id': det_id,
                    'features': features,
                    'area_px': area_px,
                    'area_um2': area_um2,
                    'uid': uid
                }
                inventory[uid] = sample

                # Also index by truncated-coordinate UID (int() instead of round())
                # to match annotations from earlier HTML versions
                center = feat_dict.get('center')
                if center:
                    trunc_uid = f"{slide_name}_{cell_type}_{int(center[0])}_{int(center[1])}"
                    if trunc_uid != uid and trunc_uid not in inventory:
                        inventory[trunc_uid] = sample
                
    return inventory


def convert_annotations(annotations_path, base_dir, output_path, mk_min_area_um=None, mk_max_area_um=None):
    """Convert annotations to training data format using robust IDs."""

    print("\n" + "="*70)
    print("CONVERTING ANNOTATIONS TO TRAINING DATA (ROBUST ID MODE)")
    print("="*70)

    # Load annotations
    print(f"\nLoading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)

    # Handle multiple annotation JSON formats:
    # (A) Combined index export: {"mk": {"positive": [...], "negative": [...]}, "hspc": {...}}
    # (B) Per-celltype page export: {"cell_type": "mk", "positive": [...], "negative": [...]}

    if 'cell_type' in all_annotations:
        # Format (B): single cell type export from page-level Export button
        ct = all_annotations['cell_type']
        ann = {'positive': all_annotations.get('positive', []),
               'negative': all_annotations.get('negative', []),
               'unsure': all_annotations.get('unsure', [])}
        mk_ann = ann if ct == 'mk' else {'positive': [], 'negative': []}
        hspc_ann = ann if ct == 'hspc' else {'positive': [], 'negative': []}
        print(f"  Format: per-celltype ({ct})")
    else:
        # Format (A): combined export from index page
        mk_ann = all_annotations.get('mk', {'positive': [], 'negative': []})
        hspc_ann = all_annotations.get('hspc', {'positive': [], 'negative': []})
        print(f"  Format: combined (mk + hspc)")
    
    # Convert µm² to px² if provided
    mk_size_filter = None
    if mk_min_area_um is not None and mk_max_area_um is not None:
        PIXEL_SIZE_UM = 0.1725
        um_to_px_factor = PIXEL_SIZE_UM ** 2
        mk_min_px = int(mk_min_area_um / um_to_px_factor)
        mk_max_px = int(mk_max_area_um / um_to_px_factor)
        mk_size_filter = (mk_min_px, mk_max_px)
        print(f"\nMK area filter: {mk_min_area_um}-{mk_max_area_um} µm² = {mk_min_px}-{mk_max_px} px²")

    # Load Cell Inventories
    print(f"\nLoading MK inventory from: {base_dir}")
    mk_inventory = load_cell_samples(
        base_dir=base_dir,
        cell_type='mk',
        size_filter=mk_size_filter
    )
    print(f"  Loaded {len(mk_inventory)} MK cells (indexed by UID)")

    print(f"\nLoading HSPC inventory from: {base_dir}")
    hspc_inventory = load_cell_samples(
        base_dir=base_dir,
        cell_type='hspc',
        size_filter=None
    )
    print(f"  Loaded {len(hspc_inventory)} HSPC cells (indexed by UID)")

    # Map annotations to samples
    print(f"\nMapping annotations to cells...")

    training_samples = []
    
    def process_category(id_list, label, cell_type, inventory):
        count = 0
        missing = 0
        for uid in id_list:
            # Handle potential legacy int indices (skip them or warn)
            if isinstance(uid, int) or (isinstance(uid, str) and uid.isdigit()):
                print(f"  ⚠ Warning: Found integer ID '{uid}' - incompatible with robust ID mode. Skipping.")
                continue
                
            if uid in inventory:
                sample = inventory[uid].copy()
                sample['label'] = label
                sample['cell_type'] = cell_type
                training_samples.append(sample)
                count += 1
            else:
                missing += 1
                # Only warn if it looks like a valid UID (contains underscores)
                if '_' in str(uid):
                    print(f"  ⚠ Warning: Annotated cell not found in inventory: {uid}")
        return count, missing

    # MK
    pos, miss = process_category(mk_ann.get('positive', []), 1, 'mk', mk_inventory)
    print(f"  MK Positive: Found {pos}, Missing {miss}")
    
    neg, miss = process_category(mk_ann.get('negative', []), 0, 'mk', mk_inventory)
    print(f"  MK Negative: Found {neg}, Missing {miss}")

    # HSPC
    pos, miss = process_category(hspc_ann.get('positive', []), 1, 'hspc', hspc_inventory)
    print(f"  HSPC Positive: Found {pos}, Missing {miss}")
    
    neg, miss = process_category(hspc_ann.get('negative', []), 0, 'hspc', hspc_inventory)
    print(f"  HSPC Negative: Found {neg}, Missing {miss}")

    if len(training_samples) == 0:
        print("\nERROR: No training samples collected!")
        return

    # Get all feature names from first sample
    feature_names = sorted(training_samples[0]['features'].keys())
    print(f"\n  Feature count: {len(feature_names)}")
    print(f"    - Morphological: {len([f for f in feature_names if not f.startswith('sam2_') and not f.startswith('resnet_')])}")
    print(f"    - SAM2 embeddings: {len([f for f in feature_names if f.startswith('sam2_')])}")
    print(f"    - ResNet embeddings: {len([f for f in feature_names if f.startswith('resnet_')])}")

    # Build X and y - keep X as list of dicts for compatibility with retrain script
    X = []
    y = []
    sample_ids = []

    for sample in training_samples:
        features = sample['features']
        X.append(features)  # Keep as dict
        y.append(sample['label'])
        sample_ids.append(f"{sample['cell_type']}_{sample['slide']}_{sample['tile_id']}_{sample['det_id']}")

    print(f"\n  Final dataset shape: ({len(X)}, {len(feature_names)})")
    print(f"  Positive: {sum(y)}, Negative: {len(y) - sum(y)}")

    # Break down by cell type
    mk_count = sum(1 for s in training_samples if s['cell_type'] == 'mk')
    hspc_count = sum(1 for s in training_samples if s['cell_type'] == 'hspc')
    print(f"  MK samples: {mk_count}")
    print(f"  HSPC samples: {hspc_count}")

    # Save training data
    output_data = {
        'X': X,  # List of dicts
        'y': y,  # List of labels
        'feature_names': feature_names,
        'sample_ids': sample_ids,
        'training_samples': training_samples,  # Keep full sample info for reference
        'source': {
            'annotations': str(annotations_path),
            'base_dir': str(base_dir),
            'mk_samples_total': len(mk_inventory),
            'hspc_samples_total': len(hspc_inventory)
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved training data to: {output_path}")
    print("\nTo train classifier, run:")
    print(f"  python train_separate_classifiers.py \\")
    print(f"      --training-data {output_path} \\")
    print(f"      --output-mk /viper/ptmp2/edrod/MKsegmentation/mk_classifier.pkl \\")
    print(f"      --output-hspc /viper/ptmp2/edrod/MKsegmentation/hspc_classifier.pkl \\")
    print(f"      --morph-only  # Optional: use only morphological features")

    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Convert HTML annotations to training data format'
    )
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to all_labels_combined-2.json')
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Path to segmentation output directory (e.g., /ptmp/edrod/unified_10pct)')
    parser.add_argument('--mk-min-area-um', type=float,
                       help='Minimum MK area in µm² (must match segmentation filter)')
    parser.add_argument('--mk-max-area-um', type=float,
                       help='Maximum MK area in µm² (must match segmentation filter)')
    parser.add_argument('--output', type=str,
                       default='/viper/ptmp2/edrod/MKsegmentation/annotations/training_data.json',
                       help='Output path for training data')

    args = parser.parse_args()

    convert_annotations(
        annotations_path=args.annotations,
        base_dir=args.base_dir,
        output_path=args.output,
        mk_min_area_um=args.mk_min_area_um,
        mk_max_area_um=args.mk_max_area_um
    )


if __name__ == "__main__":
    main()
