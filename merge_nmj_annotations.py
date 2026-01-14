#!/usr/bin/env python3
"""
Merge old and new NMJ annotations.
New annotations overwrite old ones for the same sample ID.
Unsure annotations are excluded from training.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def parse_annotations(data, slide_prefix=""):
    """Parse annotations from either format.

    Format 1 (old): {"positive": [...], "negative": [...]}
    Format 2 (new): {"annotations": {"id": "yes"|"no"}}
    """
    positive = set()
    negative = set()
    unsure = set()

    if 'annotations' in data:
        # New format: {"annotations": {"id": "yes"|"no"}}
        for sample_id, label in data['annotations'].items():
            # Add slide prefix if needed (new format has shorter IDs)
            full_id = f"{slide_prefix}{sample_id}" if slide_prefix else sample_id
            if label == 'yes':
                positive.add(full_id)
            elif label == 'no':
                negative.add(full_id)
            elif label == 'unsure':
                unsure.add(full_id)
    else:
        # Old format: {"positive": [...], "negative": [...]}
        positive = set(data.get('positive', []))
        negative = set(data.get('negative', []))
        unsure = set(data.get('unsure', []))

    return positive, negative, unsure


def merge_annotations(old_path, new_path, output_path, slide_prefix="20251109_PMCA1_647_nuc488-EDFvar-stitch_tile_"):
    """Merge annotations with new overwriting old."""

    # Load old annotations
    print(f"Loading old annotations from: {old_path}")
    with open(old_path) as f:
        old = json.load(f)

    old_positive, old_negative, _ = parse_annotations(old)

    print(f"  Old: {len(old_positive)} positive, {len(old_negative)} negative")

    # Load new annotations
    print(f"\nLoading new annotations from: {new_path}")
    with open(new_path) as f:
        new = json.load(f)

    # Check if new format needs slide prefix
    if 'annotations' in new:
        # Check if IDs already have prefix
        sample_id = next(iter(new['annotations'].keys()), "")
        if not sample_id.startswith("20"):
            print(f"  Adding slide prefix: {slide_prefix}")
        else:
            slide_prefix = ""
    else:
        slide_prefix = ""

    new_positive, new_negative, new_unsure = parse_annotations(new, slide_prefix)

    print(f"  New: {len(new_positive)} positive, {len(new_negative)} negative, {len(new_unsure)} unsure")

    # Track conflicts
    conflicts = {
        'pos_to_neg': 0,  # Was positive, now negative
        'neg_to_pos': 0,  # Was negative, now positive
        'to_unsure': 0,   # Was labeled, now unsure (excluded)
    }

    # Build merged set - start with old
    merged_positive = set(old_positive)
    merged_negative = set(old_negative)

    # Apply new annotations (overwrites)
    for sample_id in new_positive:
        if sample_id in merged_negative:
            merged_negative.remove(sample_id)
            conflicts['neg_to_pos'] += 1
        merged_positive.add(sample_id)

    for sample_id in new_negative:
        if sample_id in merged_positive:
            merged_positive.remove(sample_id)
            conflicts['pos_to_neg'] += 1
        merged_negative.add(sample_id)

    # Remove unsure samples (don't use for training)
    for sample_id in new_unsure:
        if sample_id in merged_positive:
            merged_positive.remove(sample_id)
            conflicts['to_unsure'] += 1
        if sample_id in merged_negative:
            merged_negative.remove(sample_id)
            conflicts['to_unsure'] += 1

    # Report conflicts
    print(f"\nConflicts resolved (new overwrites old):")
    print(f"  Positive → Negative: {conflicts['pos_to_neg']}")
    print(f"  Negative → Positive: {conflicts['neg_to_pos']}")
    print(f"  Labeled → Unsure (excluded): {conflicts['to_unsure']}")

    # Save merged
    merged = {
        'positive': sorted(list(merged_positive)),
        'negative': sorted(list(merged_negative)),
        'metadata': {
            'old_file': str(old_path),
            'new_file': str(new_path),
            'old_counts': {'positive': len(old_positive), 'negative': len(old_negative)},
            'new_counts': {'positive': len(new_positive), 'negative': len(new_negative), 'unsure': len(new_unsure)},
            'conflicts': conflicts
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged annotations saved to: {output_path}")
    print(f"  Final: {len(merged_positive)} positive, {len(merged_negative)} negative")
    print(f"  Total training samples: {len(merged_positive) + len(merged_negative)}")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Merge NMJ annotations')
    parser.add_argument('--old', type=str,
                        default='/home/dude/nmj_test_output/nmj_annotations.json',
                        help='Path to old annotations JSON')
    parser.add_argument('--new', type=str,
                        required=True,
                        help='Path to new annotations JSON (exported from HTML)')
    parser.add_argument('--output', type=str,
                        default='/home/dude/nmj_output/nmj_annotations_merged.json',
                        help='Output path for merged annotations')
    args = parser.parse_args()

    merge_annotations(args.old, args.new, args.output)


if __name__ == '__main__':
    main()
