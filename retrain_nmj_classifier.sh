#!/bin/bash
# Retrain NMJ classifier with merged annotations
# Usage: ./retrain_nmj_classifier.sh /path/to/new_annotations.json

set -e

NEW_ANNOTATIONS="${1:?Usage: $0 /path/to/new_annotations.json}"
OLD_ANNOTATIONS="/home/dude/nmj_test_output/nmj_annotations.json"
MERGED_ANNOTATIONS="/home/dude/nmj_output/nmj_annotations_merged.json"
HTML_DIR_OLD="/home/dude/nmj_output/html"
HTML_DIR_NEW="/home/dude/nmj_output/20251109_PMCA1_647_nuc488-EDFvar-stitch/inference/html"
OUTPUT_DIR="/home/dude/nmj_output"

echo "============================================"
echo "NMJ Classifier Retraining Pipeline"
echo "============================================"
echo ""
echo "Old annotations: $OLD_ANNOTATIONS"
echo "New annotations: $NEW_ANNOTATIONS"
echo ""

# Step 1: Merge annotations
echo "Step 1: Merging annotations..."
python3 /home/dude/code/xldvp_seg_repo/merge_nmj_annotations.py \
    --old "$OLD_ANNOTATIONS" \
    --new "$NEW_ANNOTATIONS" \
    --output "$MERGED_ANNOTATIONS"

echo ""
echo "Step 2: Training classifier..."
python3 /home/dude/code/xldvp_seg_repo/train_nmj_classifier.py \
    --annotations "$MERGED_ANNOTATIONS" \
    --html-dir "$HTML_DIR_OLD" "$HTML_DIR_NEW" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 20

echo ""
echo "============================================"
echo "Retraining complete!"
echo "Model saved to: $OUTPUT_DIR/nmj_classifier.pth"
echo "============================================"
