#!/bin/bash
# NMJ Multi-Node 100% Detection — 4 nodes (16 GPUs)
# "Detect once, classify later" workflow:
#   Step 1: Detection array (4 nodes, each processes 1/4 tiles)
#   Step 2: Merge (1 node, dedup + HTML)
#   Step 3 (manual, after merge): apply_classifier.py + regenerate_html.py
#
# Usage: bash slurm/run_nmj_multinode_100pct.sh [CLASSIFIER] [ANNOTATIONS]
#   No args = detect-only (recommended for new workflow)
#   With args = old workflow (classifier applied during detection)

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
OUTPUT_BASE=/fs/pool/pool-mann-edwin/nmj_output
SLIDE_NAME="20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch"
SAMPLE_FRACTION=1.0
NUM_NODES=4

# Optional classifier + annotations (pass as args or leave empty for detect-only)
CLASSIFIER="${1:-}"
ANNOTATIONS="${2:-}"

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCT=$(echo "$SAMPLE_FRACTION * 100" | bc | cut -d. -f1)
OUTPUT_DIR="${OUTPUT_BASE}/${SLIDE_NAME}_${TIMESTAMP}_${PCT}pct"

echo "=========================================="
echo "NMJ Multi-Node 100% Detection"
echo "Nodes: $NUM_NODES"
echo "Classifier: ${CLASSIFIER:-none (detect-only)}"
echo "Annotations: ${ANNOTATIONS:-none}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Validate classifier/annotations if provided
if [[ -n "$CLASSIFIER" && ! -f "$CLASSIFIER" ]]; then
    echo "ERROR: Classifier not found: $CLASSIFIER"
    exit 1
fi
if [[ -n "$ANNOTATIONS" && ! -f "$ANNOTATIONS" ]]; then
    echo "ERROR: Annotations not found: $ANNOTATIONS"
    exit 1
fi

# Create output directory + tiles subdir (all shards will write here)
mkdir -p "$OUTPUT_DIR/tiles"
mkdir -p "$REPO/slurm/logs"

# Submit detection array job (nodes 0..NUM_NODES-1)
ARRAY_SPEC="0-$((NUM_NODES - 1))"
DETECT_JOB=$(sbatch \
    --array="$ARRAY_SPEC" \
    --parsable \
    "$REPO/slurm/run_nmj_detect_shard.sh" \
    "$OUTPUT_DIR" \
    "$SAMPLE_FRACTION" \
    "$CLASSIFIER" \
    "$ANNOTATIONS")

echo "Detection array job submitted: $DETECT_JOB (${NUM_NODES} nodes)"

# Submit merge job (depends on all detection shards completing successfully)
MERGE_JOB=$(sbatch \
    --dependency=afterok:"$DETECT_JOB" \
    --parsable \
    "$REPO/slurm/run_nmj_merge.sh" \
    "$OUTPUT_DIR" \
    "$SAMPLE_FRACTION" \
    "$CLASSIFIER" \
    "$ANNOTATIONS")

echo "Merge job submitted: $MERGE_JOB (depends on $DETECT_JOB)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO/slurm/logs/nmj_shard_${DETECT_JOB}_*.out"
echo "  tail -f $REPO/slurm/logs/nmj_merge_${MERGE_JOB}.out"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
if [[ -z "$CLASSIFIER" ]]; then
    echo "Next steps (after merge completes):"
    echo "  1. Generate annotation HTML:"
    echo "     python scripts/regenerate_html.py --output-dir $OUTPUT_DIR --czi-path ... --max-samples 1500 --html-dir $OUTPUT_DIR/html_annotation"
    echo "  2. Annotate in browser, export annotations JSON"
    echo "  3. Train classifier:"
    echo "     python train_classifier.py --detections $OUTPUT_DIR/nmj_detections.json --annotations <annotations.json> --output-dir ./checkpoints"
    echo "  4. Score all detections:"
    echo "     python scripts/apply_classifier.py --detections $OUTPUT_DIR/nmj_detections.json --classifier ./checkpoints/nmj_classifier_rf_*.pkl"
    echo "  5. Review filtered HTML:"
    echo "     python scripts/regenerate_html.py --output-dir $OUTPUT_DIR --czi-path ... --detections $OUTPUT_DIR/nmj_detections_scored.json --score-threshold 0.5 --prior-annotations <annotations.json>"
fi
