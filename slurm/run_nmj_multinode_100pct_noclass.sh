#!/bin/bash
# NMJ Multi-Node 100% Detection (No Classifier) — 4 nodes (16 GPUs)
# "Detect once, classify later" workflow:
#   1. This script: 100% detection across all tiles (no classifier needed)
#   2. After merge: regenerate_html.py --max-samples 1500 → annotation subset
#   3. Annotate in HTML → export annotations JSON
#   4. train_classifier.py → train RF classifier
#   5. scripts/apply_classifier.py → score ALL detections (CPU-only, seconds)
#   6. regenerate_html.py --score-threshold 0.5 --prior-annotations ann.json → review HTML
#
# Usage: bash slurm/run_nmj_multinode_100pct_noclass.sh

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
OUTPUT_BASE=/fs/pool/pool-mann-edwin/nmj_output
SLIDE_NAME="20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch"
SAMPLE_FRACTION=1.0
NUM_NODES=4

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCT=$(echo "$SAMPLE_FRACTION * 100" | bc | cut -d. -f1)
OUTPUT_DIR="${OUTPUT_BASE}/${SLIDE_NAME}_${TIMESTAMP}_${PCT}pct"

echo "=========================================="
echo "NMJ Multi-Node 100% Detection (No Classifier)"
echo "Nodes: $NUM_NODES"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Workflow after this completes:"
echo "  1. regenerate_html.py --max-samples 1500 --html-dir \$OUTPUT/html_annotation"
echo "  2. Annotate → export annotations JSON"
echo "  3. train_classifier.py"
echo "  4. scripts/apply_classifier.py → score all detections"
echo "  5. regenerate_html.py --score-threshold 0.5 --prior-annotations ann.json"
echo ""

# Create output directory + tiles subdir (all shards will write here)
mkdir -p "$OUTPUT_DIR/tiles"
mkdir -p "$REPO/slurm/logs"

# Submit detection array job (nodes 0..NUM_NODES-1)
# No classifier or annotations args — pure detection
ARRAY_SPEC="0-$((NUM_NODES - 1))"
DETECT_JOB=$(sbatch \
    --array="$ARRAY_SPEC" \
    --parsable \
    "$REPO/slurm/run_nmj_detect_shard.sh" \
    "$OUTPUT_DIR" \
    "$SAMPLE_FRACTION")

echo "Detection array job submitted: $DETECT_JOB (${NUM_NODES} nodes)"

# Submit merge job (depends on all detection shards completing successfully)
# No classifier or annotations — just dedup + HTML of ALL detections
MERGE_JOB=$(sbatch \
    --dependency=afterok:"$DETECT_JOB" \
    --parsable \
    "$REPO/slurm/run_nmj_merge.sh" \
    "$OUTPUT_DIR" \
    "$SAMPLE_FRACTION")

echo "Merge job submitted: $MERGE_JOB (depends on $DETECT_JOB)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $REPO/slurm/logs/nmj_shard_${DETECT_JOB}_*.out"
echo "  tail -f $REPO/slurm/logs/nmj_merge_${MERGE_JOB}.out"
echo ""
echo "Output directory: $OUTPUT_DIR"
