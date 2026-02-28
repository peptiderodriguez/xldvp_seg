#!/bin/bash
# NMJ Multi-Node 15% Annotation Run — 4 nodes (16 GPUs)
# Step 1: Create timestamped output directory
# Step 2: Submit detection array job (4 nodes, each processes 1/4 tiles)
# Step 3: Submit merge job (1 node, dedup + HTML, depends on detection)
#
# Usage: bash slurm/run_nmj_multinode_15pct.sh

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
OUTPUT_BASE=/fs/pool/pool-mann-edwin/nmj_output
SLIDE_NAME="20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch"
SAMPLE_FRACTION=0.15
NUM_NODES=4

# Timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCT=$(echo "$SAMPLE_FRACTION * 100" | bc | cut -d. -f1)
OUTPUT_DIR="${OUTPUT_BASE}/${SLIDE_NAME}_${TIMESTAMP}_${PCT}pct"

echo "=========================================="
echo "NMJ Multi-Node 15% Annotation Run"
echo "Nodes: $NUM_NODES"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

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
    "$SAMPLE_FRACTION")

echo "Detection array job submitted: $DETECT_JOB (${NUM_NODES} nodes)"

# Submit merge job (depends on all detection shards completing successfully)
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
