#!/bin/bash
#SBATCH --job-name=nmj_merge
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --time=4:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_merge_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_merge_%j.err

# NMJ merge step — dedup + HTML + CSV after all detection shards complete
# Usage: sbatch --dependency=afterok:<DETECT_JOB_ID> run_nmj_merge.sh <OUTPUT_DIR> <SAMPLE_FRACTION> [CLASSIFIER] [ANNOTATIONS]
#
# Arguments:
#   $1 = OUTPUT_DIR (shared directory where all shards wrote tiles)
#   $2 = SAMPLE_FRACTION (must match what detection used)
#   $3 = CLASSIFIER path (optional)
#   $4 = ANNOTATIONS path (optional)

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi"
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

OUTPUT_DIR="${1:?ERROR: OUTPUT_DIR required as first argument}"
SAMPLE_FRACTION="${2:?ERROR: SAMPLE_FRACTION required as second argument}"
CLASSIFIER="${3:-}"
ANNOTATIONS="${4:-}"

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p slurm/logs

# Count tiles written by all shards
TILE_COUNT=$(ls -d "$OUTPUT_DIR"/tiles/tile_*/ 2>/dev/null | wc -l)

echo "=========================================="
echo "NMJ Merge Step (dedup + HTML + CSV)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Output dir: $OUTPUT_DIR"
echo "Tiles found: $TILE_COUNT"
echo "Sample fraction: $SAMPLE_FRACTION"
echo "Classifier: ${CLASSIFIER:-none}"
echo "=========================================="

# Build command: --resume auto-detects shard manifests and enables --merge-shards
# Checkpointed: merged_detections.json → deduped detections.json → HTML
CMD=(
    $PYTHON run_segmentation.py
    --czi-path "$CZI"
    --cell-type nmj
    --channel 1
    --output-dir "$(dirname "$OUTPUT_DIR")"
    --resume "$OUTPUT_DIR"
    --merge-shards
    --sample-fraction "$SAMPLE_FRACTION"
    --tile-size 3000
    --tile-overlap 0.10
    --num-gpus 1
    --load-to-ram
    --all-channels
    --random-seed 42
    --no-serve
)

# Add classifier args if provided
if [[ -n "$CLASSIFIER" ]]; then
    CMD+=(--nmj-classifier "$CLASSIFIER")
fi
if [[ -n "$ANNOTATIONS" ]]; then
    CMD+=(--prior-annotations "$ANNOTATIONS")
fi

"${CMD[@]}"

echo "=========================================="
echo "Merge done: $(date)"
echo "Tiles processed: $TILE_COUNT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
