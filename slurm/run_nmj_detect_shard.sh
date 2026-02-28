#!/bin/bash
#SBATCH --job-name=nmj_shard
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=12:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_shard_%A_%a.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_shard_%A_%a.err

# NMJ detection shard — called by array job
# Usage: sbatch --array=0-3 run_nmj_detect_shard.sh <OUTPUT_DIR> <SAMPLE_FRACTION> [CLASSIFIER] [ANNOTATIONS]
#
# Arguments:
#   $1 = OUTPUT_DIR (shared across all shards, pre-created by wrapper)
#   $2 = SAMPLE_FRACTION (e.g. 0.15 or 1.0)
#   $3 = CLASSIFIER path (optional, for 100% run)
#   $4 = ANNOTATIONS path (optional, for 100% run)

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi"
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

OUTPUT_DIR="${1:?ERROR: OUTPUT_DIR required as first argument}"
SAMPLE_FRACTION="${2:?ERROR: SAMPLE_FRACTION required as second argument}"
CLASSIFIER="${3:-}"
ANNOTATIONS="${4:-}"

SHARD_IDX=${SLURM_ARRAY_TASK_ID}
# Compute total from MIN/MAX (SLURM_ARRAY_TASK_COUNT not available in all SLURM versions)
SHARD_TOTAL=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p slurm/logs

echo "=========================================="
echo "NMJ Detection Shard ${SHARD_IDX}/${SHARD_TOTAL}"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CZI: $CZI"
echo "Output: $OUTPUT_DIR"
echo "Sample fraction: $SAMPLE_FRACTION"
echo "Classifier: ${CLASSIFIER:-none}"
echo "=========================================="

# Build command
CMD=(
    $PYTHON run_segmentation.py
    --czi-path "$CZI"
    --cell-type nmj
    --channel 1
    --output-dir "$(dirname "$OUTPUT_DIR")"
    --resume "$OUTPUT_DIR"
    --sample-fraction "$SAMPLE_FRACTION"
    --tile-size 3000
    --tile-overlap 0.10
    --multi-gpu
    --num-gpus 4
    --load-to-ram
    --all-channels
    --tile-shard "${SHARD_IDX}/${SHARD_TOTAL}"
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
echo "Shard ${SHARD_IDX}/${SHARD_TOTAL} done: $(date)"
echo "=========================================="
