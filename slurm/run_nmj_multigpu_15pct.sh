#!/bin/bash
#SBATCH --job-name=nmj_15pct
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/xldvp_seg/slurm/logs/nmj_15pct_%j.out
#SBATCH --error=/path/to/xldvp_seg/slurm/logs/nmj_15pct_%j.err

# NMJ multi-GPU segmentation — 15% annotation run
# Slide: 20251107_Fig5 (3-channel: nuc488, Bgtx647, NfL750)
# Channel 1 = BTX (647nm) — NMJ detection channel

set -euo pipefail

XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"
REPO="${REPO:-/path/to/xldvp_seg}"
CZI="/path/to/data/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi"
OUTPUT_DIR=/path/to/output/nmj_output
PYTHON="$XLDVP_PYTHON"

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

echo "=========================================="
echo "NMJ Multi-GPU 15% Annotation Run"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CZI: $CZI"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

$PYTHON run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type nmj \
    --channel 1 \
    --output-dir "$OUTPUT_DIR" \
    --sample-fraction 0.15 \
    --tile-size 3000 \
    --tile-overlap 0.10 \
    --multi-gpu \
    --num-gpus 4 \
    --load-to-ram \
    --all-channels

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
