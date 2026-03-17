#!/bin/bash
#SBATCH --job-name=islet_test
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=8:00:00
#SBATCH --output=/path/to/xldvp_seg/slurm/logs/islet_test_%j.out
#SBATCH --error=/path/to/xldvp_seg/slurm/logs/islet_test_%j.err

# Islet 10% test run — 4 GPUs
# BS-100 slide: 35 tiles, 29 tissue → 3 tiles at 10%

set -euo pipefail

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
REPO="${REPO:-/path/to/xldvp_seg}"
CZI=/path/to/data/2025_09_03_30610012_BS-100.czi
OUTPUT_DIR=/path/to/output/islet_output
PYTHON="$MKSEG_PYTHON"

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

echo "=== Islet 10% test run ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4"

$PYTHON run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type islet \
    --channel 1 \
    --membrane-channel 1 \
    --nuclear-channel 4 \
    --all-channels \
    --tile-size 4000 \
    --tile-overlap 0.25 \
    --sample-fraction 0.10 \
    --load-to-ram \
    --multi-gpu --num-gpus 4 \
    --output-dir "$OUTPUT_DIR" \
    --no-serve

echo "Done: $(date)"
