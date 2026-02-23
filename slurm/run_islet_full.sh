#!/bin/bash
#SBATCH --job-name=islet_full
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=12:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_full_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_full_%j.err

# Islet 100% run — 4 GPUs
# BS-100 slide: 14 tissue tiles, ~7K cells/tile, ~4h expected

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI=/fs/pool/pool-mann-edwin/marvin_test/2025_09_03_30610012_BS-100.czi
OUTPUT_DIR=/fs/pool/pool-mann-edwin/islet_output
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

echo "=== Islet 100% full run ==="
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
    --sample-fraction 1.0 \
    --load-to-ram \
    --multi-gpu --num-gpus 4 \
    --output-dir "$OUTPUT_DIR" \
    --no-serve

echo "Detection done: $(date)"

# Find the run directory (most recent islet output)
RUN_DIR=$(ls -dt "$OUTPUT_DIR"/2025_09_03_30610012_BS-100_* | head -1)
echo "Run dir: $RUN_DIR"

# Post-hoc islet analysis with Otsu quality filter
$PYTHON "$REPO/scripts/analyze_islets.py" \
    --run-dir "$RUN_DIR" \
    --czi-path "$CZI" \
    --threshold-factor 2.0 \
    --quality-filter otsu \
    --buffer-um 25 --min-cells 5 --no-recruit

echo "Done: $(date)"
