#!/bin/bash
#SBATCH --job-name=islet_test
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=8:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_test_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_test_%j.err

# Islet 10% test run — 2 GPUs for faster scheduling
# BS-100 slide: 35 tiles, 29 tissue → 3 tiles at 10%

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

echo "=== Islet 10% test run ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 2"

$PYTHON run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type islet \
    --channel 1 \
    --membrane-channel 1 \
    --nuclear-channel 4 \
    --all-channels \
    --tile-size 3000 \
    --tile-overlap 0.10 \
    --sample-fraction 0.10 \
    --load-to-ram \
    --multi-gpu --num-gpus 2 \
    --output-dir "$OUTPUT_DIR" \
    --no-serve

echo "Done: $(date)"
