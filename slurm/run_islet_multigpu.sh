#!/bin/bash
#SBATCH --job-name=islet_seg
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_%j.err

# Islet multi-GPU segmentation
# 6-channel CZI: Bright(0), AF633/membrane(1), AF555/Gcg(2), AF488/Ins(3), DAPI(4), Cy7/Sst(5)
# Detection uses membrane(ch1) + DAPI(ch4) as Cellpose input
# HTML display: R=Gcg(ch2), G=Ins(ch3), B=Sst(ch5)

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="${1:?Usage: sbatch run_islet_multigpu.sh <czi_path> [classifier_path]}"
CLASSIFIER="${2:-}"
OUTPUT_DIR=/fs/pool/pool-mann-edwin/islet_output
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

echo "=========================================="
echo "Islet Multi-GPU Segmentation"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CZI: $CZI"
echo "Classifier: ${CLASSIFIER:-none (annotation run)}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Build classifier arguments
CLASSIFIER_ARGS=""
if [ -n "$CLASSIFIER" ]; then
    CLASSIFIER_ARGS="--islet-classifier $CLASSIFIER"
fi

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
    --no-serve \
    $CLASSIFIER_ARGS

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
