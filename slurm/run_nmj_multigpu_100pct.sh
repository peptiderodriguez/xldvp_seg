#!/bin/bash
#SBATCH --job-name=nmj_100pct
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=36:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_100pct_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_100pct_%j.err

# NMJ multi-GPU segmentation — 100% with classifier
# Slide: 20251107_Fig5 (3-channel: nuc488, Bgtx647, NfL750)
# Channel 1 = BTX (647nm) — NMJ detection channel

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi"
OUTPUT_DIR=/fs/pool/pool-mann-edwin/nmj_output
CLASSIFIER=/fs/pool/pool-mann-edwin/nmj_output/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch_20260210_112830/classifier/nmj_classifier_rf.pkl
ANNOTATIONS=/fs/pool/pool-mann-edwin/nmj_output/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch_20260210_112830/nmj_20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch_1770728512_annotations.json
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

echo "=========================================="
echo "NMJ Multi-GPU 100% Classifier Run"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CZI: $CZI"
echo "Classifier: $CLASSIFIER"
echo "Prior annotations: $ANNOTATIONS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

$PYTHON run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type nmj \
    --channel 1 \
    --output-dir "$OUTPUT_DIR" \
    --sample-fraction 1.0 \
    --tile-size 3000 \
    --tile-overlap 0.10 \
    --multi-gpu \
    --num-gpus 4 \
    --load-to-ram \
    --all-channels \
    --nmj-classifier "$CLASSIFIER" \
    --prior-annotations "$ANNOTATIONS"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
