#!/bin/bash
#SBATCH --job-name=nmj_100pct
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=36:00:00
#SBATCH --output=/path/to/xldvp_seg/slurm/logs/nmj_100pct_%j.out
#SBATCH --error=/path/to/xldvp_seg/slurm/logs/nmj_100pct_%j.err

# NMJ multi-GPU segmentation — 100% with classifier
# Slide: 20251107_Fig5 (3-channel: nuc488, Bgtx647, NfL750)
# Channel 1 = BTX (647nm) — NMJ detection channel

set -euo pipefail

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
REPO="${REPO:-/path/to/xldvp_seg}"
CZI="/path/to/data/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi"
OUTPUT_DIR=/path/to/output/nmj_output
# UPDATE THESE after 10% annotation + classifier training:
CLASSIFIER=/path/to/output/nmj_output/FILL_IN_10PCT_DIR/classifier/nmj_classifier_rf.pkl
ANNOTATIONS=/path/to/output/nmj_output/FILL_IN_10PCT_DIR/FILL_IN_ANNOTATIONS.json
PYTHON="$MKSEG_PYTHON"

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
