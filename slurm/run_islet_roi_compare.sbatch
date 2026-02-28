#!/bin/bash
#SBATCH --job-name=islet_roi
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --gres=gpu:rtx5000:2
#SBATCH --time=1:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_roi_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_roi_%j.err

# =============================================================================
# ROI-based islet segmentation: nuclei-only vs PM+nuclei comparison
# Slide: BS-100 (6-channel: Bright, AF633, Gcg, Ins, DAPI, Sst)
#
# Finds islet regions first, then runs Cellpose+SAM2 on ROI crops only.
# 2 GPUs process ROIs in parallel. Much faster than full-slide (~10-20 min vs hours).
# =============================================================================

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-edwin/marvin_test/2025_09_03_30610012_BS-100.czi"
OUTPUT_DIR=/fs/pool/pool-mann-edwin/islet_output/BS100_roi_comparison
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Project: Islet ROI — Nuclei vs PM+Nuclei comparison (2 GPUs)"
echo "Output:  $OUTPUT_DIR"
echo "============================================================"

$PYTHON $REPO/scripts/segment_islet_regions.py \
    --czi-path "$CZI" \
    --marker-channels gcg:2,ins:3,sst:5 \
    --membrane-channel 1 \
    --nuclear-channel 4 \
    --display-channels 2,3,5 \
    --otsu-multiplier 2.0 \
    --buffer-um 25 \
    --num-gpus 2 \
    --output-dir "$OUTPUT_DIR"

echo "=== Done at $(date) ==="
