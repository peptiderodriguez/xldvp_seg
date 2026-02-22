#!/bin/bash
#SBATCH --job-name=vessel_ms
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.err

# =============================================================================
# Vessel Multi-Scale Detection — 4x L40S (Unified: ring + lumen-first)
#
# Input:  259 GB CZI (4-channel: nuc488, CD31_555, SMA647, PM750)
# Output: /fs/pool/pool-mann-edwin/vessel_output/multiscale/
# Scales: 1/32 → 1/16 → 1/8 → 1/4 → 1/2
# Tile:   3000x3000, 10% overlap
# Detection: Ring (Canny+hierarchy) + supplementary lumen-first (Otsu+wall)
# Max vessel area: 90% of tile area
# Contour smoothing: ON (factor 3.0, scale-aware after upscaling)
# Dedup: keep largest contour area, IoU 0.3 cross-method merge
# =============================================================================

set -e

CZI_PATH="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT_DIR="/fs/pool/pool-mann-edwin/vessel_output/multiscale"

echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"
echo "Mode:       Multi-GPU Shared Memory (4x L40S)"
echo "Pipeline:   Vessel Multi-Scale Detection"
echo "CZI:        $CZI_PATH"
echo "Output:     $OUTPUT_DIR"
echo "============================================================"

# Activate conda
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Show GPUs
nvidia-smi -L

# Environment
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run vessel detection
# Channel 2 = SMA647 (smooth muscle actin, main vessel detection channel)
# --candidate-mode: keep all candidates for RF training/annotation
# Unified detection: ring + supplementary lumen-first (default, no flag needed)
# --load-to-ram: faster for network mount
python run_segmentation.py \
    --czi-path "$CZI_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --cell-type vessel \
    --channel 2 \
    --tile-size 3000 \
    --tile-overlap 0.10 \
    --sample-fraction 1.0 \
    --num-gpus 4 \
    --load-to-ram \
    --candidate-mode \
    --multi-scale \
    --scales "32,16,8,4,2" \
    --multiscale-iou-threshold 0.3 \
    --smooth-contours-factor 3.0 \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo "============================================================"
echo "Job complete at $(date)"
echo "============================================================"
