#!/bin/bash
#SBATCH --job-name=test_cleanup
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/test_cleanup_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/test_cleanup_%j.err

# =============================================================================
# Test --cleanup-masks feature with 1% sampling
# =============================================================================

set -e

INPUT_LIST=${1:?"Usage: sbatch test_cleanup_masks.sbatch <input_files.txt> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch test_cleanup_masks.sbatch <input_files.txt> <output_dir>"}

NUM_GPUS=4
TILE_SIZE=3000
SAMPLE_FRACTION=0.01  # 1% for quick testing

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Mode: Multi-GPU with --cleanup-masks"
echo "Sample fraction: ${SAMPLE_FRACTION} (1%)"
echo "============================================================"

# Activate conda
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Show GPUs
nvidia-smi -L

# Force unbuffered output
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

# Read input files
mapfile -t CZI_FILES < "$INPUT_LIST"
echo "Processing ${#CZI_FILES[@]} CZI files with $NUM_GPUS GPUs"
echo "Input files:"
printf '  %s\n' "${CZI_FILES[@]}"

# =============================================================================
# PHASE 1-4: Segmentation with multi-GPU + cleanup-masks
# =============================================================================
python run_unified_FAST.py \
    --czi-paths "${CZI_FILES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size "$TILE_SIZE" \
    --sample-fraction "$SAMPLE_FRACTION" \
    --multi-gpu \
    --num-gpus "$NUM_GPUS" \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-max-area-um 500 \
    --cleanup-masks

echo ""
echo "============================================================"
echo "PHASE 5: Generating HTML annotation viewer"
echo "============================================================"

# Generate HTML from saved crops (no CZI reload needed)
python regenerate_html_fast.py \
    --output-base "$OUTPUT_DIR" \
    --samples-per-page 300 \
    --workers 16

echo ""
echo "============================================================"
echo "Job complete at $(date)"
echo "HTML viewer: $OUTPUT_DIR/html/index.html"
echo "============================================================"
