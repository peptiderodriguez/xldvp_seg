#!/bin/bash
#SBATCH --job-name=mkseg
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MK/HSPC Segmentation Pipeline - Slurm Batch Script
# =============================================================================
#
# Example usage:
#   sbatch slurm/run_segmentation.sbatch /path/to/input.czi /path/to/output
#
# For multiple files:
#   sbatch --array=0-9 slurm/run_segmentation.sbatch file_list.txt /path/to/output
#
# Partition guide for HPCL8:
#   p.hpcl8:  RTX 5000 GPUs, 377GB RAM, 24 cores (default)
#   p.hpcl93: L40s GPUs, 755GB RAM, 128 cores (recommended for large slides)
#   p.hpcl9:  A40 GPUs, 1TB RAM, 72 cores (Dept. Briggs only)
#   p.hpcl91: H100 GPUs, 1TB RAM, 76 cores (b_borgwardt group only)
#
# =============================================================================

set -e

# --- Configuration ---
CZI_PATH=${1:?"Usage: sbatch run_segmentation.sbatch <czi_path> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch run_segmentation.sbatch <czi_path> <output_dir>"}

# Adjust these based on your partition's resources
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-16}
TILE_SIZE=3000
SAMPLE_FRACTION=0.10

# --- Environment Setup ---
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Load conda environment
# Adjust this path for your system
if [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
    source ~/miniforge3/etc/profile.d/conda.sh
elif [ -f /etc/profile.d/conda.sh ]; then
    source /etc/profile.d/conda.sh
fi

conda activate mkseg

# Check GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# --- Run Segmentation ---
echo ""
echo "Starting segmentation..."
echo "Input: $CZI_PATH"
echo "Output: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS"
echo ""

cd /home/dude/code/xldvp_seg_repo

python run_unified_FAST.py \
    --czi-path "$CZI_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$NUM_WORKERS" \
    --tile-size "$TILE_SIZE" \
    --sample-fraction "$SAMPLE_FRACTION"

echo ""
echo "============================================================"
echo "Job completed successfully"
echo "============================================================"
