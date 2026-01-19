#!/bin/bash
#SBATCH --job-name=mkseg_2gpu
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=370G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx5000:2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Multi-GPU on p.hpcl8 (2x RTX 5000, 380GB RAM, 24 CPUs)
# =============================================================================

set -e

INPUT_LIST=${1:?"Usage: sbatch run_multigpu_hpcl8.sbatch <input_files.txt> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch run_multigpu_hpcl8.sbatch <input_files.txt> <output_dir>"}

NUM_GPUS=2
TILE_SIZE=3000
SAMPLE_FRACTION=0.10

# --- Environment Setup ---
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs requested: $NUM_GPUS"
echo "Mode: Multi-GPU (1 tile per GPU)"
echo "============================================================"

# Load conda (cluster-specific path)
CONDA_BASE="/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
    source ~/miniforge3/etc/profile.d/conda.sh
fi
conda activate mkseg

# Verify GPUs
nvidia-smi --list-gpus

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# --- Read input files into array ---
mapfile -t CZI_FILES < "$INPUT_LIST"

echo ""
echo "Processing ${#CZI_FILES[@]} CZI files with $NUM_GPUS GPUs"
echo "Each GPU processes 1 tile at a time (true parallel tile processing)"
echo ""

# --- Run with multi-GPU mode ---
python run_unified_FAST.py \
    --czi-paths "${CZI_FILES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size "$TILE_SIZE" \
    --sample-fraction "$SAMPLE_FRACTION" \
    --multi-gpu \
    --num-gpus "$NUM_GPUS"

echo ""
echo "============================================================"
echo "Multi-GPU processing completed"
echo "============================================================"
