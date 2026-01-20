#!/bin/bash
#SBATCH --job-name=mkseg_4gpu
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=700G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Multi-GPU Parallel Processing (4 GPUs, 1 tile per GPU at a time)
# =============================================================================
#
# This script uses Python's built-in multi-GPU processing where:
# - All slides are loaded into RAM
# - 4 GPU workers are spawned, each pinned to one GPU
# - Each GPU processes ONE tile at a time from the tile queue
# - Tiles from the same slide are processed in parallel across GPUs
#
# Usage:
#   sbatch slurm/run_multigpu.sbatch input_files.txt /path/to/output
#
# Where input_files.txt contains one CZI path per line.
#
# Partition recommendations for 4 GPUs:
#   p.hpcl93: 4x L40s, 755GB RAM, 128 cores (recommended)
#   p.hpcl9:  4x A40, 1TB RAM, 72 cores (Dept. Briggs only)
#   p.hpcl91: 4x H100, 1TB RAM, 76 cores (b_borgwardt only)
#
# =============================================================================

set -e

INPUT_LIST=${1:?"Usage: sbatch run_multigpu.sbatch <input_files.txt> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch run_multigpu.sbatch <input_files.txt> <output_dir>"}

NUM_GPUS=4
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

# Set thread counts for CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# PyTorch CPU threads (per process, so divide by num workers + 1)
export PYTORCH_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 5))

echo "CPU threads: OMP=$OMP_NUM_THREADS, PyTorch=$PYTORCH_NUM_THREADS"

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
