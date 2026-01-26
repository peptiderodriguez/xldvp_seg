#!/bin/bash
#SBATCH --job-name=mkseg_l40s
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=600G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.err

# =============================================================================
# L40S Multi-GPU Mode with HSPC <500um² filter
# =============================================================================

set -e

INPUT_LIST=${1:?"Usage: sbatch run_l40s_4gpu.sbatch <input_files.txt> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch run_l40s_4gpu.sbatch <input_files.txt> <output_dir>"}

NUM_GPUS=4
TILE_SIZE=3000
SAMPLE_FRACTION=0.10

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Mode: Multi-GPU Shared Memory (4x L40S)"
echo "HSPC filter: <500 um²"
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

# Run with multi-GPU shared memory mode
python run_unified_FAST.py \
    --czi-paths "${CZI_FILES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size "$TILE_SIZE" \
    --sample-fraction "$SAMPLE_FRACTION" \
    --multi-gpu \
    --num-gpus "$NUM_GPUS" \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-max-area-um 500

echo "Job complete at $(date)"
