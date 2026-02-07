#!/bin/bash
#SBATCH --job-name=mkseg_4gpu
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.err

# =============================================================================
# RAM-First Mode with 4 GPUs
# =============================================================================

set -e

INPUT_LIST=${1:?"Usage: sbatch run_ram_4gpu.sbatch <input_files.txt> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch run_ram_4gpu.sbatch <input_files.txt> <output_dir>"}

NUM_WORKERS=4
TILE_SIZE=3000
SAMPLE_FRACTION=0.10

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Mode: RAM-First (4 GPUs)"
echo "============================================================"

# Activate conda
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Show GPUs
nvidia-smi -L

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Read input files
mapfile -t CZI_FILES < "$INPUT_LIST"
echo "Processing ${#CZI_FILES[@]} CZI files with 4 GPUs"

# Run - RAM-first mode (no --multi-gpu flag)
python run_unified_FAST.py \
    --czi-paths "${CZI_FILES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size "$TILE_SIZE" \
    --sample-fraction "$SAMPLE_FRACTION" \
    --num-workers "$NUM_WORKERS" \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000

echo "Job complete at $(date)"
