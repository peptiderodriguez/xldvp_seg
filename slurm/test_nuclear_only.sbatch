#!/bin/bash
#SBATCH --job-name=nuclear_only
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/nuclear_only_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/nuclear_only_%j.err

set -e

INPUT_LIST=${1:?\"Usage: sbatch test_nuclear_only.sbatch <input_files.txt> <output_dir>\"}
OUTPUT_DIR=${2:?\"Usage: sbatch test_nuclear_only.sbatch <input_files.txt> <output_dir>\"}

NUM_GPUS=4
TILE_SIZE=3000
SAMPLE_FRACTION=0.10  # 10%

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Mode: Multi-GPU with --hspc-nuclear-only --cleanup-masks"
echo "Sample fraction: ${SAMPLE_FRACTION} (10%)"
echo "============================================================"

source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

nvidia-smi -L

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

mapfile -t CZI_FILES < "$INPUT_LIST"
echo "Processing ${#CZI_FILES[@]} CZI files with $NUM_GPUS GPUs"
echo "Input files:"
printf '  %s\n' "${CZI_FILES[@]}"

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
    --cleanup-masks \
    --hspc-nuclear-only

echo ""
echo "============================================================"
echo "Job complete at $(date)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
