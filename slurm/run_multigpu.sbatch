#!/bin/bash
#SBATCH --job-name=mkseg_4gpu
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Multi-GPU Parallel Processing (4 GPUs, 4 parallel tile processors)
# =============================================================================
#
# This script runs 4 parallel processes, each pinned to a different GPU.
# Each process handles a subset of the input files.
#
# Usage:
#   sbatch slurm/run_multigpu.sbatch input_files.txt /path/to/output
#
# Where input_files.txt contains one CZI path per line.
#
# Partition recommendations for 4 GPUs:
#   p.hpcl93: 4x L40s, 755GB RAM, 128 cores
#   p.hpcl9:  4x A40, 1TB RAM, 72 cores (Dept. Briggs only)
#   p.hpcl91: 4x H100, 1TB RAM, 76 cores (b_borgwardt only)
#
# =============================================================================

set -e

INPUT_LIST=${1:?"Usage: sbatch run_multigpu.sbatch <input_files.txt> <output_dir>"}
OUTPUT_DIR=${2:?"Usage: sbatch run_multigpu.sbatch <input_files.txt> <output_dir>"}

NUM_GPUS=4
WORKERS_PER_GPU=8
TILE_SIZE=3000
SAMPLE_FRACTION=0.10

# --- Environment Setup ---
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs requested: $NUM_GPUS"
echo "============================================================"

# Load conda
if [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
    source ~/miniforge3/etc/profile.d/conda.sh
elif [ -f /etc/profile.d/conda.sh ]; then
    source /etc/profile.d/conda.sh
fi
conda activate mkseg

# Verify GPUs
nvidia-smi --list-gpus

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

cd /home/dude/code/xldvp_seg_repo

# --- Split input files across GPUs ---
# Count total files
TOTAL_FILES=$(wc -l < "$INPUT_LIST")
FILES_PER_GPU=$(( (TOTAL_FILES + NUM_GPUS - 1) / NUM_GPUS ))

echo "Total files: $TOTAL_FILES"
echo "Files per GPU: ~$FILES_PER_GPU"
echo ""

# Create temporary file lists for each GPU
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    start=$((gpu_id * FILES_PER_GPU + 1))
    end=$((start + FILES_PER_GPU - 1))
    sed -n "${start},${end}p" "$INPUT_LIST" > "/tmp/gpu${gpu_id}_files_${SLURM_JOB_ID}.txt"
    file_count=$(wc -l < "/tmp/gpu${gpu_id}_files_${SLURM_JOB_ID}.txt")
    echo "GPU $gpu_id: $file_count files"
done

echo ""
echo "Starting parallel processing..."
echo ""

# --- Launch parallel processes, each pinned to a different GPU ---
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    file_list="/tmp/gpu${gpu_id}_files_${SLURM_JOB_ID}.txt"

    if [ -s "$file_list" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id

            # Read files into array
            mapfile -t czi_files < "$file_list"

            if [ ${#czi_files[@]} -gt 0 ]; then
                echo "[GPU $gpu_id] Processing ${#czi_files[@]} files..."

                python run_unified_FAST.py \
                    --czi-paths "${czi_files[@]}" \
                    --output-dir "$OUTPUT_DIR" \
                    --num-workers "$WORKERS_PER_GPU" \
                    --tile-size "$TILE_SIZE" \
                    --sample-fraction "$SAMPLE_FRACTION" \
                    2>&1 | while read line; do echo "[GPU $gpu_id] $line"; done

                echo "[GPU $gpu_id] Completed."
            fi
        ) &
    fi
done

# Wait for all background processes
wait

# Cleanup temp files
rm -f /tmp/gpu*_files_${SLURM_JOB_ID}.txt

echo ""
echo "============================================================"
echo "All GPU processes completed"
echo "============================================================"
