#!/bin/bash
#SBATCH --job-name=mkseg_batch
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# MK/HSPC Batch Segmentation - Process Multiple CZI Files
# =============================================================================
#
# This script processes multiple CZI files in a single job.
# Models are loaded once and reused across all slides.
#
# Usage:
#   sbatch slurm/run_batch_segmentation.sbatch /path/to/files/*.czi /path/to/output
#
# Or with a file list:
#   sbatch slurm/run_batch_segmentation.sbatch --files file_list.txt --output /path/to/output
#
# =============================================================================

set -e

# --- Parse Arguments ---
CZI_PATHS=()
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --files)
            # Read file paths from a text file
            while IFS= read -r line; do
                [[ -n "$line" && ! "$line" =~ ^# ]] && CZI_PATHS+=("$line")
            done < "$2"
            shift 2
            ;;
        *)
            # Assume it's a CZI path
            CZI_PATHS+=("$1")
            shift
            ;;
    esac
done

if [ ${#CZI_PATHS[@]} -eq 0 ]; then
    echo "Error: No CZI files specified"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not specified (use --output)"
    exit 1
fi

# --- Configuration ---
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
echo "Files to process: ${#CZI_PATHS[@]}"
echo "============================================================"

# Load conda environment
if [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
    source ~/miniforge3/etc/profile.d/conda.sh
elif [ -f /etc/profile.d/conda.sh ]; then
    source /etc/profile.d/conda.sh
fi

conda activate mkseg

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# --- Run Batch Segmentation ---
cd /home/dude/code/xldvp_seg_repo

echo ""
echo "Processing ${#CZI_PATHS[@]} files..."
echo ""

# Convert array to space-separated string for --czi-paths
python run_unified_FAST.py \
    --czi-paths "${CZI_PATHS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$NUM_WORKERS" \
    --tile-size "$TILE_SIZE" \
    --sample-fraction "$SAMPLE_FRACTION"

echo ""
echo "============================================================"
echo "Batch job completed: ${#CZI_PATHS[@]} files processed"
echo "============================================================"
