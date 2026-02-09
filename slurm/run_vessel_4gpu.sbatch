#!/bin/bash
#SBATCH --job-name=vessel_sam2
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=450G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/%x_%j.err

# =============================================================================
# SAM2 Multi-Scale Vessel Detection — 4x L40S (48GB each)
#
# Input:  259 GB CZI (4-channel: nuc488, CD31_555, SMA647, PM750)
# Output: /fs/pool/pool-mann-edwin/vessel_output/sam2_multiscale/
# Scales: 1/64 → 1/32 → 1/16 → 1/8 → 1/4 → 1/2
#
# Resume after interruption:
#   sbatch run_vessel_4gpu.sbatch --resume "1/16"
# =============================================================================

set -e

# Parse optional --resume argument
RESUME_ARG=""
if [ "$1" == "--resume" ] && [ -n "$2" ]; then
    RESUME_ARG="--resume-from $2"
    echo "Resuming from scale $2"
fi

echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"
echo "Mode:       Multi-GPU Shared Memory (4x L40S)"
echo "Pipeline:   SAM2 Multi-Scale Vessel Detection"
echo "============================================================"

# Activate conda
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

# Working directory
cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Show GPUs
nvidia-smi -L

# Environment
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

# Create output directory
mkdir -p /fs/pool/pool-mann-edwin/vessel_output/sam2_multiscale

# Run vessel detection
python scripts/sam2_multiscale_vessels.py \
    --num-gpus 4 \
    $RESUME_ARG \
    2>&1 | tee /fs/pool/pool-mann-edwin/vessel_output/sam2_multiscale/run.log

echo "============================================================"
echo "Job complete at $(date)"
echo "============================================================"
