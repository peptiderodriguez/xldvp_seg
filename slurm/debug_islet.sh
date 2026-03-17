#!/bin/bash
#SBATCH --job-name=islet_debug
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/islet_debug_%j.out
#SBATCH --error=slurm/logs/islet_debug_%j.err

REPO="${REPO:-/path/to/xldvp_seg}"
MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
PYTHON="$MKSEG_PYTHON"

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

echo "Debug islet detection"
echo "Start: $(date)"
echo "Node: $(hostname)"

$PYTHON $REPO/debug_islet_tile.py

echo "Done: $(date)"
