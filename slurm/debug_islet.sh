#!/bin/bash
#SBATCH --job-name=islet_debug
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=0:30:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_debug_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_debug_%j.err

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

echo "Debug islet detection"
echo "Start: $(date)"
echo "Node: $(hostname)"

$PYTHON $REPO/debug_islet_tile.py

echo "Done: $(date)"
