#!/bin/bash
#SBATCH --job-name=compute_norm_params
#SBATCH --account=mel_apu
#SBATCH --partition=apu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/output/logs/compute_norm_params_%j.out
#SBATCH --error=/path/to/output/logs/compute_norm_params_%j.err

echo "============================================================"
echo "STEP 1: Compute Global Normalization Parameters"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"
export PYTHONUNBUFFERED=1

cd /path/to/output

python compute_normalization_params.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Next step: Run step2_launch_parallel_normalized.sh"
