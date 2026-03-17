#!/bin/bash
#SBATCH --job-name=reinhard_params_8slides
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/path/to/output/logs/reinhard_params_8slides_%j.out
#SBATCH --error=/path/to/output/logs/reinhard_params_8slides_%j.err

echo "============================================================"
echo "STEP 1: Compute Reinhard Normalization Parameters (8 slides)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 200GB"
echo "Start time: $(date)"
echo ""
echo "Slides: FGC1,3 + FHU2,4 + MGC1,3 + MHU2,4"
echo "Method: Tissue-aware sampling (500k pixels per slide)"
echo ""

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
export PYTHONUNBUFFERED=1

cd /path/to/output

python compute_reinhard_params_8slides.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output: reinhard_params_8slides.json"
echo ""
echo "Next step: Review parameters, then run:"
echo "  bash slurm/step2_launch_parallel_reinhard_8slides.sh"
