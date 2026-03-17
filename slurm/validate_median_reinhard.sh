#!/bin/bash
#SBATCH --job-name=validate_median_reinhard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/output/logs/validate_median_reinhard_%j.out
#SBATCH --error=/path/to/output/logs/validate_median_reinhard_%j.err

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Environment
MKSEG_PYTHON="${MKSEG_PYTHON:-python}"

cd /path/to/output
python validate_median_reinhard.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output directory: /path/to/output/validation_median_vs_mean"
echo ""
echo "To view results:"
echo "  ls -lh /path/to/output/validation_median_vs_mean/"
