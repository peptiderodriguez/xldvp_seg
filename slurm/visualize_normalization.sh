#!/bin/bash
#SBATCH --job-name=viz_norm_16slides
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/output/logs/visualize_normalization_%j.out
#SBATCH --error=/path/to/output/logs/visualize_normalization_%j.err

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Environment
XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"

cd /path/to/output
python visualize_normalization.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output directory: /path/to/output/verification_tiles/"
echo ""
echo "To view results:"
echo "  ls -lh /path/to/output/verification_tiles/"
