#!/bin/bash
#SBATCH --job-name=validate_median_reinhard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/validate_median_reinhard_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/validate_median_reinhard_%j.err

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Activate conda environment
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate

# Run validation
cd /viper/u2/edrod
python validate_median_reinhard.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output directory: /viper/ptmp2/edrod/xldvp_seg_fresh/validation_median_vs_mean"
echo ""
echo "To view results:"
echo "  ls -lh /viper/ptmp2/edrod/xldvp_seg_fresh/validation_median_vs_mean/"
