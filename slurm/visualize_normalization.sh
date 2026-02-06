#!/bin/bash
#SBATCH --job-name=viz_norm_16slides
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/visualize_normalization_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/visualize_normalization_%j.err

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Activate environment
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate

# Run visualization (working dir must contain segmentation/ package)
cd /viper/ptmp2/edrod/xldvp_seg_fresh
python visualize_normalization.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output directory: /viper/ptmp2/edrod/xldvp_seg_fresh/verification_tiles/"
echo ""
echo "To view results:"
echo "  ls -lh /viper/ptmp2/edrod/xldvp_seg_fresh/verification_tiles/"
