#!/bin/bash
#SBATCH --job-name=compute_norm_params
#SBATCH --account=mel_apu
#SBATCH --partition=apu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/compute_norm_params_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/compute_norm_params_%j.err

echo "============================================================"
echo "STEP 1: Compute Global Normalization Parameters"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1

cd /viper/ptmp2/edrod/xldvp_seg_fresh

python compute_normalization_params.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Next step: Run step2_launch_parallel_normalized.sh"
