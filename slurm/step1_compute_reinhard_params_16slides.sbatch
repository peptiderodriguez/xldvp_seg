#!/bin/bash
#SBATCH --job-name=reinhard_params_16slides
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/reinhard_params_16slides_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/reinhard_params_16slides_%j.err

echo "============================================================"
echo "STEP 1: Compute Reinhard Normalization Parameters (16 slides)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 200GB"
echo "Start time: $(date)"
echo ""
echo "Slides: ALL 16 slides (complete cohort)"
echo "Method: Tissue-aware sampling (500k pixels per slide)"
echo ""

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1

cd /viper/ptmp2/edrod/xldvp_seg_fresh

python compute_reinhard_params_8slides.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output: reinhard_params_16slides.json"
echo ""
echo "Next step: Run validation on all 16 slides"
