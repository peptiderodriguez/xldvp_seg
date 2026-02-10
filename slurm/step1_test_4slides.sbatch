#!/bin/bash
#SBATCH --job-name=reinhard_test4
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=1:30:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/reinhard_test4_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/reinhard_test4_%j.err

echo "============================================================"
echo "QUICK TEST: Reinhard params on 4 slides (Otsu-only pipeline)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd /viper/ptmp2/edrod/xldvp_seg_fresh

python compute_normalization_params.py \
    --n-slides 4 \
    --output reinhard_params_TEST_4slides.json

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output: reinhard_params_TEST_4slides.json"
echo "Verification tiles: verification_tiles/"
