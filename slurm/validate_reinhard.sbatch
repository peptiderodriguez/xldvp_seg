#!/bin/bash
#SBATCH --job-name=validate_reinhard
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/validate_reinhard_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/validate_reinhard_%j.err

echo "============================================================"
echo "Validating Reinhard Normalization"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Run validation on 2 test slides (NOT in the 8 training slides)
python validate_reinhard_normalization.py \
    --params reinhard_params_8slides.json \
    --slides 2025_11_18_FGC2.czi 2025_11_18_MHU1.czi \
    --slide-dir /viper/ptmp2/edrod/czi \
    --output-dir validation_reinhard \
    --n-tiles 6 \
    --tile-size 3000

echo ""
echo "============================================================"
echo "Complete! $(date)"
echo "============================================================"
echo ""
echo "Check visualizations: validation_reinhard/*.png"
