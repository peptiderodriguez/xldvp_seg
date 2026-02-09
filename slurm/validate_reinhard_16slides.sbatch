#!/bin/bash
#SBATCH --job-name=validate_reinhard_16
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/validate_reinhard_16_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/validate_reinhard_16_%j.err

echo "============================================================"
echo "Validating Reinhard Normalization (All 16 Slides)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Run validation on ALL 16 slides
python validate_reinhard_normalization.py \
    --params reinhard_params_16slides_MEDIAN_NEW.json \
    --slides \
        2025_11_18_FGC1.czi \
        2025_11_18_FGC2.czi \
        2025_11_18_FGC3.czi \
        2025_11_18_FGC4.czi \
        2025_11_18_FHU1.czi \
        2025_11_18_FHU2.czi \
        2025_11_18_FHU3.czi \
        2025_11_18_FHU4.czi \
        2025_11_18_MGC1.czi \
        2025_11_18_MGC2.czi \
        2025_11_18_MGC3.czi \
        2025_11_18_MGC4.czi \
        2025_11_18_MHU1.czi \
        2025_11_18_MHU2.czi \
        2025_11_18_MHU3.czi \
        2025_11_18_MHU4.czi \
    --slide-dir /viper/ptmp2/edrod/2025_11_18 \
    --output-dir validation_reinhard_16slides \
    --n-tiles 6 \
    --tile-size 3000

echo ""
echo "============================================================"
echo "Complete! $(date)"
echo "============================================================"
echo ""
echo "Check visualizations: validation_reinhard_16slides/*.png"
echo ""
echo "This will generate 16 PNG files (one per slide)"
