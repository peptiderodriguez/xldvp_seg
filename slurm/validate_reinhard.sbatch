#!/bin/bash
#SBATCH --job-name=validate_reinhard
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH --output=/path/to/output/logs/validate_reinhard_%j.out
#SBATCH --error=/path/to/output/logs/validate_reinhard_%j.err

echo "============================================================"
echo "Validating Reinhard Normalization"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
export PYTHONUNBUFFERED=1

cd /path/to/output

# Run validation on 2 test slides (NOT in the 8 training slides)
python validate_reinhard_normalization.py \
    --params reinhard_params_8slides.json \
    --slides 2025_11_18_FGC2.czi 2025_11_18_MHU1.czi \
    --slide-dir /path/to/data/czi_slides \
    --output-dir validation_reinhard \
    --n-tiles 6 \
    --tile-size 3000

echo ""
echo "============================================================"
echo "Complete! $(date)"
echo "============================================================"
echo ""
echo "Check visualizations: validation_reinhard/*.png"
