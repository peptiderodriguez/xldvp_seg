#!/bin/bash
#SBATCH --job-name=reinhard_MEDIAN_16
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/output/logs/reinhard_params_16slides_MEDIAN_%j.out
#SBATCH --error=/path/to/output/logs/reinhard_params_16slides_MEDIAN_%j.err

echo "============================================================"
echo "Computing MEDIAN-based Reinhard Parameters (16 slides)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""
echo "Using MEDIAN and MAD for robust statistics"
echo "(Resistant to outliers - prevents darkening from background)"
echo ""

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd /path/to/output

python compute_normalization_params.py

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Output: reinhard_params_16slides_MEDIAN_NEW.json"
echo ""
echo "Next step: Run validation on all 16 slides"
