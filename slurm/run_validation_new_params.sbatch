#!/bin/bash
#SBATCH --job-name=validate_norm
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=logs/validate_norm_%j.out
#SBATCH --error=logs/validate_norm_%j.err

# Load modules
MKSEG_PYTHON="${MKSEG_PYTHON:-python}"

# Run validation script
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2

cd /path/to/output

$MKSEG_PYTHON validate_median_reinhard.py

echo ""
echo "Validation complete! Images saved to:"
echo "/path/to/output/validation_median_vs_mean/"
