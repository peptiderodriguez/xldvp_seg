#!/bin/bash
#SBATCH --job-name=norm_validate
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=logs/norm_validate_%j.out
#SBATCH --error=logs/norm_validate_%j.err

XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2

cd /path/to/output

python -c "
import json
from pathlib import Path
from compute_normalization_params import generate_visual_validation

params_file = Path('reinhard_params_16slides_MEDIAN_NEW.json')
with open(params_file) as f:
    params = json.load(f)

czi_dir = Path('/path/to/data/czi_slides')
slides = sorted(czi_dir.glob('2025_11_18_*.czi'))
print(f'Found {len(slides)} slides: {[s.name for s in slides]}')

generate_visual_validation(slides, params, Path('verification_tiles'))
"
