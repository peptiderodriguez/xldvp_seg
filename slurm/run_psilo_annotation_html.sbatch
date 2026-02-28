#!/bin/bash
#SBATCH --job-name=whole_mouse_ann_html
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/psilo_ann_html_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/psilo_ann_html_%j.err

# Generate annotation HTML (1500 random cells) from WM PM 100% detections
# "Detect once, classify later" step 1: subsample for annotation

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1.czi"
RUN=/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p slurm/logs

echo "=========================================="
echo "WM PM Annotation HTML Generation"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "RAM: $(free -g | awk '/Mem:/{print $2}')G"
echo "CZI: $CZI"
echo "Run: $RUN"
echo "=========================================="

$PYTHON scripts/regenerate_html.py \
    --output-dir "$RUN" \
    --czi-path "$CZI" \
    --cell-type tissue_pattern \
    --display-channels 1,2,0 \
    --max-samples 1500 \
    --html-dir "$RUN/html_annotation"

echo "=========================================="
echo "Done: $(date)"
echo "HTML at: $RUN/html_annotation"
echo "=========================================="
