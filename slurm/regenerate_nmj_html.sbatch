#!/bin/bash
#SBATCH --job-name=regen_nmj_html
#SBATCH --partition=p.hpcl93
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/regen_nmj_html_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/regen_nmj_html_%j.err

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python
CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi"
OUTPUT_DIR=/fs/pool/pool-mann-edwin/nmj_output/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "=========================================="
echo "Regenerate NMJ HTML — sorted by area (ascending)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

$PYTHON "$REPO/regenerate_html.py" \
    --output-dir "$OUTPUT_DIR" \
    --czi-path "$CZI" \
    --channel 1 \
    --sort-by area \
    --sort-order asc \
    --samples-per-page 300

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
