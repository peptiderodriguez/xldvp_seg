#!/bin/bash
#SBATCH --job-name=wm_pm_html
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/wm_pm_html_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/wm_pm_html_%j.err

# =============================================================================
# Whole Mouse PM — regenerate HTML (366K detections, HTML never generated)
#
# Uses the consolidated scripts/regenerate_html.py — no GPU needed.
# Loads 3 display channels (~150 GB), samples 15K of 366K detections,
# loads per-tile masks from HDF5, generates crops, exports HTML.
#
# Display channels: 3 (Pdgfra/red), 1 (PM647/green), 2 (Msln750/blue)
# — matches the original full run sbatch (--tp-display-channels 3,1,2)
# =============================================================================

set -e

CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/MPIP_psilo/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1.czi"
OUTPUT="/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Project: Whole Mouse PM"
echo "Mode:   HTML regeneration only (no detection)"
echo "============================================================"

source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

python scripts/regenerate_html.py \
    --output-dir "$OUTPUT" \
    --czi-path "$CZI" \
    --cell-type tissue_pattern \
    --display-channels 3,1,2 \
    --channels 1,2,3 \
    --tile-size 4000 \
    --max-samples 15000 \
    --samples-per-page 300 \
    --sort-by area \
    --sort-order asc

echo "=== HTML generation complete at $(date) ==="
