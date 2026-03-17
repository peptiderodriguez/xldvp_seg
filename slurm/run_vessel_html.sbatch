#!/bin/bash
#SBATCH --job-name=vessel_html
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/xldvp_seg/logs/vessel_html_%j.out
#SBATCH --error=/path/to/xldvp_seg/logs/vessel_html_%j.err

# =============================================================================
# Vessel Multiscale — HTML regeneration (no GPU needed)
#
# 4-channel CZI (234K x 104K): nuc488, CD31_555, SMA647, PM750
# Display channels 0,1,2 → R=nuc488, G=CD31_555, B=SMA647
# 16,568 vessels — contour-based cropping (no HDF5 masks for multiscale)
# Max 15,000 HTML samples
# =============================================================================

set -e

CZI="/path/to/data/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT="/path/to/output/vessel_output/multiscale/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch_20260223_093915_100pct"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Project: Vessel Multiscale"
echo "Mode:   HTML regeneration only (contour-based, no GPU)"
echo "============================================================"

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
REPO="${REPO:-/path/to/xldvp_seg}"
cd "$REPO"

echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

python scripts/regenerate_html.py \
    --output-dir "$OUTPUT" \
    --czi-path "$CZI" \
    --cell-type vessel \
    --display-channels 0,1,2 \
    --channels 0,1,2 \
    --max-samples 0 \
    --samples-per-page 300 \
    --sort-by outer_diameter_um \
    --sort-order desc

echo "=== HTML generation complete at $(date) ==="
