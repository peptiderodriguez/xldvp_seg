#!/bin/bash
#SBATCH --job-name=msln_html
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/msln_html_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/msln_html_%j.err

# =============================================================================
# Whole Mouse PM — Msln+ HTML annotation viewer
#
# Generates HTML for the Msln+ clustered detections (gated ch2 >= p90, then
# shape-clustered). Uses the consolidated scripts/regenerate_html.py.
# No GPU needed — loads 3 display channels from CZI, reads per-tile masks
# from HDF5, generates crops, exports HTML sorted by area descending.
#
# Display channels: ch1 (PM647/red), ch2 (Msln750/green), ch0 (nuc488/blue)
# =============================================================================

set -e

CZI="/fs/pool/pool-mann-axioscan/lm_data/MPIP_Psilo/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1.czi"
OUTPUT="/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/msln_plus"
DETECTIONS="${OUTPUT}/detections_clustered.json"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Detections: $DETECTIONS"
echo "Project: Whole Mouse PM — Msln+ HTML"
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
    --detections "$DETECTIONS" \
    --cell-type tissue_pattern \
    --display-channels 1,2,0 \
    --channels 0,1,2 \
    --max-samples 15000 \
    --sort-by area \
    --sort-order desc \
    --samples-per-page 300

echo "=== Msln+ HTML generation complete at $(date) ==="
