#!/bin/bash
#SBATCH --job-name=islet_analysis
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/path/to/xldvp_seg/logs/islet_analysis_%j.out
#SBATCH --error=/path/to/xldvp_seg/logs/islet_analysis_%j.err

# =============================================================================
# Islet Analysis — BS-100 (84,610 cells)
# Marker classification + spatial islet definition + HTML
# No GPU needed
# =============================================================================

set -e

RUN_DIR="/path/to/output/islet_output/2025_09_03_30610012_BS-100_20260224_084344_100pct"
CZI="/path/to/output/marvin_test/2025_09_03_30610012_BS-100.czi"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Project: Islet Analysis — BS-100"
echo "Run dir: $RUN_DIR"
echo "============================================================"

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
REPO="${REPO:-/path/to/xldvp_seg}"
cd "$REPO"

echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1

python scripts/analyze_islets.py \
    --run-dir "$RUN_DIR" \
    --czi-path "$CZI" \
    --display-channels 2,3,5 \
    --marker-channels gcg:2,ins:3,sst:5 \
    --marker-percentile 95 \
    --buffer-um 10 \
    --min-cells 5 \
    --otsu-multiplier 2.0

echo "=== Islet analysis complete at $(date) ==="
