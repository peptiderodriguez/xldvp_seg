#!/bin/bash
#SBATCH --job-name=islet_analyze
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_analyze_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_analyze_%j.err

# Islet analysis: reclassify markers + generate HTML overview
# No GPU needed — CPU only (classification + HTML rendering)

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python
CZI="${1:?Usage: sbatch run_analyze_islets.sh <czi_path> <run_dir>}"
RUN_DIR="${2:?Usage: sbatch run_analyze_islets.sh <czi_path> <run_dir>}"

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p slurm/logs

echo "=========================================="
echo "Islet Analysis (GMM classification)"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "CZI: $CZI"
echo "Run dir: $RUN_DIR"
echo "=========================================="

MARKER_TOP_PCT="${3:-5}"
MARKER_PCT_CHANNELS="${4:-sst}"
GMM_P_CUTOFF="${5:-0.75}"
RATIO_MIN="${6:-1.5}"

$PYTHON scripts/analyze_islets.py \
    --run-dir "$RUN_DIR" \
    --czi-path "$CZI" \
    --buffer-um 25 \
    --min-cells 5 \
    --display-channels 2,3,5 \
    --marker-channels gcg:2,ins:3,sst:5 \
    --marker-top-pct "$MARKER_TOP_PCT" \
    --marker-pct-channels "$MARKER_PCT_CHANNELS" \
    --gmm-p-cutoff "$GMM_P_CUTOFF" \
    --ratio-min "$RATIO_MIN"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
