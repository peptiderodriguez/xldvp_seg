#!/bin/bash
#SBATCH --job-name=islet_pm
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --gres=gpu:rtx5000:2
#SBATCH --time=4:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_pm_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/islet_pm_%j.err

# =============================================================================
# Islet segmentation — PM + NUCLEI (membrane ch1 + DAPI ch4, channels=[1,2])
# Slide: BS-100 (6-channel: Bright, AF633, Gcg, Ins, DAPI, Sst)
# Comparison run: nuclei-only vs PM+nuclei
# =============================================================================

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-edwin/marvin_test/2025_09_03_30610012_BS-100.czi"
OUTPUT_DIR=/fs/pool/pool-mann-edwin/islet_output/BS100_pm_nuclei
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Project: Islet — PM + Nuclei (membrane ch1 + DAPI ch4)"
echo "Output:  $OUTPUT_DIR"
echo "============================================================"

# --- Step 1: Segmentation (PM+nuclei Cellpose + SAM2 embeddings) ---
echo "=== Step 1: Segmentation (PM+nuclei) ==="
$PYTHON $REPO/run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type islet \
    --membrane-channel 1 \
    --nuclear-channel 4 \
    --all-channels \
    --sample-fraction 1.0 \
    --load-to-ram \
    --num-gpus 2 \
    --output-dir "$OUTPUT_DIR"

echo "=== Segmentation complete at $(date) ==="

# --- Step 2: Find the run directory (timestamped) ---
RUN_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_100pct" -type d | sort | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No run directory found in $OUTPUT_DIR"
    exit 1
fi
echo "Run dir: $RUN_DIR"

# --- Step 3: Islet analysis (marker classification + tissue-level islets) ---
echo "=== Step 2: Islet analysis ==="
$PYTHON $REPO/scripts/analyze_islets.py \
    --run-dir "$RUN_DIR" \
    --czi-path "$CZI" \
    --display-channels 2,3,5 \
    --marker-channels gcg:2,ins:3,sst:5 \
    --otsu-multiplier 2.0

echo "=== All done at $(date) ==="
