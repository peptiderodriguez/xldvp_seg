#!/bin/bash
#SBATCH --job-name=coronal_tp10
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/path/to/output/code_bin/xldvp_seg/logs/coronal_tp10_%j.out
#SBATCH --error=/path/to/output/code_bin/xldvp_seg/logs/coronal_tp10_%j.err

# =============================================================================
# Whole-mount coronal section — tissue_pattern 10% test run
# 2-channel: ch0=AF488 (nuclear), ch1=AF647 (PM/membrane)
# Cellpose segmentation on PM+nuclear, extract morph+SAM2+channel features
# =============================================================================

set -e

CZI="/path/to/data/20250905_xlDVP_PM647_conjugatedAbs_nuc488_test-EDFvar-stitch.czi"
OUTPUT="/path/to/output/coronal_output/tp_10pct"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Mode:   tissue_pattern, 10% tiles, 4x L40S"
echo "============================================================"

XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"
REPO="${REPO:-/path/to/xldvp_seg}"
cd "$REPO"

nvidia-smi -L
echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

python run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type tissue_pattern \
    --tp-detection-channels 1 \
    --tp-nuclear-channel 0 \
    --tp-display-channels 1,0 \
    --channel-names "nuc488,PM647" \
    --all-channels \
    --tile-size 3000 \
    --sample-fraction 0.10 \
    --tile-overlap 0.10 \
    --load-to-ram \
    --multi-gpu --num-gpus 4 \
    --no-serve \
    --output-dir "$OUTPUT"

echo "=== 10% test complete at $(date) ==="
echo "Detections: $(python -c "import json; d=json.load(open('$OUTPUT/tissue_pattern_detections.json')); print(len(d) if isinstance(d,list) else len(d.get('detections',[])))" 2>/dev/null || echo 'check manually')"
