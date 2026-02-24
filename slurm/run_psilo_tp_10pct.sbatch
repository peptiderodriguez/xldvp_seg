#!/bin/bash
#SBATCH --job-name=wm_pm_tp10
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/wm_pm_tp10_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/wm_pm_tp10_%j.err

# =============================================================================
# Whole Mouse PM — tissue_pattern 10% test run
# 4-channel CZI (metadata reports 12 due to EDF layers, but C=4 in data):
#   ch0: AF488 (nuc488)    — nuclear
#   ch1: AF647 (PM647)     — plasma membrane (detection)
#   ch2: AF750 (Msln750)   — mesothelin
#   ch3: AF555 (Pdgfra546) — PDGFRa
# Cellpose segmentation on PM(ch1) + nuclear(ch0), features from all 4 channels
# 252K x 107K px, ~1728 tiles at 4000px, ~172 tiles at 10%
# =============================================================================

set -e

CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/MPIP_psilo/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1.czi"
OUTPUT="/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/psilo_output/tp_10pct"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Project: Whole Mouse PM"
echo "Mode:   tissue_pattern, 10% tiles, 4x L40S"
echo "============================================================"

source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

nvidia-smi -L
echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

python run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type tissue_pattern \
    --tp-detection-channels 1 \
    --tp-nuclear-channel 0 \
    --tp-display-channels 3,1,2 \
    --channel-names "nuc488,PM647,Msln750,Pdgfra546" \
    --tile-size 4000 \
    --sample-fraction 0.10 \
    --tile-overlap 0.10 \
    --load-to-ram \
    --no-normalize-features \
    --num-gpus 4 \
    --no-serve \
    --output-dir "$OUTPUT"

echo "=== 10% test complete at $(date) ==="
echo "Detections: $(python -c "import json; d=json.load(open('$OUTPUT/tissue_pattern_detections.json')); print(len(d) if isinstance(d,list) else len(d.get('detections',[])))" 2>/dev/null || echo 'check manually')"
