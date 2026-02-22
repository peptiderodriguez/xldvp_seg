#!/bin/bash
#SBATCH --job-name=psilo_full
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/psilo_full_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/psilo_full_%j.err

# =============================================================================
# MPIP psilocybin coronal — tissue_pattern FULL 100% run
# 4-channel CZI (252K x 107K px, 1728 tiles at 4000px):
#   ch0: AF488 (nuc488)    — nuclear
#   ch1: AF647 (PM647)     — plasma membrane (detection)
#   ch2: AF750 (Msln750)   — mesothelin
#   ch3: AF555 (Pdgfra546) — PDGFRa
# Memory budget: 202 GB shm + 5 GB detections + 30 GB overhead = ~237 GB steady
# HTML capped at 15000 samples (50 pages x 300) to avoid base64 accumulation
# =============================================================================

set -e

CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/MPIP_psilo/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1.czi"
OUTPUT="/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/psilo_output/tp_full"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Mode:   tissue_pattern, 100% tiles, 4x L40S"
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
    --sample-fraction 1.0 \
    --tile-overlap 0.10 \
    --load-to-ram \
    --no-normalize-features \
    --max-html-samples 15000 \
    --num-gpus 4 \
    --no-serve \
    --output-dir "$OUTPUT"

echo "=== Full run complete at $(date) ==="
echo "Detections: $(python -c "import json; d=json.load(open('$OUTPUT/tissue_pattern_detections.json')); print(len(d) if isinstance(d,list) else len(d.get('detections',[])))" 2>/dev/null || echo 'check manually')"
