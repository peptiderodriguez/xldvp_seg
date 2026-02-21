#!/bin/bash
#SBATCH --job-name=tp_100
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=6:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/tp_test_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/tp_test_%j.err

# Tissue pattern 1% smoke test — brain FISH (5ch)
# Channels: 0=Slc17a7, 1=Htr2a, 2=Ntrk2, 3=Gad1, 4=Hoechst
# Detection: sum ch0+ch3 (excitatory+inhibitory neurons)
# Display: R=Slc17a7(0), G=Gad1(3), B=Htr2a(1)

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
CZI="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/MPIP_psilo/19022026_gold_fish_ctrl_veh_488Slc17a7_555Gad1_647Htr2a_750Ntrk2_Hoechst-EDFvar-stitch.czi"
OUTPUT_DIR=/fs/pool/pool-mann-edwin/brain_tissue_pattern
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python

export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

cd "$REPO"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/logs

echo "=== Tissue Pattern 100% run ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 2"
echo "CZI: $CZI"

$PYTHON run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type tissue_pattern \
    --channel 0 \
    --tp-detection-channels 0,3 \
    --tp-nuclear-channel 2 \
    --tp-display-channels 0,3,1 \
    --tp-min-area 20 \
    --tp-max-area 300 \
    --all-channels \
    --tile-size 3000 \
    --tile-overlap 0.10 \
    --sample-fraction 1.0 \
    --load-to-ram \
    --multi-gpu --num-gpus 2 \
    --output-dir "$OUTPUT_DIR" \
    --no-serve

echo "Done: $(date)"
