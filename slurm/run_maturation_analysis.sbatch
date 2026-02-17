#!/bin/bash
#SBATCH --job-name=mk_maturation_v2
#SBATCH --partition=apu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/slurm/logs/maturation_v2_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/slurm/logs/maturation_v2_%j.err

# MK Maturation Staging v2 — Nuclear Deep Features (inv. blue + Otsu)
# Runs: load → nuclear features (GPU) → cluster → validation/plots
#
# Usage: sbatch slurm/run_maturation_analysis.sbatch

set -euo pipefail

cd /viper/ptmp2/edrod/xldvp_seg_fresh
source mkseg_rocm_env/bin/activate

INPUT_DIR="/viper/ptmp2/edrod/unified_2026-02-11_100pct_2gpu"
OUTPUT_BASE="/viper/ptmp2/edrod/maturation_analysis_v2"
mkdir -p "${OUTPUT_BASE}"

DATA_FILE="${OUTPUT_BASE}/maturation_data.npz"
NUCLEAR_FILE="${OUTPUT_BASE}/nuclear_deep_features.npz"
CLUSTERS_FILE="${OUTPUT_BASE}/maturation_clusters.npz"
RESULTS_DIR="${OUTPUT_BASE}/results"

SAM2_CHECKPOINT="/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"
NUM_GPUS=2

echo "============================================"
echo "MK Maturation Staging v2 — Nuclear Deep Features"
echo "Start: $(date)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_BASE}"
echo "GPUs: ${NUM_GPUS}"
echo "============================================"

# Phase 1: Load, filter, dedup (CPU only)
echo ""
echo ">>> PHASE 1: Loading and preprocessing..."
python maturation_analysis.py phase1_load \
    --input-dir "${INPUT_DIR}" \
    --min-area-um 200 \
    --max-area-um 2000 \
    --min-clf-score 0.80 \
    --output "${DATA_FILE}"

# Phase 2: Nuclear deep feature extraction (GPU)
echo ""
echo ">>> PHASE 2: Nuclear features (inv. blue Otsu + SAM2 + ResNet on ${NUM_GPUS} GPUs)..."
python maturation_analysis.py phase2_nuclear \
    --input "${DATA_FILE}" \
    --num-gpus ${NUM_GPUS} \
    --sam2-checkpoint "${SAM2_CHECKPOINT}" \
    --sam2-config "${SAM2_CONFIG}" \
    --output "${NUCLEAR_FILE}"

# Phase 3: Clustering on nuclear features (CPU)
echo ""
echo ">>> PHASE 3: Clustering..."
python maturation_analysis.py phase3_cluster \
    --input "${NUCLEAR_FILE}" \
    --data "${DATA_FILE}" \
    --k-range 3,4,5,6,7,8 \
    --output "${CLUSTERS_FILE}"

# Phase 4: Validation, pseudotime, group comparison (CPU)
echo ""
echo ">>> PHASE 4: Validation and plotting..."
python maturation_analysis.py phase4_validate \
    --data "${DATA_FILE}" \
    --nuclear "${NUCLEAR_FILE}" \
    --clusters "${CLUSTERS_FILE}" \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "============================================"
echo "All phases complete: $(date)"
echo "Results in: ${RESULTS_DIR}"
echo "============================================"
