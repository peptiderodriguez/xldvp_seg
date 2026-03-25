#!/bin/bash
#SBATCH --job-name=sam2_rescue
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=700G
#SBATCH --time=01:00:00
#SBATCH --output=sam2_rescue_%j.log

# Re-extract SAM2 for 3 rescued slides (FGC2, FGC4, MHU4).
# Single node — completed slides are auto-skipped via resume check.

XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"
REPO="${REPO:-/path/to/xldvp_seg}"

DETECTIONS=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset/all_mks_with_rejected3_full.json
TRAINING=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset/mk_clf_export_2026-02-11/mk_training_data_2026-02-11.json
CZI_DIR=/path/to/data/bonemarrow/2025_11_18
OUTPUT_DIR=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset/sam2_embeddings

echo "=========================================="
echo "SAM2 MK Extraction — Rescued slides fix"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "Started: $(date)"
echo "=========================================="

cd "$REPO"

# Run all 16 slides on 1 node — 13 completed will be skipped via resume check
PYTHONPATH="$REPO" "$XLDVP_PYTHON" scripts/extract_sam2_for_mk.py extract \
    --detections "$DETECTIONS" \
    --training-data "$TRAINING" \
    --czi-dir "$CZI_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Completed: $(date)"
ls -la "${OUTPUT_DIR}"/sam2_*.json 2>/dev/null
