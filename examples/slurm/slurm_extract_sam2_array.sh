#!/bin/bash
#SBATCH --job-name=sam2_mk
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=sam2_mk_%A_%a.log

# SAM2 embedding extraction for 16 MK slides.
# 4 full nodes (all GPUs, all RAM, all CPUs), each processes 4 slides
# across 4 L40S GPUs in parallel. Total: 16 slides simultaneously.
#
# Expected runtime: ~5-10 minutes per node (100-400 cells per slide,
# ~10-50 virtual tiles per slide, SAM2 encoder is fast on L40S).

XLDVP_PYTHON="${XLDVP_PYTHON:-${MKSEG_PYTHON:-python}}"
REPO="${REPO:-/path/to/xldvp_seg}"

DETECTIONS=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset/all_mks_with_rejected3_full.json
TRAINING=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset/mk_clf_export_2026-02-11/mk_training_data_2026-02-11.json
CZI_DIR=/path/to/data/bonemarrow/2025_11_18
OUTPUT_DIR=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset/sam2_embeddings

echo "=========================================="
echo "SAM2 MK Extraction — Node ${SLURM_ARRAY_TASK_ID}/4"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "CPUs: $(nproc), RAM: $(free -g | awk '/Mem:/{print $2}')G"
echo "Started: $(date)"
echo "=========================================="

cd "$REPO"

PYTHONPATH="$REPO" "$XLDVP_PYTHON" examples/bone_marrow/extract_sam2_for_mk.py extract \
    --detections "$DETECTIONS" \
    --training-data "$TRAINING" \
    --czi-dir "$CZI_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --node-index "${SLURM_ARRAY_TASK_ID}" \
    --num-nodes 4

echo ""
echo "Node ${SLURM_ARRAY_TASK_ID} completed: $(date)"
ls -la "${OUTPUT_DIR}"/sam2_*.json 2>/dev/null
