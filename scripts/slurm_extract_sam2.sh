#!/bin/bash
#SBATCH --job-name=sam2_extract
#SBATCH --partition=p.hpcl93
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=sam2_extract_%j.log

# Extract SAM2 embeddings for ALL 16 MK slides.
# SAM2 features are all zeros for both original (13) AND rescued (3) slides.
# Two extraction passes: one per base directory.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

REPO=/home/edrod/xldvp_seg
CZI_DIR=/path/to/data/bonemarrow/2025_11_18
OUTDIR=/path/to/output

# Base dirs for the two slide groups
BASE_ORIGINAL=/path/to/output/unified_2026-02-11_100pct_2gpu
BASE_RESCUED=/path/to/output/bm_lmd_feb2026/mk_clf084_dataset

cd $REPO

# --- Pass 1: 13 original slides ---
echo "=== Pass 1: 13 original slides ==="
echo "Starting at $(date)"
python scripts/extract_sam2_embeddings.py extract \
    --base-dir "$BASE_ORIGINAL" \
    --czi-dir "$CZI_DIR" \
    --output "$OUTDIR/sam2_embeddings_original13.json"
echo "Pass 1 complete at $(date)"
echo ""

# --- Pass 2: 3 rescued slides (FGC2, FGC4, MHU4) ---
echo "=== Pass 2: 3 rescued slides ==="
echo "Starting at $(date)"
python scripts/extract_sam2_embeddings.py extract \
    --base-dir "$BASE_RESCUED" \
    --czi-dir "$CZI_DIR" \
    --slides 2025_11_18_FGC2 2025_11_18_FGC4 2025_11_18_MHU4 \
    --output "$OUTDIR/sam2_embeddings_rescued3.json"
echo "Pass 2 complete at $(date)"
echo ""

echo "Both passes complete. Output files:"
echo "  $OUTDIR/sam2_embeddings_original13.json"
echo "  $OUTDIR/sam2_embeddings_rescued3.json"
echo ""
echo "Next step — retrain classifier (can run on CPU):"
echo "  python scripts/retrain_mk_classifier.py train \\"
echo "      --original-training /path/to/mk_training_data_2026-02-11.json \\"
echo "      --rescued-base-dir $BASE_RESCUED \\"
echo "      --rescued-annotations /path/to/mk_annotations_2026-03-06_rejected3_unnorm_100pct.json \\"
echo "      --sam2-embeddings $OUTDIR/sam2_embeddings_original13.json $OUTDIR/sam2_embeddings_rescued3.json \\"
echo "      --output-dir $OUTDIR/retrained_classifier/"
