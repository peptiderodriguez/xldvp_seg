#!/bin/bash
#SBATCH --job-name=sam2_extract
#SBATCH --partition=p.hpcl93
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=sam2_extract_%j.log

# Extract SAM2 embeddings for 13 original MK slides
# These slides were processed with a pipeline version that saved sam2_0..sam2_255
# as zeros — this job re-extracts real embeddings from CZI tiles.

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

REPO=/home/edrod/xldvp_seg
BASE_DIR=/viper/ptmp2/edrod/unified_2026-02-11_100pct_2gpu
CZI_DIR=/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/bonemarrow/2025_11_18
OUTPUT=/viper/ptmp2/edrod/sam2_embeddings_original13.json

cd $REPO

echo "Starting SAM2 embedding extraction at $(date)"
echo "Base dir: $BASE_DIR"
echo "CZI dir: $CZI_DIR"
echo "Output: $OUTPUT"

python scripts/extract_sam2_embeddings.py extract \
    --base-dir "$BASE_DIR" \
    --czi-dir "$CZI_DIR" \
    --output "$OUTPUT"

echo "Extraction complete at $(date)"
echo ""

# After extraction completes, merge into the full detection JSON:
# python scripts/extract_sam2_embeddings.py merge \
#     --target /path/to/all_mks_with_rejected3_full.json \
#     --embeddings $OUTPUT
