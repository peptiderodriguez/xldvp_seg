#!/bin/bash
#SBATCH --job-name=regen_html
#SBATCH --partition=p.hpcl93
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/regen_html_%j.out
#SBATCH --error=logs/regen_html_%j.err

# Activate conda environment
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Regenerate HTML for all 16 slides
OUTPUT_BASE="/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/mk_output/2026_01_21_BM_16slides_2gpu"
CZI_BASE="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/bonemarrow/2025_11_18"

for slide in FGC1 FGC2 FGC3 FGC4 FHU1 FHU2 FHU3 FHU4 MGC1 MGC2 MGC3 MGC4 MHU1 MHU2 MHU3 MHU4; do
    echo "Processing 2025_11_18_${slide} MK..."
    python regenerate_html.py \
        --output-dir "${OUTPUT_BASE}/2025_11_18_${slide}" \
        --czi-path "${CZI_BASE}/2025_11_18_${slide}.czi" \
        --channel 0 \
        --sort-by area \
        --sort-order asc \
        --samples-per-page 300 \
        --cell-type mk \
        --experiment-name "2026_01_21_BM_mk"

    echo "Processing 2025_11_18_${slide} HSPC..."
    python regenerate_html.py \
        --output-dir "${OUTPUT_BASE}/2025_11_18_${slide}" \
        --czi-path "${CZI_BASE}/2025_11_18_${slide}.czi" \
        --channel 0 \
        --sort-by area \
        --sort-order asc \
        --samples-per-page 300 \
        --cell-type hspc \
        --experiment-name "2026_01_21_BM_hspc"
done

echo "All slides processed"
