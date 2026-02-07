#!/bin/bash
#SBATCH --job-name=regen_html
#SBATCH --partition=p.hpcl93
#SBATCH --time=04:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=32
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/regen_html_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/regen_html_%j.err

source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Use the batch regenerator for 3-GPU output
python regenerate_batch_html.py \
    --output-base "/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/mk_output/2026_01_22_BM_16slides_3gpu" \
    --czi-base "/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/bonemarrow/2025_11_18" \
    --channel 0 \
    --samples-per-page 300

echo "Done at $(date)"
