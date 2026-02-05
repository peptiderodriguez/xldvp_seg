#!/bin/bash
#SBATCH --job-name=regen_html
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/regen_html_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/regen_html_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=p.hpcl93
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G

# Activate conda environment
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

# Change to repo directory
cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

echo "=== Regenerating HTML sorted by classifier score ==="
python regenerate_nmj_html_sorted.py \
    --output-dir /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/nmj_output/multigpu_full \
    --czi-path /fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi \
    --samples-per-page 300

echo "=== Done ==="
