#!/bin/bash
#SBATCH --job-name=test_multigpu_nmj
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/test_multigpu_nmj_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/test_multigpu_nmj_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=p.hpcl93
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G

# Activate conda environment
source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

# Change to repo directory
cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

# Show GPU info
echo "=== GPU Info ==="
nvidia-smi -L

# Show CPU/memory info
echo "=== CPU/Memory Info ==="
echo "CPUs available: $(nproc)"
free -h

# Run multi-GPU NMJ full run with 10% tile overlap
echo "=== Starting multi-GPU NMJ full run (4 GPUs, 100% tiles, 10% overlap) ==="
python run_segmentation.py \
    --czi-path /fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/xDVP/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi \
    --cell-type nmj \
    --channel 1 \
    --all-channels \
    --sample-fraction 1.0 \
    --tile-overlap 0.1 \
    --multi-gpu --num-gpus 4 \
    --no-serve \
    --output-dir /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/nmj_output/multigpu_full

echo "=== Test complete ==="
