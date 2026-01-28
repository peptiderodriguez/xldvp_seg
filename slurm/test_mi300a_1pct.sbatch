#!/bin/bash
#SBATCH --job-name=mkseg_mi300a_test
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/test_mi300a_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/test_mi300a_%j.err

# =============================================================================
# Test run on AMD MI300A with ROCm 6.3 - 1% sampling
# Tests 2 slides with minimal sampling to verify pipeline works
# =============================================================================

set -e

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Mode: Multi-GPU Test (1% sampling)"
echo "============================================================"

# Load Python and ROCm modules
module load python-waterboa/2024.06
module load rocm/6.3

# Activate virtual environment
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate

# Verify GPU detection
echo ""
echo "GPU Detection:"
rocm-smi --showproductname || echo "rocm-smi not available"

# Verify PyTorch sees GPUs
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Environment variables
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2  # MI300A is gfx942

# Test with first 2 slides
TEST_SLIDES=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC2.czi"
)

OUTPUT_DIR="/viper/ptmp2/edrod/test_1pct_mi300a"
mkdir -p "$OUTPUT_DIR"

cd /viper/ptmp2/edrod/xldvp_seg_fresh

echo ""
echo "Processing ${#TEST_SLIDES[@]} test slides with 1% sampling"
echo "Output: $OUTPUT_DIR"
echo ""

# Run with 1% sampling for quick test
python run_unified_FAST.py \
    --czi-paths "${TEST_SLIDES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size 3000 \
    --sample-fraction 0.01 \
    --multi-gpu \
    --num-gpus 2 \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-max-area-um 500

echo ""
echo "============================================================"
echo "Test completed at $(date)"
echo "Check output at: $OUTPUT_DIR"
echo "============================================================"
