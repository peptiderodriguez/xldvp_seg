#!/bin/bash
#SBATCH --job-name=mkseg_mi300a_10pct
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --array=0-7%1
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/mi300a_10pct_%A_%a.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/mi300a_10pct_%A_%a.err

# =============================================================================
# Production array job for AMD MI300A with ROCm 6.3 - 10% sampling
# Processes 16 slides in 8 sequential jobs (2 slides per job)
# Array indices 0-7, with %1 limiting to 1 concurrent job (MaxJobs=1 constraint)
# =============================================================================

set -e

echo "============================================================"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "============================================================"

# Load Python and ROCm modules
module load python-waterboa/2024.06
module load rocm/6.3

# Activate virtual environment
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate

# Environment variables
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2  # MI300A gfx942

# Verify GPU detection
echo ""
echo "GPU Detection:"
rocm-smi --showproductname || echo "rocm-smi not available"

# Define slide pairs for each array index
# Array 0-7 corresponds to 8 pairs of slides
ALL_SLIDES=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU4.czi"
)

# Get 2 slides for this task (array index 0-7 → slides 0-1, 2-3, ..., 14-15)
SLIDE_IDX_START=$((SLURM_ARRAY_TASK_ID * 2))
SLIDE_IDX_END=$((SLIDE_IDX_START + 1))
SLIDE_1="${ALL_SLIDES[$SLIDE_IDX_START]}"
SLIDE_2="${ALL_SLIDES[$SLIDE_IDX_END]}"

echo ""
echo "Task $SLURM_ARRAY_TASK_ID processing slides:"
echo "  1. $(basename $SLIDE_1)"
echo "  2. $(basename $SLIDE_2)"
echo ""

# Output directory
OUTPUT_DIR="/viper/ptmp2/edrod/unified_10pct_mi300a"
mkdir -p "$OUTPUT_DIR"

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Run segmentation with 10% sampling
python run_unified_FAST.py \
    --czi-paths "$SLIDE_1" "$SLIDE_2" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size 3000 \
    --sample-fraction 0.10 \
    --multi-gpu \
    --num-gpus 2 \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-max-area-um 500 \
    --cleanup-masks \
    --hspc-nuclear-only

echo ""
echo "============================================================"
echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "Slides processed: $(basename $SLIDE_1), $(basename $SLIDE_2)"
echo "============================================================"
