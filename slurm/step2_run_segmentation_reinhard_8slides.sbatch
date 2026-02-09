#!/bin/bash
#SBATCH --job-name=seg_reinhard_8slides
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/seg_reinhard_8slides_%A_%a.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/seg_reinhard_8slides_%A_%a.err

echo "============================================================"
echo "STEP 2: Segmentation with Reinhard Normalization (8 slides)"
echo "============================================================"
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: 2"
echo "Memory: 200GB"
echo "Start time: $(date)"
echo ""

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Define all 16 slides (pairs of 2 per job)
ALL_SLIDES=(
    "2025_11_18_FGC1.czi"
    "2025_11_18_FGC2.czi"
    "2025_11_18_FGC3.czi"
    "2025_11_18_FGC4.czi"
    "2025_11_18_FHU1.czi"
    "2025_11_18_FHU2.czi"
    "2025_11_18_FHU3.czi"
    "2025_11_18_FHU4.czi"
    "2025_11_18_MGC1.czi"
    "2025_11_18_MGC2.czi"
    "2025_11_18_MGC3.czi"
    "2025_11_18_MGC4.czi"
    "2025_11_18_MHU1.czi"
    "2025_11_18_MHU2.czi"
    "2025_11_18_MHU3.czi"
    "2025_11_18_MHU4.czi"
)

# Get 2 slides for this task
SLIDE1_IDX=$((SLURM_ARRAY_TASK_ID * 2))
SLIDE2_IDX=$((SLURM_ARRAY_TASK_ID * 2 + 1))

SLIDE1="${ALL_SLIDES[$SLIDE1_IDX]}"
SLIDE2="${ALL_SLIDES[$SLIDE2_IDX]}"

SLIDE1_PATH="/viper/ptmp2/edrod/czi/${SLIDE1}"
SLIDE2_PATH="/viper/ptmp2/edrod/czi/${SLIDE2}"

echo "Processing pair ${SLURM_ARRAY_TASK_ID}: ${SLIDE1} + ${SLIDE2}"
echo ""

# Segmentation parameters
OUTPUT_BASE="/viper/ptmp2/edrod/xldvp_seg_fresh/output_reinhard_8slides"
NORM_PARAMS="/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_8slides_tissue.json"
MK_CLASSIFIER="/viper/ptmp2/edrod/xldvp_seg_fresh/segmentation/classifiers/rf_mk_classifier.joblib"
HSPC_CLASSIFIER="/viper/ptmp2/edrod/xldvp_seg_fresh/segmentation/classifiers/rf_hspc_classifier.joblib"

# Tile parameters (15% overlap)
TILE_SIZE=4096
OVERLAP=615  # 15% of 4096

echo "Parameters:"
echo "  Tile size: ${TILE_SIZE}px"
echo "  Overlap: ${OVERLAP}px (15%)"
echo "  Normalization: Reinhard (tile-by-tile)"
echo "  Norm params: ${NORM_PARAMS}"
echo ""

python run_unified_FAST.py \
    "${SLIDE1_PATH}" "${SLIDE2_PATH}" \
    --output-base "${OUTPUT_BASE}" \
    --tile-size ${TILE_SIZE} \
    --overlap ${OVERLAP} \
    --normalize-slides \
    --normalization-method reinhard \
    --norm-params-file "${NORM_PARAMS}" \
    --mk-classifier "${MK_CLASSIFIER}" \
    --hspc-classifier "${HSPC_CLASSIFIER}" \
    --num-gpus 2 \
    --multi-gpu

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Complete!"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "============================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Pair ${SLURM_ARRAY_TASK_ID} completed"
else
    echo "FAILED: Pair ${SLURM_ARRAY_TASK_ID} exited with code ${EXIT_CODE}"
fi

exit $EXIT_CODE
