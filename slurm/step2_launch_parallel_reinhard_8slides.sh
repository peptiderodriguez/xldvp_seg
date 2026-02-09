#!/bin/bash
# STEP 2: Launch 4 parallel jobs with Reinhard normalization (8 slides total)
# Run this AFTER step1_compute_reinhard_params_8slides.sbatch completes

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Check if Reinhard params file exists
REINHARD_PARAMS="/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_8slides.json"
if [ ! -f "$REINHARD_PARAMS" ]; then
    echo "ERROR: Reinhard parameters file not found: $REINHARD_PARAMS"
    echo "Please run step1_compute_reinhard_params_8slides.sbatch first!"
    exit 1
fi

echo "Using Reinhard parameters from: $REINHARD_PARAMS"
echo ""
cat "$REINHARD_PARAMS"
echo ""

# Define slide pairs (8 slides = 4 batches × 2 slides)
# Balanced sampling: 2 from each condition (FGC, FHU, MGC, MHU)
SLIDE_PAIRS=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU2.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU2.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU4.czi"
)

OUTPUT_DIR="/viper/ptmp2/edrod/unified_8slides_reinhard"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "============================================================"
echo "Submitting 4 parallel jobs with REINHARD normalization"
echo "============================================================"
echo "Slides: FGC1,3 + FHU2,4 + MGC1,3 + MHU2,4"
echo "Output directory: $OUTPUT_DIR"
echo ""

for i in {0..3}; do
    SLIDES="${SLIDE_PAIRS[$i]}"
    JOB_NAME="reinhard_batch$((i+1))"

    sbatch --job-name="$JOB_NAME" \
           --account=mel_apu \
           --partition=apu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=48 \
           --mem=200G \
           --time=2:00:00 \
           --gres=gpu:2 \
           --output="logs/${JOB_NAME}_%j.out" \
           --error="logs/${JOB_NAME}_%j.err" \
           --wrap="
module load python-waterboa/2024.06
module load rocm/6.3
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2

python run_unified_FAST.py \
    --czi-paths $SLIDES \
    --output-dir $OUTPUT_DIR \
    --tile-size 3000 \
    --sample-fraction 0.10 \
    --multi-gpu \
    --num-gpus 2 \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-min-area-um 25 \
    --hspc-max-area-um 150 \
    --cleanup-masks \
    --normalization-method reinhard \
    --norm-params-file $REINHARD_PARAMS
"

    echo "  Submitted batch $((i+1)): $SLIDES"
done

echo ""
echo "============================================================"
echo "All 4 jobs submitted!"
echo "============================================================"
echo ""
echo "Each job will apply Reinhard normalization (Lab color space)"
echo "using global statistics from all 8 slides."
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs: tail -f logs/reinhard_batch*"
