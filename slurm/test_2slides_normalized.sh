#!/bin/bash
# TEST: Process 2 slides with new slide-level normalization
# Tests the corrected normalization implementation before running all 16 slides

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Check if normalization params file exists
NORM_PARAMS="/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json"
if [ ! -f "$NORM_PARAMS" ]; then
    echo "ERROR: Normalization parameters file not found: $NORM_PARAMS"
    exit 1
fi

echo "============================================================"
echo "TEST: Processing 2 slides with slide-level normalization"
echo "============================================================"
echo "Using normalization parameters: $NORM_PARAMS"
echo "Slides: FGC1 and FGC2"
echo ""

# Test with just 2 slides
SLIDES="/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC2.czi"
OUTPUT_DIR="/viper/ptmp2/edrod/unified_10pct_mi300a_normalized_TEST"
mkdir -p "$OUTPUT_DIR"

JOB_ID=$(sbatch --job-name="test_norm_2slides" \
       --partition=apu \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task=48 \
       --mem=240G \
       --time=2:00:00 \
       --gres=gpu:2 \
       --output="logs/test_norm_2slides_%j.out" \
       --error="logs/test_norm_2slides_%j.err" \
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
    --normalize-slides \
    --normalization-method reinhard \
    --norm-params-file $NORM_PARAMS
" | tail -1 | awk '{print $4}')

echo "Job submitted: $JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs: tail -f logs/test_norm_2slides_${JOB_ID}.err"
