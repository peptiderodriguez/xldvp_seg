#!/bin/bash
# BASELINE: Launch 4 parallel jobs WITHOUT normalization (8 slides total)
# This provides baseline comparison for normalization validation

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Define slide pairs (same 8 slides as Reinhard test)
SLIDE_PAIRS=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU2.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU2.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU4.czi"
)

OUTPUT_DIR="/viper/ptmp2/edrod/unified_8slides_unnormalized"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "============================================================"
echo "Submitting 4 parallel jobs WITHOUT normalization (BASELINE)"
echo "============================================================"
echo "Slides: FGC1,3 + FHU2,4 + MGC1,3 + MHU2,4"
echo "Output directory: $OUTPUT_DIR"
echo ""

for i in {0..3}; do
    SLIDES="${SLIDE_PAIRS[$i]}"
    JOB_NAME="unnorm_batch$((i+1))"

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
    --cleanup-masks
"

    echo "  Submitted batch $((i+1)): $SLIDES"
done

echo ""
echo "============================================================"
echo "All 4 jobs submitted!"
echo "============================================================"
echo ""
echo "These jobs run WITHOUT normalization for baseline comparison."
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs: tail -f logs/unnorm_batch*"
