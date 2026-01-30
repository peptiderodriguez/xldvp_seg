#!/bin/bash
# Launch 8 parallel jobs, each processing 2 slides

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Define slide pairs
SLIDE_PAIRS=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU4.czi"
)

OUTPUT_DIR="/viper/ptmp2/edrod/unified_10pct_mi300a"
mkdir -p "$OUTPUT_DIR"

echo "Submitting 8 parallel jobs..."

for i in {0..7}; do
    SLIDES="${SLIDE_PAIRS[$i]}"
    JOB_NAME="mkseg_batch$((i+1))"

    sbatch --job-name="$JOB_NAME" \
           --partition=apu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=48 \
           --mem=180G \
           --time=2:00:00 \
           --gres=gpu:2 \
           --output="logs/batch$((i+1))_%j.out" \
           --error="logs/batch$((i+1))_%j.err" \
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
echo "All 8 jobs submitted!"
echo "Monitor with: squeue -u \$USER"
