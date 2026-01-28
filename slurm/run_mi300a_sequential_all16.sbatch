#!/bin/bash
#SBATCH --job-name=mkseg_all16_10pct
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/all16_sequential_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/all16_sequential_%j.err

# =============================================================================
# Sequential processing of all 16 slides on one node - 10% sampling
# Processes 8 batches of 2 slides each, keeping the node allocated throughout
# Total estimated time: 16-20 hours (based on test: ~2 hours per batch)
# =============================================================================

set -e

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Processing all 16 slides sequentially (8 batches of 2)"
echo "============================================================"

# Load Python and ROCm modules
module load python-waterboa/2024.06
module load rocm/6.3

# Activate virtual environment
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate

# Environment variables
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2

# Verify GPU detection once
echo ""
echo "GPU Detection:"
rocm-smi --showproductname || echo "rocm-smi not available"
echo ""

# Define all 16 slides
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

OUTPUT_DIR="/viper/ptmp2/edrod/unified_10pct_mi300a"
mkdir -p "$OUTPUT_DIR"

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Process in batches of 2 slides
TOTAL_BATCHES=8
FAILED_BATCHES=()

for BATCH in $(seq 0 7); do
    SLIDE_IDX_START=$((BATCH * 2))
    SLIDE_IDX_END=$((SLIDE_IDX_START + 1))
    SLIDE_1="${ALL_SLIDES[$SLIDE_IDX_START]}"
    SLIDE_2="${ALL_SLIDES[$SLIDE_IDX_END]}"

    echo ""
    echo "============================================================"
    echo "BATCH $((BATCH+1))/$TOTAL_BATCHES - Started at $(date)"
    echo "  Slide 1: $(basename $SLIDE_1)"
    echo "  Slide 2: $(basename $SLIDE_2)"
    echo "============================================================"

    # Run segmentation
    if python run_unified_FAST.py \
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
        --hspc-nuclear-only; then

        echo ""
        echo "✓ BATCH $((BATCH+1)) completed successfully at $(date)"

        # Quick summary of detections
        for SLIDE in "$SLIDE_1" "$SLIDE_2"; do
            SLIDE_NAME=$(basename "$SLIDE" .czi)
            if [ -d "$OUTPUT_DIR/$SLIDE_NAME" ]; then
                echo "  $SLIDE_NAME: Output directory created"
            fi
        done
    else
        EXIT_CODE=$?
        echo ""
        echo "✗ BATCH $((BATCH+1)) FAILED with exit code $EXIT_CODE at $(date)"
        FAILED_BATCHES+=($BATCH)

        # Continue with next batch instead of exiting
        # (remove this if you want to stop on first failure)
        echo "  Continuing with next batch..."
    fi

    echo ""
    echo "Progress: $((BATCH+1))/$TOTAL_BATCHES batches complete"
    echo "============================================================"
done

# Final summary
echo ""
echo "============================================================"
echo "ALL BATCHES COMPLETE"
echo "Job end time: $(date)"
echo "============================================================"

if [ ${#FAILED_BATCHES[@]} -eq 0 ]; then
    echo "✓ All 8 batches completed successfully"
    echo "✓ All 16 slides processed"
else
    echo "✗ ${#FAILED_BATCHES[@]} batch(es) failed: ${FAILED_BATCHES[@]}"
    echo "Failed slide pairs:"
    for BATCH in "${FAILED_BATCHES[@]}"; do
        IDX_START=$((BATCH * 2))
        IDX_END=$((IDX_START + 1))
        echo "  Batch $((BATCH+1)): $(basename ${ALL_SLIDES[$IDX_START]}), $(basename ${ALL_SLIDES[$IDX_END]})"
    done
    exit 1
fi

echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Total runtime: Check job stats above"
echo "============================================================"
