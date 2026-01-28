#!/bin/bash
#SBATCH --job-name=mkseg_all16_resume
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/all16_resume_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/all16_resume_%j.err

# =============================================================================
# Sequential processing with checkpoint/resume - 10% sampling
# Skips already-processed slides, allowing you to resubmit if job times out
# =============================================================================

set -e

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Mode: Resume-capable sequential processing"
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

# Output directory
OUTPUT_DIR="/viper/ptmp2/edrod/unified_10pct_mi300a"
mkdir -p "$OUTPUT_DIR"

# Checkpoint file to track progress
CHECKPOINT_FILE="$OUTPUT_DIR/.checkpoint_batches_completed.txt"
touch "$CHECKPOINT_FILE"

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

# Read completed batches
COMPLETED_BATCHES=($(cat "$CHECKPOINT_FILE" 2>/dev/null || echo ""))

# Function to check if batch is complete
is_batch_complete() {
    local BATCH=$1
    for completed in "${COMPLETED_BATCHES[@]}"; do
        if [ "$completed" == "$BATCH" ]; then
            return 0
        fi
    done
    return 1
}

cd /viper/ptmp2/edrod/xldvp_seg_fresh

echo ""
echo "Checkpoint status:"
echo "  Completed batches: ${COMPLETED_BATCHES[@]:-none}"
echo ""

# Verify GPU detection once
echo "GPU Detection:"
rocm-smi --showproductname || echo "rocm-smi not available"
echo ""

# Process in batches of 2 slides
TOTAL_BATCHES=8
SKIPPED=0
PROCESSED=0
FAILED_BATCHES=()

for BATCH in $(seq 0 7); do
    # Check if already complete
    if is_batch_complete $BATCH; then
        echo "✓ BATCH $((BATCH+1))/$TOTAL_BATCHES - SKIPPED (already completed)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

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

        # Mark as complete in checkpoint file
        echo "$BATCH" >> "$CHECKPOINT_FILE"
        PROCESSED=$((PROCESSED+1))

        # Quick summary
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

        # Stop on first failure (can change to continue if preferred)
        echo "  Stopping due to failure. Resubmit this job to resume."
        break
    fi

    echo ""
    TOTAL_DONE=$((SKIPPED+PROCESSED))
    echo "Progress: $TOTAL_DONE/$TOTAL_BATCHES batches complete ($SKIPPED skipped, $PROCESSED newly processed)"
    echo "============================================================"
done

# Final summary
echo ""
echo "============================================================"
echo "JOB SUMMARY"
echo "Job end time: $(date)"
echo "============================================================"
echo "Batches skipped (already done): $SKIPPED"
echo "Batches processed this run: $PROCESSED"
echo "Batches failed: ${#FAILED_BATCHES[@]}"

if [ ${#FAILED_BATCHES[@]} -eq 0 ] && [ $((SKIPPED+PROCESSED)) -eq 8 ]; then
    echo ""
    echo "✓ ALL 16 SLIDES COMPLETE!"
    echo "  Output: $OUTPUT_DIR"
elif [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
    echo ""
    echo "✗ Job stopped due to failure"
    echo "  Failed batch: ${FAILED_BATCHES[@]}"
    echo "  To resume: sbatch slurm/run_mi300a_sequential_resume.sbatch"
    exit 1
else
    echo ""
    echo "Job completed partial work"
    echo "  To continue: sbatch slurm/run_mi300a_sequential_resume.sbatch"
fi

echo "============================================================"
