#!/bin/bash
# Process all 16 slides on lab machine (24 CPU, 512GB RAM, RTX 3090)
# Optimized for local workstation with fast storage

# ========================================
# CONFIGURATION
# ========================================

# Input/Output paths - UPDATE THESE FOR YOUR MACHINE
CZI_BASE="/path/to/your/czi/files"  # Directory containing slide subdirectories
OUTPUT_BASE="$HOME/xldvp_seg_output"  # Output directory

# Processing parameters (optimized for RTX 3090 + 512GB RAM)
TILE_SIZE=4096           # Use maximum resolution
NUM_WORKERS=4            # 4-6 workers safe for RTX 3090 (24GB VRAM)
SAMPLE_FRACTION=0.10     # 10% for annotation, 1.0 for full processing
MK_MIN_AREA_UM=200       # Minimum MK area in µm²
MK_MAX_AREA_UM=2000      # Maximum MK area in µm²

# Optional: Use trained classifiers (leave empty if not using)
MK_CLASSIFIER=""         # Path to mk_classifier.pkl or empty
HSPC_CLASSIFIER=""       # Path to hspc_classifier.pkl or empty

# ========================================
# SLIDE LIST (all 16 slides)
# ========================================
SLIDES=(
    "2025_11_18_FGC1"
    "2025_11_18_FGC2"
    "2025_11_18_FGC3"
    "2025_11_18_FGC4"
    "2025_11_18_FHU1"
    "2025_11_18_FHU2"
    "2025_11_18_FHU3"
    "2025_11_18_FHU4"
    "2025_11_18_MGC1"
    "2025_11_18_MGC2"
    "2025_11_18_MGC3"
    "2025_11_18_MGC4"
    "2025_11_18_MHU1"
    "2025_11_18_MHU2"
    "2025_11_18_MHU3"
    "2025_11_18_MHU4"
)

# ========================================
# SETUP
# ========================================

# Ensure output directory exists
mkdir -p "$OUTPUT_BASE"

# Build classifier arguments if provided
CLASSIFIER_ARGS=""
if [ -n "$MK_CLASSIFIER" ] && [ -f "$MK_CLASSIFIER" ]; then
    CLASSIFIER_ARGS="$CLASSIFIER_ARGS --mk-classifier $MK_CLASSIFIER"
    echo "Using MK classifier: $MK_CLASSIFIER"
fi
if [ -n "$HSPC_CLASSIFIER" ] && [ -f "$HSPC_CLASSIFIER" ]; then
    CLASSIFIER_ARGS="$CLASSIFIER_ARGS --hspc-classifier $HSPC_CLASSIFIER"
    echo "Using HSPC classifier: $HSPC_CLASSIFIER"
fi

# ========================================
# PROCESS SLIDES
# ========================================

TOTAL=${#SLIDES[@]}
COMPLETED=0
FAILED=0

for i in "${!SLIDES[@]}"; do
    SLIDE="${SLIDES[$i]}"
    SLIDE_NUM=$((i + 1))

    echo "=========================================="
    echo "Slide $SLIDE_NUM/$TOTAL: $SLIDE"
    echo "Start time: $(date)"
    echo "=========================================="

    # Build full CZI path - check both flat and nested structures
    if [ -f "$CZI_BASE/${SLIDE}.czi" ]; then
        CZI_PATH="$CZI_BASE/${SLIDE}.czi"
    elif [ -f "$CZI_BASE/${SLIDE}/${SLIDE}.czi" ]; then
        CZI_PATH="$CZI_BASE/${SLIDE}/${SLIDE}.czi"
    else
        echo "ERROR: Cannot find CZI file for $SLIDE"
        echo "Checked:"
        echo "  - $CZI_BASE/${SLIDE}.czi"
        echo "  - $CZI_BASE/${SLIDE}/${SLIDE}.czi"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Run segmentation
    python run_unified_FAST.py \
        --czi-path "$CZI_PATH" \
        --output-dir "$OUTPUT_BASE/$SLIDE" \
        --tile-size $TILE_SIZE \
        --num-workers $NUM_WORKERS \
        --mk-min-area-um $MK_MIN_AREA_UM \
        --mk-max-area-um $MK_MAX_AREA_UM \
        --sample-fraction $SAMPLE_FRACTION \
        --calibration-samples 100 \
        $CLASSIFIER_ARGS

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $SLIDE completed"
        COMPLETED=$((COMPLETED + 1))
        echo "End time: $(date)"
    else
        echo "✗ ERROR: $SLIDE failed"
        FAILED=$((FAILED + 1))
        echo "Check GPU memory (nvidia-smi) and reduce --num-workers if needed"
    fi

    echo ""
done

# ========================================
# SUMMARY
# ========================================

echo "=========================================="
echo "PROCESSING COMPLETE"
echo "=========================================="
echo "Total slides: $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Results in: $OUTPUT_BASE"
echo "=========================================="

# Exit with error code if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
