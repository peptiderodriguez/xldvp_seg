#!/bin/bash
# ============================================================================
# LOCAL RUN SCRIPT - Optimized for this workstation
# ============================================================================
# Hardware: 48 cores, 432GB RAM, RTX 4090 (24GB VRAM)
# Settings leave ~6GB GPU and 50GB RAM for other tasks
# ============================================================================

set -e  # Exit on error

# ============================================================================
# PATHS - Configured for this machine
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CZI_BASE="/mnt/x/01_Users/EdRo_axioscan/bonemarrow/2025_11_18"
OUTPUT_BASE="$HOME/xldvp_seg_output"

# ============================================================================
# PROCESSING PARAMETERS - Optimized for RTX 4090 + 432GB RAM
# ============================================================================
# Resource targets: 75% CPU (36 of 48 threads), 75% RAM (324 of 432GB)
# HOWEVER: GPU memory (24GB) is the bottleneck - each worker needs ~8-12GB VRAM
# With 1 worker we use ~50% GPU, leaving headroom for stability
# ============================================================================
TILE_SIZE=3000              # 3000x3000 tiles
NUM_WORKERS=1               # Limited by GPU memory, not CPU/RAM
SAMPLE_FRACTION=0.10        # 10% sampling for annotation
MK_MIN_AREA_UM=200          # Minimum MK area in um2
MK_MAX_AREA_UM=2000         # Maximum MK area in um2
CALIBRATION_SAMPLES=100     # For auto-calibration

# ============================================================================
# PROCESSING MODE
# ============================================================================
# Set to "single" for one slide, "batch" for all 16 slides
# Batch mode loads models ONCE and processes all slides (much faster)
MODE="batch"

# Single slide to process (used when MODE="single")
SLIDE="2025_11_18_FGC1"

# All 16 slides (used when MODE="batch")
ALL_SLIDES=(
    "2025_11_18_FGC1" "2025_11_18_FGC2" "2025_11_18_FGC3" "2025_11_18_FGC4"
    "2025_11_18_FHU1" "2025_11_18_FHU2" "2025_11_18_FHU3" "2025_11_18_FHU4"
    "2025_11_18_MGC1" "2025_11_18_MGC2" "2025_11_18_MGC3" "2025_11_18_MGC4"
    "2025_11_18_MHU1" "2025_11_18_MHU2" "2025_11_18_MHU3" "2025_11_18_MHU4"
)

# ============================================================================
# SETUP
# ============================================================================
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Cell Segmentation - Local Run"
echo "=============================================="
echo "Mode:          $MODE"
echo "CZI source:    $CZI_BASE"
echo "Output:        $OUTPUT_BASE"
echo "Tile size:     ${TILE_SIZE}x${TILE_SIZE}"
echo "Workers:       $NUM_WORKERS"
echo "Sample:        ${SAMPLE_FRACTION} (10%)"
echo "MK area range: ${MK_MIN_AREA_UM}-${MK_MAX_AREA_UM} um2"
echo "=============================================="

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total --format=csv
echo ""

# ============================================================================
# RUN SEGMENTATION
# ============================================================================
cd "$SCRIPT_DIR"

if [ "$MODE" = "batch" ]; then
    # BATCH MODE: Process all slides with models loaded ONCE
    echo "BATCH MODE: Processing ${#ALL_SLIDES[@]} slides"
    echo "Models will be loaded ONCE and reused for all slides"
    echo "Start time: $(date)"
    echo ""

    # Build list of CZI paths
    CZI_PATHS=()
    for SLIDE_NAME in "${ALL_SLIDES[@]}"; do
        CZI_PATH="$CZI_BASE/${SLIDE_NAME}.czi"
        if [ ! -f "$CZI_PATH" ]; then
            echo "WARNING: CZI file not found: $CZI_PATH (skipping)"
            continue
        fi
        CZI_PATHS+=("$CZI_PATH")
    done

    if [ ${#CZI_PATHS[@]} -eq 0 ]; then
        echo "ERROR: No valid CZI files found"
        exit 1
    fi

    echo "Found ${#CZI_PATHS[@]} valid CZI files"

    # HTML export is integrated - happens automatically while slides are in RAM
    DOCS_DIR="$SCRIPT_DIR/docs"
    mkdir -p "$DOCS_DIR"

    python -u run_unified_FAST.py \
        --czi-paths "${CZI_PATHS[@]}" \
        --output-dir "$OUTPUT_BASE" \
        --tile-size $TILE_SIZE \
        --num-workers $NUM_WORKERS \
        --mk-min-area-um $MK_MIN_AREA_UM \
        --mk-max-area-um $MK_MAX_AREA_UM \
        --sample-fraction $SAMPLE_FRACTION \
        --calibration-samples $CALIBRATION_SAMPLES \
        --html-output-dir "$DOCS_DIR" \
        --samples-per-page 300

    echo ""
    echo "=============================================="
    echo "BATCH SEGMENTATION COMPLETE"
    echo "End time: $(date)"
    echo "=============================================="

else
    # SINGLE MODE: Process one slide
    CZI_PATH="$CZI_BASE/${SLIDE}.czi"

    if [ ! -f "$CZI_PATH" ]; then
        echo "ERROR: CZI file not found: $CZI_PATH"
        exit 1
    fi

    echo "Processing: $SLIDE"
    echo "Start time: $(date)"
    echo ""

    python -u run_unified_FAST.py \
        --czi-path "$CZI_PATH" \
        --output-dir "$OUTPUT_BASE/$SLIDE" \
        --tile-size $TILE_SIZE \
        --num-workers $NUM_WORKERS \
        --mk-min-area-um $MK_MIN_AREA_UM \
        --mk-max-area-um $MK_MAX_AREA_UM \
        --sample-fraction $SAMPLE_FRACTION \
        --calibration-samples $CALIBRATION_SAMPLES

    echo ""
    echo "=============================================="
    echo "SEGMENTATION COMPLETE: $SLIDE"
    echo "End time: $(date)"
    echo "=============================================="
fi

# ============================================================================
# DONE - HTML export happened automatically during segmentation
# ============================================================================
echo ""
echo "=============================================="
echo "ALL COMPLETE"
echo "=============================================="
echo "Output:  $OUTPUT_BASE"
echo "HTML:    $SCRIPT_DIR/docs"
echo ""
echo "To view locally:"
echo "  cd $SCRIPT_DIR/docs && python -m http.server 8080"
echo "  Open: http://localhost:8080"
echo "=============================================="

# ============================================================================
# RUN ALL SLIDES (if SLIDES array is defined)
# ============================================================================
if [ ${#SLIDES[@]} -gt 0 ]; then
    TOTAL=${#SLIDES[@]}
    COMPLETED=0
    FAILED=0

    for i in "${!SLIDES[@]}"; do
        SLIDE="${SLIDES[$i]}"
        SLIDE_NUM=$((i + 1))
        CZI_PATH="$CZI_BASE/${SLIDE}.czi"

        echo "=============================================="
        echo "Slide $SLIDE_NUM/$TOTAL: $SLIDE"
        echo "Start time: $(date)"
        echo "=============================================="

        if [ ! -f "$CZI_PATH" ]; then
            echo "ERROR: CZI not found: $CZI_PATH"
            FAILED=$((FAILED + 1))
            continue
        fi

        cd "$SCRIPT_DIR"
        python -u run_unified_FAST.py \
            --czi-path "$CZI_PATH" \
            --output-dir "$OUTPUT_BASE/$SLIDE" \
            --tile-size $TILE_SIZE \
            --num-workers $NUM_WORKERS \
            --mk-min-area-um $MK_MIN_AREA_UM \
            --mk-max-area-um $MK_MAX_AREA_UM \
            --sample-fraction $SAMPLE_FRACTION \
            --calibration-samples $CALIBRATION_SAMPLES

        if [ $? -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
            echo "SUCCESS: $SLIDE"
        else
            FAILED=$((FAILED + 1))
            echo "FAILED: $SLIDE"
        fi

        echo "End time: $(date)"
        echo ""
    done

    echo "=============================================="
    echo "BATCH COMPLETE"
    echo "Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED"
    echo "Output: $OUTPUT_BASE"
    echo "=============================================="
fi
