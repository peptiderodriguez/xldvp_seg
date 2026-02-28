#!/bin/bash
# =============================================================================
# General-purpose pipeline chain launcher
#
# Chains SLURM jobs with --dependency=afterok so each step waits for the
# previous one to succeed. Steps can be skipped by omitting their flags.
#
# Steps:
#   1. DETECT   — run_segmentation.py (GPU, multi-node sharding optional)
#   2. MERGE    — run_segmentation.py --merge-shards (CPU, after detect)
#   3. HTML     — regenerate_html.py (CPU, annotation HTML)
#   4. SCORE    — apply_classifier.py (CPU, requires --classifier)
#   5. ANALYSIS — spatial_cell_analysis.py (CPU, UMAP + network)
#   6. LMD      — run_lmd_export.py (CPU, requires --crosses)
#
# Usage examples:
#   # Full NMJ pipeline: detect 4 nodes -> merge -> annotation HTML
#   bash slurm/launch_pipeline.sh \
#       --czi /path/to/slide.czi \
#       --cell-type nmj --channel 1 \
#       --nodes 4 --sample-fraction 1.0 \
#       --steps detect,merge,html
#
#   # Score + analysis + LMD (post-annotation, reuses existing output)
#   bash slurm/launch_pipeline.sh \
#       --output-dir /path/to/existing/run \
#       --czi /path/to/slide.czi \
#       --cell-type nmj \
#       --classifier /path/to/rf_classifier.pkl \
#       --annotations /path/to/annotations.json \
#       --crosses /path/to/crosses.json \
#       --steps score,analysis,lmd
#
#   # Single-node vessel detection + HTML
#   bash slurm/launch_pipeline.sh \
#       --czi /path/to/slide.czi \
#       --cell-type vessel --channel 0 \
#       --nodes 1 --gpus 4 --partition p.hpcl93 \
#       --steps detect,html
#
#   # Everything on one node
#   bash slurm/launch_pipeline.sh \
#       --czi /path/to/slide.czi \
#       --cell-type cell --channel 0 \
#       --nodes 1 --sample-fraction 0.10 \
#       --steps detect,html
# =============================================================================

set -euo pipefail

REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python
LOG_DIR="$REPO/slurm/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CZI=""
CELL_TYPE=""
CHANNEL=""
OUTPUT_DIR=""
OUTPUT_BASE=""
SAMPLE_FRACTION="1.0"
TILE_SIZE="3000"
TILE_OVERLAP="0.10"
NUM_NODES=1
NUM_GPUS=4
PARTITION_GPU="p.hpcl93"
PARTITION_CPU="p.hpcl8"
GPU_TYPE="l40s"
CLASSIFIER=""
ANNOTATIONS=""
CROSSES=""
CLUSTERS=""
STEPS=""
SCORE_THRESHOLD="0.5"
MAX_SAMPLES="1500"
DISPLAY_CHANNELS=""
EXTRA_SEG_ARGS=""
EXTRA_HTML_ARGS=""
EXTRA_LMD_ARGS=""
ANALYSIS_MODES="morph-umap,spatial-network"
SEED="42"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --czi)              CZI="$2"; shift 2 ;;
        --cell-type)        CELL_TYPE="$2"; shift 2 ;;
        --channel)          CHANNEL="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --output-base)      OUTPUT_BASE="$2"; shift 2 ;;
        --sample-fraction)  SAMPLE_FRACTION="$2"; shift 2 ;;
        --tile-size)        TILE_SIZE="$2"; shift 2 ;;
        --tile-overlap)     TILE_OVERLAP="$2"; shift 2 ;;
        --nodes|--num-nodes) NUM_NODES="$2"; shift 2 ;;
        --gpus|--num-gpus)  NUM_GPUS="$2"; shift 2 ;;
        --partition)        PARTITION_GPU="$2"; PARTITION_CPU="$2"; shift 2 ;;
        --partition-gpu)    PARTITION_GPU="$2"; shift 2 ;;
        --partition-cpu)    PARTITION_CPU="$2"; shift 2 ;;
        --resume)           OUTPUT_DIR="$2"; shift 2 ;;
        --gpu-type)         GPU_TYPE="$2"; shift 2 ;;
        --classifier)       CLASSIFIER="$2"; shift 2 ;;
        --annotations)      ANNOTATIONS="$2"; shift 2 ;;
        --crosses)          CROSSES="$2"; shift 2 ;;
        --clusters)         CLUSTERS="$2"; shift 2 ;;
        --steps)            STEPS="$2"; shift 2 ;;
        --score-threshold)  SCORE_THRESHOLD="$2"; shift 2 ;;
        --max-samples)      MAX_SAMPLES="$2"; shift 2 ;;
        --display-channels) DISPLAY_CHANNELS="$2"; shift 2 ;;
        --extra-seg-args)   EXTRA_SEG_ARGS="$2"; shift 2 ;;
        --extra-html-args)  EXTRA_HTML_ARGS="$2"; shift 2 ;;
        --extra-lmd-args)   EXTRA_LMD_ARGS="$2"; shift 2 ;;
        --analysis-modes)   ANALYSIS_MODES="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: bash slurm/launch_pipeline.sh --czi FILE --cell-type TYPE --steps STEPS [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --czi PATH              CZI slide path"
            echo "  --cell-type TYPE        nmj|mk|cell|vessel|islet|tissue_pattern|mesothelium"
            echo "  --steps LIST            Comma-separated: detect,merge,html,score,analysis,lmd"
            echo ""
            echo "Detection:"
            echo "  --channel N             Detection channel index"
            echo "  --nodes|--num-nodes N   Number of SLURM nodes for detect (default: 1)"
            echo "  --gpus|--num-gpus N     GPUs per node (default: 4)"
            echo "  --sample-fraction F     Tile sampling fraction (default: 1.0)"
            echo "  --tile-size N           Tile size in pixels (default: 3000)"
            echo "  --partition NAME        Set both GPU and CPU partition"
            echo "  --partition-gpu NAME    GPU partition (default: p.hpcl93)"
            echo "  --partition-cpu NAME    CPU partition (default: p.hpcl8)"
            echo "  --gpu-type TYPE         GPU type for --gres (default: l40s)"
            echo "  --extra-seg-args 'ARGS' Extra args passed to run_segmentation.py"
            echo ""
            echo "Output:"
            echo "  --output-dir|--resume   Reuse existing output directory (skip detect)"
            echo "  --output-base PATH      Base directory for new output (default: per cell type)"
            echo ""
            echo "Classification:"
            echo "  --classifier PATH       RF classifier .pkl (required for score step)"
            echo "  --annotations PATH      Annotations JSON (for training or HTML)"
            echo "  --score-threshold F     RF score filter (default: 0.5)"
            echo ""
            echo "HTML:"
            echo "  --max-samples N         Max HTML samples (default: 1500)"
            echo "  --display-channels L    Channel indices for RGB display (e.g. 1,2,0)"
            echo "  --extra-html-args 'A'   Extra args for regenerate_html.py"
            echo ""
            echo "Analysis:"
            echo "  --analysis-modes LIST   Comma-separated: rf-embedding,morph-umap,spatial-network"
            echo "                          (default: morph-umap,spatial-network)"
            echo ""
            echo "LMD Export:"
            echo "  --crosses PATH          Reference crosses JSON (required for lmd step)"
            echo "  --clusters PATH         Biological clusters JSON"
            echo "  --extra-lmd-args 'A'    Extra args for run_lmd_export.py"
            echo ""
            echo "Other:"
            echo "  --seed N                Random seed (default: 42)"
            echo "  --dry-run               Print commands without submitting"
            exit 0
            ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$STEPS" ]]; then
    echo "ERROR: --steps required (e.g. --steps detect,merge,html)"
    exit 1
fi

# Parse steps into flags
HAS_DETECT=false; HAS_MERGE=false; HAS_HTML=false
HAS_SCORE=false; HAS_ANALYSIS=false; HAS_LMD=false
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"
for step in "${STEP_ARRAY[@]}"; do
    case "$step" in
        detect)   HAS_DETECT=true ;;
        merge)    HAS_MERGE=true ;;
        html)     HAS_HTML=true ;;
        score)    HAS_SCORE=true ;;
        analysis) HAS_ANALYSIS=true ;;
        lmd)      HAS_LMD=true ;;
        *) echo "ERROR: Unknown step '$step'. Valid: detect,merge,html,score,analysis,lmd"; exit 1 ;;
    esac
done

if [[ -z "$CELL_TYPE" ]]; then
    echo "ERROR: --cell-type required"
    exit 1
fi

if $HAS_DETECT && [[ -z "$CZI" ]]; then
    echo "ERROR: --czi required for detect step"
    exit 1
fi

if ! $DRY_RUN && $HAS_DETECT && [[ -n "$CZI" && ! -f "$CZI" ]]; then
    echo "ERROR: CZI file not found: $CZI"
    exit 1
fi

if $HAS_SCORE && [[ -z "$CLASSIFIER" ]]; then
    echo "ERROR: --classifier required for score step"
    exit 1
fi

if $HAS_LMD && [[ -z "$CROSSES" ]]; then
    echo "ERROR: --crosses required for lmd step"
    exit 1
fi

# Merge requires multi-node detect
if $HAS_MERGE && [[ "$NUM_NODES" -le 1 ]]; then
    echo "WARNING: --steps merge is only needed with --nodes > 1, skipping merge step"
    HAS_MERGE=false
fi

# ---------------------------------------------------------------------------
# Resolve output directory
# ---------------------------------------------------------------------------
if [[ -z "$OUTPUT_DIR" ]]; then
    # Create timestamped output directory
    if [[ -z "$OUTPUT_BASE" ]]; then
        OUTPUT_BASE="/fs/pool/pool-mann-edwin/${CELL_TYPE}_output"
    fi
    SLIDE_NAME=$(basename "$CZI" .czi)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PCT=$(echo "$SAMPLE_FRACTION * 100" | bc | cut -d. -f1)
    OUTPUT_DIR="${OUTPUT_BASE}/${SLIDE_NAME}_${TIMESTAMP}_${PCT}pct"
fi

# Derive paths
DETECTIONS_FILE="${OUTPUT_DIR}/${CELL_TYPE}_detections.json"
SCORED_FILE="${OUTPUT_DIR}/${CELL_TYPE}_detections_scored.json"
ANALYSIS_DIR="${OUTPUT_DIR}/spatial_analysis"
CLASSIFIER_DIR="${OUTPUT_DIR}/classifier"
HTML_DIR="${OUTPUT_DIR}/html"
LMD_DIR="${OUTPUT_DIR}/lmd_export"

# ---------------------------------------------------------------------------
# Print plan
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Pipeline Chain Launcher"
echo "=========================================="
echo "Cell type:       $CELL_TYPE"
echo "CZI:             ${CZI:-<using existing output>}"
echo "Output:          $OUTPUT_DIR"
echo "Steps:           $STEPS"
echo "Nodes:           $NUM_NODES"
echo "GPUs/node:       $NUM_GPUS"
echo "Sample fraction: $SAMPLE_FRACTION"
if [[ -n "$CLASSIFIER" ]]; then echo "Classifier:      $CLASSIFIER"; fi
if [[ -n "$ANNOTATIONS" ]]; then echo "Annotations:     $ANNOTATIONS"; fi
if [[ -n "$CROSSES" ]]; then echo "Crosses:         $CROSSES"; fi
echo "=========================================="
echo ""

if $DRY_RUN; then
    echo "[DRY RUN — commands will be printed but not submitted]"
    echo ""
fi

# Helper: submit or print
LAST_JOB=""
submit_job() {
    local job_name="$1"
    local partition="$2"
    local resources="$3"  # e.g. "--cpus-per-task=24 --mem=64G"
    local time_limit="$4"
    local script="$5"
    local depend=""

    if [[ -n "$LAST_JOB" ]]; then
        depend="--dependency=afterok:$LAST_JOB"
    fi

    local cmd="sbatch --parsable --job-name=${job_name} --partition=${partition} ${resources} --time=${time_limit} --output=${LOG_DIR}/${job_name}_%j.out --error=${LOG_DIR}/${job_name}_%j.err ${depend} ${script}"

    if $DRY_RUN; then
        echo "[DRY RUN] $cmd"
        LAST_JOB="DRY_${job_name}"
    else
        LAST_JOB=$(eval "$cmd")
        echo "  Submitted $job_name: job $LAST_JOB"
    fi
}

# ---------------------------------------------------------------------------
# Common environment block (inline script preamble)
# ---------------------------------------------------------------------------
ENV_BLOCK="
set -euo pipefail
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
cd $REPO
"

# ---------------------------------------------------------------------------
# Step 1: DETECT
# ---------------------------------------------------------------------------
if $HAS_DETECT; then
    echo "--- Step 1: DETECT ($NUM_NODES node(s), $NUM_GPUS GPU(s) each) ---"
    if ! $DRY_RUN; then
        mkdir -p "$OUTPUT_DIR/tiles"
    fi

    # Build segmentation args
    SEG_ARGS=(
        "$PYTHON" run_segmentation.py
        --czi-path "$CZI"
        --cell-type "$CELL_TYPE"
        --output-dir "$(dirname "$OUTPUT_DIR")"
        --resume "$OUTPUT_DIR"
        --sample-fraction "$SAMPLE_FRACTION"
        --tile-size "$TILE_SIZE"
        --tile-overlap "$TILE_OVERLAP"
        --num-gpus "$NUM_GPUS"
        --load-to-ram
        --all-channels
        --random-seed "$SEED"
        --no-serve
    )

    if [[ -n "$CHANNEL" ]]; then
        SEG_ARGS+=(--channel "$CHANNEL")
    fi
    if [[ -n "$CLASSIFIER" ]]; then
        SEG_ARGS+=(--nmj-classifier "$CLASSIFIER")
    fi
    if [[ -n "$ANNOTATIONS" ]]; then
        SEG_ARGS+=(--prior-annotations "$ANNOTATIONS")
    fi
    if [[ -n "$EXTRA_SEG_ARGS" ]]; then
        SEG_ARGS+=($EXTRA_SEG_ARGS)
    fi

    if [[ "$NUM_NODES" -gt 1 ]]; then
        # Multi-node: submit independent shard jobs (all run in parallel)
        SHARD_JOBS=""
        SAVED_LAST_JOB="$LAST_JOB"
        for shard_idx in $(seq 0 $((NUM_NODES - 1))); do
            # Each shard depends only on whatever came before detect (not on other shards)
            LAST_JOB="$SAVED_LAST_JOB"
            SHARD_CMD="${ENV_BLOCK}
${SEG_ARGS[*]} --tile-shard ${shard_idx}/${NUM_NODES}
"
            submit_job \
                "${CELL_TYPE}_shard${shard_idx}" \
                "$PARTITION_GPU" \
                "--nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G --gres=gpu:${GPU_TYPE}:${NUM_GPUS}" \
                "12:00:00" \
                "--wrap \"$SHARD_CMD\""
            if $DRY_RUN; then
                SHARD_JOBS="${SHARD_JOBS}:DRY_${CELL_TYPE}_shard${shard_idx}"
            else
                SHARD_JOBS="${SHARD_JOBS}:${LAST_JOB}"
            fi
        done
        # Next step must depend on ALL shards completing
        # Set LAST_JOB to a comma-separated list for afterok dependency
        LAST_JOB="${SHARD_JOBS#:}"  # remove leading colon
        echo "  (All $NUM_NODES shard jobs submitted in parallel)"
    else
        # Single node
        DETECT_CMD="${ENV_BLOCK}
${SEG_ARGS[*]}
"
        submit_job \
            "${CELL_TYPE}_detect" \
            "$PARTITION_GPU" \
            "--nodes=1 --ntasks=1 --cpus-per-task=64 --mem=500G --gres=gpu:${GPU_TYPE}:${NUM_GPUS}" \
            "24:00:00" \
            "--wrap \"$DETECT_CMD\""
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: MERGE (multi-node only)
# ---------------------------------------------------------------------------
if $HAS_MERGE; then
    echo "--- Step 2: MERGE (dedup + combine shards) ---"
    MERGE_CMD="${ENV_BLOCK}
$PYTHON run_segmentation.py \\
    --czi-path '$CZI' \\
    --cell-type $CELL_TYPE \\
    --output-dir '$(dirname "$OUTPUT_DIR")' \\
    --resume '$OUTPUT_DIR' \\
    --merge-shards \\
    --sample-fraction $SAMPLE_FRACTION \\
    --tile-size $TILE_SIZE \\
    --tile-overlap $TILE_OVERLAP \\
    --num-gpus 1 \\
    --load-to-ram \\
    --all-channels \\
    --random-seed $SEED \\
    --no-serve
"
    submit_job \
        "${CELL_TYPE}_merge" \
        "$PARTITION_CPU" \
        "--nodes=1 --ntasks=1 --cpus-per-task=24 --mem=350G" \
        "4:00:00" \
        "--wrap \"$MERGE_CMD\""
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: HTML (annotation viewer)
# ---------------------------------------------------------------------------
if $HAS_HTML; then
    echo "--- Step 3: HTML (annotation viewer, max $MAX_SAMPLES samples) ---"

    # Use scored detections if score step already ran (html after score)
    HTML_DET_ARGS=""
    if $HAS_SCORE; then
        HTML_DET_ARGS="--detections '$SCORED_FILE' --score-threshold $SCORE_THRESHOLD"
    fi

    HTML_CMD="${ENV_BLOCK}
$PYTHON scripts/regenerate_html.py \\
    --output-dir '$OUTPUT_DIR' \\
    --czi-path '$CZI' \\
    --cell-type $CELL_TYPE \\
    --max-samples $MAX_SAMPLES \\
    --html-dir '$HTML_DIR' \\
    $HTML_DET_ARGS
"
    if [[ -n "$DISPLAY_CHANNELS" ]]; then
        HTML_CMD+=" --display-channels $DISPLAY_CHANNELS"
    fi
    if [[ -n "$ANNOTATIONS" ]]; then
        HTML_CMD+=" --prior-annotations '$ANNOTATIONS'"
    fi
    if [[ -n "$EXTRA_HTML_ARGS" ]]; then
        HTML_CMD+=" $EXTRA_HTML_ARGS"
    fi

    submit_job \
        "${CELL_TYPE}_html" \
        "$PARTITION_CPU" \
        "--nodes=1 --ntasks=1 --cpus-per-task=24 --mem=200G" \
        "2:00:00" \
        "--wrap \"$HTML_CMD\""
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 4: SCORE (apply classifier)
# ---------------------------------------------------------------------------
if $HAS_SCORE; then
    echo "--- Step 4: SCORE (apply classifier, threshold $SCORE_THRESHOLD) ---"
    SCORE_CMD="${ENV_BLOCK}
$PYTHON scripts/apply_classifier.py \\
    --detections '$DETECTIONS_FILE' \\
    --classifier '$CLASSIFIER' \\
    --output '$SCORED_FILE'
"
    submit_job \
        "${CELL_TYPE}_score" \
        "$PARTITION_CPU" \
        "--nodes=1 --ntasks=1 --cpus-per-task=24 --mem=64G" \
        "00:30:00" \
        "--wrap \"$SCORE_CMD\""
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 5: ANALYSIS (spatial cell analysis)
# ---------------------------------------------------------------------------
if $HAS_ANALYSIS; then
    echo "--- Step 5: ANALYSIS ($ANALYSIS_MODES) ---"

    # Use scored detections if score step ran, otherwise raw
    if $HAS_SCORE; then
        ANALYSIS_DET="$SCORED_FILE"
    elif [[ -n "$CLASSIFIER" ]]; then
        ANALYSIS_DET="$SCORED_FILE"
    else
        ANALYSIS_DET="$DETECTIONS_FILE"
    fi

    ANALYSIS_CMD="${ENV_BLOCK}
$PYTHON scripts/spatial_cell_analysis.py \\
    --detections '$ANALYSIS_DET' \\
    --output-dir '$ANALYSIS_DIR'
"
    # Add mode flags
    IFS=',' read -ra MODES <<< "$ANALYSIS_MODES"
    for mode in "${MODES[@]}"; do
        ANALYSIS_CMD+=" --${mode}"
    done

    # Add classifier for rf-embedding if available
    if [[ -n "$CLASSIFIER" ]]; then
        ANALYSIS_CMD+=" --classifier '$CLASSIFIER'"
    fi

    # Add score threshold if scoring was applied
    if $HAS_SCORE || [[ -n "$CLASSIFIER" ]]; then
        ANALYSIS_CMD+=" --score-threshold $SCORE_THRESHOLD"
    fi

    submit_job \
        "${CELL_TYPE}_analysis" \
        "$PARTITION_CPU" \
        "--nodes=1 --ntasks=1 --cpus-per-task=24 --mem=64G" \
        "1:00:00" \
        "--wrap \"$ANALYSIS_CMD\""
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 6: LMD (export for laser microdissection)
# ---------------------------------------------------------------------------
if $HAS_LMD; then
    echo "--- Step 6: LMD (export XML for microdissection) ---"

    if $HAS_SCORE; then
        LMD_DET="$SCORED_FILE"
    elif [[ -n "$CLASSIFIER" ]]; then
        LMD_DET="$SCORED_FILE"
    else
        LMD_DET="$DETECTIONS_FILE"
    fi

    LMD_CMD="${ENV_BLOCK}
$PYTHON run_lmd_export.py \\
    --detections '$LMD_DET' \\
    --cell-type $CELL_TYPE \\
    --crosses '$CROSSES' \\
    --output-dir '$LMD_DIR' \\
    --export --generate-controls
"
    if [[ -n "$CLUSTERS" ]]; then
        LMD_CMD+=" --clusters '$CLUSTERS'"
    fi
    if $HAS_SCORE || [[ -n "$CLASSIFIER" ]]; then
        LMD_CMD+=" --min-score $SCORE_THRESHOLD"
    fi
    if [[ -n "$EXTRA_LMD_ARGS" ]]; then
        LMD_CMD+=" $EXTRA_LMD_ARGS"
    fi

    submit_job \
        "${CELL_TYPE}_lmd" \
        "$PARTITION_CPU" \
        "--nodes=1 --ntasks=1 --cpus-per-task=24 --mem=64G" \
        "1:00:00" \
        "--wrap \"$LMD_CMD\""
    echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Pipeline submitted!"
echo "=========================================="
echo "Output:    $OUTPUT_DIR"
echo "Steps:     $STEPS"
echo "Last job:  $LAST_JOB"
echo ""
echo "Monitor:   squeue -u \$USER"
echo "Logs:      $LOG_DIR/"
echo ""

if $HAS_DETECT && ! $HAS_SCORE && ! $HAS_LMD; then
    echo "Next steps (after detection completes):"
    echo "  1. Serve HTML:  python serve_html.py --dir $HTML_DIR"
    echo "  2. Annotate in browser, export annotations JSON"
    echo "  3. Train classifier:"
    echo "     python train_classifier.py --detections $DETECTIONS_FILE --annotations <annotations.json> --output-dir $CLASSIFIER_DIR"
    echo "  4. Chain remaining steps:"
    echo "     bash slurm/launch_pipeline.sh --output-dir $OUTPUT_DIR --czi $CZI --cell-type $CELL_TYPE --classifier $CLASSIFIER_DIR/*.pkl --steps score,analysis"
    echo ""
fi
