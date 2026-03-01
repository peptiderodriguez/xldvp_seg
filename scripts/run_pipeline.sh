#!/bin/bash
set -euo pipefail

# run_pipeline.sh - Read a YAML config and submit SLURM jobs for the
# segmentation -> marker classification -> spatial analysis pipeline.
#
# Usage:
#   scripts/run_pipeline.sh configs/senescence.yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MKSEG_PYTHON="${MKSEG_PYTHON:-/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"

# ---------------------------------------------------------------------------
# Argument check
# ---------------------------------------------------------------------------
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <config.yaml>"
    exit 1
fi
CONFIG="$(realpath "$1")"
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse YAML config with Python (needs PyYAML)
# ---------------------------------------------------------------------------
read_yaml() {
    # Usage: read_yaml KEY [DEFAULT]
    local key="$1"
    local default="${2:-}"
    local val
    val=$("$MKSEG_PYTHON" -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
keys = sys.argv[2].split('.')
v = cfg
for k in keys:
    if v is None:
        v = None
        break
    v = v.get(k) if isinstance(v, dict) else None
if v is None:
    print(sys.argv[3])
elif isinstance(v, list):
    print(' '.join(str(x) for x in v))
elif isinstance(v, bool):
    print('true' if v else 'false')
else:
    print(v)
" "$CONFIG" "$key" "$default")
    echo "$val"
}

# ---------------------------------------------------------------------------
# Read config values
# ---------------------------------------------------------------------------
NAME=$(read_yaml name pipeline)
CZI_DIR=$(read_yaml czi_dir "")
CZI_GLOB=$(read_yaml czi_glob "")
CZI_PATH=$(read_yaml czi_path "")
OUTPUT_DIR=$(read_yaml output_dir "")
CELL_TYPE=$(read_yaml cell_type cell)
CP_CHANNELS=$(read_yaml cellpose_input_channels "")
PHOTOBLEACH=$(read_yaml photobleach_correction false)
ALL_CHANNELS=$(read_yaml all_channels false)
LOAD_CHANNELS=$(read_yaml load_channels "")
NUM_GPUS=$(read_yaml num_gpus 1)
MIN_AREA=$(read_yaml min_area_um "")
MAX_AREA=$(read_yaml max_area_um "")

# Channel map (wavelength/name-based channel specs, resolved at runtime)
# Supported YAML keys: channel_map.detect, channel_map.cyto, channel_map.nuc
CHANNEL_SPEC=$("$MKSEG_PYTHON" -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
cm = cfg.get('channel_map')
if not cm or not isinstance(cm, dict):
    sys.exit(0)
pairs = []
for role, spec in cm.items():
    pairs.append(f'{role}={spec}')
print(','.join(pairs))
" "$CONFIG" || echo "")
CORRECT_ALL_CHANNELS=$(read_yaml correct_all_channels false)
SAMPLE_FRACTION=$(read_yaml sample_fraction "")

# Post-dedup processing (contour dilation + background correction)
BACKGROUND_CORRECTION=$(read_yaml background_correction true)
CONTOUR_PROCESSING=$(read_yaml contour_processing true)
DILATION_UM=$(read_yaml dilation_um "")
RDP_EPSILON=$(read_yaml rdp_epsilon "")
BG_NEIGHBORS=$(read_yaml bg_neighbors "")

# SLURM settings
PARTITION=$(read_yaml slurm.partition p.hpcl8)
CPUS=$(read_yaml slurm.cpus 24)
MEM_GB=$(read_yaml slurm.mem_gb 350)
GPUS=$(read_yaml slurm.gpus "rtx5000:2")
TIME=$(read_yaml slurm.time "2-00:00:00")
SLIDES_PER_JOB=$(read_yaml slurm.slides_per_job 1)
NUM_JOBS=$(read_yaml slurm.num_jobs 1)

# Markers (parsed as JSON list, supports both 'channel' and 'wavelength' keys)
MARKERS_JSON=$("$MKSEG_PYTHON" -c "
import yaml, json, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
print(json.dumps(cfg.get('markers', [])))
" "$CONFIG")

# Spatial network
SPATIAL_ENABLED=$(read_yaml spatial_network.enabled "")
SPATIAL_FILTER=$(read_yaml spatial_network.marker_filter "")
SPATIAL_EDGE=$(read_yaml spatial_network.max_edge_distance 50)
SPATIAL_MIN_COMP=$(read_yaml spatial_network.min_component_cells 3)
PIXEL_SIZE=$(read_yaml pixel_size_um "")
# If spatial_network section exists but enabled is not explicitly set, treat as enabled
# But respect explicit enabled: false
# Note: SPATIAL_EDGE and SPATIAL_MIN_COMP always have defaults (50, 3) so they can't
# distinguish "user set a value" from "default". Only marker_filter has empty default,
# so it's the reliable signal that spatial_network was intentionally configured.
if [[ "$SPATIAL_ENABLED" == "false" ]]; then
    : # Explicitly disabled, do nothing
elif [[ -n "$SPATIAL_FILTER" && -z "$SPATIAL_ENABLED" ]]; then
    SPATIAL_ENABLED="true"
fi

# SpatialData export
SPATIALDATA_ENABLED=$(read_yaml spatialdata.enabled true)
SPATIALDATA_SHAPES=$(read_yaml spatialdata.extract_shapes true)
SPATIALDATA_SQUIDPY=$(read_yaml spatialdata.run_squidpy false)
SPATIALDATA_CLUSTER_KEY=$(read_yaml spatialdata.squidpy_cluster_key "")

# Spatial viewer
VIEWER_ENABLED=$(read_yaml spatial_viewer.enabled false)
VIEWER_GROUP_FIELD=$(read_yaml spatial_viewer.group_field "")
VIEWER_TITLE=$(read_yaml spatial_viewer.title "Multi-Slide Spatial Viewer")
VIEWER_TITLE_ESC="${VIEWER_TITLE//\"/\\\"}"

# Validate pixel_size_um is set when spatial analysis is enabled
if [[ "$SPATIAL_ENABLED" == "true" && -z "$PIXEL_SIZE" ]]; then
    echo "Error: spatial_network requires pixel_size_um in config" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Determine multi-slide vs single-slide mode
# ---------------------------------------------------------------------------
MULTI_SLIDE=false
if [[ -n "$CZI_DIR" && -n "$CZI_GLOB" ]]; then
    MULTI_SLIDE=true
elif [[ -z "$CZI_PATH" ]]; then
    echo "Error: config must provide either czi_dir+czi_glob (multi-slide) or czi_path (single-slide)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build run_segmentation.py command flags
# ---------------------------------------------------------------------------
build_seg_cmd() {
    local czi_arg="$1"
    local out_arg="$2"
    local cmd="$MKSEG_PYTHON $REPO/run_segmentation.py"
    cmd+=" --czi-path \"$czi_arg\""
    cmd+=" --cell-type \"$CELL_TYPE\""
    cmd+=" --output-dir \"$out_arg\""
    cmd+=" --num-gpus \"$NUM_GPUS\""
    cmd+=" --no-serve"

    # Cellpose input channels: passed as "cyto,nuc" to --cellpose-input-channels
    if [[ -n "$CP_CHANNELS" ]]; then
        local ch_cyto ch_nuc
        ch_cyto=$(echo "$CP_CHANNELS" | awk '{print $1}')
        ch_nuc=$(echo "$CP_CHANNELS" | awk '{print $2}')
        cmd+=" --cellpose-input-channels ${ch_cyto},${ch_nuc}"
    fi

    if [[ "$PHOTOBLEACH" == "true" ]]; then
        cmd+=" --photobleaching-correction"
    fi
    if [[ "$ALL_CHANNELS" == "true" ]]; then
        cmd+=" --all-channels"
    fi
    if [[ -n "$LOAD_CHANNELS" ]]; then
        cmd+=" --channels \"$LOAD_CHANNELS\""
    fi
    if [[ -n "$MIN_AREA" ]]; then
        cmd+=" --min-cell-area $MIN_AREA"
    fi
    if [[ -n "$MAX_AREA" ]]; then
        cmd+=" --max-cell-area $MAX_AREA"
    fi
    if [[ -n "$SAMPLE_FRACTION" ]]; then
        cmd+=" --sample-fraction $SAMPLE_FRACTION"
    fi
    if [[ -n "$CHANNEL_SPEC" ]]; then
        cmd+=" --channel-spec \"$CHANNEL_SPEC\""
    fi
    # Post-dedup processing flags
    if [[ "$BACKGROUND_CORRECTION" == "false" ]]; then
        cmd+=" --no-background-correction"
    fi
    if [[ "$CONTOUR_PROCESSING" == "false" ]]; then
        cmd+=" --no-contour-processing"
    fi
    if [[ -n "$DILATION_UM" ]]; then
        cmd+=" --dilation-um $DILATION_UM"
    fi
    if [[ -n "$RDP_EPSILON" ]]; then
        cmd+=" --rdp-epsilon $RDP_EPSILON"
    fi
    if [[ -n "$BG_NEIGHBORS" ]]; then
        cmd+=" --bg-neighbors $BG_NEIGHBORS"
    fi
    echo "$cmd"
}

# ---------------------------------------------------------------------------
# Build comma-separated marker args for single classify_markers.py invocation
# ---------------------------------------------------------------------------
# Markers can use 'channel' (index) or 'wavelength' (resolved at runtime)
MARKER_USE_WAVELENGTH=$("$MKSEG_PYTHON" -c "
import json, sys
markers = json.loads(sys.argv[1])
if not markers:
    sys.exit(0)
# If any marker uses 'wavelength' key instead of 'channel', use wavelength mode
if any('wavelength' in m for m in markers):
    print('true')
else:
    print('false')
" "$MARKERS_JSON" || echo "false")

MARKER_CHANNELS=$("$MKSEG_PYTHON" -c "
import json, sys
markers = json.loads(sys.argv[1])
if not markers:
    sys.exit(0)
# Support both 'channel' and 'wavelength' keys
parts = []
for m in markers:
    if 'channel' in m:
        parts.append(str(m['channel']))
    elif 'wavelength' in m:
        parts.append(str(m['wavelength']))
print(','.join(parts))
" "$MARKERS_JSON" || echo "")

MARKER_NAMES=$("$MKSEG_PYTHON" -c "
import json, sys
markers = json.loads(sys.argv[1])
print(','.join(m['name'] for m in markers))
" "$MARKERS_JSON" || echo "")

MARKER_METHOD=$("$MKSEG_PYTHON" -c "
import json, sys
markers = json.loads(sys.argv[1])
if not markers:
    sys.exit(0)
methods = set(m.get('method', 'otsu_half') for m in markers)
if len(methods) > 1:
    print(f'ERROR: Mixed marker methods not supported in single invocation: {methods}', file=sys.stderr)
    sys.exit(1)
print(methods.pop())
" "$MARKERS_JSON" || echo "")

# Validate: if markers are defined but method/channels extraction failed, abort
if [[ -n "$MARKERS_JSON" && "$MARKERS_JSON" != "[]" ]]; then
    if [[ -n "$MARKER_CHANNELS" && -z "$MARKER_METHOD" ]]; then
        echo "Error: markers defined but method could not be determined. Check for mixed methods in config." >&2
        exit 1
    fi
    if [[ -z "$MARKER_CHANNELS" && -n "$MARKER_NAMES" ]] || [[ -n "$MARKER_CHANNELS" && -z "$MARKER_NAMES" ]]; then
        echo "Error: marker channels/names mismatch. Check markers section in config." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Write sbatch script
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
SBATCH_FILE="${OUTPUT_DIR}/pipeline_${NAME}_$$.sbatch"

{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${NAME}"
    echo "#SBATCH --partition=${PARTITION}"
    echo "#SBATCH --cpus-per-task=${CPUS}"
    echo "#SBATCH --mem=${MEM_GB}G"
    echo "#SBATCH --gres=gpu:${GPUS}"
    echo "#SBATCH --time=${TIME}"
    if [[ "$MULTI_SLIDE" == "true" ]]; then
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.err"
        echo "#SBATCH --array=0-$((NUM_JOBS - 1))"
    else
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%j.err"
    fi

    echo ""
    echo "set -euo pipefail"
    echo ""
    echo "export PYTHONPATH=\"$REPO\""
    echo "MKSEG_PYTHON=\"$MKSEG_PYTHON\""
    echo ""

    if [[ "$MULTI_SLIDE" == "true" ]]; then
        # Multi-slide array job: each task processes a slice of the slide list
        echo "# Discover slides matching glob"
        echo "mapfile -t ALL_SLIDES < <(find \"$CZI_DIR\" -maxdepth 1 -name '$CZI_GLOB' -type f | sort)"
        echo "TOTAL_SLIDES=\${#ALL_SLIDES[@]}"
        echo ""
        echo "if [[ \$TOTAL_SLIDES -eq 0 ]]; then"
        echo "    echo \"Error: no slides matched glob '$CZI_GLOB' in $CZI_DIR\""
        echo "    exit 1"
        echo "fi"
        echo ""
        echo "# Compute slide range for this array task"
        echo "START=\$(( SLURM_ARRAY_TASK_ID * $SLIDES_PER_JOB ))"
        echo "END=\$(( START + $SLIDES_PER_JOB ))"
        echo "if [[ \$END -gt \$TOTAL_SLIDES ]]; then END=\$TOTAL_SLIDES; fi"
        echo ""
        echo "echo \"Task \$SLURM_ARRAY_TASK_ID: processing slides \$START..\$((END-1)) of \$TOTAL_SLIDES\""
        echo ""
        echo "for (( i=START; i<END; i++ )); do"
        echo "    CZI_FILE=\"\${ALL_SLIDES[\$i]}\""
        echo "    SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
        echo "    SLIDE_OUT=\"${OUTPUT_DIR}/\${SLIDE_NAME}\""
        echo "    mkdir -p \"\$SLIDE_OUT\""
        echo "    echo \"=== Processing slide \$((i+1))/\$TOTAL_SLIDES: \$CZI_FILE ===\""
        echo ""
        echo "    # Step 1: Segmentation"
        echo "    $(build_seg_cmd '${CZI_FILE}' '${SLIDE_OUT}')"
        echo ""
        echo "    # Step 2: Find detection JSON in latest run subdir"
        echo "    RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
        echo "    DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "    if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"

        if [[ -n "$MARKER_CHANNELS" ]]; then
            classify_extra=""
            if [[ "$CORRECT_ALL_CHANNELS" == "true" ]]; then
                classify_extra+=" --correct-all-channels"
            fi
            echo "        echo \"  Classifying markers: $MARKER_NAMES\""
            if [[ "$MARKER_USE_WAVELENGTH" == "true" ]]; then
                echo "        \$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-wavelength \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --czi-path \"\$CZI_FILE\" --output-dir \"\$RUN_DIR\"${classify_extra}"
            else
                echo "        \$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --output-dir \"\$RUN_DIR\"${classify_extra}"
            fi
            echo "        DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\""
        fi

        if [[ "$SPATIAL_ENABLED" == "true" ]]; then
            echo ""
            echo "        # Step 3: Spatial analysis"
            echo "        echo \"  Running spatial analysis...\""
            echo "        \$MKSEG_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$RUN_DIR\" --spatial-network --marker-filter \"$SPATIAL_FILTER\" --max-edge-distance \"$SPATIAL_EDGE\" --min-component-cells \"$SPATIAL_MIN_COMP\" --pixel-size \"$PIXEL_SIZE\""
        fi

        if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
            echo ""
            echo "        # SpatialData export"
            echo "        echo \"  Exporting to SpatialData...\""
            sd_cmd="        \$MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
            if [[ "$SPATIALDATA_SHAPES" == "true" ]]; then
                sd_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
            else
                sd_cmd+=" --no-shapes"
            fi
            if [[ "$SPATIALDATA_SQUIDPY" == "true" ]]; then
                sd_cmd+=" --run-squidpy"
                if [[ -n "$SPATIALDATA_CLUSTER_KEY" ]]; then
                    sd_cmd+=" --squidpy-cluster-key \"$SPATIALDATA_CLUSTER_KEY\""
                fi
            fi
            echo "$sd_cmd"
        fi

        echo "    else"
        echo "        echo \"  WARNING: ${CELL_TYPE}_detections.json not found, skipping marker/spatial steps\""
        echo "    fi"
        echo "    echo \"\""
        echo "done"

    else
        # Single-slide job
        echo "SLIDE_OUT=\"${OUTPUT_DIR}\""
        echo "mkdir -p \"\$SLIDE_OUT\""
        echo ""
        echo "# Step 1: Segmentation"
        echo "$(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}')"
        echo ""
        echo "# Step 2: Find detection JSON in latest run subdir"
        echo "RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
        echo "DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"

        if [[ -n "$MARKER_CHANNELS" ]]; then
            classify_extra=""
            if [[ "$CORRECT_ALL_CHANNELS" == "true" ]]; then
                classify_extra+=" --correct-all-channels"
            fi
            echo "    echo \"Classifying markers: $MARKER_NAMES\""
            if [[ "$MARKER_USE_WAVELENGTH" == "true" ]]; then
                echo "    \$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-wavelength \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --czi-path \"$CZI_PATH\" --output-dir \"\$RUN_DIR\"${classify_extra}"
            else
                echo "    \$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --output-dir \"\$RUN_DIR\"${classify_extra}"
            fi
            echo "    DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\""
        fi

        if [[ "$SPATIAL_ENABLED" == "true" ]]; then
            echo ""
            echo "    # Step 3: Spatial analysis"
            echo "    echo \"Running spatial analysis...\""
            echo "    \$MKSEG_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$RUN_DIR\" --spatial-network --marker-filter \"$SPATIAL_FILTER\" --max-edge-distance \"$SPATIAL_EDGE\" --min-component-cells \"$SPATIAL_MIN_COMP\" --pixel-size \"$PIXEL_SIZE\""
        fi

        if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
            echo ""
            echo "    # SpatialData export"
            echo "    echo \"Exporting to SpatialData...\""
            sd_cmd="    \$MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
            if [[ "$SPATIALDATA_SHAPES" == "true" ]]; then
                sd_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
            else
                sd_cmd+=" --no-shapes"
            fi
            if [[ "$SPATIALDATA_SQUIDPY" == "true" ]]; then
                sd_cmd+=" --run-squidpy"
                if [[ -n "$SPATIALDATA_CLUSTER_KEY" ]]; then
                    sd_cmd+=" --squidpy-cluster-key \"$SPATIALDATA_CLUSTER_KEY\""
                fi
            fi
            echo "$sd_cmd"
        fi

        # Step 4: Spatial viewer (single-slide, inline)
        if [[ "$VIEWER_ENABLED" == "true" && -n "$VIEWER_GROUP_FIELD" ]]; then
            echo ""
            echo "    # Step 4: Spatial viewer"
            echo "    echo \"Generating spatial viewer...\""
            echo "    \$MKSEG_PYTHON $REPO/scripts/generate_multi_slide_spatial_viewer.py --input-dir \"\$SLIDE_OUT\" --group-field \"$VIEWER_GROUP_FIELD\" --title \"$VIEWER_TITLE_ESC\" --output \"\${SLIDE_OUT}/spatial_viewer.html\""
        fi

        echo "else"
        echo "    echo \"WARNING: ${CELL_TYPE}_detections.json not found, skipping marker/spatial steps\""
        echo "fi"
    fi

} > "$SBATCH_FILE"

chmod +x "$SBATCH_FILE"

# ---------------------------------------------------------------------------
# Summary and submit
# ---------------------------------------------------------------------------
echo "================================================"
echo "Pipeline: $NAME"
echo "Config:   $CONFIG"
echo "Sbatch:   $SBATCH_FILE"
if [[ "$MULTI_SLIDE" == "true" ]]; then
    echo "Mode:     multi-slide array ($NUM_JOBS jobs, $SLIDES_PER_JOB slides/job)"
    echo "CZI dir:  $CZI_DIR"
    echo "Glob:     $CZI_GLOB"
else
    echo "Mode:     single-slide"
    echo "CZI:      $CZI_PATH"
fi
echo "Output:   $OUTPUT_DIR"
echo "SLURM:    $PARTITION | ${CPUS} CPUs | ${MEM_GB}G RAM | gpu:${GPUS} | ${TIME}"
echo "================================================"
echo ""
echo "Submitting job..."
mkdir -p "$OUTPUT_DIR"
MAIN_JOB_OUTPUT=$(sbatch "$SBATCH_FILE") || { echo "Error: sbatch submission failed" >&2; exit 1; }
echo "$MAIN_JOB_OUTPUT"
MAIN_JOB_ID=$(echo "$MAIN_JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+') || true
if [[ -z "$MAIN_JOB_ID" ]]; then
    echo "Error: could not parse job ID from: $MAIN_JOB_OUTPUT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 4: Submit dependent spatial viewer job (multi-slide only)
# ---------------------------------------------------------------------------
if [[ "$MULTI_SLIDE" == "true" && "$VIEWER_ENABLED" == "true" && -n "$VIEWER_GROUP_FIELD" ]]; then
    VIEWER_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_viewer_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_viewer"
        echo "#SBATCH --partition=p.hpcl8"
        echo "#SBATCH --cpus-per-task=4"
        echo "#SBATCH --mem=32G"
        echo "#SBATCH --time=00:30:00"
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_viewer_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_viewer_%j.err"
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "MKSEG_PYTHON=\"$MKSEG_PYTHON\""
        echo ""
        echo "echo \"Generating multi-slide spatial viewer...\""
        echo "\$MKSEG_PYTHON $REPO/scripts/generate_multi_slide_spatial_viewer.py \\"
        echo "    --input-dir \"$OUTPUT_DIR\" \\"
        echo "    --detection-glob \"${CELL_TYPE}_detections_classified.json\" \\"
        echo "    --group-field \"$VIEWER_GROUP_FIELD\" \\"
        echo "    --title \"$VIEWER_TITLE_ESC\" \\"
        echo "    --output \"${OUTPUT_DIR}/spatial_viewer.html\""
        echo ""
        echo "echo \"Spatial viewer saved to ${OUTPUT_DIR}/spatial_viewer.html\""
    } > "$VIEWER_SBATCH"
    chmod +x "$VIEWER_SBATCH"

    echo ""
    echo "Submitting viewer job (depends on $MAIN_JOB_ID)..."
    sbatch --dependency=afterok:"$MAIN_JOB_ID" "$VIEWER_SBATCH"
fi
