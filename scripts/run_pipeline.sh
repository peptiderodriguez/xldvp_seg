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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"

# Find xldvp_seg python: explicit env var > conda env detection > common paths
if [ -n "${XLDVP_PYTHON:-}" ] && [ -x "${XLDVP_PYTHON:-}" ]; then
    : # User-specified, use as-is
elif [ -n "${MKSEG_PYTHON:-}" ] && [ -x "${MKSEG_PYTHON:-}" ]; then
    XLDVP_PYTHON="$MKSEG_PYTHON"
else
    XLDVP_PYTHON=""
    # Try conda info to find env paths
    for _env_name in xldvp_seg mkseg; do
        _conda_prefix=$(conda info --envs 2>/dev/null | grep "^${_env_name} " | awk '{print $NF}')
        if [ -n "$_conda_prefix" ] && [ -x "$_conda_prefix/bin/python" ]; then
            XLDVP_PYTHON="$_conda_prefix/bin/python"
            break
        fi
    done
    # Fallback: search common paths (HOME may differ from actual conda location)
    if [ -z "$XLDVP_PYTHON" ]; then
        for _base in "$HOME" "$(getent passwd "$(whoami)" 2>/dev/null | cut -d: -f6)"; do
            for _env in xldvp_seg mkseg; do
                for _mgr in miniforge3 miniconda3 anaconda3 mambaforge; do
                    _p="$_base/$_mgr/envs/$_env/bin/python"
                    if [ -x "$_p" ]; then
                        XLDVP_PYTHON="$_p"
                        break 3
                    fi
                done
            done
        done
    fi
    # Last resort: check if current python has torch
    if [ -z "$XLDVP_PYTHON" ]; then
        _p=$(which python 2>/dev/null)
        if [ -n "$_p" ] && "$_p" -c "import torch" 2>/dev/null; then
            XLDVP_PYTHON="$_p"
        fi
    fi
    if [ -z "$XLDVP_PYTHON" ]; then
        echo "ERROR: Cannot find xldvp_seg/mkseg conda python."
        echo "Set XLDVP_PYTHON=/path/to/python or activate the conda env."
        exit 1
    fi
    echo "Auto-detected python: $XLDVP_PYTHON"
fi

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
    val=$("$XLDVP_PYTHON" -c "
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
# Validate YAML keys (catch typos and misplaced keys early)
# ---------------------------------------------------------------------------
_unknown_keys=$("$XLDVP_PYTHON" -c "
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}

VALID_TOP = {
    'name', 'czi_path', 'czi_dir', 'czi_glob', 'output_dir', 'cell_type',
    'cellpose_input_channels', 'channel_map', 'load_channels', 'channels',
    'all_channels', 'num_gpus', 'pixel_size_um', 'sample_fraction',
    'html_sample_fraction', 'photobleach_correction', 'resume_dir',
    'min_area_um', 'max_area_um', 'dedup_method', 'iou_threshold',
    'contour_processing', 'background_correction', 'dilation_um',
    'rdp_epsilon', 'bg_neighbors', 'correct_all_channels',
    'markers', 'slurm', 'sharding', 'downstream', 'scenes', 'scene_parallel',
    'spatial_network', 'spatial_viewer', 'spatialdata',
}
unknown = [k for k in cfg if k not in VALID_TOP]
if unknown:
    print(' '.join(unknown))
" "$CONFIG")

if [ -n "$_unknown_keys" ]; then
    echo "ERROR: Unknown YAML keys in config: $_unknown_keys" >&2
    echo "  Valid top-level keys: name, czi_path, czi_dir, output_dir, cell_type," >&2
    echo "    channel_map, markers, slurm, sharding, downstream, spatialdata, ..." >&2
    echo "  Common mistakes:" >&2
    echo "    - 'num_shards' should be under 'sharding:' (sharding.num_shards)" >&2
    echo "    - 'partition/cpus/gpus' should be under 'slurm:'" >&2
    echo "    - 'quality_filter/nuclei/html' should be under 'downstream:'" >&2
    exit 1
fi

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
# Default num_gpus from SLURM GPU count (e.g. "l40s:4" -> 4)
_slurm_gpus=$(read_yaml slurm.gpus "")
_default_ngpu=1
if [[ "$_slurm_gpus" =~ :([0-9]+)$ ]]; then
    _default_ngpu="${BASH_REMATCH[1]}"
elif [[ "$_slurm_gpus" =~ ^[0-9]+$ ]]; then
    _default_ngpu="$_slurm_gpus"
fi
NUM_GPUS=$(read_yaml num_gpus "$_default_ngpu")
MIN_AREA=$(read_yaml min_area_um "")
MAX_AREA=$(read_yaml max_area_um "")

# Channel map (wavelength/name-based channel specs, resolved at runtime)
# Supported YAML keys: channel_map.detect, channel_map.cyto, channel_map.nuc
CHANNEL_SPEC=$("$XLDVP_PYTHON" -c "
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
HTML_SAMPLE_FRACTION=$(read_yaml html_sample_fraction "")
RESUME_DIR=$(read_yaml resume_dir "")   # Explicit resume path — no auto-discovery

# Post-dedup processing (contour dilation + background correction)
BACKGROUND_CORRECTION=$(read_yaml background_correction true)
CONTOUR_PROCESSING=$(read_yaml contour_processing true)
DILATION_UM=$(read_yaml dilation_um "")
RDP_EPSILON=$(read_yaml rdp_epsilon "")
BG_NEIGHBORS=$(read_yaml bg_neighbors "")

# Deduplication method
DEDUP_METHOD=$(read_yaml dedup_method "")
IOU_THRESHOLD=$(read_yaml iou_threshold "")

# Multi-scene support (single CZI with multiple scenes)
# scenes: "0-9" or scenes: [0, 1, 2] in YAML
# scene_parallel: true (default) → array job, one task per scene
#                 false → sequential loop in one job
SCENES=$(read_yaml scenes "")
SCENE_PARALLEL=false  # only meaningful when scenes is set
SCENE_START=""
SCENE_END=""
if [[ -n "$SCENES" ]]; then
    SCENE_PARALLEL=$(read_yaml scene_parallel true)
    if [[ "$SCENES" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        SCENE_START="${BASH_REMATCH[1]}"
        SCENE_END="${BASH_REMATCH[2]}"
    else
        # Space-separated list from YAML array → derive range
        _first=$(echo "$SCENES" | awk '{print $1}')
        _last=$(echo "$SCENES" | awk '{print $NF}')
        SCENE_START="$_first"
        SCENE_END="$_last"
    fi
fi
MULTI_SCENE=false
if [[ -n "$SCENE_START" && -n "$SCENE_END" ]]; then
    MULTI_SCENE=true
fi

# Multi-node sharding
NUM_SHARDS=$(read_yaml sharding.num_shards 0)

# Downstream steps (each is a separate SLURM job with dependency)
DS_QUALITY_FILTER=$(read_yaml downstream.quality_filter.enabled false)
DS_QF_MIN_AREA=$(read_yaml downstream.quality_filter.min_area_um2 50)
DS_QF_MAX_AREA=$(read_yaml downstream.quality_filter.max_area_um2 2000)
DS_QF_MIN_SOLIDITY=$(read_yaml downstream.quality_filter.min_solidity 0.85)
DS_NUCLEI=$(read_yaml downstream.nuclei.enabled false)
DS_NUC_CHANNEL_SPEC=$(read_yaml downstream.nuclei.channel_spec "")
DS_HTML=$(read_yaml downstream.annotation_html.enabled false)
DS_HTML_MAX_SAMPLES=$(read_yaml downstream.annotation_html.max_samples 5000)
DS_HTML_DISPLAY_CHANNELS=$(read_yaml downstream.annotation_html.display_channels "")
DS_CLUSTERING=$(read_yaml downstream.clustering.enabled false)
DS_CLUSTERING_FEATURES=$(read_yaml downstream.clustering.feature_groups "morph")
DS_CLUSTERING_METHODS=$(read_yaml downstream.clustering.methods "both")
DS_CLUSTERING_RESOLUTION=$(read_yaml downstream.clustering.resolution "0.1")

# SLURM settings
PARTITION=$(read_yaml slurm.partition p.hpcl8)
CPUS=$(read_yaml slurm.cpus 24)
MEM_GB=$(read_yaml slurm.mem_gb 350)
# GPUS: the gres suffix (e.g. "l40s:4" or "rtx5000:2").  On GPU partitions we
# always request all 4 GPUs so SLURM doesn't place us on a non-GPU node.
# The Python code's --num-gpus controls how many GPUs it actually uses.
GPUS=$(read_yaml slurm.gpus "rtx5000:2")
# Ensure GPU gres always requests all 4 GPUs: replace a trailing :N with :4
# (keeps the GPU type, e.g. "l40s:4" instead of "l40s:2")
if [[ "$GPUS" =~ ^[a-zA-Z0-9_]+:[0-9]+$ ]]; then
    GPU_TYPE="${GPUS%:*}"
    GPUS="${GPU_TYPE}:4"
elif [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    GPUS="4"
fi
TIME=$(read_yaml slurm.time "3-00:00:00")
SLIDES_PER_JOB=$(read_yaml slurm.slides_per_job 1)
NUM_JOBS=$(read_yaml slurm.num_jobs 1)

# Markers (parsed as JSON list, supports both 'channel' and 'wavelength' keys)
MARKERS_JSON=$("$XLDVP_PYTHON" -c "
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
    # Usage: build_seg_cmd CZI_PATH OUTPUT_DIR [SCENE_VAR]
    # SCENE_VAR: if provided, adds --scene with this value (literal or shell var)
    local czi_arg="$1"
    local out_arg="$2"
    local scene_arg="${3:-}"
    local cmd="$XLDVP_PYTHON $REPO/run_segmentation.py"
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
    if [[ -n "$HTML_SAMPLE_FRACTION" ]]; then
        cmd+=" --html-sample-fraction $HTML_SAMPLE_FRACTION"
    fi
    # Dedup method
    if [[ -n "$DEDUP_METHOD" ]]; then
        cmd+=" --dedup-method $DEDUP_METHOD"
    fi
    if [[ -n "$IOU_THRESHOLD" ]]; then
        cmd+=" --iou-threshold $IOU_THRESHOLD"
    fi
    if [[ -n "$scene_arg" ]]; then
        cmd+=" --scene $scene_arg"
    fi
    echo "$cmd"
}

# ---------------------------------------------------------------------------
# Build comma-separated marker args for single classify_markers.py invocation
# ---------------------------------------------------------------------------
# Markers can use 'channel' (index) or 'wavelength' (resolved at runtime)
MARKER_USE_WAVELENGTH=$("$XLDVP_PYTHON" -c "
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

MARKER_CHANNELS=$("$XLDVP_PYTHON" -c "
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

MARKER_NAMES=$("$XLDVP_PYTHON" -c "
import json, sys
markers = json.loads(sys.argv[1])
print(','.join(m['name'] for m in markers))
" "$MARKERS_JSON" || echo "")

MARKER_METHOD=$("$XLDVP_PYTHON" -c "
import json, sys
markers = json.loads(sys.argv[1])
if not markers:
    sys.exit(0)
methods = set(m.get('method', 'snr') for m in markers)
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
# Pre-compute slide list (once, shared by SBATCH header + body + downstream)
# ---------------------------------------------------------------------------
_GEN_SLIDES=()
_N_SLIDES=0
if [[ "$MULTI_SLIDE" == "true" ]]; then
    mapfile -t _GEN_SLIDES < <(find "$CZI_DIR" -maxdepth 1 -name "$CZI_GLOB" -type f | sort)
    _N_SLIDES=${#_GEN_SLIDES[@]}
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
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --cpus-per-task=${CPUS}"
    echo "#SBATCH --mem=${MEM_GB}G"
    echo "#SBATCH --gres=gpu:${GPUS}"
    echo "#SBATCH --time=${TIME}"
    if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
        # Flat array: one task per (slide, scene) pair
        _N_SCENES=$(( SCENE_END - SCENE_START + 1 ))
        _FLAT_TOTAL=$(( _N_SLIDES * _N_SCENES ))
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.err"
        echo "#SBATCH --array=0-$((_FLAT_TOTAL - 1))"
    elif [[ "$MULTI_SLIDE" == "true" ]]; then
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.err"
        echo "#SBATCH --array=0-$((NUM_JOBS - 1))"
    elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_scene%a_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_scene%a_%j.err"
        echo "#SBATCH --array=${SCENE_START}-${SCENE_END}"
    else
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%j.err"
    fi

    echo ""
    echo "set -euo pipefail"
    echo ""
    echo "export PYTHONPATH=\"$REPO\""
    echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
    echo ""

    if [[ "$MULTI_SLIDE" == "true" ]]; then
        # Bake slide list into sbatch as a fixed array (no runtime glob)
        echo "ALL_SLIDES=("
        for _s in "${_GEN_SLIDES[@]}"; do
            echo "    \"$_s\""
        done
        echo ")"
        echo "TOTAL_SLIDES=${_N_SLIDES}"
        echo ""

        # Resume paths: only if explicitly set in YAML (no auto-discovery)
        echo "# Resume paths (only set if resume_dir explicitly configured in YAML)"
        echo "declare -A RESUME_PATHS"
        if [[ -n "$RESUME_DIR" ]]; then
            for _s in "${_GEN_SLIDES[@]}"; do
                _sname=$(basename "$_s" .czi)
                echo "RESUME_PATHS[$_sname]=\"$RESUME_DIR\""
            done
        fi
        echo ""

        echo "if [[ \$TOTAL_SLIDES -eq 0 ]]; then"
        echo "    echo \"Error: no slides found\""
        echo "    exit 1"
        echo "fi"
        echo ""

        # Multi-slide + multi-scene parallel: flat array over (slide, scene) pairs
        if [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            _N_SCENES=$(( SCENE_END - SCENE_START + 1 ))
            echo "N_SCENES=${_N_SCENES}"
            echo "SCENE_START=${SCENE_START}"
            echo "SLIDE_IDX=\$(( SLURM_ARRAY_TASK_ID / N_SCENES ))"
            echo "SCENE=\$(( SLURM_ARRAY_TASK_ID % N_SCENES + SCENE_START ))"
            echo "CZI_FILE=\"\${ALL_SLIDES[\$SLIDE_IDX]}\""
            echo "SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            echo "SLIDE_OUT=\"${OUTPUT_DIR}/\${SLIDE_NAME}/scene_\${SCENE}\""
            echo "mkdir -p \"\$SLIDE_OUT\""
            echo "echo \"=== Slide \$SLIDE_IDX scene \$SCENE: \$CZI_FILE ===\""
            echo ""
            echo "RESUME_FLAG=\"\""
            echo "if [[ -n \"\${RESUME_PATHS[\$SLIDE_NAME]:-}\" ]]; then RESUME_FLAG=\"--resume \${RESUME_PATHS[\$SLIDE_NAME]}\"; fi"
            echo "$(build_seg_cmd '${CZI_FILE}' '${SLIDE_OUT}' '${SCENE}') \$RESUME_FLAG"
            echo ""
            echo "# Step 2: Find detection JSON"
            echo "RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
            echo "DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
            echo "if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"
            if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
                echo "    echo \"  Exporting to SpatialData...\""
                sd_cmd="    \$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
                if [[ "$SPATIALDATA_SHAPES" == "true" ]]; then
                    sd_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
                else
                    sd_cmd+=" --no-shapes"
                fi
                echo "$sd_cmd"
            fi
            echo "else"
            echo "    echo \"WARNING: ${CELL_TYPE}_detections.json not found for slide \${SLIDE_NAME} scene \${SCENE}\""
            echo "fi"

        # Multi-slide + multi-scene sequential: scene loop inside each slide task
        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" ]]; then
            echo "# Compute slide range for this array task"
            echo "START=\$(( SLURM_ARRAY_TASK_ID * $SLIDES_PER_JOB ))"
            echo "END=\$(( START + $SLIDES_PER_JOB ))"
            echo "if [[ \$END -gt \$TOTAL_SLIDES ]]; then END=\$TOTAL_SLIDES; fi"
            echo ""
            echo "for (( i=START; i<END; i++ )); do"
            echo "    CZI_FILE=\"\${ALL_SLIDES[\$i]}\""
            echo "    SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            echo "    echo \"=== Processing slide \$((i+1))/\$TOTAL_SLIDES: \$CZI_FILE ===\""
            echo "    for SCENE in \$(seq ${SCENE_START} ${SCENE_END}); do"
            echo "        SLIDE_OUT=\"${OUTPUT_DIR}/\${SLIDE_NAME}/scene_\${SCENE}\""
            echo "        mkdir -p \"\$SLIDE_OUT\""
            echo "        echo \"  Scene \${SCENE}\""
            echo "        RESUME_FLAG=\"\""
            echo "        if [[ -n \"\${RESUME_PATHS[\$SLIDE_NAME]:-}\" ]]; then RESUME_FLAG=\"--resume \${RESUME_PATHS[\$SLIDE_NAME]}\"; fi"
            echo "        $(build_seg_cmd '${CZI_FILE}' '${SLIDE_OUT}' '${SCENE}') \$RESUME_FLAG"
            echo ""
            echo "        # Step 2: Find detection JSON"
            echo "        RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
            echo "        DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
            echo "        if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"
            if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
                echo "            echo \"  Exporting to SpatialData...\""
                sd_cmd="            \$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
                if [[ "$SPATIALDATA_SHAPES" == "true" ]]; then
                    sd_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
                else
                    sd_cmd+=" --no-shapes"
                fi
                echo "$sd_cmd"
            fi
            echo "        else"
            echo "            echo \"WARNING: ${CELL_TYPE}_detections.json not found for slide \${SLIDE_NAME} scene \${SCENE}\""
            echo "        fi"
            echo "    done"
            echo "done"

        # Multi-slide only (no scenes — existing behavior)
        else
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
            echo "    # Step 1: Segmentation (resume from pre-computed path if available)"
            echo "    RESUME_FLAG=\"\""
            echo "    if [[ -n \"\${RESUME_PATHS[\$SLIDE_NAME]:-}\" ]]; then RESUME_FLAG=\"--resume \${RESUME_PATHS[\$SLIDE_NAME]}\"; fi"
            echo "    $(build_seg_cmd '${CZI_FILE}' '${SLIDE_OUT}') \$RESUME_FLAG"
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
                    echo "        \$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-wavelength \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --czi-path \"\$CZI_FILE\" --output-dir \"\$RUN_DIR\"${classify_extra}"
                else
                    echo "        \$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --output-dir \"\$RUN_DIR\"${classify_extra}"
                fi
                echo "        # Discover classified output (may be _filtered_classified or _classified)"
                echo "        for _cf in \"\${RUN_DIR}${CELL_TYPE}_detections_filtered_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\"; do"
                echo "            if [[ -f \"\$_cf\" ]]; then DET_JSON=\"\$_cf\"; break; fi"
                echo "        done"
            fi

            if [[ "$SPATIAL_ENABLED" == "true" ]]; then
                echo ""
                echo "        # Step 3: Spatial analysis"
                echo "        echo \"  Running spatial analysis...\""
                echo "        \$XLDVP_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$RUN_DIR\" --spatial-network --marker-filter \"$SPATIAL_FILTER\" --max-edge-distance \"$SPATIAL_EDGE\" --min-component-cells \"$SPATIAL_MIN_COMP\" --pixel-size \"$PIXEL_SIZE\""
            fi

            if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
                echo ""
                echo "        # SpatialData export"
                echo "        echo \"  Exporting to SpatialData...\""
                sd_cmd="        \$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
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
        fi

    elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
        # Multi-scene parallel: one array task per scene
        echo "SCENE=\${SLURM_ARRAY_TASK_ID}"
        echo "SLIDE_OUT=\"${OUTPUT_DIR}/scene_\${SCENE}\""
        echo "mkdir -p \"\$SLIDE_OUT\""
        echo ""
        echo "# Step 1: Segmentation (scene \${SCENE})"
        if [[ -n "$RESUME_DIR" ]]; then
            echo "$(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}' '${SCENE}') --resume \"$RESUME_DIR\""
        else
            echo "$(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}' '${SCENE}')"
        fi
        echo ""
        echo "# Step 2: Find detection JSON"
        echo "RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
        echo "DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"
        if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
            echo "    echo \"Exporting to SpatialData...\""
            sd_cmd="    \$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
            if [[ "$SPATIALDATA_SHAPES" == "true" ]]; then
                sd_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
            else
                sd_cmd+=" --no-shapes"
            fi
            echo "$sd_cmd"
        fi
        echo "else"
        echo "    echo \"WARNING: ${CELL_TYPE}_detections.json not found for scene \${SCENE}\""
        echo "fi"
    elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" ]]; then
        # Multi-scene sequential: loop over scenes in one job
        echo "for SCENE in \$(seq ${SCENE_START} ${SCENE_END}); do"
        echo "    SLIDE_OUT=\"${OUTPUT_DIR}/scene_\${SCENE}\""
        echo "    mkdir -p \"\$SLIDE_OUT\""
        echo "    echo \"=== Scene \${SCENE} ===\""
        echo ""
        echo "    # Step 1: Segmentation (scene \${SCENE})"
        if [[ -n "$RESUME_DIR" ]]; then
            echo "    $(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}' '${SCENE}') --resume \"$RESUME_DIR\""
        else
            echo "    $(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}' '${SCENE}')"
        fi
        echo ""
        echo "    # Step 2: Find detection JSON"
        echo "    RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
        echo "    DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "    if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"
        if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
            echo "        echo \"Exporting to SpatialData...\""
            sd_cmd="        \$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
            if [[ "$SPATIALDATA_SHAPES" == "true" ]]; then
                sd_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
            else
                sd_cmd+=" --no-shapes"
            fi
            echo "$sd_cmd"
        fi
        echo "    else"
        echo "        echo \"WARNING: ${CELL_TYPE}_detections.json not found for scene \${SCENE}\""
        echo "    fi"
        echo "done"
    else
        # Single-slide job
        echo "SLIDE_OUT=\"${OUTPUT_DIR}\""
        echo "mkdir -p \"\$SLIDE_OUT\""
        echo ""
        echo "# Step 1: Segmentation"
        if [[ -n "$RESUME_DIR" ]]; then
            echo "$(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}') --resume \"$RESUME_DIR\""
        else
            echo "$(build_seg_cmd "$CZI_PATH" '${SLIDE_OUT}')"
        fi
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
                echo "    \$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-wavelength \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --czi-path \"$CZI_PATH\" --output-dir \"\$RUN_DIR\"${classify_extra}"
            else
                echo "    \$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --output-dir \"\$RUN_DIR\"${classify_extra}"
            fi
            echo "    # Discover classified output (may be _filtered_classified or _classified)"
            echo "    for _cf in \"\${RUN_DIR}${CELL_TYPE}_detections_filtered_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\"; do"
            echo "        if [[ -f \"\$_cf\" ]]; then DET_JSON=\"\$_cf\"; break; fi"
            echo "    done"
        fi

        if [[ "$SPATIAL_ENABLED" == "true" ]]; then
            echo ""
            echo "    # Step 3: Spatial analysis"
            echo "    echo \"Running spatial analysis...\""
            echo "    \$XLDVP_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$RUN_DIR\" --spatial-network --marker-filter \"$SPATIAL_FILTER\" --max-edge-distance \"$SPATIAL_EDGE\" --min-component-cells \"$SPATIAL_MIN_COMP\" --pixel-size \"$PIXEL_SIZE\""
        fi

        if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
            echo ""
            echo "    # SpatialData export"
            echo "    echo \"Exporting to SpatialData...\""
            sd_cmd="    \$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
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
            echo "    \$XLDVP_PYTHON $REPO/scripts/generate_multi_slide_spatial_viewer.py --input-dir \"\$SLIDE_OUT\" --group-field \"$VIEWER_GROUP_FIELD\" --title \"$VIEWER_TITLE_ESC\" --output \"\${SLIDE_OUT}/spatial_viewer.html\""
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
if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" ]]; then
    _strategy="parallel (flat array)"
    if [[ "$SCENE_PARALLEL" == "false" ]]; then _strategy="sequential (scenes loop per slide)"; fi
    echo "Mode:     multi-slide + multi-scene ${_strategy} (scenes ${SCENE_START}-${SCENE_END})"
    echo "CZI dir:  $CZI_DIR"
    echo "Glob:     $CZI_GLOB"
elif [[ "$MULTI_SLIDE" == "true" ]]; then
    echo "Mode:     multi-slide array ($NUM_JOBS jobs, $SLIDES_PER_JOB slides/job)"
    echo "CZI dir:  $CZI_DIR"
    echo "Glob:     $CZI_GLOB"
elif [[ "$MULTI_SCENE" == "true" ]]; then
    _strategy="parallel (array)"
    if [[ "$SCENE_PARALLEL" == "false" ]]; then _strategy="sequential (loop)"; fi
    echo "Mode:     multi-scene ${_strategy} (scenes ${SCENE_START}-${SCENE_END})"
    echo "CZI:      $CZI_PATH"
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

# ---------------------------------------------------------------------------
# Multi-node sharding: generate shard array + merge job
# ---------------------------------------------------------------------------
# Warn if sharding + multi-slide or multi-scene (not supported)
if [[ "$NUM_SHARDS" -gt 1 && ( "$MULTI_SLIDE" == "true" || "$MULTI_SCENE" == "true" ) ]]; then
    echo "WARNING: sharding.num_shards=$NUM_SHARDS ignored in multi-slide/multi-scene mode." >&2
    NUM_SHARDS=0
fi

# Block unsupported downstream jobs in multi-slide or multi-scene mode
# Nuclei counting and annotation HTML ARE supported
if [[ "$MULTI_SLIDE" == "true" || "$MULTI_SCENE" == "true" ]]; then
    for _ds_flag in "$DS_QUALITY_FILTER" "$DS_CLUSTERING"; do
        if [[ "$_ds_flag" == "true" ]]; then
            echo "Error: quality_filter/clustering downstream jobs not yet supported in multi-slide/multi-scene mode." >&2
            echo "Run downstream steps manually after detection completes." >&2
            exit 1
        fi
    done
fi

if [[ "$NUM_SHARDS" -gt 1 && "$MULTI_SLIDE" == "false" ]]; then
    echo "Multi-node sharding: $NUM_SHARDS shards"

    # Pre-create shared run directory so all shards write to the same place
    _slide_name=$(basename "$CZI_PATH" .czi)
    _slide_dir="${OUTPUT_DIR}/${_slide_name}"
    mkdir -p "$_slide_dir"
    SHARED_RUN_DIR="${_slide_dir}/${_slide_name}_$(date +%Y%m%d_%H%M%S)_100pct"
    mkdir -p "$SHARED_RUN_DIR"
    echo "  Shared run dir: $SHARED_RUN_DIR"

    # Shard detection array — all shards --resume into the shared dir
    SHARD_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_shards_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_shard"
        echo "#SBATCH --partition=${PARTITION}"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --cpus-per-task=${CPUS}"
        echo "#SBATCH --mem=${MEM_GB}G"
        echo "#SBATCH --gres=gpu:${GPUS}"
        echo "#SBATCH --time=${TIME}"
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_s%a_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_s%a_%j.err"
        echo "#SBATCH --array=0-$((NUM_SHARDS - 1))"
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        echo ""
        echo "# All shards write to the same shared directory"
        echo "$(build_seg_cmd "$CZI_PATH" "$_slide_dir") --resume \"$SHARED_RUN_DIR\" --tile-shard \${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}"
    } > "$SHARD_SBATCH"
    chmod +x "$SHARD_SBATCH"

    SHARD_OUTPUT=$(sbatch "$SHARD_SBATCH")
    SHARD_JOB_ID=$(echo "$SHARD_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Shard array job: $SHARD_JOB_ID (${NUM_SHARDS} tasks)"

    # Merge job — uses the known shared dir (no ls -td discovery)
    # Only needs 1 GPU for post-dedup processing
    MERGE_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_merge_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_merge"
        echo "#SBATCH --partition=${PARTITION}"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --cpus-per-task=${CPUS}"
        echo "#SBATCH --mem=${MEM_GB}G"
        echo "#SBATCH --gres=gpu:${GPU_TYPE}:1"
        echo "#SBATCH --time=${TIME}"
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_merge_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_merge_%j.err"
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        echo ""
        echo "echo \"Merging shards in $SHARED_RUN_DIR\""
        echo ""
        echo "# CRITICAL: --merge-shards REQUIRES --resume (without it, argparse errors silently)"
        echo "$(build_seg_cmd "$CZI_PATH" "$_slide_dir") --resume \"$SHARED_RUN_DIR\" --merge-shards"
    } > "$MERGE_SBATCH"
    chmod +x "$MERGE_SBATCH"

    MERGE_OUTPUT=$(sbatch --dependency=afterok:"$SHARD_JOB_ID" "$MERGE_SBATCH")
    MAIN_JOB_ID=$(echo "$MERGE_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Merge job: $MAIN_JOB_ID (dep: $SHARD_JOB_ID)"

    # For downstream jobs, set the known run dir
    KNOWN_RUN_DIR="$SHARED_RUN_DIR"

else
    # Standard single-job submission
    KNOWN_RUN_DIR=""  # will be discovered at runtime
    MAIN_JOB_OUTPUT=$(sbatch "$SBATCH_FILE") || { echo "Error: sbatch submission failed" >&2; exit 1; }
    echo "$MAIN_JOB_OUTPUT"
    MAIN_JOB_ID=$(echo "$MAIN_JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+') || true
fi

if [[ -z "$MAIN_JOB_ID" ]]; then
    echo "Error: could not parse job ID" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Downstream dependent jobs (quality filter, nuclei, HTML, clustering)
# Each is a separate lightweight SLURM job with correct dependency.
# This eliminates the need to write manual sbatch scripts.
# ---------------------------------------------------------------------------
DETECT_JOB_ID="$MAIN_JOB_ID"
LAST_DET_JSON="\${RUN_DIR}${CELL_TYPE}_detections.json"

# For single-slide: determine CZI path for downstream steps
DS_CZI_PATH="${CZI_PATH}"

# Quality filter (CPU, depends on detection)
if [[ "$DS_QUALITY_FILTER" == "true" ]]; then
    QF_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_qf_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_qf"
        echo "#SBATCH --partition=p.hpcl8"
        echo "#SBATCH --cpus-per-task=4"
        echo "#SBATCH --mem=32G"
        echo "#SBATCH --time=00:30:00"
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_qf_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_qf_%j.err"
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        if [[ -n "$KNOWN_RUN_DIR" ]]; then
            echo "RUN_DIR=\"${KNOWN_RUN_DIR}/\""
        else
            echo "RUN_DIR=\$(ls -td \"${OUTPUT_DIR}\"/*/  2>/dev/null | head -1)"
        fi
        echo "DET=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "if [[ -z \"\$RUN_DIR\" || ! -f \"\$DET\" ]]; then echo 'ERROR: Detection output not found'; exit 1; fi"
        echo "echo \"=== \$(date): Quality filter ===\""
        echo "\$XLDVP_PYTHON $REPO/scripts/quality_filter_detections.py \\"
        echo "    --detections \"\$DET\" \\"
        echo "    --output \"\${RUN_DIR}${CELL_TYPE}_detections_filtered.json\" \\"
        echo "    --min-area-um2 $DS_QF_MIN_AREA --max-area-um2 $DS_QF_MAX_AREA --min-solidity $DS_QF_MIN_SOLIDITY"
        echo "echo \"=== \$(date): Done ===\""
    } > "$QF_SBATCH"
    chmod +x "$QF_SBATCH"
    QF_OUTPUT=$(sbatch --dependency=afterok:"$DETECT_JOB_ID" "$QF_SBATCH")
    QF_JOB_ID=$(echo "$QF_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Quality filter job: $QF_JOB_ID (dep: $DETECT_JOB_ID)"
    LAST_DET_JSON="\${RUN_DIR}${CELL_TYPE}_detections_filtered.json"
    DETECT_JOB_ID="$QF_JOB_ID"
fi

# Marker classification on filtered output (CPU, depends on quality filter)
# If quality filter ran AND markers are configured, re-run markers on the filtered
# detections so marker classes reflect filtered (debris-free) data.
if [[ "$DS_QUALITY_FILTER" == "true" && -n "$MARKER_CHANNELS" ]]; then
    MK_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_markers_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_markers"
        echo "#SBATCH --partition=p.hpcl8"
        echo "#SBATCH --cpus-per-task=24"
        echo "#SBATCH --mem=64G"
        echo "#SBATCH --time=1:00:00"
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_markers_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_markers_%j.err"
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        if [[ -n "$KNOWN_RUN_DIR" ]]; then
            echo "RUN_DIR=\"${KNOWN_RUN_DIR}/\""
        else
            echo "RUN_DIR=\$(ls -td \"${OUTPUT_DIR}\"/*/  2>/dev/null | head -1)"
        fi
        echo "FILTERED=\"\${RUN_DIR}${CELL_TYPE}_detections_filtered.json\""
        echo "if [[ -z \"\$RUN_DIR\" || ! -f \"\$FILTERED\" ]]; then echo 'ERROR: Filtered detections not found'; exit 1; fi"
        echo "echo \"=== \$(date): Marker classification on filtered detections ===\""
        classify_extra=""
        if [[ "$CORRECT_ALL_CHANNELS" == "true" ]]; then
            classify_extra+=" --correct-all-channels"
        fi
        if [[ "$MARKER_USE_WAVELENGTH" == "true" ]]; then
            echo "\$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$FILTERED\" --marker-wavelength \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --czi-path \"$CZI_PATH\" --output-dir \"\$RUN_DIR\"${classify_extra}"
        else
            echo "\$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$FILTERED\" --marker-channel \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --output-dir \"\$RUN_DIR\"${classify_extra}"
        fi
        echo "echo \"=== \$(date): Done ===\""
    } > "$MK_SBATCH"
    chmod +x "$MK_SBATCH"
    MK_OUTPUT=$(sbatch --dependency=afterok:"$QF_JOB_ID" "$MK_SBATCH")
    MK_JOB_ID=$(echo "$MK_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Markers (on filtered) job: $MK_JOB_ID (dep: $QF_JOB_ID)"
    DETECT_JOB_ID="$MK_JOB_ID"
fi

# Nuclear counting (GPU, depends on detection — uses UNFILTERED detections)
# Strategy mirrors detection: multi-scene parallel → array, sequential → loop
if [[ "$DS_NUCLEI" == "true" ]]; then
    NUC_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_nuc_$$.sbatch"

    # Build the per-scene nuclei command (reused by all modes)
    # Args: indent, scene_flag, run_dir_expr, czi_path_expr
    _nuc_cmd_body() {
        local indent="$1"  # "" or "    " for loop indent
        local scene_flag="$2"  # "" or "--scene \$SCENE"
        local run_dir_expr="$3"  # expression to find RUN_DIR
        local czi_expr="${4:-$DS_CZI_PATH}"  # CZI path (literal or shell var)
        echo "${indent}RUN_DIR=${run_dir_expr}"
        echo "${indent}if [[ -z \"\$RUN_DIR\" || ! -f \"\${RUN_DIR}${CELL_TYPE}_detections.json\" ]]; then echo 'ERROR: Detection output not found'; exit 1; fi"
        echo "${indent}echo \"=== \$(date): Nuclear counting ===\""
        local nuc_cmd="${indent}\$XLDVP_PYTHON $REPO/scripts/count_nuclei_per_cell.py"
        nuc_cmd+=" --detections \"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        nuc_cmd+=" --czi-path \"$czi_expr\""
        nuc_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
        if [[ -n "$DS_NUC_CHANNEL_SPEC" ]]; then
            nuc_cmd+=" --channel-spec \"$DS_NUC_CHANNEL_SPEC\""
        fi
        if [[ -n "$scene_flag" ]]; then
            nuc_cmd+=" $scene_flag"
        fi
        nuc_cmd+=" --output \"\${RUN_DIR}${CELL_TYPE}_detections_nuclei.json\""
        echo "$nuc_cmd"
        echo "${indent}echo \"=== \$(date): Done ===\""
    }

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_nuc"
        echo "#SBATCH --partition=p.hpcl93"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --cpus-per-task=64"
        echo "#SBATCH --mem=300G"
        echo "#SBATCH --gres=gpu:l40s:1"
        echo "#SBATCH --time=6:00:00"
        # Array header: mirrors detection strategy exactly
        if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            # Flat array: one task per (slide, scene)
            _N_SCENES=$(( SCENE_END - SCENE_START + 1 ))
            _FLAT_TOTAL=$(( ${#_GEN_SLIDES[@]} * _N_SCENES ))
            echo "#SBATCH --array=0-$((_FLAT_TOTAL - 1))"
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_nuc_%A_%a.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_nuc_%A_%a.err"
        elif [[ "$MULTI_SLIDE" == "true" ]]; then
            # Multi-slide (with or without sequential scenes): no array, loops internally
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_nuc_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_nuc_%j.err"
        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            echo "#SBATCH --array=${SCENE_START}-${SCENE_END}"
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_nuc_scene%a_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_nuc_scene%a_%j.err"
        else
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_nuc_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_nuc_%j.err"
        fi
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""

        if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            # Multi-slide + multi-scene parallel: flat array, same as detection
            _N_SCENES=$(( SCENE_END - SCENE_START + 1 ))
            echo "# Baked slide list (same as detection job)"
            echo "ALL_SLIDES=("
            for _s in "${_GEN_SLIDES[@]}"; do echo "    \"$_s\""; done
            echo ")"
            echo "N_SCENES=${_N_SCENES}"
            echo "SCENE_START=${SCENE_START}"
            echo "SLIDE_IDX=\$(( SLURM_ARRAY_TASK_ID / N_SCENES ))"
            echo "SCENE=\$(( SLURM_ARRAY_TASK_ID % N_SCENES + SCENE_START ))"
            echo "CZI_FILE=\"\${ALL_SLIDES[\$SLIDE_IDX]}\""
            echo "SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            _nuc_cmd_body "" "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/\${SLIDE_NAME}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)" "\$CZI_FILE"

        elif [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" ]]; then
            # Multi-slide + multi-scene sequential: loop slides and scenes
            echo "ALL_SLIDES=("
            for _s in "${_GEN_SLIDES[@]}"; do echo "    \"$_s\""; done
            echo ")"
            echo "for CZI_FILE in \"\${ALL_SLIDES[@]}\"; do"
            echo "    SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            echo "    for SCENE in \$(seq ${SCENE_START} ${SCENE_END}); do"
            echo "        echo \"=== \$SLIDE_NAME scene \$SCENE ===\""
            _nuc_cmd_body "        " "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/\${SLIDE_NAME}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)" "\$CZI_FILE"
            echo "    done"
            echo "done"

        elif [[ "$MULTI_SLIDE" == "true" ]]; then
            # Multi-slide only (no scenes): loop over slides
            echo "ALL_SLIDES=("
            for _s in "${_GEN_SLIDES[@]}"; do echo "    \"$_s\""; done
            echo ")"
            echo "for CZI_FILE in \"\${ALL_SLIDES[@]}\"; do"
            echo "    SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            echo "    echo \"=== \$SLIDE_NAME ===\""
            _nuc_cmd_body "    " "" "\$(ls -td \"${OUTPUT_DIR}/\${SLIDE_NAME}\"/*/  2>/dev/null | head -1)" "\$CZI_FILE"
            echo "done"

        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            # Single-slide + multi-scene parallel
            echo "SCENE=\${SLURM_ARRAY_TASK_ID}"
            _nuc_cmd_body "" "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)"
        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" ]]; then
            # Single-slide + multi-scene sequential
            echo "for SCENE in \$(seq ${SCENE_START} ${SCENE_END}); do"
            echo "    echo \"=== Scene \${SCENE} ===\""
            _nuc_cmd_body "    " "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)"
            echo "done"
        else
            # Single-scene (original behavior)
            if [[ -n "$KNOWN_RUN_DIR" ]]; then
                _nuc_cmd_body "" "" "\"${KNOWN_RUN_DIR}/\""
            else
                _nuc_cmd_body "" "" "\$(ls -td \"${OUTPUT_DIR}\"/*/  2>/dev/null | head -1)"
            fi
        fi
    } > "$NUC_SBATCH"
    chmod +x "$NUC_SBATCH"
    NUC_OUTPUT=$(sbatch --dependency=afterok:"$MAIN_JOB_ID" "$NUC_SBATCH")
    NUC_JOB_ID=$(echo "$NUC_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Nuclear counting job: $NUC_JOB_ID (dep: $MAIN_JOB_ID)"
fi

# Annotation HTML (CPU, depends on markers if markers ran, else detection)
# Strategy mirrors detection: multi-scene parallel → array, sequential → loop
if [[ "$DS_HTML" == "true" ]]; then
    HTML_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_html_$$.sbatch"
    HTML_DEP_ID="$DETECT_JOB_ID"

    # Per-item HTML command (reused by all modes)
    # Args: indent, scene_flag, run_dir_expr, czi_path_expr
    _html_cmd_body() {
        local indent="$1"
        local scene_flag="$2"
        local run_dir_expr="$3"
        local czi_expr="${4:-$DS_CZI_PATH}"
        echo "${indent}RUN_DIR=${run_dir_expr}"
        echo "${indent}if [[ -z \"\$RUN_DIR\" ]]; then echo 'ERROR: Run directory not found'; exit 1; fi"
        echo "${indent}DET=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "${indent}for _f in \"\${RUN_DIR}${CELL_TYPE}_detections_filtered_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_filtered.json\"; do"
        echo "${indent}    if [[ -f \"\$_f\" ]]; then DET=\"\$_f\"; break; fi"
        echo "${indent}done"
        echo "${indent}echo \"=== \$(date): Annotation HTML from \$DET ===\""
        local html_cmd="${indent}\$XLDVP_PYTHON $REPO/scripts/regenerate_html.py"
        html_cmd+=" --detections \"\$DET\""
        html_cmd+=" --output-dir \"\$RUN_DIR\""
        html_cmd+=" --czi-path \"$czi_expr\""
        if [[ -n "$DS_HTML_DISPLAY_CHANNELS" ]]; then
            html_cmd+=" --display-channels \"$DS_HTML_DISPLAY_CHANNELS\""
        fi
        if [[ -n "$scene_flag" ]]; then
            html_cmd+=" $scene_flag"
        fi
        html_cmd+=" --dashed-contour"
        html_cmd+=" --max-samples $DS_HTML_MAX_SAMPLES"
        html_cmd+=" --html-dir \"\${RUN_DIR}html_annotation\""
        echo "$html_cmd"
        echo "${indent}echo \"=== \$(date): Done ===\""
    }

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_html"
        echo "#SBATCH --partition=p.hpcl8"
        echo "#SBATCH --cpus-per-task=8"
        echo "#SBATCH --mem=64G"
        echo "#SBATCH --time=2:00:00"
        # Array header mirrors detection strategy
        if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            _N_SCENES=$(( SCENE_END - SCENE_START + 1 ))
            _FLAT_TOTAL=$(( _N_SLIDES * _N_SCENES ))
            echo "#SBATCH --array=0-$((_FLAT_TOTAL - 1))"
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_html_%A_%a.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_html_%A_%a.err"
        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            echo "#SBATCH --array=${SCENE_START}-${SCENE_END}"
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_html_scene%a_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_html_scene%a_%j.err"
        else
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_html_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_html_%j.err"
        fi
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""

        if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            _N_SCENES=$(( SCENE_END - SCENE_START + 1 ))
            echo "ALL_SLIDES=("
            for _s in "${_GEN_SLIDES[@]}"; do echo "    \"$_s\""; done
            echo ")"
            echo "N_SCENES=${_N_SCENES}"
            echo "SCENE_START=${SCENE_START}"
            echo "SLIDE_IDX=\$(( SLURM_ARRAY_TASK_ID / N_SCENES ))"
            echo "SCENE=\$(( SLURM_ARRAY_TASK_ID % N_SCENES + SCENE_START ))"
            echo "CZI_FILE=\"\${ALL_SLIDES[\$SLIDE_IDX]}\""
            echo "SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            _html_cmd_body "" "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/\${SLIDE_NAME}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)" "\$CZI_FILE"

        elif [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" ]]; then
            echo "ALL_SLIDES=("
            for _s in "${_GEN_SLIDES[@]}"; do echo "    \"$_s\""; done
            echo ")"
            echo "for CZI_FILE in \"\${ALL_SLIDES[@]}\"; do"
            echo "    SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            echo "    for SCENE in \$(seq ${SCENE_START} ${SCENE_END}); do"
            echo "        echo \"=== \$SLIDE_NAME scene \$SCENE ===\""
            _html_cmd_body "        " "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/\${SLIDE_NAME}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)" "\$CZI_FILE"
            echo "    done"
            echo "done"

        elif [[ "$MULTI_SLIDE" == "true" ]]; then
            echo "ALL_SLIDES=("
            for _s in "${_GEN_SLIDES[@]}"; do echo "    \"$_s\""; done
            echo ")"
            echo "for CZI_FILE in \"\${ALL_SLIDES[@]}\"; do"
            echo "    SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
            echo "    echo \"=== \$SLIDE_NAME ===\""
            _html_cmd_body "    " "" "\$(ls -td \"${OUTPUT_DIR}/\${SLIDE_NAME}\"/*/  2>/dev/null | head -1)" "\$CZI_FILE"
            echo "done"

        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "true" ]]; then
            echo "SCENE=\${SLURM_ARRAY_TASK_ID}"
            _html_cmd_body "" "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)"

        elif [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" ]]; then
            echo "for SCENE in \$(seq ${SCENE_START} ${SCENE_END}); do"
            echo "    echo \"=== Scene \${SCENE} ===\""
            _html_cmd_body "    " "--scene \$SCENE" "\$(ls -td \"${OUTPUT_DIR}/scene_\${SCENE}\"/*/  2>/dev/null | head -1)"
            echo "done"

        else
            # Single-scene (original behavior)
            if [[ -n "$KNOWN_RUN_DIR" ]]; then
                _html_cmd_body "" "" "\"${KNOWN_RUN_DIR}/\""
            else
                _html_cmd_body "" "" "\$(ls -td \"${OUTPUT_DIR}\"/*/  2>/dev/null | head -1)"
            fi
        fi
    } > "$HTML_SBATCH"
    chmod +x "$HTML_SBATCH"
    HTML_OUTPUT=$(sbatch --dependency=afterok:"$HTML_DEP_ID" "$HTML_SBATCH")
    HTML_JOB_ID=$(echo "$HTML_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Annotation HTML job: $HTML_JOB_ID (dep: $HTML_DEP_ID)"
fi

# Clustering (CPU, depends on markers)
if [[ "$DS_CLUSTERING" == "true" ]]; then
    CLUST_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_cluster_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_cluster"
        echo "#SBATCH --partition=p.hpcl8"
        echo "#SBATCH --cpus-per-task=24"
        echo "#SBATCH --mem=200G"
        echo "#SBATCH --time=2:00:00"
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_cluster_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_cluster_%j.err"
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        if [[ -n "$KNOWN_RUN_DIR" ]]; then
            echo "RUN_DIR=\"${KNOWN_RUN_DIR}/\""
        else
            echo "RUN_DIR=\$(ls -td \"${OUTPUT_DIR}\"/*/  2>/dev/null | head -1)"
        fi
        echo "DET=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "if [[ -z \"\$RUN_DIR\" || ! -f \"\$DET\" ]]; then echo 'ERROR: Detection output not found'; exit 1; fi"
        echo "for _f in \"\${RUN_DIR}${CELL_TYPE}_detections_filtered_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_filtered.json\"; do"
        echo "    if [[ -f \"\$_f\" ]]; then DET=\"\$_f\"; break; fi"
        echo "done"
        echo "echo \"=== \$(date): Clustering ===\""
        echo "\$XLDVP_PYTHON $REPO/scripts/cluster_by_features.py \\"
        echo "    --detections \"\$DET\" \\"
        echo "    --output-dir \"\${RUN_DIR}clustering\" \\"
        echo "    --feature-groups \"$DS_CLUSTERING_FEATURES\" \\"
        echo "    --methods \"$DS_CLUSTERING_METHODS\" \\"
        echo "    --clustering leiden --resolution $DS_CLUSTERING_RESOLUTION"
        echo "echo \"=== \$(date): Done ===\""
    } > "$CLUST_SBATCH"
    chmod +x "$CLUST_SBATCH"
    CLUST_OUTPUT=$(sbatch --dependency=afterok:"$DETECT_JOB_ID" "$CLUST_SBATCH")
    CLUST_JOB_ID=$(echo "$CLUST_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Clustering job: $CLUST_JOB_ID (dep: $DETECT_JOB_ID)"
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
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        echo ""
        echo "echo \"Generating multi-slide spatial viewer...\""
        echo "\$XLDVP_PYTHON $REPO/scripts/generate_multi_slide_spatial_viewer.py \\"
        echo "    --input-dir \"$OUTPUT_DIR\" \\"
        echo "    --detection-glob \"${CELL_TYPE}_detections*classified.json\" \\"
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
