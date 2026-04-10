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
# Validate YAML values against shell injection
# ---------------------------------------------------------------------------
validate_yaml_value() {
    local val="$1" name="$2"
    if [[ "$val" =~ [\`\$\(\)\;\|\&\>\<\!\\] ]] || [[ "$val" == *$'\n'* ]]; then
        echo "ERROR: Unsafe character in $name. Aborting." >&2
        exit 1
    fi
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

# Built-in marker SNR classification (format: "SMA:1,CD31:3")
MARKER_SNR_CHANNELS=$(read_yaml marker_snr_channels "")

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
# GPU gres: use exactly what the YAML specifies (e.g. "l40s:4" or "rtx5000:2").
# Default depends on partition: p.hpcl93 has 4x L40S, p.hpcl8 has 2x RTX 5000.
_default_gpus="l40s:4"
if [[ "$PARTITION" == "p.hpcl8" ]]; then _default_gpus="rtx5000:2"; fi
GPUS=$(read_yaml slurm.gpus "$_default_gpus")
# Extract GPU type for merge job gres (needs type:count format)
GPU_TYPE=""
if [[ "$GPUS" =~ ^([a-zA-Z0-9_]+):([0-9]+)$ ]]; then
    GPU_TYPE="${BASH_REMATCH[1]}"
fi
TIME=$(read_yaml slurm.time "3-00:00:00")

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
# Validate all YAML-sourced values that are interpolated into shell commands
# ---------------------------------------------------------------------------
# Core pipeline values
validate_yaml_value "$NAME" "name"
validate_yaml_value "$CZI_PATH" "czi_path"
validate_yaml_value "$CZI_DIR" "czi_dir"
validate_yaml_value "$CZI_GLOB" "czi_glob"
validate_yaml_value "$OUTPUT_DIR" "output_dir"
validate_yaml_value "$CELL_TYPE" "cell_type"
validate_yaml_value "$CP_CHANNELS" "cellpose_input_channels"
validate_yaml_value "$LOAD_CHANNELS" "load_channels"
validate_yaml_value "$NUM_GPUS" "num_gpus"
validate_yaml_value "$MIN_AREA" "min_area_um"
validate_yaml_value "$MAX_AREA" "max_area_um"
validate_yaml_value "$SAMPLE_FRACTION" "sample_fraction"
validate_yaml_value "$HTML_SAMPLE_FRACTION" "html_sample_fraction"
validate_yaml_value "$CHANNEL_SPEC" "channel_map"
validate_yaml_value "$RESUME_DIR" "resume_dir"
validate_yaml_value "$MARKER_SNR_CHANNELS" "marker_snr_channels"

# Post-dedup processing
validate_yaml_value "$DILATION_UM" "dilation_um"
validate_yaml_value "$RDP_EPSILON" "rdp_epsilon"
validate_yaml_value "$BG_NEIGHBORS" "bg_neighbors"
validate_yaml_value "$DEDUP_METHOD" "dedup_method"
validate_yaml_value "$IOU_THRESHOLD" "iou_threshold"

# SLURM settings
validate_yaml_value "$PARTITION" "slurm.partition"
validate_yaml_value "$CPUS" "slurm.cpus"
validate_yaml_value "$MEM_GB" "slurm.mem_gb"
validate_yaml_value "$GPUS" "slurm.gpus"
validate_yaml_value "$TIME" "slurm.time"

# Spatial network
validate_yaml_value "$SPATIAL_FILTER" "spatial_network.marker_filter"
validate_yaml_value "$SPATIAL_EDGE" "spatial_network.max_edge_distance"
validate_yaml_value "$SPATIAL_MIN_COMP" "spatial_network.min_component_cells"
validate_yaml_value "$PIXEL_SIZE" "pixel_size_um"

# SpatialData
validate_yaml_value "$SPATIALDATA_CLUSTER_KEY" "spatialdata.squidpy_cluster_key"

# Spatial viewer
validate_yaml_value "$VIEWER_GROUP_FIELD" "spatial_viewer.group_field"
validate_yaml_value "$VIEWER_TITLE" "spatial_viewer.title"

# Downstream steps
validate_yaml_value "$DS_QF_MIN_AREA" "downstream.quality_filter.min_area_um2"
validate_yaml_value "$DS_QF_MAX_AREA" "downstream.quality_filter.max_area_um2"
validate_yaml_value "$DS_QF_MIN_SOLIDITY" "downstream.quality_filter.min_solidity"
validate_yaml_value "$DS_NUC_CHANNEL_SPEC" "downstream.nuclei.channel_spec"
validate_yaml_value "$DS_HTML_MAX_SAMPLES" "downstream.annotation_html.max_samples"
validate_yaml_value "$DS_HTML_DISPLAY_CHANNELS" "downstream.annotation_html.display_channels"
validate_yaml_value "$DS_CLUSTERING_FEATURES" "downstream.clustering.feature_groups"
validate_yaml_value "$DS_CLUSTERING_METHODS" "downstream.clustering.methods"
validate_yaml_value "$DS_CLUSTERING_RESOLUTION" "downstream.clustering.resolution"

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
    # Usage: build_seg_cmd CZI_PATH OUTPUT_DIR
    # Scene is handled by the generic body (appended at runtime if non-empty)
    local czi_arg="$1"
    local out_arg="$2"
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
    # Built-in marker SNR classification
    if [[ -n "$MARKER_SNR_CHANNELS" ]]; then
        cmd+=" --marker-snr-channels \"$MARKER_SNR_CHANNELS\""
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

# Validate marker values (parsed from markers JSON list)
validate_yaml_value "$MARKER_CHANNELS" "markers[].channel/wavelength"
validate_yaml_value "$MARKER_NAMES" "markers[].name"
validate_yaml_value "$MARKER_METHOD" "markers[].method"

# ---------------------------------------------------------------------------
# Pre-compute work items: (czi_path, output_dir, scene)
# Every mode is just a different enumeration of these tuples.
# Single-slide/single-scene is one work item — the universal base case.
# ---------------------------------------------------------------------------
_GEN_SLIDES=()
if [[ "$MULTI_SLIDE" == "true" ]]; then
    mapfile -t _GEN_SLIDES < <(find "$CZI_DIR" -maxdepth 1 -name "$CZI_GLOB" -type f | sort)
fi

_WORK_CZI=()
_WORK_OUT=()
_WORK_SCENE=()

# Build scene list (empty string means no scene)
_SCENE_LIST=("")
if [[ "$MULTI_SCENE" == "true" ]]; then
    _SCENE_LIST=()
    for (( _s=SCENE_START; _s<=SCENE_END; _s++ )); do
        _SCENE_LIST+=("$_s")
    done
fi

# Build slide list (single-slide: just CZI_PATH)
_SLIDE_LIST=()
_SLIDE_NAMES=()
if [[ "$MULTI_SLIDE" == "true" ]]; then
    _SLIDE_LIST=("${_GEN_SLIDES[@]}")
    for _s in "${_GEN_SLIDES[@]}"; do
        _SLIDE_NAMES+=("$(basename "$_s" .czi)")
    done
else
    _SLIDE_LIST=("$CZI_PATH")
    _SLIDE_NAMES+=("")  # no subdirectory for single-slide
fi

# Cross product: slides × scenes
for (( _si=0; _si<${#_SLIDE_LIST[@]}; _si++ )); do
    _czi="${_SLIDE_LIST[$_si]}"
    _sname="${_SLIDE_NAMES[$_si]}"
    for _sc in "${_SCENE_LIST[@]}"; do
        _WORK_CZI+=("$_czi")
        # Output directory: slide_name/scene_K, slide_name, scene_K, or flat
        _out="$OUTPUT_DIR"
        if [[ -n "$_sname" ]]; then _out+="/$_sname"; fi
        if [[ -n "$_sc" ]]; then _out+="/scene_${_sc}"; fi
        _WORK_OUT+=("$_out")
        _WORK_SCENE+=("$_sc")
    done
done

_N_WORK=${#_WORK_CZI[@]}
if [[ $_N_WORK -eq 0 ]]; then
    echo "Error: no work items generated (no CZI files match glob?)" >&2
    exit 1
fi

# Determine if work items run in parallel (array) or sequential (loop)
USE_ARRAY=true
if [[ "$MULTI_SCENE" == "true" && "$SCENE_PARALLEL" == "false" && "$MULTI_SLIDE" == "false" ]]; then
    USE_ARRAY=false
fi
# Single work item never needs an array
if [[ $_N_WORK -eq 1 ]]; then
    USE_ARRAY=false
fi

# ---------------------------------------------------------------------------
# Helper: emit post-detection steps (markers, spatial, spatialdata)
# Usage: _emit_post_detection INDENT CZI_VAR
#   INDENT: indentation string
#   CZI_VAR: CZI path expression in the generated script (e.g. "\$CZI_FILE")
# ---------------------------------------------------------------------------
_emit_post_detection() {
    local I="$1"
    local CZI="$2"

    if [[ -n "$MARKER_CHANNELS" ]]; then
        local classify_extra=""
        if [[ "$CORRECT_ALL_CHANNELS" == "true" ]]; then
            classify_extra+=" --correct-all-channels"
        fi
        echo "${I}echo \"  Classifying markers: $MARKER_NAMES\""
        if [[ "$MARKER_USE_WAVELENGTH" == "true" ]]; then
            echo "${I}\$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-wavelength \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --czi-path \"$CZI\" --output-dir \"\$RUN_DIR\"${classify_extra}"
        else
            echo "${I}\$XLDVP_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel \"$MARKER_CHANNELS\" --marker-name \"$MARKER_NAMES\" --method \"$MARKER_METHOD\" --output-dir \"\$RUN_DIR\"${classify_extra}"
        fi
        echo "${I}for _cf in \"\${RUN_DIR}${CELL_TYPE}_detections_filtered_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\"; do"
        echo "${I}    if [[ -f \"\$_cf\" ]]; then DET_JSON=\"\$_cf\"; break; fi"
        echo "${I}done"
    fi

    if [[ "$SPATIAL_ENABLED" == "true" ]]; then
        echo "${I}echo \"  Running spatial analysis...\""
        echo "${I}\$XLDVP_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$RUN_DIR\" --spatial-network --marker-filter \"$SPATIAL_FILTER\" --max-edge-distance \"$SPATIAL_EDGE\" --min-component-cells \"$SPATIAL_MIN_COMP\" --pixel-size \"$PIXEL_SIZE\""
    fi

    if [[ "$SPATIALDATA_ENABLED" == "true" ]]; then
        echo "${I}echo \"  Exporting to SpatialData...\""
        local sd_cmd="${I}\$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py --detections \"\$DET_JSON\" --output \"\${RUN_DIR}${CELL_TYPE}_spatialdata.zarr\" --cell-type \"$CELL_TYPE\" --overwrite"
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
}

# ---------------------------------------------------------------------------
# Helper: bake work item arrays into a generated sbatch script
# ---------------------------------------------------------------------------
_emit_work_arrays() {
    echo "# Work items: ${_N_WORK} total (czi_path, output_dir, scene)"
    echo "ALL_CZI=("
    for _c in "${_WORK_CZI[@]}"; do echo "    \"$_c\""; done
    echo ")"
    echo "ALL_OUT=("
    for _o in "${_WORK_OUT[@]}"; do echo "    \"$_o\""; done
    echo ")"
    echo "ALL_SCENE=("
    for _s in "${_WORK_SCENE[@]}"; do echo "    \"$_s\""; done
    echo ")"
    echo "N_WORK=${_N_WORK}"
    echo ""

    # Resume paths (associative array keyed by slide name)
    echo "declare -A RESUME_PATHS"
    if [[ -n "$RESUME_DIR" ]]; then
        if [[ "$MULTI_SLIDE" == "true" ]]; then
            for _s in "${_GEN_SLIDES[@]}"; do
                echo "RESUME_PATHS[$(basename "$_s" .czi)]=\"$RESUME_DIR\""
            done
        else
            # Single-slide: key by CZI basename
            echo "RESUME_PATHS[$(basename "$CZI_PATH" .czi)]=\"$RESUME_DIR\""
        fi
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Write sbatch script — ONE generic body for all modes
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
    if [[ "$USE_ARRAY" == "true" ]]; then
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%A_%a.err"
        echo "#SBATCH --array=0-$((_N_WORK - 1))"
    else
        echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_%j.out"
        echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_%j.err"
    fi

    echo ""
    echo "set -euo pipefail"
    echo "export PYTHONPATH=\"$REPO\""
    echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
    echo ""

    # Bake work items into sbatch
    _emit_work_arrays

    # --- Generic work item body (same for ALL modes) ---
    # Parallel mode: process one work item per array task
    # Sequential mode: loop over all work items
    if [[ "$USE_ARRAY" == "true" ]]; then
        echo "IDX=\$SLURM_ARRAY_TASK_ID"
    else
        echo "for (( IDX=0; IDX<N_WORK; IDX++ )); do"
    fi

    # Indent: top-level for array, one level for loop
    I=""
    if [[ "$USE_ARRAY" == "false" ]]; then I="    "; fi

    echo "${I}CZI_FILE=\"\${ALL_CZI[\$IDX]}\""
    echo "${I}SLIDE_OUT=\"\${ALL_OUT[\$IDX]}\""
    echo "${I}SCENE=\"\${ALL_SCENE[\$IDX]}\""
    echo "${I}mkdir -p \"\$SLIDE_OUT\""
    echo "${I}SLIDE_NAME=\$(basename \"\$CZI_FILE\" .czi)"
    echo "${I}echo \"=== [\$((IDX+1))/${_N_WORK}] \$SLIDE_NAME scene=\$SCENE ===\""
    echo ""

    # Segmentation command
    echo "${I}RESUME_FLAG=\"\""
    echo "${I}if [[ -n \"\${RESUME_PATHS[\$SLIDE_NAME]:-}\" ]]; then RESUME_FLAG=\"--resume \\\"\${RESUME_PATHS[\$SLIDE_NAME]}\\\"\"; fi"
    echo "${I}SCENE_FLAG=\"\""
    echo "${I}if [[ -n \"\$SCENE\" ]]; then SCENE_FLAG=\"--scene \$SCENE\"; fi"
    echo "${I}$(build_seg_cmd '${CZI_FILE}' '${SLIDE_OUT}') \$SCENE_FLAG \$RESUME_FLAG"
    echo ""

    # Find detection JSON
    echo "${I}RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
    echo "${I}DET_JSON=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
    echo "${I}if [[ -n \"\$RUN_DIR\" && -f \"\$DET_JSON\" ]]; then"
    _emit_post_detection "${I}    " "\$CZI_FILE"

    # Spatial viewer (inline, after post-detection)
    if [[ "$VIEWER_ENABLED" == "true" && -n "$VIEWER_GROUP_FIELD" ]]; then
        echo "${I}    echo \"Generating spatial viewer...\""
        echo "${I}    \$XLDVP_PYTHON $REPO/scripts/generate_multi_slide_spatial_viewer.py --input-dir \"\$SLIDE_OUT\" --group-field \"$VIEWER_GROUP_FIELD\" --title \"$VIEWER_TITLE_ESC\" --output \"\${SLIDE_OUT}/spatial_viewer.html\""
    fi

    echo "${I}else"
    echo "${I}    echo \"WARNING: ${CELL_TYPE}_detections.json not found\""
    echo "${I}fi"

    # Close loop for sequential mode
    if [[ "$USE_ARRAY" == "false" ]]; then
        echo "done"
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
_mode="single-slide"
if [[ "$MULTI_SLIDE" == "true" && "$MULTI_SCENE" == "true" ]]; then
    _mode="multi-slide + multi-scene"
elif [[ "$MULTI_SLIDE" == "true" ]]; then
    _mode="multi-slide"
elif [[ "$MULTI_SCENE" == "true" ]]; then
    _mode="multi-scene"
fi
_strategy="parallel (array)"
if [[ "$USE_ARRAY" == "false" ]]; then _strategy="sequential (loop)"; fi
echo "Mode:     ${_mode}, ${_strategy}, ${_N_WORK} work items"
if [[ "$MULTI_SLIDE" == "true" ]]; then
    echo "CZI dir:  $CZI_DIR ($CZI_GLOB)"
else
    echo "CZI:      $CZI_PATH"
fi
if [[ "$MULTI_SCENE" == "true" ]]; then
    echo "Scenes:   ${SCENE_START}-${SCENE_END}"
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
        if [[ -n "$GPU_TYPE" ]]; then
            echo "#SBATCH --gres=gpu:${GPU_TYPE}:1"
        else
            echo "#SBATCH --gres=gpu:1"
        fi
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
# Same work item arrays as detection, same parallel/sequential strategy
if [[ "$DS_NUCLEI" == "true" ]]; then
    NUC_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_nuc_$$.sbatch"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_nuc"
        echo "#SBATCH --partition=p.hpcl93"
        echo "#SBATCH --nodes=1"
        echo "#SBATCH --cpus-per-task=64"
        echo "#SBATCH --mem=300G"
        echo "#SBATCH --gres=gpu:l40s:1"
        echo "#SBATCH --time=6:00:00"
        if [[ "$USE_ARRAY" == "true" ]]; then
            echo "#SBATCH --array=0-$((_N_WORK - 1))"
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_nuc_%A_%a.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_nuc_%A_%a.err"
        else
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_nuc_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_nuc_%j.err"
        fi
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        echo ""
        _emit_work_arrays

        if [[ "$USE_ARRAY" == "true" ]]; then
            echo "IDX=\$SLURM_ARRAY_TASK_ID"
        else
            echo "for (( IDX=0; IDX<N_WORK; IDX++ )); do"
        fi

        I=""
        if [[ "$USE_ARRAY" == "false" ]]; then I="    "; fi

        echo "${I}CZI_FILE=\"\${ALL_CZI[\$IDX]}\""
        echo "${I}SLIDE_OUT=\"\${ALL_OUT[\$IDX]}\""
        echo "${I}SCENE=\"\${ALL_SCENE[\$IDX]}\""
        echo "${I}echo \"=== \$(date): Nuclear counting [\$((IDX+1))/${_N_WORK}] ===\""
        echo "${I}RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
        echo "${I}if [[ -z \"\$RUN_DIR\" || ! -f \"\${RUN_DIR}${CELL_TYPE}_detections.json\" ]]; then echo 'ERROR: Detection output not found for \$SLIDE_OUT'; exit 1; fi"

        local_nuc_cmd="\$XLDVP_PYTHON $REPO/scripts/count_nuclei_per_cell.py"
        local_nuc_cmd+=" --detections \"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        local_nuc_cmd+=" --czi-path \"\$CZI_FILE\""
        local_nuc_cmd+=" --tiles-dir \"\${RUN_DIR}tiles\""
        if [[ -n "$DS_NUC_CHANNEL_SPEC" ]]; then
            local_nuc_cmd+=" --channel-spec \"$DS_NUC_CHANNEL_SPEC\""
        fi
        echo "${I}SCENE_FLAG=\"\""
        echo "${I}if [[ -n \"\$SCENE\" ]]; then SCENE_FLAG=\"--scene \$SCENE\"; fi"
        echo "${I}${local_nuc_cmd} \$SCENE_FLAG --output \"\${RUN_DIR}${CELL_TYPE}_detections_nuclei.json\""
        echo "${I}echo \"=== \$(date): Done ===\""

        if [[ "$USE_ARRAY" == "false" ]]; then
            echo "done"
        fi
    } > "$NUC_SBATCH"
    chmod +x "$NUC_SBATCH"
    NUC_OUTPUT=$(sbatch --dependency=afterok:"$MAIN_JOB_ID" "$NUC_SBATCH")
    NUC_JOB_ID=$(echo "$NUC_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    echo "  Nuclear counting job: $NUC_JOB_ID (dep: $MAIN_JOB_ID)"
fi

# Annotation HTML (CPU, depends on detection)
# Same work item arrays, same parallel/sequential strategy
if [[ "$DS_HTML" == "true" ]]; then
    HTML_SBATCH="${OUTPUT_DIR}/pipeline_${NAME}_html_$$.sbatch"
    HTML_DEP_ID="$DETECT_JOB_ID"
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=${NAME}_html"
        echo "#SBATCH --partition=p.hpcl8"
        echo "#SBATCH --cpus-per-task=8"
        echo "#SBATCH --mem=64G"
        echo "#SBATCH --time=2:00:00"
        if [[ "$USE_ARRAY" == "true" ]]; then
            echo "#SBATCH --array=0-$((_N_WORK - 1))"
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_html_%A_%a.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_html_%A_%a.err"
        else
            echo "#SBATCH --output=${OUTPUT_DIR}/slurm_${NAME}_html_%j.out"
            echo "#SBATCH --error=${OUTPUT_DIR}/slurm_${NAME}_html_%j.err"
        fi
        echo ""
        echo "set -euo pipefail"
        echo "export PYTHONPATH=\"$REPO\""
        echo "XLDVP_PYTHON=\"$XLDVP_PYTHON\""
        echo ""
        _emit_work_arrays

        if [[ "$USE_ARRAY" == "true" ]]; then
            echo "IDX=\$SLURM_ARRAY_TASK_ID"
        else
            echo "for (( IDX=0; IDX<N_WORK; IDX++ )); do"
        fi

        I=""
        if [[ "$USE_ARRAY" == "false" ]]; then I="    "; fi

        echo "${I}CZI_FILE=\"\${ALL_CZI[\$IDX]}\""
        echo "${I}SLIDE_OUT=\"\${ALL_OUT[\$IDX]}\""
        echo "${I}SCENE=\"\${ALL_SCENE[\$IDX]}\""
        echo "${I}RUN_DIR=\$(ls -td \"\${SLIDE_OUT}\"/*/  2>/dev/null | head -1)"
        echo "${I}if [[ -z \"\$RUN_DIR\" ]]; then echo 'ERROR: Run directory not found for \$SLIDE_OUT'; exit 1; fi"
        echo "${I}DET=\"\${RUN_DIR}${CELL_TYPE}_detections.json\""
        echo "${I}for _f in \"\${RUN_DIR}${CELL_TYPE}_detections_filtered_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_classified.json\" \"\${RUN_DIR}${CELL_TYPE}_detections_filtered.json\"; do"
        echo "${I}    if [[ -f \"\$_f\" ]]; then DET=\"\$_f\"; break; fi"
        echo "${I}done"
        echo "${I}echo \"=== \$(date): Annotation HTML [\$((IDX+1))/${_N_WORK}] ===\""

        local_html_cmd="\$XLDVP_PYTHON $REPO/scripts/regenerate_html.py"
        local_html_cmd+=" --detections \"\$DET\""
        local_html_cmd+=" --output-dir \"\$RUN_DIR\""
        local_html_cmd+=" --czi-path \"\$CZI_FILE\""
        if [[ -n "$DS_HTML_DISPLAY_CHANNELS" ]]; then
            local_html_cmd+=" --display-channels \"$DS_HTML_DISPLAY_CHANNELS\""
        fi
        local_html_cmd+=" --dashed-contour --max-samples $DS_HTML_MAX_SAMPLES"
        local_html_cmd+=" --html-dir \"\${RUN_DIR}html_annotation\""
        echo "${I}SCENE_FLAG=\"\""
        echo "${I}if [[ -n \"\$SCENE\" ]]; then SCENE_FLAG=\"--scene \$SCENE\"; fi"
        echo "${I}${local_html_cmd} \$SCENE_FLAG"
        echo "${I}echo \"=== \$(date): Done ===\""

        if [[ "$USE_ARRAY" == "false" ]]; then
            echo "done"
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
# Spatial viewer job (multi-slide only)
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
