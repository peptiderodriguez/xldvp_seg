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
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
keys = '$key'.split('.')
v = cfg
for k in keys:
    if v is None:
        v = None
        break
    v = v.get(k) if isinstance(v, dict) else None
if v is None:
    print('$default')
elif isinstance(v, list):
    print(' '.join(str(x) for x in v))
elif isinstance(v, bool):
    print('true' if v else 'false')
else:
    print(v)
")
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
NUM_GPUS=$(read_yaml num_gpus 1)
MIN_AREA=$(read_yaml min_area_um "")
MAX_AREA=$(read_yaml max_area_um "")

# SLURM settings
PARTITION=$(read_yaml slurm.partition p.hpcl8)
CPUS=$(read_yaml slurm.cpus 24)
MEM_GB=$(read_yaml slurm.mem_gb 350)
GPUS=$(read_yaml slurm.gpus "rtx5000:2")
TIME=$(read_yaml slurm.time "2-00:00:00")
SLIDES_PER_JOB=$(read_yaml slurm.slides_per_job 1)
NUM_JOBS=$(read_yaml slurm.num_jobs 1)

# Markers (parsed as JSON list)
MARKERS_JSON=$("$MKSEG_PYTHON" -c "
import yaml, json
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(json.dumps(cfg.get('markers', [])))
")

# Spatial network
SPATIAL_ENABLED=$(read_yaml spatial_network.enabled "")
SPATIAL_FILTER=$(read_yaml spatial_network.marker_filter "")
SPATIAL_EDGE=$(read_yaml spatial_network.max_edge_distance 50)
SPATIAL_MIN_COMP=$(read_yaml spatial_network.min_component_cells 3)
# If spatial_network section exists but enabled is not explicitly set, treat as enabled
if [[ -n "$SPATIAL_FILTER" && -z "$SPATIAL_ENABLED" ]]; then
    SPATIAL_ENABLED="true"
fi

# ---------------------------------------------------------------------------
# Determine multi-slide vs single-slide mode
# ---------------------------------------------------------------------------
MULTI_SLIDE=false
if [[ -n "$CZI_DIR" && -n "$CZI_GLOB" ]]; then
    MULTI_SLIDE=true
fi

# ---------------------------------------------------------------------------
# Build run_segmentation.py command flags
# ---------------------------------------------------------------------------
build_seg_cmd() {
    local czi_arg="$1"
    local out_arg="$2"
    local cmd="$MKSEG_PYTHON $REPO/run_segmentation.py"
    cmd+=" --czi-path $czi_arg"
    cmd+=" --cell-type $CELL_TYPE"
    cmd+=" --output-dir $out_arg"
    cmd+=" --num-gpus $NUM_GPUS"

    # Cellpose input channels: passed as "cyto,nuc" to --cellpose-input-channels
    if [[ -n "$CP_CHANNELS" ]]; then
        local ch_cyto ch_nuc
        ch_cyto=$(echo "$CP_CHANNELS" | awk '{print $1}')
        ch_nuc=$(echo "$CP_CHANNELS" | awk '{print $2}')
        cmd+=" --cellpose-input-channels ${ch_cyto},${ch_nuc}"
    fi

    if [[ "$PHOTOBLEACH" == "true" ]]; then
        cmd+=" --photobleach-correction"
    fi
    if [[ "$ALL_CHANNELS" == "true" ]]; then
        cmd+=" --all-channels"
    fi
    if [[ -n "$MIN_AREA" ]]; then
        cmd+=" --min-cell-area $MIN_AREA"
    fi
    if [[ -n "$MAX_AREA" ]]; then
        cmd+=" --max-cell-area $MAX_AREA"
    fi
    echo "$cmd"
}

# ---------------------------------------------------------------------------
# Build marker classification commands
# ---------------------------------------------------------------------------
build_marker_cmds() {
    local det_json="$1"
    local out_dir="$2"
    "$MKSEG_PYTHON" -c "
import json
markers = json.loads('$MARKERS_JSON')
for m in markers:
    ch = m['channel']
    name = m['name']
    method = m.get('method', 'otsu_half')
    print(f'$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections {det_json} --marker-channel {ch} --marker-name {name} --method {method} --output-dir {out_dir}')
"
}

# ---------------------------------------------------------------------------
# Build spatial analysis command
# ---------------------------------------------------------------------------
build_spatial_cmd() {
    local det_json="$1"
    local out_dir="$2"
    echo "$MKSEG_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections $det_json --output-dir $out_dir --marker-filter $SPATIAL_FILTER --max-edge-distance $SPATIAL_EDGE --min-component-cells $SPATIAL_MIN_COMP"
}

# ---------------------------------------------------------------------------
# Write sbatch script
# ---------------------------------------------------------------------------
SBATCH_FILE="/tmp/pipeline_${NAME}.sbatch"

{
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${NAME}"
    echo "#SBATCH --partition=${PARTITION}"
    echo "#SBATCH --cpus-per-task=${CPUS}"
    echo "#SBATCH --mem=${MEM_GB}G"
    echo "#SBATCH --gres=gpu:${GPUS}"
    echo "#SBATCH --time=${TIME}"
    echo "#SBATCH --output=/tmp/slurm_${NAME}_%A_%a.out"
    echo "#SBATCH --error=/tmp/slurm_${NAME}_%A_%a.err"

    if [[ "$MULTI_SLIDE" == "true" ]]; then
        echo "#SBATCH --array=0-$((NUM_JOBS - 1))"
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
        echo "    # Step 2: Marker classification"
        echo "    DET_JSON=\"\${SLIDE_OUT}/${CELL_TYPE}_detections.json\""
        echo "    if [[ -f \"\$DET_JSON\" ]]; then"

        # Emit marker commands
        "$MKSEG_PYTHON" -c "
import json
markers = json.loads('$MARKERS_JSON')
for m in markers:
    ch = m['channel']
    name = m['name']
    method = m.get('method', 'otsu_half')
    print(f'        echo \"  Classifying marker: {name} (ch{ch})\"')
    print(f'        \$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel {ch} --marker-name {name} --method {method} --output-dir \"\$SLIDE_OUT\"')
"

        if [[ "$SPATIAL_ENABLED" == "true" ]]; then
            echo ""
            echo "        # Step 3: Spatial analysis"
            echo "        echo \"  Running spatial analysis...\""
            echo "        \$MKSEG_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$SLIDE_OUT\" --marker-filter $SPATIAL_FILTER --max-edge-distance $SPATIAL_EDGE --min-component-cells $SPATIAL_MIN_COMP"
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
        echo "# Step 2: Marker classification"
        echo "DET_JSON=\"\${SLIDE_OUT}/${CELL_TYPE}_detections.json\""
        echo "if [[ -f \"\$DET_JSON\" ]]; then"

        "$MKSEG_PYTHON" -c "
import json
markers = json.loads('$MARKERS_JSON')
for m in markers:
    ch = m['channel']
    name = m['name']
    method = m.get('method', 'otsu_half')
    print(f'    echo \"Classifying marker: {name} (ch{ch})\"')
    print(f'    \$MKSEG_PYTHON $REPO/scripts/classify_markers.py --detections \"\$DET_JSON\" --marker-channel {ch} --marker-name {name} --method {method} --output-dir \"\$SLIDE_OUT\"')
"

        if [[ "$SPATIAL_ENABLED" == "true" ]]; then
            echo ""
            echo "    # Step 3: Spatial analysis"
            echo "    echo \"Running spatial analysis...\""
            echo "    \$MKSEG_PYTHON $REPO/scripts/spatial_cell_analysis.py --detections \"\$DET_JSON\" --output-dir \"\$SLIDE_OUT\" --marker-filter $SPATIAL_FILTER --max-edge-distance $SPATIAL_EDGE --min-component-cells $SPATIAL_MIN_COMP"
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
sbatch "$SBATCH_FILE"
