#!/bin/bash
#SBATCH --job-name=brain_full
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --array=0-7
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/brain_full_%A_%a.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/brain_full_%A_%a.err

# =============================================================================
# Brain FISH (goldfish) — tissue_pattern FULL 100% run
# 4 CZIs x 2 scenes = 8 jobs (array 0-7)
# 5-channel CZI (EDFvar):
#   ch0: AF488 — Slc17a7 (excitatory neuron marker)
#   ch1: AF647 — Htr2a   (serotonin receptor)
#   ch2: AF750 — Ntrk2   (neurotrophin receptor)
#   ch3: AF555 — Gad1    (inhibitory neuron marker)
#   ch4: H3258 — Hoechst (nuclear)
# Detection: Cellpose on Slc17a7+Gad1 sum (ch0+ch3), SAM2 refinement
# Clustering (post-hoc): all markers except Hoechst (ch0,1,2,3)
# =============================================================================

set -e

# Map array index -> CZI + scene
CZI_DIR="/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/MPIP_psilo"
OUT_BASE="/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/brain_fish_output"

CZIS=(
    "19022026_gold_fish_ctrl_psil_488Slc17a7_555Gad1_647Htr2a_750Ntrk2_Hoechst-EDFvar-stitch.czi"
    "19022026_gold_fish_ctrl_veh_488Slc17a7_555Gad1_647Htr2a_750Ntrk2_Hoechst-EDFvar-stitch.czi"
    "19022026_gold_fish_pnels_veh_488Slc17a7_555Gad1_647Htr2a_750Ntrk2_Hoechst-EDFvar-stitch.czi"
    "19022026_gold_fish_pnels_psil_488Slc17a7_555Gad1_647Htr2a_750Ntrk2_Hoechst-EDFvar-stitch-1.czi"
)

NAMES=(
    "ctrl_psil"
    "ctrl_veh"
    "pnels_veh"
    "pnels_psil"
)

CZI_IDX=$((SLURM_ARRAY_TASK_ID / 2))
SCENE=$((SLURM_ARRAY_TASK_ID % 2))

CZI="${CZI_DIR}/${CZIS[$CZI_IDX]}"
NAME="${NAMES[$CZI_IDX]}"
OUTPUT="${OUT_BASE}/${NAME}/scene${SCENE}"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:   $SLURMD_NODENAME"
echo "Slide:  $NAME"
echo "Scene:  $SCENE"
echo "CZI:    $CZI"
echo "Output: $OUTPUT"
echo "Mode:   tissue_pattern, 100% tiles, 4x L40S"
echo "============================================================"

source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

nvidia-smi -L
echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$OUTPUT"

python run_segmentation.py \
    --czi-path "$CZI" \
    --cell-type tissue_pattern \
    --scene "$SCENE" \
    --tp-detection-channels 0,3 \
    --tp-nuclear-channel 4 \
    --tp-display-channels 0,3,1 \
    --channel-names "Slc17a7,Htr2a,Ntrk2,Gad1,Hoechst" \
    --tile-size 4000 \
    --sample-fraction 1.0 \
    --tile-overlap 0.10 \
    --load-to-ram \
    --no-normalize-features \
    --max-html-samples 15000 \
    --num-gpus 4 \
    --no-serve \
    --output-dir "$OUTPUT"

# Find the detections JSON (pipeline creates a timestamped subdirectory)
DETECTIONS=$(find "$OUTPUT" -name "tissue_pattern_detections.json" -type f | head -1)
if [ -z "$DETECTIONS" ]; then
    echo "ERROR: No detections JSON found in $OUTPUT"
    exit 1
fi
N_DET=$(python -c "import json; d=json.load(open('$DETECTIONS')); print(len(d) if isinstance(d,list) else len(d.get('detections',[])))" 2>/dev/null || echo '0')
echo "=== Detection complete at $(date): $N_DET cells ==="
echo "Detections file: $DETECTIONS"

# --- Post-processing: clustering by expression (exclude Hoechst ch4) ---
echo "=== Starting clustering at $(date) ==="
python scripts/cluster_by_features.py \
    --detections "$DETECTIONS" \
    --output-dir "$OUTPUT/clustering" \
    --marker-channels "Slc17a7:0,Htr2a:1,Ntrk2:2,Gad1:3" \
    --exclude-channels 4 \
    --feature-groups "channel" \
    --marker-only \
    --threshold 0.0

echo "=== Clustering complete at $(date) ==="

# --- Post-processing: tissue zone assignment (exclude Hoechst ch4) ---
CLUSTERED="$OUTPUT/clustering/detections_clustered.json"
if [ -f "$CLUSTERED" ]; then
    echo "=== Starting zone assignment at $(date) ==="
    python scripts/assign_tissue_zones.py \
        --detections "$CLUSTERED" \
        --output-dir "$OUTPUT/zones" \
        --marker-channels "Slc17a7:0,Htr2a:1,Ntrk2:2,Gad1:3" \
        --exclude-channels 4 \
        --visualize
    echo "=== Zone assignment complete at $(date) ==="
fi

# --- Post-processing: spatial frequency analysis (exclude Hoechst ch4) ---
ZONED="$OUTPUT/zones/zoned_detections.json"
INPUT_FOR_SPATIAL="${ZONED}"
if [ ! -f "$INPUT_FOR_SPATIAL" ]; then
    INPUT_FOR_SPATIAL="$CLUSTERED"
fi
if [ ! -f "$INPUT_FOR_SPATIAL" ]; then
    INPUT_FOR_SPATIAL="$DETECTIONS"
fi

echo "=== Starting spatial frequency analysis at $(date) ==="
python scripts/spatial_frequency_analysis.py \
    --detections "$INPUT_FOR_SPATIAL" \
    --channels 0,1,2,3 \
    --output-dir "$OUTPUT/spatial"
echo "=== Spatial analysis complete at $(date) ==="

# --- Visualization: cluster gallery ---
echo "=== Generating cluster gallery at $(date) ==="
python scripts/generate_cluster_gallery.py \
    --detections "$CLUSTERED" \
    --czi-path "$CZI" \
    --output-dir "$OUTPUT/gallery" \
    --display-channels "0,3,1" \
    --marker-channels "Slc17a7:0,Htr2a:1,Ntrk2:2,Gad1:3" \
    --scene "$SCENE" \
    --n-per-cluster 100
echo "=== Cluster gallery complete at $(date) ==="

# --- Visualization: tissue overlay (cells colored by cluster) ---
SPATIAL_CSV="$OUTPUT/clustering/spatial.csv"
if [ -f "$SPATIAL_CSV" ]; then
    echo "=== Generating tissue overlay (cluster) at $(date) ==="
    python scripts/generate_tissue_overlay.py \
        --czi-path "$CZI" \
        --spatial-csv "$SPATIAL_CSV" \
        --display-channels "0,3,1" \
        --cluster-field cluster_label \
        --scene "$SCENE" \
        --output-dir "$OUTPUT/overlay_cluster"
    echo "=== Tissue overlay (cluster) complete at $(date) ==="
fi

# --- Visualization: tissue overlay (cells colored by zone) ---
ZONE_CSV="$OUTPUT/zones/spatial.csv"
if [ -f "$ZONE_CSV" ]; then
    echo "=== Generating tissue overlay (zone) at $(date) ==="
    python scripts/generate_tissue_overlay.py \
        --czi-path "$CZI" \
        --spatial-csv "$ZONE_CSV" \
        --display-channels "0,3,1" \
        --cluster-field zone \
        --scene "$SCENE" \
        --output-dir "$OUTPUT/overlay_zone"
    echo "=== Tissue overlay (zone) complete at $(date) ==="
fi

echo "=== All done at $(date): $NAME scene $SCENE ==="
