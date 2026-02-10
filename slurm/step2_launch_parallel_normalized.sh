#!/bin/bash
# STEP 2: Launch 8 parallel jobs using pre-computed normalization parameters
# Then auto-submit STEP 3 (combined HTML generation) with dependency on all step 2 jobs
#
# Usage: ./step2_launch_parallel_normalized.sh [STEP1_JOBID]
#   If STEP1_JOBID provided, step 2 jobs wait for it to complete (--dependency=afterok)

cd /viper/ptmp2/edrod/xldvp_seg_fresh

DEPEND_ARG=""
if [ -n "$1" ]; then
    DEPEND_ARG="--dependency=afterok:$1"
    echo "Jobs will wait for step 1 job $1 to complete"
fi

# Check if normalization params file exists (skip check if dependency set — file won't exist yet)
NORM_PARAMS="/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json"
if [ -z "$DEPEND_ARG" ] && [ ! -f "$NORM_PARAMS" ]; then
    echo "ERROR: Normalization parameters file not found: $NORM_PARAMS"
    echo "Please run step1_compute_norm_params.sbatch first!"
    exit 1
fi

# Experiment metadata
TIMESTAMP=$(date +%Y-%m-%d)
SAMPLE_PCT=10
NUM_GPUS=2
EXPERIMENT="${TIMESTAMP}_${SAMPLE_PCT}pct_${NUM_GPUS}gpu"

OUTPUT_DIR="/viper/ptmp2/edrod/unified_${EXPERIMENT}"
HTML_DIR="/viper/ptmp2/edrod/docs_${EXPERIMENT}"
mkdir -p "$OUTPUT_DIR" "$HTML_DIR" logs

echo "Using normalization parameters from: $NORM_PARAMS"
echo "Experiment: $EXPERIMENT"
echo "Output:     $OUTPUT_DIR"
echo "HTML:       $HTML_DIR"
echo ""

# Define slide pairs
SLIDE_PAIRS=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU1.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU3.czi /viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU4.czi"
)

echo "============================================================"
echo "Submitting 8 parallel step 2 jobs with GLOBAL normalization"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo "HTML directory:   $HTML_DIR"
echo "Experiment name:  $EXPERIMENT"
echo ""

# Collect step 2 job IDs for step 3 dependency
STEP2_JOBS=""

for i in {0..7}; do
    SLIDES="${SLIDE_PAIRS[$i]}"
    JOB_NAME="mkseg_${EXPERIMENT}_b$((i+1))"

    JID=$(sbatch --parsable --job-name="$JOB_NAME" \
           --partition=apu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=48 \
           --mem=200G \
           --time=2:00:00 \
           --gres=gpu:2 \
           --output="logs/${JOB_NAME}_%j.out" \
           --error="logs/${JOB_NAME}_%j.err" \
           $DEPEND_ARG \
           --wrap="
module load python-waterboa/2024.06
module load rocm/6.3
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2

python run_unified_FAST.py \
    --czi-paths $SLIDES \
    --output-dir $OUTPUT_DIR \
    --html-output-dir $HTML_DIR/batch$((i+1)) \
    --experiment-name $EXPERIMENT \
    --tile-size 3000 \
    --sample-fraction 0.10 \
    --multi-gpu \
    --num-gpus $NUM_GPUS \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-min-area-um 25 \
    --hspc-max-area-um 150 \
    --cleanup-masks \
    --normalize-slides \
    --normalization-method reinhard \
    --norm-params-file $NORM_PARAMS
")

    STEP2_JOBS="${STEP2_JOBS}:${JID}"
    echo "  Submitted batch $((i+1)) (job $JID): $SLIDES"
done

echo ""
echo "============================================================"
echo "All 8 step 2 jobs submitted!"
echo "============================================================"
echo ""

# Submit step 3: combined HTML generation (depends on all step 2 jobs)
STEP3_JOB_NAME="mkseg_${EXPERIMENT}_html"
STEP3_JID=$(sbatch --parsable \
    --job-name="$STEP3_JOB_NAME" \
    --partition=apu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=24 \
    --mem=16G \
    --time=0:15:00 \
    --gres=gpu:1 \
    --output="logs/${STEP3_JOB_NAME}_%j.out" \
    --error="logs/${STEP3_JOB_NAME}_%j.err" \
    --dependency=afterok${STEP2_JOBS} \
    slurm/step3_generate_html.sbatch \
    "$OUTPUT_DIR" "$HTML_DIR" "$EXPERIMENT")

echo "Step 3 submitted (job $STEP3_JID) with dependency on step 2 jobs"
echo ""
echo "============================================================"
echo "Full pipeline submitted!"
echo "============================================================"
echo ""
echo "Experiment:  $EXPERIMENT"
echo "Step 2 jobs: ${STEP2_JOBS#:}"
echo "Step 3 job:  $STEP3_JID (HTML generation, depends on all step 2)"
echo ""
echo "Output:      $OUTPUT_DIR"
echo "HTML:        $HTML_DIR/"
echo "Annotations: annotations_${EXPERIMENT}.json"
echo ""
echo "Monitor with: squeue -u \$USER"
