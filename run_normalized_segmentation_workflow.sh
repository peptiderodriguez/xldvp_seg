#!/bin/bash
# Master workflow script for normalized segmentation
# Runs in 2 steps:
#   1. Compute global normalization params from all 16 slides
#   2. Launch 8 parallel jobs that apply those params

echo "============================================================"
echo "NORMALIZED SEGMENTATION WORKFLOW"
echo "============================================================"
echo ""
echo "This workflow will:"
echo "  1. Compute global normalization parameters from all 16 slides"
echo "  2. Launch 8 parallel segmentation jobs (2 slides each)"
echo "  3. Each job applies the SAME global normalization"
echo ""
echo "Result: True cross-slide normalization + parallel processing!"
echo ""
echo "============================================================"
echo ""

cd /viper/ptmp2/edrod/xldvp_seg_fresh

# Step 1: Submit normalization parameter computation
echo "STEP 1: Submitting normalization parameter computation..."
NORM_JOB=$(sbatch --parsable slurm/step1_compute_norm_params.sbatch)

if [ -z "$NORM_JOB" ]; then
    echo "ERROR: Failed to submit step 1 job"
    exit 1
fi

echo "  Job ID: $NORM_JOB"
echo ""
echo "Waiting for normalization parameters to be computed..."
echo "(This will take ~10-30 minutes depending on queue)"
echo ""

# Wait for the normalization job to complete
while true; do
    # Check if job is still in queue or running
    JOB_STATE=$(squeue -j $NORM_JOB -h -o "%T" 2>/dev/null)

    if [ -z "$JOB_STATE" ]; then
        # Job no longer in queue - it completed
        break
    fi

    echo "  Status: $JOB_STATE (waiting...)"
    sleep 30
done

# Check if normalization params file was created
NORM_PARAMS="/viper/ptmp2/edrod/xldvp_seg_fresh/normalization_params_all16.json"
if [ ! -f "$NORM_PARAMS" ]; then
    echo ""
    echo "ERROR: Normalization parameters file not found!"
    echo "Check the log: logs/compute_norm_params_${NORM_JOB}.out"
    exit 1
fi

echo ""
echo "✓ Normalization parameters computed successfully!"
echo ""

# Show the parameters
echo "Global normalization parameters:"
cat "$NORM_PARAMS"
echo ""
echo "============================================================"
echo ""

# VALIDATION: Run validation script to visually confirm normalization
echo "VALIDATION: Running normalization validation..."
echo "This will:"
echo "  - Sample pixels from 5 representative slides"
echo "  - Generate intensity distribution plots"
echo "  - Confirm normalization parameters are correct"
echo ""

source mkseg_rocm_env/bin/activate
python validate_normalization.py

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: Validation script failed, but continuing anyway..."
    echo "Check validation_plots/ directory after workflow completes"
else
    echo ""
    echo "✓ Validation complete! Check validation_plots/ for results"
fi

echo ""
echo "============================================================"
echo ""

# Step 2: Launch parallel segmentation jobs
echo "STEP 2: Launching 8 parallel segmentation jobs..."
bash slurm/step2_launch_parallel_normalized.sh

echo ""
echo "============================================================"
echo "WORKFLOW COMPLETE!"
echo "============================================================"
echo ""
echo "Monitor progress with: squeue -u \$USER"
echo "Output will be in: /viper/ptmp2/edrod/unified_10pct_mi300a_normalized/"
