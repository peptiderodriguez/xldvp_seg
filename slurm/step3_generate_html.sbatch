#!/bin/bash
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1
#
# STEP 3: Generate combined HTML annotation pages from saved crops
#
# This script is auto-submitted by step2_launch_parallel_normalized.sh
# with --dependency=afterok on all step 2 jobs.
#
# Usage (standalone):
#   sbatch step3_generate_html.sbatch OUTPUT_DIR HTML_DIR EXPERIMENT_NAME
#
# Args passed positionally from the launcher:
#   $1 = OUTPUT_DIR (segmentation output with features.json files)
#   $2 = HTML_DIR   (where to write combined HTML)
#   $3 = EXPERIMENT (experiment name for localStorage isolation)

OUTPUT_DIR="${1:?ERROR: OUTPUT_DIR required as first argument}"
HTML_DIR="${2:?ERROR: HTML_DIR required as second argument}"
EXPERIMENT="${3:?ERROR: EXPERIMENT name required as third argument}"

echo "============================================================"
echo "STEP 3: Generate Combined HTML"
echo "============================================================"
echo "Output dir:      $OUTPUT_DIR"
echo "HTML dir:        $HTML_DIR"
echo "Experiment:      $EXPERIMENT"
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "Start time:      $(date)"
echo "============================================================"

cd /viper/ptmp2/edrod/xldvp_seg_fresh

module load python-waterboa/2024.06
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1

python generate_combined_html.py \
    --output-dir "$OUTPUT_DIR" \
    --html-output-dir "$HTML_DIR" \
    --experiment-name "$EXPERIMENT" \
    --samples-per-page 300 \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000

echo ""
echo "Step 3 completed at $(date)"
echo "HTML output: $HTML_DIR/"
