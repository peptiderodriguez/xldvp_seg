#!/bin/bash
#SBATCH --job-name=html_combined
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/path/to/output/logs/html_combined_%j.out
#SBATCH --error=/path/to/output/logs/html_combined_%j.err

echo "============================================================"
echo "Generate Combined HTML for All 16 Slides"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Environment
MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
cd /path/to/output

OUTPUT_DIR="/path/to/output/unified_10pct"
CZI_DIR="/path/to/data/czi_slides"

echo "============================================================"
echo "Processing MK detections..."
echo "============================================================"
python generate_combined_html.py \
    --output-dir "$OUTPUT_DIR" \
    --czi-dir "$CZI_DIR" \
    --cell-type mk \
    --channel 0 \
    --sort-by area \
    --sort-order desc \
    --samples-per-page 300 \
    --experiment-name mi300a_10pct_all16

echo ""
echo "============================================================"
echo "Processing HSPC detections..."
echo "============================================================"
python generate_combined_html.py \
    --output-dir "$OUTPUT_DIR" \
    --czi-dir "$CZI_DIR" \
    --cell-type hspc \
    --channel 1 \
    --sort-by area \
    --sort-order desc \
    --samples-per-page 300 \
    --experiment-name mi300a_10pct_all16

echo ""
echo "============================================================"
echo "Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "View results at:"
echo "  MK:   $OUTPUT_DIR/html_combined/mk/index.html"
echo "  HSPC: $OUTPUT_DIR/html_combined/hspc/index.html"
