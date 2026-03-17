#!/bin/bash
#SBATCH --job-name=nmj_clf
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/xldvp_seg/slurm/logs/nmj_clf_%j.out
#SBATCH --error=/path/to/xldvp_seg/slurm/logs/nmj_clf_%j.err

# Paths
MKSEG_PYTHON="${MKSEG_PYTHON:-python}"
REPO="${REPO:-/path/to/xldvp_seg}"
PYTHON="$MKSEG_PYTHON"
RUN_DIR=/path/to/output/nmj_output/RUN_DIR

DETECTIONS=$RUN_DIR/nmj_detections.json
ANNOTATIONS=$RUN_DIR/nmj_20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch_1770728512_annotations.json
OUTPUT_DIR=$RUN_DIR/classifier

echo "=========================================="
echo "NMJ Classifier Training"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Detections: $DETECTIONS"
echo "Annotations: $ANNOTATIONS"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

cd $REPO
PYTHONPATH=$REPO $PYTHON train_classifier.py \
    --detections "$DETECTIONS" \
    --annotations "$ANNOTATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --feature-set all \
    --n-estimators 200

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
