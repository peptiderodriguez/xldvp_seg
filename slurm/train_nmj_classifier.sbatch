#!/bin/bash
#SBATCH --job-name=nmj_clf
#SBATCH --partition=p.hpcl93
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_clf_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/slurm/logs/nmj_clf_%j.err

# Paths
REPO=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg
PYTHON=/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/bin/python
RUN_DIR=/fs/pool/pool-mann-edwin/nmj_output/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch_20260210_112830

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
