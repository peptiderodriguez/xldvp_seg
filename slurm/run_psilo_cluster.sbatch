#!/bin/bash
#SBATCH --job-name=wm_pm_cluster
#SBATCH --partition=p.hpcl8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/wm_pm_cluster_%j.out
#SBATCH --error=/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/logs/wm_pm_cluster_%j.err

# =============================================================================
# Whole Mouse PM — feature-based clustering (366K detections)
# Shape + SAM2 features only (no color — ch3 Pdgfra is bad)
# No GPU needed
# =============================================================================

set -e

DETECTIONS="/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/tissue_pattern_detections.json"
OUTPUT="/fs/pool/pool-mann-edwin/psilo_output/tp_full/20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct"

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Project: Whole Mouse PM — Feature Clustering"
echo "Detections: $DETECTIONS"
echo "============================================================"

source /fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

cd /fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg

echo "CPUs: $(nproc), RAM: $(free -h | awk '/^Mem:/{print $2}')"

export PYTHONUNBUFFERED=1

python scripts/cluster_by_features.py \
    --detections "$DETECTIONS" \
    --output-dir "$OUTPUT" \
    --feature-groups "shape,sam2" \
    --exclude-channels 3 \
    --min-cluster-size 50 \
    --threshold 0.0

echo "=== Clustering complete at $(date) ==="
