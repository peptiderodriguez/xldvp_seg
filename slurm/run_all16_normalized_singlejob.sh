#!/bin/bash
#SBATCH --job-name=mkseg_all16_normalized
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=230G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/all16_normalized_%j.out
#SBATCH --error=/viper/ptmp2/edrod/xldvp_seg_fresh/logs/all16_normalized_%j.err

echo "============================================================"
echo "Process all 16 slides with GLOBAL CROSS-SLIDE NORMALIZATION"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

module load python-waterboa/2024.06
module load rocm/6.3
source /viper/ptmp2/edrod/xldvp_seg_fresh/mkseg_rocm_env/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export HSA_OVERRIDE_GFX_VERSION=9.4.2

cd /viper/ptmp2/edrod/xldvp_seg_fresh

OUTPUT_DIR="/viper/ptmp2/edrod/unified_10pct_mi300a_normalized"
CZI_DIR="/viper/ptmp2/edrod/2025_11_18"

python run_unified_FAST.py \
    --czi-paths \
        "$CZI_DIR/2025_11_18_FGC1.czi" \
        "$CZI_DIR/2025_11_18_FGC2.czi" \
        "$CZI_DIR/2025_11_18_FGC3.czi" \
        "$CZI_DIR/2025_11_18_FGC4.czi" \
        "$CZI_DIR/2025_11_18_FHU1.czi" \
        "$CZI_DIR/2025_11_18_FHU2.czi" \
        "$CZI_DIR/2025_11_18_FHU3.czi" \
        "$CZI_DIR/2025_11_18_FHU4.czi" \
        "$CZI_DIR/2025_11_18_MGC1.czi" \
        "$CZI_DIR/2025_11_18_MGC2.czi" \
        "$CZI_DIR/2025_11_18_MGC3.czi" \
        "$CZI_DIR/2025_11_18_MGC4.czi" \
        "$CZI_DIR/2025_11_18_MHU1.czi" \
        "$CZI_DIR/2025_11_18_MHU2.czi" \
        "$CZI_DIR/2025_11_18_MHU3.czi" \
        "$CZI_DIR/2025_11_18_MHU4.czi" \
    --output-dir "$OUTPUT_DIR" \
    --tile-size 3000 \
    --sample-fraction 0.10 \
    --multi-gpu \
    --num-gpus 2 \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --hspc-min-area-um 25 \
    --hspc-max-area-um 150 \
    --cleanup-masks \
    --normalize-slides \
    --norm-percentile-low 1.0 \
    --norm-percentile-high 99.0

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
