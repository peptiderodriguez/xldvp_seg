#!/bin/bash
#SBATCH --job-name=stain_consensus
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/stain_consensus_%j.out
#SBATCH --error=logs/stain_consensus_%j.err

MKSEG_PYTHON="${MKSEG_PYTHON:-python}"

cd /path/to/output

# All 16 slides
SLIDES=(
    "/path/to/data/czi_slides/2025_11_18_FGC1.czi"
    "/path/to/data/czi_slides/2025_11_18_FGC2.czi"
    "/path/to/data/czi_slides/2025_11_18_FGC3.czi"
    "/path/to/data/czi_slides/2025_11_18_FGC4.czi"
    "/path/to/data/czi_slides/2025_11_18_FHU1.czi"
    "/path/to/data/czi_slides/2025_11_18_FHU2.czi"
    "/path/to/data/czi_slides/2025_11_18_FHU3.czi"
    "/path/to/data/czi_slides/2025_11_18_FHU4.czi"
    "/path/to/data/czi_slides/2025_11_18_MGC1.czi"
    "/path/to/data/czi_slides/2025_11_18_MGC2.czi"
    "/path/to/data/czi_slides/2025_11_18_MGC3.czi"
    "/path/to/data/czi_slides/2025_11_18_MGC4.czi"
    "/path/to/data/czi_slides/2025_11_18_MHU1.czi"
    "/path/to/data/czi_slides/2025_11_18_MHU2.czi"
    "/path/to/data/czi_slides/2025_11_18_MHU3.czi"
    "/path/to/data/czi_slides/2025_11_18_MHU4.czi"
)

python compute_stain_consensus.py \
    --slides "${SLIDES[@]}" \
    --output stain_normalizer_reinhard.pkl \
    --n-patches 50 \
    --patch-size 256 \
    --method reinhard

echo "Consensus stain normalization parameters computed!"
