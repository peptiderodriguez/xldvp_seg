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

cd /viper/ptmp2/edrod/xldvp_seg_fresh

module load python-waterboa/2024.06
source mkseg_rocm_env/bin/activate

# All 16 slides
SLIDES=(
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_FHU4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MGC4.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU1.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU2.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU3.czi"
    "/viper/ptmp2/edrod/2025_11_18/2025_11_18_MHU4.czi"
)

python compute_stain_consensus.py \
    --slides "${SLIDES[@]}" \
    --output stain_normalizer_reinhard.pkl \
    --n-patches 50 \
    --patch-size 256 \
    --method reinhard

echo "Consensus stain normalization parameters computed!"
