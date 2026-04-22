#!/bin/bash
# Sourced by GPU sbatch files. Exports LD_LIBRARY_PATH so cuML + cupy
# pip-wheels find their CUDA-11 shared libraries under the mkseg env.
#
# Usage:  source scripts/slurm/nvidia_ld_library_path.sh

NVIDIA_LIBS="/fs/gpfs41/lv07/fileset03/home/b_mann/rodriguez/miniforge3/envs/mkseg/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}/cuda_runtime/lib:${NVIDIA_LIBS}/cuda_nvrtc/lib:${NVIDIA_LIBS}/cublas/lib:${NVIDIA_LIBS}/cufft/lib:${NVIDIA_LIBS}/curand/lib:${NVIDIA_LIBS}/cusolver/lib:${NVIDIA_LIBS}/cusparse/lib:${NVIDIA_LIBS}/cudnn/lib:${NVIDIA_LIBS}/nccl/lib:${NVIDIA_LIBS}/nvjitlink/lib:${NVIDIA_LIBS}/nvtx/lib:${NVIDIA_LIBS}/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"
