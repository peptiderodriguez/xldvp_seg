#!/bin/bash
# Quick activation script for xldvp_seg ROCm environment

module load python-waterboa/2024.06
module load rocm/6.3
source /path/to/xldvp_seg_env/bin/activate

echo "Activated xldvp_seg_rocm environment"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Ready to run segmentation pipeline!"
