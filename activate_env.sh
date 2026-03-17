#!/bin/bash
# Quick activation script for mkseg ROCm environment

module load python-waterboa/2024.06
module load rocm/6.3
source /path/to/mkseg_env/bin/activate

echo "✓ Activated mkseg_rocm environment"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Ready to run segmentation pipeline!"
