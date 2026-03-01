#!/bin/bash
# =============================================================================
# MKSeg Installation Script
# =============================================================================
#
# This script installs the mkseg package and all dependencies including:
# - PyTorch with CUDA support
# - SAM2 (Segment Anything Model 2) from Facebook Research
# - All other Python dependencies
#
# Usage:
#   # Create conda environment first (recommended):
#   conda create -n mkseg python=3.11 -y
#   conda activate mkseg
#
#   # Then run this script:
#   ./install.sh
#
# Options:
#   --cuda 11.8|12.1|12.4   Specify CUDA version (default: auto-detect or 12.1)
#   --rocm                   Install for AMD GPUs (ROCm)
#   --cpu                    CPU-only installation (no GPU)
#   --dev                    Install development dependencies
#
# =============================================================================

set -e

# Default values
CUDA_VERSION=""
ROCM=false
CPU_ONLY=false
DEV=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --rocm)
            ROCM=true
            shift
            ;;
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --dev)
            DEV=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "MKSeg Installation"
echo "============================================================"

# Detect CUDA version if not specified
if [ -z "$CUDA_VERSION" ] && [ "$ROCM" = false ] && [ "$CPU_ONLY" = false ]; then
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_CUDA=$(nvidia-smi | grep "CUDA Version" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/')
        if [ -n "$DETECTED_CUDA" ]; then
            echo "Detected CUDA version: $DETECTED_CUDA"
            # Map to PyTorch-supported versions
            case $DETECTED_CUDA in
                11.*) CUDA_VERSION="11.8" ;;
                12.0|12.1) CUDA_VERSION="12.1" ;;
                12.*) CUDA_VERSION="12.4" ;;
                *) CUDA_VERSION="12.1" ;;
            esac
        fi
    fi

    if [ -z "$CUDA_VERSION" ]; then
        echo "Could not detect CUDA. Defaulting to CUDA 12.1"
        CUDA_VERSION="12.1"
    fi
fi

# Install PyTorch
echo ""
echo "Step 1: Installing PyTorch..."
echo "------------------------------------------------------------"

if [ "$CPU_ONLY" = true ]; then
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
elif [ "$ROCM" = true ]; then
    echo "Installing PyTorch with ROCm support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
else
    echo "Installing PyTorch with CUDA $CUDA_VERSION..."
    case $CUDA_VERSION in
        11.8)
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            ;;
        12.1)
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            ;;
        12.4)
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
            ;;
        *)
            echo "Unsupported CUDA version: $CUDA_VERSION"
            echo "Supported versions: 11.8, 12.1, 12.4"
            exit 1
            ;;
    esac
fi

# Verify PyTorch installation
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install SAM2
echo ""
echo "Step 2: Installing SAM2..."
echo "------------------------------------------------------------"
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Install this package
echo ""
echo "Step 3: Installing mkseg package..."
echo "------------------------------------------------------------"

if [ "$DEV" = true ]; then
    pip install -e ".[dev]"
else
    pip install -e .
fi

# Verify installation
echo ""
echo "Step 4: Verifying installation..."
echo "------------------------------------------------------------"

python -c "
import torch
import cellpose
import numpy
import cv2
import h5py
import anndata
import scanpy
import spatialdata
import squidpy
import geopandas
from segmentation.processing.multigpu_worker import MultiGPUTileProcessor
print('All imports successful!')
print(f'  torch: {torch.__version__}')
print(f'  cellpose: {cellpose.__version__}')
print(f'  numpy: {numpy.__version__}')
print(f'  cv2: {cv2.__version__}')
print(f'  anndata: {anndata.__version__}')
print(f'  scanpy: {scanpy.__version__}')
print(f'  spatialdata: {spatialdata.__version__}')
print(f'  squidpy: {squidpy.__version__}')
print(f'  geopandas: {geopandas.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""
echo "Usage:"
echo "  # NMJ detection (single node, 2 GPUs):"
echo "  python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --channel 1 --num-gpus 2"
echo ""
echo "  # MK detection (single node, 4 GPUs):"
echo "  python run_segmentation.py --czi-path /path/to/slide.czi --cell-type mk --channel 0 --num-gpus 4"
echo ""
echo "  # Vessel detection:"
echo "  python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 --candidate-mode"
echo ""
echo "  # SLURM chain launcher (multi-node):"
echo "  bash slurm/launch_pipeline.sh --czi /path/to/slide.czi --cell-type nmj --channel 1 --nodes 4 --steps detect,merge,html"
echo ""
