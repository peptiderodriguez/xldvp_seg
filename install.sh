#!/bin/bash
# =============================================================================
# xldvp_seg Installation Script
# =============================================================================
#
# This script installs the xldvp_seg package and all dependencies including:
# - PyTorch with CUDA support
# - SAM2 (Segment Anything Model 2) from Facebook Research
# - All other Python dependencies
#
# Usage:
#   # Create conda environment first (recommended):
#   conda create -n xldvp_seg python=3.10 -y
#   conda activate xldvp_seg
#
#   # Then run this script:
#   ./install.sh
#
# Options:
#   --latest                 Use latest compatible versions (auto-detect CUDA)
#   --cuda 11.8|12.1|12.4   Specify CUDA version (implies --latest)
#   --rocm                   Install for AMD GPUs (ROCm, implies --latest)
#   --cpu                    CPU-only installation (implies --latest)
#   --dev                    Install development dependencies
#
# By default, installs exact pinned versions from requirements-lock.txt
# for reproducibility. Use --latest to auto-detect CUDA and grab the
# latest compatible versions instead.
#
# =============================================================================

set -e

# Default values
CUDA_VERSION=""
ROCM=false
CPU_ONLY=false
DEV=false
LATEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --latest)
            LATEST=true
            shift
            ;;
        --cuda)
            CUDA_VERSION="$2"
            LATEST=true
            shift 2
            ;;
        --rocm)
            ROCM=true
            LATEST=true
            shift
            ;;
        --cpu)
            CPU_ONLY=true
            LATEST=true
            shift
            ;;
        --dev)
            DEV=true
            shift
            ;;
        --reproducible)
            # Kept for backwards compat — this is now the default
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "xldvp_seg Installation"
echo "============================================================"

# Default: reproducible install from lock file
# Use --latest to opt into auto-detect CUDA + latest versions
if [ "$LATEST" = false ]; then
    echo ""
    echo "REPRODUCIBLE INSTALL (default): exact pinned versions from requirements-lock.txt"
    echo "Use './install.sh --latest' to auto-detect CUDA and install latest versions instead."
    echo "------------------------------------------------------------"
    LOCK_FILE="$SCRIPT_DIR/requirements-lock.txt"
    if [ ! -f "$LOCK_FILE" ]; then
        echo "ERROR: requirements-lock.txt not found at $LOCK_FILE"
        echo "Run './install.sh --latest' to install without the lock file."
        exit 1
    fi
    pip install -r "$LOCK_FILE"
    if [ "$DEV" = true ]; then
        pip install -e "$SCRIPT_DIR[dev]"
    else
        pip install -e "$SCRIPT_DIR"
    fi
    echo ""
    echo "============================================================"
    echo "Reproducible installation complete!"
    echo "All packages pinned to exact versions from requirements-lock.txt."
    echo "============================================================"
    exit 0
fi

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
echo "Step 3: Installing xldvp_seg package..."
echo "------------------------------------------------------------"

if [ "$DEV" = true ]; then
    pip install -e ".[dev]"
else
    pip install -e .
fi

# Install optional but recommended packages
echo ""
echo "Step 4: Installing optional packages..."
echo "------------------------------------------------------------"
echo "Installing py-lmd (required for LMD XML export)..."
if ! pip install py-lmd; then
    echo ""
    echo "  WARNING: py-lmd failed to install."
    echo "  Impact: LMD export (run_lmd_export.py) will not work."
    echo "  Fix: pip install py-lmd"
    echo ""
fi

echo "Installing orjson (3-5x faster JSON parsing for large detection files)..."
if ! pip install orjson; then
    echo ""
    echo "  WARNING: orjson failed to install."
    echo "  Impact: JSON loading will use stdlib (slower for files >100MB)."
    echo "  Fix: pip install orjson  (requires Rust compiler on some platforms)"
    echo ""
fi

echo "Installing plotly (interactive UMAP/t-SNE HTML viewer with hover)..."
if ! pip install plotly; then
    echo ""
    echo "  WARNING: plotly failed to install."
    echo "  Impact: Interactive UMAP HTML (cluster_by_features.py --interactive) will not"
    echo "          be generated. Static PNG plots still work."
    echo "  Fix: pip install plotly"
    echo ""
fi

echo "Installing fa2-modified (fast ForceAtlas2 layout for trajectory analysis)..."
if ! pip install fa2-modified; then
    echo ""
    echo "  WARNING: fa2-modified failed to install."
    echo "  Impact: Trajectory force-directed layout (cluster_by_features.py --trajectory)"
    echo "          will fall back to slow Fruchterman-Reingold layout."
    echo "  Fix: pip install fa2-modified"
    echo ""
fi

echo "Installing napari (interactive viewer for cross placement + LMD overlay)..."
if ! pip install napari[all]; then
    echo ""
    echo "  WARNING: napari failed to install."
    echo "  Impact: Cross placement (napari_place_crosses.py) and LMD overlay"
    echo "          (napari_view_lmd_export.py) will not work. Detection, classification,"
    echo "          and HTML viewers are unaffected."
    echo "  Fix: pip install 'napari[all]'  (requires Qt backend — try: conda install napari -c conda-forge)"
    echo ""
fi

# Download SAM2 checkpoint
echo ""
echo "Step 5: Downloading SAM2 checkpoint (~890 MB)..."
echo "------------------------------------------------------------"

CHECKPOINT_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/sam2.1_hiera_large.pt"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "SAM2 checkpoint already exists at $CHECKPOINT_FILE — skipping download."
else
    mkdir -p "$CHECKPOINT_DIR"
    if command -v wget &> /dev/null; then
        wget -O "$CHECKPOINT_FILE" \
            https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    elif command -v curl &> /dev/null; then
        curl -L -o "$CHECKPOINT_FILE" \
            https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    else
        echo "WARNING: Neither wget nor curl found. Download the SAM2 checkpoint manually:"
        echo "  mkdir -p $CHECKPOINT_DIR"
        echo "  wget -O $CHECKPOINT_FILE https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fi
fi

# Install cloudflared for remote HTML viewing (optional)
echo ""
echo "Step 6: Installing cloudflared (for remote viewing)..."
echo "------------------------------------------------------------"

if command -v cloudflared &> /dev/null; then
    echo "cloudflared already installed — skipping."
elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    mkdir -p ~/.local/bin
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared
    echo "Installed cloudflared to ~/.local/bin/cloudflared"
elif [ "$(uname -s)" = "Darwin" ]; then
    if command -v brew &> /dev/null; then
        brew install cloudflared 2>/dev/null || echo "Note: brew install cloudflared failed (optional)"
    else
        echo "Note: Install cloudflared manually (brew install cloudflared) for remote HTML viewing"
    fi
else
    echo "Note: Install cloudflared manually for remote HTML viewing"
    echo "  See: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
fi

# Verify installation
echo ""
echo "Step 7: Verifying installation..."
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
from xldvp_seg.processing.multigpu_worker import MultiGPUTileProcessor
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

# Check SAM2 checkpoint
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "  SAM2 checkpoint: OK ($CHECKPOINT_FILE)"
else
    echo "  SAM2 checkpoint: MISSING — download it before running detection"
fi

# Check cloudflared
if command -v cloudflared &> /dev/null || [ -f ~/.local/bin/cloudflared ]; then
    echo "  cloudflared: OK"
else
    echo "  cloudflared: not installed (optional — needed for remote HTML viewing)"
fi

echo ""
echo "============================================================"
echo "Installation complete!"
echo "============================================================"
echo ""
echo "Quick start:"
echo "  # With Claude Code (recommended):"
echo "  npm install -g @anthropic-ai/claude-code && claude"
echo "  # Then type: /analyze"
echo ""
echo "  # Without Claude Code:"
echo "  python scripts/czi_info.py /path/to/slide.czi    # inspect channels first"
echo "  python run_segmentation.py --czi-path /path/to/slide.czi --cell-type cell --channel-spec 'cyto=PM,nuc=488' --all-channels"
echo ""
echo "  # SLURM batch:"
echo "  scripts/run_pipeline.sh configs/my_experiment.yaml"
echo ""
