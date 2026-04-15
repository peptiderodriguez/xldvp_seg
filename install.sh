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
#   conda create -n xldvp_seg python=3.11 -y
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

    # Step A: pin numpy early. Torch wheels declare `numpy>=1.22` with no upper
    # bound, and without a pre-pin pip pulls numpy 2.x — which breaks several
    # compiled transitive deps (cv2, spatialdata). Extract the exact version
    # from the lock file and install it first.
    NUMPY_PIN=$(grep -E '^numpy==' "$LOCK_FILE")
    echo "Step A: pinning $NUMPY_PIN first (prevents torch pulling in numpy 2.x)..."
    pip install "$NUMPY_PIN"

    # Step B: install torch with the right index. Detect CUDA so GPU users don't
    # silently land on CPU wheels. If the user passed --cpu, --rocm, or --cuda,
    # those override detection.
    TORCH_PIN=$(grep -E '^torch==' "$LOCK_FILE")
    TORCHVISION_PIN=$(grep -E '^torchvision==' "$LOCK_FILE")
    if [ "$CPU_ONLY" = true ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        echo "Step B: installing CPU PyTorch ($TORCH_PIN) ..."
    elif [ "$ROCM" = true ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/rocm6.0"
        echo "Step B: installing ROCm PyTorch ($TORCH_PIN) ..."
    else
        # Auto-detect CUDA from nvidia-smi
        if command -v nvidia-smi &> /dev/null; then
            DETECTED_CUDA=$(nvidia-smi | grep "CUDA Version" | sed -E 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/' | head -1)
            case "$DETECTED_CUDA" in
                11.*)      TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
                12.0|12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
                12.*)      TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
                13.*)      TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;  # torch has no cu130 wheels yet; cu124 is backward-compat
                *)         TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
            esac
            echo "Step B: detected CUDA $DETECTED_CUDA — installing $TORCH_PIN from $TORCH_INDEX ..."
        elif command -v sinfo &> /dev/null; then
            # SLURM present but nvidia-smi missing (typical login node on a GPU cluster).
            # Defaulting to CUDA 12.4 is the right call for modern GPU clusters —
            # users almost never mean "install CPU torch on my HPC cluster".
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            echo "Step B: SLURM detected but no nvidia-smi on this node (login?) —"
            echo "        assuming compute nodes have CUDA GPUs; using CUDA 12.4 wheel."
            echo "        (Override with './install.sh --cuda 11.8|12.1|12.4' or '--cpu'.)"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            echo "Step B: no nvidia-smi, no SLURM — installing CPU PyTorch ($TORCH_PIN) ..."
        fi
    fi
    # Uninstall any previous torch so re-running with a different backend
    # (e.g. first --cpu, then --cuda 12.4) actually swaps the wheel instead of
    # silently no-op'ing because torch is already installed.
    pip uninstall -y torch torchvision 2>/dev/null || true
    pip install "$TORCH_PIN" "$TORCHVISION_PIN" --index-url "$TORCH_INDEX"

    # Step C: install the rest of the lock. numpy + torch already satisfied,
    # so pip skips them. Everything else resolves against the exact pins.
    echo "Step C: installing remaining dependencies from lock file..."
    pip install -r "$LOCK_FILE"

    # Step D: install our package (editable) without re-resolving deps.
    if [ "$DEV" = true ]; then
        pip install --no-deps -e "$SCRIPT_DIR[dev]"
    else
        pip install --no-deps -e "$SCRIPT_DIR"
    fi

    # Step E: SAM2 — required for detection. Not on PyPI, install from git.
    # --no-deps: SAM2's runtime deps (torch, hydra-core, iopath, pillow) are all
    # already satisfied by the lock; re-resolving triggers ResolutionTooDeep.
    echo "Step E: installing SAM2 (--no-deps; transitive deps already in lock)..."
    pip install --no-deps "git+https://github.com/facebookresearch/segment-anything-2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"

    # Step F: SAM2 checkpoint (700MB). Skip if already present.
    CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
    CHECKPOINT_FILE="$CHECKPOINT_DIR/sam2.1_hiera_large.pt"
    if [ ! -f "$CHECKPOINT_FILE" ]; then
        mkdir -p "$CHECKPOINT_DIR"
        echo "Step F: downloading SAM2 checkpoint (700MB)..."
        if command -v curl &> /dev/null; then
            curl -L -o "$CHECKPOINT_FILE" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
        elif command -v wget &> /dev/null; then
            wget -O "$CHECKPOINT_FILE" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
        else
            echo "  WARNING: curl/wget not found — download manually to $CHECKPOINT_FILE"
        fi
    else
        echo "Step F: SAM2 checkpoint already present at $CHECKPOINT_FILE"
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
                13.*) CUDA_VERSION="12.4" ;;  # no cu130 wheels yet; cu124 is backward-compat
                *) CUDA_VERSION="12.4" ;;    # default to newest for unrecognized (was 12.1)
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

# If torch is already installed, uninstall it so the new backend (CPU/CUDA/ROCm)
# can actually be applied. Without this, re-running install.sh with a different
# backend flag is a silent no-op (pip sees torch already installed and skips).
pip uninstall -y torch torchvision 2>/dev/null || true

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
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install SAM2
echo ""
echo "Step 2: Installing SAM2..."
echo "------------------------------------------------------------"
pip install git+https://github.com/facebookresearch/segment-anything-2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4

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

python3 -c "
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
print(f'  cellpose: {getattr(cellpose, \"__version__\", None) or cellpose.version}')
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
