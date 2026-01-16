#!/usr/bin/env python3
"""
Unified Cell Segmentation Pipeline

A general-purpose pipeline for detecting and classifying cells in CZI microscopy images.
Supports multiple cell types with shared infrastructure and full feature extraction.

Cell Types:
    - nmj: Neuromuscular junctions (intensity + elongation filter)
    - mk: Megakaryocytes (SAM2 automatic mask generation)
    - cell: Hematopoietic stem/progenitor cells (Cellpose-SAM + SAM2 refinement)
    - vessel: Blood vessel cross-sections (ring detection via contour hierarchy)

Features extracted per cell: 2326 total
    - 22 morphological/intensity features
    - 256 SAM2 embedding features
    - 2048 ResNet-50 deep features
    + Cell-type specific features (elongation for NMJ, wall thickness for vessel, etc.)

Outputs:
    - {cell_type}_detections.json: All detections with universal IDs and global coordinates
    - {cell_type}_coordinates.csv: Quick export with center coordinates in pixels and µm
    - {cell_type}_masks.h5: Per-tile mask arrays
    - html/: Interactive HTML viewer for annotation

After processing, automatically starts HTTP server and Cloudflare tunnel for remote viewing.
Server runs in background by default - script exits but server keeps running.

Server options:
    --serve-background  Start server in background (default) - script exits, server persists
    --serve             Start server in foreground - wait for Ctrl+C to stop
    --no-serve          Don't start server
    --port PORT         HTTP server port (default: 8081)
    --stop-server       Stop any running background server and exit

Usage:
    # NMJ detection
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --channel 1

    # Vessel detection (SMA staining)
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --min-vessel-diameter 10 --max-vessel-diameter 500

    # Vessel with CD31 validation
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --cd31-channel 1
"""

import os
import gc
import re
import json
import argparse
import subprocess
import signal
import atexit
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from PIL import Image

# Import segmentation modules
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
)
from segmentation.io.html_export import (
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
    create_hdf5_dataset,  # Import shared HDF5 utilities
    HDF5_COMPRESSION_KWARGS,
    HDF5_COMPRESSION_NAME,
)
from segmentation.utils.logging import get_logger, setup_logging, log_parameters
from segmentation.utils.feature_extraction import (
    extract_resnet_features_batch,
    preprocess_crop_for_resnet,
    extract_morphological_features as _shared_extract_morphological_features,  # Import shared version
    compute_hsv_features,
)
from segmentation.io.czi_loader import get_loader, CZILoader

# Import new CellDetector and strategies
from segmentation.detection.cell_detector import CellDetector
from segmentation.detection.strategies.mk import MKStrategy
from segmentation.detection.strategies.nmj import NMJStrategy
from segmentation.detection.strategies.vessel import VesselStrategy
from segmentation.detection.strategies.cell import CellStrategy
from segmentation.detection.strategies.mesothelium import MesotheliumStrategy

logger = get_logger(__name__)

# Global list to track spawned processes for cleanup (foreground mode only)
_spawned_processes = []

# PID file location for background servers
SERVER_PID_FILE = Path.home() / '.segmentation_server.pid'


def _cleanup_processes():
    """Cleanup spawned HTTP server and tunnel processes on exit."""
    for proc in _spawned_processes:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            pass


atexit.register(_cleanup_processes)


def stop_background_server():
    """Stop any running background server using saved PID file."""
    if not SERVER_PID_FILE.exists():
        logger.info("No background server running (no PID file found)")
        return False

    try:
        data = json.loads(SERVER_PID_FILE.read_text())
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        stopped = False

        for name, pid in [('HTTP server', http_pid), ('Cloudflare tunnel', tunnel_pid)]:
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Stopped {name} (PID {pid})")
                    stopped = True
                except ProcessLookupError:
                    logger.info(f"{name} (PID {pid}) was not running")
                except PermissionError:
                    logger.error(f"Permission denied stopping {name} (PID {pid})")

        SERVER_PID_FILE.unlink()
        return stopped
    except Exception as e:
        logger.error(f"Error stopping server: {e}")
        return False


def start_server_and_tunnel(html_dir: Path, port: int = 8081, background: bool = False) -> tuple:
    """
    Start HTTP server and Cloudflare tunnel for viewing results.

    Args:
        html_dir: Path to the HTML directory to serve
        port: Port for HTTP server (default 8081)
        background: If True, detach processes so they survive script exit

    Returns:
        Tuple of (http_process, tunnel_process, tunnel_url)
    """
    global _spawned_processes

    html_dir = Path(html_dir)
    if not html_dir.exists():
        logger.error(f"HTML directory does not exist: {html_dir}")
        return None, None, None

    # Stop any existing background server first
    if background and SERVER_PID_FILE.exists():
        logger.info("Stopping existing background server...")
        stop_background_server()

    # Common args for background mode (detach from parent)
    bg_kwargs = {}
    if background:
        bg_kwargs = {
            'start_new_session': True,  # Detach from parent process group
            'stdin': subprocess.DEVNULL,
        }

    # Start HTTP server
    logger.info(f"Starting HTTP server on port {port}...")
    http_proc = subprocess.Popen(
        ['python', '-m', 'http.server', str(port)],
        cwd=str(html_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **bg_kwargs,
    )
    if not background:
        _spawned_processes.append(http_proc)
    time.sleep(1)  # Give server time to start

    if http_proc.poll() is not None:
        logger.error("HTTP server failed to start")
        return None, None, None

    logger.info(f"HTTP server running: http://localhost:{port}")

    # Start Cloudflare tunnel
    cloudflared_path = os.path.expanduser('~/cloudflared')
    if not os.path.exists(cloudflared_path):
        logger.warning("Cloudflare tunnel not found at ~/cloudflared")
        logger.info("Install with: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared && chmod +x ~/cloudflared")
        if background:
            # Save HTTP server PID even without tunnel
            SERVER_PID_FILE.write_text(json.dumps({
                'http_pid': http_proc.pid,
                'tunnel_pid': None,
                'port': port,
                'html_dir': str(html_dir),
                'url': None,
            }))
        return http_proc, None, None

    logger.info("Starting Cloudflare tunnel...")

    # For background mode, we need to capture output to get URL but still detach
    if background:
        # Create a log file for tunnel output
        tunnel_log = html_dir / '.tunnel.log'
        tunnel_log_file = open(tunnel_log, 'w')
        tunnel_proc = subprocess.Popen(
            [cloudflared_path, 'tunnel', '--url', f'http://localhost:{port}'],
            stdout=tunnel_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
        # Wait briefly and parse log for URL
        time.sleep(5)
        tunnel_log_file.flush()
        tunnel_url = None
        try:
            log_content = tunnel_log.read_text()
            match = re.search(r'(https://[^\s]+\.trycloudflare\.com)', log_content)
            if match:
                tunnel_url = match.group(1)
        except Exception:
            pass
    else:
        tunnel_proc = subprocess.Popen(
            [cloudflared_path, 'tunnel', '--url', f'http://localhost:{port}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _spawned_processes.append(tunnel_proc)

        # Wait for tunnel URL (usually appears within 5-10 seconds)
        tunnel_url = None
        start_time = time.time()
        while time.time() - start_time < 30:
            line = tunnel_proc.stdout.readline()
            if not line:
                if tunnel_proc.poll() is not None:
                    logger.error("Cloudflare tunnel exited unexpectedly")
                    break
                continue
            # Look for the tunnel URL in the output
            if 'trycloudflare.com' in line:
                match = re.search(r'(https://[^\s]+\.trycloudflare\.com)', line)
                if match:
                    tunnel_url = match.group(1)
                    break

    # Save PID file for background mode
    if background:
        SERVER_PID_FILE.write_text(json.dumps({
            'http_pid': http_proc.pid,
            'tunnel_pid': tunnel_proc.pid if tunnel_proc else None,
            'port': port,
            'html_dir': str(html_dir),
            'url': tunnel_url,
        }))

    if tunnel_url:
        logger.info("=" * 60)
        logger.info("REMOTE ACCESS AVAILABLE")
        logger.info("=" * 60)
        logger.info(f"Public URL: {tunnel_url}")
        logger.info(f"Local URL:  http://localhost:{port}")
        if background:
            logger.info("")
            logger.info("Server running in BACKGROUND")
            logger.info(f"To stop: python run_segmentation.py --stop-server")
            logger.info(f"PID file: {SERVER_PID_FILE}")
        else:
            logger.info("")
            logger.info("Press Ctrl+C to stop server and tunnel")
        logger.info("=" * 60)
    else:
        logger.warning("Could not get tunnel URL (tunnel may still be starting)")
        logger.info(f"Local URL: http://localhost:{port}")
        if background:
            logger.info(f"Check tunnel log: {html_dir / '.tunnel.log'}")

    return http_proc, tunnel_proc, tunnel_url


def wait_for_server_shutdown(http_proc, tunnel_proc):
    """Wait for user to press Ctrl+C, then cleanup."""
    if http_proc is None:
        return

    try:
        logger.info("Server running. Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if http_proc.poll() is not None:
                logger.warning("HTTP server stopped unexpectedly")
                break
            if tunnel_proc and tunnel_proc.poll() is not None:
                logger.warning("Tunnel stopped unexpectedly")
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        _cleanup_processes()
        logger.info("Server stopped")


# =============================================================================
# STRATEGY HELPER FUNCTIONS
# =============================================================================

def create_strategy_for_cell_type(cell_type, params, pixel_size_um):
    """
    Create the appropriate detection strategy for a cell type.

    Args:
        cell_type: One of 'nmj', 'mk', 'cell', 'vessel'
        params: Cell-type specific parameters dict
        pixel_size_um: Pixel size in microns

    Returns:
        DetectionStrategy instance

    Raises:
        ValueError: If cell_type is not supported by the new strategy pattern
    """
    if cell_type == 'nmj':
        return NMJStrategy(
            intensity_percentile=params.get('intensity_percentile', 99.0),
            min_area_px=params.get('min_area', 150),
            min_skeleton_length=params.get('min_skeleton_length', 30),
            max_solidity=params.get('max_solidity', 0.85),
        )
    elif cell_type == 'mk':
        # Convert area from pixels to um^2 for the strategy
        min_area_px = params.get('mk_min_area', 4132)  # ~200 um^2 at 0.22 um/px
        max_area_px = params.get('mk_max_area', 41322)  # ~2000 um^2 at 0.22 um/px
        min_area_um = min_area_px * (pixel_size_um ** 2)
        max_area_um = max_area_px * (pixel_size_um ** 2)
        return MKStrategy(
            min_area_um=min_area_um,
            max_area_um=max_area_um,
        )
    elif cell_type == 'cell':
        return CellStrategy(
            min_area_um=params.get('min_area_um', 50),
            max_area_um=params.get('max_area_um', 200),
        )
    elif cell_type == 'vessel':
        return VesselStrategy(
            min_diameter_um=params.get('min_vessel_diameter_um', 10),
            max_diameter_um=params.get('max_vessel_diameter_um', 1000),
            min_wall_thickness_um=params.get('min_wall_thickness_um', 2),
            max_aspect_ratio=params.get('max_aspect_ratio', 4.0),
            min_circularity=params.get('min_circularity', 0.3),
            min_ring_completeness=params.get('min_ring_completeness', 0.5),
            classify_vessel_types=params.get('classify_vessel_types', False),
        )
    elif cell_type == 'mesothelium':
        return MesotheliumStrategy(
            target_chunk_area_um2=params.get('target_chunk_area_um2', 1500.0),
            min_ribbon_width_um=params.get('min_ribbon_width_um', 5.0),
            max_ribbon_width_um=params.get('max_ribbon_width_um', 30.0),
            min_fragment_area_um2=params.get('min_fragment_area_um2', 1500.0),
            pixel_size_um=pixel_size_um,
        )
    else:
        raise ValueError(f"Cell type '{cell_type}' does not have a strategy implementation. "
                         f"Supported types: nmj, mk, cell, vessel, mesothelium")


def detections_to_features_list(detections, cell_type):
    """
    Convert a list of Detection objects to the expected features_list format.

    The old format is:
        features_list = [
            {
                'id': 'nmj_1',
                'center': [cx, cy],  # local coordinates
                'features': {...}    # morphological + deep features
            },
            ...
        ]

    For vessels, contours are at the top level (not in features):
        {
            'id': 'vessel_1',
            'center': [cx, cy],
            'outer_contour': [...],  # At top level
            'inner_contour': [...],  # At top level
            'features': {...}
        }

    Args:
        detections: List of Detection objects from strategy.detect()
        cell_type: Cell type string for ID prefix

    Returns:
        List of feature dicts in the old format
    """
    features_list = []
    for i, det in enumerate(detections, start=1):
        feat_dict = {
            'id': det.id if det.id else f'{cell_type}_{i}',
            'center': det.centroid,  # [x, y] format
            'features': det.features.copy(),
        }
        # Include score in features if available
        if det.score is not None:
            feat_dict['features']['score'] = det.score

        # For vessels, lift contours from features to top level (old format compatibility)
        if cell_type == 'vessel':
            if 'outer_contour' in feat_dict['features']:
                feat_dict['outer_contour'] = feat_dict['features'].pop('outer_contour')
            if 'inner_contour' in feat_dict['features']:
                feat_dict['inner_contour'] = feat_dict['features'].pop('inner_contour')

        features_list.append(feat_dict)
    return features_list


def detections_to_label_array(detections, shape):
    """
    Convert a list of Detection objects to a label array.

    Args:
        detections: List of Detection objects
        shape: (height, width) tuple for the output array

    Returns:
        uint32 array with labels
    """
    label_array = np.zeros(shape, dtype=np.uint32)
    for i, det in enumerate(detections, start=1):
        label_array[det.mask] = i
    return label_array


# NOTE: HDF5 compression utilities (create_hdf5_dataset, HDF5_COMPRESSION_KWARGS, HDF5_COMPRESSION_NAME)
# are now imported from segmentation.io.html_export (Issue #7 - consolidated)
# Alias for backwards compatibility
HDF5_COMPRESSION = HDF5_COMPRESSION_KWARGS


# =============================================================================
# FEATURE EXTRACTION (shared across all cell types)
# =============================================================================
# NOTE: extract_morphological_features is now imported from
# segmentation.utils.feature_extraction (Issue #7 - consolidated from 7 files)
# This alias preserves backwards compatibility for code in this file.
extract_morphological_features = _shared_extract_morphological_features


# =============================================================================
# UNIFIED SEGMENTER CLASS (DEPRECATED)
# =============================================================================
# NOTE: This class is deprecated. Use CellDetector with strategy classes instead:
#   from segmentation.detection.cell_detector import CellDetector
#   from segmentation.detection.strategies import MKStrategy, NMJStrategy, VesselStrategy, CellStrategy
#
#   detector = CellDetector(device="cuda")
#   strategy = MKStrategy(min_area_um=200, max_area_um=2000)
#   label_array, detections = strategy.detect(tile, detector.models, pixel_size_um)
#
# This class is kept for backwards compatibility with mesothelium detection,
# which does not yet have a strategy implementation.
# =============================================================================

class UnifiedSegmenter:
    """
    DEPRECATED: Use CellDetector with strategy classes instead.

    Unified segmenter for all cell types (MK, HSPC, NMJ, Vessel).

    Loads models once and provides cell-type specific detection with
    consistent feature extraction (2326 features per cell + type-specific).

    Detection methods:
        - NMJ: Intensity threshold + skeleton elongation filter
        - MK: SAM2 automatic mask generation + size filter
        - HSPC: Cellpose-SAM nuclei detection + SAM2 refinement
        - Vessel: Contour hierarchy for ring detection + ellipse fitting
    """

    def __init__(self, device=None, load_sam2=True, load_cellpose=True):
        """
        Initialize segmenter with required models.

        Args:
            device: torch device (auto-detect if None)
            load_sam2: Whether to load SAM2 (needed for MK, HSPC, and full features)
            load_cellpose: Whether to load Cellpose-SAM (needed for HSPC)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        logger.info(f"Initializing UnifiedSegmenter on {self.device}")

        # ResNet for deep features (always loaded - 2048D features)
        logger.info("  Loading ResNet-50...")
        resnet = tv_models.resnet50(weights='DEFAULT')
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.resnet.eval().to(self.device)
        self.resnet_transform = tv_transforms.Compose([
            tv_transforms.Resize(224),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # SAM2 for mask generation and embeddings
        self.sam2_auto = None
        self.sam2_predictor = None
        if load_sam2:
            self._load_sam2()

        # Cellpose-SAM for HSPC detection
        self.cellpose = None
        if load_cellpose:
            self._load_cellpose()

    def _load_sam2(self):
        """Load SAM2 models."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        # Find checkpoint
        script_dir = Path(__file__).parent.resolve()
        checkpoint_candidates = [
            script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
            Path("/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt"),
        ]
        checkpoint_path = None
        for cp in checkpoint_candidates:
            if cp.exists():
                checkpoint_path = cp
                break

        if checkpoint_path is None:
            logger.warning("SAM2 checkpoint not found, skipping SAM2")
            return

        logger.info(f"  Loading SAM2 from {checkpoint_path}...")
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=self.device)

        # Auto mask generator for MK detection
        self.sam2_auto = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=24,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.4,
            min_mask_region_area=500,
            crop_n_layers=1
        )

        # Predictor for point prompts (HSPC) and embeddings
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _load_cellpose(self):
        """Load Cellpose-SAM model."""
        from cellpose.models import CellposeModel

        logger.info("  Loading Cellpose-SAM...")
        self.cellpose = CellposeModel(pretrained_model='cpsam', gpu=True, device=self.device)

    def extract_resnet_features(self, crop):
        """Extract 2048D ResNet features from crop."""
        if crop.size == 0:
            return np.zeros(2048)

        # Convert uint16 to uint8 if needed (CZI images are often 16-bit)
        if crop.dtype == np.uint16:
            crop = (crop / 256).astype(np.uint8)
        elif crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)

        # Ensure RGB format (3 channels)
        if crop.ndim == 2:
            crop = np.stack([crop, crop, crop], axis=-1)
        elif crop.shape[-1] != 3:
            crop = np.ascontiguousarray(crop[..., :3])

        # Force PIL to use RGB mode
        try:
            pil_img = Image.fromarray(crop, mode='RGB')
            tensor = self.resnet_transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.resnet(tensor).cpu().numpy().flatten()
            return features
        except Exception as e:
            # Return zeros if feature extraction fails
            return np.zeros(2048)

    def extract_resnet_features_batch(self, crops, batch_size=16):
        """
        Extract ResNet features for multiple crops in batches for GPU efficiency.

        This method significantly improves GPU utilization by processing multiple
        crops at once instead of one at a time.

        Args:
            crops: List of image crops as numpy arrays
            batch_size: Number of crops to process at once (default 16)

        Returns:
            List of feature vectors as numpy arrays (2048D each)
        """
        return extract_resnet_features_batch(
            crops,
            self.resnet,
            self.resnet_transform,
            self.device,
            batch_size=batch_size
        )

    def extract_sam2_embedding(self, cy, cx):
        """Extract 256D SAM2 embedding at a point."""
        if self.sam2_predictor is None:
            return np.zeros(256)

        try:
            shape = self.sam2_predictor._features["image_embed"].shape
            emb_h, emb_w = shape[2], shape[3]
            img_h, img_w = self.sam2_predictor._orig_hw

            emb_y = int(cy / img_h * emb_h)
            emb_x = int(cx / img_w * emb_w)
            emb_y = min(max(emb_y, 0), emb_h - 1)
            emb_x = min(max(emb_x, 0), emb_w - 1)

            return self.sam2_predictor._features["image_embed"][0, :, emb_y, emb_x].cpu().numpy()
        except Exception as e:
            logger.debug(f"Failed to extract SAM2 embedding: {e}")
            return np.zeros(256)

    def extract_full_features(self, mask, image_rgb, cy, cx):
        """
        Extract all 2326 features for a detection.

        Args:
            mask: Binary mask
            image_rgb: RGB image
            cy, cx: Centroid coordinates

        Returns:
            Dict with all features
        """
        # 22 morphological features
        features = extract_morphological_features(mask, image_rgb)

        # 256 SAM2 embedding features
        sam2_emb = self.extract_sam2_embedding(cy, cx)
        for i, v in enumerate(sam2_emb):
            features[f'sam2_emb_{i}'] = float(v)

        # 2048 ResNet features from masked crop
        ys, xs = np.where(mask)
        if len(ys) > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = image_rgb[y1:y2+1, x1:x2+1].copy()
            crop_mask = mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self.extract_resnet_features(crop)
            for i, v in enumerate(resnet_feats):
                features[f'resnet_{i}'] = float(v)
        else:
            for i in range(2048):
                features[f'resnet_{i}'] = 0.0

        return features

    def extract_full_features_batch(self, masks_list, image_rgb, centroids, batch_size=16):
        """
        Extract all 2326 features for multiple detections with batch ResNet processing.

        This method improves GPU utilization by batching ResNet feature extraction
        across multiple detections, instead of processing one at a time.

        Args:
            masks_list: List of binary masks
            image_rgb: RGB image
            centroids: List of (cy, cx) tuples for each mask
            batch_size: Batch size for ResNet (default 16)

        Returns:
            List of feature dicts, one per detection
        """
        if not masks_list:
            return []

        # First pass: extract morphological and SAM2 features, collect crops
        all_features = []
        crops = []
        crop_indices = []  # Track which detections have valid crops

        for idx, (mask, (cy, cx)) in enumerate(zip(masks_list, centroids)):
            # 22 morphological features
            features = extract_morphological_features(mask, image_rgb)

            # 256 SAM2 embedding features
            sam2_emb = self.extract_sam2_embedding(cy, cx)
            for i, v in enumerate(sam2_emb):
                features[f'sam2_emb_{i}'] = float(v)

            # Prepare crop for batch ResNet processing
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                crop = image_rgb[y1:y2+1, x1:x2+1].copy()
                crop_mask = mask[y1:y2+1, x1:x2+1]
                crop[~crop_mask] = 0
                crops.append(crop)
                crop_indices.append(idx)
            else:
                # No valid crop - will fill with zeros later
                pass

            all_features.append(features)

        # Batch ResNet feature extraction
        if crops:
            resnet_features_list = self.extract_resnet_features_batch(crops, batch_size=batch_size)

            # Assign ResNet features to correct detections
            for crop_idx, resnet_feats in zip(crop_indices, resnet_features_list):
                for i, v in enumerate(resnet_feats):
                    all_features[crop_idx][f'resnet_{i}'] = float(v)

        # Fill zeros for detections without valid crops
        for idx in range(len(all_features)):
            if f'resnet_0' not in all_features[idx]:
                for i in range(2048):
                    all_features[idx][f'resnet_{i}'] = 0.0

        return all_features

    def detect_nmj(self, image_rgb, params, resnet_batch_size=16):
        """
        Detect NMJs using intensity threshold + elongation filter.

        Uses batch ResNet feature extraction for improved GPU utilization.

        Args:
            image_rgb: RGB image array
            params: Dict with intensity_percentile, min_area, min_skeleton_length, max_solidity
            resnet_batch_size: Batch size for ResNet feature extraction (default 16)

        Note: This method is DEPRECATED. Use NMJStrategy with CellDetector instead.

        Returns:
            Tuple of (masks, features_list)
        """
        from skimage.morphology import skeletonize, remove_small_objects, binary_opening, binary_closing, disk
        from skimage.measure import label, regionprops
        from scipy import ndimage

        # Convert to grayscale
        if image_rgb.ndim == 3:
            gray = np.mean(image_rgb[:, :, :3], axis=2)
        else:
            gray = image_rgb.astype(float)

        # Threshold bright regions
        threshold = np.percentile(gray, params['intensity_percentile'])
        bright_mask = gray > threshold

        # Morphological cleanup
        bright_mask = binary_opening(bright_mask, disk(1))
        bright_mask = binary_closing(bright_mask, disk(2))
        bright_mask = remove_small_objects(bright_mask, min_size=params['min_area'])

        # Label connected components
        labeled = label(bright_mask)
        props = regionprops(labeled, intensity_image=gray)

        # Debug: log candidate counts
        if len(props) > 0:
            logger.debug(f"    NMJ: {len(props)} candidates after threshold={threshold:.0f}")

        # First pass: filter by elongation and collect valid detections
        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        valid_detections = []  # (det_id, region_mask, cy, cx, skeleton_length, elongation, eccentricity, mean_intensity)
        det_id = 1

        for prop in props:
            if prop.area < params['min_area']:
                continue

            region_mask = labeled == prop.label
            skeleton = skeletonize(region_mask)
            skeleton_length = skeleton.sum()
            elongation = skeleton_length / np.sqrt(prop.area) if prop.area > 0 else 0

            # Use solidity filter (branched NMJs have low solidity)
            if skeleton_length >= params['min_skeleton_length'] and prop.solidity <= params.get('max_solidity', 0.85):
                masks[region_mask] = det_id
                cy, cx = prop.centroid

                valid_detections.append({
                    'det_id': det_id,
                    'mask': region_mask,
                    'cy': cy,
                    'cx': cx,
                    'skeleton_length': skeleton_length,
                    'elongation': elongation,
                    'eccentricity': prop.eccentricity,
                    'mean_intensity': prop.mean_intensity
                })
                det_id += 1

        # Set image for SAM2 embeddings if available
        if self.sam2_predictor is not None:
            self.sam2_predictor.set_image(image_rgb if image_rgb.dtype == np.uint8 else (image_rgb / 256).astype(np.uint8))

        # Batch feature extraction for all valid detections
        if valid_detections:
            masks_list = [d['mask'] for d in valid_detections]
            centroids = [(d['cy'], d['cx']) for d in valid_detections]
            all_features = self.extract_full_features_batch(
                masks_list, image_rgb, centroids, batch_size=resnet_batch_size
            )
        else:
            all_features = []

        # Build final features list
        features_list = []
        for det, features in zip(valid_detections, all_features):
            features['skeleton_length'] = int(det['skeleton_length'])
            features['elongation'] = float(det['elongation'])
            features['eccentricity'] = float(det['eccentricity'])
            features['mean_intensity'] = float(det['mean_intensity'])

            features_list.append({
                'id': f'nmj_{det["det_id"]}',
                'center': [float(det['cx']), float(det['cy'])],
                'features': features
            })

        # Clear SAM2 cache
        if self.sam2_predictor is not None:
            self.sam2_predictor.reset_predictor()

        return masks, features_list

    def detect_mk(self, image_rgb, params, resnet_batch_size=16):
        """
        Detect MKs using SAM2 automatic mask generation.

        Uses batch ResNet feature extraction for improved GPU utilization.

        Args:
            image_rgb: RGB image array
            params: Dict with mk_min_area, mk_max_area
            resnet_batch_size: Batch size for ResNet feature extraction (default 16)

        Returns:
            Tuple of (masks, features_list)
        """
        from scipy import ndimage

        if self.sam2_auto is None:
            raise RuntimeError("SAM2 not loaded - required for MK detection")

        # Generate masks (SAM2 expects uint8)
        sam2_img = image_rgb if image_rgb.dtype == np.uint8 else (image_rgb / 256).astype(np.uint8)
        sam2_results = self.sam2_auto.generate(sam2_img)

        # Filter by size
        valid_results = []
        for result in sam2_results:
            area = result['segmentation'].sum()
            if params['mk_min_area'] <= area <= params['mk_max_area']:
                result['area'] = area
                valid_results.append(result)
        valid_results.sort(key=lambda x: x['area'], reverse=True)

        del sam2_results
        gc.collect()

        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        valid_detections = []  # Collect valid detections for batch processing
        det_id = 1

        # Set image for embeddings
        self.sam2_predictor.set_image(image_rgb if image_rgb.dtype == np.uint8 else (image_rgb / 256).astype(np.uint8))

        # First pass: filter overlaps and collect valid detections
        for result in valid_results:
            mask = result['segmentation']
            if mask.dtype != bool:
                mask = (mask > 0.5).astype(bool)

            # Check overlap
            if masks.max() > 0:
                overlap = ((mask > 0) & (masks > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            masks[mask] = det_id
            cy, cx = ndimage.center_of_mass(mask)

            valid_detections.append({
                'det_id': det_id,
                'mask': mask,
                'cy': cy,
                'cx': cx,
                'sam2_iou': float(result.get('predicted_iou', 0)),
                'sam2_stability': float(result.get('stability_score', 0))
            })
            det_id += 1

        del valid_results
        gc.collect()

        # Batch feature extraction for all valid detections
        if valid_detections:
            masks_list = [d['mask'] for d in valid_detections]
            centroids = [(d['cy'], d['cx']) for d in valid_detections]
            all_features = self.extract_full_features_batch(
                masks_list, image_rgb, centroids, batch_size=resnet_batch_size
            )
        else:
            all_features = []

        # Build final features list
        features_list = []
        for det, features in zip(valid_detections, all_features):
            features['sam2_iou'] = det['sam2_iou']
            features['sam2_stability'] = det['sam2_stability']

            features_list.append({
                'id': f'mk_{det["det_id"]}',
                'center': [float(det['cx']), float(det['cy'])],
                'features': features
            })

        torch.cuda.empty_cache()
        self.sam2_predictor.reset_predictor()

        return masks, features_list

    def detect_cell(self, image_rgb, params, resnet_batch_size=16):
        """
        Detect HSPCs using Cellpose-SAM + SAM2 refinement.

        Uses batch ResNet feature extraction for improved GPU utilization.

        Args:
            image_rgb: RGB image array
            params: Dict (currently unused, Cellpose auto-detects)
            resnet_batch_size: Batch size for ResNet feature extraction (default 16)

        Returns:
            Tuple of (masks, features_list)
        """
        from scipy import ndimage

        if self.cellpose is None:
            raise RuntimeError("Cellpose not loaded - required for HSPC detection")
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 not loaded - required for HSPC detection")

        # Cellpose detection
        cellpose_masks, _, _ = self.cellpose.eval(image_rgb, channels=[0, 0])

        # Get centroids
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        # Limit candidates
        MAX_CANDIDATES = 500
        if len(cellpose_ids) > MAX_CANDIDATES:
            areas = [(cp_id, (cellpose_masks == cp_id).sum()) for cp_id in cellpose_ids]
            areas.sort(key=lambda x: x[1], reverse=True)
            cellpose_ids = np.array([a[0] for a in areas[:MAX_CANDIDATES]])

        # Set image for SAM2
        self.sam2_predictor.set_image(image_rgb if image_rgb.dtype == np.uint8 else (image_rgb / 256).astype(np.uint8))

        # Collect candidates with SAM2 refinement
        candidates = []
        for cp_id in cellpose_ids:
            cp_mask = cellpose_masks == cp_id
            cy, cx = ndimage.center_of_mass(cp_mask)

            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            masks_pred, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            best_idx = np.argmax(scores)
            sam2_mask = masks_pred[best_idx]
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)

            if sam2_mask.sum() < 10:
                continue

            candidates.append({
                'mask': sam2_mask,
                'score': float(scores[best_idx]),
                'center': (cx, cy),
                'cp_id': cp_id
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)

        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        valid_detections = []  # Collect valid detections for batch processing
        det_id = 1

        # First pass: filter overlaps and collect valid detections
        for cand in candidates:
            sam2_mask = cand['mask']
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)

            # Check overlap
            if masks.max() > 0:
                overlap = ((sam2_mask > 0) & (masks > 0)).sum()
                if overlap > 0.5 * sam2_mask.sum():
                    continue

            masks[sam2_mask] = det_id
            cx, cy = cand['center']

            valid_detections.append({
                'det_id': det_id,
                'mask': sam2_mask,
                'cy': cy,
                'cx': cx,
                'sam2_score': cand['score'],
                'cellpose_id': int(cand['cp_id'])
            })
            det_id += 1

        del candidates, cellpose_masks
        gc.collect()

        # Batch feature extraction for all valid detections
        if valid_detections:
            masks_list = [d['mask'] for d in valid_detections]
            centroids = [(d['cy'], d['cx']) for d in valid_detections]
            all_features = self.extract_full_features_batch(
                masks_list, image_rgb, centroids, batch_size=resnet_batch_size
            )
        else:
            all_features = []

        # Build final features list
        features_list = []
        for det, features in zip(valid_detections, all_features):
            features['sam2_score'] = det['sam2_score']
            features['cellpose_id'] = det['cellpose_id']

            features_list.append({
                'id': f'cell_{det["det_id"]}',
                'center': [float(det['cx']), float(det['cy'])],
                'features': features
            })

        self.sam2_predictor.reset_predictor()
        torch.cuda.empty_cache()

        return masks, features_list

    def detect_vessel(self, image_rgb, params, cd31_channel=None):
        """
        Detect vessel cross-sections (ring structures) using Canny edge detection.

        Vessels appear as ring-like structures in SMA staining - an outer contour
        (adventitial side) with an inner contour (lumen). Uses Canny edge detection
        to find ring edges, then contour hierarchy to pair outer/inner boundaries.

        Args:
            image_rgb: RGB image array (SMA channel as grayscale repeated 3x, or actual RGB)
            params: Dict with:
                - min_vessel_diameter_um: minimum outer diameter
                - max_vessel_diameter_um: maximum outer diameter
                - min_wall_thickness_um: minimum wall thickness
                - max_aspect_ratio: maximum major/minor axis ratio (exclude longitudinal)
                - min_circularity: minimum circularity (0-1)
                - min_ring_completeness: minimum fraction of ring that must be SMA+
                - pixel_size_um: for converting pixels to microns
                - classify_vessel_types: whether to auto-classify by size
                - canny_low: Canny low threshold (default: auto)
                - canny_high: Canny high threshold (default: auto)
            cd31_channel: Optional CD31 channel for vessel validation

        Returns:
            Tuple of (masks, features_list)
        """
        import cv2
        from scipy import ndimage
        from scipy.ndimage import distance_transform_edt

        pixel_size = params.get('pixel_size_um', 0.22)

        # Convert size parameters from µm to pixels
        min_diameter_px = params.get('min_vessel_diameter_um', 10) / pixel_size
        max_diameter_px = params.get('max_vessel_diameter_um', 1000) / pixel_size
        min_wall_px = params.get('min_wall_thickness_um', 2) / pixel_size

        # Convert to grayscale
        if image_rgb.ndim == 3:
            gray = np.mean(image_rgb[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = image_rgb.astype(np.float32)

        # Normalize to 0-255 for OpenCV
        gray_norm = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)

        # Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray_norm, (5, 5), 1.5)

        # Auto-calculate Canny thresholds using Otsu's method
        if params.get('canny_low') is None or params.get('canny_high') is None:
            otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = int(otsu_thresh * 0.5)
            canny_high = int(otsu_thresh * 1.0)
        else:
            canny_low = params.get('canny_low')
            canny_high = params.get('canny_high')

        # Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Dilate edges slightly to close small gaps, then find contours
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

        # Fill the detected edges to create binary regions
        # Use flood fill from edges to create filled regions
        binary = np.zeros_like(edges_dilated)

        # Find contours from edges
        edge_contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill closed contours
        for cnt in edge_contours:
            if cv2.contourArea(cnt) > 50:  # Skip tiny noise
                cv2.drawContours(binary, [cnt], 0, 255, -1)

        # Morphological cleanup
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Find contours with hierarchy for ring detection
        # RETR_CCOMP gives 2-level hierarchy: outer contours and their direct holes
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None or len(contours) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint32), []

        hierarchy = hierarchy[0]  # Shape: (N, 4) where 4 = [next, prev, child, parent]

        # Find ring candidates: outer contours (parent=-1) that have children (holes)
        ring_candidates = []
        for i, (next_c, prev_c, child, parent) in enumerate(hierarchy):
            if parent == -1 and child != -1:  # Outer contour with at least one hole
                outer_contour = contours[i]

                # Collect all child contours (holes)
                inner_contours = []
                child_idx = child
                while child_idx != -1:
                    inner_contours.append(contours[child_idx])
                    child_idx = hierarchy[child_idx][0]  # Next sibling

                # Take the largest inner contour as the main lumen
                if inner_contours:
                    inner_contour = max(inner_contours, key=cv2.contourArea)
                    ring_candidates.append({
                        'outer': outer_contour,
                        'inner': inner_contour,
                        'all_inner': inner_contours
                    })

        # Process ring candidates
        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        det_id = 1

        # Set image for SAM2 embeddings if available
        if self.sam2_predictor is not None:
            self.sam2_predictor.set_image(image_rgb if image_rgb.dtype == np.uint8 else (image_rgb / 256).astype(np.uint8))

        for cand in ring_candidates:
            outer = cand['outer']
            inner = cand['inner']

            # Need at least 5 points for ellipse fitting
            if len(outer) < 5 or len(inner) < 5:
                continue

            # Fit ellipses
            try:
                outer_ellipse = cv2.fitEllipse(outer)
                inner_ellipse = cv2.fitEllipse(inner)
            except cv2.error:
                continue

            # Extract ellipse parameters
            # fitEllipse returns: ((cx, cy), (minor_axis, major_axis), angle)
            (cx_out, cy_out), (minor_out, major_out), angle_out = outer_ellipse
            (cx_in, cy_in), (minor_in, major_in), angle_in = inner_ellipse

            # Calculate areas
            outer_area = cv2.contourArea(outer)
            inner_area = cv2.contourArea(inner)
            wall_area = outer_area - inner_area

            if wall_area <= 0 or inner_area <= 0:
                continue

            # Convert to diameters in µm
            outer_diameter_um = max(major_out, minor_out) * pixel_size
            inner_diameter_um = max(major_in, minor_in) * pixel_size

            # Size filtering
            if outer_diameter_um < params.get('min_vessel_diameter_um', 10):
                continue
            if outer_diameter_um > params.get('max_vessel_diameter_um', 1000):
                continue

            # Aspect ratio filtering (exclude longitudinal sections)
            aspect_ratio_out = max(major_out, minor_out) / (min(major_out, minor_out) + 1e-8)
            if aspect_ratio_out > params.get('max_aspect_ratio', 4.0):
                continue

            # Circularity filtering
            perimeter_out = cv2.arcLength(outer, True)
            circularity = 4 * np.pi * outer_area / (perimeter_out ** 2 + 1e-8)
            if circularity < params.get('min_circularity', 0.3):
                continue

            # Calculate wall thickness using distance transform (more accurate for irregular shapes)
            # Create wall mask
            wall_mask_temp = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(wall_mask_temp, [outer], 0, 255, -1)
            cv2.drawContours(wall_mask_temp, [inner], 0, 0, -1)
            wall_region = wall_mask_temp > 0

            if wall_region.sum() == 0:
                continue

            # Distance transform from inner boundary (lumen edge)
            lumen_mask_temp = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(lumen_mask_temp, [inner], 0, 255, -1)

            # Distance from lumen boundary into wall
            dist_from_lumen = distance_transform_edt(~(lumen_mask_temp > 0))

            # Distance from outer boundary into wall
            outer_mask_temp = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(outer_mask_temp, [outer], 0, 255, -1)
            dist_from_outer = distance_transform_edt(outer_mask_temp > 0)

            # Wall thickness at each wall pixel = distance to lumen + distance to outer
            # But we want local thickness, so sample along the medial axis
            wall_thickness_values = []

            # Sample thickness at points along inner contour
            for pt in inner[::max(1, len(inner)//36)]:  # Sample ~36 points
                px, py = pt[0]
                if 0 <= py < wall_region.shape[0] and 0 <= px < wall_region.shape[1]:
                    # Find thickness by ray casting outward
                    # Use the distance transform value at the medial axis
                    if wall_region[py, px] or (lumen_mask_temp[py, px] > 0):
                        # Cast ray outward to find wall thickness
                        ray_dist = dist_from_lumen[py, px]
                        if ray_dist > 0:
                            wall_thickness_values.append(ray_dist * pixel_size)

            # Also measure using the skeleton/medial axis approach
            from skimage.morphology import skeletonize
            skeleton = skeletonize(wall_region)
            skeleton_distances = dist_from_lumen[skeleton]
            if len(skeleton_distances) > 0:
                # Thickness is roughly 2x the distance to medial axis
                medial_thicknesses = skeleton_distances * 2 * pixel_size
                wall_thickness_values.extend(medial_thicknesses.tolist())

            if len(wall_thickness_values) < 5:
                continue

            wall_thicknesses = np.array(wall_thickness_values)
            wall_thickness_mean = float(np.mean(wall_thicknesses))
            wall_thickness_std = float(np.std(wall_thicknesses))
            wall_thickness_min = float(np.min(wall_thicknesses))
            wall_thickness_max = float(np.max(wall_thicknesses))
            wall_thickness_median = float(np.median(wall_thicknesses))

            # Wall thickness filtering
            if wall_thickness_mean < params.get('min_wall_thickness_um', 2):
                continue

            # Calculate ring completeness (fraction of perimeter with SMA signal)
            # Sample points along the expected ring and check if they're in the mask
            ring_points = 0
            ring_positive = 0
            for theta in np.linspace(0, 2 * np.pi, 72):
                # Mid-wall radius
                a_out, b_out = major_out / 2, minor_out / 2
                a_in, b_in = major_in / 2, minor_in / 2
                angle_out_rad = np.radians(angle_out)

                cos_t = np.cos(theta - angle_out_rad)
                sin_t = np.sin(theta - angle_out_rad)
                r_out = (a_out * b_out) / np.sqrt((b_out * cos_t) ** 2 + (a_out * sin_t) ** 2 + 1e-8)
                r_in = (a_in * b_in) / np.sqrt((b_in * cos_t) ** 2 + (a_in * sin_t) ** 2 + 1e-8)
                r_mid = (r_out + r_in) / 2

                # Point at mid-wall
                px = int(cx_out + r_mid * np.cos(theta))
                py = int(cy_out + r_mid * np.sin(theta))

                if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                    ring_points += 1
                    if binary[py, px] > 0:
                        ring_positive += 1

            ring_completeness = ring_positive / (ring_points + 1e-8)
            if ring_completeness < params.get('min_ring_completeness', 0.5):
                continue

            # CD31 validation (if channel provided)
            cd31_validated = True
            cd31_score = 0.0
            if cd31_channel is not None:
                # Create masks for lumen and wall
                lumen_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                wall_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

                cv2.drawContours(lumen_mask, [inner], 0, 255, -1)
                cv2.drawContours(wall_mask, [outer], 0, 255, -1)
                cv2.drawContours(wall_mask, [inner], 0, 0, -1)

                cd31_in_lumen = cd31_channel[lumen_mask > 0].mean() if (lumen_mask > 0).any() else 0
                cd31_in_wall = cd31_channel[wall_mask > 0].mean() if (wall_mask > 0).any() else 0

                # CD31 should be at lumen boundary, not in wall
                cd31_score = cd31_in_lumen / (cd31_in_wall + 1e-8)
                cd31_validated = cd31_in_lumen > cd31_in_wall * 0.8  # Some tolerance

            # Create mask for this vessel (wall region only)
            vessel_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(vessel_mask, [outer], 0, 255, -1)
            cv2.drawContours(vessel_mask, [inner], 0, 0, -1)
            vessel_mask_bool = vessel_mask > 0

            # Check overlap with existing detections
            if masks.max() > 0:
                overlap = (vessel_mask_bool & (masks > 0)).sum()
                if overlap > 0.5 * vessel_mask_bool.sum():
                    continue

            masks[vessel_mask_bool] = det_id

            # Extract full features
            cy, cx = cy_out, cx_out
            features = self.extract_full_features(vessel_mask_bool, image_rgb, cy, cx)

            # Add vessel-specific features
            features['outer_diameter_um'] = float(outer_diameter_um)
            features['inner_diameter_um'] = float(inner_diameter_um)
            features['major_axis_um'] = float(max(major_out, minor_out) * pixel_size)
            features['minor_axis_um'] = float(min(major_out, minor_out) * pixel_size)
            features['wall_thickness_mean_um'] = float(wall_thickness_mean)
            features['wall_thickness_median_um'] = float(wall_thickness_median)
            features['wall_thickness_std_um'] = float(wall_thickness_std)
            features['wall_thickness_min_um'] = float(wall_thickness_min)
            features['wall_thickness_max_um'] = float(wall_thickness_max)
            features['lumen_area_um2'] = float(inner_area * pixel_size ** 2)
            features['wall_area_um2'] = float(wall_area * pixel_size ** 2)
            features['orientation_deg'] = float(angle_out)
            features['aspect_ratio'] = float(aspect_ratio_out)
            features['circularity'] = float(circularity)
            features['ring_completeness'] = float(ring_completeness)
            features['cd31_validated'] = cd31_validated
            features['cd31_score'] = float(cd31_score)

            # Auto-classify vessel type by size (if enabled)
            vessel_type = 'unknown'
            if params.get('classify_vessel_types', False):
                if outer_diameter_um < 10:
                    vessel_type = 'capillary'
                elif outer_diameter_um < 100:
                    vessel_type = 'arteriole'
                else:
                    vessel_type = 'artery'
            features['vessel_type'] = vessel_type

            # Determine confidence level
            if ring_completeness > 0.8 and circularity > 0.6 and aspect_ratio_out < 2.0:
                confidence = 'high'
            elif ring_completeness > 0.6 and circularity > 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
            features['confidence'] = confidence

            features_list.append({
                'id': f'vessel_{det_id}',
                'center': [float(cx), float(cy)],
                'outer_contour': outer.tolist(),
                'inner_contour': inner.tolist(),
                'features': features
            })

            det_id += 1

        # Clear SAM2 cache
        if self.sam2_predictor is not None:
            self.sam2_predictor.reset_predictor()

        return masks, features_list

    def detect_mesothelium(self, image_rgb, params):
        """
        Detect mesothelial ribbon structures and divide into ~1500 µm² chunks.

        Uses ridge detection (Meijering filter) to find thin ribbon structures,
        then extracts skeleton, walks along paths, and chunks by area.

        Args:
            image_rgb: RGB image array (mesothelin channel as grayscale repeated 3x)
            params: Dict with:
                - target_chunk_area_um2: Target area for each chunk (default 1500)
                - min_ribbon_width_um: Expected minimum ribbon width
                - max_ribbon_width_um: Expected maximum ribbon width
                - min_fragment_area_um2: Skip fragments smaller than this
                - pixel_size_um: For converting pixels to microns

        Returns:
            Tuple of (masks, features_list) where features_list contains chunk polygons
        """
        import cv2
        from scipy import ndimage
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize, medial_axis, remove_small_objects
        from skimage.morphology import binary_closing, binary_opening, disk
        from skimage.filters import meijering, threshold_local
        from skimage.measure import label, regionprops

        pixel_size = params.get('pixel_size_um', 0.22)
        target_area_um2 = params.get('target_chunk_area_um2', 1500)
        min_fragment_um2 = params.get('min_fragment_area_um2', 1500)  # Skip small fragments

        # Convert width parameters from µm to pixels
        min_width_px = params.get('min_ribbon_width_um', 5) / pixel_size
        max_width_px = params.get('max_ribbon_width_um', 30) / pixel_size

        # Convert to grayscale
        if image_rgb.ndim == 3:
            gray = np.mean(image_rgb[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = image_rgb.astype(np.float32)

        # Normalize to 0-1 for ridge detection
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Ridge detection using Meijering filter (optimized for neurite/line structures)
        sigmas = np.linspace(min_width_px * 0.5, max_width_px * 0.5, 5)
        ridges = meijering(gray_norm, sigmas=sigmas, black_ridges=False)

        # Threshold ridge response
        ridge_thresh = threshold_local(ridges, block_size=51, offset=-0.01)
        binary = ridges > ridge_thresh

        # Morphological cleanup
        binary = binary_opening(binary, disk(1))
        binary = binary_closing(binary, disk(2))
        binary = remove_small_objects(binary, min_size=int(min_fragment_um2 / (pixel_size ** 2) * 0.1))

        # Label connected components and filter by total area
        labeled = label(binary)
        props = regionprops(labeled)

        # Keep only fragments large enough to chunk
        valid_labels = []
        for prop in props:
            area_um2 = prop.area * (pixel_size ** 2)
            if area_um2 >= min_fragment_um2:
                valid_labels.append(prop.label)

        if len(valid_labels) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint32), []

        # Create cleaned binary with only valid fragments
        binary_clean = np.isin(labeled, valid_labels)

        # Extract medial axis with distance transform
        skeleton, distance = medial_axis(binary_clean, return_distance=True)
        local_width = distance * 2  # Full width at skeleton points

        # Parse skeleton into paths using skan if available, else simple approach
        try:
            from skan import Skeleton as SkanSkeleton, summarize
            skel_obj = SkanSkeleton(skeleton)
            paths = skel_obj.paths_list()
        except ImportError:
            # Fallback: trace paths manually
            paths = self._trace_skeleton_paths(skeleton)

        # Chunk each path by area
        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        chunk_id = 1

        for path_idx, path_coords in enumerate(paths):
            if len(path_coords) < 3:
                continue

            # Get local width at each path point
            widths_px = []
            for pt in path_coords:
                r, c = int(pt[0]), int(pt[1])
                if 0 <= r < local_width.shape[0] and 0 <= c < local_width.shape[1]:
                    widths_px.append(max(local_width[r, c], 1))
                else:
                    widths_px.append(min_width_px)
            widths_um = np.array(widths_px) * pixel_size

            # Walk along path, accumulating area until target reached
            chunks = []
            accumulated_area = 0
            chunk_start_idx = 0

            for i in range(1, len(path_coords)):
                # Segment length
                dx = (path_coords[i][1] - path_coords[i-1][1]) * pixel_size
                dy = (path_coords[i][0] - path_coords[i-1][0]) * pixel_size
                seg_length = np.sqrt(dx**2 + dy**2)

                # Average width
                avg_width = (widths_um[i] + widths_um[i-1]) / 2

                # Segment area
                accumulated_area += seg_length * avg_width

                # Check if we've reached target
                if accumulated_area >= target_area_um2:
                    chunks.append({
                        'start_idx': chunk_start_idx,
                        'end_idx': i,
                        'path_points': path_coords[chunk_start_idx:i+1],
                        'widths_px': widths_px[chunk_start_idx:i+1],
                        'area_um2': accumulated_area
                    })
                    chunk_start_idx = i
                    accumulated_area = 0

            # Handle remainder - merge with previous if too small
            if chunk_start_idx < len(path_coords) - 1:
                remainder_area = accumulated_area
                if remainder_area < target_area_um2 * 0.5 and len(chunks) > 0:
                    # Merge with previous chunk
                    prev = chunks[-1]
                    prev['end_idx'] = len(path_coords) - 1
                    prev['path_points'] = np.vstack([prev['path_points'], path_coords[chunk_start_idx+1:]])
                    prev['widths_px'] = list(prev['widths_px']) + widths_px[chunk_start_idx+1:]
                    prev['area_um2'] += remainder_area
                elif remainder_area >= min_fragment_um2 * 0.5:
                    # Keep as separate chunk if not too small
                    chunks.append({
                        'start_idx': chunk_start_idx,
                        'end_idx': len(path_coords) - 1,
                        'path_points': path_coords[chunk_start_idx:],
                        'widths_px': widths_px[chunk_start_idx:],
                        'area_um2': remainder_area
                    })

            # Convert chunks to polygons
            for chunk in chunks:
                polygon = self._skeleton_chunk_to_polygon(
                    chunk['path_points'],
                    chunk['widths_px'],
                    pixel_size
                )

                if polygon is None or len(polygon) < 4:
                    continue

                # Create mask for this chunk
                chunk_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                cv2.fillPoly(chunk_mask, [polygon.astype(np.int32)], 255)
                chunk_mask_bool = chunk_mask > 0

                if chunk_mask_bool.sum() == 0:
                    continue

                masks[chunk_mask_bool] = chunk_id

                # Calculate centroid
                cy, cx = ndimage.center_of_mass(chunk_mask_bool)

                # Create feature dict
                features = {
                    'area_um2': float(chunk['area_um2']),
                    'path_length_um': float(len(chunk['path_points']) * pixel_size),
                    'mean_width_um': float(np.mean(chunk['widths_px']) * pixel_size),
                    'n_vertices': len(polygon),
                    'branch_id': path_idx,
                }

                features_list.append({
                    'id': f'meso_{chunk_id}',
                    'center': [float(cx), float(cy)],
                    'polygon_image': polygon.tolist(),  # In image coordinates (pixels)
                    'features': features
                })

                chunk_id += 1

        return masks, features_list

    def _trace_skeleton_paths(self, skeleton):
        """
        Simple skeleton path tracing (fallback if skan not available).
        Returns list of paths, each path is Nx2 array of (row, col) coordinates.
        """
        from scipy import ndimage
        from collections import deque

        # Find endpoints and branch points
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
        neighbor_count = neighbor_count * skeleton

        endpoints = (neighbor_count == 1) & skeleton
        branchpoints = (neighbor_count >= 3) & skeleton

        # Label skeleton segments
        labeled, n_segments = ndimage.label(skeleton)

        paths = []
        visited = np.zeros_like(skeleton, dtype=bool)

        # Start from each endpoint
        endpoint_coords = np.argwhere(endpoints)

        for start in endpoint_coords:
            if visited[start[0], start[1]]:
                continue

            # Trace path from this endpoint
            path = [start]
            visited[start[0], start[1]] = True
            current = start

            while True:
                # Find unvisited neighbors
                r, c = current
                found_next = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < skeleton.shape[0] and
                            0 <= nc < skeleton.shape[1] and
                            skeleton[nr, nc] and
                            not visited[nr, nc]):
                            path.append(np.array([nr, nc]))
                            visited[nr, nc] = True
                            current = np.array([nr, nc])
                            found_next = True
                            break
                    if found_next:
                        break

                if not found_next:
                    break

            if len(path) >= 3:
                paths.append(np.array(path))

        return paths

    def _skeleton_chunk_to_polygon(self, path_points, widths_px, pixel_size):
        """
        Convert skeleton path with widths to closed polygon.
        """
        path_points = np.array(path_points)
        widths_px = np.array(widths_px)

        if len(path_points) < 2:
            return None

        left_boundary = []
        right_boundary = []

        for i in range(len(path_points)):
            half_width = widths_px[i] / 2

            # Get tangent direction
            if i == 0:
                tangent = path_points[1] - path_points[0]
            elif i == len(path_points) - 1:
                tangent = path_points[-1] - path_points[-2]
            else:
                tangent = path_points[i+1] - path_points[i-1]

            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                continue
            tangent = tangent / norm

            # Perpendicular
            perp = np.array([-tangent[1], tangent[0]])

            # Boundary points (row, col format)
            left_pt = path_points[i] + perp * half_width
            right_pt = path_points[i] - perp * half_width

            left_boundary.append(left_pt)
            right_boundary.append(right_pt)

        if len(left_boundary) < 2:
            return None

        # Create closed polygon: left forward, then right backward
        # Convert to (col, row) = (x, y) for cv2
        polygon = np.vstack([
            np.array(left_boundary)[:, ::-1],  # (row,col) to (col,row)
            np.array(right_boundary)[::-1, ::-1]
        ])

        return polygon

    def process_tile(self, image_rgb, cell_type, params, cd31_channel=None):
        """
        Process a tile for the specified cell type.

        Args:
            image_rgb: RGB image array
            cell_type: 'nmj', 'mk', 'cell', 'vessel', or 'mesothelium'
            params: Cell-type specific parameters
            cd31_channel: Optional CD31 channel for vessel validation

        Returns:
            Tuple of (masks, features_list)
        """
        if cell_type == 'nmj':
            return self.detect_nmj(image_rgb, params)
        elif cell_type == 'mk':
            return self.detect_mk(image_rgb, params)
        elif cell_type == 'cell':
            return self.detect_cell(image_rgb, params)
        elif cell_type == 'vessel':
            return self.detect_vessel(image_rgb, params, cd31_channel=cd31_channel)
        elif cell_type == 'mesothelium':
            return self.detect_mesothelium(image_rgb, params)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")


# =============================================================================
# CZI LOADING
# =============================================================================

def get_czi_metadata(czi_path):
    """
    Extract metadata from CZI file without loading image data.

    Returns dict with:
        - channels: list of channel info (name, wavelength, fluor)
        - pixel_size_um: pixel size in microns
        - mosaic_size: (width, height) in pixels
        - n_channels: number of channels
    """
    import xml.etree.ElementTree as ET

    czi_path = str(czi_path)
    metadata = {
        'channels': [],
        'pixel_size_um': 0.22,  # default
        'mosaic_size': None,
        'n_channels': 0,
    }

    # Try pylibCZIrw first (better for large files)
    try:
        from pylibCZIrw import czi as pylibczi

        with pylibczi.open_czi(czi_path) as czidoc:
            # Get dimensions
            dims = czidoc.total_bounding_box
            metadata['mosaic_size'] = (dims['X'][1] - dims['X'][0], dims['Y'][1] - dims['Y'][0])

            # Get metadata XML
            meta_xml = czidoc.raw_metadata
            root = ET.fromstring(meta_xml)

    except ImportError:
        # Fall back to aicspylibczi
        from aicspylibczi import CziFile
        reader = CziFile(czi_path)

        bbox = reader.get_mosaic_bounding_box()
        metadata['mosaic_size'] = (bbox.w, bbox.h)

        meta_xml = reader.meta
        if isinstance(meta_xml, str):
            root = ET.fromstring(meta_xml)
        else:
            root = meta_xml

    # Parse XML for channel info
    channels = []
    for i, channel in enumerate(root.iter('Channel')):
        ch_info = {
            'index': i,
            'name': channel.get('Name', f'Channel_{i}'),
            'id': channel.get('Id', ''),
        }

        # Get fluorophore
        fluor = channel.find('.//Fluor')
        ch_info['fluorophore'] = fluor.text if fluor is not None else 'N/A'

        # Get emission wavelength
        emission = channel.find('.//EmissionWavelength')
        ch_info['emission_nm'] = float(emission.text) if emission is not None and emission.text else None

        # Get excitation wavelength
        excitation = channel.find('.//ExcitationWavelength')
        ch_info['excitation_nm'] = float(excitation.text) if excitation is not None and excitation.text else None

        # Get dye name/description
        dye_name = channel.find('.//DyeName')
        ch_info['dye'] = dye_name.text if dye_name is not None else ch_info['fluorophore']

        channels.append(ch_info)

    metadata['channels'] = channels
    metadata['n_channels'] = len(channels)

    # Parse pixel size
    for scaling in root.iter('Scaling'):
        for items in scaling.iter('Items'):
            for distance in items.iter('Distance'):
                if distance.get('Id') == 'X':
                    value = distance.find('Value')
                    if value is not None and value.text:
                        metadata['pixel_size_um'] = float(value.text) * 1e6
                        break

    return metadata


def print_czi_metadata(czi_path):
    """Print CZI metadata in human-readable format."""
    logger.info(f"CZI Metadata: {Path(czi_path).name}")
    logger.info("=" * 60)

    try:
        meta = get_czi_metadata(czi_path)

        if meta['mosaic_size']:
            w, h = meta['mosaic_size']
            logger.info(f"Mosaic size: {w:,} x {h:,} px")

        logger.info(f"Pixel size: {meta['pixel_size_um']:.4f} µm/px")
        logger.info(f"Number of channels: {meta['n_channels']}")

        logger.info("Channels:")
        logger.info("-" * 60)
        for ch in meta['channels']:
            ex = f"{ch['excitation_nm']:.0f}" if ch['excitation_nm'] else "N/A"
            em = f"{ch['emission_nm']:.0f}" if ch['emission_nm'] else "N/A"
            logger.info(f"  [{ch['index']}] {ch['name']}")
            logger.info(f"      Fluorophore: {ch['fluorophore']}")
            logger.info(f"      Excitation: {ex} nm | Emission: {em} nm")

        logger.info("=" * 60)
        return meta

    except Exception as e:
        logger.error(f"ERROR reading metadata: {e}")
        logger.error("  File may be on slow network mount - try copying locally first")
        return None


def load_czi(czi_path):
    """
    Load CZI file and return reader + metadata.

    DEPRECATED: Use CZILoader from shared.czi_loader instead for RAM-first architecture.
    This function is kept for backwards compatibility with external scripts.
    """
    from aicspylibczi import CziFile

    logger.info(f"Loading CZI: {czi_path}")
    reader = CziFile(str(czi_path))

    bbox = reader.get_mosaic_bounding_box()
    mosaic_info = {
        'x': bbox.x,
        'y': bbox.y,
        'width': bbox.w,
        'height': bbox.h,
    }
    logger.info(f"  Mosaic: {mosaic_info['width']:,} x {mosaic_info['height']:,} px")

    # Get pixel size from pre-parsed metadata or parse now
    pixel_size_um = 0.22
    try:
        meta = get_czi_metadata(czi_path)
        pixel_size_um = meta['pixel_size_um']

        # Print channel info
        if meta['channels']:
            logger.info(f"  Channels: {len(meta['channels'])}")
            for ch in meta['channels']:
                em = f"{ch['emission_nm']:.0f}nm" if ch['emission_nm'] else ""
                logger.info(f"    [{ch['index']}] {ch['name']} {em}")
    except Exception as e:
        logger.warning(f"Could not parse metadata ({e}), using defaults")

    logger.info(f"  Pixel size: {pixel_size_um:.4f} µm/px")

    return reader, mosaic_info, pixel_size_um


def generate_tile_grid(mosaic_info, tile_size):
    """Generate tile coordinates covering the mosaic."""
    tiles = []
    x_start = mosaic_info['x']
    y_start = mosaic_info['y']
    width = mosaic_info['width']
    height = mosaic_info['height']

    for y in range(y_start, y_start + height, tile_size):
        for x in range(x_start, x_start + width, tile_size):
            tiles.append({'x': x, 'y': y})

    return tiles


def load_all_channels_to_ram(reader, mosaic_info, n_channels=None, strip_height=2000):
    """
    Load ALL channels into RAM as numpy array using strip-based loading.

    DEPRECATED: Use CZILoader from shared.czi_loader with load_to_ram=True instead.
    This function is kept for backwards compatibility with external scripts.

    For large network-mounted files, this reads in horizontal strips (rows)
    to show progress and be more robust than one huge read.

    Args:
        reader: CziFile reader
        mosaic_info: Dict with x, y, width, height
        n_channels: Number of channels (auto-detect if None)
        strip_height: Height of each strip to read (default 2000 rows)

    Returns:
        Dict mapping channel index to numpy array of shape (height, width)
    """
    import time

    x_start = mosaic_info['x']
    y_start = mosaic_info['y']
    width = mosaic_info['width']
    height = mosaic_info['height']

    # Auto-detect number of channels
    if n_channels is None:
        dims = reader.get_dims_shape()
        for dim in dims:
            if 'C' in dim:
                n_channels = dim['C'][1]
                break
        if n_channels is None:
            n_channels = 1

    logger.info(f"  Loading {n_channels} channels into RAM (strip-based)...")
    per_channel_gb = width * height * 2 / 1e9
    total_gb = per_channel_gb * n_channels
    logger.info(f"  Size: {width:,} x {height:,} px x {n_channels} channels = {total_gb:.1f} GB")

    n_strips = (height + strip_height - 1) // strip_height
    logger.info(f"  Reading in {n_strips} strips of {strip_height} rows each")

    start_time = time.time()

    channels = {}
    for c in range(n_channels):
        logger.info(f"    Channel {c}:")
        ch_start = time.time()

        # Pre-allocate full array for this channel
        arr = np.empty((height, width), dtype=np.uint16)

        # Read in strips
        for strip_idx in range(n_strips):
            y_off = strip_idx * strip_height
            h = min(strip_height, height - y_off)

            strip_start = time.time()
            strip = reader.read_mosaic(
                region=(x_start, y_start + y_off, width, h),
                scale_factor=1,
                C=c
            )
            strip = np.squeeze(strip)
            arr[y_off:y_off + h, :] = strip

            strip_elapsed = time.time() - strip_start
            strip_mb = width * h * 2 / 1e6
            pct = (strip_idx + 1) * 100 // n_strips
            logger.debug(f"      Strip {strip_idx + 1}/{n_strips} ({pct}%): {strip_mb:.0f} MB in {strip_elapsed:.1f}s ({strip_mb / strip_elapsed:.0f} MB/s)")

        channels[c] = arr

        ch_elapsed = time.time() - ch_start
        logger.info(f"    Channel {c} done: {per_channel_gb:.1f} GB in {ch_elapsed:.1f}s ({per_channel_gb * 1000 / ch_elapsed:.1f} MB/s)")

    elapsed = time.time() - start_time
    logger.info(f"  Total load time: {elapsed:.1f}s ({total_gb * 1000 / elapsed:.1f} MB/s)")

    return channels


# =============================================================================
# SAMPLE CREATION FOR HTML
# =============================================================================

# =============================================================================
# LMD EXPORT (for mesothelium)
# =============================================================================

def export_to_leica_lmd(detections, output_path, pixel_size_um, image_height_px,
                        image_width_px=None, calibration_points=None,
                        add_fiducials=True, fiducial_positions=None,
                        flip_y=True):
    """
    Export mesothelium chunks to Leica LMD XML format using py-lmd library.

    Args:
        detections: List of detection dicts with 'polygon_image' (pixel coords)
        output_path: Path to save XML file
        pixel_size_um: Microns per pixel
        image_height_px: Image height in pixels (for Y flip)
        image_width_px: Image width in pixels (for calibration)
        calibration_points: Optional 3x2 array of calibration points in µm
        add_fiducials: Whether to add calibration cross markers
        fiducial_positions: List of (x, y) positions in µm for fiducial crosses
                           If None and add_fiducials=True, uses image corners
        flip_y: Whether to flip Y axis for stage coordinates

    Returns:
        Path to saved file, also saves metadata CSV with both coordinate systems
    """
    try:
        from lmd.lib import Collection, Shape
        from lmd.tools import makeCross
        has_pylmd = True
    except ImportError:
        logger.warning("py-lmd not installed. Install with: pip install py-lmd")
        logger.warning("  Falling back to simple XML export...")
        has_pylmd = False

    if image_width_px is None:
        image_width_px = 10000 / pixel_size_um  # Default estimate

    img_width_um = image_width_px * pixel_size_um
    img_height_um = image_height_px * pixel_size_um

    # Default calibration points (corners of image in µm)
    if calibration_points is None:
        calibration_points = np.array([
            [0, 0],
            [0, img_height_um],
            [img_width_um, img_height_um]
        ])

    # Default fiducial positions (corners + center)
    if fiducial_positions is None and add_fiducials:
        margin = 500  # µm margin from edges
        fiducial_positions = [
            (margin, margin),  # Top-left
            (img_width_um - margin, margin),  # Top-right
            (margin, img_height_um - margin),  # Bottom-left
            (img_width_um - margin, img_height_um - margin),  # Bottom-right
        ]

    if not has_pylmd:
        return _export_lmd_simple(detections, output_path, pixel_size_um,
                                  image_height_px, flip_y, fiducial_positions)

    # Create collection
    collection = Collection(calibration_points=calibration_points)

    # Add fiducial crosses
    if add_fiducials and fiducial_positions:
        for i, (fx, fy) in enumerate(fiducial_positions):
            # makeCross creates a cross shape at specified location
            cross = makeCross(
                center=np.array([fx, fy]),
                arm_length=100,  # µm
                arm_width=10,    # µm
            )
            collection.add_shape(Shape(
                points=cross,
                well="CAL",  # Special well for calibration
                name=f"Fiducial_{i+1}"
            ))

    # Add each chunk as a shape
    for i, det in enumerate(detections):
        if 'polygon_image' not in det:
            continue

        polygon_px = np.array(det['polygon_image'])

        # Convert to µm coordinates
        polygon_um = polygon_px * pixel_size_um

        # Flip Y if needed (image Y increases down, stage Y may increase up)
        if flip_y:
            polygon_um[:, 1] = img_height_um - polygon_um[:, 1]

        # Close polygon if not already closed
        if not np.allclose(polygon_um[0], polygon_um[-1]):
            polygon_um = np.vstack([polygon_um, polygon_um[0]])

        # Add to collection
        chunk_name = det.get('uid', det.get('id', f'Chunk_{i+1:04d}'))
        collection.new_shape(polygon_um, well="A1", name=chunk_name)

    # Save to XML
    collection.save(str(output_path))
    logger.info(f"  Exported {len(detections)} chunks to LMD XML: {output_path}")
    if add_fiducials:
        logger.info(f"  Added {len(fiducial_positions)} fiducial crosses for calibration")

    # Also save metadata CSV with both coordinate systems
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write('chunk_name,centroid_x_px,centroid_y_px,centroid_x_um,centroid_y_um,area_um2,n_vertices\n')
        for det in detections:
            if 'polygon_image' not in det:
                continue
            name = det.get('uid', det.get('id', ''))
            cx_px, cy_px = det['center']
            cx_um = cx_px * pixel_size_um
            cy_um = cy_px * pixel_size_um
            if flip_y:
                cy_um = img_height_um - cy_um
            area = det['features'].get('area_um2', 0)
            n_verts = det['features'].get('n_vertices', len(det.get('polygon_image', [])))
            f.write(f'{name},{cx_px:.1f},{cy_px:.1f},{cx_um:.2f},{cy_um:.2f},{area:.2f},{n_verts}\n')
    logger.info(f"  Saved coordinates CSV: {csv_path}")

    return output_path


def _export_lmd_simple(detections, output_path, pixel_size_um, image_height_px,
                       flip_y=True, fiducial_positions=None):
    """
    Simple XML export fallback when py-lmd is not installed.
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    root = ET.Element("ImageData")

    # Global coordinates
    global_coords = ET.SubElement(root, "GlobalCoordinates")
    ET.SubElement(global_coords, "OffsetX").text = "0"
    ET.SubElement(global_coords, "OffsetY").text = "0"

    # Shape list
    shape_list = ET.SubElement(root, "ShapeList")

    for i, det in enumerate(detections):
        if 'polygon_image' not in det:
            continue

        polygon_px = np.array(det['polygon_image'])

        # Convert to µm
        polygon_um = polygon_px * pixel_size_um

        # Flip Y if needed
        if flip_y:
            polygon_um[:, 1] = (image_height_px * pixel_size_um) - polygon_um[:, 1]

        shape = ET.SubElement(shape_list, "Shape")

        # Name
        name = det.get('uid', det.get('id', f'Chunk_{i+1:04d}'))
        ET.SubElement(shape, "Name").text = name
        ET.SubElement(shape, "ShapeType").text = "Polygon"

        # Points
        point_list = ET.SubElement(shape, "PointList")
        for x, y in polygon_um:
            point = ET.SubElement(point_list, "Point")
            ET.SubElement(point, "X").text = f"{x:.2f}"
            ET.SubElement(point, "Y").text = f"{y:.2f}"

    # Pretty print
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    with open(output_path, 'w') as f:
        f.write(xml_str)

    logger.info(f"  Exported {len(detections)} chunks to simple XML: {output_path}")
    return output_path


# =============================================================================
# SAMPLE CREATION FOR HTML
# =============================================================================

def create_sample_from_detection(tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name, crop_size=224):
    """Create an HTML sample from a detection.

    Uses fixed 224x224 crop to match ResNet feature extraction input,
    so annotators see exactly what the classifier will see.
    """
    det_id = feat['id']
    det_num = int(det_id.split('_')[-1])
    mask = masks == det_num

    if mask.sum() == 0:
        return None

    # Get centroid
    cy, cx = feat['center'][1], feat['center'][0]

    # Use fixed crop size (224) to match ResNet classifier input
    half = crop_size // 2

    # Calculate ideal crop bounds (may extend beyond tile)
    y1_ideal = int(cy) - half
    y2_ideal = int(cy) + half
    x1_ideal = int(cx) - half
    x2_ideal = int(cx) + half

    # Clamp to tile bounds
    y1 = max(0, y1_ideal)
    y2 = min(tile_rgb.shape[0], y2_ideal)
    x1 = max(0, x1_ideal)
    x2 = min(tile_rgb.shape[1], x2_ideal)

    # Validate crop bounds before extracting
    if y2 <= y1 or x2 <= x1:
        logger.warning(f"Invalid crop bounds: y1={y1}, y2={y2}, x1={x1}, x2={x2}, skipping detection {det_id}")
        return None

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Pad to center the mask if crop was clamped at edges
    # Use max(0, ...) to ensure non-negative padding values
    pad_top = max(0, y1 - y1_ideal)
    pad_bottom = max(0, y2_ideal - y2)
    pad_left = max(0, x1 - x1_ideal)
    pad_right = max(0, x2_ideal - x2)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        # Pad crop with zeros (black)
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=False)

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5)
    crop_with_contour = draw_mask_contour(crop_norm, crop_mask, color=(0, 255, 0), thickness=2)

    # Keep at 224x224 (same as classifier input) - already correct size from crop
    pil_img = Image.fromarray(crop_with_contour)

    img_b64, mime = image_to_base64(pil_img, format='PNG')

    # Create unique ID
    uid = f"{slide_name}_{tile_x}_{tile_y}_{det_id}"

    # Get stats from features
    features = feat['features']
    area_um2 = features.get('area', 0) * (pixel_size_um ** 2)

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    # Add cell-type specific stats
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'sam2_score' in features:
        stats['confidence'] = features['sam2_score']

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args):
    """Main pipeline execution."""
    # Setup logging
    setup_logging(level="DEBUG" if getattr(args, 'verbose', False) else "INFO")

    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    slide_name = czi_path.stem

    logger.info("=" * 60)
    logger.info("UNIFIED SEGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Slide: {slide_name}")
    logger.info(f"Cell type: {args.cell_type}")
    logger.info(f"Channel: {args.channel}")
    logger.info("=" * 60)

    # RAM-first architecture: Load CZI channel into RAM ONCE at pipeline start
    # This eliminates repeated network I/O for files on network mounts
    # Default is RAM loading for single slides (best performance on network mounts)
    use_ram = args.load_to_ram  # Default True for single slide processing

    logger.info("Loading CZI file with get_loader() (RAM-first architecture)...")
    loader = get_loader(
        czi_path,
        load_to_ram=use_ram,
        channel=args.channel,
        quiet=False
    )

    # Get mosaic bounds from loader properties
    x_start, y_start = loader.mosaic_origin
    width, height = loader.mosaic_size

    # Build mosaic_info dict for compatibility with existing functions
    mosaic_info = {
        'x': x_start,
        'y': y_start,
        'width': width,
        'height': height,
    }
    pixel_size_um = loader.get_pixel_size()

    logger.info(f"  Mosaic: {mosaic_info['width']} x {mosaic_info['height']} px")
    logger.info(f"  Origin: ({mosaic_info['x']}, {mosaic_info['y']})")
    logger.info(f"  Pixel size: {pixel_size_um:.4f} um/px")
    if use_ram:
        logger.info(f"  Channel {args.channel} loaded to RAM")

    # Load additional channels if --all-channels specified (for NMJ specificity checking)
    all_channel_data = {args.channel: loader.channel_data}  # Primary channel
    if getattr(args, 'all_channels', False) and use_ram:
        # Get number of channels from CZI metadata
        try:
            dims = loader.reader.get_dims_shape()[0]
            n_channels = dims.get('C', (0, 3))[1]  # Default to 3 channels
        except:
            n_channels = 3  # Fallback

        logger.info(f"Loading all {n_channels} channels for multi-channel analysis...")
        for ch in range(n_channels):
            if ch != args.channel:
                logger.info(f"  Loading channel {ch}...")
                loader = get_loader(czi_path, load_to_ram=True, channel=ch, quiet=True)
                all_channel_data[ch] = loader.get_channel_data(ch)
        logger.info(f"  Loaded channels: {sorted(all_channel_data.keys())}")

    # Also load CD31 channel to RAM if specified (for vessel validation)
    # Note: get_loader() with the same path returns the cached loader and just adds the new channel
    if args.cell_type == 'vessel' and args.cd31_channel is not None:
        logger.info(f"Loading CD31 channel {args.cd31_channel} to RAM...")
        # Use get_loader to add channel to the same cached loader instance
        loader = get_loader(
            czi_path,
            load_to_ram=use_ram,
            channel=args.cd31_channel,
            quiet=False
        )

    # Generate tile grid (using global coordinates)
    logger.info(f"Generating tile grid (size={args.tile_size})...")
    all_tiles = generate_tile_grid(mosaic_info, args.tile_size)
    logger.info(f"  Total tiles: {len(all_tiles)}")

    # Calibrate tissue threshold using RAM-loaded data
    logger.info("Calibrating tissue threshold...")
    variance_threshold = calibrate_tissue_threshold(
        all_tiles,
        reader=None,  # No reader needed - use image_array
        x_start=0,    # Not needed when using image_array
        y_start=0,
        calibration_samples=min(50, len(all_tiles)),
        channel=args.channel,
        tile_size=args.tile_size,
        image_array=loader.channel_data,  # Pass RAM array directly
    )

    # Filter to tissue-containing tiles using RAM-loaded data
    logger.info("Filtering to tissue-containing tiles...")
    tissue_tiles = filter_tissue_tiles(
        all_tiles,
        variance_threshold,
        reader=None,  # No reader needed - use image_array
        x_start=0,
        y_start=0,
        channel=args.channel,
        tile_size=args.tile_size,
        image_array=loader.channel_data,  # Pass RAM array directly
    )

    if len(tissue_tiles) == 0:
        logger.error("No tissue-containing tiles found!")
        return

    # Sample from tissue tiles
    n_sample = max(1, int(len(tissue_tiles) * args.sample_fraction))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    logger.info(f"Sampled {len(sampled_tiles)} tiles ({args.sample_fraction*100:.0f}% of {len(tissue_tiles)} tissue tiles)")

    # Setup output directories
    slide_output_dir = output_dir / slide_name
    tiles_dir = slide_output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    # Use CellDetector + strategy pattern for all cell types
    logger.info("Initializing detector...")

    # All cell types now use the new CellDetector pattern
    use_new_detector = args.cell_type in ('nmj', 'mk', 'cell', 'vessel', 'mesothelium')

    if use_new_detector:
        # New CellDetector with strategy pattern
        # Note: mesothelium strategy doesn't need SAM2 (uses ridge detection)
        detector = CellDetector(device="cuda")
        segmenter = None  # Not used for these cell types

        # Load NMJ classifier if provided
        if args.cell_type == 'nmj' and getattr(args, 'nmj_classifier', None):
            from segmentation.detection.strategies.nmj import load_nmj_classifier
            from torchvision import transforms

            logger.info(f"Loading NMJ classifier from {args.nmj_classifier}...")
            nmj_classifier = load_nmj_classifier(args.nmj_classifier, device=detector.device)
            classifier_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # Add classifier to models dict
            detector.models['classifier'] = nmj_classifier
            detector.models['transform'] = classifier_transform
            logger.info("NMJ classifier loaded successfully")
    else:
        # No cell types fall back to UnifiedSegmenter anymore
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    # Detection parameters
    if args.cell_type == 'nmj':
        params = {
            'intensity_percentile': args.intensity_percentile,
            'min_area': args.min_area,
            'min_skeleton_length': args.min_skeleton_length,
            'max_solidity': args.max_solidity,
        }
    elif args.cell_type == 'mk':
        params = {
            'mk_min_area': args.mk_min_area,
            'mk_max_area': args.mk_max_area,
        }
    elif args.cell_type == 'cell':
        params = {}
    elif args.cell_type == 'vessel':
        params = {
            'min_vessel_diameter_um': args.min_vessel_diameter,
            'max_vessel_diameter_um': args.max_vessel_diameter,
            'min_wall_thickness_um': args.min_wall_thickness,
            'max_aspect_ratio': args.max_aspect_ratio,
            'min_circularity': args.min_circularity,
            'min_ring_completeness': args.min_ring_completeness,
            'pixel_size_um': pixel_size_um,
            'classify_vessel_types': args.classify_vessel_types,
        }
    elif args.cell_type == 'mesothelium':
        params = {
            'target_chunk_area_um2': args.target_chunk_area,
            'min_ribbon_width_um': args.min_ribbon_width,
            'max_ribbon_width_um': args.max_ribbon_width,
            'min_fragment_area_um2': args.min_fragment_area,
            'pixel_size_um': pixel_size_um,
        }
    else:
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    logger.info(f"Detection params: {params}")

    # Create strategy for new detector pattern
    strategy = None
    if use_new_detector:
        strategy = create_strategy_for_cell_type(args.cell_type, params, pixel_size_um)
        logger.info(f"Using {strategy.name} strategy: {strategy.get_config()}")

    # Process tiles
    logger.info("Processing tiles...")
    all_samples = []
    all_detections = []  # Universal list with global coordinates
    total_detections = 0

    for tile in tqdm(sampled_tiles, desc="Tiles"):
        tile_x = tile['x']
        tile_y = tile['y']

        try:
            # Use loader.get_tile() for consistent tile extraction
            # This uses RAM-loaded data if available, or falls back to on-demand reading
            tile_data = loader.get_tile(tile_x, tile_y, args.tile_size, channel=args.channel)

            if tile_data is None or tile_data.size == 0:
                continue

            if tile_data.max() == 0:
                continue

            # Convert to RGB
            if tile_data.ndim == 2:
                tile_rgb = np.stack([tile_data] * 3, axis=-1)
            else:
                tile_rgb = tile_data

            # Get CD31 channel if specified (for vessel validation)
            # Uses the same loader - CD31 channel was already loaded to RAM via get_loader()
            cd31_channel_data = None
            if args.cell_type == 'vessel' and args.cd31_channel is not None:
                cd31_tile = loader.get_tile(tile_x, tile_y, args.tile_size, channel=args.cd31_channel)
                if cd31_tile is not None and cd31_tile.size > 0:
                    cd31_channel_data = cd31_tile.astype(np.float32)

            # Detect cells
            if use_new_detector:
                # New CellDetector + strategy pattern
                # The strategy's detect() method returns (label_array, list[Detection])
                if args.cell_type == 'vessel':
                    # Vessel strategy needs cd31_channel parameter
                    masks, detections = strategy.detect(
                        tile_rgb, detector.models, pixel_size_um,
                        cd31_channel=cd31_channel_data
                    )
                else:
                    # Other strategies: nmj, mk, cell, mesothelium
                    masks, detections = strategy.detect(
                        tile_rgb, detector.models, pixel_size_um
                    )
                # Convert Detection objects to the expected features_list format
                features_list = detections_to_features_list(detections, args.cell_type)
            else:
                # This branch is no longer used - all cell types use new detector
                masks, features_list = segmenter.process_tile(
                    tile_rgb, args.cell_type, params, cd31_channel=cd31_channel_data
                )

            if len(features_list) == 0:
                continue

            # Multi-channel intensity measurement for NMJ specificity checking
            if args.cell_type == 'nmj' and len(all_channel_data) > 1:
                for i, feat in enumerate(features_list):
                    # Get mask for this detection
                    if i < masks.max():
                        mask = (masks == (i + 1))
                        if mask.sum() > 0:
                            channel_intensities = {}
                            for ch, ch_data in all_channel_data.items():
                                # Extract tile from channel data
                                ch_tile = ch_data[tile_y:tile_y + args.tile_size,
                                                  tile_x:tile_x + args.tile_size]
                                if ch_tile.shape == mask.shape:
                                    mean_int = float(ch_tile[mask].mean())
                                    channel_intensities[f'ch{ch}_mean'] = mean_int
                            # Add to features
                            feat['features'].update(channel_intensities)
                            # Calculate specificity metrics
                            # BTX (ch1) / nuclear (ch0) - real NMJs have high BTX, low nuclear
                            # Autofluorescence appears in all channels including nuclear
                            btx_int = channel_intensities.get('ch1_mean', 0)
                            nuclear_int = channel_intensities.get('ch0_mean', 1)  # avoid div by zero
                            nfl_int = channel_intensities.get('ch2_mean', 0)

                            feat['features']['btx_nuclear_ratio'] = btx_int / max(nuclear_int, 1)
                            feat['features']['btx_nfl_ratio'] = btx_int / max(nfl_int, 1) if nfl_int > 0 else float('inf')
                            # Legacy: overall specificity (BTX vs max other)
                            primary_int = channel_intensities.get(f'ch{args.channel}_mean', 0)
                            other_ints = [v for k, v in channel_intensities.items()
                                         if k != f'ch{args.channel}_mean']
                            if other_ints and max(other_ints) > 0:
                                feat['features']['channel_specificity'] = primary_int / max(other_ints)
                            else:
                                feat['features']['channel_specificity'] = float('inf')

            # Add universal IDs and global coordinates to each detection
            for feat in features_list:
                local_cx, local_cy = feat['center']
                global_cx = tile_x + local_cx
                global_cy = tile_y + local_cy

                # Create universal ID: slide_celltype_globalX_globalY
                # Use round() instead of int() to reduce collision probability for nearby cells
                uid = f"{slide_name}_{args.cell_type}_{round(global_cx)}_{round(global_cy)}"
                feat['uid'] = uid
                feat['global_center'] = [float(global_cx), float(global_cy)]
                feat['global_center_um'] = [float(global_cx * pixel_size_um), float(global_cy * pixel_size_um)]
                feat['tile_origin'] = [tile_x, tile_y]
                feat['slide_name'] = slide_name

                # Convert contours to global coordinates if present (vessels)
                if 'outer_contour' in feat:
                    feat['outer_contour_global'] = [[pt[0][0] + tile_x, pt[0][1] + tile_y]
                                                    for pt in feat['outer_contour']]
                if 'inner_contour' in feat:
                    feat['inner_contour_global'] = [[pt[0][0] + tile_x, pt[0][1] + tile_y]
                                                    for pt in feat['inner_contour']]

                all_detections.append(feat)

            # Save masks and features
            tile_id = f"tile_{tile_x}_{tile_y}"
            tile_out = tiles_dir / tile_id
            tile_out.mkdir(exist_ok=True)

            with h5py.File(tile_out / f"{args.cell_type}_masks.h5", 'w') as f:
                create_hdf5_dataset(f, 'masks', masks.astype(np.uint16))

            with open(tile_out / f"{args.cell_type}_features.json", 'w') as f:
                json.dump(features_list, f)

            # Create samples for HTML (filter by minimum area 25 µm²)
            min_area_um2 = 25.0
            for feat in features_list:
                # Check area filter
                area_um2 = feat['features'].get('area', 0) * (pixel_size_um ** 2)
                if area_um2 < min_area_um2:
                    continue

                sample = create_sample_from_detection(
                    tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name
                )
                if sample:
                    all_samples.append(sample)
                    total_detections += 1

            del tile_data, tile_rgb, masks
            if cd31_channel_data is not None:
                del cd31_channel_data
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing tile ({tile_x}, {tile_y}): {e}")
            continue

    logger.info(f"Total detections: {total_detections}")

    # Sort samples by area (ascending - smallest first)
    all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0))

    # Export to HTML
    logger.info(f"Exporting to HTML ({len(all_samples)} samples)...")
    html_dir = slide_output_dir / "html"

    export_samples_to_html(
        all_samples,
        html_dir,
        args.cell_type,
        samples_per_page=args.samples_per_page,
        title=f"{args.cell_type.upper()} Annotation Review",
        page_prefix=f'{args.cell_type}_page',
        file_name=f"{slide_name}.czi",
        pixel_size_um=pixel_size_um,
        tiles_processed=len(sampled_tiles),
        tiles_total=len(all_tiles),
    )

    # Save all detections with universal IDs and global coordinates
    detections_file = slide_output_dir / f'{args.cell_type}_detections.json'
    with open(detections_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    logger.info(f"  Saved {len(all_detections)} detections to {detections_file}")

    # Export CSV with contour coordinates for easy import
    csv_file = slide_output_dir / f'{args.cell_type}_coordinates.csv'
    with open(csv_file, 'w') as f:
        # Header
        if args.cell_type == 'vessel':
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,outer_diameter_um,wall_thickness_um,confidence\n')
            for det in all_detections:
                feat = det.get('features', {})
                f.write(f"{det['uid']},{det['global_center'][0]:.1f},{det['global_center'][1]:.1f},"
                        f"{det['global_center_um'][0]:.2f},{det['global_center_um'][1]:.2f},"
                        f"{feat.get('outer_diameter_um', 0):.2f},{feat.get('wall_thickness_mean_um', 0):.2f},"
                        f"{feat.get('confidence', 'unknown')}\n")
        else:
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2\n')
            for det in all_detections:
                feat = det.get('features', {})
                area_um2 = feat.get('area', 0) * (pixel_size_um ** 2)
                f.write(f"{det['uid']},{det['global_center'][0]:.1f},{det['global_center'][1]:.1f},"
                        f"{det['global_center_um'][0]:.2f},{det['global_center_um'][1]:.2f},{area_um2:.2f}\n")
    logger.info(f"  Saved coordinates to {csv_file}")

    # Export to Leica LMD format for mesothelium
    if args.cell_type == 'mesothelium' and len(all_detections) > 0:
        logger.info("Exporting to Leica LMD XML format...")
        lmd_file = slide_output_dir / f'{args.cell_type}_chunks.xml'
        export_to_leica_lmd(
            all_detections,
            lmd_file,
            pixel_size_um,
            image_height_px=mosaic_info['height'],
            image_width_px=mosaic_info['width'],
            add_fiducials=args.add_fiducials,
            flip_y=True,
        )

    # Save summary
    summary = {
        'slide_name': slide_name,
        'cell_type': args.cell_type,
        'pixel_size_um': pixel_size_um,
        'mosaic_width': mosaic_info['width'],
        'mosaic_height': mosaic_info['height'],
        'total_tiles': len(all_tiles),
        'tissue_tiles': len(tissue_tiles),
        'sampled_tiles': len(sampled_tiles),
        'total_detections': total_detections,
        'params': params,
        'detections_file': str(detections_file),
        'coordinates_file': str(csv_file),
    }

    with open(slide_output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup detector resources
    if use_new_detector and detector is not None:
        detector.cleanup()

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output: {slide_output_dir}")
    logger.info(f"HTML viewer: {html_dir / 'index.html'}")

    # Start HTTP server and Cloudflare tunnel if requested
    no_serve = getattr(args, 'no_serve', False)
    serve_foreground = getattr(args, 'serve', False)
    serve_background = getattr(args, 'serve_background', True)
    port = getattr(args, 'port', 8081)

    if no_serve:
        logger.info("Server disabled (--no-serve)")
    elif serve_foreground:
        # Foreground mode: start and wait for Ctrl+C
        http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(html_dir, port, background=False)
        if http_proc is not None:
            wait_for_server_shutdown(http_proc, tunnel_proc)
    elif serve_background:
        # Background mode: start and exit script
        start_server_and_tunnel(html_dir, port, background=True)


def main():
    parser = argparse.ArgumentParser(
        description='Unified Cell Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument('--czi-path', type=str, required=True, help='Path to CZI file')
    parser.add_argument('--cell-type', type=str, default=None,
                        choices=['nmj', 'mk', 'cell', 'vessel', 'mesothelium'],
                        help='Cell type to detect (not required if --show-metadata)')

    # Metadata inspection
    parser.add_argument('--show-metadata', action='store_true',
                        help='Show CZI channel/dimension info and exit (no processing)')

    # Performance options - RAM loading is the default for single slides (best for network mounts)
    parser.add_argument('--load-to-ram', action='store_true', default=True,
                        help='Load entire channel into RAM first (default: True for best performance on network mounts)')
    parser.add_argument('--no-ram', dest='load_to_ram', action='store_false',
                        help='Disable RAM loading (use on-demand tile reading instead)')

    # Output
    parser.add_argument('--output-dir', type=str, default='/home/dude/nmj_output', help='Output directory')

    # Tile processing
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size in pixels')
    parser.add_argument('--sample-fraction', type=float, default=0.20, help='Fraction of tissue tiles (default: 20%)')
    parser.add_argument('--channel', type=int, default=1, help='Primary channel index for detection')
    parser.add_argument('--all-channels', action='store_true',
                        help='Load all channels for multi-channel analysis (NMJ specificity checking)')

    # NMJ parameters
    parser.add_argument('--intensity-percentile', type=float, default=99)
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--min-skeleton-length', type=int, default=30)
    parser.add_argument('--max-solidity', type=float, default=0.85,
                        help='Maximum solidity for NMJ detection (branched structures have low solidity)')
    parser.add_argument('--nmj-classifier', type=str, default=None,
                        help='Path to trained NMJ classifier (.pth file)')

    # MK parameters
    parser.add_argument('--mk-min-area', type=int, default=1000)
    parser.add_argument('--mk-max-area', type=int, default=100000)

    # Vessel parameters
    parser.add_argument('--min-vessel-diameter', type=float, default=10,
                        help='Minimum vessel outer diameter in µm')
    parser.add_argument('--max-vessel-diameter', type=float, default=1000,
                        help='Maximum vessel outer diameter in µm')
    parser.add_argument('--min-wall-thickness', type=float, default=2,
                        help='Minimum vessel wall thickness in µm')
    parser.add_argument('--max-aspect-ratio', type=float, default=4.0,
                        help='Maximum aspect ratio (exclude longitudinal sections)')
    parser.add_argument('--min-circularity', type=float, default=0.3,
                        help='Minimum circularity for vessel detection')
    parser.add_argument('--min-ring-completeness', type=float, default=0.5,
                        help='Minimum ring completeness (fraction of SMA+ perimeter)')
    parser.add_argument('--cd31-channel', type=int, default=None,
                        help='CD31 channel index for vessel validation (optional)')
    parser.add_argument('--classify-vessel-types', action='store_true',
                        help='Auto-classify vessels by size (capillary/arteriole/artery)')

    # Mesothelium parameters (for LMD chunking)
    parser.add_argument('--target-chunk-area', type=float, default=1500,
                        help='Target area for mesothelium chunks in µm²')
    parser.add_argument('--min-ribbon-width', type=float, default=5,
                        help='Minimum expected ribbon width in µm')
    parser.add_argument('--max-ribbon-width', type=float, default=30,
                        help='Maximum expected ribbon width in µm')
    parser.add_argument('--min-fragment-area', type=float, default=1500,
                        help='Skip mesothelium fragments smaller than this (µm²)')
    parser.add_argument('--add-fiducials', action='store_true', default=True,
                        help='Add calibration cross markers to LMD export')
    parser.add_argument('--no-fiducials', dest='add_fiducials', action='store_false',
                        help='Do not add calibration markers')

    # Feature extraction (always enabled - 2326 features per detection)
    # Kept for backwards compatibility but no longer needed
    parser.add_argument('--extract-full-features', action='store_true',
                        help='(Deprecated) Full features always extracted')

    # HTML export
    parser.add_argument('--samples-per-page', type=int, default=300)

    # Server options
    parser.add_argument('--serve', action='store_true', default=False,
                        help='Start HTTP server and wait for Ctrl+C (foreground mode)')
    parser.add_argument('--serve-background', action='store_true', default=True,
                        help='Start HTTP server in background and exit (default: True)')
    parser.add_argument('--no-serve', action='store_true',
                        help='Do not start server after processing')
    parser.add_argument('--port', type=int, default=8081,
                        help='Port for HTTP server (default: 8081)')
    parser.add_argument('--stop-server', action='store_true',
                        help='Stop any running background server and exit')

    args = parser.parse_args()

    # Handle --stop-server (exit early)
    if args.stop_server:
        setup_logging()
        stop_background_server()
        return

    # Handle --show-metadata (exit early)
    if args.show_metadata:
        print_czi_metadata(args.czi_path)
        return

    # Require --cell-type if not showing metadata
    if args.cell_type is None:
        parser.error("--cell-type is required unless using --show-metadata")

    run_pipeline(args)


if __name__ == '__main__':
    main()
