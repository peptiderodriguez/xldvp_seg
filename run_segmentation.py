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
    - islet: Pancreatic islet cells (Cellpose membrane+nuclear + SAM2)

Features extracted per cell: up to 6478 total (with --extract-deep-features and --all-channels)
    - ~78 morphological features (22 base + NMJ-specific + multi-channel stats)
    - 256 SAM2 embedding features (always)
    - 4096 ResNet-50 features (2x2048 masked+context, opt-in via --extract-deep-features)
    - 2048 DINOv2-L features (2x1024 masked+context, opt-in via --extract-deep-features)
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
    --server-status     Show status of running server (including public URL)

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
# Note: torchvision imports (tv_models, tv_transforms) are loaded lazily inside
# CellDetector strategy classes to avoid loading them when not needed.
from PIL import Image

# Import segmentation modules
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
    has_tissue,
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
from segmentation.detection.strategies.islet import IsletStrategy
from segmentation.detection.strategies.tissue_pattern import TissuePatternStrategy

# Import vessel classifier for ML-based classification
from segmentation.classification.vessel_classifier import VesselClassifier, classify_vessel
from segmentation.classification.vessel_type_classifier import VesselTypeClassifier

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Global list to track spawned processes for cleanup (foreground mode only)
_spawned_processes = []

# PID file directory for background servers (one file per port)
SERVER_PID_DIR = Path.home() / '.segmentation_servers'
# Legacy single PID file (for backwards compatibility)
SERVER_PID_FILE = Path.home() / '.segmentation_server.pid'


def parse_channel_legend_from_filename(filename: str) -> dict:
    """
    Parse channel information from filename to create legend.

    Looks for patterns like:
    - nuc488, nuc405 -> nuclear stain (keeps original like 'nuc488')
    - Bgtx647, BTX647 -> bungarotoxin (keeps original)
    - NfL750, NFL750 -> neurofilament (keeps original)
    - DAPI -> nuclear
    - SMA -> smooth muscle actin
    - _647_ -> standalone wavelength

    Args:
        filename: Slide filename (e.g., '20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch')

    Returns:
        Dict with 'red', 'green', 'blue' keys mapping to channel names,
        or None if no channels detected.
    """
    import re

    channels = []

    # Specific channel patterns - use original text from filename
    # Order: patterns that include wavelength first, then standalone wavelengths
    patterns = [
        # Patterns with wavelength embedded (capture the whole thing)
        r'nuc\d{3}',           # nuc488, nuc405
        r'bgtx\d{3}',          # Bgtx647
        r'btx\d{3}',           # BTX647
        r'nfl?\d{3}',          # NfL750, NFL750
        r'sma\d*',             # SMA, SMA488
        r'cd\d+',              # CD31, CD34
        # Named stains without wavelength
        r'dapi',
        r'bungarotoxin',
        r'neurofilament',
        # Standalone wavelengths (must be bounded by _ or - or start/end)
        r'(?:^|[_-])(\d{3})(?:[_-]|$)',  # _647_, -488-
    ]

    # Find all channel mentions with their positions
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, filename, re.IGNORECASE):
            # For grouped patterns, use group(1) if it exists
            if match.lastindex:
                text = match.group(1)
                pos = match.start(1)
            else:
                text = match.group(0)
                pos = match.start()
            found.append((pos, text))

    # Sort by position in filename and deduplicate
    found.sort(key=lambda x: x[0])
    seen = set()
    for pos, name in found:
        name_lower = name.lower()
        if name_lower not in seen:
            channels.append(name)
            seen.add(name_lower)

    if len(channels) >= 3:
        return {
            'red': channels[0],
            'green': channels[1],
            'blue': channels[2]
        }
    elif len(channels) == 2:
        return {
            'red': channels[0],
            'green': channels[1]
        }
    elif len(channels) == 1:
        return {
            'green': channels[0]  # Single channel shown as green
        }

    return None


def _get_pid_file(port: int) -> Path:
    """Get PID file path for a specific port."""
    SERVER_PID_DIR.mkdir(exist_ok=True)
    return SERVER_PID_DIR / f'server_{port}.json'


def _get_all_servers() -> list:
    """Get list of all server info dicts."""
    servers = []

    # Check legacy single PID file first
    if SERVER_PID_FILE.exists():
        try:
            data = json.loads(SERVER_PID_FILE.read_text())
            data['_pid_file'] = SERVER_PID_FILE
            servers.append(data)
        except Exception:
            pass

    # Check new per-port PID files
    if SERVER_PID_DIR.exists():
        for pid_file in SERVER_PID_DIR.glob('server_*.json'):
            try:
                data = json.loads(pid_file.read_text())
                data['_pid_file'] = pid_file
                # Skip if already covered by legacy file (same port)
                if not any(s.get('port') == data.get('port') for s in servers):
                    servers.append(data)
            except Exception:
                pass

    return servers


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


def show_server_status():
    """Show status of all running background servers."""
    servers = _get_all_servers()

    if not servers:
        print("No background servers running")
        return False

    # Filter to only running servers
    running_servers = []
    for data in servers:
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        http_running = http_pid and _pid_exists(http_pid)
        tunnel_running = tunnel_pid and _pid_exists(tunnel_pid)

        if http_running or tunnel_running:
            data['_http_running'] = http_running
            data['_tunnel_running'] = tunnel_running
            running_servers.append(data)

    if not running_servers:
        print("No background servers running")
        return False

    print("=" * 70)
    print(f"ACTIVE SEGMENTATION SERVERS ({len(running_servers)} running)")
    print("=" * 70)

    for i, data in enumerate(running_servers):
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        url = data.get('url')
        port = data.get('port', 8081)
        slide_name = data.get('slide_name', 'unknown')
        cell_type = data.get('cell_type', 'unknown')
        http_running = data.get('_http_running', False)
        tunnel_running = data.get('_tunnel_running', False)

        # Build human-readable name
        if slide_name and slide_name != 'unknown' and cell_type and cell_type != 'unknown':
            serving_name = f"{slide_name} ({cell_type.upper()})"
        elif slide_name and slide_name != 'unknown':
            serving_name = slide_name
        else:
            serving_name = f"Server on port {port}"

        if i > 0:
            print("-" * 70)

        print(f"\n[{i+1}] {serving_name}")
        print(f"    Slide:      {slide_name}")
        print(f"    Cell Type:  {cell_type}")
        print(f"    Port:       {port}")
        print(f"    Status:     HTTP={'OK' if http_running else 'DOWN'}, Tunnel={'OK' if tunnel_running else 'DOWN'}")
        if url and tunnel_running:
            print(f"    PUBLIC:     {url}")
        print(f"    LOCAL:      http://localhost:{port}")

    print("\n" + "=" * 70)
    print(f"To stop all: python run_segmentation.py --stop-server")
    print("=" * 70)
    return True


def _pid_exists(pid):
    """Check if a process with given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def start_server_and_tunnel(html_dir: Path, port: int = 8081, background: bool = False,
                            slide_name: str = None, cell_type: str = None) -> tuple:
    """
    Start HTTP server and Cloudflare tunnel for viewing results.

    Args:
        html_dir: Path to the HTML directory to serve
        port: Port for HTTP server (default 8081)
        background: If True, detach processes so they survive script exit
        slide_name: Name of the slide being served (for status display)
        cell_type: Type of cells being detected (for status display)

    Returns:
        Tuple of (http_process, tunnel_process, tunnel_url)
    """
    global _spawned_processes

    html_dir = Path(html_dir)
    if not html_dir.exists():
        logger.error(f"HTML directory does not exist: {html_dir}")
        return None, None, None

    # Check for existing server - reuse tunnel if possible
    existing_tunnel_url = None
    existing_tunnel_pid = None
    if background and SERVER_PID_FILE.exists():
        try:
            data = json.loads(SERVER_PID_FILE.read_text())
            old_http_pid = data.get('http_pid')
            old_tunnel_pid = data.get('tunnel_pid')
            old_port = data.get('port', 8081)
            existing_tunnel_url = data.get('url')

            # Check if tunnel is still running
            tunnel_running = old_tunnel_pid and _pid_exists(old_tunnel_pid)

            if tunnel_running and old_port == port:
                # Tunnel is running on same port - keep it, just restart HTTP server
                logger.info(f"Reusing existing Cloudflare tunnel: {existing_tunnel_url}")
                existing_tunnel_pid = old_tunnel_pid

                # Stop only the HTTP server (not the tunnel)
                if old_http_pid and _pid_exists(old_http_pid):
                    try:
                        os.kill(old_http_pid, signal.SIGTERM)
                        logger.info(f"Stopped old HTTP server (PID {old_http_pid})")
                        time.sleep(0.5)  # Give it time to release the port
                    except Exception:
                        pass
            else:
                # Tunnel not running or different port - start fresh
                logger.info("Starting new server and tunnel...")
                stop_background_server()
                existing_tunnel_url = None
        except Exception:
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

    # Start Cloudflare tunnel (or reuse existing one)
    tunnel_proc = None
    tunnel_url = existing_tunnel_url

    if existing_tunnel_pid:
        # Reusing existing tunnel - create a dummy process reference
        logger.info(f"Tunnel already running (PID {existing_tunnel_pid})")
        tunnel_proc = None  # We don't have the process object, just the PID

        # Update PID file with new HTTP server but keep tunnel info
        SERVER_PID_FILE.write_text(json.dumps({
            'http_pid': http_proc.pid,
            'tunnel_pid': existing_tunnel_pid,
            'port': port,
            'html_dir': str(html_dir),
            'url': existing_tunnel_url,
            'slide_name': slide_name,
            'cell_type': cell_type,
        }))
    else:
        # Need to start a new tunnel
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
                    'slide_name': slide_name,
                    'cell_type': cell_type,
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
                'slide_name': slide_name,
                'cell_type': cell_type,
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
            intensity_percentile=params.get('intensity_percentile', 98.0),
            min_area_px=params.get('min_area', 150),
            min_skeleton_length=params.get('min_skeleton_length', 30),
            max_solidity=params.get('max_solidity', 0.85),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        )
    elif cell_type == 'mk':
        return MKStrategy(
            min_area_um=params.get('mk_min_area', 200.0),
            max_area_um=params.get('mk_max_area', 2000.0),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        )
    elif cell_type == 'cell':
        return CellStrategy(
            min_area_um=params.get('min_area_um', 50),
            max_area_um=params.get('max_area_um', 200),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
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
            candidate_mode=params.get('candidate_mode', False),
            lumen_first=params.get('lumen_first', False),
            parallel_detection=params.get('parallel_detection', False),
            parallel_workers=params.get('parallel_workers', 3),
            multi_marker=params.get('multi_marker', False),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        )
    elif cell_type == 'mesothelium':
        return MesotheliumStrategy(
            target_chunk_area_um2=params.get('target_chunk_area_um2', 1500.0),
            min_ribbon_width_um=params.get('min_ribbon_width_um', 5.0),
            max_ribbon_width_um=params.get('max_ribbon_width_um', 30.0),
            min_fragment_area_um2=params.get('min_fragment_area_um2', 1500.0),
            pixel_size_um=pixel_size_um,
        )
    elif cell_type == 'islet':
        return IsletStrategy(
            membrane_channel=params.get('membrane_channel', 1),
            nuclear_channel=params.get('nuclear_channel', 4),
            min_area_um=params.get('min_area_um', 30),
            max_area_um=params.get('max_area_um', 500),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        )
    elif cell_type == 'tissue_pattern':
        return TissuePatternStrategy(
            detection_channels=params.get('detection_channels', [0, 3]),
            nuclear_channel=params.get('nuclear_channel', 4),
            min_area_um=params.get('min_area_um', 20),
            max_area_um=params.get('max_area_um', 300),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        )
    else:
        raise ValueError(f"Cell type '{cell_type}' does not have a strategy implementation. "
                         f"Supported types: nmj, mk, cell, vessel, mesothelium, islet, tissue_pattern")


# Import from canonical location (segmentation.processing.tile_processing)
from segmentation.processing.tile_processing import detections_to_features_list, process_single_tile


def apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier):
    """Apply vessel binary + 6-class classifiers to detection features in-place."""
    if vessel_classifier is not None:
        for feat in features_list:
            try:
                vessel_type, confidence = vessel_classifier.predict(feat['features'])
                feat['features']['vessel_type'] = vessel_type
                feat['features']['vessel_type_confidence'] = float(confidence)
                feat['features']['classification_method'] = 'ml'
            except Exception as e:
                vessel_type, confidence = VesselClassifier.rule_based_classify(feat['features'])
                feat['features']['vessel_type'] = vessel_type
                feat['features']['vessel_type_confidence'] = float(confidence)
                feat['features']['classification_method'] = 'rule_based_fallback'
                logger.debug(f"ML classification failed, using rule-based: {e}")

    if vessel_type_classifier is not None:
        for feat in features_list:
            try:
                vessel_type, confidence = vessel_type_classifier.predict(feat['features'])
                probs = vessel_type_classifier.predict_proba(feat['features'])
                feat['features']['vessel_type_6class'] = vessel_type
                feat['features']['vessel_type_6class_confidence'] = float(confidence)
                feat['features']['vessel_type_6class_probabilities'] = {
                    k: float(v) for k, v in probs.items()
                } if probs else {}
                feat['features']['classification_method_6class'] = 'ml_vessel_type_classifier'
            except Exception as e:
                try:
                    vessel_type, confidence = vessel_type_classifier.rule_based_classify(feat['features'])
                    feat['features']['vessel_type_6class'] = vessel_type
                    feat['features']['vessel_type_6class_confidence'] = float(confidence)
                    feat['features']['classification_method_6class'] = 'rule_based_fallback'
                except Exception as e2:
                    logger.debug(f"VesselTypeClassifier failed: {e}, {e2}")


def _compute_tile_percentiles(tile_rgb, p_low=1, p_high=99.5):
    """Compute per-channel percentiles from an entire tile for uniform HTML normalization.

    Returns dict {channel_idx: (low_val, high_val)} suitable for percentile_normalize().
    """
    valid_mask = np.max(tile_rgb, axis=2) > 0
    percentiles = {}
    for ch in range(tile_rgb.shape[2]):
        valid = tile_rgb[:, :, ch][valid_mask]
        if len(valid) > 0:
            percentiles[ch] = (float(np.percentile(valid, p_low)), float(np.percentile(valid, p_high)))
    return percentiles if percentiles else None


def classify_islet_marker(features_dict, marker_thresholds=None):
    """Classify an islet cell by dominant hormone marker.

    Uses ch2=Gcg (alpha), ch3=Ins (beta), ch5=Sst (delta).
    Returns (class_name, contour_color_rgb).
    """
    gcg = features_dict.get('ch2_mean', 0)
    ins = features_dict.get('ch3_mean', 0)
    sst = features_dict.get('ch5_mean', 0)

    if marker_thresholds:
        gcg_t, ins_t, sst_t = marker_thresholds
    else:
        # Fallback thresholds (will be overridden by population stats)
        gcg_t, ins_t, sst_t = 22, 550, 210

    # Check which markers are above background
    gcg_pos = gcg > gcg_t
    ins_pos = ins > ins_t
    sst_pos = sst > sst_t

    if not (gcg_pos or ins_pos or sst_pos):
        return 'none', (128, 128, 128)  # gray

    # Dominant = highest fold-change over threshold
    scores = []
    if gcg_pos:
        scores.append(('alpha', gcg / max(gcg_t, 1), (255, 50, 50)))    # red
    if ins_pos:
        scores.append(('beta', ins / max(ins_t, 1), (50, 255, 50)))     # green
    if sst_pos:
        scores.append(('delta', sst / max(sst_t, 1), (50, 200, 255)))   # cyan

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0], scores[0][2]


def compute_islet_marker_thresholds(all_detections):
    """Compute background thresholds for islet marker channels from population stats.

    Uses mean+3sd of the lower half (below median) as background threshold.
    """
    gcg = np.array([d['features'].get('ch2_mean', 0) for d in all_detections])
    ins = np.array([d['features'].get('ch3_mean', 0) for d in all_detections])
    sst = np.array([d['features'].get('ch5_mean', 0) for d in all_detections])

    thresholds = []
    for arr in [gcg, ins, sst]:
        med = np.median(arr)
        bg = arr[arr <= med]
        t = bg.mean() + 3 * bg.std() if len(bg) > 0 else med
        thresholds.append(float(t))

    logger.info(f"Islet marker thresholds: Gcg>{thresholds[0]:.0f}, Ins>{thresholds[1]:.0f}, Sst>{thresholds[2]:.0f}")
    return tuple(thresholds)


def filter_and_create_html_samples(
    features_list, tile_x, tile_y, tile_rgb, masks, pixel_size_um,
    slide_name, cell_type, html_score_threshold, min_area_um2=25.0,
    tile_percentiles=None, marker_thresholds=None,
):
    """Filter detections by quality and create HTML samples.

    Returns list of (sample, count) tuples for accepted detections.
    """
    samples = []
    for feat in features_list:
        features_dict = feat.get('features', {})

        rf_score = feat.get('rf_prediction', feat.get('score', 1.0))
        if rf_score is None:
            rf_score = 1.0  # No classifier loaded; show all candidates
        if rf_score < html_score_threshold:
            continue

        area_um2 = features_dict.get('area', 0) * (pixel_size_um ** 2)
        if area_um2 < min_area_um2:
            continue

        if cell_type == 'vessel':
            if features_dict.get('ring_completeness', 1.0) < 0.30:
                continue
            if features_dict.get('circularity', 1.0) < 0.15:
                continue
            if features_dict.get('wall_thickness_mean_um', 10.0) < 1.5:
                continue

        sample = create_sample_from_detection(
            tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name,
            cell_type=cell_type, tile_percentiles=tile_percentiles,
            marker_thresholds=marker_thresholds,
        )
        if sample:
            samples.append(sample)
    return samples


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


def generate_tile_grid(mosaic_info, tile_size, overlap_fraction=0.0):
    """Generate tile coordinates covering the mosaic.

    Args:
        mosaic_info: Dict with x, y, width, height
        tile_size: Size of each tile in pixels
        overlap_fraction: Fraction of tile to overlap (0.0 = no overlap, 0.1 = 10% overlap)

    Returns:
        List of tile coordinate dicts with 'x' and 'y' keys
    """
    tiles = []
    x_start = mosaic_info['x']
    y_start = mosaic_info['y']
    width = mosaic_info['width']
    height = mosaic_info['height']

    # Calculate stride (step size) based on overlap
    stride = int(tile_size * (1 - overlap_fraction))
    stride = max(stride, 1)  # Ensure at least 1 pixel stride

    for y in range(y_start, y_start + height, stride):
        for x in range(x_start, x_start + width, stride):
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

def create_sample_from_detection(tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name, cell_type='nmj', crop_size=None, tile_percentiles=None, marker_thresholds=None):
    """Create an HTML sample from a detection.

    Crop size is calculated dynamically to be 100% larger than the mask,
    ensuring the full mask is visible with context around it.
    Minimum crop size is 224px, maximum is 800px.
    """
    det_id = feat['id']
    # Use mask_label if available (multi-GPU path), otherwise parse from id (single-GPU path)
    if 'mask_label' in feat:
        det_num = feat['mask_label']
    else:
        det_num = int(det_id.split('_')[-1])
    mask = masks == det_num

    if mask.sum() == 0:
        return None

    # Get centroid
    cy, cx = feat['center'][1], feat['center'][0]

    # Calculate crop size based on mask bounding box
    # Crop should be 100% larger than mask (mask fills ~50% of crop)
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return None
    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = max(mask_h, mask_w)

    # Make crop 2x the mask size (100% larger), with min 224, max 800
    if crop_size is None:
        crop_size = max(224, min(800, int(mask_size * 2)))

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

    # Determine contour color
    features = feat['features']
    contour_color = (0, 255, 0)  # default green
    marker_class = None
    if cell_type == 'islet' and marker_thresholds is not None:
        marker_class, contour_color = classify_islet_marker(features, marker_thresholds)

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5, global_percentiles=tile_percentiles)
    crop_with_contour = draw_mask_contour(crop_norm, crop_mask, color=contour_color, thickness=2)

    # Keep at 224x224 (same as classifier input) - already correct size from crop
    pil_img = Image.fromarray(crop_with_contour)

    img_b64, mime = image_to_base64(pil_img, format='PNG')

    # Create unique ID using global coordinates (consistent with detection JSON)
    # Global center = tile origin + local center
    local_cx, local_cy = feat['center'][0], feat['center'][1]
    global_cx = tile_x + local_cx
    global_cy = tile_y + local_cy
    uid = f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}"

    # Get stats from features
    area_um2 = features.get('area', 0) * (pixel_size_um ** 2)

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    # Add marker classification for islet
    if marker_class is not None:
        stats['marker_class'] = marker_class

    # Add cell-type specific stats
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'sam2_score' in features:
        stats['confidence'] = features['sam2_score']

    # Add classifier score if available (from NMJ multi-GPU pipeline)
    if 'rf_prediction' in feat and feat['rf_prediction'] is not None:
        stats['rf_prediction'] = feat['rf_prediction']
    elif 'score' in feat and feat['score'] is not None:
        stats['rf_prediction'] = feat['score']

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

    from datetime import datetime
    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    slide_name = czi_path.stem
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 60)
    logger.info("UNIFIED SEGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Slide: {slide_name}")
    logger.info(f"Run: {run_timestamp}")
    logger.info(f"Cell type: {args.cell_type}")
    logger.info(f"Channel: {args.channel}")
    if getattr(args, 'multi_marker', False):
        logger.info("Multi-marker mode: ENABLED (auto-enabled --all-channels and --parallel-detection)")
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
        except Exception:
            n_channels = 3  # Fallback

        logger.info(f"Loading all {n_channels} channels for multi-channel analysis...")
        for ch in range(n_channels):
            if ch != args.channel:
                logger.info(f"  Loading channel {ch}...")
                ch_loader = get_loader(czi_path, load_to_ram=True, channel=ch, quiet=True)
                all_channel_data[ch] = ch_loader.get_channel_data(ch)
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

    # Apply photobleaching correction if requested (slide-wide, before tiling)
    if getattr(args, 'photobleaching_correction', False) and use_ram:
        from segmentation.preprocessing.illumination import normalize_rows_columns, estimate_band_severity

        logger.info("Applying slide-wide photobleaching correction...")

        for ch, ch_data in all_channel_data.items():
            original_dtype = ch_data.dtype

            # Report severity before
            severity_before = estimate_band_severity(ch_data)
            logger.info(f"  Channel {ch} before: row_cv={severity_before['row_cv']:.1f}%, "
                       f"col_cv={severity_before['col_cv']:.1f}% ({severity_before['severity']})")

            # Apply row/column normalization to fix banding
            # Note: uses float64 internally, may need ~4x memory temporarily
            corrected = normalize_rows_columns(ch_data)

            # Convert back to original dtype
            if original_dtype == np.uint16:
                corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
            elif original_dtype == np.uint8:
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            else:
                # Keep as float but match original dtype if possible
                corrected = corrected.astype(original_dtype)

            # Update in-place
            all_channel_data[ch] = corrected

            # Also update loader.channel_data if this is the primary channel
            if ch == args.channel:
                loader.channel_data = corrected

            # Report severity after
            severity_after = estimate_band_severity(corrected)
            logger.info(f"  Channel {ch} after:  row_cv={severity_after['row_cv']:.1f}%, "
                       f"col_cv={severity_after['col_cv']:.1f}% ({severity_after['severity']})")

            # Force garbage collection after each channel to free float64 intermediate
            gc.collect()

        logger.info("Photobleaching correction complete.")

    # Apply flat-field illumination correction (smooth out regional intensity gradients)
    if getattr(args, 'normalize_features', True) and use_ram:
        from segmentation.preprocessing.flat_field import estimate_illumination_profile

        logger.info(f"\n{'='*70}")
        logger.info("FLAT-FIELD ILLUMINATION CORRECTION")
        logger.info(f"{'='*70}")
        logger.info("Estimating slide-level illumination profile...")

        illumination_profile = estimate_illumination_profile(all_channel_data)

        for ch in all_channel_data:
            illumination_profile.correct_channel_inplace(all_channel_data[ch], ch)
            if ch == args.channel:
                loader.channel_data = all_channel_data[ch]

        gc.collect()
        logger.info("Flat-field correction complete.")

    elif getattr(args, 'normalize_features', True) and not use_ram:
        logger.warning("--normalize-features requires --load-to-ram. Skipping flat-field correction.")

    # Apply Reinhard normalization if params file provided (whole-slide, before tiling)
    if getattr(args, 'norm_params_file', None) and use_ram:
        import json as _json
        from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN

        logger.info(f"\n{'='*70}")
        logger.info("CROSS-SLIDE REINHARD NORMALIZATION (median/MAD)")
        logger.info(f"{'='*70}")
        logger.info(f"Loading params from: {args.norm_params_file}")

        with open(args.norm_params_file, 'r') as f:
            norm_params = _json.load(f)

        # Validate required keys
        required_keys = {'L_median', 'L_mad', 'a_median', 'a_mad', 'b_median', 'b_mad'}
        missing_keys = required_keys - set(norm_params.keys())
        if missing_keys:
            raise ValueError(
                f"Normalization params file missing required keys: {missing_keys}. "
                f"Required: {required_keys}"
            )

        logger.info(f"  Target: L_median={norm_params['L_median']:.2f}, L_mad={norm_params['L_mad']:.2f}")
        logger.info(f"  Target: a_median={norm_params['a_median']:.2f}, a_mad={norm_params['a_mad']:.2f}")
        logger.info(f"  Target: b_median={norm_params['b_median']:.2f}, b_mad={norm_params['b_mad']:.2f}")
        if 'n_slides' in norm_params:
            logger.info(f"  Computed from {norm_params['n_slides']} slides, {norm_params.get('n_total_pixels', '?')} pixels")

        # Build RGB image for normalization
        primary_data = loader.channel_data
        if primary_data.ndim == 3 and primary_data.shape[2] >= 3:
            # Already RGB (or more channels) — use first 3
            rgb_for_norm = primary_data[:, :, :3].copy()
        elif primary_data.ndim == 2:
            # Single channel — stack 3x (same as compute_normalization_params.py)
            rgb_for_norm = np.stack([primary_data] * 3, axis=-1)
        else:
            raise ValueError(f"Unexpected channel data shape for normalization: {primary_data.shape}")

        # Convert to uint8 if needed (Reinhard expects uint8)
        if rgb_for_norm.dtype == np.uint16:
            logger.info(f"  Converting uint16 → uint8 for normalization ({rgb_for_norm.nbytes / 1e9:.1f} GB)")
            rgb_for_norm = (rgb_for_norm / 256).astype(np.uint8)
        elif rgb_for_norm.dtype != np.uint8:
            rgb_for_norm = rgb_for_norm.astype(np.uint8)

        logger.info(f"  RGB shape: {rgb_for_norm.shape}, dtype: {rgb_for_norm.dtype} ({rgb_for_norm.nbytes / 1e9:.1f} GB)")
        logger.info(f"  Applying Reinhard normalization (this normalizes tissue blocks, preserves background)...")

        normalized_rgb = apply_reinhard_normalization_MEDIAN(rgb_for_norm, norm_params)
        del rgb_for_norm
        gc.collect()

        # Update channel data with normalized values
        if primary_data.ndim == 3 and primary_data.shape[2] >= 3:
            # Replace the 3 RGB channels
            loader.channel_data = normalized_rgb
            all_channel_data[args.channel] = normalized_rgb
            # Also update individual extra channels if loaded
            ch_keys = sorted(all_channel_data.keys())
            for i, ch_key in enumerate(ch_keys[:3]):
                all_channel_data[ch_key] = normalized_rgb[:, :, i]
        else:
            # Single channel — take first channel from normalized RGB
            normalized_single = normalized_rgb[:, :, 0].copy()
            loader.channel_data = normalized_single
            all_channel_data[args.channel] = normalized_single
            del normalized_single

        del normalized_rgb
        gc.collect()

        logger.info("  Reinhard normalization complete.")

    elif getattr(args, 'norm_params_file', None) and not use_ram:
        logger.warning("--norm-params-file requires --load-to-ram (whole-slide normalization). Skipping normalization.")

    # Generate tile grid (using global coordinates)
    overlap = getattr(args, 'tile_overlap', 0.0)
    logger.info(f"Generating tile grid (size={args.tile_size}, overlap={overlap*100:.0f}%)...")
    all_tiles = generate_tile_grid(mosaic_info, args.tile_size, overlap_fraction=overlap)
    logger.info(f"  Total tiles: {len(all_tiles)}")

    # Determine tissue detection channel BEFORE calibration
    # For islet/tissue_pattern: use nuclear channel (universal cell marker)
    tissue_channel = args.channel
    if args.cell_type == 'islet':
        tissue_channel = getattr(args, 'nuclear_channel', 4)
        logger.info(f"Islet: using DAPI (ch{tissue_channel}) for tissue detection")
    elif args.cell_type == 'tissue_pattern':
        tissue_channel = getattr(args, 'tp_nuclear_channel', 4)
        logger.info(f"Tissue pattern: using nuclear (ch{tissue_channel}) for tissue detection")

    # Calibrate tissue threshold on the SAME channel used for filtering
    logger.info("Calibrating tissue threshold...")
    variance_threshold = calibrate_tissue_threshold(
        all_tiles,
        calibration_samples=min(50, len(all_tiles)),
        channel=tissue_channel,
        tile_size=args.tile_size,
        loader=loader,  # Loader handles mosaic origin offset correctly
    )
    logger.info("Filtering to tissue-containing tiles...")
    tissue_tiles = filter_tissue_tiles(
        all_tiles,
        variance_threshold,
        channel=tissue_channel,
        tile_size=args.tile_size,
        loader=loader,  # Loader handles mosaic origin offset correctly
    )

    if len(tissue_tiles) == 0:
        logger.error("No tissue-containing tiles found!")
        return

    # Sample from tissue tiles
    n_sample = max(1, int(len(tissue_tiles) * args.sample_fraction))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    logger.info(f"Sampled {len(sampled_tiles)} tiles ({args.sample_fraction*100:.0f}% of {len(tissue_tiles)} tissue tiles)")

    # Setup output directories (timestamped to avoid overwriting previous runs)
    pct = int(args.sample_fraction * 100)
    slide_output_dir = output_dir / f'{slide_name}_{run_timestamp}_{pct}pct'
    tiles_dir = slide_output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    # Use CellDetector + strategy pattern for all cell types
    logger.info("Initializing detector...")

    # All cell types now use the new CellDetector pattern
    use_new_detector = args.cell_type in ('nmj', 'mk', 'cell', 'vessel', 'mesothelium', 'islet', 'tissue_pattern')

    if use_new_detector:
        # New CellDetector with strategy pattern
        # Note: mesothelium strategy doesn't need SAM2 (uses ridge detection)
        detector = CellDetector(device="cuda")
        segmenter = None  # Not used for these cell types

        # Load NMJ classifier if provided (supports CNN .pth or RF .pkl)
        classifier_loaded = False
        if args.cell_type == 'nmj' and getattr(args, 'nmj_classifier', None):
            from segmentation.detection.strategies.nmj import load_classifier

            logger.info(f"Loading NMJ classifier from {args.nmj_classifier}...")
            classifier_data = load_classifier(args.nmj_classifier, device=detector.device)

            if classifier_data['type'] == 'cnn':
                # CNN classifier - use transform pipeline
                from torchvision import transforms
                classifier_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                detector.models['classifier'] = classifier_data['model']
                detector.models['classifier_type'] = 'cnn'
                detector.models['transform'] = classifier_transform
                detector.models['device'] = classifier_data['device']
                logger.info("CNN classifier loaded successfully")
                classifier_loaded = True
            else:
                # RF classifier - use features directly
                # New format uses 'pipeline', legacy uses 'model'
                if 'pipeline' in classifier_data:
                    detector.models['classifier'] = classifier_data['pipeline']
                    detector.models['scaler'] = None  # Pipeline handles scaling internally
                else:
                    detector.models['classifier'] = classifier_data['model']
                    detector.models['scaler'] = classifier_data.get('scaler')
                detector.models['classifier_type'] = 'rf'
                detector.models['feature_names'] = classifier_data['feature_names']
                logger.info(f"RF classifier loaded successfully ({len(classifier_data['feature_names'])} features)")
                classifier_loaded = True
        # Load islet classifier if provided (generic RF loading)
        elif args.cell_type == 'islet' and getattr(args, 'islet_classifier', None):
            from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
            logger.info(f"Loading islet RF classifier from {args.islet_classifier}...")
            classifier_data = load_nmj_rf_classifier(args.islet_classifier)
            # load_nmj_rf_classifier always returns 'pipeline' key (wraps legacy format)
            detector.models['classifier'] = classifier_data['pipeline']
            detector.models['scaler'] = None
            detector.models['classifier_type'] = 'rf'
            detector.models['feature_names'] = classifier_data['feature_names']
            logger.info(f"Islet RF classifier loaded ({len(classifier_data['feature_names'])} features)")
            classifier_loaded = True
        # Load tissue_pattern classifier if provided (generic RF loading)
        elif args.cell_type == 'tissue_pattern' and getattr(args, 'tp_classifier', None):
            from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
            logger.info(f"Loading tissue_pattern RF classifier from {args.tp_classifier}...")
            classifier_data = load_nmj_rf_classifier(args.tp_classifier)
            detector.models['classifier'] = classifier_data['pipeline']
            detector.models['scaler'] = None
            detector.models['classifier_type'] = 'rf'
            detector.models['feature_names'] = classifier_data['feature_names']
            logger.info(f"Tissue pattern RF classifier loaded ({len(classifier_data['feature_names'])} features)")
            classifier_loaded = True

    else:
        # All cell types use CellDetector with strategy classes
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    # Auto-detect annotation run: no classifier → show ALL candidates in HTML
    # Must happen BEFORE tile processing so filter_and_create_html_samples uses threshold=0.0
    if args.cell_type in ('nmj', 'islet', 'tissue_pattern') and not classifier_loaded and args.html_score_threshold > 0:
        logger.info(f"No classifier loaded — annotation run detected. "
                     f"Overriding --html-score-threshold from {args.html_score_threshold} to 0.0 "
                     f"(will show ALL candidates for annotation)")
        args.html_score_threshold = 0.0

    # Detection parameters
    if args.cell_type == 'nmj':
        params = {
            'intensity_percentile': args.intensity_percentile,
            'min_area': args.min_area,
            'min_skeleton_length': args.min_skeleton_length,
            'max_solidity': args.max_solidity,
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
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
            'use_ml_classification': args.use_ml_classification,
            'vessel_classifier_path': args.vessel_classifier_path,
            'candidate_mode': args.candidate_mode,
            'lumen_first': getattr(args, 'lumen_first', False),
            'parallel_detection': getattr(args, 'parallel_detection', False),
            'parallel_workers': getattr(args, 'parallel_workers', 3),
            'multi_marker': getattr(args, 'multi_marker', False),
        }
    elif args.cell_type == 'mesothelium':
        params = {
            'target_chunk_area_um2': args.target_chunk_area,
            'min_ribbon_width_um': args.min_ribbon_width,
            'max_ribbon_width_um': args.max_ribbon_width,
            'min_fragment_area_um2': args.min_fragment_area,
            'pixel_size_um': pixel_size_um,
        }
    elif args.cell_type == 'islet':
        params = {
            'membrane_channel': getattr(args, 'membrane_channel', 1),
            'nuclear_channel': getattr(args, 'nuclear_channel', 4),
            'min_area_um': getattr(args, 'islet_min_area', 30.0),
            'max_area_um': getattr(args, 'islet_max_area', 500.0),
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
        }
    elif args.cell_type == 'tissue_pattern':
        params = {
            'detection_channels': [int(x) for x in args.tp_detection_channels.split(',')],
            'nuclear_channel': getattr(args, 'tp_nuclear_channel', 4),
            'min_area_um': getattr(args, 'tp_min_area', 20.0),
            'max_area_um': getattr(args, 'tp_max_area', 300.0),
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
        }
    else:
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    logger.info(f"Detection params: {params}")

    # Create strategy for new detector pattern
    strategy = None
    if use_new_detector:
        strategy = create_strategy_for_cell_type(args.cell_type, params, pixel_size_um)
        logger.info(f"Using {strategy.name} strategy: {strategy.get_config()}")

    # Load vessel classifier if ML classification requested
    vessel_classifier = None
    if args.cell_type == 'vessel' and args.use_ml_classification:
        classifier_path = args.vessel_classifier_path
        if classifier_path and Path(classifier_path).exists():
            try:
                vessel_classifier = VesselClassifier.load(classifier_path)
                logger.info(f"Loaded vessel classifier from: {classifier_path}")
                logger.info(f"  CV accuracy: {vessel_classifier.metrics.get('cv_accuracy_mean', 'N/A'):.4f}")
            except Exception as e:
                logger.warning(f"Failed to load vessel classifier: {e}")
                logger.warning("Falling back to rule-based classification")
        else:
            logger.warning("--use-ml-classification specified but no model path provided or file not found")
            logger.warning("Falling back to rule-based classification")
            if args.classify_vessel_types:
                logger.info("Using rule-based diameter thresholds for vessel classification")

    # Load VesselTypeClassifier if path provided (for multi-marker 6-type classification)
    vessel_type_classifier = None
    if args.cell_type == 'vessel' and getattr(args, 'vessel_type_classifier', None):
        classifier_path = args.vessel_type_classifier
        if Path(classifier_path).exists():
            try:
                vessel_type_classifier = VesselTypeClassifier.load(classifier_path)
                logger.info(f"Loaded VesselTypeClassifier from: {classifier_path}")
                if vessel_type_classifier.metrics:
                    accuracy = vessel_type_classifier.metrics.get('cv_accuracy_mean', 'N/A')
                    if isinstance(accuracy, float):
                        logger.info(f"  CV accuracy: {accuracy:.4f}")
                    else:
                        logger.info(f"  CV accuracy: {accuracy}")
            except Exception as e:
                logger.warning(f"Failed to load VesselTypeClassifier: {e}")
                vessel_type_classifier = None
        else:
            logger.warning(f"VesselTypeClassifier path does not exist: {classifier_path}")

    # Process tiles
    logger.info("Processing tiles...")
    all_samples = []
    all_detections = []  # Universal list with global coordinates
    total_detections = 0
    deferred_html_tiles = []  # For islet: defer HTML until marker thresholds computed

    # Multi-scale vessel detection mode
    if args.cell_type == 'vessel' and getattr(args, 'multi_scale', False):
        logger.info("=" * 60)
        logger.info("MULTI-SCALE VESSEL DETECTION ENABLED")
        logger.info("=" * 60)

        # Parse scale factors
        scales = [int(s.strip()) for s in args.scales.split(',')]
        iou_threshold = getattr(args, 'multiscale_iou_threshold', 0.3)

        logger.info(f"Scales: {scales} (coarse to fine)")
        logger.info(f"IoU threshold for deduplication: {iou_threshold}")

        # Run multi-scale detection
        # The strategy.detect_multiscale() handles all tile iteration internally
        from tqdm import tqdm as tqdm_progress

        def progress_callback(scale, done, total):
            """Progress callback for multi-scale detection."""
            pass  # tqdm handles progress display

        all_masks, detections = strategy.detect_multiscale(
            tile_getter=lambda x, y, size, ch, sf: loader.get_tile(x, y, size, ch, scale_factor=sf),
            models=detector.models,
            mosaic_width=mosaic_info['width'],
            mosaic_height=mosaic_info['height'],
            tile_size=args.tile_size,
            scales=scales,
            pixel_size_um=pixel_size_um,
            channel=args.channel,
            iou_threshold=iou_threshold,
            sample_fraction=args.sample_fraction,
            progress_callback=progress_callback,
        )

        # Convert to features list format
        features_list = detections_to_features_list(detections, args.cell_type)
        total_detections = len(features_list)

        logger.info(f"Multi-scale detection complete: {total_detections} vessels")

        # Create samples for HTML (using full-resolution crops)
        for feat in features_list:
            features_dict = feat.get('features', {})

            # Get center coordinates (already in full resolution)
            center = features_dict.get('global_center', features_dict.get('center', [0, 0]))
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = int(center[0]), int(center[1])
            else:
                continue

            # Calculate crop region in full resolution
            # Use mask bounding box if available, otherwise estimate from diameter
            diameter_um = features_dict.get('outer_diameter_um', 50)
            diameter_px = int(diameter_um / pixel_size_um)
            crop_size = max(300, min(800, int(diameter_px * 2)))

            # Generate UID
            uid = f"{slide_name}_vessel_{cx}_{cy}"

            # Create detection dict for export
            detection_dict = {
                'uid': uid,
                'slide': slide_name,
                'center': [cx, cy],
                'center_um': [cx * pixel_size_um, cy * pixel_size_um],
                'features': features_dict,
                'scale_detected': features_dict.get('scale_detected', 1),
            }
            all_detections.append(detection_dict)

            # Create HTML sample with a crop around the detection center
            half = crop_size // 2
            y1 = max(0, cy - half)
            x1 = max(0, cx - half)
            y2 = min(mosaic_info['height'], cy + half)
            x2 = min(mosaic_info['width'], cx + half)

            if len(all_channel_data) >= 3:
                ch_keys = sorted(all_channel_data.keys())[:3]
                crop_rgb = np.stack([
                    all_channel_data[ch_keys[i]][y1:y2, x1:x2]
                    for i in range(3)
                ], axis=-1)
            else:
                crop_data = loader.channel_data[y1:y2, x1:x2]
                crop_rgb = np.stack([crop_data] * 3, axis=-1)
            if crop_rgb.dtype == np.uint16:
                crop_rgb = (crop_rgb / 256).astype(np.uint8)

            if crop_rgb.size > 0:
                from segmentation.io.html_export import image_to_base64
                b64_str, _ = image_to_base64(crop_rgb)
                sample = {
                    'uid': uid,
                    'image': b64_str,
                    'stats': features_dict,
                }
                all_samples.append(sample)

        # Skip the regular tile loop - go directly to HTML export
        logger.info(f"Multi-scale mode: {len(all_detections)} detections, {len(all_samples)} HTML samples")

    elif getattr(args, 'multi_gpu', False):
        # Multi-GPU processing (all cell types)
        logger.info("=" * 60)
        logger.info(f"MULTI-GPU {args.cell_type.upper()} DETECTION ENABLED")
        logger.info("=" * 60)

        num_gpus = getattr(args, 'num_gpus', 4)
        logger.info(f"Using {num_gpus} GPUs for parallel {args.cell_type} detection")

        # Multi-channel data is recommended but not required
        if len(all_channel_data) < 2:
            logger.warning("Multi-GPU mode works best with --all-channels for multi-channel features")

        from segmentation.processing.multigpu_shm import SharedSlideManager
        from segmentation.processing.multigpu_worker import MultiGPUTileProcessor

        # Create shared memory manager
        shm_manager = SharedSlideManager()

        try:
            # Load ALL channels to shared memory (RGB display uses first 3, feature extraction needs all)
            ch_keys = sorted(all_channel_data.keys())
            n_channels = len(ch_keys)
            h, w = all_channel_data[ch_keys[0]].shape
            logger.info(f"Creating shared memory for {n_channels} channels ({h}x{w})...")
            logger.info(f"  Channel mapping: {ch_keys}")

            # Create shared memory buffer and load directly
            slide_shm_arr = shm_manager.create_slide_buffer(
                slide_name, (h, w, n_channels), all_channel_data[ch_keys[0]].dtype
            )
            for i, ch_key in enumerate(ch_keys):
                slide_shm_arr[:, :, i] = all_channel_data[ch_key]
                logger.info(f"  Loaded channel {ch_key} to shared memory slot {i}")

            # Build strategy parameters from the already-constructed params dict
            strategy_params = dict(params)  # params built by build_detection_params()

            # Get classifier path — only use if explicitly specified
            classifier_path = None
            if args.cell_type == 'nmj':
                classifier_path = getattr(args, 'nmj_classifier', None)
                if classifier_path:
                    classifier_loaded = True
                    logger.info(f"Using specified NMJ classifier: {classifier_path}")
                else:
                    logger.info("No --nmj-classifier specified — will return all candidates (annotation run)")
            elif args.cell_type == 'islet':
                classifier_path = getattr(args, 'islet_classifier', None)
                if classifier_path:
                    classifier_loaded = True
                    logger.info(f"Using specified islet classifier: {classifier_path}")
                else:
                    logger.info("No --islet-classifier specified — will return all candidates (annotation run)")
            elif args.cell_type == 'tissue_pattern':
                classifier_path = getattr(args, 'tp_classifier', None)
                if classifier_path:
                    classifier_loaded = True
                    logger.info(f"Using specified tissue_pattern classifier: {classifier_path}")
                else:
                    logger.info("No --tp-classifier specified — will return all candidates (annotation run)")

            # Create multi-GPU processor (supports all cell types)
            extract_deep = getattr(args, 'extract_deep_features', False)

            # Vessel-specific params for multi-GPU
            mgpu_cd31_channel = getattr(args, 'cd31_channel', None) if args.cell_type == 'vessel' else None
            mgpu_channel_names = None
            if args.cell_type == 'vessel' and getattr(args, 'channel_names', None):
                names = args.channel_names.split(',')
                ch_keys = sorted(all_channel_data.keys())
                mgpu_channel_names = {ch_keys[i]: name.strip()
                                      for i, name in enumerate(names)
                                      if i < len(ch_keys)}

            # Add mosaic origin to slide_info so workers can convert global→relative coords
            mgpu_slide_info = shm_manager.get_slide_info()
            mgpu_slide_info[slide_name]['mosaic_origin'] = (x_start, y_start)

            with MultiGPUTileProcessor(
                num_gpus=num_gpus,
                slide_info=mgpu_slide_info,
                cell_type=args.cell_type,
                strategy_params=strategy_params,
                pixel_size_um=pixel_size_um,
                classifier_path=classifier_path,
                extract_deep_features=extract_deep,
                extract_sam2_embeddings=True,
                detection_channel=tissue_channel,
                cd31_channel=mgpu_cd31_channel,
                channel_names=mgpu_channel_names,
                variance_threshold=variance_threshold,
                channel_keys=ch_keys,
            ) as processor:

                # Submit all tiles (add tile dimensions for worker)
                logger.info(f"Submitting {len(sampled_tiles)} tiles to {num_gpus} GPUs...")
                tile_size = args.tile_size
                for tile in sampled_tiles:
                    # Worker expects 'x', 'y', 'w', 'h' keys
                    tile_with_dims = {
                        'x': tile['x'],
                        'y': tile['y'],
                        'w': tile_size,
                        'h': tile_size,
                    }
                    processor.submit_tile(slide_name, tile_with_dims)

                # Collect results with progress bar
                from tqdm import tqdm as tqdm_progress
                pbar = tqdm_progress(total=len(sampled_tiles), desc="Processing tiles")

                results_collected = 0
                while results_collected < len(sampled_tiles):
                    result = processor.collect_result(timeout=14400)  # 4h timeout per tile (islet: ~7K cells @ 33/min = ~3.5h)
                    if result is None:
                        logger.warning("Timeout waiting for result")
                        break

                    results_collected += 1
                    pbar.update(1)

                    if result['status'] == 'success':
                        try:
                            tile = result['tile']
                            tile_x, tile_y = tile['x'], tile['y']
                            masks = result['masks']
                            features_list = result['features_list']

                            # Apply vessel classifier post-processing BEFORE saving
                            if args.cell_type == 'vessel':
                                apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                            # Save tile outputs
                            tile_id = f"tile_{tile_x}_{tile_y}"
                            tile_out = tiles_dir / tile_id
                            tile_out.mkdir(exist_ok=True)

                            # Save masks
                            with h5py.File(tile_out / f"{args.cell_type}_masks.h5", 'w') as f:
                                create_hdf5_dataset(f, 'masks', masks)

                            # Save features (includes vessel classification if applicable)
                            with open(tile_out / f"{args.cell_type}_features.json", 'w') as f:
                                json.dump(features_list, f, cls=NumpyEncoder)

                            # Add detections to global list
                            for feat in features_list:
                                all_detections.append(feat)

                            # Create samples for HTML
                            # Convert global tile coords to 0-based array indices
                            rel_tx = tile_x - x_start
                            rel_ty = tile_y - y_start
                            # Use masks.shape to handle edge tiles (smaller than tile_size at boundaries)
                            tile_h, tile_w = masks.shape[:2]
                            if args.cell_type == 'islet' and all(k in all_channel_data for k in (2, 3, 5)):
                                # Islet: R=Gcg(ch2), G=Ins(ch3), B=Sst(ch5)
                                tile_rgb_html = np.stack([
                                    all_channel_data[2][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
                                    all_channel_data[3][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
                                    all_channel_data[5][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
                                ], axis=-1)
                            elif args.cell_type == 'tissue_pattern':
                                # Tissue pattern: configurable R/G/B display channels
                                tp_disp = [int(x) for x in args.tp_display_channels.split(',')]
                                if len(tp_disp) >= 3 and all(k in all_channel_data for k in tp_disp[:3]):
                                    tile_rgb_html = np.stack([
                                        all_channel_data[tp_disp[0]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
                                        all_channel_data[tp_disp[1]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
                                        all_channel_data[tp_disp[2]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
                                    ], axis=-1)
                                else:
                                    ch_keys = sorted(all_channel_data.keys())[:3]
                                    tile_rgb_html = np.stack([
                                        all_channel_data[ch_keys[i]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w]
                                        for i in range(min(3, len(ch_keys)))
                                    ], axis=-1)
                            elif len(all_channel_data) >= 3:
                                ch_keys = sorted(all_channel_data.keys())[:3]
                                tile_rgb_html = np.stack([
                                    all_channel_data[ch_keys[i]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w]
                                    for i in range(3)
                                ], axis=-1)
                            else:
                                tile_data = loader.channel_data[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w]
                                tile_rgb_html = np.stack([tile_data] * 3, axis=-1)
                            # Convert uint16→uint8 for non-islet (NMJ/MK channels are high-signal, /256 works).
                            # For islet: keep uint16 — low-signal fluorescence (Gcg mean=16.7)
                            # would become all-zero after /256. percentile_normalize() handles uint16.
                            if tile_rgb_html.dtype == np.uint16 and args.cell_type not in ('islet', 'tissue_pattern'):
                                tile_rgb_html = (tile_rgb_html / 256).astype(np.uint8)

                            tile_pct = _compute_tile_percentiles(tile_rgb_html) if getattr(args, 'html_normalization', 'crop') == 'tile' else None

                            if args.cell_type == 'islet':
                                # Defer HTML generation until marker thresholds are computed
                                deferred_html_tiles.append({
                                    'features_list': features_list,
                                    'tile_x': tile_x, 'tile_y': tile_y,
                                    'tile_rgb_html': tile_rgb_html,
                                    'masks': masks,
                                    'tile_pct': tile_pct,
                                })
                            else:
                                html_samples = filter_and_create_html_samples(
                                    features_list, tile_x, tile_y, tile_rgb_html, masks,
                                    pixel_size_um, slide_name, args.cell_type,
                                    args.html_score_threshold,
                                    tile_percentiles=tile_pct,
                                )
                                all_samples.extend(html_samples)
                            total_detections += len(features_list)

                        except Exception as e:
                            import traceback
                            logger.error(f"Error post-processing tile ({tile_x}, {tile_y}): {e}")
                            logger.error(f"Traceback:\n{traceback.format_exc()}")

                    elif result['status'] in ('empty', 'no_tissue'):
                        pass  # Normal - no tissue in tile
                    elif result['status'] == 'error':
                        logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

                pbar.close()

                # Deferred HTML generation for islet (needs population-level marker thresholds)
                if deferred_html_tiles and args.cell_type == 'islet':
                    marker_thresholds = compute_islet_marker_thresholds(all_detections) if all_detections else None
                    # Add marker_class to each detection for JSON export
                    if marker_thresholds:
                        counts = {}
                        for det in all_detections:
                            mc, _ = classify_islet_marker(det.get('features', {}), marker_thresholds)
                            det['marker_class'] = mc
                            counts[mc] = counts.get(mc, 0) + 1
                        logger.info(f"Islet marker classification: {counts}")
                    for dt in deferred_html_tiles:
                        html_samples = filter_and_create_html_samples(
                            dt['features_list'], dt['tile_x'], dt['tile_y'],
                            dt['tile_rgb_html'], dt['masks'],
                            pixel_size_um, slide_name, args.cell_type,
                            args.html_score_threshold,
                            tile_percentiles=dt['tile_pct'],
                            marker_thresholds=marker_thresholds,
                        )
                        all_samples.extend(html_samples)
                    del deferred_html_tiles  # free memory

            logger.info(f"Multi-GPU processing complete: {total_detections} {args.cell_type} detections from {results_collected} tiles")

        finally:
            # Cleanup shared memory
            shm_manager.cleanup()

    else:
        # Standard single-scale tile processing
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

                # Convert global tile coords to 0-based array indices
                # (tiles have global mosaic coords, but RAM arrays are 0-indexed)
                rel_x = tile_x - x_start
                rel_y = tile_y - y_start

                # Build extra_channel_tiles for ALL cell types when multi-channel available
                extra_channel_tiles = None
                if len(all_channel_data) >= 2:
                    extra_channel_tiles = {}
                    for ch_idx, ch_data in all_channel_data.items():
                        extra_channel_tiles[ch_idx] = ch_data[
                            rel_y:rel_y + args.tile_size,
                            rel_x:rel_x + args.tile_size
                        ]

                # Build RGB tile from channels (matches multi-GPU path in multigpu_worker.py)
                ch_keys_sorted = sorted(all_channel_data.keys())
                n_ch = len(ch_keys_sorted)
                if n_ch >= 3:
                    tile_rgb = np.stack([
                        all_channel_data[k][rel_y:rel_y + args.tile_size, rel_x:rel_x + args.tile_size]
                        for k in ch_keys_sorted[:3]
                    ], axis=-1)
                elif n_ch == 2:
                    ch0 = all_channel_data[ch_keys_sorted[0]][rel_y:rel_y + args.tile_size, rel_x:rel_x + args.tile_size]
                    ch1 = all_channel_data[ch_keys_sorted[1]][rel_y:rel_y + args.tile_size, rel_x:rel_x + args.tile_size]
                    tile_rgb = np.stack([ch0, ch1, ch1], axis=-1)
                elif n_ch == 1:
                    ch0 = all_channel_data[ch_keys_sorted[0]][rel_y:rel_y + args.tile_size, rel_x:rel_x + args.tile_size]
                    tile_rgb = np.stack([ch0, ch0, ch0], axis=-1)
                else:
                    # Fallback: use detection channel tile
                    tile_rgb = np.stack([tile_data] * 3, axis=-1)

                # has_tissue() check on uint16 BEFORE conversion
                # For islet: use DAPI (nuclear) — more reliable tissue indicator
                try:
                    if args.cell_type == 'islet':
                        tissue_ch = getattr(args, 'nuclear_channel', 4)
                    elif args.cell_type == 'tissue_pattern':
                        tissue_ch = getattr(args, 'tp_nuclear_channel', 4)
                    else:
                        tissue_ch = args.channel
                    det_ch = extra_channel_tiles.get(tissue_ch, tile_rgb[:, :, 0]) if extra_channel_tiles else tile_rgb[:, :, 0]
                    has_tissue_flag, _ = has_tissue(det_ch, variance_threshold=variance_threshold)
                except Exception:
                    has_tissue_flag = True
                if not has_tissue_flag:
                    continue

                # Convert to uint8 for visual models (extra_channel_tiles stay uint16)
                if tile_rgb.dtype != np.uint8:
                    if tile_rgb.dtype == np.uint16:
                        tile_rgb = (tile_rgb / 256).astype(np.uint8)
                    else:
                        tile_rgb = tile_rgb.astype(np.uint8)

                # Get CD31 channel if specified (for vessel validation)
                cd31_channel_data = None
                if args.cell_type == 'vessel' and getattr(args, 'cd31_channel', None) is not None:
                    cd31_tile = loader.get_tile(tile_x, tile_y, args.tile_size, channel=args.cd31_channel)
                    if cd31_tile is not None and cd31_tile.size > 0:
                        cd31_channel_data = cd31_tile.astype(np.float32)

                # Parse vessel channel names
                channel_names = None
                if args.cell_type == 'vessel' and getattr(args, 'channel_names', None) and extra_channel_tiles:
                    names = args.channel_names.split(',')
                    actual_indices = sorted(extra_channel_tiles.keys())
                    channel_names = {actual_indices[i]: name.strip()
                                     for i, name in enumerate(names)
                                     if i < len(actual_indices)}

                # Detect using common function (CUDA retry, mask_label, UID enrichment)
                result = process_single_tile(
                    tile_rgb=tile_rgb,
                    extra_channel_tiles=extra_channel_tiles,
                    strategy=strategy,
                    models=detector.models,
                    pixel_size_um=pixel_size_um,
                    cell_type=args.cell_type,
                    slide_name=slide_name,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    cd31_channel_data=cd31_channel_data,
                    channel_names=channel_names,
                    max_retries=3,
                )

                if result is None:
                    continue

                masks, features_list = result

                # Apply vessel classifier post-processing
                if args.cell_type == 'vessel':
                    apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                # Add to global list
                for feat in features_list:
                    all_detections.append(feat)

                # Save masks and features
                tile_id = f"tile_{tile_x}_{tile_y}"
                tile_out = tiles_dir / tile_id
                tile_out.mkdir(exist_ok=True)

                with h5py.File(tile_out / f"{args.cell_type}_masks.h5", 'w') as f:
                    create_hdf5_dataset(f, 'masks', masks)

                with open(tile_out / f"{args.cell_type}_features.json", 'w') as f:
                    json.dump(features_list, f, cls=NumpyEncoder)

                # Create samples for HTML with quality filtering
                # For islet: use Gcg(ch2)/Ins(ch3)/Sst(ch5) as RGB display
                if args.cell_type == 'islet' and extra_channel_tiles and 2 in extra_channel_tiles and 3 in extra_channel_tiles and 5 in extra_channel_tiles:
                    tile_rgb_display = np.stack([
                        extra_channel_tiles[2],  # R = Gcg
                        extra_channel_tiles[3],  # G = Ins
                        extra_channel_tiles[5],  # B = Sst
                    ], axis=-1)
                    # Keep uint16 for islet — low-signal fluorescence (Gcg mean=16.7)
                    # would become all-zero after /256. percentile_normalize() handles uint16.
                elif args.cell_type == 'tissue_pattern' and extra_channel_tiles:
                    # Tissue pattern: configurable R/G/B display channels
                    tp_disp = [int(x) for x in args.tp_display_channels.split(',')]
                    if len(tp_disp) >= 3 and all(k in extra_channel_tiles for k in tp_disp[:3]):
                        tile_rgb_display = np.stack([
                            extra_channel_tiles[tp_disp[0]],
                            extra_channel_tiles[tp_disp[1]],
                            extra_channel_tiles[tp_disp[2]],
                        ], axis=-1)
                        # Keep uint16 — FISH channels can have low signal like islet
                    else:
                        tile_rgb_display = tile_rgb
                else:
                    tile_rgb_display = tile_rgb
                tile_pct = _compute_tile_percentiles(tile_rgb_display) if getattr(args, 'html_normalization', 'crop') == 'tile' else None

                if args.cell_type == 'islet':
                    # Defer HTML generation until marker thresholds are computed
                    deferred_html_tiles.append({
                        'features_list': features_list,
                        'tile_x': tile_x, 'tile_y': tile_y,
                        'tile_rgb_html': tile_rgb_display,
                        'masks': masks.copy(),
                        'tile_pct': tile_pct,
                    })
                else:
                    html_samples = filter_and_create_html_samples(
                        features_list, tile_x, tile_y, tile_rgb_display, masks,
                        pixel_size_um, slide_name, args.cell_type,
                        args.html_score_threshold,
                        tile_percentiles=tile_pct,
                    )
                    all_samples.extend(html_samples)
                total_detections += len(features_list)

                del tile_data, tile_rgb, masks
                if extra_channel_tiles is not None:
                    del extra_channel_tiles
                    extra_channel_tiles = None
                if cd31_channel_data is not None:
                    del cd31_channel_data
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                logger.error(f"Error processing tile ({tile_x}, {tile_y}): {e}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

    # Deferred HTML generation for islet (needs population-level marker thresholds)
    if deferred_html_tiles and args.cell_type == 'islet':
        marker_thresholds = compute_islet_marker_thresholds(all_detections) if all_detections else None
        if marker_thresholds:
            counts = {}
            for det in all_detections:
                mc, _ = classify_islet_marker(det.get('features', {}), marker_thresholds)
                det['marker_class'] = mc
                counts[mc] = counts.get(mc, 0) + 1
            logger.info(f"Islet marker classification: {counts}")
        for dt in deferred_html_tiles:
            html_samples = filter_and_create_html_samples(
                dt['features_list'], dt['tile_x'], dt['tile_y'],
                dt['tile_rgb_html'], dt['masks'],
                pixel_size_um, slide_name, args.cell_type,
                args.html_score_threshold,
                tile_percentiles=dt['tile_pct'],
                marker_thresholds=marker_thresholds,
            )
            all_samples.extend(html_samples)
        del deferred_html_tiles

    logger.info(f"Total detections (pre-dedup): {len(all_detections)}")

    # Deduplication: tile overlap causes same detection in adjacent tiles
    # Uses actual mask pixel overlap (loads HDF5 mask files) for accurate dedup
    if getattr(args, 'tile_overlap', 0) > 0 and len(all_detections) > 0:
        from segmentation.processing.deduplication import deduplicate_by_mask_overlap
        pre_dedup = len(all_detections)
        mask_fn = f'{args.cell_type}_masks.h5'
        dedup_sort = 'confidence' if getattr(args, 'dedup_by_confidence', False) else 'area'
        all_detections = deduplicate_by_mask_overlap(
            all_detections, tiles_dir, min_overlap_fraction=0.1,
            mask_filename=mask_fn, sort_by=dedup_sort,
        )

        # Filter HTML samples to match deduped detections and remove duplicate UIDs
        deduped_uids = {det.get('uid', det.get('id', '')) for det in all_detections}
        seen_uids = set()
        unique_samples = []
        for s in all_samples:
            uid = s.get('uid', '')
            if uid in deduped_uids and uid not in seen_uids:
                seen_uids.add(uid)
                unique_samples.append(s)
        logger.info(f"Dedup: {len(all_samples)} HTML samples -> {len(unique_samples)} (removed {len(all_samples) - len(unique_samples)} duplicate UIDs)")
        all_samples = unique_samples

    # (annotation threshold override already applied before tile loop)

    # Sort samples: with classifier → descending RF score; without → ascending area
    if args.cell_type in ('nmj', 'islet', 'tissue_pattern') and classifier_loaded:
        all_samples.sort(key=lambda x: x['stats'].get('rf_prediction') or 0, reverse=True)
    else:
        all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0))

    # Export to HTML
    if args.cell_type in ('nmj', 'islet', 'tissue_pattern') and len(all_detections) > len(all_samples):
        logger.info(f"Total detections (all scores): {len(all_detections)}, "
                     f"shown in HTML (rf_prediction >= {args.html_score_threshold}): {len(all_samples)}")
    logger.info(f"Exporting to HTML ({len(all_samples)} samples)...")
    html_dir = slide_output_dir / "html"

    # Channel legend for multi-channel images
    channel_legend = None
    if args.cell_type == 'nmj' and getattr(args, 'all_channels', False):
        channel_legend = parse_channel_legend_from_filename(slide_name)
    elif args.cell_type == 'islet':
        channel_legend = {
            'red': 'Gcg (alpha)',
            'green': 'Ins (beta)',
            'blue': 'Sst (delta)',
        }
    elif args.cell_type == 'tissue_pattern':
        tp_disp = [int(x) for x in args.tp_display_channels.split(',')]
        probe_legend = parse_channel_legend_from_filename(slide_name)
        if probe_legend:
            channel_legend = probe_legend
        else:
            channel_legend = {
                'red': f'Ch{tp_disp[0]}' if len(tp_disp) > 0 else 'Ch0',
                'green': f'Ch{tp_disp[1]}' if len(tp_disp) > 1 else 'Ch1',
                'blue': f'Ch{tp_disp[2]}' if len(tp_disp) > 2 else 'Ch2',
            }

    # Use slide_name + timestamp as experiment_name so each run gets its own
    # localStorage namespace (prevents stale annotations from prior runs).
    # The timestamp ensures re-runs of the same slide don't collide.
    # For round-2 (--prior-annotations), prior labels come from the preload JS
    # file (not localStorage), so a unique key is safe for all runs.
    prior_ann = getattr(args, 'prior_annotations', None)
    experiment_name = f"{slide_name}_{run_timestamp}_{pct}pct"

    export_samples_to_html(
        all_samples,
        html_dir,
        args.cell_type,
        samples_per_page=args.samples_per_page,
        title=f"{args.cell_type.upper()} Annotation Review",
        page_prefix=f'{args.cell_type}_page',
        experiment_name=experiment_name,
        file_name=f"{slide_name}.czi",
        pixel_size_um=pixel_size_um,
        tiles_processed=len(sampled_tiles),
        tiles_total=len(all_tiles),
        channel_legend=channel_legend,
        prior_annotations=prior_ann,
    )

    # Save all detections with universal IDs and global coordinates
    detections_file = slide_output_dir / f'{args.cell_type}_detections.json'
    with open(detections_file, 'w') as f:
        json.dump(all_detections, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  Saved {len(all_detections)} detections to {detections_file}")

    # Export CSV with contour coordinates for easy import
    csv_file = slide_output_dir / f'{args.cell_type}_coordinates.csv'
    with open(csv_file, 'w') as f:
        # Header
        if args.cell_type == 'vessel':
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,outer_diameter_um,wall_thickness_um,confidence\n')
            for det in all_detections:
                # Skip detections with missing or invalid global coordinates
                g_center = det.get('global_center')
                g_center_um = det.get('global_center_um')
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get('features', {})
                f.write(f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                        f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},"
                        f"{feat.get('outer_diameter_um', 0):.2f},{feat.get('wall_thickness_mean_um', 0):.2f},"
                        f"{feat.get('confidence', 'unknown')}\n")
        else:
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2\n')
            for det in all_detections:
                # Skip detections with missing or invalid global coordinates
                g_center = det.get('global_center')
                g_center_um = det.get('global_center_um')
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get('features', {})
                area_um2 = feat.get('area', 0) * (pixel_size_um ** 2)
                f.write(f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                        f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},{area_um2:.2f}\n")
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
        'total_detections': len(all_detections),
        'html_displayed': len(all_samples),
        'params': params,
        'detections_file': str(detections_file),
        'coordinates_file': str(csv_file),
    }

    with open(slide_output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Cleanup detector resources
    if use_new_detector and detector is not None:
        detector.cleanup()

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total detections: {len(all_detections)}")
    logger.info(f"Displayed in HTML: {len(all_samples)} (score >= {args.html_score_threshold})")
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
        http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
            html_dir, port, background=False,
            slide_name=slide_name, cell_type=args.cell_type
        )
        if http_proc is not None:
            wait_for_server_shutdown(http_proc, tunnel_proc)
    elif serve_background:
        # Background mode: start and exit script
        start_server_and_tunnel(
            html_dir, port, background=True,
            slide_name=slide_name, cell_type=args.cell_type
        )
        # Show final server status
        print("")
        show_server_status()


def main():
    parser = argparse.ArgumentParser(
        description='Unified Cell Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required (unless using utility commands like --stop-server or --server-status)
    parser.add_argument('--czi-path', type=str, required=False, help='Path to CZI file')
    parser.add_argument('--cell-type', type=str, default=None,
                        choices=['nmj', 'mk', 'cell', 'vessel', 'mesothelium', 'islet', 'tissue_pattern'],
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
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ~/nmj_output for NMJ, ~/mk_output for MK, ~/vessel_output for vessel)')

    # Tile processing
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size in pixels')
    parser.add_argument('--tile-overlap', type=float, default=0.10, help='Tile overlap fraction (0.0-0.5, default: 0.10 = 10%% overlap)')
    parser.add_argument('--sample-fraction', type=float, default=0.10, help='Fraction of tissue tiles (default: 10%%)')
    parser.add_argument('--channel', type=int, default=None,
                        help='Primary channel index for detection (default: 1 for NMJ, 0 for MK/vessel/cell)')
    parser.add_argument('--all-channels', action='store_true',
                        help='Load all channels for multi-channel analysis (NMJ specificity checking)')
    parser.add_argument('--channel-names', type=str, default=None,
                        help='Comma-separated channel names for feature naming (e.g., "nuclear,sma,pm,cd31" or "nuclear,sma,pm,lyve1")')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    parser.add_argument('--photobleaching-correction', action='store_true',
                        help='Apply slide-wide photobleaching correction (fixes horizontal/vertical banding)')
    parser.add_argument('--norm-params-file', type=str, default=None,
                        help='Path to pre-computed Reinhard normalization params JSON (from compute_normalization_params.py). '
                             'Applies whole-slide Lab-space normalization before tile processing.')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='Apply flat-field illumination correction (default: ON)')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false',
                        help='Disable flat-field correction (use raw intensities)')
    parser.add_argument('--html-normalization', choices=['tile', 'crop'], default='tile',
                        help='HTML crop normalization scope: tile=shared percentiles per tile, crop=per-crop (default: tile)')

    # NMJ parameters
    parser.add_argument('--intensity-percentile', type=float, default=98)
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--min-skeleton-length', type=int, default=30)
    parser.add_argument('--max-solidity', type=float, default=0.85,
                        help='Maximum solidity for NMJ detection (branched structures have low solidity)')
    parser.add_argument('--nmj-classifier', type=str, default=None,
                        help='Path to trained NMJ classifier (.pth file)')
    parser.add_argument('--html-score-threshold', type=float, default=0.5,
                        help='Minimum rf_prediction score to show in HTML (default 0.5). '
                             'All detections still saved to JSON regardless. '
                             'Auto-set to 0.0 when no classifier is loaded (annotation run). '
                             'Use --html-score-threshold 0.0 to show ALL candidates explicitly.')
    parser.add_argument('--prior-annotations', type=str, default=None,
                        help='Path to prior annotations JSON file (from round-1 annotation). '
                             'Pre-loads annotations into HTML localStorage so round-1 labels '
                             'are visible during round-2 review after classifier training.')

    # MK parameters (area in um²)
    parser.add_argument('--mk-min-area', type=float, default=200.0,
                        help='Minimum MK area in um² (default 200)')
    parser.add_argument('--mk-max-area', type=float, default=2000.0,
                        help='Maximum MK area in um² (default 2000)')

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
                        help='Auto-classify vessels by size (capillary/arteriole/artery) using rule-based method')
    parser.add_argument('--use-ml-classification', action='store_true',
                        help='Use ML-based vessel classification (requires trained model)')
    parser.add_argument('--vessel-classifier-path', type=str, default=None,
                        help='Path to trained vessel classifier (.joblib). If not provided with '
                             '--use-ml-classification, falls back to rule-based classification.')
    parser.add_argument('--candidate-mode', action='store_true',
                        help='Enable candidate generation mode for vessel detection. '
                             'Relaxes all thresholds to catch more potential vessels (higher recall). '
                             'Includes detection_confidence score (0-1) for each candidate. '
                             'Use for generating training data for manual annotation + RF classifier.')
    parser.add_argument('--lumen-first', action='store_true',
                        help='Enable lumen-first vessel detection mode. '
                             'Instead of detecting SMA+ walls first (contour hierarchy), this mode '
                             'finds dark lumens using Otsu threshold, then validates bright SMA+ wall '
                             'surrounding each lumen. Better for detecting vessels with incomplete walls. '
                             'Fits ellipses to candidates and filters by shape quality.')
    parser.add_argument('--parallel-detection', action='store_true',
                        help='Enable parallel multi-marker vessel detection. '
                             'Runs SMA, CD31, and LYVE1 detection in parallel using CPU threads. '
                             'Requires --channel-names to specify marker channels. '
                             'Example: --channel-names "nuclear,sma,cd31,lyve1" --parallel-detection')
    parser.add_argument('--parallel-workers', type=int, default=3,
                        help='Number of parallel workers for multi-marker detection (default: 3). '
                             'One worker per marker type (SMA, CD31, LYVE1).')
    parser.add_argument('--multi-marker', action='store_true',
                        help='Enable full multi-marker vessel detection pipeline. '
                             'Automatically enables --all-channels and --parallel-detection. '
                             'Detects SMA+ rings, CD31+ capillaries, and LYVE1+ lymphatics. '
                             'Merges overlapping candidates from different markers. '
                             'Extracts multi-channel features for downstream classification. '
                             'Example: --multi-marker --channel-names "nuclear,sma,cd31,lyve1"')
    parser.add_argument('--vessel-type-classifier', type=str, default=None,
                        help='Path to trained VesselTypeClassifier model (.joblib) for 6-type '
                             'vessel classification (artery/arteriole/vein/capillary/lymphatic/'
                             'collecting_lymphatic). Used with --multi-marker for automated '
                             'vessel type prediction based on marker profiles and morphology.')

    # Multi-scale vessel detection
    parser.add_argument('--multi-scale', action='store_true',
                        help='Enable multi-scale vessel detection. Detects at multiple resolutions '
                             '(1/8x, 1/4x, 1x) to capture all vessel sizes and avoid cross-tile '
                             'fragmentation. Large vessels are detected at coarse scale (1/8x) '
                             'where they fit within a single tile. Requires --cell-type vessel.')
    parser.add_argument('--scales', type=str, default='8,4,1',
                        help='Comma-separated scale factors for multi-scale detection (default: "8,4,1"). '
                             'Numbers represent downsampling factors: 8=1/8x (coarse, large vessels), '
                             '4=1/4x (medium), 1=full resolution (small vessels, capillaries). '
                             'Detection runs coarse-to-fine with IoU deduplication.')
    parser.add_argument('--multiscale-iou-threshold', type=float, default=0.3,
                        help='IoU threshold for deduplicating vessels detected at multiple scales '
                             '(default: 0.3). If a vessel is detected at both coarse and fine scales '
                             'with IoU > threshold, the finer scale detection is kept.')

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

    # Islet parameters
    parser.add_argument('--membrane-channel', type=int, default=1,
                        help='Membrane marker channel index for islet Cellpose input (default: 1, AF633)')
    parser.add_argument('--nuclear-channel', type=int, default=4,
                        help='Nuclear marker channel index for islet Cellpose input (default: 4, DAPI)')
    parser.add_argument('--islet-classifier', type=str, default=None,
                        help='Path to trained islet RF classifier (.pkl)')
    parser.add_argument('--islet-min-area', type=float, default=30.0,
                        help='Minimum islet cell area in um² (default 30)')
    parser.add_argument('--islet-max-area', type=float, default=500.0,
                        help='Maximum islet cell area in um² (default 500)')
    parser.add_argument('--dedup-by-confidence', action='store_true', default=False,
                        help='Sort by confidence (score) instead of area during deduplication. '
                             'Automatically enabled for islet cell type.')

    # Tissue pattern parameters
    parser.add_argument('--tp-detection-channels', type=str, default='0,3',
                        help='Comma-separated channel indices to sum for tissue_pattern detection (default: 0,3 = Slc17a7+Gad1)')
    parser.add_argument('--tp-nuclear-channel', type=int, default=4,
                        help='Nuclear channel for tissue detection (default: 4, Hoechst)')
    parser.add_argument('--tp-display-channels', type=str, default='0,3,1',
                        help='Comma-separated R,G,B channel indices for HTML display (default: 0,3,1 = Slc17a7/Gad1/Htr2a)')
    parser.add_argument('--tp-classifier', type=str, default=None,
                        help='Path to trained tissue_pattern RF classifier (.pkl)')
    parser.add_argument('--tp-min-area', type=float, default=20.0,
                        help='Minimum cell area in um² for tissue_pattern (default 20)')
    parser.add_argument('--tp-max-area', type=float, default=300.0,
                        help='Maximum cell area in um² for tissue_pattern (default 300)')

    # Feature extraction options
    parser.add_argument('--extract-full-features', action='store_true',
                        help='Extract full features including SAM2 embeddings')
    parser.add_argument('--extract-deep-features', action='store_true',
                        help='Extract ResNet and DINOv2 features (opt-in, default morph+SAM2 only)')
    parser.add_argument('--skip-deep-features', action='store_true',
                        help='Deprecated: deep features are off by default now. Use --extract-deep-features to enable.')

    # Multi-GPU processing
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Enable multi-GPU processing. Distributes tiles across GPUs via shared memory.')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use for multi-GPU processing (default: 4)')

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
    parser.add_argument('--server-status', action='store_true',
                        help='Show status of running server (including public URL) and exit')

    args = parser.parse_args()

    # Handle --stop-server (exit early)
    if args.stop_server:
        setup_logging()
        stop_background_server()
        return

    # Handle --server-status (exit early)
    if args.server_status:
        show_server_status()
        return

    # Handle --show-metadata (exit early)
    if args.show_metadata:
        print_czi_metadata(args.czi_path)
        return

    # Require --czi-path for actual pipeline runs
    if args.czi_path is None:
        parser.error("--czi-path is required (unless using --stop-server, --server-status, or --show-metadata)")

    # Require --cell-type if not showing metadata
    if args.cell_type is None:
        parser.error("--cell-type is required unless using --show-metadata")

    # Cell-type-dependent defaults for output-dir and channel
    if args.output_dir is None:
        default_dirs = {'nmj': '/home/dude/nmj_output', 'mk': '/home/dude/mk_output',
                        'vessel': '/home/dude/vessel_output', 'cell': '/home/dude/cell_output',
                        'mesothelium': '/home/dude/mesothelium_output',
                        'islet': '/home/dude/islet_output'}
        args.output_dir = default_dirs.get(args.cell_type, f'/home/dude/{args.cell_type}_output')
    if args.channel is None:
        if args.cell_type == 'nmj':
            args.channel = 1
        elif args.cell_type == 'islet':
            args.channel = getattr(args, 'membrane_channel', 1)
        elif args.cell_type == 'tissue_pattern':
            # Primary channel = first detection channel (for tissue loading)
            args.channel = int(args.tp_detection_channels.split(',')[0])
        else:
            args.channel = 0

    # Handle --cell-type islet: auto-enable all-channels, dedup by area (largest wins)
    if args.cell_type == 'islet':
        args.all_channels = True

    # Handle --cell-type tissue_pattern: auto-enable all-channels
    if args.cell_type == 'tissue_pattern':
        args.all_channels = True

    # Handle --multi-marker: automatically enable dependent flags
    if getattr(args, 'multi_marker', False):
        if args.cell_type != 'vessel':
            parser.error("--multi-marker is only valid with --cell-type vessel")
        # Auto-enable all-channels and parallel-detection
        args.all_channels = True
        args.parallel_detection = True
        # Note: logger not available yet, will log in run_pipeline()

    run_pipeline(args)


if __name__ == '__main__':
    main()
