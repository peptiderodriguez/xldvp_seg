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
            ring_only=params.get('ring_only', False),
            parallel_detection=params.get('parallel_detection', False),
            parallel_workers=params.get('parallel_workers', 3),
            multi_marker=params.get('multi_marker', False),
            extract_deep_features=params.get('extract_deep_features', False),
            extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
            smooth_contours=params.get('smooth_contours', True),
            smooth_contours_factor=params.get('smooth_contours_factor', 3.0),
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
            marker_signal_factor=params.get('marker_signal_factor', 2.0),
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


def classify_islet_marker(features_dict, marker_thresholds=None, marker_map=None):
    """Classify an islet cell by dominant hormone marker.

    Uses NORMALIZED channel values (same as HTML display) so contour color
    matches what the user sees. Marker channels are defined by marker_map.

    Args:
        features_dict: dict with 'ch{N}_mean' feature keys
        marker_thresholds: (norm_ranges, ch_thresholds, ratio_min) from compute_islet_marker_thresholds()
        marker_map: dict mapping marker name → CZI channel index, e.g. {'gcg': 2, 'ins': 3, 'sst': 5}

    Returns (class_name, contour_color_rgb).
    """
    if marker_map is None:
        marker_map = {'gcg': 2, 'ins': 3, 'sst': 5}

    # Build ordered marker list with colors (R, G, B for first 3 markers)
    _marker_colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
    marker_names = list(marker_map.keys())
    marker_vals = {}
    for name in marker_names:
        ch_idx = marker_map[name]
        marker_vals[name] = features_dict.get(f'ch{ch_idx}_mean', 0)

    if marker_thresholds is None:
        return 'none', (128, 128, 128)

    norm_ranges, ch_thresholds, ratio_min = marker_thresholds

    # Normalize to 0-1 using same percentiles as HTML display
    def _norm(val, ch_key):
        lo, hi = norm_ranges.get(ch_key, (0, 1))
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    normed = {}
    positive = {}
    for name in marker_names:
        ch_key = f'ch{marker_map[name]}'
        normed[name] = _norm(marker_vals[name], ch_key)
        positive[name] = normed[name] >= ch_thresholds.get(ch_key, 0.5)

    # Gate: must exceed at least one channel's threshold
    if not any(positive.values()):
        return 'none', (128, 128, 128)

    # Ratio classification among gated cells
    markers = []
    for i, name in enumerate(marker_names):
        color = _marker_colors[i] if i < len(_marker_colors) else (200, 200, 200)
        markers.append((name, normed[name], color))
    markers.sort(key=lambda x: x[1], reverse=True)
    best_name, best_val, best_color = markers[0]
    second_val = markers[1][1] if len(markers) > 1 else 0

    if second_val > 0 and best_val / second_val < ratio_min:
        return 'multi', (255, 170, 0)  # orange

    return best_name, best_color


def compute_islet_marker_thresholds(all_detections, vis_threshold_overrides=None, ratio_min=1.5,
                                    marker_map=None):
    """Compute per-channel thresholds for islet marker classification.

    For each marker channel:
      1. Normalize raw values to [0,1] using population p1-p99.5 percentiles
      2. Otsu threshold for channels with bimodal separation
      3. median+3*MAD for channels with low dynamic range
      4. Fallback: if Otsu gives >15% positive, use med+3*MAD instead

    Args:
        all_detections: list of detection dicts with features
        vis_threshold_overrides: optional dict {ch_key: float} to manually override
            per-channel thresholds (e.g. {'ch5': 0.6})
        ratio_min: dominant marker must be >= ratio_min * second-highest
            to be classified as single-marker. Otherwise → "multi".
        marker_map: dict mapping marker name → CZI channel index,
            e.g. {'gcg': 2, 'ins': 3, 'sst': 5}

    Returns (norm_ranges, ch_thresholds, ratio_min) for classify_islet_marker(),
        or None if too few detections for reliable thresholds.
    """
    if marker_map is None:
        marker_map = {'gcg': 2, 'ins': 3, 'sst': 5}

    if len(all_detections) < 10:
        logger.warning(f"Only {len(all_detections)} detections — too few for reliable "
                       "marker thresholds. Skipping marker classification.")
        return None

    from skimage.filters import threshold_otsu

    # Build arrays from features using marker_map channel indices
    marker_arrays = {}
    for name, ch_idx in marker_map.items():
        marker_arrays[name] = np.array([
            d.get('features', {}).get(f'ch{ch_idx}_mean', 0) for d in all_detections
        ])

    norm_ranges = {}
    ch_thresholds = {}

    for name, ch_idx in marker_map.items():
        ch_key = f'ch{ch_idx}'
        ch_name = name.capitalize()
        arr = marker_arrays[name]
        # Exclude zero-valued entries (cells with no signal or missing features)
        # to prevent pulling down p1 percentile and skewing Otsu thresholds
        arr_pos = arr[arr > 0]
        if len(arr_pos) < 10:
            logger.warning(f"Only {len(arr_pos)} cells with nonzero {ch_name} — using full array for percentiles")
            arr_pos = arr
        lo = float(np.percentile(arr_pos, 1))
        hi = float(np.percentile(arr_pos, 99.5))
        norm_ranges[ch_key] = (lo, hi)

        # Normalize to [0, 1]
        if hi > lo:
            norm_vals = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            norm_vals = np.zeros_like(arr)

        # Otsu on normalized values
        try:
            otsu_t = float(threshold_otsu(norm_vals))
        except ValueError:
            otsu_t = 0.5

        # median + 3*MAD — robust for unimodal distributions with a signal tail
        med = float(np.median(norm_vals))
        mad = float(np.median(np.abs(norm_vals - med)))
        mad3_t = med + 3 * 1.4826 * mad  # 1.4826 scales MAD to std for normal dist

        # Use Otsu if it gives <=15% positive, otherwise fall back to med+3*MAD.
        # Channels with low dynamic range (like Sst) produce poor Otsu thresholds
        # because the distribution isn't clearly bimodal.
        n_otsu = int(np.sum(norm_vals > otsu_t))
        otsu_pct = 100 * n_otsu / len(arr) if len(arr) > 0 else 0

        if otsu_pct <= 15:
            auto_t = otsu_t
            method = 'otsu'
        else:
            auto_t = mad3_t
            method = 'med+3MAD'

        # Allow manual override
        if vis_threshold_overrides and ch_key in vis_threshold_overrides:
            ch_thresholds[ch_key] = vis_threshold_overrides[ch_key]
            logger.info(f"Islet {ch_name}({ch_key}): p1={lo:.1f}, p99.5={hi:.1f}, "
                        f"auto={auto_t:.3f} ({method}), OVERRIDE={vis_threshold_overrides[ch_key]:.3f}")
        else:
            ch_thresholds[ch_key] = auto_t
            logger.info(f"Islet {ch_name}({ch_key}): p1={lo:.1f}, p99.5={hi:.1f}, "
                        f"threshold={auto_t:.3f} ({method})")

        # Count positive cells at this threshold
        raw_cutoff = lo + ch_thresholds[ch_key] * (hi - lo)
        n_pos = int(np.sum(arr > raw_cutoff))
        logger.info(f"  {ch_name}-positive: {n_pos} cells ({100*n_pos/len(arr):.1f}%) "
                    f"[raw > {raw_cutoff:.0f}]")

    logger.info(f"Islet marker ratio_min: {ratio_min}x (dominant must be >= {ratio_min}x runner-up)")
    return (norm_ranges, ch_thresholds, ratio_min)


def filter_and_create_html_samples(
    features_list, tile_x, tile_y, tile_rgb, masks, pixel_size_um,
    slide_name, cell_type, html_score_threshold, min_area_um2=25.0,
    tile_percentiles=None, marker_thresholds=None, marker_map=None,
    candidate_mode=False,
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

        if cell_type == 'vessel' and not candidate_mode:
            if features_dict.get('ring_completeness', 1.0) < 0.30:
                continue
            if features_dict.get('circularity', 1.0) < 0.15:
                continue
            wt = features_dict.get('wall_thickness_mean_um')
            if wt is not None and wt < 1.5:
                continue

        sample = create_sample_from_detection(
            tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name,
            cell_type=cell_type, tile_percentiles=tile_percentiles,
            marker_thresholds=marker_thresholds, marker_map=marker_map,
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

def create_sample_from_detection(tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name, cell_type='nmj', crop_size=None, tile_percentiles=None, marker_thresholds=None, marker_map=None):
    """Create an HTML sample from a detection.

    Crop size is calculated dynamically to be 100% larger than the mask,
    ensuring the full mask is visible with context around it.
    Minimum crop size is 224px, maximum is 800px.
    """
    det_id = feat['id']
    # Use mask_label if available, otherwise parse from id (legacy fallback)
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
        marker_class, contour_color = classify_islet_marker(features, marker_thresholds, marker_map=marker_map)

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
        stats['marker_color'] = f'#{contour_color[0]:02x}{contour_color[1]:02x}{contour_color[2]:02x}'

    # Add vessel detection method provenance
    if 'detection_method' in features:
        dm = features['detection_method']
        stats['detection_method'] = ', '.join(dm) if isinstance(dm, list) else dm

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

    # Always log channel metadata so the log is self-documenting
    try:
        _czi_meta = get_czi_metadata(czi_path)
        logger.info(f"  CZI channels ({_czi_meta['n_channels']}):")
        for _ch in _czi_meta['channels']:
            _ex = f"{_ch['excitation_nm']:.0f}" if _ch['excitation_nm'] else "?"
            _em = f"{_ch['emission_nm']:.0f}" if _ch['emission_nm'] else "?"
            _label = _ch['fluorophore'] if _ch['fluorophore'] != 'N/A' else _ch['name']
            logger.info(f"    [{_ch['index']}] {_ch['name']:<20s}  Ex {_ex} → Em {_em} nm  ({_label})")
    except Exception as _e:
        logger.warning(f"  Could not read channel metadata: {_e}")

    # Load additional channels if --all-channels specified (for NMJ specificity checking)
    all_channel_data = {args.channel: loader.channel_data}  # Primary channel
    if getattr(args, 'all_channels', False) and use_ram:
        # Determine which channels to load
        if getattr(args, 'channels', None):
            ch_list = [int(x.strip()) for x in args.channels.split(',')]
            logger.info(f"Loading specified channels {ch_list} for multi-channel analysis...")
        else:
            # Load all channels from CZI
            try:
                dims = loader.reader.get_dims_shape()[0]
                n_channels = dims.get('C', (0, 3))[1]  # Default to 3 channels
            except Exception:
                n_channels = 3  # Fallback
            ch_list = list(range(n_channels))
            logger.info(f"Loading all {len(ch_list)} channels for multi-channel analysis...")

        for ch in ch_list:
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
    if getattr(args, 'resume_from', None):
        slide_output_dir = Path(args.resume_from)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    else:
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
            'ring_only': getattr(args, 'ring_only', False),
            'parallel_detection': getattr(args, 'parallel_detection', False),
            'parallel_workers': getattr(args, 'parallel_workers', 3),
            'multi_marker': getattr(args, 'multi_marker', False),
            'smooth_contours': not getattr(args, 'no_smooth_contours', False),
            'smooth_contours_factor': getattr(args, 'smooth_contours_factor', 3.0),
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
            'marker_signal_factor': getattr(args, 'marker_signal_factor', 2.0),
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
    is_multiscale = args.cell_type == 'vessel' and getattr(args, 'multi_scale', False)

    # ---- Shared memory creation (used by BOTH regular and multiscale paths) ----
    num_gpus = getattr(args, 'num_gpus', 1)

    if len(all_channel_data) < 2:
        logger.warning("Pipeline works best with --all-channels for multi-channel features")

    from segmentation.processing.multigpu_shm import SharedSlideManager
    from segmentation.processing.multigpu_worker import MultiGPUTileProcessor

    shm_manager = SharedSlideManager()

    try:
        # Load ALL channels to shared memory (RGB display uses first 3, feature extraction needs all)
        ch_keys = sorted(all_channel_data.keys())
        n_channels = len(ch_keys)
        h, w = all_channel_data[ch_keys[0]].shape
        logger.info(f"Creating shared memory for {n_channels} channels ({h}x{w})...")
        logger.info(f"  Channel mapping: {ch_keys}")

        slide_shm_arr = shm_manager.create_slide_buffer(
            slide_name, (h, w, n_channels), all_channel_data[ch_keys[0]].dtype
        )
        for i, ch_key in enumerate(ch_keys):
            slide_shm_arr[:, :, i] = all_channel_data[ch_key]
            logger.info(f"  Loaded channel {ch_key} to shared memory slot {i}")

        ch_to_slot = {ch_key: i for i, ch_key in enumerate(ch_keys)}

        # Free original channel data — everything is now in shared memory
        mem_freed_gb = sum(arr.nbytes for arr in all_channel_data.values()) / (1024**3)
        del all_channel_data
        loader.channel_data = None
        gc.collect()
        logger.info(f"Freed all_channel_data ({mem_freed_gb:.1f} GB) — using shared memory for all reads")

        # Build strategy parameters from the already-constructed params dict
        strategy_params = dict(params)

        # Get classifier path
        classifier_path = None
        if args.cell_type == 'nmj':
            classifier_path = getattr(args, 'nmj_classifier', None)
            if classifier_path:
                logger.info(f"Using specified NMJ classifier: {classifier_path}")
            else:
                logger.info("No --nmj-classifier specified — will return all candidates (annotation run)")
        elif args.cell_type == 'islet':
            classifier_path = getattr(args, 'islet_classifier', None)
            if classifier_path:
                logger.info(f"Using specified islet classifier: {classifier_path}")
            else:
                logger.info("No --islet-classifier specified — will return all candidates (annotation run)")
        elif args.cell_type == 'tissue_pattern':
            classifier_path = getattr(args, 'tp_classifier', None)
            if classifier_path:
                logger.info(f"Using specified tissue_pattern classifier: {classifier_path}")
            else:
                logger.info("No --tp-classifier specified — will return all candidates (annotation run)")

        extract_deep = getattr(args, 'extract_deep_features', False)

        # Vessel-specific params for multi-GPU
        mgpu_cd31_channel = getattr(args, 'cd31_channel', None) if args.cell_type == 'vessel' else None
        mgpu_channel_names = None
        if args.cell_type == 'vessel' and getattr(args, 'channel_names', None):
            names = args.channel_names.split(',')
            mgpu_channel_names = {ch_keys[i]: name.strip()
                                  for i, name in enumerate(names)
                                  if i < len(ch_keys)}

        # Add mosaic origin to slide_info so workers can convert global→relative coords
        mgpu_slide_info = shm_manager.get_slide_info()
        mgpu_slide_info[slide_name]['mosaic_origin'] = (x_start, y_start)

        # ---- Multi-scale vessel detection mode ----
        if is_multiscale:
            logger.info("=" * 60)
            logger.info(f"MULTI-SCALE VESSEL DETECTION — {num_gpus} GPU(s)")
            logger.info("=" * 60)

            from segmentation.utils.multiscale import (
                get_scale_params, generate_tile_grid_at_scale,
                convert_detection_to_full_res, merge_detections_across_scales,
            )
            from tqdm import tqdm as tqdm_progress

            scales = [int(s.strip()) for s in args.scales.split(',')]
            iou_threshold = getattr(args, 'multiscale_iou_threshold', 0.3)
            tile_size = args.tile_size

            logger.info(f"Scales: {scales} (coarse to fine)")
            logger.info(f"IoU threshold for deduplication: {iou_threshold}")

            # One MultiGPUTileProcessor for all scales (workers stay alive, models stay loaded)
            with MultiGPUTileProcessor(
                num_gpus=num_gpus,
                slide_info=mgpu_slide_info,
                cell_type='vessel',
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

                all_scale_detections = []  # Accumulate full-res detections across scales
                total_tiles_submitted = 0

                # Resume from checkpoints if available
                completed_scales = set()
                if getattr(args, 'resume_from', None):
                    checkpoint_dir = Path(args.resume_from) / "checkpoints"
                    if checkpoint_dir.exists():
                        # Sort by modification time (not lexicographic — scale_8x > scale_16x lex)
                        checkpoint_files = sorted(
                            checkpoint_dir.glob("scale_*x.json"),
                            key=lambda p: p.stat().st_mtime,
                        )
                        if checkpoint_files:
                            latest = checkpoint_files[-1]
                            with open(latest) as f:
                                all_scale_detections = json.load(f)
                            # Restore numpy arrays for contours (json.load produces lists)
                            for det in all_scale_detections:
                                for key in ('outer', 'inner', 'outer_contour', 'inner_contour'):
                                    if key in det and det[key] is not None:
                                        det[key] = np.array(det[key], dtype=np.int32)
                            for cf in checkpoint_files:
                                # Parse scale from filename like "scale_32x.json"
                                s = int(cf.stem.split('_')[1].rstrip('x'))
                                completed_scales.add(s)
                            logger.info(
                                f"Resumed from {latest}: {len(all_scale_detections)} detections, "
                                f"completed scales: {sorted(completed_scales)}"
                            )

                for scale in scales:
                    if scale in completed_scales:
                        logger.info(f"Scale 1/{scale}x: skipping (checkpointed)")
                        continue

                    scale_params = get_scale_params(scale)
                    scale_tiles = generate_tile_grid_at_scale(
                        mosaic_info['width'], mosaic_info['height'],
                        tile_size, scale, overlap=0,
                    )

                    # Sample tiles if requested
                    if args.sample_fraction < 1.0:
                        n_sample = max(1, int(len(scale_tiles) * args.sample_fraction))
                        indices = np.random.choice(len(scale_tiles), n_sample, replace=False)
                        scale_tiles = [scale_tiles[i] for i in indices]

                    logger.info(
                        f"Scale 1/{scale}x: {len(scale_tiles)} tiles, "
                        f"pixel_size={pixel_size_um * scale:.3f} µm, "
                        f"target: {scale_params.get('description', '')}"
                    )

                    # Submit tiles for this scale with scale metadata
                    for tx_s, ty_s in scale_tiles:
                        # Tile coords in full-res space (for shm extraction):
                        # worker subtracts mosaic_origin, then strides by sf
                        tile_with_dims = {
                            'x': x_start + tx_s * scale,
                            'y': y_start + ty_s * scale,
                            'w': tile_size * scale,
                            'h': tile_size * scale,
                            'scale_factor': scale,
                            'scale_params': scale_params,
                            'tile_x_scaled': tx_s,
                            'tile_y_scaled': ty_s,
                        }
                        processor.submit_tile(slide_name, tile_with_dims)

                    # Collect results for this scale
                    pbar = tqdm_progress(total=len(scale_tiles), desc=f"Scale 1/{scale}x")
                    scale_det_count = 0
                    results_collected = 0

                    while results_collected < len(scale_tiles):
                        result = processor.collect_result(timeout=3600)
                        if result is None:
                            logger.warning(f"Timeout at scale 1/{scale}x")
                            break

                        results_collected += 1
                        pbar.update(1)

                        if result['status'] == 'success':
                            tile_dict = result['tile']
                            features_list = result['features_list']
                            sf = result.get('scale_factor', scale)
                            tx_s = tile_dict.get('tile_x_scaled', 0)
                            ty_s = tile_dict.get('tile_y_scaled', 0)

                            # Apply vessel classifiers
                            apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                            # Convert each detection from downscaled-local to full-res global
                            for feat in features_list:
                                # FIX 1: Promote contour keys — detections_to_features_list
                                # outputs 'outer_contour'/'inner_contour' but
                                # convert_detection_to_full_res expects 'outer'/'inner'
                                if 'outer_contour' in feat and 'outer' not in feat:
                                    feat['outer'] = feat.pop('outer_contour')
                                if 'inner_contour' in feat and 'inner' not in feat:
                                    feat['inner'] = feat.pop('inner_contour')
                                if feat.get('features', {}).get('detection_type') == 'arc':
                                    feat['is_arc'] = True

                                det_fullres = convert_detection_to_full_res(
                                    feat, sf, tx_s, ty_s,
                                    smooth=True,
                                    smooth_base_factor=getattr(args, 'smooth_contours_factor', 3.0),
                                )

                                # FIX 2: Add mosaic origin — convert_detection_to_full_res
                                # produces mosaic-relative coords (0-indexed into shm).
                                # Add (x_start, y_start) for CZI-global coords matching
                                # the regular pipeline.
                                for key in ('center', 'centroid'):
                                    if key in det_fullres:
                                        det_fullres[key][0] += x_start
                                        det_fullres[key][1] += y_start
                                feats_d = det_fullres.get('features', {})
                                if isinstance(feats_d, dict):
                                    # features['center'] not scaled by convert_detection_to_full_res
                                    if 'center' in feats_d and feats_d['center'] is not None:
                                        fc = feats_d['center']
                                        feats_d['center'] = [
                                            (fc[0] + tx_s) * sf + x_start,
                                            (fc[1] + ty_s) * sf + y_start,
                                        ]
                                    # outer_center/inner_center already scaled, add mosaic origin
                                    for ck in ('outer_center', 'inner_center'):
                                        if ck in feats_d and feats_d[ck] is not None:
                                            feats_d[ck][0] += x_start
                                            feats_d[ck][1] += y_start
                                mosaic_offset = np.array([x_start, y_start], dtype=np.int32)
                                if 'outer' in det_fullres and det_fullres['outer'] is not None:
                                    det_fullres['outer'] = det_fullres['outer'] + mosaic_offset
                                if 'inner' in det_fullres and det_fullres['inner'] is not None:
                                    det_fullres['inner'] = det_fullres['inner'] + mosaic_offset

                                # FIX 3: Rebuild outer_contour/outer_contour_global from
                                # scaled+offset contours (worker created these in downscaled
                                # local space with tile_x=0, tile_y=0)
                                for ckey in ('outer', 'inner'):
                                    if ckey in det_fullres and det_fullres[ckey] is not None:
                                        det_fullres[f'{ckey}_contour'] = det_fullres[ckey]
                                        det_fullres[f'{ckey}_contour_global'] = [
                                            [int(pt[0][0]), int(pt[0][1])]
                                            for pt in det_fullres[ckey]
                                        ]

                                det_fullres['scale_detected'] = sf
                                all_scale_detections.append(det_fullres)
                                scale_det_count += 1

                        elif result['status'] == 'error':
                            logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

                    pbar.close()
                    total_tiles_submitted += results_collected
                    logger.info(f"Scale 1/{scale}x: {scale_det_count} detections")

                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()

                    # Save checkpoint after each scale
                    checkpoint_dir = slide_output_dir / "checkpoints"
                    checkpoint_dir.mkdir(exist_ok=True)
                    checkpoint_file = checkpoint_dir / f"scale_{scale}x.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(all_scale_detections, f, cls=NumpyEncoder)
                    logger.info(f"Checkpoint saved: {checkpoint_file} ({len(all_scale_detections)} detections)")

            # Merge across scales (contour-based IoU dedup)
            logger.info(f"Merging {len(all_scale_detections)} detections across scales...")
            merged_detections = merge_detections_across_scales(
                all_scale_detections, iou_threshold=iou_threshold,
            )
            logger.info(f"After merge: {len(merged_detections)} vessels")

            # Regenerate UIDs from full-res global coords and build all_detections
            for det in merged_detections:
                features_dict = det.get('features', {})
                center = features_dict.get('center', det.get('center', [0, 0]))
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    cx, cy = int(center[0]), int(center[1])
                else:
                    cx, cy = 0, 0

                uid = f"{slide_name}_vessel_{cx}_{cy}"
                det['uid'] = uid
                det['slide'] = slide_name
                det['center'] = [cx, cy]
                det['center_um'] = [cx * pixel_size_um, cy * pixel_size_um]
                det['global_center'] = [cx, cy]
                det['global_center_um'] = [cx * pixel_size_um, cy * pixel_size_um]
                all_detections.append(det)

            total_detections = len(all_detections)

            # Generate HTML crops from shared memory with percentile normalization
            logger.info(f"Generating HTML crops for {len(all_detections)} multiscale detections...")
            from segmentation.io.html_export import image_to_base64

            for det in all_detections:
                features_dict = det.get('features', {})
                cx, cy = det['center']

                diameter_um = features_dict.get('outer_diameter_um', 50)
                diameter_px = int(diameter_um / pixel_size_um)
                crop_size = max(300, min(800, int(diameter_px * 2)))
                half = crop_size // 2

                # Crop from shared memory (0-indexed, subtract mosaic origin)
                rel_cy = cy - y_start
                rel_cx = cx - x_start
                y1 = max(0, rel_cy - half)
                x1 = max(0, rel_cx - half)
                y2 = min(h, rel_cy + half)
                x2 = min(w, rel_cx + half)

                if n_channels >= 3:
                    crop_rgb = np.stack([
                        slide_shm_arr[y1:y2, x1:x2, i] for i in range(3)
                    ], axis=-1)
                else:
                    crop_rgb = np.stack([slide_shm_arr[y1:y2, x1:x2, 0]] * 3, axis=-1)

                if crop_rgb.size == 0:
                    continue

                # Percentile normalize (not /256) for proper dynamic range
                crop_rgb = percentile_normalize(crop_rgb, p_low=1, p_high=99.5)

                b64_str, _ = image_to_base64(crop_rgb)
                sample = {
                    'uid': det['uid'],
                    'image': b64_str,
                    'stats': features_dict,
                }
                all_samples.append(sample)

            logger.info(f"Multi-scale mode: {total_detections} detections, {len(all_samples)} HTML samples "
                        f"from {total_tiles_submitted} tiles on {num_gpus} GPUs")

        # ---- Regular (non-multiscale) tile processing ----
        else:
            logger.info("=" * 60)
            logger.info(f"{args.cell_type.upper()} DETECTION — {num_gpus} GPU(s)")
            logger.info("=" * 60)

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
                islet_display_channels=getattr(args, 'islet_display_chs', None),
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
                            # Read HTML crops from shared memory (all_channel_data freed after shm creation)
                            if args.cell_type == 'islet' and hasattr(args, 'islet_display_chs'):
                                # Islet: display channels from --islet-display-channels (R, G, B)
                                _islet_disp = args.islet_display_chs
                                _shm_dtype = slide_shm_arr.dtype
                                rgb_channels = []
                                for _ci in range(3):
                                    if _ci < len(_islet_disp) and _islet_disp[_ci] in ch_to_slot:
                                        rgb_channels.append(
                                            slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, ch_to_slot[_islet_disp[_ci]]]
                                        )
                                    else:
                                        rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_shm_dtype))
                                tile_rgb_html = np.stack(rgb_channels, axis=-1)
                            elif args.cell_type == 'tissue_pattern':
                                # Tissue pattern: configurable R/G/B display channels
                                # Handles any number of display channels (1, 2, or 3+), always produces (h, w, 3)
                                tp_disp = [int(x) for x in args.tp_display_channels.split(',')]
                                _shm_dtype = slide_shm_arr.dtype
                                rgb_channels = []
                                for _ci in range(3):
                                    if _ci < len(tp_disp) and tp_disp[_ci] in ch_to_slot:
                                        rgb_channels.append(
                                            slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, ch_to_slot[tp_disp[_ci]]]
                                        )
                                    else:
                                        rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_shm_dtype))
                                tile_rgb_html = np.stack(rgb_channels, axis=-1)
                            elif n_channels >= 3:
                                tile_rgb_html = np.stack([
                                    slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, i]
                                    for i in range(3)
                                ], axis=-1)
                            else:
                                tile_rgb_html = np.stack([
                                    slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, 0]
                                ] * 3, axis=-1)
                            # Convert uint16→uint8 for non-islet (NMJ/MK channels are high-signal, /256 works).
                            # For islet: keep uint16 — low-signal fluorescence (Gcg mean=16.7)
                            # would become all-zero after /256. percentile_normalize() handles uint16.
                            if tile_rgb_html.dtype == np.uint16 and args.cell_type not in ('islet', 'tissue_pattern'):
                                tile_rgb_html = (tile_rgb_html / 256).astype(np.uint8)

                            tile_pct = _compute_tile_percentiles(tile_rgb_html) if getattr(args, 'html_normalization', 'crop') == 'tile' else None

                            if args.cell_type == 'islet':
                                # Flush tile data to disk — keep only lightweight metadata in memory
                                # to avoid OOM from accumulating masks+tile_rgb across all tiles
                                np.save(tile_out / 'tile_rgb_html.npy', tile_rgb_html)
                                if tile_pct is not None:
                                    with open(tile_out / 'tile_pct.json', 'w') as f_pct:
                                        json.dump(tile_pct, f_pct)
                                deferred_html_tiles.append({
                                    'tile_dir': str(tile_out),
                                    'tile_x': tile_x, 'tile_y': tile_y,
                                    'tile_pct': tile_pct,
                                })
                                total_detections += len(features_list)
                                del masks, tile_rgb_html, features_list
                                result['masks'] = None  # Release array ref from result dict
                                result['features_list'] = None
                                gc.collect()
                            else:
                                _max_html = getattr(args, 'max_html_samples', 0)
                                if _max_html > 0 and len(all_samples) >= _max_html:
                                    pass  # Skip HTML crop generation — cap reached
                                else:
                                    html_samples = filter_and_create_html_samples(
                                        features_list, tile_x, tile_y, tile_rgb_html, masks,
                                        pixel_size_um, slide_name, args.cell_type,
                                        args.html_score_threshold,
                                        tile_percentiles=tile_pct,
                                        candidate_mode=args.candidate_mode,
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
                    _islet_mm = getattr(args, 'islet_marker_map', None)
                    marker_thresholds = compute_islet_marker_thresholds(
                        all_detections, marker_map=_islet_mm) if all_detections else None
                    # Add marker_class to each detection for JSON export
                    if marker_thresholds:
                        counts = {}
                        for det in all_detections:
                            mc, _ = classify_islet_marker(
                                det.get('features', {}), marker_thresholds, marker_map=_islet_mm)
                            det['marker_class'] = mc
                            counts[mc] = counts.get(mc, 0) + 1
                        logger.info(f"Islet marker classification: {counts}")
                    # Build UID→marker_class lookup for injecting into reloaded features
                    uid_to_marker = {d.get('uid', ''): d.get('marker_class') for d in all_detections}
                    for dt in deferred_html_tiles:
                        # Reload tile data from disk (one at a time to control memory)
                        _td = Path(dt['tile_dir'])
                        _tile_rgb = np.load(_td / 'tile_rgb_html.npy')
                        with h5py.File(_td / f'{args.cell_type}_masks.h5', 'r') as _hf:
                            _tile_masks = _hf['masks'][:]
                        with open(_td / f'{args.cell_type}_features.json', 'r') as _ff:
                            _tile_feats = json.load(_ff)
                        # Inject marker_class into reloaded features
                        for _feat in _tile_feats:
                            _mc = uid_to_marker.get(_feat.get('uid', ''))
                            if _mc:
                                _feat['marker_class'] = _mc
                        html_samples = filter_and_create_html_samples(
                            _tile_feats, dt['tile_x'], dt['tile_y'],
                            _tile_rgb, _tile_masks,
                            pixel_size_um, slide_name, args.cell_type,
                            args.html_score_threshold,
                            tile_percentiles=dt['tile_pct'],
                            marker_thresholds=marker_thresholds,
                            marker_map=_islet_mm,
                            candidate_mode=args.candidate_mode,
                        )
                        all_samples.extend(html_samples)
                        del _tile_rgb, _tile_masks, _tile_feats
                        gc.collect()
                    deferred_html_tiles = []  # clear metadata
                    gc.collect()  # free ~1.6 GB of retained tile data

            logger.info(f"Processing complete: {total_detections} {args.cell_type} detections from {results_collected} tiles")

    finally:
        # Cleanup shared memory
        shm_manager.cleanup()

    logger.info(f"Total detections (pre-dedup): {len(all_detections)}")

    # Deduplication: tile overlap causes same detection in adjacent tiles
    # Uses actual mask pixel overlap (loads HDF5 mask files) for accurate dedup
    # Skip for multiscale — already deduped by contour IoU in merge_detections_across_scales()
    if not is_multiscale and getattr(args, 'tile_overlap', 0) > 0 and len(all_detections) > 0:
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

    # Channel legend for multi-channel images — derived from CZI metadata when available
    channel_legend = None

    def _channel_label(ch_idx):
        """Get human-readable label for a channel index from CZI metadata."""
        try:
            for ch in _czi_meta['channels']:
                if ch['index'] == ch_idx:
                    name = ch['name']
                    em = f" ({ch['emission_nm']:.0f}nm)" if ch.get('emission_nm') else ''
                    return f'{name}{em}'
        except (NameError, TypeError, KeyError):
            pass
        return f'Ch{ch_idx}'

    if args.cell_type == 'nmj' and getattr(args, 'all_channels', False):
        # Try CZI metadata first, fall back to filename parsing
        try:
            channel_legend = {
                'red': _channel_label(0),
                'green': _channel_label(1),
                'blue': _channel_label(2),
            }
        except Exception:
            channel_legend = parse_channel_legend_from_filename(slide_name)
    elif args.cell_type == 'islet':
        _islet_disp = getattr(args, 'islet_display_chs', [2, 3, 5])
        channel_legend = {
            'red': _channel_label(_islet_disp[0]) if len(_islet_disp) > 0 else 'none',
            'green': _channel_label(_islet_disp[1]) if len(_islet_disp) > 1 else 'none',
            'blue': _channel_label(_islet_disp[2]) if len(_islet_disp) > 2 else 'none',
        }
    elif args.cell_type == 'tissue_pattern':
        tp_disp = [int(x) for x in args.tp_display_channels.split(',')]
        channel_legend = {
            'red': _channel_label(tp_disp[0]) if len(tp_disp) > 0 else 'none',
            'green': _channel_label(tp_disp[1]) if len(tp_disp) > 1 else 'none',
            'blue': _channel_label(tp_disp[2]) if len(tp_disp) > 2 else 'none',
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

    # Assign globally unique mask_labels encoding centroid in global coordinates
    # Format: x_y (integer pixel coords) — spatially meaningful + unique
    # Preserve original per-tile label as 'tile_mask_label' for HDF5 lookups
    for det in all_detections:
        det['tile_mask_label'] = det.get('mask_label', 0)
        gc = det.get('global_center', det.get('center', [0, 0]))
        det['mask_label'] = f"{int(round(gc[0]))}_{int(round(gc[1]))}"

    # Save all detections with universal IDs and global coordinates
    detections_file = slide_output_dir / f'{args.cell_type}_detections.json'
    with open(detections_file, 'w') as f:
        json.dump(all_detections, f, indent=2, cls=NumpyEncoder)
    logger.info(f"  Saved {len(all_detections)} detections to {detections_file}")

    # Clean up multiscale checkpoints after successful completion
    if is_multiscale:
        checkpoint_dir = slide_output_dir / "checkpoints"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info("Multiscale checkpoints cleaned up after successful completion")

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
                        help='[DEPRECATED] Lumen-first detection now runs automatically as a '
                             'supplementary pass alongside ring detection. This flag is a no-op. '
                             'Use --ring-only to disable the supplementary lumen-first pass.')
    parser.add_argument('--ring-only', action='store_true',
                        help='Disable the supplementary lumen-first detection pass. '
                             'Only use Canny edge + contour hierarchy ring detection. '
                             'Useful if you know there are no great vessels in the tissue.')
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
    parser.add_argument('--no-smooth-contours', action='store_true',
                        help='Disable B-spline contour smoothing (on by default). '
                             'Smoothing removes stair-step artifacts from coarse-scale detection.')
    parser.add_argument('--smooth-contours-factor', type=float, default=3.0,
                        help='Spline smoothing factor for vessel contours (default: 3.0). '
                             'Higher = smoother. 0 = interpolating spline (passes through all points).')
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
    parser.add_argument('--scales', type=str, default='32,16,8,4,2',
                        help='Comma-separated scale factors for multi-scale detection (default: "32,16,8,4,2"). '
                             'Numbers represent downsampling factors: 32=1/32x (large arteries), '
                             '16=1/16x, 8=1/8x, 4=1/4x (medium), 2=1/2x (small vessels). '
                             'Detection runs coarse-to-fine with IoU deduplication.')
    parser.add_argument('--multiscale-iou-threshold', type=float, default=0.3,
                        help='IoU threshold for deduplicating vessels detected at multiple scales '
                             '(default: 0.3). If a vessel is detected at both coarse and fine scales '
                             'with IoU > threshold, the detection with larger contour area is kept.')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume multiscale run from checkpoints in a previous run directory. '
                             'Skips already-completed scales and reuses the output directory.')

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
    parser.add_argument('--islet-display-channels', type=str, default='2,3,5',
                        help='Comma-separated R,G,B channel indices for islet HTML display (default: 2,3,5). '
                             'Channels are mapped to R/G/B in order.')
    parser.add_argument('--islet-marker-channels', type=str, default='gcg:2,ins:3,sst:5',
                        help='Marker-to-channel mapping for islet classification, as name:index pairs. '
                             'Format: "gcg:2,ins:3,sst:5". Names are used in logs and legends.')
    parser.add_argument('--islet-min-area', type=float, default=30.0,
                        help='Minimum islet cell area in um² (default 30)')
    parser.add_argument('--islet-max-area', type=float, default=500.0,
                        help='Maximum islet cell area in um² (default 500)')
    parser.add_argument('--marker-signal-factor', type=float, default=2.0,
                        help='Pre-filter divisor for Otsu threshold. Cells need marker '
                             'signal > auto_threshold/N to get full features + SAM2. '
                             'Higher = more permissive. 0 = disable. (default 2.0)')
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

    # Channel selection
    parser.add_argument('--channels', type=str, default=None,
                        help='Comma-separated list of CZI channel indices to load (e.g. "8,9,10,11"). '
                             'If not specified, all channels are loaded when --all-channels is active. '
                             'Use with multi-channel CZIs that have EDF/processing layers to avoid loading unnecessary data.')

    # Feature extraction options
    parser.add_argument('--extract-full-features', action='store_true',
                        help='Extract full features including SAM2 embeddings')
    parser.add_argument('--extract-deep-features', action='store_true',
                        help='Extract ResNet and DINOv2 features (opt-in, default morph+SAM2 only)')
    parser.add_argument('--skip-deep-features', action='store_true',
                        help='Deprecated: deep features are off by default now. Use --extract-deep-features to enable.')

    # GPU processing (always uses multi-GPU infrastructure, even with 1 GPU)
    parser.add_argument('--multi-gpu', action='store_true', default=True,
                        help='[DEPRECATED - always enabled] Multi-GPU processing is now the only code path. '
                             'Use --num-gpus to control how many GPUs are used (default: auto-detect).')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect via torch.cuda.device_count(), '
                             'minimum 1). The pipeline always uses the multi-GPU infrastructure, '
                             'even with --num-gpus 1.')

    # HTML export
    parser.add_argument('--samples-per-page', type=int, default=300)
    parser.add_argument('--max-html-samples', type=int, default=0,
                        help='Maximum HTML samples to keep in memory (0=unlimited). '
                             'For full runs with 500K+ cells, set to e.g. 5000 to avoid OOM from base64 crop accumulation.')

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
        # Parse --islet-display-channels into list of ints
        args.islet_display_chs = [int(x.strip()) for x in args.islet_display_channels.split(',')]
        # Parse --islet-marker-channels into dict: {name: channel_index}
        args.islet_marker_map = {}
        for pair in args.islet_marker_channels.split(','):
            name, ch = pair.strip().split(':')
            args.islet_marker_map[name.strip()] = int(ch.strip())

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

    # Auto-detect number of GPUs if not specified
    if args.num_gpus is None:
        try:
            args.num_gpus = max(1, torch.cuda.device_count())
        except Exception:
            args.num_gpus = 1

    # --multi-gpu is always True now (kept for backward compatibility)
    args.multi_gpu = True

    run_pipeline(args)


if __name__ == '__main__':
    main()
