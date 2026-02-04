"""
Multi-GPU NMJ detection with shared memory (zero-copy).

This module provides multi-GPU support for NMJ detection, distributing tiles
across multiple GPUs with each worker processing one tile at a time.

Architecture:
- Main process: loads slide channels into shared memory, sends tile coordinates to queue
- N GPU workers: each pinned to one GPU, reads tiles from shared memory, runs NMJStrategy

NMJ Pipeline uses:
- Intensity thresholding + morphological filtering (no learned models)
- SAM2 embeddings (256D) - required for morph+SAM2 classifier
- Optional RF classifier for filtering

Note: Unlike MK/HSPC pipeline, NMJ does NOT use Cellpose, ResNet, or DINOv2 by default.
The morph+SAM2 classifier achieves 95% precision with only 334 features.

Coordinate System:
    Workers receive tiles with RELATIVE coordinates (array indices), NOT global CZI
    coordinates. Tile dict format:
        tile = {
            'x': 0,      # Relative X - direct array column index (used as tile origin for UID)
            'y': 0,      # Relative Y - direct array row index (used as tile origin for UID)
            'w': 3000,   # Tile width
            'h': 3000,   # Tile height
        }

    Global coordinates for UIDs are computed as:
        global_cx = tile['x'] + local_cx
        global_cy = tile['y'] + local_cy
        uid = f"{slide_name}_nmj_{global_cx}_{global_cy}"
"""

import os
import gc
import json
import queue
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np

# Use 'spawn' start method for CUDA compatibility in subprocesses
# 'fork' (Linux default) copies CUDA state and causes hangs
_mp_context = mp.get_context('spawn')
Process = _mp_context.Process
Queue = _mp_context.Queue
Event = _mp_context.Event

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def _nmj_gpu_worker(
    gpu_id: int,
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    slide_info: Dict[str, Dict[str, Any]],
    strategy_params: Dict[str, Any],
    classifier_path: Optional[str],
    pixel_size_um: float,
    extract_sam2_embeddings: bool = True,
    sam2_checkpoint: Optional[str] = None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
):
    """
    Worker process for NMJ detection on a single GPU.

    Each worker:
    1. Pins to assigned GPU via CUDA_VISIBLE_DEVICES
    2. Loads SAM2 predictor (for embeddings)
    3. Loads RF classifier if path provided
    4. Creates NMJStrategy with parameters
    5. Processes tiles from shared memory via queue

    Args:
        gpu_id: Which GPU this worker owns (0, 1, 2, ...)
        input_queue: Queue of (slide_name, tile_dict) tuples
        output_queue: Queue for results
        stop_event: Event to signal shutdown
        slide_info: Dict mapping slide_name -> {shm_name, shape, dtype}
        strategy_params: Dict of NMJ strategy parameters
        classifier_path: Path to RF classifier (.pkl or .joblib), or None
        pixel_size_um: Pixel size for area calculations
        extract_sam2_embeddings: Whether to extract SAM2 embeddings (default True)
        sam2_checkpoint: Optional path to SAM2 checkpoint
        sam2_config: SAM2 config file path
    """
    # Pin to specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from segmentation.detection.strategies.nmj import NMJStrategy, load_nmj_rf_classifier
    from segmentation.detection.tissue import has_tissue

    worker_name = f"GPU-{gpu_id}"
    logger.info(f"[{worker_name}] Starting NMJ worker, attaching to shared memory...")

    # Attach to shared memory for all slides
    shared_slides: Dict[str, Tuple[SharedMemory, np.ndarray]] = {}
    try:
        for slide_name, info in slide_info.items():
            shm = SharedMemory(name=info['shm_name'])
            arr = np.ndarray(
                info['shape'],
                dtype=np.dtype(info['dtype']),
                buffer=shm.buf
            )
            shared_slides[slide_name] = (shm, arr)
            logger.debug(f"[{worker_name}] Attached to {slide_name} shared memory")
    except Exception as e:
        logger.error(f"[{worker_name}] Failed to attach to shared memory: {e}")
        output_queue.put({'status': 'init_error', 'gpu_id': gpu_id, 'error': str(e)})
        return

    # Initialize device
    device = torch.device('cuda:0')
    logger.info(f"[{worker_name}] Using device: {device}")

    # Build models dict for NMJStrategy
    models: Dict[str, Any] = {'device': device}

    # Load SAM2 predictor for embeddings
    sam2_predictor = None
    if extract_sam2_embeddings:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2

            # Find SAM2 checkpoint
            checkpoint_path = None
            if sam2_checkpoint and Path(sam2_checkpoint).exists():
                checkpoint_path = Path(sam2_checkpoint)
            else:
                # Auto-detect from common locations
                script_dir = Path(__file__).parent.parent.parent.resolve()
                checkpoint_candidates = [
                    script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
                    script_dir / "checkpoints" / "sam2.1_hiera_l.pt",
                    Path("/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt"),
                ]
                for cp in checkpoint_candidates:
                    if cp.exists():
                        checkpoint_path = cp
                        break

            if checkpoint_path:
                logger.info(f"[{worker_name}] Loading SAM2 from {checkpoint_path}...")
                sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=device)
                sam2_predictor = SAM2ImagePredictor(sam2_model)
                models['sam2_predictor'] = sam2_predictor
                logger.info(f"[{worker_name}] SAM2 loaded successfully")
            else:
                logger.warning(f"[{worker_name}] SAM2 checkpoint not found, embeddings will be zeros")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load SAM2: {e}")
            logger.warning(f"[{worker_name}] SAM2 embeddings will be zeros")

    # Load RF classifier if provided
    classifier = None
    feature_names = []
    if classifier_path and Path(classifier_path).exists():
        try:
            classifier_data = load_nmj_rf_classifier(classifier_path)
            classifier = classifier_data['pipeline']
            feature_names = classifier_data['feature_names']
            models['classifier'] = classifier
            models['classifier_type'] = 'rf'
            models['scaler'] = None  # Pipeline handles scaling
            models['feature_names'] = feature_names
            logger.info(f"[{worker_name}] RF classifier loaded ({len(feature_names)} features)")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load classifier: {e}")
            logger.warning(f"[{worker_name}] Will return all candidates without filtering")

    # Create NMJStrategy with parameters
    strategy = NMJStrategy(
        intensity_percentile=strategy_params.get('intensity_percentile', 99.0),
        max_solidity=strategy_params.get('max_solidity', 0.85),
        min_skeleton_length=strategy_params.get('min_skeleton_length', 30),
        min_area_px=strategy_params.get('min_area', 150),
        max_area_px=strategy_params.get('max_area_px'),
        min_area_um=strategy_params.get('min_area_um', 25.0),
        classifier_threshold=strategy_params.get('classifier_threshold', 0.75),
        use_classifier=classifier is not None,
        extract_resnet_features=False,  # NMJ multi-GPU uses morph+SAM2 only
        extract_sam2_embeddings=extract_sam2_embeddings,
    )

    logger.info(f"[{worker_name}] NMJStrategy created: {strategy.get_config()}")

    # Signal ready
    output_queue.put({'status': 'ready', 'gpu_id': gpu_id})

    tiles_processed = 0
    detections_found = 0

    while not stop_event.is_set():
        try:
            try:
                work_item = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if work_item is None:
                logger.info(f"[{worker_name}] Received shutdown signal")
                break

            slide_name, tile = work_item
            tid = tile.get('id', f"{tile['x']}_{tile['y']}")

            # Read tile directly from shared memory
            if slide_name not in shared_slides:
                logger.error(f"[{worker_name}] Unknown slide: {slide_name}")
                output_queue.put({
                    'status': 'error',
                    'tid': tid,
                    'slide_name': slide_name,
                    'tile': tile,
                    'error': f'Unknown slide: {slide_name}'
                })
                tiles_processed += 1
                continue

            _, slide_arr = shared_slides[slide_name]

            # Extract tile using RELATIVE coordinates (direct array indices)
            # slide_arr is 3D: (H, W, C) for multi-channel or (H, W) for single channel
            y_start = tile['y']
            x_start = tile['x']
            tile_h = tile['h']
            tile_w = tile['w']

            # Handle both single-channel and multi-channel shared memory
            if slide_arr.ndim == 3:
                # Multi-channel: (H, W, C) - expected for NMJ with --all-channels
                y_end = min(y_start + tile_h, slide_arr.shape[0])
                x_end = min(x_start + tile_w, slide_arr.shape[1])
                tile_rgb = slide_arr[y_start:y_end, x_start:x_end, :].copy()
            else:
                # Single channel: (H, W) - should not happen with --all-channels requirement
                y_end = min(y_start + tile_h, slide_arr.shape[0])
                x_end = min(x_start + tile_w, slide_arr.shape[1])
                tile_gray = slide_arr[y_start:y_end, x_start:x_end].copy()
                tile_rgb = np.stack([tile_gray] * 3, axis=-1)

            # Validate tile
            if tile_rgb.size == 0 or tile_rgb.max() == 0:
                output_queue.put({
                    'status': 'empty',
                    'tid': tid,
                    'slide_name': slide_name,
                    'tile': tile,
                })
                tiles_processed += 1
                continue

            # IMPORTANT: Keep ORIGINAL uint16 data for extra_channel_tiles (intensity features)
            # The classifier was trained on uint16 intensity values (0-65535).
            # Converting to uint8 would make intensity features ~256x smaller and break classification.
            # Create extra_channels dict from ORIGINAL data BEFORE any conversion
            # Channel mapping: ch0=nuclear(R), ch1=BTX(G), ch2=NFL(B)
            extra_channel_tiles = {
                0: tile_rgb[:, :, 0].copy(),  # Nuclear (488nm) - ORIGINAL uint16
                1: tile_rgb[:, :, 1].copy(),  # BTX (647nm) - used for segmentation, ORIGINAL uint16
                2: tile_rgb[:, :, 2].copy() if tile_rgb.shape[2] > 2 else tile_rgb[:, :, 1].copy(),  # NFL (750nm)
            }

            # Now convert to uint8 for SAM2/visual processing (SAM2 expects uint8 RGB)
            if tile_rgb.dtype != np.uint8:
                if tile_rgb.dtype == np.uint16:
                    tile_rgb = (tile_rgb / 256).astype(np.uint8)
                else:
                    tile_rgb = tile_rgb.astype(np.uint8)

            # Check for tissue (optional - can skip for speed)
            try:
                has_tissue_flag, _ = has_tissue(tile_rgb, threshold=100.0, block_size=64)
            except Exception:
                has_tissue_flag = True  # Assume tissue on error

            if not has_tissue_flag:
                output_queue.put({
                    'status': 'no_tissue',
                    'tid': tid,
                    'slide_name': slide_name,
                    'tile': tile,
                })
                tiles_processed += 1
                continue

            # Run NMJ detection
            try:
                masks, detections = strategy.detect(
                    tile_rgb,
                    models,
                    pixel_size_um,
                    extract_full_features=True,
                    extra_channels=extra_channel_tiles
                )

                # Build features list with global UIDs
                features_list = []
                for det in detections:
                    # Get local centroid
                    local_cx, local_cy = det.centroid  # [x, y]

                    # Compute global coordinates
                    global_cx = tile['x'] + local_cx
                    global_cy = tile['y'] + local_cy

                    # Create UID
                    uid = f"{slide_name}_nmj_{round(global_cx)}_{round(global_cy)}"

                    # Build feature dict
                    feat = {
                        'uid': uid,
                        'slide_name': slide_name,
                        'center': [float(local_cx), float(local_cy)],
                        'global_center': [float(global_cx), float(global_cy)],
                        'global_center_um': [float(global_cx * pixel_size_um), float(global_cy * pixel_size_um)],
                        'tile_origin': [tile['x'], tile['y']],
                        'features': det.features,
                        'score': det.score,
                    }
                    features_list.append(feat)

                detections_found += len(features_list)

                output_queue.put({
                    'status': 'success',
                    'tid': tid,
                    'slide_name': slide_name,
                    'tile': tile,
                    'masks': masks,
                    'features_list': features_list,
                })

            except Exception as e:
                logger.error(f"[{worker_name}] Error processing tile {tid}: {e}")
                logger.error(traceback.format_exc())
                output_queue.put({
                    'status': 'error',
                    'tid': tid,
                    'slide_name': slide_name,
                    'tile': tile,
                    'error': str(e),
                })

            tiles_processed += 1

            # Cleanup after every tile
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[{worker_name}] Worker loop error: {e}")
            logger.error(traceback.format_exc())

    # Cleanup
    logger.info(f"[{worker_name}] Shutting down, processed {tiles_processed} tiles, found {detections_found} NMJs")
    gc.collect()
    torch.cuda.empty_cache()

    # Close shared memory attachments (but don't unlink - main process does that)
    for shm, _ in shared_slides.values():
        shm.close()


class MultiGPUNMJProcessor:
    """
    Multi-GPU NMJ processor using shared memory.

    Usage:
        from segmentation.processing.multigpu_shm import SharedSlideManager
        from segmentation.processing.multigpu_nmj import MultiGPUNMJProcessor

        manager = SharedSlideManager()
        # Load slide data into shared memory
        slide_arr = manager.create_slide_buffer('slide1', shape, dtype)
        # ... load data into slide_arr ...

        with MultiGPUNMJProcessor(
            num_gpus=4,
            slide_info=manager.get_slide_info(),
            strategy_params={'intensity_percentile': 99.0},
            pixel_size_um=0.1725,
        ) as processor:
            for tile in tiles:
                processor.submit_tile('slide1', tile)

            for _ in range(len(tiles)):
                result = processor.collect_result()
                process(result)

        manager.cleanup()
    """

    def __init__(
        self,
        num_gpus: int,
        slide_info: Dict[str, Dict[str, Any]],
        strategy_params: Dict[str, Any],
        pixel_size_um: float,
        classifier_path: Optional[str] = None,
        extract_sam2_embeddings: bool = True,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    ):
        """
        Initialize multi-GPU NMJ processor.

        Args:
            num_gpus: Number of GPUs to use
            slide_info: Dict from SharedSlideManager.get_slide_info()
            strategy_params: Parameters for NMJStrategy
            pixel_size_um: Pixel size for area calculations
            classifier_path: Path to RF classifier (.pkl or .joblib)
            extract_sam2_embeddings: Whether to extract SAM2 embeddings
            sam2_checkpoint: Optional path to SAM2 checkpoint
            sam2_config: SAM2 config file path
        """
        self.num_gpus = num_gpus
        self.slide_info = slide_info
        self.strategy_params = strategy_params
        self.pixel_size_um = pixel_size_um
        self.classifier_path = classifier_path
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config

        self.workers: List[Process] = []
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.stop_event: Optional[Event] = None
        self.tiles_submitted = 0
        self._local_checkpoint_path: Optional[Path] = None  # For cleanup

    def _copy_checkpoint_to_local(self) -> str:
        """Copy SAM2 checkpoint to local /tmp for faster loading.

        Network storage I/O is slow when multiple workers load simultaneously.
        Copying to local /tmp (SSD) first allows fast parallel loading.

        Returns:
            Path to local checkpoint copy
        """
        # Find the checkpoint
        if self.sam2_checkpoint:
            checkpoint_path = Path(self.sam2_checkpoint)
        else:
            # Default location
            checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "sam2.1_hiera_large.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        # Create unique local path to avoid conflicts
        local_dir = Path("/tmp") / f"sam2_cache_{os.getpid()}"
        local_dir.mkdir(exist_ok=True)
        local_path = local_dir / checkpoint_path.name

        if local_path.exists():
            logger.info(f"SAM2 checkpoint already cached at {local_path}")
            self._local_checkpoint_path = local_path
            return str(local_path)

        # Copy checkpoint to local storage
        logger.info(f"Copying SAM2 checkpoint to local /tmp for faster loading...")
        logger.info(f"  Source: {checkpoint_path} ({checkpoint_path.stat().st_size / 1e9:.2f} GB)")
        start = time.time()
        shutil.copy2(checkpoint_path, local_path)
        elapsed = time.time() - start
        logger.info(f"  Copied to {local_path} in {elapsed:.1f}s")

        self._local_checkpoint_path = local_path
        return str(local_path)

    def _cleanup_local_checkpoint(self):
        """Remove local checkpoint copy."""
        if self._local_checkpoint_path and self._local_checkpoint_path.exists():
            try:
                # Remove the directory and its contents
                local_dir = self._local_checkpoint_path.parent
                shutil.rmtree(local_dir)
                logger.info(f"Cleaned up local checkpoint cache: {local_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup local checkpoint: {e}")

    def start(self):
        """Start worker processes with staggered initialization.

        Workers are started one at a time, waiting for each to finish SAM2 loading
        before starting the next. This avoids GPU/CUDA contention when multiple
        workers try to initialize simultaneously.
        """
        logger.info(f"Starting {self.num_gpus} GPU workers for NMJ detection...")

        # Copy checkpoint to local /tmp for faster parallel loading
        local_checkpoint = self._copy_checkpoint_to_local()

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.stop_event = Event()

        # Start workers one at a time and wait for each to be ready
        # This avoids CUDA/GPU contention during SAM2 model initialization
        ready_count = 0
        errors = []
        timeout_per_worker = 120  # 2 min timeout per worker (local SSD is fast)

        for gpu_id in range(self.num_gpus):
            p = Process(
                target=_nmj_gpu_worker,
                args=(
                    gpu_id,
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.slide_info,
                    self.strategy_params,
                    self.classifier_path,
                    self.pixel_size_um,
                    self.extract_sam2_embeddings,
                    local_checkpoint,  # Use local /tmp copy for fast loading
                    self.sam2_config,
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Started NMJ worker for GPU {gpu_id} (PID: {p.pid}), waiting for ready...")

            # Wait for THIS worker to be ready before starting the next
            # This ensures sequential SAM2 initialization (no contention)
            try:
                msg = self.output_queue.get(timeout=timeout_per_worker)
                if msg.get('status') == 'ready':
                    ready_count += 1
                    logger.info(f"NMJ worker GPU-{msg['gpu_id']} ready ({ready_count}/{self.num_gpus})")
                elif msg.get('status') == 'init_error':
                    errors.append(f"GPU-{msg['gpu_id']}: {msg['error']}")
                    logger.error(f"Worker GPU-{gpu_id} failed to initialize: {msg['error']}")
            except queue.Empty:
                logger.error(f"Timeout waiting for GPU-{gpu_id} to initialize")
                errors.append(f"GPU-{gpu_id}: timeout during initialization")

        if errors:
            logger.error(f"Worker initialization errors: {errors}")
            self.stop()
            raise RuntimeError(f"Failed to initialize NMJ workers: {errors}")

        # Verify all workers are ready before proceeding
        if ready_count < self.num_gpus:
            self.stop()
            raise RuntimeError(f"Only {ready_count}/{self.num_gpus} workers initialized - aborting")

        logger.info(f"All {self.num_gpus} NMJ workers ready")

    def submit_tile(self, slide_name: str, tile: Dict[str, Any]):
        """Submit a tile for processing."""
        self.input_queue.put((slide_name, tile))
        self.tiles_submitted += 1

    def collect_result(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Collect one tile result from workers.

        Skips non-tile messages (like stray 'ready' messages from race conditions).
        """
        start = time.time()
        remaining = timeout if timeout else float('inf')

        while remaining > 0:
            try:
                result = self.output_queue.get(timeout=min(remaining, 1.0))
                # Filter out non-tile results (like stray 'ready' messages)
                if result.get('status') in ('success', 'error', 'empty', 'no_tissue'):
                    return result
                # Skip other message types (ready, init_error)
                logger.debug(f"Skipping non-tile message: {result.get('status')}")
            except queue.Empty:
                if timeout:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return None
                else:
                    return None
        return None

    def stop(self):
        """Stop all workers."""
        # Send shutdown signals
        if self.input_queue:
            for _ in range(self.num_gpus):
                try:
                    self.input_queue.put(None, timeout=1.0)
                except (queue.Full, Exception):
                    pass

        # Set stop event as fallback
        if self.stop_event:
            self.stop_event.set()

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"NMJ worker {p.pid} did not stop gracefully, terminating")
                p.terminate()

        self.workers.clear()
        logger.info("All NMJ workers stopped")

        # Cleanup local checkpoint copy
        self._cleanup_local_checkpoint()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
