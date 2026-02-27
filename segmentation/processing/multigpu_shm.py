"""
Multi-GPU tile processing with shared memory (zero-copy).

This version uses shared memory to avoid serializing tile data through queues.
Slides are loaded into shared memory once, and workers read tiles directly.

Architecture:
- Main process: loads slides into shared memory, sends tile coordinates to queue
- N GPU workers: each pinned to one GPU, reads tiles from shared memory, processes, returns results

Performance improvement: ~27MB per tile no longer serialized, only ~100 bytes of coordinates.

Coordinate System:
    Workers receive tiles with RELATIVE coordinates (array indices), NOT global CZI
    coordinates. This is different from the `CZILoader.get_tile()` API which expects
    global coordinates.

    Tile dict format:
        tile = {
            'x': 0,      # Relative X - direct array column index
            'y': 0,      # Relative Y - direct array row index
            'w': 3000,   # Tile width
            'h': 3000,   # Tile height
        }

    Worker tile extraction:
        # Direct array slicing with relative coords (no conversion needed)
        tile_img = slide_arr[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]

    Why relative coords for workers?
        - Shared memory contains the loaded mosaic data starting at array index (0, 0)
        - Direct array slicing is faster than going through get_tile() conversion
        - Tiles are created with relative coords specifically for this use case

    See docs/COORDINATE_SYSTEM.md for the complete specification.
"""

import atexit
import os
import gc
import queue
import signal
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import numpy as np

from segmentation.utils.logging import get_logger
from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization, extract_slide_norm_params

logger = get_logger(__name__)

# Global registry of shared memory names for cleanup on crash
_shm_registry: Set[str] = set()

def _cleanup_shared_memory_on_exit():
    """Emergency cleanup of shared memory on process exit.

    This ensures shared memory is released even if the main process crashes
    or is killed. Without this, shared memory persists until system reboot.
    """
    for shm_name in list(_shm_registry):
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
            logger.info(f"Emergency cleanup: unlinked shared memory {shm_name}")
        except FileNotFoundError:
            pass  # Already cleaned up
        except Exception as e:
            logger.warning(f"Failed to cleanup shared memory {shm_name}: {e}")
    _shm_registry.clear()

# Register cleanup on normal exit
atexit.register(_cleanup_shared_memory_on_exit)

# Register cleanup on SIGTERM (e.g., SLURM job cancellation, timeout)
def _signal_cleanup(signum, frame):
    """Clean up shared memory on SIGTERM/SIGINT before exiting."""
    _cleanup_shared_memory_on_exit()
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, _signal_cleanup)


class SharedSlideManager:
    """Manages shared memory for slide data."""

    def __init__(self):
        self.shared_memories: Dict[str, SharedMemory] = {}
        self.slide_info: Dict[str, Dict[str, Any]] = {}

    def add_slide(self, name: str, data: np.ndarray) -> Dict[str, Any]:
        """
        Copy slide data into shared memory.

        Args:
            name: Slide name
            data: Numpy array with slide data

        Returns:
            Dict with shared memory info (shm_name, shape, dtype)
        """
        # Create shared memory
        shm = SharedMemory(create=True, size=data.nbytes)

        # Register for emergency cleanup (in case of crash)
        _shm_registry.add(shm.name)

        # Create numpy array backed by shared memory and copy data
        shared_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared_arr[:] = data[:]

        # Store references
        self.shared_memories[name] = shm
        info = {
            'shm_name': shm.name,
            'shape': data.shape,
            'dtype': str(data.dtype),
        }
        self.slide_info[name] = info

        logger.debug(f"Created shared memory for {name}: {data.nbytes / 1e9:.2f} GB")
        return info

    def create_slide_buffer(self, name: str, shape: tuple, dtype) -> np.ndarray:
        """Create shared memory and return numpy array backed by it for direct loading.

        Args:
            name: Slide name
            shape: Shape of the array to create
            dtype: Data type of the array

        Returns:
            numpy array backed by shared memory (caller can load data directly into it)
        """
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        shm = SharedMemory(create=True, size=size)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # Register for emergency cleanup (in case of crash)
        _shm_registry.add(shm.name)

        self.shared_memories[name] = shm
        self.slide_info[name] = {'shm_name': shm.name, 'shape': shape, 'dtype': str(dtype)}

        logger.info(f"Created shared memory for {name}: {size/1e9:.2f} GB")
        return arr

    def get_slide_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info dict for all slides (to pass to workers)."""
        return self.slide_info.copy()

    def cleanup_slide(self, name: str):
        """Release shared memory for a single slide."""
        if name in self.shared_memories:
            try:
                shm = self.shared_memories[name]
                # Remove from emergency cleanup registry
                _shm_registry.discard(shm.name)
                shm.close()
                shm.unlink()
                logger.debug(f"Released shared memory for {name}")
            except Exception as e:
                logger.warning(f"Error releasing shared memory for {name}: {e}")
            del self.shared_memories[name]
        if name in self.slide_info:
            del self.slide_info[name]

    def cleanup(self):
        """Release all shared memory."""
        for name, shm in self.shared_memories.items():
            try:
                # Remove from emergency cleanup registry
                _shm_registry.discard(shm.name)
                shm.close()
                shm.unlink()
                logger.debug(f"Released shared memory for {name}")
            except Exception as e:
                logger.warning(f"Error releasing shared memory for {name}: {e}")
        self.shared_memories.clear()
        self.slide_info.clear()


def _gpu_worker_shm(
    gpu_id: int,
    input_queue,
    output_queue,
    stop_event,
    slide_info: Dict[str, Dict[str, Any]],
    segmenter_kwargs: Dict[str, Any],
    mk_min_area: int,
    mk_max_area: int,
    hspc_min_area: Optional[int],
    hspc_max_area: Optional[int],
    variance_threshold: float,
    calibration_block_size: int,
    cleanup_config: Optional[Dict[str, Any]] = None,
    norm_params: Optional[Dict[str, float]] = None,
    normalization_method: str = 'none',
    intensity_threshold: float = 220.0,
    modality: Optional[str] = None,
    per_slide_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    skip_hspc: bool = False,
):
    """
    Worker process that reads tiles from shared memory.

    Args:
        gpu_id: Which GPU this worker owns
        input_queue: Queue of (slide_name, tile_dict) tuples (NO tile data!)
        output_queue: Queue for results
        stop_event: Event to signal shutdown
        slide_info: Dict mapping slide_name -> {shm_name, shape, dtype}
        segmenter_kwargs: kwargs for UnifiedSegmenter
        mk_min_area: Minimum MK area in pixels
        mk_max_area: Maximum MK area in pixels
        hspc_min_area: Minimum HSPC area in pixels (None = no filter)
        hspc_max_area: Maximum HSPC area in pixels (None = no filter)
        variance_threshold: Threshold for tissue detection
        calibration_block_size: Block size for variance calculation
        cleanup_config: Dict with cleanup options (cleanup_masks, fill_holes, pixel_size_um)
        intensity_threshold: Max background intensity for tissue detection (Otsu-derived)
        modality: 'brightfield' for H&E, None for fluorescence (OR logic)
        per_slide_thresholds: Dict mapping slide_name -> {variance_threshold, intensity_threshold}
            from step 1. If present, overrides scalar thresholds for per-slide tissue detection.
        skip_hspc: If True, skip HSPC detection entirely (MK only mode)
    """
    # Default cleanup config if not provided
    if cleanup_config is None:
        cleanup_config = {'cleanup_masks': False, 'fill_holes': True, 'pixel_size_um': 0.1725}
    # Pin to specific GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from run_unified_FAST import UnifiedSegmenter, process_detection_with_cleanup
    from segmentation.detection.tissue import has_tissue
    from segmentation.processing.mk_hspc_utils import (
        ensure_rgb_array,
        check_tile_validity,
        prepare_tile_for_detection,
        build_mk_hspc_result,
    )

    worker_name = f"GPU-{gpu_id}"
    logger.info(f"[{worker_name}] Starting worker, attaching to shared memory...")

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

    # Initialize segmenter
    try:
        device = torch.device('cuda:0')
        segmenter = UnifiedSegmenter(device=device, **segmenter_kwargs)
        logger.info(f"[{worker_name}] Models loaded on GPU {gpu_id}")
    except Exception as e:
        logger.error(f"[{worker_name}] Failed to initialize: {e}")
        output_queue.put({'status': 'init_error', 'gpu_id': gpu_id, 'error': str(e)})
        # Cleanup shared memory attachments
        for shm, _ in shared_slides.values():
            shm.close()
        return

    output_queue.put({'status': 'ready', 'gpu_id': gpu_id})

    tiles_processed = 0

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

            # Read tile directly from shared memory (ZERO COPY!)
            if slide_name not in shared_slides:
                logger.error(f"[{worker_name}] Unknown slide: {slide_name}")
                output_queue.put(build_mk_hspc_result(
                    tid=tid,
                    status='error',
                    tile=tile,
                    slide_name=slide_name,
                    error=f'Unknown slide: {slide_name}'
                ))
                tiles_processed += 1
                continue

            _, slide_arr = shared_slides[slide_name]

            # Extract tile using RELATIVE coordinates (direct array indices).
            # Unlike CZILoader.get_tile() which expects global CZI coords, shared
            # memory workers use tile['x'], tile['y'] directly as array indices.
            # See module docstring for coordinate system details.
            y_start = tile['y']
            x_start = tile['x']
            y_end = min(tile['y'] + tile['h'], slide_arr.shape[0])
            x_end = min(tile['x'] + tile['w'], slide_arr.shape[1])

            if y_start >= slide_arr.shape[0] or x_start >= slide_arr.shape[1]:
                logger.warning(f"[{worker_name}] Tile {tid} out of bounds, skipping")
                output_queue.put(build_mk_hspc_result(
                    tid=tid,
                    status='invalid',
                    tile=tile,
                    slide_name=slide_name,
                    error='Tile coordinates out of bounds'
                ))
                tiles_processed += 1
                continue

            tile_img = slide_arr[y_start:y_end, x_start:x_end].copy()

            # Validate tile
            is_valid, reason = check_tile_validity(tile_img, tid)
            if not is_valid:
                output_queue.put(build_mk_hspc_result(
                    tid=tid,
                    status='empty' if reason == 'empty' else 'invalid',
                    tile=tile,
                    slide_name=slide_name
                ))
                tiles_processed += 1
                continue

            # Prepare tile
            tile_rgb = ensure_rgb_array(tile_img)

            # Apply Reinhard normalization if requested (tile-by-tile to avoid OOM)
            if normalization_method == 'reinhard' and norm_params is not None:
                _pst = per_slide_thresholds.get(slide_name) if per_slide_thresholds and slide_name else None
                _otsu, _slab = extract_slide_norm_params(_pst)
                tile_rgb = apply_reinhard_normalization(
                    tile_rgb,
                    norm_params,
                    otsu_threshold=_otsu,
                    slide_lab_stats=_slab,
                )

            # Prepare for detection (percentile normalization, disabled for Reinhard)
            use_percentile_norm = (normalization_method != 'reinhard')
            tile_normalized = prepare_tile_for_detection(tile_rgb, normalize=use_percentile_norm)

            # Check for tissue (always run — Reinhard normalizes but does not filter out background)
            # Use per-slide thresholds if available
            slide_vt = variance_threshold
            slide_it = intensity_threshold
            if per_slide_thresholds and slide_name in per_slide_thresholds:
                slide_vt = 0.0  # unused for brightfield
                slide_it = per_slide_thresholds[slide_name].get(
                    'otsu_threshold',
                    per_slide_thresholds[slide_name].get('intensity_threshold', 220.0))

            try:
                has_tissue_flag, _ = has_tissue(tile_normalized, slide_vt, block_size=calibration_block_size, intensity_threshold=slide_it, modality=modality)
            except Exception:
                has_tissue_flag = True  # Assume tissue on error

            if not has_tissue_flag:
                output_queue.put(build_mk_hspc_result(
                    tid=tid,
                    status='no_tissue',
                    tile=tile,
                    slide_name=slide_name
                ))
                tiles_processed += 1
                continue

            # Process tile
            try:
                # process_tile returns 4-tuple: (mk_masks, hspc_masks, mk_features, hspc_features)
                mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
                    tile_normalized,
                    mk_min_area=mk_min_area,
                    mk_max_area=mk_max_area,
                    hspc_min_area=hspc_min_area,
                    hspc_max_area=hspc_max_area,
                    hspc_nuclear_only=cleanup_config.get('hspc_nuclear_only', False),
                    skip_hspc=skip_hspc,
                )

                # Generate crops for each detection (with optional cleanup, same as non-SHM modes)
                for feat in mk_feats:
                    _, crop_result = process_detection_with_cleanup(
                        feat, mk_masks, tile_normalized, 'mk',
                        cleanup_masks=cleanup_config['cleanup_masks'],
                        fill_holes=cleanup_config['fill_holes'],
                        pixel_size_um=cleanup_config['pixel_size_um'],
                    )
                    if crop_result:
                        feat['crop_b64'] = crop_result['crop']
                        feat['mask_b64'] = crop_result['mask']

                for feat in hspc_feats:
                    _, crop_result = process_detection_with_cleanup(
                        feat, hspc_masks, tile_normalized, 'hspc',
                        cleanup_masks=cleanup_config['cleanup_masks'],
                        fill_holes=cleanup_config['fill_holes'],
                        pixel_size_um=cleanup_config['pixel_size_um'],
                    )
                    if crop_result:
                        feat['crop_b64'] = crop_result['crop']
                        feat['mask_b64'] = crop_result['mask']

                output_queue.put(build_mk_hspc_result(
                    tid=tid,
                    status='success',
                    mk_masks=mk_masks,
                    mk_feats=mk_feats,
                    hspc_masks=hspc_masks,
                    hspc_feats=hspc_feats,
                    tile=tile,
                    slide_name=slide_name
                ))

            except Exception as e:
                logger.error(f"[{worker_name}] Error processing tile {tid}: {e}")
                output_queue.put(build_mk_hspc_result(
                    tid=tid,
                    status='error',
                    tile=tile,
                    slide_name=slide_name,
                    error=str(e)
                ))

            tiles_processed += 1

            # Cleanup after every tile to prevent memory accumulation
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[{worker_name}] Worker loop error: {e}")
            logger.error(traceback.format_exc())

    # Cleanup
    logger.info(f"[{worker_name}] Shutting down, processed {tiles_processed} tiles")
    del segmenter
    gc.collect()
    torch.cuda.empty_cache()

    # Close shared memory attachments (but don't unlink - main process does that)
    for shm, _ in shared_slides.values():
        shm.close()


class MultiGPUTileProcessorSHM:
    """
    Multi-GPU tile processor using shared memory.

    Usage:
        manager = SharedSlideManager()
        for name, data in slide_data.items():
            manager.add_slide(name, data['image'])

        with MultiGPUTileProcessorSHM(
            num_gpus=4,
            slide_info=manager.get_slide_info(),
            ...
        ) as processor:
            for slide_name, tile in tiles:
                processor.submit_tile(slide_name, tile)

            for _ in range(len(tiles)):
                result = processor.collect_result()
                process(result)

        manager.cleanup()
    """

    def __init__(
        self,
        num_gpus: int,
        slide_info: Dict[str, Dict[str, Any]],
        mk_classifier_path: Optional[Path] = None,
        hspc_classifier_path: Optional[Path] = None,
        mk_min_area: int = 500,
        mk_max_area: int = 50000,
        hspc_min_area: Optional[int] = None,
        hspc_max_area: Optional[int] = None,
        variance_threshold: float = 100.0,
        calibration_block_size: int = 512,
        cleanup_config: Optional[Dict[str, Any]] = None,
        norm_params: Optional[Dict[str, float]] = None,
        normalization_method: str = 'none',
        intensity_threshold: float = 220.0,
        modality: Optional[str] = None,
        per_slide_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        skip_hspc: bool = False,
    ):
        self.num_gpus = num_gpus
        self.slide_info = slide_info
        self.mk_min_area = mk_min_area
        self.mk_max_area = mk_max_area
        self.hspc_min_area = hspc_min_area
        self.hspc_max_area = hspc_max_area
        self.variance_threshold = variance_threshold
        self.calibration_block_size = calibration_block_size
        self.cleanup_config = cleanup_config or {'cleanup_masks': False, 'fill_holes': True, 'pixel_size_um': 0.1725}
        self.norm_params = norm_params
        self.normalization_method = normalization_method
        self.intensity_threshold = intensity_threshold
        self.modality = modality
        self.per_slide_thresholds = per_slide_thresholds
        self.skip_hspc = skip_hspc

        self.segmenter_kwargs = {
            'mk_classifier_path': str(mk_classifier_path) if mk_classifier_path else None,
            'hspc_classifier_path': str(hspc_classifier_path) if hspc_classifier_path else None,
        }

        self.workers: List[multiprocessing.Process] = []
        self.input_queue: Optional[multiprocessing.Queue] = None
        self.output_queue: Optional[multiprocessing.Queue] = None
        self.stop_event: Optional[multiprocessing.Event] = None
        self.tiles_submitted = 0

    def start(self):
        """Start worker processes."""
        logger.info(f"Starting {self.num_gpus} GPU workers with shared memory (spawn context)...")

        ctx = multiprocessing.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.stop_event = ctx.Event()

        for gpu_id in range(self.num_gpus):
            p = ctx.Process(
                target=_gpu_worker_shm,
                args=(
                    gpu_id,
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.slide_info,
                    self.segmenter_kwargs,
                    self.mk_min_area,
                    self.mk_max_area,
                    self.hspc_min_area,
                    self.hspc_max_area,
                    self.variance_threshold,
                    self.calibration_block_size,
                    self.cleanup_config,
                    self.norm_params,
                    self.normalization_method,
                    self.intensity_threshold,
                    self.modality,
                    self.per_slide_thresholds,
                    self.skip_hspc,
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Started worker for GPU {gpu_id} (PID: {p.pid})")

        # Wait for all workers to be ready
        ready_count = 0
        errors = []
        while ready_count < self.num_gpus:
            try:
                msg = self.output_queue.get(timeout=120)
                if msg.get('status') == 'ready':
                    ready_count += 1
                    logger.info(f"Worker GPU-{msg['gpu_id']} ready ({ready_count}/{self.num_gpus})")
                elif msg.get('status') == 'init_error':
                    errors.append(f"GPU-{msg['gpu_id']}: {msg['error']}")
            except queue.Empty:
                logger.error("Timeout waiting for workers to initialize")
                break

        if errors:
            logger.error(f"Worker initialization errors: {errors}")
            self.stop()
            raise RuntimeError(f"Failed to initialize workers: {errors}")

        logger.info(f"All {self.num_gpus} workers ready (shared memory mode)")

    def submit_tile(self, slide_name: str, tile: Dict[str, Any]):
        """Submit a tile for processing (only coordinates, not data!)."""
        self.input_queue.put((slide_name, tile))
        self.tiles_submitted += 1

    def collect_result(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Collect one result from workers, filtering stray init messages."""
        import time
        start = time.time()
        remaining = timeout if timeout is not None else float('inf')

        while remaining > 0:
            try:
                wait = min(remaining, 1.0) if timeout is not None else 1.0
                result = self.output_queue.get(timeout=wait)
                status = result.get('status')
                if status in ('ready', 'init_error'):
                    continue  # Skip stray worker init messages
                return result
            except queue.Empty:
                if timeout is not None:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return None
                continue
        return None

    def stop(self):
        """Stop all workers."""
        # First, try to send shutdown signals (sentinel values) BEFORE setting stop_event
        # This avoids race condition where workers exit before reading sentinels
        if self.input_queue:
            for _ in range(self.num_gpus):
                try:
                    # Use timeout to avoid blocking if queue is full
                    self.input_queue.put(None, timeout=1.0)
                except (queue.Full, Exception):
                    pass

        # Now set stop event as fallback
        if self.stop_event:
            self.stop_event.set()

        # Wait for workers to finish
        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} did not stop gracefully, terminating")
                p.terminate()

        self.workers.clear()
        logger.info("All workers stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
