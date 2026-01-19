"""
Multi-GPU tile processing for parallel segmentation.

This module enables true multi-GPU processing where each GPU processes
one tile at a time. For 4 GPUs, 4 tiles are processed in parallel.

Architecture:
- Main process: feeds tiles to input queue, collects results from output queue
- N GPU workers: each pinned to one GPU, pulls tiles from queue, processes, returns results

Usage:
    from segmentation.processing.multigpu import MultiGPUTileProcessor

    with MultiGPUTileProcessor(num_gpus=4, segmenter_kwargs={...}) as processor:
        results = processor.process_tiles(tiles, slide_data, ...)
"""

import os
import gc
import queue
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from multiprocessing import Process, Queue, Event
import numpy as np
import torch

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def _gpu_worker(
    gpu_id: int,
    input_queue: Queue,
    output_queue: Queue,
    stop_event: Event,
    segmenter_kwargs: Dict[str, Any],
    mk_min_area: int,
    mk_max_area: int,
    variance_threshold: float,
    calibration_block_size: int,
):
    """
    Worker process that runs on a single GPU.

    Each worker:
    1. Sets CUDA_VISIBLE_DEVICES to see only its assigned GPU
    2. Loads models on that GPU (appears as cuda:0 to the worker)
    3. Pulls tiles from input_queue, processes them, puts results in output_queue
    4. Runs until stop_event is set

    Args:
        gpu_id: Which GPU this worker owns (0, 1, 2, or 3)
        input_queue: Queue of (tile_data, slide_name, tile_dict) tuples
        output_queue: Queue for results
        stop_event: Event to signal worker shutdown
        segmenter_kwargs: kwargs for UnifiedSegmenter initialization
        mk_min_area: Minimum MK area in pixels
        mk_max_area: Maximum MK area in pixels
        variance_threshold: Threshold for tissue detection
        calibration_block_size: Block size for variance calculation
    """
    # Pin to specific GPU - this worker only sees one GPU as cuda:0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Now import torch and models (after setting CUDA_VISIBLE_DEVICES)
    import torch

    # Import segmenter and utilities
    # These imports happen AFTER CUDA_VISIBLE_DEVICES is set
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from run_unified_FAST import UnifiedSegmenter
    from segmentation.io.html_export import percentile_normalize
    from segmentation.detection.tissue import has_tissue
    from segmentation.processing.mk_hspc_utils import (
        ensure_rgb_array,
        check_tile_validity,
        prepare_tile_for_detection,
        build_mk_hspc_result,
    )

    worker_name = f"GPU-{gpu_id}"
    logger.info(f"[{worker_name}] Starting on GPU {gpu_id}")
    logger.info(f"[{worker_name}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"[{worker_name}] Device name: {torch.cuda.get_device_name(0)}")

    # Initialize segmenter on this GPU (appears as cuda:0 to us)
    try:
        segmenter = UnifiedSegmenter(
            device="cuda:0",  # Always cuda:0 since we set CUDA_VISIBLE_DEVICES
            **segmenter_kwargs
        )
        logger.info(f"[{worker_name}] Models loaded successfully")
    except Exception as e:
        logger.error(f"[{worker_name}] Failed to load models: {e}")
        logger.error(traceback.format_exc())
        output_queue.put({'status': 'init_error', 'gpu_id': gpu_id, 'error': str(e)})
        return

    # Signal ready
    output_queue.put({'status': 'ready', 'gpu_id': gpu_id})

    tiles_processed = 0

    # Process tiles until stop signal
    while not stop_event.is_set():
        try:
            # Get next tile (with timeout to check stop_event)
            try:
                work_item = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if work_item is None:
                # Poison pill - shutdown
                logger.info(f"[{worker_name}] Received shutdown signal")
                break

            tile_img, slide_name, tile = work_item
            tid = tile['id']

            try:
                # Convert to RGB
                img_rgb = ensure_rgb_array(tile_img)

                # Check validity (is_valid=False means empty)
                is_valid, status = check_tile_validity(img_rgb, tid)
                if not is_valid:
                    output_queue.put({
                        'status': status,  # 'empty'
                        'tid': tid,
                        'slide_name': slide_name,
                        'tile': tile,
                    })
                    tiles_processed += 1
                    continue

                # Normalize
                img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)

                # Check for tissue
                try:
                    has_tissue_content, _ = has_tissue(
                        img_rgb, variance_threshold, block_size=calibration_block_size
                    )
                except Exception:
                    has_tissue_content = True  # Assume tissue on error

                if not has_tissue_content:
                    output_queue.put({
                        'status': 'no_tissue',
                        'tid': tid,
                        'slide_name': slide_name,
                        'tile': tile,
                    })
                    tiles_processed += 1
                    continue

                # Process tile with segmenter
                # process_tile returns: (mk_masks, hspc_masks, mk_features, hspc_features)
                mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
                    img_rgb, mk_min_area, mk_max_area
                )

                # Build result
                output_queue.put({
                    'status': 'success',
                    'tid': tid,
                    'slide_name': slide_name,
                    'tile': tile,
                    'mk_masks': mk_masks,
                    'mk_feats': mk_feats,
                    'hspc_masks': hspc_masks,
                    'hspc_feats': hspc_feats,
                })

                tiles_processed += 1

                # Cleanup after every tile to prevent GPU memory fragmentation
                gc.collect()
                torch.cuda.empty_cache()

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

        except Exception as e:
            logger.error(f"[{worker_name}] Worker loop error: {e}")
            logger.error(traceback.format_exc())

    # Cleanup
    logger.info(f"[{worker_name}] Shutting down after processing {tiles_processed} tiles")
    try:
        del segmenter
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


class MultiGPUTileProcessor:
    """
    Multi-GPU tile processor that distributes tiles across GPUs.

    Each GPU processes one tile at a time, enabling true parallel processing
    where N GPUs process N tiles simultaneously.

    Usage:
        processor = MultiGPUTileProcessor(
            num_gpus=4,
            mk_classifier_path=None,
            hspc_classifier_path=None,
        )
        processor.start()

        for tile_img, slide_name, tile in tiles:
            processor.submit_tile(tile_img, slide_name, tile)

        results = processor.collect_all_results()
        processor.stop()

    Or as context manager:
        with MultiGPUTileProcessor(num_gpus=4) as processor:
            for ...:
                processor.submit_tile(...)
            results = processor.collect_all_results()
    """

    def __init__(
        self,
        num_gpus: int = 4,
        mk_classifier_path: Optional[str] = None,
        hspc_classifier_path: Optional[str] = None,
        mk_min_area: int = 1000,
        mk_max_area: int = 50000,
        variance_threshold: float = 100.0,
        calibration_block_size: int = 256,
    ):
        """
        Initialize the multi-GPU processor.

        Args:
            num_gpus: Number of GPUs to use (default 4)
            mk_classifier_path: Path to MK classifier (optional)
            hspc_classifier_path: Path to HSPC classifier (optional)
            mk_min_area: Minimum MK area in pixels
            mk_max_area: Maximum MK area in pixels
            variance_threshold: Threshold for tissue detection
            calibration_block_size: Block size for variance calculation
        """
        # Validate GPU availability
        available_gpus = torch.cuda.device_count()
        if available_gpus < num_gpus:
            logger.warning(
                f"Requested {num_gpus} GPUs but only {available_gpus} available. "
                f"Using {available_gpus} GPUs."
            )
            num_gpus = max(1, available_gpus)

        self.num_gpus = num_gpus
        self.segmenter_kwargs = {
            'mk_classifier_path': mk_classifier_path,
            'hspc_classifier_path': hspc_classifier_path,
        }
        self.mk_min_area = mk_min_area
        self.mk_max_area = mk_max_area
        self.variance_threshold = variance_threshold
        self.calibration_block_size = calibration_block_size

        # Queues and workers
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.stop_event: Optional[Event] = None
        self.workers: List[Process] = []
        self.started = False
        self.tiles_submitted = 0

    def start(self) -> bool:
        """
        Start the GPU worker processes.

        Returns:
            True if all workers started successfully, False otherwise.
        """
        if self.started:
            logger.warning("MultiGPUTileProcessor already started")
            return True

        logger.info(f"Starting {self.num_gpus} GPU workers...")

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.stop_event = Event()

        # Start workers
        for gpu_id in range(self.num_gpus):
            p = Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.segmenter_kwargs,
                    self.mk_min_area,
                    self.mk_max_area,
                    self.variance_threshold,
                    self.calibration_block_size,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Started worker process {p.pid} for GPU {gpu_id}")

        # Wait for all workers to signal ready
        ready_count = 0
        errors = []

        while ready_count < self.num_gpus:
            try:
                msg = self.output_queue.get(timeout=120)  # 2 min timeout for model loading
                if msg['status'] == 'ready':
                    ready_count += 1
                    logger.info(f"GPU {msg['gpu_id']} ready ({ready_count}/{self.num_gpus})")
                elif msg['status'] == 'init_error':
                    errors.append(f"GPU {msg['gpu_id']}: {msg['error']}")
            except queue.Empty:
                logger.error("Timeout waiting for workers to initialize")
                self.stop()
                return False

        if errors:
            logger.error(f"Worker initialization errors: {errors}")
            self.stop()
            return False

        self.started = True
        logger.info(f"All {self.num_gpus} GPU workers ready")
        return True

    def submit_tile(
        self,
        tile_img: np.ndarray,
        slide_name: str,
        tile: Dict[str, Any],
    ):
        """
        Submit a tile for processing.

        Args:
            tile_img: The tile image as numpy array (H, W, C)
            slide_name: Name of the slide this tile belongs to
            tile: Tile metadata dict with 'id', 'x', 'y', 'w', 'h' keys
        """
        if not self.started:
            raise RuntimeError("Processor not started. Call start() first.")

        self.input_queue.put((tile_img, slide_name, tile))
        self.tiles_submitted += 1

    def collect_result(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Collect one result from the output queue.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            Result dict or None if timeout
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def collect_all_results(self, progress_callback: Callable[[int, int], None] = None) -> List[Dict[str, Any]]:
        """
        Collect all submitted tile results.

        Args:
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            List of all result dicts
        """
        results = []
        collected = 0

        while collected < self.tiles_submitted:
            result = self.collect_result(timeout=300)  # 5 min timeout per tile
            if result is None:
                logger.error(f"Timeout collecting results ({collected}/{self.tiles_submitted})")
                break

            # Skip ready messages (shouldn't happen here, but be safe)
            if result.get('status') == 'ready':
                continue

            results.append(result)
            collected += 1

            if progress_callback:
                progress_callback(collected, self.tiles_submitted)

        return results

    def stop(self):
        """Stop all worker processes."""
        if not self.started and not self.workers:
            return

        logger.info("Stopping GPU workers...")

        # Signal stop
        if self.stop_event:
            self.stop_event.set()

        # Send poison pills
        if self.input_queue:
            for _ in range(self.num_gpus):
                try:
                    self.input_queue.put(None)
                except Exception:
                    pass

        # Wait for workers to finish
        for p in self.workers:
            try:
                p.join(timeout=10)
                if p.is_alive():
                    logger.warning(f"Worker {p.pid} did not stop gracefully, terminating")
                    p.terminate()
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")

        self.workers = []
        self.started = False
        self.tiles_submitted = 0

        logger.info("All GPU workers stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def process_tiles_multi_gpu(
    sampled_tiles: List[Tuple[str, Dict]],
    slide_data: Dict[str, Dict],
    num_gpus: int = 4,
    mk_classifier_path: Optional[str] = None,
    hspc_classifier_path: Optional[str] = None,
    mk_min_area: int = 1000,
    mk_max_area: int = 50000,
    variance_threshold: float = 100.0,
    calibration_block_size: int = 256,
    progress_callback: Callable[[int, int], None] = None,
) -> List[Dict[str, Any]]:
    """
    Process tiles using multiple GPUs.

    Convenience function that creates a MultiGPUTileProcessor, submits all tiles,
    collects results, and cleans up.

    Args:
        sampled_tiles: List of (slide_name, tile_dict) tuples
        slide_data: Dict mapping slide_name -> {'image': np.array, ...}
        num_gpus: Number of GPUs to use
        mk_classifier_path: Path to MK classifier
        hspc_classifier_path: Path to HSPC classifier
        mk_min_area: Minimum MK area in pixels
        mk_max_area: Maximum MK area in pixels
        variance_threshold: Threshold for tissue detection
        calibration_block_size: Block size for variance calculation
        progress_callback: Optional callback(completed, total)

    Returns:
        List of result dicts from all tiles
    """
    with MultiGPUTileProcessor(
        num_gpus=num_gpus,
        mk_classifier_path=mk_classifier_path,
        hspc_classifier_path=hspc_classifier_path,
        mk_min_area=mk_min_area,
        mk_max_area=mk_max_area,
        variance_threshold=variance_threshold,
        calibration_block_size=calibration_block_size,
    ) as processor:

        # Submit all tiles
        logger.info(f"Submitting {len(sampled_tiles)} tiles to {num_gpus} GPUs...")
        for slide_name, tile in sampled_tiles:
            img = slide_data[slide_name]['image']
            tile_img = img[tile['y']:tile['y']+tile['h'],
                         tile['x']:tile['x']+tile['w']].copy()
            processor.submit_tile(tile_img, slide_name, tile)

        # Collect results
        logger.info("Collecting results...")
        results = processor.collect_all_results(progress_callback)

    return results
