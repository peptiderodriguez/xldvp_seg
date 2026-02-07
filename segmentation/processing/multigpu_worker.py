"""
Generic multi-GPU tile processor with shared memory (zero-copy).

This module generalizes the NMJ-specific multi-GPU worker to support all cell types.
Architecture is identical to multigpu_nmj.py:
- Main process: loads slide channels into shared memory, sends tile coordinates to queue
- N GPU workers: each pinned to one GPU, reads tiles from shared memory, runs detection

Supports all cell types: nmj, mk, cell, vessel, mesothelium.

Coordinate System:
    Workers receive tiles with RELATIVE coordinates (array indices), NOT global CZI.
    Tile dict format:
        tile = {'x': 0, 'y': 0, 'w': 3000, 'h': 3000}

    Global coordinates for UIDs:
        global_cx = tile['x'] + local_cx
        uid = f"{slide_name}_{cell_type}_{global_cx}_{global_cy}"
"""

import atexit
import os
import gc
import json
import queue
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry of shared memory names for crash cleanup
_shm_registry: Set[str] = set()


def _cleanup_shared_memory_on_exit():
    """Emergency cleanup of shared memory on process exit."""
    for shm_name in list(_shm_registry):
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except Exception:
            pass
    _shm_registry.clear()


atexit.register(_cleanup_shared_memory_on_exit)

# Use 'spawn' start method for CUDA compatibility in subprocesses
_mp_context = mp.get_context('spawn')
Process = _mp_context.Process
Queue = _mp_context.Queue
Event = _mp_context.Event


def _gpu_worker(
    gpu_id: int,
    input_queue,
    output_queue,
    stop_event,
    slide_info: Dict[str, Dict[str, Any]],
    cell_type: str,
    strategy_params: Dict[str, Any],
    classifier_path: Optional[str],
    pixel_size_um: float,
    extract_deep_features: bool = False,
    extract_sam2_embeddings: bool = True,
    detection_channel: int = 1,
    sam2_checkpoint: Optional[str] = None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    cd31_channel: Optional[int] = None,
    channel_names: Optional[Dict[int, str]] = None,
):
    """
    Generic GPU worker for any cell type.

    Each worker:
    1. Pins to assigned GPU
    2. Loads models appropriate for the cell type
    3. Creates strategy via create_strategy_for_cell_type()
    4. Processes tiles from shared memory via queue using process_single_tile()

    Args:
        gpu_id: Which GPU this worker owns
        input_queue: Queue of (slide_name, tile_dict) tuples
        output_queue: Queue for results
        stop_event: Event to signal shutdown
        slide_info: Dict mapping slide_name -> {shm_name, shape, dtype}
        cell_type: 'nmj', 'mk', 'cell', 'vessel', 'mesothelium'
        strategy_params: Strategy-specific parameters
        classifier_path: Path to classifier (.pkl/.joblib/.pth), or None
        pixel_size_um: Pixel size for area calculations
        extract_deep_features: Whether to load ResNet+DINOv2 (default False)
        extract_sam2_embeddings: Whether to extract SAM2 embeddings
        detection_channel: Which channel for tissue detection (default 1)
        sam2_checkpoint: Path to SAM2 checkpoint
        sam2_config: SAM2 config file
        cd31_channel: Optional channel index for CD31 (vessel validation)
        channel_names: Optional channel name mapping for vessels
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from segmentation.detection.tissue import has_tissue
    from segmentation.processing.tile_processing import process_single_tile

    worker_name = f"GPU-{gpu_id}"
    logger.info(f"[{worker_name}] Starting {cell_type} worker...")

    # --- Attach to shared memory ---
    shared_slides: Dict[str, Tuple[SharedMemory, np.ndarray]] = {}
    try:
        for slide_name, info in slide_info.items():
            shm = SharedMemory(name=info['shm_name'])
            arr = np.ndarray(info['shape'], dtype=np.dtype(info['dtype']), buffer=shm.buf)
            shared_slides[slide_name] = (shm, arr)
    except Exception as e:
        logger.error(f"[{worker_name}] Failed to attach shared memory: {e}")
        output_queue.put({'status': 'init_error', 'gpu_id': gpu_id, 'error': str(e)})
        return

    device = torch.device('cuda:0')
    models: Dict[str, Any] = {'device': device}

    # --- Load SAM2 predictor ---
    if extract_sam2_embeddings:
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2

            checkpoint_path = None
            if sam2_checkpoint and Path(sam2_checkpoint).exists():
                checkpoint_path = Path(sam2_checkpoint)
            else:
                script_dir = Path(__file__).parent.parent.parent.resolve()
                for cp in [
                    script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
                    script_dir / "checkpoints" / "sam2.1_hiera_l.pt",
                ]:
                    if cp.exists():
                        checkpoint_path = cp
                        break

            if checkpoint_path:
                logger.info(f"[{worker_name}] Loading SAM2 from {checkpoint_path}...")
                sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=device)
                sam2_predictor = SAM2ImagePredictor(sam2_model)
                models['sam2_predictor'] = sam2_predictor
                logger.info(f"[{worker_name}] SAM2 loaded")
            else:
                logger.warning(f"[{worker_name}] SAM2 checkpoint not found")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load SAM2: {e}")

    # --- Load SAM2 auto mask generator (MK/Cell need it) ---
    if cell_type in ('mk',):
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            if 'sam2_predictor' in models:
                sam2_auto = SAM2AutomaticMaskGenerator(
                    models['sam2_predictor'].model,
                    points_per_side=24,
                    pred_iou_thresh=0.5,
                    stability_score_thresh=0.4,
                    min_mask_region_area=500,
                    crop_n_layers=1,
                )
                models['sam2_auto'] = sam2_auto
                logger.info(f"[{worker_name}] SAM2 auto mask generator loaded")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load SAM2 auto: {e}")

    # --- Load Cellpose (Cell strategy needs it) ---
    if cell_type in ('cell',):
        try:
            from cellpose import models as cellpose_models
            cellpose_model = cellpose_models.CellposeModel(gpu=True, model_type='cyto3')
            models['cellpose'] = cellpose_model
            logger.info(f"[{worker_name}] Cellpose loaded")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load Cellpose: {e}")

    # --- Load deep feature models if requested ---
    if extract_deep_features:
        try:
            from segmentation.models.manager import load_resnet_feature_extractor, load_dinov2
            resnet, resnet_transform = load_resnet_feature_extractor(device)
            models['resnet'] = resnet
            models['resnet_transform'] = resnet_transform
            logger.info(f"[{worker_name}] ResNet50 loaded")

            dinov2, dinov2_transform = load_dinov2(device)
            models['dinov2'] = dinov2
            models['dinov2_transform'] = dinov2_transform
            logger.info(f"[{worker_name}] DINOv2 loaded")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load deep feature models: {e}")

    # --- Load classifier ---
    if classifier_path and Path(classifier_path).exists():
        try:
            if classifier_path.endswith('.pth'):
                from segmentation.detection.strategies.nmj import load_nmj_classifier
                model, transform, dev = load_nmj_classifier(classifier_path, device)
                models['classifier'] = model
                models['transform'] = transform
                models['classifier_type'] = 'cnn'
            else:
                from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
                classifier_data = load_nmj_rf_classifier(classifier_path)
                models['classifier'] = classifier_data['pipeline']
                models['classifier_type'] = 'rf'
                models['scaler'] = None
                models['feature_names'] = classifier_data['feature_names']
            logger.info(f"[{worker_name}] Classifier loaded from {classifier_path}")
        except Exception as e:
            logger.warning(f"[{worker_name}] Failed to load classifier: {e}")

    # --- Create strategy ---
    from segmentation.detection.strategies.nmj import NMJStrategy
    from segmentation.detection.strategies.mk import MKStrategy
    from segmentation.detection.strategies.cell import CellStrategy

    if cell_type == 'nmj':
        strategy = NMJStrategy(
            intensity_percentile=strategy_params.get('intensity_percentile', 98.0),
            max_solidity=strategy_params.get('max_solidity', 0.85),
            min_skeleton_length=strategy_params.get('min_skeleton_length', 30),
            min_area_px=strategy_params.get('min_area', 150),
            min_area_um=strategy_params.get('min_area_um', 25.0),
            classifier_threshold=strategy_params.get('classifier_threshold', 0.5),
            use_classifier='classifier' in models,
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
        )
    elif cell_type == 'mk':
        strategy = MKStrategy(
            min_area_um=strategy_params.get('min_area_um', 200.0),
            max_area_um=strategy_params.get('max_area_um', 2000.0),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
        )
    elif cell_type == 'cell':
        strategy = CellStrategy(
            min_area_um=strategy_params.get('min_area_um', 50),
            max_area_um=strategy_params.get('max_area_um', 200),
        )
    elif cell_type == 'vessel':
        from segmentation.detection.strategies.vessel import VesselStrategy
        strategy = VesselStrategy(
            min_diameter_um=strategy_params.get('min_vessel_diameter_um', 10),
            max_diameter_um=strategy_params.get('max_vessel_diameter_um', 1000),
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
        )
    else:
        from segmentation.detection.strategies.mesothelium import MesotheliumStrategy
        strategy = MesotheliumStrategy(
            pixel_size_um=pixel_size_um,
            **{k: v for k, v in strategy_params.items() if k != 'pixel_size_um'},
        )

    logger.info(f"[{worker_name}] {cell_type} strategy created: {strategy.get_config()}")

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

            if slide_name not in shared_slides:
                logger.error(f"[{worker_name}] Unknown slide: {slide_name}")
                output_queue.put({
                    'status': 'error', 'tid': tid,
                    'slide_name': slide_name, 'tile': tile,
                    'error': f'Unknown slide: {slide_name}'
                })
                tiles_processed += 1
                continue

            _, slide_arr = shared_slides[slide_name]

            # Extract tile from shared memory
            y_start, x_start = tile['y'], tile['x']
            tile_h, tile_w = tile['h'], tile['w']

            y_end = min(y_start + tile_h, slide_arr.shape[0])
            x_end = min(x_start + tile_w, slide_arr.shape[1])

            if slide_arr.ndim == 3:
                tile_raw = slide_arr[y_start:y_end, x_start:x_end, :].copy()
                n_ch = tile_raw.shape[2]
                if n_ch >= 3:
                    tile_rgb = tile_raw[:, :, :3]
                elif n_ch == 2:
                    # Pad 2-channel to 3-channel (repeat last channel)
                    tile_rgb = np.stack([tile_raw[:, :, 0], tile_raw[:, :, 1], tile_raw[:, :, 1]], axis=-1)
                else:
                    tile_rgb = np.stack([tile_raw[:, :, 0]] * 3, axis=-1)
            else:
                tile_gray = slide_arr[y_start:y_end, x_start:x_end].copy()
                tile_rgb = np.stack([tile_gray] * 3, axis=-1)

            # Validate tile
            if tile_rgb.size == 0 or tile_rgb.max() == 0:
                output_queue.put({
                    'status': 'empty', 'tid': tid,
                    'slide_name': slide_name, 'tile': tile,
                })
                tiles_processed += 1
                continue

            # Build extra_channel_tiles from REAL channels (before RGB padding/uint8 conversion)
            extra_channel_tiles = None
            if slide_arr.ndim == 3:
                n_real_ch = slide_arr.shape[2]
                if n_real_ch >= 2:
                    extra_channel_tiles = {}
                    for ch_idx in range(n_real_ch):
                        extra_channel_tiles[ch_idx] = slide_arr[y_start:y_end, x_start:x_end, ch_idx].copy()

            # has_tissue() check on uint16 BEFORE conversion
            try:
                det_ch = extra_channel_tiles.get(detection_channel, tile_rgb[:, :, 0]) if extra_channel_tiles else tile_rgb[:, :, 0]
                has_tissue_flag, _ = has_tissue(det_ch, variance_threshold=1000.0, block_size=64)
            except Exception:
                has_tissue_flag = True

            # Convert to uint8 for visual models
            if tile_rgb.dtype != np.uint8:
                if tile_rgb.dtype == np.uint16:
                    tile_rgb = (tile_rgb / 256).astype(np.uint8)
                else:
                    tile_rgb = tile_rgb.astype(np.uint8)

            if not has_tissue_flag:
                output_queue.put({
                    'status': 'no_tissue', 'tid': tid,
                    'slide_name': slide_name, 'tile': tile,
                })
                tiles_processed += 1
                continue

            # Extract CD31 channel tile for vessel validation (if available in shared memory)
            cd31_channel_data = None
            if cell_type == 'vessel' and cd31_channel is not None and extra_channel_tiles is not None:
                cd31_tile = extra_channel_tiles.get(cd31_channel)
                if cd31_tile is not None:
                    cd31_channel_data = cd31_tile.astype(np.float32)

            # Process tile using common function (includes CUDA retry)
            try:
                result = process_single_tile(
                    tile_rgb=tile_rgb,
                    extra_channel_tiles=extra_channel_tiles,
                    strategy=strategy,
                    models=models,
                    pixel_size_um=pixel_size_um,
                    cell_type=cell_type,
                    slide_name=slide_name,
                    tile_x=tile['x'],
                    tile_y=tile['y'],
                    cd31_channel_data=cd31_channel_data,
                    channel_names=channel_names,
                    max_retries=3,
                )

                if result is None:
                    output_queue.put({
                        'status': 'success', 'tid': tid,
                        'slide_name': slide_name, 'tile': tile,
                        'masks': np.zeros(tile_rgb.shape[:2], dtype=np.uint32),
                        'features_list': [],
                    })
                else:
                    masks, features_list = result
                    detections_found += len(features_list)
                    output_queue.put({
                        'status': 'success', 'tid': tid,
                        'slide_name': slide_name, 'tile': tile,
                        'masks': masks,
                        'features_list': features_list,
                    })

            except Exception as e:
                logger.error(f"[{worker_name}] Error on tile {tid}: {e}")
                logger.error(traceback.format_exc())
                output_queue.put({
                    'status': 'error', 'tid': tid,
                    'slide_name': slide_name, 'tile': tile,
                    'error': str(e),
                })

            tiles_processed += 1
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[{worker_name}] Worker loop error: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"[{worker_name}] Done, processed {tiles_processed} tiles, found {detections_found} detections")
    gc.collect()
    torch.cuda.empty_cache()

    for shm, _ in shared_slides.values():
        shm.close()


class MultiGPUTileProcessor:
    """
    Generic multi-GPU tile processor using shared memory.

    Supports all cell types. Drop-in replacement for MultiGPUNMJProcessor
    with additional cell_type parameter.

    Usage:
        processor = MultiGPUTileProcessor(
            num_gpus=4,
            slide_info=manager.get_slide_info(),
            cell_type='nmj',
            strategy_params={'intensity_percentile': 98.0},
            pixel_size_um=0.1725,
        )
        with processor:
            for tile in tiles:
                processor.submit_tile('slide1', tile)
            for _ in range(len(tiles)):
                result = processor.collect_result()
    """

    def __init__(
        self,
        num_gpus: int,
        slide_info: Dict[str, Dict[str, Any]],
        cell_type: str,
        strategy_params: Dict[str, Any],
        pixel_size_um: float,
        classifier_path: Optional[str] = None,
        extract_deep_features: bool = False,
        extract_sam2_embeddings: bool = True,
        detection_channel: int = 1,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        cd31_channel: Optional[int] = None,
        channel_names: Optional[Dict[int, str]] = None,
    ):
        self.num_gpus = num_gpus
        self.slide_info = slide_info
        self.cell_type = cell_type
        self.strategy_params = strategy_params
        self.pixel_size_um = pixel_size_um
        self.classifier_path = classifier_path
        self.extract_deep_features = extract_deep_features
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.detection_channel = detection_channel
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.cd31_channel = cd31_channel
        self.channel_names = channel_names

        self.workers: List[Process] = []
        self.input_queue: Optional[Queue] = None
        self.output_queue: Optional[Queue] = None
        self.stop_event: Optional[Event] = None
        self.tiles_submitted = 0
        self._local_checkpoint_path: Optional[Path] = None
        self._workers_started = False

    def _copy_checkpoint_to_local(self) -> str:
        """Copy SAM2 checkpoint to local /tmp for faster loading."""
        if self.sam2_checkpoint:
            checkpoint_path = Path(self.sam2_checkpoint)
        else:
            checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "sam2.1_hiera_large.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        local_dir = Path("/tmp") / f"sam2_cache_{os.getpid()}"
        local_dir.mkdir(exist_ok=True)
        local_path = local_dir / checkpoint_path.name

        if local_path.exists():
            logger.info(f"SAM2 checkpoint already cached at {local_path}")
            self._local_checkpoint_path = local_path
            return str(local_path)

        logger.info(f"Copying SAM2 checkpoint to /tmp...")
        start = time.time()
        shutil.copy2(checkpoint_path, local_path)
        logger.info(f"Copied in {time.time() - start:.1f}s")
        self._local_checkpoint_path = local_path
        return str(local_path)

    def _cleanup_local_checkpoint(self):
        """Remove local checkpoint copy."""
        if self._local_checkpoint_path and self._local_checkpoint_path.exists():
            try:
                shutil.rmtree(self._local_checkpoint_path.parent)
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint: {e}")

    def start(self):
        """Start worker processes with staggered initialization."""
        logger.info(f"Starting {self.num_gpus} GPU workers for {self.cell_type} detection...")

        for info in self.slide_info.values():
            shm_name = info.get('shm_name')
            if shm_name:
                _shm_registry.add(shm_name)

        local_checkpoint = self._copy_checkpoint_to_local()

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.stop_event = Event()

        ready_count = 0
        errors = []

        for gpu_id in range(self.num_gpus):
            p = Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    self.input_queue,
                    self.output_queue,
                    self.stop_event,
                    self.slide_info,
                    self.cell_type,
                    self.strategy_params,
                    self.classifier_path,
                    self.pixel_size_um,
                    self.extract_deep_features,
                    self.extract_sam2_embeddings,
                    self.detection_channel,
                    local_checkpoint,
                    self.sam2_config,
                    self.cd31_channel,
                    self.channel_names,
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)
            logger.info(f"Started {self.cell_type} worker GPU-{gpu_id} (PID: {p.pid})")

            try:
                msg = self.output_queue.get(timeout=120)
                if msg.get('status') == 'ready':
                    ready_count += 1
                    logger.info(f"Worker GPU-{msg['gpu_id']} ready ({ready_count}/{self.num_gpus})")
                elif msg.get('status') == 'init_error':
                    errors.append(f"GPU-{msg['gpu_id']}: {msg['error']}")
            except queue.Empty:
                errors.append(f"GPU-{gpu_id}: timeout")

        if errors:
            self.stop()
            raise RuntimeError(f"Worker init errors: {errors}")

        if ready_count < self.num_gpus:
            self.stop()
            raise RuntimeError(f"Only {ready_count}/{self.num_gpus} workers ready")

        logger.info(f"All {self.num_gpus} workers ready")
        self._workers_started = True

    def submit_tile(self, slide_name: str, tile: Dict[str, Any]):
        """Submit a tile for processing."""
        self.input_queue.put((slide_name, tile))
        self.tiles_submitted += 1

    def collect_result(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """Collect one tile result from workers."""
        if not self._workers_started:
            raise RuntimeError("Call start() first")

        start = time.time()
        remaining = timeout if timeout else float('inf')

        while remaining > 0:
            try:
                result = self.output_queue.get(timeout=min(remaining, 1.0))
                status = result.get('status')
                if status in ('success', 'error', 'empty', 'no_tissue'):
                    return result
                if status in ('ready', 'init_error'):
                    continue
                return result
            except queue.Empty:
                if timeout:
                    remaining = timeout - (time.time() - start)
                    if remaining <= 0:
                        return None
                continue
        return None

    def stop(self):
        """Stop all workers."""
        if self.input_queue:
            for _ in range(self.num_gpus):
                try:
                    self.input_queue.put(None, timeout=1.0)
                except Exception:
                    pass

        if self.stop_event:
            self.stop_event.set()

        for p in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

        self.workers.clear()
        self._cleanup_local_checkpoint()

        for info in self.slide_info.values():
            shm_name = info.get('shm_name')
            if shm_name:
                _shm_registry.discard(shm_name)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
