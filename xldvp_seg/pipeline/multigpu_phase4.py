"""Multi-GPU orchestrator for Phase 4 (nuclear counting).

Each GPU worker holds its own Cellpose (+ optional SAM2) instance and pulls
tiles from a shared queue. Per-tile results are written to a scratch directory
as JSON; the main process reads them back and enriches detection dicts in-place.

**Why scratch JSON instead of mp.Queue:**
    Per-tile results contain ~hundreds of cells with per-nucleus dicts that
    include SAM2 embeddings (256D). Pickling that volume through ``mp.Queue``
    is slower than writing to GPFS once.

**Tile dispatch:** largest-first (sorted by detection count) for load balance,
so straggling mega-tiles start early and workers drain evenly at the end.

**Requirements:** shared memory only. The loader fallback path is not
multi-GPU compatible — callers should fall back to the in-process Phase 4
loop in ``post_detection.py`` when SHM is unavailable.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Use 'spawn' for CUDA compatibility in subprocesses (same as multigpu_worker)
_mp_context = mp.get_context("spawn")
Process = _mp_context.Process
Queue = _mp_context.Queue
Event = _mp_context.Event

_SUMMARY_KEYS = (
    "n_nuclei",
    "nuclear_area_um2",
    "nuclear_area_fraction",
    "largest_nucleus_um2",
    "nuclear_solidity",
    "nuclear_eccentricity",
)


def _phase4_worker(
    gpu_id: int,
    input_queue,
    output_queue,
    stop_event,
    config: dict[str, Any],
) -> None:
    """Worker process: load models, drain tile queue, write per-tile JSON."""
    import h5py
    import hdf5plugin  # noqa: F401 — register LZ4 codec before h5py
    import numpy as np

    from xldvp_seg.analysis.nuclear_count import (
        _percentile_normalize_to_uint8,
        count_nuclei_for_tile,
    )
    from xldvp_seg.processing.shm_attach import attach_slide_shm
    from xldvp_seg.utils.device import (
        cellpose_supports_bfloat16,
        device_supports_gpu,
        empty_cache,
        set_device_for_worker,
    )

    device = set_device_for_worker(gpu_id)
    worker_name = f"GPU-{gpu_id}"
    logger.info("[%s] Phase 4 worker starting on %s", worker_name, device)

    shm = None
    try:
        shm, slide_arr = attach_slide_shm(
            config["shm_name"], config["shm_shape"], config["shm_dtype"]
        )
    except Exception as e:
        logger.error("[%s] SHM attach failed: %s", worker_name, e)
        output_queue.put({"status": "init_error", "gpu_id": gpu_id, "error": str(e)})
        return

    cellpose_model = None
    sam2_predictor = None
    try:
        from cellpose import models as cellpose_models

        cellpose_model = cellpose_models.CellposeModel(
            gpu=device_supports_gpu(device),
            pretrained_model="cpsam",
            use_bfloat16=cellpose_supports_bfloat16(),
        )
    except Exception as e:
        logger.error("[%s] Cellpose load failed: %s", worker_name, e)
        output_queue.put({"status": "init_error", "gpu_id": gpu_id, "error": str(e)})
        if shm is not None:
            shm.close()
        return

    sam2_checkpoint = config.get("sam2_checkpoint")
    sam2_cfg = config.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml")
    if config.get("extract_sam2_embeddings", True):
        # Auto-discover checkpoint if not explicitly passed (mirrors multigpu_worker)
        if sam2_checkpoint is None or not Path(sam2_checkpoint).exists():
            script_dir = Path(__file__).parent.parent.parent.resolve()
            for cp in [
                script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
                script_dir / "checkpoints" / "sam2.1_hiera_l.pt",
            ]:
                if cp.exists():
                    sam2_checkpoint = cp
                    break
        if sam2_checkpoint and Path(sam2_checkpoint).exists():
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam2_model = build_sam2(sam2_cfg, str(sam2_checkpoint), device=str(device))
                sam2_predictor = SAM2ImagePredictor(sam2_model)
            except Exception as e:
                # SAM2 is optional — log and continue without per-nucleus embeddings
                logger.warning(
                    "[%s] SAM2 load failed (continuing without embeddings): %s",
                    worker_name,
                    e,
                )
        else:
            logger.warning(
                "[%s] SAM2 checkpoint not found — continuing without embeddings",
                worker_name,
            )

    output_queue.put({"status": "ready", "gpu_id": gpu_id})

    tiles_dir = Path(config["tiles_dir"])
    scratch_dir = Path(config["scratch_dir"])
    mask_filename = config["mask_filename"]
    pixel_size_um = config["pixel_size_um"]
    min_nuclear_area = config["min_nuclear_area"]
    nuc_slot = config["nuc_slot"]
    x_start = config["x_start"]
    y_start = config["y_start"]
    cache_clear_every = config.get("cache_clear_every", 20)

    n_processed = 0
    while not stop_event.is_set():
        try:
            work = input_queue.get(timeout=1.0)
        except mp.queues.Empty:
            continue
        except (BrokenPipeError, EOFError):
            logger.warning("[%s] Queue broken, exiting", worker_name)
            break

        if work is None:
            logger.info("[%s] Shutdown signal received", worker_name)
            break

        tile_x, tile_y = work["tile_x"], work["tile_y"]
        tile_key = f"{tile_x}_{tile_y}"

        try:
            tile_dir = tiles_dir / f"tile_{tile_x}_{tile_y}"
            mask_path = tile_dir / mask_filename
            if not mask_path.exists():
                output_queue.put({"status": "skip", "tile_key": tile_key, "reason": "no_mask"})
                continue

            with h5py.File(str(mask_path), "r") as hf:
                cell_masks = hf["masks"][:]
            tile_h, tile_w = cell_masks.shape[:2]

            nuc_tile = slide_arr[
                tile_y - y_start : tile_y - y_start + tile_h,
                tile_x - x_start : tile_x - x_start + tile_w,
                nuc_slot,
            ]
            if nuc_tile.size == 0:
                output_queue.put(
                    {"status": "skip", "tile_key": tile_key, "reason": "empty_nuc_tile"}
                )
                continue

            if sam2_predictor is not None:
                nuc_uint8 = _percentile_normalize_to_uint8(nuc_tile)
                nuc_rgb = np.stack([nuc_uint8] * 3, axis=-1)
                sam2_predictor.set_image(nuc_rgb)

            results, n_nuc = count_nuclei_for_tile(
                cell_masks,
                nuc_tile,
                cellpose_model,
                pixel_size_um,
                min_nuclear_area,
                tile_x,
                tile_y,
                sam2_predictor=sam2_predictor,
            )

            if sam2_predictor is not None:
                try:
                    sam2_predictor.reset_predictor()
                except Exception as e:
                    logger.debug("[%s] SAM2 reset failed: %s", worker_name, e)

            # JSON requires str keys; cells are keyed by integer mask labels
            serializable = {str(k): v for k, v in results.items()}
            atomic_json_dump(serializable, str(scratch_dir / f"{tile_key}.json"))

            output_queue.put({"status": "ok", "tile_key": tile_key, "n_nuclei": n_nuc})

            n_processed += 1
            if cache_clear_every and n_processed % cache_clear_every == 0:
                empty_cache()

        except Exception as e:
            logger.exception("[%s] Tile %s failed", worker_name, tile_key)
            output_queue.put({"status": "error", "tile_key": tile_key, "error": str(e)})

    # Cleanup
    try:
        empty_cache()
    except Exception:
        pass
    try:
        if shm is not None:
            shm.close()
    except Exception:
        pass
    logger.info("[%s] Worker exit (%d tiles processed)", worker_name, n_processed)


def run_multigpu_phase4(
    by_tile: dict[str, list[dict]],
    detections: list[dict],
    num_gpus: int,
    *,
    tiles_dir,
    mask_filename: str,
    pixel_size_um: float,
    min_nuclear_area: int,
    slide_shm_arr,
    shm_name: str,
    nuc_channel_idx: int,
    ch_to_slot: dict[int, int],
    x_start: int,
    y_start: int,
    sam2_checkpoint=None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    extract_sam2_embeddings: bool = True,
    scratch_dir=None,
    per_tile_timeout_s: float = 600.0,
) -> int:
    """Spawn ``num_gpus`` Phase 4 workers and merge their results.

    Args:
        by_tile: detections grouped by ``"X_Y"`` tile key.
        detections: full list of detections (mutated in-place with nuclear features).
        num_gpus: number of GPU worker processes to spawn.
        tiles_dir: directory containing ``tile_X_Y/<mask_filename>`` HDF5 masks.
        mask_filename: HDF5 mask filename inside each tile dir.
        pixel_size_um: pixel size in micrometers (from CZI metadata).
        min_nuclear_area: minimum nuclear area in pixels.
        slide_shm_arr: shape/dtype source for SHM attach (not pickled, just inspected).
        shm_name: shared-memory segment name (for workers to attach).
        nuc_channel_idx: CZI channel index for nuclear stain.
        ch_to_slot: mapping of CZI channel index → SHM slot.
        x_start, y_start: mosaic origin offsets.
        sam2_checkpoint: optional SAM2 checkpoint path for per-nucleus embeddings.
        sam2_config: SAM2 model config name.
        extract_sam2_embeddings: if True, workers load SAM2 for embeddings.
        scratch_dir: directory for per-tile JSON results (default: ``<tiles_dir>/../_phase4_scratch``).
        per_tile_timeout_s: max seconds to wait for any single tile result before aborting.

    Returns:
        Number of cells enriched with nuclear-count features.
    """
    from tqdm import tqdm

    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
    if nuc_channel_idx not in ch_to_slot:
        raise ValueError(f"nuc_channel_idx {nuc_channel_idx} not in ch_to_slot {list(ch_to_slot)}")

    nuc_slot = ch_to_slot[nuc_channel_idx]
    tiles_dir = Path(tiles_dir)
    if scratch_dir is None:
        scratch_dir = tiles_dir.parent / "_phase4_scratch"
    scratch_dir = Path(scratch_dir)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)

    # Largest tiles first → straggler avoidance
    tile_keys_sorted = sorted(by_tile.keys(), key=lambda k: -len(by_tile[k]))

    config = {
        "tiles_dir": str(tiles_dir),
        "scratch_dir": str(scratch_dir),
        "mask_filename": mask_filename,
        "pixel_size_um": pixel_size_um,
        "min_nuclear_area": min_nuclear_area,
        "shm_name": shm_name,
        "shm_shape": list(slide_shm_arr.shape),
        "shm_dtype": str(slide_shm_arr.dtype),
        "nuc_slot": nuc_slot,
        "x_start": x_start,
        "y_start": y_start,
        "sam2_checkpoint": str(sam2_checkpoint) if sam2_checkpoint else None,
        "sam2_config": sam2_config,
        "extract_sam2_embeddings": extract_sam2_embeddings,
    }

    input_queue = Queue()
    output_queue = Queue()
    stop_event = Event()

    workers = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=_phase4_worker,
            args=(gpu_id, input_queue, output_queue, stop_event, config),
        )
        p.start()
        workers.append(p)

    try:
        # Wait for workers ready
        n_ready = 0
        while n_ready < num_gpus:
            try:
                msg = output_queue.get(timeout=120)
            except mp.queues.Empty as exc:
                raise RuntimeError("Phase 4 workers timed out during initialization") from exc
            if msg.get("status") == "ready":
                n_ready += 1
            elif msg.get("status") == "init_error":
                raise RuntimeError(
                    f"Phase 4 worker {msg.get('gpu_id')} init failed: {msg.get('error')}"
                )
        logger.info("Phase 4: %d workers ready", num_gpus)

        # Dispatch tiles
        for tile_key in tile_keys_sorted:
            parts = tile_key.split("_")
            input_queue.put({"tile_x": int(parts[0]), "tile_y": int(parts[1])})

        # Send shutdown sentinels
        for _ in workers:
            input_queue.put(None)

        # Drain status messages
        n_total = len(tile_keys_sorted)
        n_done = n_skip = n_err = 0
        with tqdm(total=n_total, desc="Phase 4 (multi-GPU)") as pbar:
            while n_done + n_skip + n_err < n_total:
                try:
                    msg = output_queue.get(timeout=per_tile_timeout_s)
                except mp.queues.Empty as exc:
                    raise RuntimeError(
                        f"Phase 4 timed out: no progress in {per_tile_timeout_s}s"
                    ) from exc
                s = msg.get("status")
                if s == "ok":
                    n_done += 1
                elif s == "skip":
                    n_skip += 1
                elif s == "error":
                    n_err += 1
                    logger.warning(
                        "Phase 4 tile %s error: %s", msg.get("tile_key"), msg.get("error")
                    )
                pbar.update(1)

        logger.info("Phase 4 workers complete: ok=%d skip=%d err=%d", n_done, n_skip, n_err)

        # Wait for clean exits
        for p in workers:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning("Phase 4 worker %s still alive after join — terminating", p.pid)
                p.terminate()
                p.join(timeout=5)

        # Merge per-tile JSONs into detection dicts
        n_enriched = _merge_phase4_results(by_tile, scratch_dir)
        return n_enriched

    finally:
        # Always release workers and clean scratch
        stop_event.set()
        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _merge_phase4_results(by_tile: dict[str, list[dict]], scratch_dir: Path) -> int:
    """Read per-tile JSONs from *scratch_dir* and enrich detections in-place.

    Returns the number of cells enriched (i.e., assigned at least summary metrics).
    """
    scratch_dir = Path(scratch_dir)
    n_enriched = 0

    for tile_key, tile_dets in by_tile.items():
        result_path = scratch_dir / f"{tile_key}.json"
        if not result_path.exists():
            continue
        try:
            results = fast_json_load(str(result_path))
        except Exception as e:
            logger.warning("Failed to read Phase 4 result %s: %s", result_path, e)
            continue

        for det in tile_dets:
            mask_label = det.get("tile_mask_label", det.get("mask_label"))
            if mask_label is None:
                continue
            key = str(int(mask_label))
            if key not in results:
                continue
            nuc_feats = results[key]
            features = det.setdefault("features", {})
            for k in _SUMMARY_KEYS:
                if k in nuc_feats:
                    features[k] = nuc_feats[k]
            if nuc_feats.get("nuclei"):
                det["nuclei"] = nuc_feats["nuclei"]
            n_enriched += 1

    return n_enriched
