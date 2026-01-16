"""
Producer-consumer pipeline for tile I/O and processing.

Uses threads (not processes) because:
- Tile reading is I/O bound (benefits from threading)
- GPU processing happens in main thread
- Avoids serialization overhead of multiprocessing

This module provides two main components:

1. TilePipeline: Asynchronous tile loading with producer-consumer pattern
   - Background thread prefetches tiles into a bounded queue
   - Main thread consumes tiles for GPU processing
   - Limits memory usage via queue maxsize

2. preprocess_tiles_batch: Parallel CPU preprocessing for batches of tiles
   - Uses ThreadPoolExecutor for CPU-bound preprocessing
   - Useful when preprocessing multiple tiles before GPU inference

Example usage:

    from segmentation.io.tile_pipeline import TilePipeline, preprocess_tiles_batch

    # Producer-consumer for tile loading
    pipeline = TilePipeline(
        loader=czi_loader,
        tile_coords=sampled_tiles,  # List of {'x': int, 'y': int} dicts
        tile_size=3000,
        channel=1,
        num_prefetch=4
    )

    for tile_x, tile_y, tile_data in pipeline:
        # GPU processing in main thread
        results = process_tile(tile_data)

    # Batch preprocessing
    preprocessed = preprocess_tiles_batch(
        tiles=[tile1, tile2, tile3],
        preprocessor=normalize_tile,
        max_workers=4
    )
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty, Full
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic preprocessing
T = TypeVar('T')


class TilePipeline:
    """
    Asynchronous tile loading with producer-consumer pattern.

    A background thread prefetches tiles from the loader and puts them into
    a bounded queue. The main thread consumes tiles from the queue for
    GPU processing. This overlaps I/O with computation for better throughput.

    The queue is bounded (maxsize=num_prefetch) to limit memory usage.
    When the queue is full, the producer thread blocks until space is available.

    Attributes:
        loader: CZI loader instance with get_tile() method.
        tile_coords: List of tile coordinate dicts with 'x' and 'y' keys.
        tile_size: Size of tiles in pixels (assumes square tiles).
        channel: Channel index to load.
        num_prefetch: Maximum number of tiles to prefetch (queue size).

    Example:
        >>> from segmentation.io import get_loader
        >>> loader = get_loader('/path/to/slide.czi', load_to_ram=True, channel=1)
        >>> tiles = [{'x': 0, 'y': 0}, {'x': 3000, 'y': 0}, {'x': 6000, 'y': 0}]
        >>>
        >>> pipeline = TilePipeline(loader, tiles, tile_size=3000, channel=1)
        >>> for tile_x, tile_y, tile_data in pipeline:
        ...     if tile_data is not None:
        ...         # Process tile
        ...         masks = detector.detect(tile_data)

    Notes:
        - The pipeline yields (tile_x, tile_y, tile_data) tuples
        - tile_data may be None if loading failed (logged as warning)
        - The pipeline handles graceful shutdown on exceptions
        - Uses threading (not multiprocessing) for I/O-bound work
    """

    # Sentinel value to signal end of pipeline
    _SENTINEL = object()

    def __init__(
        self,
        loader: Any,
        tile_coords: List[Dict[str, int]],
        tile_size: int,
        channel: int,
        num_prefetch: int = 4,
        cd31_channel: Optional[int] = None,
    ):
        """
        Initialize the tile pipeline.

        Args:
            loader: CZI loader instance with get_tile(x, y, size, channel) method.
            tile_coords: List of tile coordinate dicts. Each dict must have
                'x' and 'y' keys specifying the tile origin in global coordinates.
            tile_size: Size of tiles in pixels (square tiles assumed).
            channel: Main channel index to load.
            num_prefetch: Maximum number of tiles to prefetch into the queue.
                Higher values increase memory usage but reduce stalls.
                Default is 4, which provides good overlap without excessive memory.
            cd31_channel: Optional secondary channel to load (e.g., for vessel
                validation). If provided, yields (tile_x, tile_y, tile_data, cd31_data).
        """
        self.loader = loader
        self.tile_coords = tile_coords
        self.tile_size = tile_size
        self.channel = channel
        self.num_prefetch = num_prefetch
        self.cd31_channel = cd31_channel

        # Bounded queue to limit memory usage
        self._queue: Queue = Queue(maxsize=num_prefetch)

        # Thread control
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._exception: Optional[Exception] = None

        # Statistics
        self._tiles_loaded = 0
        self._tiles_failed = 0
        self._lock = threading.Lock()

    def __iter__(self) -> Iterator[Tuple[int, int, Optional[np.ndarray], ...]]:
        """
        Iterate over tiles with asynchronous prefetching.

        Yields:
            Tuples of (tile_x, tile_y, tile_data) where:
            - tile_x: X coordinate of tile origin
            - tile_y: Y coordinate of tile origin
            - tile_data: Numpy array of tile data, or None if loading failed

            If cd31_channel was specified, yields (tile_x, tile_y, tile_data, cd31_data).
        """
        # Start the producer thread
        self._stop_event.clear()
        self._exception = None
        self._producer_thread = threading.Thread(
            target=self._producer_thread_func,
            name="TilePipeline-Producer",
            daemon=True,
        )
        self._producer_thread.start()

        try:
            while True:
                # Get next item from queue (blocks until available)
                try:
                    item = self._queue.get(timeout=1.0)
                except Empty:
                    # Check if producer is still alive
                    if not self._producer_thread.is_alive():
                        # Producer finished or crashed - drain remaining items
                        try:
                            item = self._queue.get_nowait()
                        except Empty:
                            break
                    continue

                # Check for sentinel (end of tiles)
                if item is self._SENTINEL:
                    break

                # Check for exception from producer
                if isinstance(item, Exception):
                    raise item

                yield item

        finally:
            # Signal producer to stop and wait for it
            self._stop_event.set()
            if self._producer_thread is not None and self._producer_thread.is_alive():
                self._producer_thread.join(timeout=5.0)

            # Log statistics
            with self._lock:
                logger.debug(
                    f"TilePipeline finished: {self._tiles_loaded} loaded, "
                    f"{self._tiles_failed} failed"
                )

    def _producer_thread_func(self) -> None:
        """
        Background thread that prefetches tiles into the queue.

        Runs until all tiles are loaded or stop_event is set.
        On exception, puts the exception into the queue for the consumer.
        """
        try:
            for tile_coord in self.tile_coords:
                # Check if we should stop
                if self._stop_event.is_set():
                    logger.debug("TilePipeline producer: stop requested")
                    break

                tile_x = tile_coord['x']
                tile_y = tile_coord['y']

                try:
                    # Load tile data
                    tile_data = self.loader.get_tile(
                        tile_x, tile_y, self.tile_size, channel=self.channel
                    )

                    # Load CD31 channel if specified
                    cd31_data = None
                    if self.cd31_channel is not None:
                        cd31_data = self.loader.get_tile(
                            tile_x, tile_y, self.tile_size, channel=self.cd31_channel
                        )

                    with self._lock:
                        self._tiles_loaded += 1

                    # Build result tuple
                    if self.cd31_channel is not None:
                        result = (tile_x, tile_y, tile_data, cd31_data)
                    else:
                        result = (tile_x, tile_y, tile_data)

                except Exception as e:
                    logger.warning(
                        f"TilePipeline: Failed to load tile ({tile_x}, {tile_y}): {e}"
                    )
                    with self._lock:
                        self._tiles_failed += 1

                    # Yield None for failed tile so consumer knows about it
                    if self.cd31_channel is not None:
                        result = (tile_x, tile_y, None, None)
                    else:
                        result = (tile_x, tile_y, None)

                # Put into queue (blocks if queue is full)
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(result, timeout=0.1)
                        break
                    except Full:
                        continue

        except Exception as e:
            logger.error(f"TilePipeline producer thread error: {e}")
            self._exception = e
            # Put exception into queue so consumer can raise it
            try:
                self._queue.put(e, timeout=1.0)
            except Full:
                pass

        finally:
            # Signal end of tiles
            try:
                self._queue.put(self._SENTINEL, timeout=5.0)
            except Full:
                logger.warning("TilePipeline: Could not put sentinel (queue full)")

    @property
    def stats(self) -> Dict[str, int]:
        """
        Get pipeline statistics.

        Returns:
            Dict with 'tiles_loaded' and 'tiles_failed' counts.
        """
        with self._lock:
            return {
                'tiles_loaded': self._tiles_loaded,
                'tiles_failed': self._tiles_failed,
            }


class TilePipelineWithPreprocessing(TilePipeline):
    """
    Tile pipeline with additional preprocessing in the producer thread.

    Extends TilePipeline to apply a preprocessing function to each tile
    in the producer thread before putting it into the queue. This is useful
    for CPU-bound preprocessing that can be overlapped with GPU inference.

    The preprocessor function is called with (tile_data, tile_x, tile_y) and
    should return the preprocessed data (or None to skip the tile).

    Example:
        >>> def preprocess(tile_data, tile_x, tile_y):
        ...     if tile_data is None or tile_data.max() == 0:
        ...         return None
        ...     # Normalize and convert to RGB
        ...     normalized = (tile_data - tile_data.min()) / (tile_data.max() - tile_data.min())
        ...     return np.stack([normalized] * 3, axis=-1)
        >>>
        >>> pipeline = TilePipelineWithPreprocessing(
        ...     loader=loader,
        ...     tile_coords=tiles,
        ...     tile_size=3000,
        ...     channel=1,
        ...     preprocessor=preprocess,
        ... )
    """

    def __init__(
        self,
        loader: Any,
        tile_coords: List[Dict[str, int]],
        tile_size: int,
        channel: int,
        preprocessor: Callable[[Optional[np.ndarray], int, int], Optional[np.ndarray]],
        num_prefetch: int = 4,
        cd31_channel: Optional[int] = None,
    ):
        """
        Initialize the tile pipeline with preprocessing.

        Args:
            loader: CZI loader instance.
            tile_coords: List of tile coordinate dicts.
            tile_size: Size of tiles in pixels.
            channel: Main channel index.
            preprocessor: Function to preprocess each tile. Called with
                (tile_data, tile_x, tile_y) and should return preprocessed
                data or None to skip the tile.
            num_prefetch: Maximum tiles to prefetch.
            cd31_channel: Optional secondary channel.
        """
        super().__init__(
            loader=loader,
            tile_coords=tile_coords,
            tile_size=tile_size,
            channel=channel,
            num_prefetch=num_prefetch,
            cd31_channel=cd31_channel,
        )
        self.preprocessor = preprocessor

    def _producer_thread_func(self) -> None:
        """
        Background thread that prefetches and preprocesses tiles.
        """
        try:
            for tile_coord in self.tile_coords:
                if self._stop_event.is_set():
                    logger.debug("TilePipelineWithPreprocessing producer: stop requested")
                    break

                tile_x = tile_coord['x']
                tile_y = tile_coord['y']

                try:
                    # Load tile data
                    tile_data = self.loader.get_tile(
                        tile_x, tile_y, self.tile_size, channel=self.channel
                    )

                    # Load CD31 channel if specified
                    cd31_data = None
                    if self.cd31_channel is not None:
                        cd31_data = self.loader.get_tile(
                            tile_x, tile_y, self.tile_size, channel=self.cd31_channel
                        )

                    # Apply preprocessing
                    preprocessed = self.preprocessor(tile_data, tile_x, tile_y)

                    # Skip tiles that preprocess to None
                    if preprocessed is None:
                        with self._lock:
                            self._tiles_failed += 1
                        continue

                    with self._lock:
                        self._tiles_loaded += 1

                    # Build result tuple with preprocessed data
                    if self.cd31_channel is not None:
                        result = (tile_x, tile_y, preprocessed, cd31_data)
                    else:
                        result = (tile_x, tile_y, preprocessed)

                except Exception as e:
                    logger.warning(
                        f"TilePipelineWithPreprocessing: Failed tile ({tile_x}, {tile_y}): {e}"
                    )
                    with self._lock:
                        self._tiles_failed += 1
                    continue  # Skip failed tiles instead of yielding None

                # Put into queue
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(result, timeout=0.1)
                        break
                    except Full:
                        continue

        except Exception as e:
            logger.error(f"TilePipelineWithPreprocessing producer error: {e}")
            self._exception = e
            try:
                self._queue.put(e, timeout=1.0)
            except Full:
                pass

        finally:
            try:
                self._queue.put(self._SENTINEL, timeout=5.0)
            except Full:
                logger.warning("TilePipelineWithPreprocessing: Could not put sentinel")


def preprocess_tiles_batch(
    tiles: List[T],
    preprocessor: Callable[[T], Any],
    max_workers: int = 4,
    preserve_order: bool = True,
) -> List[Any]:
    """
    Preprocess multiple tiles in parallel using ThreadPoolExecutor.

    Useful for CPU-bound preprocessing before GPU inference, such as:
    - Image normalization
    - Grayscale to RGB conversion
    - Resizing or cropping
    - Feature extraction

    Args:
        tiles: List of tiles to preprocess. Can be any type that the
            preprocessor function accepts.
        preprocessor: Function to apply to each tile. Should be thread-safe
            and CPU-bound (not I/O bound).
        max_workers: Maximum number of worker threads. Default is 4.
            Higher values may not improve performance due to GIL.
        preserve_order: If True, results are returned in the same order
            as input tiles. If False, results may be in completion order.
            Default is True.

    Returns:
        List of preprocessed tiles in the same order as input (if preserve_order=True)
        or in completion order (if preserve_order=False).

    Example:
        >>> def normalize(tile):
        ...     return (tile - tile.mean()) / tile.std()
        >>>
        >>> tiles = [np.random.rand(100, 100) for _ in range(10)]
        >>> normalized = preprocess_tiles_batch(tiles, normalize, max_workers=4)

    Notes:
        - Uses ThreadPoolExecutor, which is suitable for CPU-bound work
          that releases the GIL (like NumPy operations)
        - For I/O-bound preprocessing, consider using TilePipeline instead
        - Exceptions in the preprocessor are propagated to the caller
    """
    if not tiles:
        return []

    if max_workers <= 1 or len(tiles) <= 1:
        # Single-threaded fallback
        return [preprocessor(tile) for tile in tiles]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if preserve_order:
            # Use map to preserve order (blocks until all complete)
            return list(executor.map(preprocessor, tiles))
        else:
            # Use submit + as_completed for completion order
            futures = {executor.submit(preprocessor, tile): i for i, tile in enumerate(tiles)}
            results = [None] * len(tiles)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            return results


def preprocess_tiles_batch_with_coords(
    tile_data_list: List[Tuple[int, int, np.ndarray]],
    preprocessor: Callable[[np.ndarray, int, int], Any],
    max_workers: int = 4,
) -> List[Tuple[int, int, Any]]:
    """
    Preprocess tiles in parallel, passing coordinates to the preprocessor.

    Similar to preprocess_tiles_batch but the preprocessor receives
    (tile_data, tile_x, tile_y) instead of just tile_data. Results
    include the coordinates.

    Args:
        tile_data_list: List of (tile_x, tile_y, tile_data) tuples.
        preprocessor: Function called with (tile_data, tile_x, tile_y).
        max_workers: Maximum worker threads.

    Returns:
        List of (tile_x, tile_y, preprocessed_data) tuples in input order.

    Example:
        >>> def preprocess(tile_data, tile_x, tile_y):
        ...     # Use coordinates for logging or metadata
        ...     logger.debug(f"Preprocessing tile at ({tile_x}, {tile_y})")
        ...     return normalize(tile_data)
        >>>
        >>> tiles = [(0, 0, data1), (3000, 0, data2)]
        >>> results = preprocess_tiles_batch_with_coords(tiles, preprocess)
    """
    if not tile_data_list:
        return []

    def wrapped_preprocessor(item: Tuple[int, int, np.ndarray]) -> Tuple[int, int, Any]:
        tile_x, tile_y, tile_data = item
        result = preprocessor(tile_data, tile_x, tile_y)
        return (tile_x, tile_y, result)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(wrapped_preprocessor, tile_data_list))


class AsyncTileLoader:
    """
    Simple async tile loader that wraps a loader with prefetching.

    This is a simpler alternative to TilePipeline when you just need
    to prefetch tiles without the full iterator interface.

    Example:
        >>> async_loader = AsyncTileLoader(loader, channel=1, num_prefetch=4)
        >>> async_loader.start()
        >>>
        >>> for tile_coord in tiles:
        ...     tile_data = async_loader.get(tile_coord['x'], tile_coord['y'], 3000)
        ...     process(tile_data)
        >>>
        >>> async_loader.stop()
    """

    def __init__(
        self,
        loader: Any,
        channel: int,
        num_prefetch: int = 4,
    ):
        """
        Initialize the async loader.

        Args:
            loader: CZI loader with get_tile() method.
            channel: Channel to load.
            num_prefetch: Number of tiles to prefetch.
        """
        self.loader = loader
        self.channel = channel
        self.num_prefetch = num_prefetch

        self._cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._prefetch_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the prefetch worker thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_func,
            name="AsyncTileLoader-Worker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop the prefetch worker and clear cache."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        with self._cache_lock:
            self._cache.clear()

    def prefetch(self, tile_x: int, tile_y: int, tile_size: int) -> None:
        """
        Request a tile to be prefetched.

        The tile will be loaded in the background and cached.
        """
        self._prefetch_queue.put((tile_x, tile_y, tile_size))

    def get(self, tile_x: int, tile_y: int, tile_size: int) -> Optional[np.ndarray]:
        """
        Get a tile, loading it if not in cache.

        Returns:
            Tile data array, or None if loading failed.
        """
        key = (tile_x, tile_y, tile_size)

        # Check cache first
        with self._cache_lock:
            if key in self._cache:
                return self._cache.pop(key)

        # Load synchronously if not cached
        return self.loader.get_tile(tile_x, tile_y, tile_size, channel=self.channel)

    def _worker_func(self) -> None:
        """Background worker that loads prefetch requests."""
        while not self._stop_event.is_set():
            try:
                item = self._prefetch_queue.get(timeout=0.1)
            except Empty:
                continue

            tile_x, tile_y, tile_size = item
            key = (tile_x, tile_y, tile_size)

            # Skip if already cached
            with self._cache_lock:
                if key in self._cache:
                    continue

            # Load tile
            try:
                tile_data = self.loader.get_tile(
                    tile_x, tile_y, tile_size, channel=self.channel
                )
                with self._cache_lock:
                    # Limit cache size
                    if len(self._cache) >= self.num_prefetch:
                        # Remove oldest entry
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                    self._cache[key] = tile_data
            except Exception as e:
                logger.warning(f"AsyncTileLoader: Failed to prefetch ({tile_x}, {tile_y}): {e}")


__all__ = [
    'TilePipeline',
    'TilePipelineWithPreprocessing',
    'preprocess_tiles_batch',
    'preprocess_tiles_batch_with_coords',
    'AsyncTileLoader',
]
