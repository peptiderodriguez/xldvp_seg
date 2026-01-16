"""
Unified CZI loading utilities with RAM-first caching.

Provides consistent interface for loading CZI files either:
- On-demand (lower memory, slower for many tiles)
- Into RAM (higher memory, much faster for network mounts)

Features:
- Multi-channel support: load multiple channels at once
- Singleton/global cache: get_loader() returns existing loader if already loaded
- Lazy channel loading: channels loaded on first access
- Memory reporting: track RAM usage
"""

import logging
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from aicspylibczi import CziFile

logger = logging.getLogger(__name__)

# Global cache for CZILoader instances with thread-safe access
_image_cache: Dict[str, 'CZILoader'] = {}
_image_cache_lock = threading.Lock()


def get_loader(
    czi_path: Union[str, Path],
    load_to_ram: bool = True,
    channels: Optional[List[int]] = None,
    channel: Optional[int] = None,
    strip_height: int = 5000,
    quiet: bool = False
) -> 'CZILoader':
    """
    Get or create a CZILoader for this path.

    Returns existing loader if already loaded for this path.

    Args:
        czi_path: Path to CZI file
        load_to_ram: If True, load channels into RAM
        channels: List of channels to load (for multi-channel)
        channel: Single channel to load (backward compatibility)
        strip_height: Height of strips for RAM loading
        quiet: Suppress progress output

    Returns:
        CZILoader instance (new or existing)
    """
    key = str(Path(czi_path).resolve())

    with _image_cache_lock:
        if key in _image_cache:
            existing = _image_cache[key]
            logger.debug(f"Returning cached loader for {Path(czi_path).name}")

            # If additional channels requested, load them lazily
            if load_to_ram:
                if channels:
                    for ch in channels:
                        if ch not in existing.loaded_channels:
                            existing.load_channel(ch, strip_height=strip_height)
                elif channel is not None and channel not in existing.loaded_channels:
                    existing.load_channel(channel, strip_height=strip_height)

            return existing

        logger.debug(f"Creating new loader for {Path(czi_path).name}")
        loader = CZILoader(
            czi_path,
            load_to_ram=load_to_ram,
            channels=channels,
            channel=channel,
            strip_height=strip_height,
            quiet=quiet
        )
        _image_cache[key] = loader
        return loader


def clear_cache():
    """Release all cached images and loaders."""
    global _image_cache
    with _image_cache_lock:
        for key in list(_image_cache.keys()):
            loader = _image_cache[key]
            loader.close()
        _image_cache.clear()
    logger.info("Image cache cleared")


def get_cached_paths() -> List[str]:
    """Return list of paths currently in the cache."""
    with _image_cache_lock:
        return list(_image_cache.keys())


class CZILoader:
    """
    Unified CZI loader with optional RAM caching and multi-channel support.

    Usage:
        # On-demand loading (default)
        loader = CZILoader(czi_path)
        tile = loader.get_tile(tile_x, tile_y, tile_size, channel=0)

        # Single channel RAM loading (backward compatible)
        loader = CZILoader(czi_path, load_to_ram=True, channel=1)
        tile = loader.get_tile(tile_x, tile_y, tile_size)  # channel already loaded

        # Multi-channel RAM loading
        loader = CZILoader(czi_path, load_to_ram=True, channels=[0, 1, 2])
        tile_ch0 = loader.get_tile(tile_x, tile_y, tile_size, channel=0)  # From RAM
        tile_ch1 = loader.get_tile(tile_x, tile_y, tile_size, channel=1)  # From RAM

        # Using global cache (recommended)
        loader = get_loader(czi_path, load_to_ram=True, channels=[0, 1])
        # ... later in code ...
        same_loader = get_loader(czi_path)  # Returns existing instance
    """

    def __init__(
        self,
        czi_path: Union[str, Path],
        load_to_ram: bool = False,
        channel: Optional[int] = None,
        channels: Optional[List[int]] = None,
        strip_height: int = 5000,
        quiet: bool = False
    ):
        """
        Initialize CZI loader.

        Args:
            czi_path: Path to CZI file
            load_to_ram: If True, load channel(s) into RAM
            channel: Single channel to load (backward compatibility)
            channels: List of channels to load (for multi-channel support)
            strip_height: Height of strips for RAM loading (memory optimization)
            quiet: Suppress progress output

        Raises:
            FileNotFoundError: If czi_path does not exist
        """
        self.czi_path = Path(czi_path)

        # Issue #12: Validate CZI path exists before attempting to open
        if not self.czi_path.exists():
            raise FileNotFoundError(f"CZI file not found: {self.czi_path}")
        if not self.czi_path.is_file():
            raise FileNotFoundError(f"CZI path is not a file: {self.czi_path}")

        self.reader = CziFile(str(self.czi_path))
        self.quiet = quiet
        self._strip_height = strip_height

        # Get mosaic info
        self.bbox = self.reader.get_mosaic_scene_bounding_box()
        self.x_start = self.bbox.x
        self.y_start = self.bbox.y
        self.width = self.bbox.w
        self.height = self.bbox.h

        # Multi-channel RAM data storage
        self._channel_data: Dict[int, np.ndarray] = {}

        # Backward compatibility: single channel_data property
        self._primary_channel: Optional[int] = None

        if load_to_ram:
            # Handle both single channel and multi-channel
            channels_to_load = []

            if channels is not None:
                channels_to_load = list(channels)
            elif channel is not None:
                channels_to_load = [channel]
            else:
                raise ValueError("channel or channels is required when load_to_ram=True")

            for ch in channels_to_load:
                self._load_channel_to_ram(ch, strip_height)

            # Set primary channel for backward compatibility
            if channels_to_load:
                self._primary_channel = channels_to_load[0]

    def _load_channel_to_ram(self, channel: int, strip_height: int):
        """Load a single channel into RAM."""
        if channel in self._channel_data:
            logger.debug(f"Channel {channel} already loaded, skipping")
            return

        logger.info(f"Loading channel {channel} into RAM ({self.width:,} x {self.height:,} px)...")

        n_strips = (self.height + strip_height - 1) // strip_height
        channel_array = np.empty((self.height, self.width), dtype=np.uint16)

        iterator = range(n_strips)
        if not self.quiet:
            iterator = tqdm(iterator, desc=f"Loading ch{channel}")

        for i in iterator:
            y_off = i * strip_height
            h = min(strip_height, self.height - y_off)
            strip = self.reader.read_mosaic(
                region=(self.x_start, self.y_start + y_off, self.width, h),
                scale_factor=1,
                C=channel
            )
            channel_array[y_off:y_off+h, :] = np.squeeze(strip)

        self._channel_data[channel] = channel_array

        # Set primary channel if not set
        if self._primary_channel is None:
            self._primary_channel = channel

        logger.info(f"Channel {channel} loaded: {self._get_array_memory_gb(channel_array):.2f} GB")

    def load_channel(self, channel: int, strip_height: Optional[int] = None):
        """
        Load a channel to RAM if not already loaded (lazy loading).

        Args:
            channel: Channel number to load
            strip_height: Height of strips for loading (default: use init value)
        """
        if channel in self._channel_data:
            logger.debug(f"Channel {channel} already loaded")
            return

        sh = strip_height if strip_height is not None else self._strip_height
        self._load_channel_to_ram(channel, sh)

    @property
    def loaded_channels(self) -> List[int]:
        """Return list of channels currently loaded in RAM."""
        return list(self._channel_data.keys())

    @property
    def channel_data(self) -> Optional[np.ndarray]:
        """
        Backward compatibility: return primary channel data.

        Returns the first loaded channel's data, or None if no channels loaded.
        """
        if self._primary_channel is not None and self._primary_channel in self._channel_data:
            return self._channel_data[self._primary_channel]
        return None

    @channel_data.setter
    def channel_data(self, value):
        """Backward compatibility: set primary channel data."""
        if value is None:
            # Clear all channel data
            self._channel_data.clear()
            self._primary_channel = None
        else:
            # This shouldn't be used directly, but support it for compatibility
            if self._primary_channel is not None:
                self._channel_data[self._primary_channel] = value
            else:
                self._channel_data[0] = value
                self._primary_channel = 0

    @property
    def loaded_channel(self) -> Optional[int]:
        """Backward compatibility: return primary loaded channel."""
        return self._primary_channel

    @loaded_channel.setter
    def loaded_channel(self, value):
        """Backward compatibility setter."""
        self._primary_channel = value

    def get_channel_data(self, channel: int) -> Optional[np.ndarray]:
        """
        Get the RAM data for a specific channel.

        Args:
            channel: Channel number

        Returns:
            numpy array with channel data, or None if not loaded
        """
        return self._channel_data.get(channel)

    def _get_array_memory_gb(self, arr: np.ndarray) -> float:
        """Calculate memory usage of a numpy array in GB."""
        return arr.nbytes / (1024 ** 3)

    def memory_usage_gb(self) -> float:
        """
        Return approximate memory usage in GB for all loaded channels.

        Returns:
            Total memory usage in GB
        """
        total_bytes = sum(arr.nbytes for arr in self._channel_data.values())
        return total_bytes / (1024 ** 3)

    def memory_info(self) -> Dict:
        """
        Return detailed memory usage information.

        Returns:
            Dict with per-channel and total memory usage
        """
        info = {
            'total_gb': self.memory_usage_gb(),
            'channels': {}
        }
        for ch, arr in self._channel_data.items():
            info['channels'][ch] = {
                'gb': self._get_array_memory_gb(arr),
                'shape': arr.shape,
                'dtype': str(arr.dtype)
            }
        return info

    def get_tile(
        self,
        tile_x: int,
        tile_y: int,
        tile_size: int,
        channel: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get a tile from the CZI.

        Args:
            tile_x: Tile X origin (in global mosaic coords)
            tile_y: Tile Y origin (in global mosaic coords)
            tile_size: Size of tile (square)
            channel: Channel to read. If None and data loaded to RAM, uses primary channel.
                     Required if reading from disk.

        Returns:
            2D numpy array with tile data, or None if tile is empty/invalid
        """
        # Determine which channel to use
        target_channel = channel

        # Check if we have this channel in RAM
        if target_channel is not None and target_channel in self._channel_data:
            return self._get_tile_from_ram(tile_x, tile_y, tile_size, target_channel)

        # If no channel specified, try primary channel
        if target_channel is None and self._primary_channel is not None:
            return self._get_tile_from_ram(tile_x, tile_y, tile_size, self._primary_channel)

        # Fall back to reading from CZI
        if target_channel is None:
            raise ValueError("channel is required when data not loaded to RAM")

        return self._get_tile_from_czi(tile_x, tile_y, tile_size, target_channel)

    def _get_tile_from_ram(
        self,
        tile_x: int,
        tile_y: int,
        tile_size: int,
        channel: int
    ) -> Optional[np.ndarray]:
        """Extract a tile from RAM-loaded channel data."""
        data = self._channel_data.get(channel)
        if data is None:
            logger.warning(f"Channel {channel} not loaded, falling back to CZI read")
            return self._get_tile_from_czi(tile_x, tile_y, tile_size, channel)

        rel_x = tile_x - self.x_start
        rel_y = tile_y - self.y_start

        # Bounds check
        if rel_x < 0 or rel_y < 0:
            return None
        if rel_x >= self.width or rel_y >= self.height:
            return None

        y2 = min(rel_y + tile_size, self.height)
        x2 = min(rel_x + tile_size, self.width)

        tile_data = data[rel_y:y2, rel_x:x2]
        if tile_data.size == 0:
            return None
        return tile_data

    def _get_tile_from_czi(
        self,
        tile_x: int,
        tile_y: int,
        tile_size: int,
        channel: int
    ) -> Optional[np.ndarray]:
        """Read a tile directly from the CZI file."""
        tile_data = self.reader.read_mosaic(
            region=(tile_x, tile_y, tile_size, tile_size),
            scale_factor=1,
            C=channel
        )

        if tile_data is None or tile_data.size == 0:
            return None

        tile_data = np.squeeze(tile_data)
        if tile_data.ndim != 2:
            return None

        return tile_data

    def get_pixel_size(self) -> float:
        """
        Get pixel size in micrometers.

        Returns:
            Pixel size in um/px (default 0.22 if not found in metadata)
        """
        pixel_size = 0.22  # Default
        try:
            metadata = self.reader.meta
            scaling = metadata.find('.//Scaling/Items/Distance[@Id="X"]/Value')
            if scaling is not None:
                pixel_size = float(scaling.text) * 1e6
        except Exception as e:
            logger.debug(f"Could not read pixel size from metadata: {e}")
        return pixel_size

    @property
    def slide_name(self) -> str:
        """Get slide name from file path."""
        return self.czi_path.stem

    @property
    def mosaic_size(self) -> Tuple[int, int]:
        """Get mosaic dimensions (width, height)."""
        return (self.width, self.height)

    @property
    def mosaic_origin(self) -> Tuple[int, int]:
        """Get mosaic origin (x_start, y_start)."""
        return (self.x_start, self.y_start)

    def close(self):
        """Release resources."""
        self._channel_data.clear()
        self._primary_channel = None
        logger.debug(f"Closed loader for {self.slide_name}")
        # CziFile doesn't have explicit close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        channels_str = ', '.join(str(c) for c in self.loaded_channels) if self.loaded_channels else 'none'
        return (
            f"CZILoader('{self.slide_name}', "
            f"size={self.width}x{self.height}, "
            f"loaded_channels=[{channels_str}], "
            f"memory={self.memory_usage_gb():.2f}GB)"
        )
