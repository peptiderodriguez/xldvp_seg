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

import gc
import threading
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from aicspylibczi import CziFile

import re as _re

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


class ChannelResolutionError(ValueError):
    """Raised when a channel spec cannot be resolved to a CZI channel index."""


def parse_markers_from_filename(filename: str) -> list:
    """Extract antibody/marker to wavelength mappings from a CZI filename.

    Handles patterns:
      - SMA647  -> {"name": "SMA", "wavelength": 647}
      - nuc488  -> {"name": "nuc", "wavelength": 488}
      - CD31_555 -> {"name": "CD31", "wavelength": 555}
      - 488Slc17a7 -> {"name": "Slc17a7", "wavelength": 488}
      - PM750   -> {"name": "PM", "wavelength": 750}
      - NeuN647 -> {"name": "NeuN", "wavelength": 647}
      - tdTom555 -> {"name": "tdTom", "wavelength": 555}
      - Bgtx647  -> {"name": "Bgtx", "wavelength": 647}

    Returns list of dicts sorted by position in filename.
    """
    stem = Path(filename).stem if '.' in filename else filename
    results = []
    seen_positions = set()

    # Pattern 1: name followed by wavelength (e.g., SMA647, nuc488, NeuN647, tdTom555)
    for m in _re.finditer(r'(?<![0-9])([A-Za-z][A-Za-z0-9]*?)(\d{3})(?![0-9])', stem):
        name, wl = m.group(1), int(m.group(2))
        if 350 <= wl <= 900:  # valid fluorescence wavelength range
            results.append({'name': name, 'wavelength': wl, '_pos': m.start()})
            seen_positions.add(m.start())

    # Pattern 2: name_wavelength (e.g., CD31_555)
    # Negative lookahead prevents matching NAME_WL when WL is followed by a letter
    # (e.g., Slc17a7_647Gad1 — 647 belongs to Gad1 via Pattern 3, not to Slc17a7)
    for m in _re.finditer(r'([A-Za-z][A-Za-z0-9]*)_(\d{3})(?![0-9A-Za-z])', stem):
        if m.start() not in seen_positions:
            name, wl = m.group(1), int(m.group(2))
            if 350 <= wl <= 900:
                results.append({'name': name, 'wavelength': wl, '_pos': m.start()})
                seen_positions.add(m.start())

    # Pattern 3: wavelength before name (e.g., 488Slc17a7)
    for m in _re.finditer(r'(?<![0-9A-Za-z])(\d{3})([A-Z][A-Za-z0-9]+)', stem):
        if m.start() not in seen_positions:
            wl, name = int(m.group(1)), m.group(2)
            if 350 <= wl <= 900:
                results.append({'name': name, 'wavelength': wl, '_pos': m.start()})
                seen_positions.add(m.start())

    # Sort by position in filename, strip internal _pos key
    results.sort(key=lambda x: x['_pos'])
    for r in results:
        del r['_pos']

    return results


def resolve_channel_indices(
    czi_metadata: dict,
    marker_specs: list,
    filename: str = None,
) -> dict:
    """Resolve marker names/wavelengths to CZI channel indices.

    2-step lookup:
      1. If spec is an integer index, validate it exists
      2. If spec is a wavelength (3-digit number), match to CZI metadata
      3. If spec is a name, use filename to get wavelength, then match

    Args:
        czi_metadata: From get_czi_metadata() — has channels[].excitation_nm
        marker_specs: List like ["SMA", "CD31"] or ["647", "555"] or ["1", "3"]
        filename: CZI filename for name→wavelength lookup (optional)

    Returns:
        Dict mapping each spec to its resolved channel index:
        {"SMA": 1, "CD31": 3} or {"647": 1, "555": 3}

    Raises:
        ChannelResolutionError: If wavelength not found or ambiguous
    """
    channels = czi_metadata.get('channels', [])
    n_channels = czi_metadata.get('n_channels', len(channels))

    # Build wavelength→index lookup from CZI metadata
    wl_to_idx = {}
    for ch in channels:
        ex = ch.get('excitation_nm')
        if ex is not None:
            wl_to_idx[float(ex)] = ch['index']

    # Parse filename markers if provided
    filename_markers = {}
    if filename:
        parsed = parse_markers_from_filename(filename)
        for entry in parsed:
            filename_markers[entry['name'].lower()] = entry['wavelength']

    def _available_channels_str():
        """Format available channels for error messages."""
        parts = []
        for ch in channels:
            ex = ch.get('excitation_nm')
            ex_str = f"{ex:.0f}nm" if ex else "N/A"
            parts.append(f"ch{ch['index']}={ex_str} ({ch.get('name', '?')})")
        return ', '.join(parts) if parts else '(no channel metadata)'

    def _match_wavelength(target_wl: float) -> int:
        """Find channel index matching a target wavelength (±10nm tolerance)."""
        exact = wl_to_idx.get(target_wl)
        if exact is not None:
            return exact

        # Fuzzy match within ±10nm
        matches = []
        for wl, idx in wl_to_idx.items():
            if abs(wl - target_wl) <= 10:
                matches.append((abs(wl - target_wl), idx, wl))

        if len(matches) == 1:
            _, idx, actual_wl = matches[0]
            logger.debug(f"Fuzzy wavelength match: {target_wl}nm -> {actual_wl}nm (ch{idx})")
            return idx
        elif len(matches) > 1:
            matches.sort()
            raise ChannelResolutionError(
                f"Ambiguous wavelength {target_wl}nm: matches multiple channels "
                f"{[(wl, f'ch{idx}') for _, idx, wl in matches]}. "
                f"Available: {_available_channels_str()}"
            )
        else:
            raise ChannelResolutionError(
                f"No channel found for wavelength {target_wl:.0f}nm. "
                f"Available: {_available_channels_str()}"
            )

    resolved = {}
    for spec in marker_specs:
        spec_str = str(spec).strip()

        # 1. Try as integer index
        try:
            idx = int(spec_str)
            # Could be an index (0-5) or a wavelength (400-900)
            if 0 <= idx < n_channels and idx < 100:
                # Looks like a channel index
                resolved[spec_str] = idx
                logger.debug(f"Channel spec '{spec_str}' -> index {idx} (integer passthrough)")
                continue
            # If it's a 3-digit number in wavelength range, treat as wavelength
            if 350 <= idx <= 900:
                resolved[spec_str] = _match_wavelength(float(idx))
                logger.debug(f"Channel spec '{spec_str}' -> wavelength {idx}nm -> ch{resolved[spec_str]}")
                continue
            # Otherwise treat as index if valid
            if 0 <= idx < n_channels:
                resolved[spec_str] = idx
                continue
            raise ChannelResolutionError(
                f"Channel index {idx} out of range (0..{n_channels - 1}). "
                f"Available: {_available_channels_str()}"
            )
        except ValueError:
            pass

        # 2. Try as marker name via filename lookup
        name_lower = spec_str.lower()
        if name_lower in filename_markers:
            wl = filename_markers[name_lower]
            resolved[spec_str] = _match_wavelength(float(wl))
            logger.debug(
                f"Channel spec '{spec_str}' -> {wl}nm (from filename) -> ch{resolved[spec_str]}"
            )
            continue

        # 3. Try as marker name matching CZI channel metadata names
        for ch in channels:
            ch_name = (ch.get('name') or '').lower()
            ch_dye = (ch.get('dye') or '').lower()
            ch_fluor = (ch.get('fluorophore') or '').lower()
            if name_lower in (ch_name, ch_dye, ch_fluor):
                resolved[spec_str] = ch['index']
                logger.debug(
                    f"Channel spec '{spec_str}' -> ch{ch['index']} (matched CZI metadata name)"
                )
                break
        else:
            raise ChannelResolutionError(
                f"Cannot resolve channel spec '{spec_str}'. "
                f"Not a valid index, wavelength, or known marker name. "
                f"Available: {_available_channels_str()}"
                + (f"\nFilename markers: {filename_markers}" if filename_markers else "")
            )

    return resolved


def _squeeze_batch_dims(arr: np.ndarray) -> np.ndarray:
    """Remove leading singleton (batch) dimensions, preserving 2D or 3D (H, W, 3) spatial data.

    Handles shapes like (1, H, W), (1, H, W, 3), and (1, 1, H, W).
    """
    while arr.ndim > 3 or (arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] != 3):
        if arr.shape[0] != 1:
            break
        arr = arr.squeeze(axis=0)
    return arr


# Global cache for CZILoader instances with thread-safe access
_image_cache: Dict[str, 'CZILoader'] = {}
_image_cache_lock = threading.Lock()


def get_loader(
    czi_path: Union[str, Path],
    load_to_ram: bool = True,
    channels: Optional[List[int]] = None,
    channel: Optional[int] = None,
    strip_height: int = 5000,
    quiet: bool = False,
    scene: int = 0
) -> 'CZILoader':
    """
    Get or create a CZILoader for this path.

    Returns existing loader if already loaded for this path+scene.

    Args:
        czi_path: Path to CZI file
        load_to_ram: If True, load channels into RAM
        channels: List of channels to load (for multi-channel)
        channel: Single channel to load (backward compatibility)
        strip_height: Height of strips for RAM loading
        quiet: Suppress progress output
        scene: CZI scene index (0-based, default 0)

    Returns:
        CZILoader instance (new or existing)
    """
    key = f"{Path(czi_path).resolve()}:S{scene}"

    # Look up cache under lock (fast), but do expensive channel loading outside it
    existing = None
    with _image_cache_lock:
        if key in _image_cache:
            existing = _image_cache[key]
            logger.debug(f"Returning cached loader for {Path(czi_path).name}")

    if existing is not None:
        # Load additional channels outside cache lock (per-loader lock prevents double-loading)
        if load_to_ram:
            channels_needed = []
            if channels:
                channels_needed = [ch for ch in channels if ch not in existing.loaded_channels]
            elif channel is not None and channel not in existing.loaded_channels:
                channels_needed = [channel]

            if channels_needed:
                with existing._load_lock:
                    for ch in channels_needed:
                        if ch not in existing.loaded_channels:
                            existing.load_channel(ch, strip_height=strip_height)

        return existing

    # Create new loader outside the lock to avoid blocking other threads
    # Use a threading.Event as a sentinel so concurrent callers wait
    with _image_cache_lock:
        # Double-check after re-acquiring lock
        cached = _image_cache.get(key)
        if cached is not None:
            if isinstance(cached, threading.Event):
                # Another thread is already building this loader — wait below
                event = cached
            else:
                return cached
        else:
            # We will build it — insert event as placeholder
            event = threading.Event()
            _image_cache[key] = event
            cached = None  # signals we are the builder

    if cached is not None:
        # Another thread is building — wait for it to finish
        event.wait()
        with _image_cache_lock:
            result = _image_cache.get(key)
            if result is None or isinstance(result, threading.Event):
                raise RuntimeError(f"Loader construction failed for {key}")
            return result

    # We are the builder — construct outside the lock
    try:
        logger.debug(f"Creating new loader for {Path(czi_path).name} scene={scene}")
        loader = CZILoader(
            czi_path,
            load_to_ram=load_to_ram,
            channels=channels,
            channel=channel,
            strip_height=strip_height,
            quiet=quiet,
            scene=scene
        )
        with _image_cache_lock:
            _image_cache[key] = loader
        event.set()
        return loader
    except Exception:
        with _image_cache_lock:
            del _image_cache[key]
        event.set()
        raise


def clear_cache():
    """Release all cached images and loaders."""
    global _image_cache
    with _image_cache_lock:
        for key in list(_image_cache.keys()):
            loader = _image_cache[key]
            if isinstance(loader, threading.Event):
                continue  # Skip sentinels from in-progress construction
            loader.close()
        _image_cache.clear()
    logger.info("Image cache cleared")


def get_cached_paths() -> List[str]:
    """Return list of paths currently in the cache."""
    with _image_cache_lock:
        return list(_image_cache.keys())


def get_czi_metadata(czi_path, scene: int = 0):
    """
    Extract metadata from CZI file without loading image data.

    Args:
        czi_path: Path to CZI file
        scene: Scene index (0-based, default 0). Pass None for global bbox.

    Returns dict with:
        - channels: list of channel info (name, wavelength, fluor)
        - pixel_size_um: pixel size in microns
        - mosaic_size: (width, height) in pixels for the specified scene
        - n_channels: number of channels
        - n_scenes: number of scenes in the CZI
    """
    import xml.etree.ElementTree as ET

    czi_path = str(czi_path)
    metadata = {
        'channels': [],
        'pixel_size_um': 0.22,  # default
        'mosaic_size': None,
        'n_channels': 0,
        'n_scenes': 1,
    }

    n_data_channels = None  # actual channel count from file dimensions

    # Try pylibCZIrw first (better for large files)
    try:
        from pylibCZIrw import czi as pylibczi

        with pylibczi.open_czi(czi_path) as czidoc:
            # Get per-scene dimensions if available
            try:
                scene_bbox = czidoc.scenes_bounding_rectangle
                if scene_bbox and len(scene_bbox) > 0:
                    metadata['n_scenes'] = len(scene_bbox)
                    if scene is not None and scene in scene_bbox:
                        rect = scene_bbox[scene]
                        metadata['mosaic_size'] = (rect.w, rect.h)
                    else:
                        # Fallback to total bounding box
                        dims = czidoc.total_bounding_box
                        metadata['mosaic_size'] = (dims['X'][1] - dims['X'][0], dims['Y'][1] - dims['Y'][0])
                else:
                    dims = czidoc.total_bounding_box
                    metadata['mosaic_size'] = (dims['X'][1] - dims['X'][0], dims['Y'][1] - dims['Y'][0])
            except (AttributeError, Exception):
                # Older pylibCZIrw without scenes_bounding_rectangle
                dims = czidoc.total_bounding_box
                metadata['mosaic_size'] = (dims['X'][1] - dims['X'][0], dims['Y'][1] - dims['Y'][0])

            # Get metadata XML
            meta_xml = czidoc.raw_metadata
            root = ET.fromstring(meta_xml)

    except ImportError:
        # Fall back to aicspylibczi
        reader = CziFile(czi_path)

        # Query scene count and channel count from dims_shape
        # dims_shape returns a list with one entry per scene
        n_data_channels = None
        try:
            dims_shape_list = reader.get_dims_shape()
            metadata['n_scenes'] = len(dims_shape_list)
            if 'C' in dims_shape_list[0]:
                n_data_channels = dims_shape_list[0]['C'][1] - dims_shape_list[0]['C'][0]
        except Exception:
            pass

        # Get per-scene bounding box
        if scene is not None:
            bbox = reader.get_mosaic_scene_bounding_box(index=scene)
        else:
            bbox = reader.get_mosaic_bounding_box()
        metadata['mosaic_size'] = (bbox.w, bbox.h)

        meta_xml = reader.meta
        if isinstance(meta_xml, str):
            root = ET.fromstring(meta_xml)
        else:
            root = meta_xml

    # Parse XML for channel info.
    # CZI XML may contain duplicate <Channel> entries across sections (e.g.
    # EDFvar files have 3x5=15 entries for 5 actual channels). We collect
    # all entries indexed by Channel:N id, merging metadata from the richest
    # entry per index. Then limit to actual data channel count from dims_shape.
    channel_map = {}  # index -> ch_info dict
    for channel in root.iter('Channel'):
        ch_id = channel.get('Id', '')
        name = channel.get('Name', '')

        # Extract channel index from Id like "Channel:0", "Channel:1", etc.
        ch_idx = None
        if ch_id and ch_id.startswith('Channel:'):
            try:
                ch_idx = int(ch_id.split(':')[1])
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse channel index from '{ch_id}': {e}")

        fluor = channel.find('.//Fluor')
        fluor_text = fluor.text if fluor is not None else 'N/A'
        emission = channel.find('.//EmissionWavelength')
        emission_nm = float(emission.text) if emission is not None and emission.text else None
        excitation = channel.find('.//ExcitationWavelength')
        excitation_nm = float(excitation.text) if excitation is not None and excitation.text else None
        dye_name = channel.find('.//DyeName')
        dye = dye_name.text if dye_name is not None else fluor_text

        if ch_idx is not None:
            existing = channel_map.get(ch_idx)
            if existing is None:
                channel_map[ch_idx] = {
                    'index': ch_idx, 'name': name, 'id': ch_id,
                    'fluorophore': fluor_text, 'emission_nm': emission_nm,
                    'excitation_nm': excitation_nm, 'dye': dye,
                }
            else:
                # Merge: prefer non-null/non-default values
                if fluor_text and fluor_text != 'N/A' and existing['fluorophore'] == 'N/A':
                    existing['fluorophore'] = fluor_text
                if emission_nm and not existing['emission_nm']:
                    existing['emission_nm'] = emission_nm
                if excitation_nm and not existing['excitation_nm']:
                    existing['excitation_nm'] = excitation_nm
                if dye and dye != 'N/A' and existing['dye'] in ('N/A', existing['fluorophore']):
                    existing['dye'] = dye

    # Build sorted channel list
    channels = [channel_map[k] for k in sorted(channel_map.keys())]

    # Limit to actual data channel count if known (from dims_shape)
    if n_data_channels is not None and len(channels) > n_data_channels:
        channels = channels[:n_data_channels]

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


def print_czi_metadata(czi_path, scene: int = 0):
    """Print CZI metadata in human-readable format."""
    logger.info(f"CZI Metadata: {Path(czi_path).name}")
    logger.info("=" * 60)

    try:
        meta = get_czi_metadata(czi_path, scene=scene)

        if meta['mosaic_size']:
            w, h = meta['mosaic_size']
            logger.info(f"Mosaic size: {w:,} x {h:,} px")

        logger.info(f"Pixel size: {meta['pixel_size_um']:.4f} um/px")
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
        quiet: bool = False,
        scene: int = 0
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
            scene: CZI scene index (0-based, default 0)

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
        self.scene = scene
        self.quiet = quiet
        self._strip_height = strip_height
        self._load_lock = threading.Lock()

        try:
            # Get mosaic info for the selected scene
            self.bbox = self.reader.get_mosaic_scene_bounding_box(index=self.scene)
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
        except Exception:
            del self.reader
            self.reader = None
            raise

    def _load_channel_to_ram(self, channel: int, strip_height: int):
        """Load a single channel into RAM."""
        if channel in self._channel_data:
            logger.debug(f"Channel {channel} already loaded, skipping")
            return

        if strip_height <= 0:
            raise ValueError(f"strip_height must be positive, got {strip_height}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"Mosaic dimensions must be positive, got {self.width}x{self.height}"
            )

        logger.info(f"Loading channel {channel} into RAM ({self.width:,} x {self.height:,} px)...")

        n_strips = (self.height + strip_height - 1) // strip_height

        # Read a small test strip to detect if this is RGB data
        # Scene selection is implicit: self.x_start/y_start come from
        # get_mosaic_scene_bounding_box(index=self.scene), so the region
        # coordinates naturally read from the correct scene.
        test_strip = self.reader.read_mosaic(
            region=(self.x_start, self.y_start, min(100, self.width), min(100, self.height)),
            scale_factor=1,
            C=channel,
        )
        test_strip = _squeeze_batch_dims(test_strip)
        is_rgb = len(test_strip.shape) == 3 and test_strip.shape[-1] == 3

        if is_rgb:
            logger.info(f"  Detected RGB data (shape: {test_strip.shape})")
            channel_array = np.empty((self.height, self.width, 3), dtype=np.uint8)
        else:
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
                C=channel,
            )
            strip = _squeeze_batch_dims(strip)
            if is_rgb:
                # Ensure RGB data is uint8 using dtype-based normalization
                # (per-strip max normalization causes visible banding at strip boundaries)
                if strip.dtype == np.uint16:
                    strip = (strip >> 8).astype(np.uint8)  # Fast bit-shift, consistent across all strips
                elif strip.dtype not in (np.uint8,):
                    strip = np.clip(strip, 0, 255).astype(np.uint8)
                channel_array[y_off:y_off+h, :, :] = strip
            else:
                channel_array[y_off:y_off+h, :] = strip

        self._channel_data[channel] = channel_array

        # Set primary channel if not set
        if self._primary_channel is None:
            self._primary_channel = channel

        logger.info(f"Channel {channel} loaded: {self._get_array_memory_gb(channel_array):.2f} GB")

    def load_to_shared_memory(
        self,
        channel: int,
        shm_buffer: np.ndarray,
        strip_height: int = 5000
    ) -> None:
        """
        Load a CZI channel directly into a pre-allocated shared memory buffer.

        Similar to _load_channel_to_ram() but writes directly into the provided
        buffer instead of allocating a new numpy array. Useful for multi-process
        scenarios where the buffer is backed by shared memory.

        Args:
            channel: Channel number to load
            shm_buffer: Pre-allocated numpy array to write into. Must have shape
                        (height, width) for grayscale or (height, width, 3) for RGB,
                        matching the mosaic dimensions.
            strip_height: Height of strips for loading (memory optimization)

        Raises:
            ValueError: If shm_buffer shape doesn't match mosaic dimensions
        """
        if strip_height <= 0:
            raise ValueError(f"strip_height must be positive, got {strip_height}")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"Mosaic dimensions must be positive, got {self.width}x{self.height}"
            )

        # Validate buffer dimensions
        expected_2d = (self.height, self.width)
        expected_rgb = (self.height, self.width, 3)

        if shm_buffer.shape != expected_2d and shm_buffer.shape != expected_rgb:
            raise ValueError(
                f"shm_buffer shape {shm_buffer.shape} does not match expected "
                f"{expected_2d} (grayscale) or {expected_rgb} (RGB)"
            )

        is_rgb_buffer = len(shm_buffer.shape) == 3 and shm_buffer.shape[-1] == 3

        logger.info(
            f"Loading channel {channel} to shared memory ({self.width:,} x {self.height:,} px)..."
        )

        n_strips = (self.height + strip_height - 1) // strip_height

        is_rgb_data = self.is_channel_rgb(channel)

        if is_rgb_data and not is_rgb_buffer:
            raise ValueError(
                f"CZI channel {channel} contains RGB data but shm_buffer is 2D. "
                f"Provide a buffer with shape {expected_rgb}."
            )
        if not is_rgb_data and is_rgb_buffer:
            raise ValueError(
                f"CZI channel {channel} contains grayscale data but shm_buffer is RGB. "
                f"Provide a buffer with shape {expected_2d}."
            )

        if is_rgb_data:
            logger.info(f"  Detected RGB data for channel {channel}")

        iterator = range(n_strips)
        if not self.quiet:
            iterator = tqdm(iterator, desc=f"Loading ch{channel} to shared memory")

        for i in iterator:
            y_off = i * strip_height
            h = min(strip_height, self.height - y_off)

            strip = self.reader.read_mosaic(
                region=(self.x_start, self.y_start + y_off, self.width, h),
                scale_factor=1,
                C=channel,
            )
            strip = _squeeze_batch_dims(strip)

            # Validate shape after squeeze to catch dimension issues early
            if is_rgb_data:
                if strip.ndim != 3 or strip.shape[2] != 3:
                    raise ValueError(f"Expected RGB strip shape (h, w, 3), got {strip.shape}")
            else:
                if strip.ndim != 2:
                    raise ValueError(f"Expected grayscale strip shape (h, w), got {strip.shape}")

            if is_rgb_data:
                # Ensure RGB data is uint8 using dtype-based normalization
                # (per-strip max normalization causes visible banding at strip boundaries)
                if strip.dtype == np.uint16:
                    strip = (strip >> 8).astype(np.uint8)  # Fast bit-shift, consistent across all strips
                elif strip.dtype not in (np.uint8,):
                    strip = np.clip(strip, 0, 255).astype(np.uint8)
                shm_buffer[y_off:y_off + h, :, :] = strip
            else:
                shm_buffer[y_off:y_off + h, :] = strip

            # Free memory after each strip (del is sufficient for non-cyclic numpy arrays)
            del strip

            # Log memory status every 10 strips
            if (i + 1) % 10 == 0:
                gc.collect()  # Only GC periodically, not every strip
                from segmentation.processing.memory import log_memory_status
                log_memory_status(f"Loaded strip {i + 1}/{n_strips}")

        logger.info(
            f"Channel {channel} loaded to shared memory: "
            f"{shm_buffer.nbytes / (1024 ** 3):.2f} GB"
        )

    def is_channel_rgb(self, channel: int) -> bool:
        """
        Check if a channel contains RGB data by probing a small region.

        Args:
            channel: Channel number to check

        Returns:
            True if the channel contains RGB data (shape ends with 3), False for grayscale
        """
        # Read a small test region
        test_region = self.reader.read_mosaic(
            region=(self.x_start, self.y_start, min(100, self.width), min(100, self.height)),
            scale_factor=1,
            C=channel,
        )
        test_region = _squeeze_batch_dims(test_region)
        return len(test_region.shape) == 3 and test_region.shape[-1] == 3

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

    def set_channel_data(self, channel: int, data: np.ndarray) -> None:
        """Set the RAM data for a specific channel.

        Use this to update channel data after preprocessing (e.g., illumination
        correction) without accessing the private _channel_data dict directly.

        Args:
            channel: Channel number
            data: numpy array with channel data
        """
        self._channel_data[channel] = data

    def clear_all_channels(self) -> None:
        """Release all loaded channel data from RAM.

        Clears the internal channel data dict. Use this instead of accessing
        _channel_data.clear() directly.
        """
        self._channel_data.clear()

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
        channel: Optional[int] = None,
        scale_factor: int = 1
    ) -> Optional[np.ndarray]:
        """
        Get a tile from the CZI at the specified scale.

        For multi-scale vessel detection, use scale_factor > 1 to get
        downsampled tiles. The tile coordinates are in the SCALED coordinate
        system (i.e., at 1/8x, a tile at (0, 0) of size 4000 covers
        4000*8 = 32000 pixels in the original image).

        Args:
            tile_x: Tile X origin in scaled coordinates
            tile_y: Tile Y origin in scaled coordinates
            tile_size: Size of output tile (square), e.g., 4000
            channel: Channel to read. If None and data loaded to RAM, uses primary channel.
                     Required if reading from disk.
            scale_factor: Downsampling factor (1=full res, 2=1/2, 4=1/4, 8=1/8).
                          The returned tile will be tile_size x tile_size pixels.

        Returns:
            2D numpy array with tile data at requested scale, or None if invalid

        Example:
            # Get a 4000x4000 tile at 1/8x resolution (covers 32000x32000 original pixels)
            tile = loader.get_tile(0, 0, 4000, channel=1, scale_factor=8)
        """
        # Determine which channel to use
        target_channel = channel

        # Check if we have this channel in RAM
        if target_channel is not None and target_channel in self._channel_data:
            return self._get_tile_from_ram(tile_x, tile_y, tile_size, target_channel, scale_factor)

        # If no channel specified, try primary channel
        if target_channel is None and self._primary_channel is not None:
            return self._get_tile_from_ram(tile_x, tile_y, tile_size, self._primary_channel, scale_factor)

        # Fall back to reading from CZI
        if target_channel is None:
            raise ValueError("channel is required when data not loaded to RAM")

        return self._get_tile_from_czi(tile_x, tile_y, tile_size, target_channel, scale_factor)

    def _get_tile_from_ram(
        self,
        tile_x: int,
        tile_y: int,
        tile_size: int,
        channel: int,
        scale_factor: int = 1
    ) -> Optional[np.ndarray]:
        """
        Extract a tile from RAM-loaded channel data, optionally downsampled.

        Args:
            tile_x: Tile X origin in scaled coordinates
            tile_y: Tile Y origin in scaled coordinates
            tile_size: Size of output tile (square)
            channel: Channel to extract
            scale_factor: Downsampling factor (1=full res, 2=1/2, etc.)

        Returns:
            Tile data at requested scale, or None if invalid

        Raises:
            ValueError: If scale_factor is not positive
        """
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")

        data = self._channel_data.get(channel)
        if data is None:
            logger.warning(f"Channel {channel} not loaded, falling back to CZI read")
            return self._get_tile_from_czi(tile_x, tile_y, tile_size, channel, scale_factor)

        # Convert scaled coordinates to full-resolution coordinates
        # tile_x, tile_y are in the scaled coordinate space
        full_tile_x = tile_x * scale_factor
        full_tile_y = tile_y * scale_factor
        full_tile_size = tile_size * scale_factor

        # Convert to relative coordinates (from mosaic origin)
        rel_x = full_tile_x - self.x_start
        rel_y = full_tile_y - self.y_start

        # Bounds check
        if rel_x < 0 or rel_y < 0:
            return None
        if rel_x >= self.width or rel_y >= self.height:
            return None

        y2 = min(rel_y + full_tile_size, self.height)
        x2 = min(rel_x + full_tile_size, self.width)

        # Extract full-resolution region
        tile_data = data[rel_y:y2, rel_x:x2]
        if tile_data.size == 0:
            return None

        # Downsample if scale_factor > 1
        if scale_factor > 1:
            # Calculate target size (may be smaller if at edge)
            target_h = (y2 - rel_y) // scale_factor
            target_w = (x2 - rel_x) // scale_factor
            if target_h == 0 or target_w == 0:
                return None

            # Use INTER_AREA for downsampling (best for reduction)
            tile_data = cv2.resize(
                tile_data,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            )

        return tile_data

    def _get_tile_from_czi(
        self,
        tile_x: int,
        tile_y: int,
        tile_size: int,
        channel: int,
        scale_factor: int = 1
    ) -> Optional[np.ndarray]:
        """
        Read a tile directly from the CZI file at the specified scale.

        Args:
            tile_x: Tile X origin in scaled coordinates
            tile_y: Tile Y origin in scaled coordinates
            tile_size: Size of output tile (square)
            channel: Channel to read
            scale_factor: Downsampling factor (1=full res, 2=1/2, etc.)

        Returns:
            Tile data at requested scale, or None if invalid

        Raises:
            ValueError: If scale_factor is not positive
        """
        if self.reader is None:
            raise RuntimeError("CZI reader has been released. Cannot read tiles from disk. "
                               "Use --load-to-ram or reload the CZI file.")

        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")

        # Convert scaled coordinates to full-resolution coordinates
        full_tile_x = tile_x * scale_factor
        full_tile_y = tile_y * scale_factor
        full_tile_size = tile_size * scale_factor

        # Use aicspylibczi's native scale_factor parameter
        # Note: aicspylibczi reads the region at full-res coords, then downsamples
        tile_data = self.reader.read_mosaic(
            region=(full_tile_x, full_tile_y, full_tile_size, full_tile_size),
            scale_factor=scale_factor,
            C=channel,
        )

        if tile_data is None or tile_data.size == 0:
            return None

        tile_data = _squeeze_batch_dims(tile_data)

        # Accept both grayscale (2D) and RGB (3D) data
        if tile_data.ndim not in (2, 3):
            return None

        return tile_data

    def get_pixel_size(self) -> float:
        """
        Get pixel size in micrometers.

        Returns:
            Pixel size in um/px (default 0.22 if not found in metadata)
        """
        pixel_size = None
        try:
            metadata = self.reader.meta
            if isinstance(metadata, str):
                import xml.etree.ElementTree as ET
                metadata = ET.fromstring(metadata)
            scaling = metadata.find('.//Scaling/Items/Distance[@Id="X"]/Value')
            if scaling is not None:
                pixel_size = float(scaling.text) * 1e6
        except Exception as e:
            logger.warning(f"Could not read pixel size from CZI metadata: {e}")
        if pixel_size is None:
            logger.warning("pixel_size_um not found in CZI metadata — falling back to 0.22 um/px. "
                           "Verify this matches your microscope configuration.")
            pixel_size = 0.22
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

    def release_reader(self):
        """Release the CziFile reader to free memory while keeping loaded data."""
        if hasattr(self, 'reader') and self.reader is not None:
            del self.reader
            self.reader = None
            gc.collect()
            logger.debug(f"Released reader for {self.slide_name} (data retained)")

    def close(self):
        """Release all resources including loaded data."""
        self._channel_data.clear()
        self._primary_channel = None
        self.release_reader()
        logger.debug(f"Closed loader for {self.slide_name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        channels_str = ', '.join(str(c) for c in self.loaded_channels) if self.loaded_channels else 'none'
        scene_str = f", scene={self.scene}" if self.scene != 0 else ""
        return (
            f"CZILoader('{self.slide_name}'{scene_str}, "
            f"size={self.width}x{self.height}, "
            f"loaded_channels=[{channels_str}], "
            f"memory={self.memory_usage_gb():.2f}GB)"
        )
