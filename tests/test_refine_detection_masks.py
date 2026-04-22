"""Smoke tests for scripts/refine_detection_masks.py helpers.

The script itself is an orchestrator; the core refinement logic is tested in
``test_mask_cleanup.py``. These tests cover the script-specific glue:
  - tile-origin parsing from directory names
  - contour extraction from a binary mask
  - tile slicing from an in-RAM channel array (2D uint16 + 3D uint8 RGB)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "refine_detection_masks.py"

# Load the script as a module since it lives under scripts/ (not on sys.path).
_spec = importlib.util.spec_from_file_location("refine_detection_masks", SCRIPT)
rdm = importlib.util.module_from_spec(_spec)
sys.modules["refine_detection_masks"] = rdm
_spec.loader.exec_module(rdm)


def test_parse_tile_origin():
    """tile_12288_4096 -> (12288, 4096)."""
    fake = Path("/some/run/tiles/tile_12288_4096/mk_masks.h5")
    assert rdm._parse_tile_origin(fake) == (12288, 4096)


def test_parse_tile_origin_negative_coords():
    """Handles negative tile origins (CZI stage coords can be negative)."""
    # Tile dirs use underscore separator; negative coords would be encoded as
    # tile_-12288_4096. The split on '_' gives ['tile', '-12288', '4096'].
    fake = Path("/some/run/tiles/tile_-12288_4096/mk_masks.h5")
    assert rdm._parse_tile_origin(fake) == (-12288, 4096)


def test_load_tile_rgb_grayscale():
    """2D uint16 channel array slices correctly."""
    arr = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    tile = rdm._load_tile_rgb(arr, x_start=0, y_start=0, tile_x=10, tile_y=20, tile_h=30, tile_w=40)
    assert tile.shape == (30, 40)
    assert tile.dtype == np.uint16
    assert np.array_equal(tile, arr[20:50, 10:50])


def test_load_tile_rgb_3d_rgb():
    """3D (H, W, 3) uint8 RGB channel array slices correctly (trailing axis preserved)."""
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    tile = rdm._load_tile_rgb(arr, x_start=0, y_start=0, tile_x=10, tile_y=20, tile_h=30, tile_w=40)
    assert tile.shape == (30, 40, 3)
    assert tile.dtype == np.uint8
    assert np.array_equal(tile, arr[20:50, 10:50, :])


def test_load_tile_rgb_stage_offset():
    """x_start/y_start shifts global tile coords back to array-relative indices."""
    arr = np.arange(200 * 200, dtype=np.uint16).reshape(200, 200)
    # Global tile at (1000, 2000), array starts at (990, 1995) -> relative (10, 5)
    tile = rdm._load_tile_rgb(
        arr, x_start=990, y_start=1995, tile_x=1000, tile_y=2000, tile_h=10, tile_w=20
    )
    assert tile.shape == (10, 20)
    assert np.array_equal(tile, arr[5:15, 10:30])


def test_contour_from_mask_circle():
    """A solid disk gives a closed contour that fully encloses the mask pixels."""
    mask = np.zeros((100, 100), dtype=bool)
    yy, xx = np.ogrid[:100, :100]
    mask[(yy - 50) ** 2 + (xx - 50) ** 2 <= 20**2] = True

    contour = rdm._contour_from_mask(mask)
    assert contour is not None
    assert contour.ndim == 2 and contour.shape[1] == 2  # (N, 2) [X, Y]
    # Contour should be a closed-ish polygon around the disk
    assert contour.shape[0] >= 30  # rough perimeter check


def test_contour_from_mask_empty():
    """Empty mask -> None (so caller can skip the detection)."""
    mask = np.zeros((50, 50), dtype=bool)
    assert rdm._contour_from_mask(mask) is None


def test_contour_from_mask_keeps_largest():
    """Two disconnected components -> only the larger contour returned."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True  # small square, 100 px
    mask[50:90, 50:90] = True  # large square, 1600 px
    contour = rdm._contour_from_mask(mask)
    assert contour is not None
    # Largest contour should span the big square
    xs, ys = contour[:, 0], contour[:, 1]
    assert xs.min() >= 49 and xs.max() <= 90
    assert ys.min() >= 49 and ys.max() <= 90
