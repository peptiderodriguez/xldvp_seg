"""Tests for scripts/generate_lumen_annotation.py."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.generate_lumen_annotation import pick_read_scale


class TestPickReadScale:
    """Test multi-scale read selection."""

    def test_small_lumen_uses_finest(self):
        # 20um lumen: at 4x (0.69um/px), diameter = 29px < 50 → still 4x (min)
        assert pick_read_scale(20, 0.1725) == 4

    def test_medium_lumen_uses_4x(self):
        # 100um lumen: at 4x, diameter = 145px >= 50 → 4x
        # at 8x, diameter = 72px >= 50 → 8x (coarser)
        # at 16x, diameter = 36px < 50 → stop
        assert pick_read_scale(100, 0.1725) == 8

    def test_large_lumen_uses_16x(self):
        # 500um: at 16x (2.76um/px), diameter = 181px >= 50 → 16x
        # at 64x (11.04um/px), diameter = 45px < 50 → stop
        assert pick_read_scale(500, 0.1725) == 16

    def test_huge_lumen_uses_64x(self):
        # 5000um: at 64x, diameter = 453px >= 50 → 64x
        assert pick_read_scale(5000, 0.1725) == 64

    def test_tiny_lumen_returns_finest(self):
        # 5um: at 4x, diameter = 7px < 50 → still returns 4 (min available)
        assert pick_read_scale(5, 0.1725) == 4

    def test_different_pixel_size(self):
        # 200um lumen, pixel size 0.325 um
        # at 4x (1.3um/px): 154px >= 50 → 4x
        # at 8x (2.6um/px): 77px >= 50 → 8x
        # at 16x (5.2um/px): 38px < 50 → stop
        assert pick_read_scale(200, 0.325) == 8
