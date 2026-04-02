"""Scanpy-style API for xldvp_seg.

Usage:
    from xldvp_seg import api as xseg
    from xldvp_seg.core import SlideAnalysis

    slide = SlideAnalysis.load("output/my_slide/...")
    xseg.tl.markers(slide, ...)
    xseg.pl.umap(slide)
    xseg.io.export_lmd(slide, crosses="crosses.json")
"""

from xldvp_seg.api import io, pl, pp, tl

__all__ = ["pp", "tl", "pl", "io"]
