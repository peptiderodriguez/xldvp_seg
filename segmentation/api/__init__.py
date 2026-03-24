"""Scanpy-style API for xldvp_seg.

Usage:
    from segmentation import api as xseg
    from segmentation.core import SlideAnalysis

    slide = SlideAnalysis.load("output/my_slide/...")
    xseg.tl.markers(slide, ...)
    xseg.pl.umap(slide)
    xseg.io.export_lmd(slide, crosses="crosses.json")
"""

from segmentation.api import pp, tl, pl, io

__all__ = ["pp", "tl", "pl", "io"]
