"""Binary data encoding for HTML embedding."""

import base64
import json

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def encode_float32_base64(arr):
    """Encode a numpy float32 array as base64 string (little-endian)."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")


def encode_uint8_base64(arr):
    """Encode a numpy uint8 array as base64 string."""
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode("ascii")


def safe_json(obj):
    """JSON-encode an object safe for embedding in <script> blocks.

    Escapes '</' sequences to prevent premature </script> termination (XSS).
    """
    return json.dumps(obj).replace("</", "<\\/")


def build_contour_js_data(contours_raw, max_contours=100_000):
    """Convert raw contours (pixel or um coordinates) to compact um-coordinate JS objects.

    Each contour becomes:
      { pts: Float32Array([x0,y0,x1,y1,...]), bx1, by1, bx2, by2 }

    Coordinates are converted from pixels to um using per-detection pixel_size_um.
    Bounding boxes enable fast viewport culling in renderPanel().

    Args:
        contours_raw: list of (contour_pts, pixel_size_um) from _collect_contour().
        max_contours: cap to avoid huge HTML files.

    Returns:
        List of compact dicts ready for JSON embedding.
    """
    if not contours_raw:
        return []

    out = []
    step = max(1, len(contours_raw) // max_contours)
    for i in range(0, len(contours_raw), step):
        contour, pixel_size = contours_raw[i]
        try:
            pts = np.asarray(contour, dtype=np.float32)
            # Contour may be [[x,y],...] or [[x,y,z],...] — take only x,y
            if pts.ndim == 2 and pts.shape[1] >= 2:
                pts = pts[:, :2]
            elif pts.ndim == 1 and len(pts) % 2 == 0:
                pts = pts.reshape(-1, 2)
            else:
                continue
            if len(pts) < 3:
                continue
            pts_um = pts * pixel_size
            flat = pts_um.ravel().tolist()
            bx1 = float(pts_um[:, 0].min())
            bx2 = float(pts_um[:, 0].max())
            by1 = float(pts_um[:, 1].min())
            by2 = float(pts_um[:, 1].max())
            out.append(
                {
                    "pts": flat,
                    "bx1": round(bx1, 1),
                    "by1": round(by1, 1),
                    "bx2": round(bx2, 1),
                    "by2": round(by2, 1),
                }
            )
        except Exception:
            continue
    return out
