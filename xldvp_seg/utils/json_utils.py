"""JSON utilities: numpy-safe encoding, NaN/Inf sanitization, and atomic writes.

These were duplicated across run_segmentation.py, apply_classifier.py,
spatial_cell_analysis.py, analyze_islets.py, and cluster_by_features.py.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types.

    Usage::

        json.dump(data, f, cls=NumpyEncoder)
    """

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Regular Python float NaN/Inf can reach default() via subclasses
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        return super().default(obj)


def sanitize_for_json(obj):
    """Recursively replace NaN/inf with None in nested structures.

    ``json.dump(default=)`` only fires for non-serializable types.
    Python ``float('nan')`` IS serializable (outputs non-standard ``"NaN"``
    token), so we must walk the structure recursively to catch them.

    Also converts numpy scalars and arrays to native Python types.

    Usage::

        clean = sanitize_for_json(data)
        json.dump(clean, f)
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, set):
        return [sanitize_for_json(v) for v in obj]
    return obj


def atomic_json_dump(data, filepath, sanitize=True):
    """Write JSON atomically: temp file + os.replace() to prevent partial writes.

    Args:
        data: Python object to serialize.
        filepath: Target path (str or Path).
        sanitize: If True, run sanitize_for_json() first to replace Python
            float('nan')/float('inf') with None and convert numpy types
            (default: True).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if sanitize:
        data = sanitize_for_json(data)

    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
    try:
        if _HAS_ORJSON:
            with os.fdopen(fd, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))
        else:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, cls=None if sanitize else NumpyEncoder)
        os.replace(tmp_path, filepath)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_savez(path, **arrays):
    """``np.savez`` via temp + ``os.replace`` so a crash can't leave a
    half-written ``.npz`` that crashes ``np.load`` on the next run.

    Hands ``np.savez`` a file handle (rather than a string path) so its
    auto-``.npz`` suffix doesn't append to the ``.tmp`` name.

    Args:
        path: Target ``.npz`` path (str or Path).
        **arrays: Arrays to save, keyed by archive name.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def fast_json_load(filepath):
    """Load JSON file using orjson if available (2-3x faster for large files)."""
    filepath = Path(filepath)
    if _HAS_ORJSON:
        return orjson.loads(filepath.read_bytes())
    with open(filepath) as f:
        return json.load(f)
