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
    return obj


def atomic_json_dump(data, filepath, cls=NumpyEncoder, sanitize=True):
    """Write JSON atomically: temp file + os.replace() to prevent partial writes.

    A SLURM timeout or OOM during a direct json.dump() leaves a partially-written
    file. On resume, the pipeline finds the corrupt file and either crashes or
    silently loses hours of work. This function writes to a temp file first,
    then atomically replaces the target — so the file either contains the complete
    data or does not exist at all.

    Args:
        data: Python object to serialize.
        filepath: Target path (str or Path).
        cls: JSON encoder class (default: NumpyEncoder).
        sanitize: If True, run sanitize_for_json() first to replace Python
            float('nan')/float('inf') with None (default: True).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if sanitize:
        data = sanitize_for_json(data)

    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, cls=cls)
        os.replace(tmp_path, filepath)
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
