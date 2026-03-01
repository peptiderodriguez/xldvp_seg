"""JSON utilities: numpy-safe encoding and NaN/Inf sanitization.

These were duplicated across run_segmentation.py, apply_classifier.py,
spatial_cell_analysis.py, analyze_islets.py, and cluster_by_features.py.
"""

import json
import math

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
