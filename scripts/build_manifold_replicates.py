#!/usr/bin/env python
"""DEPRECATED: use ``xlseg manifold-sample`` or ``scripts/manifold_sample.py``.

Kept for backwards compatibility with ad-hoc n45 sbatch files during the
transition. Forwards all argv to the canonical entry point.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.warn(
    "build_manifold_replicates.py is deprecated -- use `xlseg manifold-sample`.",
    DeprecationWarning,
    stacklevel=2,
)

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "scripts" / "manifold_sample.py"
os.execv(sys.executable, [sys.executable, str(TARGET), *sys.argv[1:]])
