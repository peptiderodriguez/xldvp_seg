"""Shared RNG-seed helpers for reproducibility across the analysis stack.

The user-facing `--random-seed` flag (`run_segmentation.py`) sets numpy/stdlib/
torch seeds at pipeline entry, but historically many downstream analysis calls
hardcoded `random_state=42` / `np.random.default_rng(42)` and ignored it.
This helper gives callers an optional `seed: int | None = None` kwarg that:

- Uses the supplied seed when provided.
- Falls back to ``_DEFAULT_SEED = 42`` when ``None``, emitting a single
  ``UserWarning`` per-session so users know the default is in play.

Use the ``int`` return for sklearn's ``random_state=`` slot (sklearn's
``check_random_state`` rejects numpy ``Generator``); use ``np.random.default_rng()``
when you need a Generator.
"""

from __future__ import annotations

import warnings

_DEFAULT_SEED: int = 42
_warned_default: bool = False


def resolve_seed(seed: int | None, *, caller: str = "xldvp_seg") -> int:
    """Return an int seed, warning once per session on None.

    Parameters
    ----------
    seed
        Caller-supplied seed, or ``None`` to use the process-wide default.
    caller
        Short label for the warning message (e.g. module or function name).

    Returns
    -------
    int
        A seed safe to pass into sklearn's ``random_state`` or
        ``np.random.default_rng``.
    """
    global _warned_default
    if seed is not None:
        return int(seed)
    if not _warned_default:
        warnings.warn(
            f"{caller}: no seed provided; using default ({_DEFAULT_SEED}). "
            "Pass seed=... explicitly for reproducible experiments across "
            "cohorts.",
            UserWarning,
            stacklevel=3,
        )
        _warned_default = True
    return _DEFAULT_SEED


def reset_default_warning() -> None:
    """Clear the one-shot warning flag (test helper)."""
    global _warned_default
    _warned_default = False
