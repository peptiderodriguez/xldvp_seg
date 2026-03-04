"""Classifier registry: named, versioned classifiers with training metadata.

Provides a central registry at ``{REPO}/classifiers/registry.json`` so that
classifiers can be referenced by name (e.g. ``--classifier vessel_v1``) instead
of by path.  Each entry records training provenance (F1, feature set, slide,
annotation count, timestamp).

Usage::

    from segmentation.utils.classifier_registry import (
        register_classifier, resolve_classifier, list_classifiers,
    )

    # After training
    register_classifier("vessel_v1", pkl_path, meta_dict)

    # In apply_classifier.py
    resolved_path = resolve_classifier("vessel_v1")  # -> absolute .pkl path

    # CLI listing
    list_classifiers()  # prints table to stdout
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLASSIFIERS_DIR = REPO_ROOT / "classifiers"
REGISTRY_FILE = CLASSIFIERS_DIR / "registry.json"


def _load_registry():
    """Load the registry JSON, returning an empty structure if missing."""
    if REGISTRY_FILE.exists():
        try:
            return fast_json_load(REGISTRY_FILE)
        except Exception as e:
            logger.warning("Failed to load classifier registry: %s", e)
    return {"classifiers": []}


def _save_registry(data):
    """Atomically write the registry JSON."""
    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(data, REGISTRY_FILE)


def _next_version(registry, name):
    """Return the next version number for *name* in the registry."""
    existing = [e for e in registry["classifiers"] if e["name"] == name]
    if not existing:
        return 1
    return max(e.get("version", 1) for e in existing) + 1


def register_classifier(name, pkl_path, meta, *, overwrite=False):
    """Copy a .pkl into ``classifiers/`` and add/update a registry entry.

    Args:
        name: Short classifier name (e.g. ``vessel_v1``).
        pkl_path: Path to the trained .pkl file.
        meta: Dict with training metadata — at minimum ``feature_set``,
              ``cv_f1_mean``, ``n_positive``, ``n_negative``.
        overwrite: If True, replace an existing entry with the same name
                   and version instead of auto-incrementing.

    Returns:
        The registry entry dict that was written.
    """
    pkl_path = Path(pkl_path).resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Classifier file not found: {pkl_path}")

    CLASSIFIERS_DIR.mkdir(parents=True, exist_ok=True)
    registry = _load_registry()

    version = _next_version(registry, name)

    # Destination filename
    dest_name = f"{name}.pkl" if version == 1 else f"{name}_v{version}.pkl"
    dest_path = CLASSIFIERS_DIR / dest_name

    # If overwriting, remove old entry and reuse v1 filename
    if overwrite:
        registry["classifiers"] = [
            e for e in registry["classifiers"] if e["name"] != name
        ]
        dest_name = f"{name}.pkl"
        dest_path = CLASSIFIERS_DIR / dest_name
        version = 1

    # Copy the pkl
    if pkl_path != dest_path:
        shutil.copy2(pkl_path, dest_path)
    logger.info("Registered classifier '%s' (v%d) -> %s", name, version, dest_path)

    entry = {
        "name": name,
        "version": version,
        "cell_type": meta.get("cell_type", "unknown"),
        "feature_set": meta.get("feature_set", "unknown"),
        "cv_f1": meta.get("cv_f1_mean"),
        "n_positive": meta.get("n_positive"),
        "n_negative": meta.get("n_negative"),
        "trained_at": meta.get("trained_at", datetime.now().isoformat()),
        "training_slide": meta.get("training_slide"),
        "training_annotations_path": meta.get("training_annotations_path"),
        "path": str(dest_path),
        "description": meta.get("description", ""),
    }

    registry["classifiers"].append(entry)
    _save_registry(registry)
    return entry


def resolve_classifier(name_or_path):
    """Resolve a classifier reference to an absolute .pkl path.

    Resolution order:
        1. If *name_or_path* is an existing file path, return it directly.
        2. Look up by name in the registry (latest version).

    Returns:
        Absolute Path to the .pkl file.

    Raises:
        FileNotFoundError: If the classifier cannot be resolved.
    """
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate.resolve()

    # Registry lookup
    registry = _load_registry()
    matches = [
        e for e in registry["classifiers"] if e["name"] == name_or_path
    ]
    if not matches:
        raise FileNotFoundError(
            f"Classifier '{name_or_path}' not found as file or in registry. "
            f"Run `python train_classifier.py --list-classifiers` to see available classifiers."
        )

    # Pick latest version
    best = max(matches, key=lambda e: e.get("version", 1))
    resolved = Path(best["path"])
    if not resolved.exists():
        raise FileNotFoundError(
            f"Registry entry '{name_or_path}' points to {resolved} which does not exist."
        )
    logger.info("Resolved classifier '%s' -> %s (v%d, F1=%.3f)",
                name_or_path, resolved, best.get("version", 1),
                best.get("cv_f1") or 0)
    return resolved


def get_classifier_meta(name_or_path):
    """Return the registry metadata dict for a classifier, or None."""
    registry = _load_registry()

    # Try by name first
    matches = [
        e for e in registry["classifiers"] if e["name"] == name_or_path
    ]
    if matches:
        return max(matches, key=lambda e: e.get("version", 1))

    # Try by path
    resolved = str(Path(name_or_path).resolve())
    for entry in registry["classifiers"]:
        if str(Path(entry["path"]).resolve()) == resolved:
            return entry
    return None


def list_classifiers():
    """Print a formatted table of registered classifiers to stdout."""
    registry = _load_registry()
    entries = registry.get("classifiers", [])
    if not entries:
        print("No classifiers registered. Use `train_classifier.py --register` to add one.")
        return

    # Header
    fmt = "{:<20s} {:<8s} {:<12s} {:<6s} {:<12s} {:<12s}"
    print(fmt.format("Name", "Version", "Features", "F1", "Annotations", "Trained"))
    print("-" * 72)
    for e in sorted(entries, key=lambda x: (x["name"], x.get("version", 1))):
        n_ann = (e.get("n_positive") or 0) + (e.get("n_negative") or 0)
        trained = (e.get("trained_at") or "")[:10]
        f1_str = f"{e['cv_f1']:.3f}" if e.get("cv_f1") is not None else "?"
        print(fmt.format(
            e["name"],
            f"v{e.get('version', 1)}",
            e.get("feature_set", "?"),
            f1_str,
            str(n_ann),
            trained,
        ))


def build_classifier_info(clf_path, clf_meta=None):
    """Build a provenance dict to attach to scored detections.

    Args:
        clf_path: Path to the .pkl file (str or Path).
        clf_meta: Dict from joblib.load() of the .pkl (optional — used
                  to fill in feature_set / cv_f1 when not in registry).

    Returns:
        Dict suitable for ``det["classifier_info"] = ...``.
    """
    clf_path = Path(clf_path).resolve()
    clf_meta = clf_meta or {}

    # Try registry first
    reg_meta = get_classifier_meta(str(clf_path))

    def _first_not_none(*values):
        """Return first non-None value (safe for numeric 0)."""
        for v in values:
            if v is not None:
                return v
        return None

    _reg = reg_meta or {}
    info = {
        "classifier_path": str(clf_path),
        "classifier_name": (
            _reg.get("name") or clf_meta.get("name") or clf_path.stem
        ),
        "feature_set": (
            _reg.get("feature_set") or clf_meta.get("feature_set", "unknown")
        ),
        "cv_f1": _first_not_none(
            _reg.get("cv_f1"), clf_meta.get("cv_f1_mean"),
        ),
        "n_train_positive": _first_not_none(
            _reg.get("n_positive"), clf_meta.get("n_positive"),
        ),
        "n_train_negative": _first_not_none(
            _reg.get("n_negative"), clf_meta.get("n_negative"),
        ),
        "scored_at": datetime.now().isoformat(),
    }
    return info


def extract_classifier_info(detections):
    """Extract classifier provenance from a list of detections.

    Returns:
        (scored_count, provenance_count, sample_info) where sample_info
        is the classifier_info dict from the first scored detection (or None).
    """
    scored = 0
    with_prov = 0
    sample_info = None
    for d in detections:
        if "rf_prediction" in d:
            scored += 1
            if "classifier_info" in d:
                with_prov += 1
                if sample_info is None:
                    sample_info = d["classifier_info"]
    return scored, with_prov, sample_info
