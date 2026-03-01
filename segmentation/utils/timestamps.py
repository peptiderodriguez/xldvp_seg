"""Timestamp utilities for consistent output file naming.

Convention:
  - All key output files get a timestamp suffix: detections_20260227_143052.json
  - A symlink (or copy on Windows) points from the base name to the latest version
  - RUN_TIMESTAMP is set once at import and reused throughout a single pipeline run

Usage:
    from segmentation.utils.timestamps import timestamped_path, save_with_timestamp

    # Get timestamped path
    ts_path = timestamped_path("output/detections.json")
    # -> PosixPath("output/detections_20260227_143052.json")

    # Save JSON with timestamp + symlink
    save_with_timestamp("output/detections.json", data, fmt="json")
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Set once per process — all files in a single run share the same timestamp
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def timestamped_path(path, timestamp=None):
    """Add timestamp to a file path: foo.json -> foo_20260227_143052.json

    Args:
        path: Original file path (str or Path)
        timestamp: Override timestamp string (default: RUN_TIMESTAMP)

    Returns:
        Path with timestamp inserted before extension
    """
    p = Path(path)
    ts = timestamp or RUN_TIMESTAMP
    return p.with_name(f"{p.stem}_{ts}{p.suffix}")


def update_symlink(link_path, target_path):
    """Create or update a symlink. Falls back to no symlink on error."""
    link_path = Path(link_path)
    target_path = Path(target_path)
    try:
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target_path.name)
    except OSError as e:
        logger.warning(f"Could not create symlink {link_path} -> {target_path.name}: {e}")
        # Fall back to file copy so the canonical name still exists
        import shutil
        try:
            shutil.copy2(str(target_path), str(link_path))
            logger.info(f"Copied {target_path.name} -> {link_path.name} (symlink fallback)")
        except OSError as copy_err:
            logger.warning(f"Symlink fallback copy also failed: {copy_err}")


def save_with_timestamp(base_path, data, fmt="json", json_encoder=None):
    """Save data to a timestamped file and create a symlink from base_path.

    Args:
        base_path: The canonical path (e.g., "output/detections.json")
        data: Data to save (dict/list for json, str for text)
        fmt: "json" or "text"
        json_encoder: Custom JSON encoder class (default: None)

    Returns:
        Path: The actual timestamped file that was written
    """
    base = Path(base_path)
    ts_path = timestamped_path(base)
    base.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(ts_path, 'w') as f:
            json.dump(data, f, cls=json_encoder)
    elif fmt == "text":
        with open(ts_path, 'w') as f:
            f.write(data if isinstance(data, str) else str(data))
    else:
        raise ValueError(f"Unknown format: {fmt}")

    update_symlink(base, ts_path)
    logger.info(f"Saved {ts_path} (symlink: {base.name})")
    return ts_path
