"""Preprocessing functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def inspect(czi_path: str | Path) -> dict[str, Any]:
    """Inspect CZI metadata. Returns channel info dict.

    Args:
        czi_path: Path to CZI file.

    Returns:
        Dict with channel metadata from CZI.
    """
    from xldvp_seg.io.czi_loader import get_czi_metadata

    return get_czi_metadata(str(czi_path))


def detect(
    czi_path: str | Path,
    cell_type: str = "cell",
    output_dir: str | Path | None = None,
    channel_spec: str | None = None,
    num_gpus: int = 4,
    **kwargs: Any,
) -> str:
    """Run detection pipeline.

    For production runs, use run_pipeline.sh with YAML configs instead.
    This function builds and logs the equivalent command.

    Args:
        czi_path: Path to CZI file.
        cell_type: Detection strategy (cell, nmj, mk, vessel, etc.)
        output_dir: Output directory.
        channel_spec: Channel spec string (e.g., "cyto=PM,nuc=488").
        num_gpus: Number of GPUs.

    Returns:
        Command string (for now).
    """
    parts = [f"xlseg detect --czi-path {czi_path} --cell-type {cell_type}"]
    if output_dir:
        parts.append(f"--output-dir {output_dir}")
    if channel_spec:
        parts.append(f'--channel-spec "{channel_spec}"')
    parts.append(f"--num-gpus {num_gpus}")
    cmd = " ".join(parts)
    logger.info("Detection command: %s", cmd)
    return cmd
