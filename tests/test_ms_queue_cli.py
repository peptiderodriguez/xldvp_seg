"""Smoke tests for the `scripts/build_ms_queue.py` CLI wrapper.

The library is thoroughly tested via `tests/test_ms_queue.py`, but the CLI
is a subprocess entry point that argparse-drifts would break silently. These
tests exercise the actual script as a subprocess so flag renames, missing
required args, and YAML-parsing regressions surface in CI.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "build_ms_queue.py"


class TestBuildMsQueueCLI:
    def test_help_returns_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--samples" in result.stdout
        assert "--config" in result.stdout
        assert "--output-dir" in result.stdout

    def test_missing_required_args_fails(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "required" in result.stdout.lower()

    def test_happy_path_minimal_fixture(self, tmp_path):
        """End-to-end: 2-row input → 2-row queue + key."""
        samples_csv = tmp_path / "samples.csv"
        df = pd.DataFrame(
            [
                {"plate": 1, "well": "B2", "slide": "s1", "rep": 1},
                {"plate": 1, "well": "B4", "slide": "s2", "rep": 1},
            ]
        )
        df.to_csv(samples_csv, index=False)

        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text(
            textwrap.dedent(
                """
                file_name_template: "{date}_{slide}_{well_384}_{well_96}"
                autosampler_slots:
                  B2: 2
                shuffle: false
                """
            ).strip()
        )

        out_dir = tmp_path / "out"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--samples",
                str(samples_csv),
                "--config",
                str(config_yaml),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (
            result.returncode == 0
        ), f"script failed: stdout={result.stdout} stderr={result.stderr}"
        assert (out_dir / "ms_queue_B2.csv").exists()
        assert (out_dir / "ms_queue_key.csv").exists()
        assert (out_dir / "ms_queue_key.json").exists()

    def test_missing_samples_file_fails_cleanly(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("file_name_template: x\nautosampler_slots: {B2: 2}\n")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--samples",
                str(tmp_path / "does_not_exist.csv"),
                "--config",
                str(config_yaml),
                "--output-dir",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        # Should mention the missing file
        combined = result.stdout + result.stderr
        assert "samples" in combined.lower() or "not found" in combined.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
