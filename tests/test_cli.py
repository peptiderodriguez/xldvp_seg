"""Tests for xlseg CLI subcommand parsing."""

import subprocess
import sys

# ---------------------------------------------------------------------------
# Helper: run CLI via subprocess using the Python import path
# ---------------------------------------------------------------------------


def _run_cli(*argv, timeout=30):
    """Run xlseg CLI with given argv via subprocess and return CompletedProcess."""
    argv_str = ", ".join(repr(a) for a in argv)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.argv=['xlseg', {argv_str}]; "
            "from xldvp_seg.cli.main import cli; cli()",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# General CLI tests (existing)
# ---------------------------------------------------------------------------


class TestCLIHelp:

    def test_no_args_prints_help(self):
        result = _run_cli()
        assert result.returncode == 0

    def test_subcommands_listed(self):
        result = _run_cli("--help")
        for cmd in [
            "detect",
            "info",
            "classify",
            "markers",
            "serve",
            "score",
            "export-lmd",
            "system",
            "models",
            "strategies",
            "download-models",
        ]:
            assert cmd in result.stdout, f"'{cmd}' not in help"

    def test_unknown_subcommand_handled(self):
        """An unknown subcommand should be handled without crashing."""
        result = _run_cli("nonexistent-command")
        # argparse treats unknown subcommand as unknown args → dispatches None → prints help
        # Implementation exits 0 on None command, so just verify no crash
        assert result.returncode in (0, 2)


# ---------------------------------------------------------------------------
# Subcommands that print tables (no file I/O needed)
# ---------------------------------------------------------------------------


class TestCLIModels:

    def test_models_prints_table(self):
        result = _run_cli("models")
        assert result.returncode == 0
        assert "sam2" in result.stdout


class TestCLIStrategies:

    def test_strategies_prints_table(self):
        result = _run_cli("strategies")
        assert result.returncode == 0
        assert "cell" in result.stdout
        assert "nmj" in result.stdout


# ---------------------------------------------------------------------------
# Subcommands that use their own argparse (testable via --help)
# ---------------------------------------------------------------------------


class TestCLIDownloadModels:

    def test_download_models_help(self):
        """xlseg download-models --help should exit 0 with usage text."""
        result = _run_cli("download-models", "--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_download_models_no_args_prints_help(self):
        """xlseg download-models with no flags prints help (no crash)."""
        result = _run_cli("download-models")
        assert result.returncode == 0


class TestCLISystem:

    def test_system_runs(self):
        """xlseg system should run without error (prints system info)."""
        result = _run_cli("system")
        # system_info.py should complete; may print to stdout or stderr
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Subcommands that dispatch to scripts — test argparse dispatch only.
# These subcommands require file paths as arguments, so we test that the
# CLI correctly routes the command (parse_known_args identifies it) rather
# than running the full script. We use --help where the underlying script
# supports it.
# ---------------------------------------------------------------------------


class TestCLIDetect:

    def test_detect_help(self):
        """xlseg detect --help should print detection CLI usage."""
        result = _run_cli("detect", "--help")
        # The detect subcommand dispatches to run_segmentation.main() which
        # uses argparse, so --help should print help and exit 0
        assert result.returncode == 0
        assert "czi" in result.stdout.lower() or "detect" in result.stdout.lower()


class TestCLIClassify:

    def test_classify_help(self):
        """xlseg classify --help should exit 0 with usage text."""
        result = _run_cli("classify", "--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()


class TestCLIMarkers:

    def test_markers_help(self):
        """xlseg markers --help should print marker classification usage."""
        result = _run_cli("markers", "--help")
        assert result.returncode == 0
        assert "marker" in result.stdout.lower()


class TestCLIScore:

    def test_score_help(self):
        """xlseg score --help should exit 0 with usage text."""
        result = _run_cli("score", "--help")
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()


class TestCLIExportLmd:

    def test_export_lmd_help(self):
        """xlseg export-lmd --help should print LMD export usage."""
        result = _run_cli("export-lmd", "--help")
        assert result.returncode == 0
        assert "lmd" in result.stdout.lower() or "export" in result.stdout.lower()


class TestCLIServe:

    def test_serve_help(self):
        """xlseg serve --help should print HTML server usage."""
        result = _run_cli("serve", "--help")
        assert result.returncode == 0


class TestCLIInfo:

    def test_info_help(self):
        """xlseg info --help should print CZI info usage."""
        result = _run_cli("info", "--help")
        assert result.returncode == 0
        assert "czi" in result.stdout.lower() or "info" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Dispatch table completeness
# ---------------------------------------------------------------------------


class TestCLIDispatchTable:

    def test_all_subcommands_in_dispatch(self):
        """Every subcommand registered in argparse has a dispatch entry."""
        from xldvp_seg.cli.main import _DISPATCH

        expected = {
            "info",
            "detect",
            "classify",
            "cluster",
            "markers",
            "score",
            "export-lmd",
            "serve",
            "system",
            "models",
            "strategies",
            "download-models",
        }
        assert set(_DISPATCH.keys()) == expected

    def test_dispatch_values_are_callable(self):
        """Every dispatch entry is callable."""
        from xldvp_seg.cli.main import _DISPATCH

        for name, func in _DISPATCH.items():
            assert callable(func), f"Dispatch entry '{name}' is not callable"
