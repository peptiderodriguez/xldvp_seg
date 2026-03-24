"""Tests for xlseg CLI subcommand parsing."""

import subprocess
import sys


class TestCLIHelp:

    def test_no_args_prints_help(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv=['xlseg']; "
             "from segmentation.cli.main import cli; cli()"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_subcommands_listed(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv=['xlseg', '--help']; "
             "from segmentation.cli.main import cli; cli()"],
            capture_output=True, text=True, timeout=30,
        )
        for cmd in ["detect", "info", "classify", "markers", "serve"]:
            assert cmd in result.stdout, f"'{cmd}' not in help"


class TestCLIModels:

    def test_models_prints_table(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv=['xlseg', 'models']; "
             "from segmentation.cli.main import cli; cli()"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "sam2" in result.stdout


class TestCLIStrategies:

    def test_strategies_prints_table(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.argv=['xlseg', 'strategies']; "
             "from segmentation.cli.main import cli; cli()"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "cell" in result.stdout
        assert "nmj" in result.stdout
