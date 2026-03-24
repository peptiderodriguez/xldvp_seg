"""xlseg CLI: unified entry point for the xldvp_seg pipeline.

Provides subcommands that delegate to the underlying scripts,
so users can run ``xlseg detect ...`` instead of
``python run_segmentation.py ...``.
"""

import argparse
import runpy
import sys
from pathlib import Path

# Repo root -- two levels up from segmentation/cli/main.py
_REPO = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _run_script(script_relpath: str, remaining: list[str]) -> None:
    """Run a script via runpy.run_path, forwarding remaining CLI args."""
    script = _REPO / script_relpath
    old_argv = sys.argv
    try:
        sys.argv = [str(script)] + remaining
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def _run_detect(remaining):
    old_argv = sys.argv
    try:
        sys.argv = ["xlseg detect"] + remaining
        from run_segmentation import main
        main()
    finally:
        sys.argv = old_argv


def _run_classify(remaining):
    old_argv = sys.argv
    try:
        sys.argv = ["xlseg classify"] + remaining
        from train_classifier import main
        main()
    finally:
        sys.argv = old_argv


def _run_export_lmd(remaining):
    old_argv = sys.argv
    try:
        sys.argv = ["xlseg export-lmd"] + remaining
        from run_lmd_export import main
        main()
    finally:
        sys.argv = old_argv


def _run_serve(remaining):
    old_argv = sys.argv
    try:
        sys.argv = ["xlseg serve"] + remaining
        from serve_html import main
        main()
    finally:
        sys.argv = old_argv


def _run_info(remaining):
    _run_script("scripts/czi_info.py", remaining)


def _run_markers(remaining):
    _run_script("scripts/classify_markers.py", remaining)


def _run_score(remaining):
    _run_script("scripts/apply_classifier.py", remaining)


def _run_system(remaining):
    _run_script("scripts/system_info.py", remaining)


def _run_models(remaining):
    """Print registered models from the model registry."""
    from segmentation.models.registry import ModelRegistry
    ModelRegistry.print_models()


def _run_strategies(remaining):
    """Print registered detection strategies."""
    import segmentation.detection.strategies  # noqa: F401 — trigger registration
    from segmentation.detection.registry import StrategyRegistry
    StrategyRegistry.print_strategies()


def _run_download_models(remaining):
    """Download model checkpoints."""
    import argparse as _ap
    parser = _ap.ArgumentParser(prog="xlseg download-models")
    parser.add_argument("--brightfield", action="store_true",
                        help="Download brightfield FMs (UNI2, Virchow2, CONCH, Phikon-v2)")
    parser.add_argument("--all", action="store_true",
                        help="Download all registered models")
    parser.add_argument("--model", type=str, default=None,
                        help="Download a specific model by name")
    args = parser.parse_args(remaining)
    from segmentation.models.manager import get_model_manager
    manager = get_model_manager(device="cpu")
    models_to_load = []
    if args.brightfield or args.all:
        models_to_load = ["uni2", "virchow2", "conch", "phikon_v2"]
    elif args.model:
        models_to_load = [args.model]
    else:
        parser.print_help()
        return
    for name in models_to_load:
        getter = getattr(manager, f"get_{name}", None)
        if not getter:
            print(f"  Unknown model: {name}")
            continue
        print(f"  Downloading {name}...")
        try:
            getter()
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    manager.cleanup()


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "info": _run_info,
    "detect": _run_detect,
    "classify": _run_classify,
    "markers": _run_markers,
    "score": _run_score,
    "export-lmd": _run_export_lmd,
    "serve": _run_serve,
    "system": _run_system,
    "models": _run_models,
    "strategies": _run_strategies,
    "download-models": _run_download_models,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(
        prog="xlseg",
        description="xldvp_seg: spatial cell segmentation & DVP pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("info", help="Inspect CZI metadata (channels, dimensions)")
    subparsers.add_parser("detect", help="Run cell detection pipeline")
    subparsers.add_parser("classify", help="Train RF classifier from annotations")
    subparsers.add_parser("markers", help="Classify marker pos/neg per channel")
    subparsers.add_parser("score", help="Score detections with trained classifier")
    subparsers.add_parser("export-lmd", help="Export for laser microdissection")
    subparsers.add_parser("serve", help="Serve HTML viewer with tunnel")
    subparsers.add_parser("system", help="Show system info and SLURM recommendations")
    subparsers.add_parser("models", help="List registered model checkpoints")
    subparsers.add_parser("strategies", help="List registered detection strategies")
    subparsers.add_parser("download-models", help="Download model checkpoints (brightfield needs HF token)")

    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _DISPATCH[args.command](remaining)
