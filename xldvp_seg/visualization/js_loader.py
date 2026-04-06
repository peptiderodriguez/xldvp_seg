"""Load composable JavaScript components from the js/ directory."""

from functools import lru_cache
from pathlib import Path

_JS_DIR = Path(__file__).parent / "js"


@lru_cache(maxsize=32)
def _read_js_file(name: str) -> str:
    path = _JS_DIR / f"{name}.js"
    if not path.exists():
        raise FileNotFoundError(f"JS component not found: {path}")
    return path.read_text(encoding="utf-8")


def load_js(*component_names: str) -> str:
    """Load and concatenate JS components by name.

    Args:
        *component_names: Names of .js files (without extension).

    Returns:
        Concatenated JavaScript string.
    """
    parts = []
    for name in component_names:
        parts.append(f"// === {name}.js ===")
        parts.append(_read_js_file(name))
    return "\n".join(parts)
