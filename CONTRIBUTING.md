# Contributing to xldvp_seg

Thank you for your interest in contributing. This guide covers the essentials for getting started.

## Development Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone <repo-url> && cd xldvp_seg
./install.sh --dev
pre-commit install
```

Pre-commit hooks (ruff + black) run automatically on each commit.

## Code Style

- **Formatter:** Black, line-length 100
- **Linter:** Ruff (E/F/W/I/N/UP/B/C4, E501 ignored)
- **Python:** 3.11 only (`pyproject.toml` pins `requires-python = ">=3.11,<3.12"`)

```bash
make lint      # check style
make format    # auto-fix
```

Key conventions:
- Use `get_logger(__name__)` from `xldvp_seg.utils.logging` (never bare `logging.getLogger`).
- Use `atomic_json_dump()` / `fast_json_load()` from `xldvp_seg.utils.json_utils` for all JSON I/O.
- Never hardcode pixel sizes, channel indices, or file paths that should come from CZI metadata.
- All coordinates are **[x, y]** (horizontal, vertical). See `docs/COORDINATE_SYSTEM.md`.

## Testing

```bash
make test      # runs pytest with coverage
```

- Tests live in `tests/` using pytest. See `conftest.py` for shared fixtures.
- Write tests for every new feature or bug fix.
- Run the full suite before opening a PR:

```bash
make lint && make test
```

## Pull Request Process

1. **Branch from `main`** with a descriptive name (e.g., `fix/shm-cleanup`, `feat/new-strategy`).
2. **One feature per PR.** Keep changes focused and reviewable.
3. **Ensure `make lint && make test` passes** before requesting review.
4. **Update documentation** if your change affects CLI flags, pipeline behavior, or architecture.
5. **Add a changelog entry** under the `[Unreleased]` section in `CHANGELOG.md`.

## Adding a Detection Strategy

1. Create a new file in `xldvp_seg/detection/strategies/`.
2. Inherit from `DetectionStrategy` and use `MultiChannelFeatureMixin`.
3. Register with `@register_strategy("your_strategy")`.
4. Implement `detect_in_tile()`.
5. Add tests in `tests/`.

## Device Handling

Never hardcode `device="cuda"`. Use helpers from `xldvp_seg.utils.device`:

- `get_default_device()` -- detects cuda/mps/cpu
- `device_supports_gpu()` -- for Cellpose `gpu=` flag
- `empty_cache()` -- handles cuda, mps, and cpu

## Reporting Issues

Use the GitHub issue templates. For bug reports, include:
- Steps to reproduce
- Full traceback
- Output of `xlseg system`

## Questions?

Open a discussion or reach out to the maintainers.
