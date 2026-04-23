"""Thermo Xcalibur MS queue CSV builder from LMD replicate manifests.

Takes a replicates manifest (CSV/DataFrame with 384-well assignments from
an LMD export) and produces per-box Thermo Xcalibur queue CSVs for the MS
autosampler. Each 384-well quadrant (B2, B3, C2, C3) maps 1:1 to a physical
96-well "box" via strided repacking:

    384 row B/D/F/H/J/L/N  ->  96 row A/B/C/D/E/F/G
    384 col 2/4/.../22     ->  96 col 1/2/.../11    (B2, C2 quadrants)
    384 col 3/5/.../23     ->  96 col 1/2/.../11    (B3, C3 quadrants)

Each box fills 96-well positions A1..G11 (7 x 11 = 77). Row H / col 12 of
the 96-well plate are unused.

Output per-box CSV format (Thermo Xcalibur):

    Bracket Type=4
    File Name,Path,Instrument Method,Position
    20260423_OA1_EdRo_SA_E990_..._B4_A2,D:\\,C:\\Xcalibur\\...,S2:A2
    ...

A sample-key sidecar (``<prefix>_key.csv`` + ``<prefix>_key.json``) is
always emitted so downstream analysis (DIA-NN, Spectronaut, OmicLinker) can
join the raw MS file name back to full sample metadata without parsing the
filename.

Multi-plate inputs (multiple 384-well plates) are grouped by
``(plate, quadrant)`` — each plate yields up to 4 boxes, named
``plate{N}_{quadrant}``. ``autosampler_slots`` applies across plates; the
operator physically swaps boxes between plate runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.lmd.well_plate import (
    EVEN_COLS,
    EVEN_ROWS,
    ODD_ROWS,
    QUADRANT_ORDER,
)
from xldvp_seg.utils.json_utils import atomic_json_dump
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 384 <-> 96 geometry
# ---------------------------------------------------------------------------

_EVEN_ROW_TO_96 = {row: chr(ord("A") + i) for i, row in enumerate(EVEN_ROWS)}
_ODD_ROW_TO_96 = {row: chr(ord("A") + i) for i, row in enumerate(ODD_ROWS)}
_ALL_EVEN_ROWS = set(EVEN_ROWS)
_ALL_ODD_ROWS = set(ODD_ROWS)
_VALID_ROWS = _ALL_EVEN_ROWS | _ALL_ODD_ROWS

_WELL_RE = re.compile(r"^([A-Za-z])(\d{1,2})$")

# Windows-reserved or Xcalibur-unfriendly characters in File Name values.
_FORBIDDEN_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')

# Unix-like path prefixes that almost certainly won't resolve on the MS PC.
_UNIX_PREFIXES = ("/Volumes/", "/fs/pool/", "/fs/gpfs", "/Users/", "/home/", "/mnt/")


def detect_quadrant(well_384: str) -> str:
    """Identify which of the 4 inner-ring quadrants a 384-well belongs to.

    Parameters
    ----------
    well_384
        384-well address like ``"B2"``, ``"N22"``, ``"O23"``.

    Returns
    -------
    One of ``"B2"``, ``"B3"``, ``"C2"``, ``"C3"``.

    Raises
    ------
    ConfigError
        If the well is malformed or lies in the outer ring
        (row A/P, col 1/24) which the LMD pipeline never uses.
    """
    if not isinstance(well_384, str):
        raise ConfigError(f"Invalid 384-well address (not a string): {well_384!r}")
    m = _WELL_RE.match(well_384)
    if m is None:
        raise ConfigError(f"Invalid 384-well address: {well_384!r}")
    row = m.group(1).upper()
    col = int(m.group(2))

    if row not in _VALID_ROWS:
        raise ConfigError(
            f"Well {well_384!r} is in outer row {row!r} " f"(LMD quadrants cover rows B-O only)"
        )
    if col < 2 or col > 23:
        raise ConfigError(
            f"Well {well_384!r} is in outer column {col} " f"(LMD quadrants cover cols 2-23 only)"
        )

    row_even = row in _ALL_EVEN_ROWS
    col_even = col % 2 == 0
    if row_even and col_even:
        return "B2"
    if row_even and not col_even:
        return "B3"
    if not row_even and col_even:
        return "C2"
    return "C3"


def map_384_to_96(well_384: str) -> str:
    """Repack a 384-well address into its 96-well box position.

    Each quadrant fills positions A1..G11 (7 x 11 = 77) of the 96-well box.

    Parameters
    ----------
    well_384
        384-well address like ``"B2"``, ``"N22"``.

    Returns
    -------
    96-well address like ``"A1"``, ``"G11"``.
    """
    detect_quadrant(well_384)  # validates the input; raises on outer ring
    m = _WELL_RE.match(well_384)
    row = m.group(1).upper()
    col = int(m.group(2))

    row_96 = _EVEN_ROW_TO_96[row] if row in _ALL_EVEN_ROWS else _ODD_ROW_TO_96[row]
    # Even-col quadrants: 2->1, 4->2, ..., 22->11
    # Odd-col quadrants:  3->1, 5->2, ..., 23->11
    col_96 = col // 2 if col in EVEN_COLS else (col - 1) // 2

    return f"{row_96}{col_96}"


def to_windows_path(path: str | None, warn_unix: bool = True) -> str:
    """Normalise a file path to Windows form (``/`` -> ``\\``).

    Empty / ``None`` becomes ``""``. Already-Windows paths (no forward
    slashes) are returned unchanged. Unix-style prefixes (``/Volumes``,
    ``/fs/pool/``, ``/Users/``, ``/home/``, ``/mnt/``) trigger a warning —
    Xcalibur on the MS PC won't resolve them, and the operator probably
    meant a ``C:\\Xcalibur\\...`` path.

    Parameters
    ----------
    path
        Source path string (any OS flavour) or ``None``.
    warn_unix
        Emit a log warning on Unix-like prefixes.

    Returns
    -------
    Windows-formatted path, or ``""`` for empty input.
    """
    if not path:
        return ""
    if warn_unix and any(path.startswith(p) for p in _UNIX_PREFIXES):
        logger.warning(
            "MS method path %r looks like a Unix path. Xcalibur on the MS PC "
            "will not find this — you probably want a C:\\Xcalibur\\... path.",
            path,
        )
    return path.replace("/", "\\")


# ---------------------------------------------------------------------------
# Config + main entry
# ---------------------------------------------------------------------------

MSMethodSpec = str | Callable[[pd.Series], str] | None


@dataclass
class ThermoQueueConfig:
    """Configuration for :func:`build_thermo_queues`.

    Attributes
    ----------
    file_name_template
        Python ``str.format`` template for the ``File Name`` column.
        Available fields: every input column name, plus ``{date}``,
        ``{well_384}``, ``{well_96}``, ``{quadrant}``, ``{box}``
        (autosampler slot number), ``{plate}``. Example:
        ``"{date}_OA1_EdRo_SA_E990_{slide}_{bone}_rep{replicate}_{well_384}_{well_96}"``.
        Note: if an input column shares a name with a reserved field
        (``date``, ``plate``, ``quadrant``, ``box``, ``well_384``,
        ``well_96``), the derived value wins in the template context.
    autosampler_slots
        Map from quadrant name (``"B2"`` etc.) to autosampler slot number.
        Same mapping applied to every plate in multi-plate inputs. Required.
    ms_method
        Either a path string, a callable ``(row) -> str``, or ``None``.
        ``None`` leaves the ``Instrument Method`` column blank and logs a
        one-time warning. Any returned path is Windows-formatted.
    path
        Value of the ``Path`` column (default ``"D:\\"``).
    date
        Date string for ``{date}`` placeholder. Default: today's
        ``YYYYMMDD``.
    empty_marker
        ``(column, value)`` pair that identifies empty/QC wells in the
        input. Empty rows are kept in the queue (the operator wants blank
        injections for carryover QC) but use ``empty_file_name_template``.
    empty_file_name_template
        Template for empty rows. Default:
        ``"{date}_BLANK_plate{plate}_{quadrant}_{well_96}"``.
    shuffle
        Shuffle each (plate, quadrant) group independently.
    shuffle_seed
        Base RNG seed. Each group uses seed + offset for determinism.
    bracket_type
        Thermo ``Bracket Type=N`` header value (default ``4``).
    """

    file_name_template: str
    autosampler_slots: dict[str, int]
    ms_method: MSMethodSpec = None
    path: str = "D:\\"
    date: str | None = None
    empty_marker: tuple[str, Any] | None = None
    empty_file_name_template: str | None = None
    shuffle: bool = True
    shuffle_seed: int = 42
    bracket_type: int = 4
    bracketing_blanks: int = 0
    bracketing_blank_template: str | None = None
    interspersed_blanks: int = 0
    interspersed_blank_template: str | None = None
    group_by_column: str | None = None
    group_by_ascending: bool = True
    group_separator_blanks: bool = False
    group_separator_blank_template: str | None = None
    column_substitutions: dict[str, tuple[str, str]] | None = None


_DEFAULT_EMPTY_TEMPLATE = "{date}_BLANK_plate{plate}_{quadrant}_{well_96}"
_DEFAULT_BRACKETING_TEMPLATE = "{date}_BLANK_{box_key}_{position_tag}"
_DEFAULT_INTERSPERSED_TEMPLATE = "{date}_BLANK_{box_key}_{well_96}"
_MAX_BRACKETING_BLANKS = 6  # row H in the 96-well box has 12 positions; 6 lead + 6 trail
# Interspersed-blank pool: middle of row H first, then column 12 of A-G. Row H
# ends get used by bracketing, so we pull the middle 6; col 12 of A-G is
# entirely unused by the 384 -> 96 mapping (which fills A1..G11).
_INTERSPERSED_POOL_H_MIDDLE = [f"H{i}" for i in range(4, 10)]  # H4..H9 (6 wells)
_INTERSPERSED_POOL_COL12 = [f"{r}12" for r in "ABCDEFG"]  # A12..G12 (7 wells)


def _render_filename(template: str, ctx: dict) -> str:
    try:
        return template.format(**ctx)
    except KeyError as e:
        raise ConfigError(
            f"file_name_template field {e} not found. " f"Available: {sorted(ctx)}"
        ) from e
    except IndexError as e:
        raise ConfigError(
            f"file_name_template uses positional placeholders; use named "
            f"fields only (e.g. {{slide}}, not {{0}}): {e}"
        ) from e


def _resolve_method(row: pd.Series, spec: MSMethodSpec) -> str:
    if spec is None:
        return ""
    if callable(spec):
        return to_windows_path(spec(row))
    return to_windows_path(spec)


def _resolve_method_for_blank(spec: MSMethodSpec) -> str:
    """Instrument method for synthetic bracketing blanks (no row context).

    Callable ``ms_method`` can't be invoked row-wise here; fall back to empty
    string. Caller logs a one-time warning if that matters.
    """
    if isinstance(spec, str):
        return to_windows_path(spec)
    return ""


def _bracketing_wells(n: int) -> tuple[list[str], list[str]]:
    """Return (leading_wells, trailing_wells) in 96-well row H.

    Leading = H1..Hn. Trailing = H(13-n)..H12. Non-overlapping for n <= 6.
    """
    leading = [f"H{i}" for i in range(1, n + 1)]
    trailing = [f"H{i}" for i in range(13 - n, 13)]
    return leading, trailing


def _interspersed_well_pool(n_bracket: int, n_wanted: int) -> list[str]:
    """Pick up to `n_wanted` distinct 96-well addresses for interspersed BLANKs.

    Reserves row H positions already used by bracketing (H1..H{n_bracket} and
    H{13-n_bracket}..H12). Remaining pool: row H middle wells + column 12
    (A12..G12). Raises ConfigError if ``n_wanted`` exceeds pool size.
    """
    reserved = set(f"H{i}" for i in range(1, n_bracket + 1))
    reserved |= set(f"H{i}" for i in range(13 - n_bracket, 13))
    pool = [w for w in _INTERSPERSED_POOL_H_MIDDLE if w not in reserved] + list(
        _INTERSPERSED_POOL_COL12
    )
    if n_wanted > len(pool):
        raise ConfigError(
            f"interspersed_blanks={n_wanted} exceeds available pool of "
            f"{len(pool)} wells (row H middle + col 12). Reduce count or "
            f"reduce bracketing_blanks."
        )
    return pool[:n_wanted]


def _interspersed_insert_positions(n_blanks: int, n_total: int, rng) -> set[int]:
    """Return `n_blanks` slot indices in ``[0, n_total)`` for interspersed BLANKs.

    Uses segment-based placement (like `well_plate.insert_empty_wells`): the
    full sequence is divided into `n_blanks` equal segments and one random
    position is chosen per segment, guaranteeing even spread rather than
    pure-random clustering.
    """
    if n_blanks <= 0 or n_total <= 0:
        return set()
    if n_blanks >= n_total:
        raise ConfigError(f"interspersed_blanks={n_blanks} cannot fit into a sequence of {n_total}")
    segment = n_total / n_blanks
    positions: set[int] = set()
    for i in range(n_blanks):
        lo = int(i * segment)
        hi = max(lo + 1, int((i + 1) * segment))
        pos = int(rng.integers(lo, hi))
        # collisions unlikely but retry into the same segment
        while pos in positions:
            pos = int(rng.integers(lo, hi)) if hi > lo + 1 else int(rng.integers(0, n_total))
        positions.add(pos)
    return positions


def _make_bracketing_rows(
    wells_96: list[str],
    tag_prefix: str,
    *,
    box_key: str,
    slot: int,
    plate_value,
    quadrant: str,
    date: str,
    path_value: str,
    method_value: str,
    template: str,
) -> tuple[list[dict], list[dict]]:
    """Build (queue_rows, key_rows) for a run of bracketing BLANKs."""
    queue_rows: list[dict] = []
    key_rows: list[dict] = []
    for i, well_96 in enumerate(wells_96, start=1):
        position_tag = f"{tag_prefix}{i}"
        ctx = {
            "date": date,
            "plate": plate_value,
            "quadrant": quadrant,
            "well_96": well_96,
            "box_key": box_key,
            "box": slot,
            "position_tag": position_tag,
        }
        file_name = _render_filename(template, ctx)
        position = f"S{slot}:{well_96}"
        queue_rows.append(
            {
                "File Name": file_name,
                "Path": path_value,
                "Instrument Method": method_value,
                "Position": position,
            }
        )
        key_rows.append(
            {
                "File Name": file_name,
                "plate": plate_value,
                "well": "",
                "well_384": "",
                "quadrant": quadrant,
                "well_96": well_96,
                "is_empty": True,
                "slot": slot,
                "Position": position,
                "box_key": box_key,
                "box_row_index": None,  # filled after concat
                "instrument_method": method_value,
                "queue_date": date,
                "blank_kind": "bracketing",
            }
        )
    return queue_rows, key_rows


def build_thermo_queues(
    samples: pd.DataFrame | str | Path,
    config: ThermoQueueConfig,
    out_dir: str | Path,
    well_col: str = "well",
    plate_col: str | None = "plate",
    out_prefix: str = "ms_queue",
    combined: bool = False,
) -> dict[str, Path]:
    """Split samples by (plate, quadrant), repack into 96-well boxes, and
    emit Thermo queue CSVs plus a sample-key sidecar.

    Parameters
    ----------
    samples
        Replicate manifest. ``pd.DataFrame`` or path to a CSV.
    config
        :class:`ThermoQueueConfig` instance.
    out_dir
        Directory to write output files into. Created if missing.
    well_col
        Column holding 384-well addresses (default ``"well"``).
    plate_col
        Column holding plate number. If the column is absent from the
        input, single-plate mode is used (all rows assigned plate=1).
    out_prefix
        Filename prefix for output CSVs.
    combined
        Additionally write ``<out_prefix>_combined.csv`` concatenating all
        per-box rows. Per-box CSVs are always written.

    Returns
    -------
    Mapping from box key (e.g. ``"B2"`` single-plate, ``"plate2_B3"``
    multi-plate) to output Path. ``"_combined"`` / ``"_key_csv"`` /
    ``"_key_json"`` are added for those sidecars.

    Raises
    ------
    ConfigError
        On missing well column, unknown quadrant in input without a slot,
        duplicate rendered File Name, or malformed template.
    """
    if isinstance(samples, (str, Path)):
        df = pd.read_csv(samples)
    else:
        df = samples.copy()

    if well_col not in df.columns:
        raise ConfigError(f"well column {well_col!r} not in input: {list(df.columns)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date = config.date or datetime.today().strftime("%Y%m%d")
    empty_template = config.empty_file_name_template or _DEFAULT_EMPTY_TEMPLATE
    bracket_template = config.bracketing_blank_template or _DEFAULT_BRACKETING_TEMPLATE
    n_bracket = int(config.bracketing_blanks or 0)
    if n_bracket < 0 or n_bracket > _MAX_BRACKETING_BLANKS:
        raise ConfigError(
            f"bracketing_blanks must be between 0 and {_MAX_BRACKETING_BLANKS} "
            f"(row H has 12 positions; {_MAX_BRACKETING_BLANKS} lead + "
            f"{_MAX_BRACKETING_BLANKS} trail = 12, non-overlapping). Got {n_bracket}."
        )
    interspersed_template = config.interspersed_blank_template or _DEFAULT_INTERSPERSED_TEMPLATE
    n_interspersed = int(config.interspersed_blanks or 0)
    if n_interspersed < 0:
        raise ConfigError(f"interspersed_blanks must be >= 0, got {n_interspersed}.")
    group_separator_template = (
        config.group_separator_blank_template or _DEFAULT_INTERSPERSED_TEMPLATE
    )
    # Validate the interspersed pool up front (fails fast).
    if n_interspersed > 0:
        _interspersed_well_pool(n_bracket, n_interspersed)
    if config.group_by_column and config.group_by_column not in (
        # We'll also allow it after column_substitutions — check post-sub below.
        df.columns
    ):
        raise ConfigError(
            f"group_by_column {config.group_by_column!r} not in input columns: "
            f"{list(df.columns)}"
        )

    # Apply column substitutions (regex replace) before building templates.
    if config.column_substitutions:
        for col, sub in config.column_substitutions.items():
            if col not in df.columns:
                raise ConfigError(
                    f"column_substitutions references column {col!r} "
                    f"which is not in input: {list(df.columns)}"
                )
            pattern, replacement = sub
            df[col] = df[col].astype(str).str.replace(pattern, replacement, regex=True)

    # Plate handling: keep as Series so multi-plate groupby is natural.
    if plate_col and plate_col in df.columns:
        plate_series = df[plate_col]
    else:
        plate_series = pd.Series(1, index=df.index, name="plate")
        plate_col = "plate"

    # Annotate derived columns. Compute wells as str once, reuse.
    wells_str = df[well_col].astype(str)
    df = df.assign(
        plate=plate_series.values,
        well_384=wells_str,
        quadrant=wells_str.map(detect_quadrant),
        well_96=wells_str.map(map_384_to_96),
    )

    # Sanity: every present quadrant must have a slot.
    unknown = sorted(set(df["quadrant"].unique()) - set(config.autosampler_slots))
    if unknown:
        raise ConfigError(
            f"input has quadrants {unknown} but autosampler_slots only maps "
            f"{sorted(config.autosampler_slots)}. Silent skip would lose samples."
        )

    # Flag empties.
    if config.empty_marker is not None:
        em_col, em_val = config.empty_marker
        if em_col not in df.columns:
            raise ConfigError(
                f"empty_marker column {em_col!r} not in input columns: " f"{list(df.columns)}"
            )
        df["is_empty"] = df[em_col] == em_val
    else:
        df["is_empty"] = False

    multi_plate = df["plate"].nunique() > 1

    # Warn once up front if method is unset.
    if config.ms_method is None:
        logger.warning(
            "ms_method is None — 'Instrument Method' column will be empty. "
            "Fill it in before running Xcalibur."
        )
    if callable(config.ms_method) and (
        n_bracket > 0 or n_interspersed > 0 or config.group_separator_blanks
    ):
        logger.warning(
            "ms_method is a callable but synthetic BLANKs are configured "
            "(bracketing_blanks=%d, interspersed_blanks=%d, "
            "group_separator_blanks=%s): synthetic BLANKs have no row context "
            "and will use an empty Instrument Method.",
            n_bracket,
            n_interspersed,
            config.group_separator_blanks,
        )

    # Empty-DataFrame guard — fail fast before any disk work.
    if df.empty:
        raise ConfigError("input DataFrame has no rows — nothing to queue.")

    # Pre-validate worst-case pool demand per box: group_separator_blanks and
    # interspersed_blanks draw from the same well pool. Check up front so
    # nothing hits disk on failure (Phase 4b fix).
    if n_interspersed > 0 and config.group_by_column and config.group_separator_blanks:
        for _pv in sorted(df["plate"].unique()):
            for _quad in QUADRANT_ORDER:
                _grp = df[(df["plate"] == _pv) & (df["quadrant"] == _quad)]
                if _grp.empty:
                    continue
                _n_groups = _grp[config.group_by_column].nunique()
                _n_sep = max(0, _n_groups - 1)
                _total = _n_sep + n_interspersed
                if _total > 0:
                    _interspersed_well_pool(n_bracket, _total)

    outputs: dict[str, Path] = {}
    forbidden_hits: list[str] = []
    boxes: list[dict] = []  # per-box accumulated queue + key rows

    # Phase 1: iterate boxes and accumulate per-box lead/real/trail rows.
    plate_values = sorted(df["plate"].unique())
    for plate_idx, plate_value in enumerate(plate_values):
        for quad_idx, quadrant in enumerate(QUADRANT_ORDER):
            group = df[(df["plate"] == plate_value) & (df["quadrant"] == quadrant)]
            if group.empty:
                continue
            slot = config.autosampler_slots[quadrant]

            # RNG for this box: seeded so reruns are deterministic.
            seed = config.shuffle_seed + plate_idx * 100 + quad_idx
            rng = np.random.default_rng(seed)

            # Ordering: either group-by (ascending/descending, shuffle within
            # groups) or whole-box shuffle. group-by uses numeric sort when the
            # column coerces cleanly; otherwise falls back to lexical.
            if config.group_by_column:
                gcol = config.group_by_column
                try:
                    sort_key = pd.to_numeric(group[gcol], errors="raise")
                    tmp = group.assign(_gbkey=sort_key.values)
                    tmp = tmp.sort_values(
                        "_gbkey", ascending=config.group_by_ascending, kind="stable"
                    ).drop(columns="_gbkey")
                    group = tmp
                except (ValueError, TypeError):
                    group = group.sort_values(
                        gcol, ascending=config.group_by_ascending, kind="stable"
                    )
                if config.shuffle:
                    chunks = []
                    for _, sub in group.groupby(gcol, sort=False):
                        perm = rng.permutation(len(sub))
                        chunks.append(sub.iloc[perm])
                    group = pd.concat(chunks, ignore_index=True)
                else:
                    group = group.reset_index(drop=True)
            else:
                if config.shuffle:
                    perm = rng.permutation(len(group))
                    group = group.iloc[perm].reset_index(drop=True)
                else:
                    group = group.reset_index(drop=True)

            box_key = f"plate{plate_value}_{quadrant}" if multi_plate else quadrant

            # Group-transition blank pool: allocate from the interspersed pool
            # ahead of random interspersed wells so both kinds coexist.
            separator_well_queue: list[str] = []
            if config.group_by_column and config.group_separator_blanks:
                group_values = list(dict.fromkeys(group[config.group_by_column].tolist()))
                n_separators = max(0, len(group_values) - 1)
                total_non_bracket = n_separators + n_interspersed
                pool = _interspersed_well_pool(n_bracket, total_non_bracket)
                separator_well_queue = list(pool[:n_separators])
                # Remaining pool reserved for interspersed (allocated below).
                interspersed_well_pool_override = list(pool[n_separators:])
            else:
                interspersed_well_pool_override = None

            real_q: list[dict] = []
            real_k: list[dict] = []
            prev_group_value = None
            separator_idx = 0
            blank_method = _resolve_method_for_blank(config.ms_method)
            for _, row in group.iterrows():
                # Insert a separator BLANK on group boundary.
                if (
                    config.group_by_column
                    and config.group_separator_blanks
                    and prev_group_value is not None
                    and row[config.group_by_column] != prev_group_value
                ):
                    sep_well = separator_well_queue[separator_idx]
                    separator_idx += 1
                    sep_ctx = {
                        "date": date,
                        "plate": plate_value,
                        "quadrant": quadrant,
                        "well_96": sep_well,
                        "box_key": box_key,
                        "box": slot,
                        "position_tag": f"sep_{prev_group_value}to{row[config.group_by_column]}",
                    }
                    sep_name = _render_filename(group_separator_template, sep_ctx)
                    if _FORBIDDEN_FILENAME_CHARS.search(sep_name) or sep_name != sep_name.strip():
                        forbidden_hits.append(sep_name)
                    sep_position = f"S{slot}:{sep_well}"
                    real_q.append(
                        {
                            "File Name": sep_name,
                            "Path": config.path,
                            "Instrument Method": blank_method,
                            "Position": sep_position,
                        }
                    )
                    real_k.append(
                        {
                            "File Name": sep_name,
                            "plate": plate_value,
                            "well": "",
                            "well_384": "",
                            "quadrant": quadrant,
                            "well_96": sep_well,
                            "is_empty": True,
                            "slot": slot,
                            "Position": sep_position,
                            "box_key": box_key,
                            "instrument_method": blank_method,
                            "queue_date": date,
                            "blank_kind": "group_separator",
                        }
                    )

                ctx = row.to_dict()
                ctx.update(
                    date=date,
                    well_384=row["well_384"],
                    well_96=row["well_96"],
                    quadrant=quadrant,
                    box=slot,
                    plate=plate_value,
                )
                is_empty = bool(row["is_empty"])
                template = empty_template if is_empty else config.file_name_template
                file_name = _render_filename(template, ctx)

                if _FORBIDDEN_FILENAME_CHARS.search(file_name) or file_name != file_name.strip():
                    forbidden_hits.append(file_name)

                method = _resolve_method(row, config.ms_method)
                position = f"S{slot}:{row['well_96']}"

                real_q.append(
                    {
                        "File Name": file_name,
                        "Path": config.path,
                        "Instrument Method": method,
                        "Position": position,
                    }
                )
                key_row = dict(row)
                key_row.update(
                    {
                        "File Name": file_name,
                        "slot": slot,
                        "Position": position,
                        "box_key": box_key,
                        "instrument_method": method,
                        "queue_date": date,
                        "blank_kind": "empty_well" if is_empty else "",
                    }
                )
                real_k.append(key_row)
                if config.group_by_column:
                    prev_group_value = row[config.group_by_column]

            # Interspersed BLANKs: synthesize N rows and interleave evenly
            # into the shuffled real-sample stream (before bracketing is added).
            if n_interspersed > 0:
                if interspersed_well_pool_override is not None:
                    blank_wells = interspersed_well_pool_override
                else:
                    blank_wells = _interspersed_well_pool(n_bracket, n_interspersed)
                n_total = len(real_q) + n_interspersed
                insert_positions = _interspersed_insert_positions(n_interspersed, n_total, rng)
                merged_q: list[dict] = []
                merged_k: list[dict] = []
                blank_ptr = 0
                sample_ptr = 0
                for slot_idx in range(n_total):
                    if slot_idx in insert_positions:
                        well_96 = blank_wells[blank_ptr]
                        position_tag = f"interspersed{blank_ptr + 1}"
                        ctx = {
                            "date": date,
                            "plate": plate_value,
                            "quadrant": quadrant,
                            "well_96": well_96,
                            "box_key": box_key,
                            "box": slot,
                            "position_tag": position_tag,
                        }
                        file_name = _render_filename(interspersed_template, ctx)
                        if (
                            _FORBIDDEN_FILENAME_CHARS.search(file_name)
                            or file_name != file_name.strip()
                        ):
                            forbidden_hits.append(file_name)
                        position = f"S{slot}:{well_96}"
                        merged_q.append(
                            {
                                "File Name": file_name,
                                "Path": config.path,
                                "Instrument Method": blank_method,
                                "Position": position,
                            }
                        )
                        merged_k.append(
                            {
                                "File Name": file_name,
                                "plate": plate_value,
                                "well": "",
                                "well_384": "",
                                "quadrant": quadrant,
                                "well_96": well_96,
                                "is_empty": True,
                                "slot": slot,
                                "Position": position,
                                "box_key": box_key,
                                "instrument_method": blank_method,
                                "queue_date": date,
                                "blank_kind": "interspersed",
                            }
                        )
                        blank_ptr += 1
                    else:
                        merged_q.append(real_q[sample_ptr])
                        merged_k.append(real_k[sample_ptr])
                        sample_ptr += 1
                real_q = merged_q
                real_k = merged_k

            # Bracketing BLANKs (synthetic, row H of 96-well box).
            if n_bracket > 0:
                lead_wells, trail_wells = _bracketing_wells(n_bracket)
                blank_method = _resolve_method_for_blank(config.ms_method)
                lead_q, lead_k = _make_bracketing_rows(
                    lead_wells,
                    "lead",
                    box_key=box_key,
                    slot=slot,
                    plate_value=plate_value,
                    quadrant=quadrant,
                    date=date,
                    path_value=config.path,
                    method_value=blank_method,
                    template=bracket_template,
                )
                trail_q, trail_k = _make_bracketing_rows(
                    trail_wells,
                    "trail",
                    box_key=box_key,
                    slot=slot,
                    plate_value=plate_value,
                    quadrant=quadrant,
                    date=date,
                    path_value=config.path,
                    method_value=blank_method,
                    template=bracket_template,
                )
                # Check the rendered bracket names for forbidden chars too.
                for r in lead_q + trail_q:
                    name = r["File Name"]
                    if _FORBIDDEN_FILENAME_CHARS.search(name) or name != name.strip():
                        forbidden_hits.append(name)
            else:
                lead_q, lead_k, trail_q, trail_k = [], [], [], []

            boxes.append(
                {
                    "key": box_key,
                    "plate": plate_value,
                    "quadrant": quadrant,
                    "slot": slot,
                    "lead_q": lead_q,
                    "lead_k": lead_k,
                    "real_q": real_q,
                    "real_k": real_k,
                    "trail_q": trail_q,
                    "trail_k": trail_k,
                }
            )

    # Phase 2a: collect all box row + key rows; validate BEFORE any disk write
    # (Phase 1.6 fix: previously files were written first, so a duplicate-
    # filename ConfigError left partial per-box CSVs without the sidecar).
    key_rows: list[dict] = []
    all_names: list[str] = []
    for box in boxes:
        per_box_k = box["lead_k"] + box["real_k"] + box["trail_k"]
        # Assign box_row_index (per-box file position).
        for idx, k in enumerate(per_box_k):
            k["box_row_index"] = idx
        all_names.extend(r["File Name"] for r in (box["lead_q"] + box["real_q"] + box["trail_q"]))
        key_rows.extend(per_box_k)

    # Duplicate-filename detection across ALL output rows — before any write.
    if len(set(all_names)) != len(all_names):
        seen: dict[str, int] = {}
        for n in all_names:
            seen[n] = seen.get(n, 0) + 1
        dupes = sorted(n for n, c in seen.items() if c > 1)
        raise ConfigError(
            f"file_name_template produced duplicate File Name values: {dupes[:5]}"
            f"{' ...' if len(dupes) > 5 else ''}. Xcalibur overwrites silently — "
            f"add a disambiguating field (e.g. {{well_384}} or {{replicate}})."
        )

    if forbidden_hits:
        example = forbidden_hits[0]
        logger.warning(
            "%d rendered File Name value(s) contain characters Xcalibur/Windows "
            'may reject (<>:"/\\|?* or leading/trailing whitespace). Example: %r',
            len(forbidden_hits),
            example,
        )

    # Phase 2b: all validation passed — write per-box CSVs.
    for box in boxes:
        per_box_q = box["lead_q"] + box["real_q"] + box["trail_q"]
        box_path = out_dir / f"{out_prefix}_{box['key']}.csv"
        _write_queue_csv(box_path, per_box_q, config.bracket_type)
        outputs[box["key"]] = box_path
        logger.info(
            "wrote %d rows to %s (plate=%s quadrant=%s slot=S%d, bracket=%d)",
            len(per_box_q),
            box_path,
            box["plate"],
            box["quadrant"],
            box["slot"],
            n_bracket,
        )

    # Combined CSV: for each box emit lead + real; only the last box contributes
    # its trailing block. Result: a single set of bracketing blanks at each box
    # boundary (incl. start) + one set at the very end.
    if combined:
        combined_q: list[dict] = []
        for i, box in enumerate(boxes):
            combined_q.extend(box["lead_q"] + box["real_q"])
            if i == len(boxes) - 1:
                combined_q.extend(box["trail_q"])
        combined_path = out_dir / f"{out_prefix}_combined.csv"
        _write_queue_csv(combined_path, combined_q, config.bracket_type)
        outputs["_combined"] = combined_path
        logger.info("wrote %d rows to %s (combined)", len(combined_q), combined_path)

    # Sample key (always). Reorder so File Name — the join key — comes first.
    key_df = pd.DataFrame(key_rows)
    ordered_cols = ["File Name"] + [c for c in key_df.columns if c != "File Name"]
    key_df = key_df[ordered_cols]
    key_csv = out_dir / f"{out_prefix}_key.csv"
    key_df.to_csv(key_csv, index=False)
    outputs["_key_csv"] = key_csv

    key_json = out_dir / f"{out_prefix}_key.json"
    atomic_json_dump(key_df.to_dict(orient="records"), key_json)
    outputs["_key_json"] = key_json
    logger.info("wrote sample key (%d rows) to %s + .json", len(key_df), key_csv)

    return outputs


def _write_queue_csv(path: Path, rows: list[dict], bracket_type: int) -> None:
    """Write a Thermo queue CSV: ``Bracket Type=N`` header + 4-column table."""
    frame = pd.DataFrame(rows, columns=["File Name", "Path", "Instrument Method", "Position"])
    with open(path, "w") as f:
        f.write(f"Bracket Type={bracket_type}\n")
    frame.to_csv(path, mode="a", index=False)
