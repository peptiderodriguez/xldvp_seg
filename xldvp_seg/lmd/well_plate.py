"""384-well plate serpentine well generation.

Generates well addresses across 4 quadrants (B2, B3, C3, C2) of a 384-well
plate in serpentine order.  Supports single-plate (308 max) and multi-plate
overflow, plus QC empty well insertion.

Quadrant layout (excludes outer wells: row A, row P, col 1, col 24):
  B2: even rows (B,D,F,H,J,L,N) × even cols (2,4,...,22) = 77 wells
  B3: even rows × odd cols  (3,5,...,23) = 77 wells
  C2: odd rows  (C,E,G,I,K,M,O) × even cols = 77 wells
  C3: odd rows  × odd cols  = 77 wells

Default quadrant traversal order: B2 -> B3 -> C3 -> C2
Each subsequent quadrant starts from the corner nearest the last well of the
previous quadrant (minimizes laser head / stage travel).
"""

import math

from xldvp_seg.exceptions import ConfigError

WELLS_PER_PLATE = 308  # 4 quadrants × 77 wells

EVEN_ROWS = ["B", "D", "F", "H", "J", "L", "N"]
ODD_ROWS = ["C", "E", "G", "I", "K", "M", "O"]
EVEN_COLS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
ODD_COLS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

QUAD_MAP = {
    "B2": (EVEN_ROWS, EVEN_COLS),
    "B3": (EVEN_ROWS, ODD_COLS),
    "C2": (ODD_ROWS, EVEN_COLS),
    "C3": (ODD_ROWS, ODD_COLS),
}

QUADRANT_ORDER = ["B2", "B3", "C3", "C2"]


def generate_quadrant_serpentine(quadrant, start_corner="TL"):
    """Generate wells for one 384-well quadrant in serpentine order.

    Parameters
    ----------
    quadrant : str
        One of 'B2', 'B3', 'C2', 'C3'.
    start_corner : str
        Where the serpentine begins: 'TL', 'TR', 'BL', or 'BR'.

    Returns
    -------
    list[str]
        77 well addresses in serpentine order (e.g. ['B2', 'B4', ..., 'N22']).
    """
    if quadrant not in QUAD_MAP:
        raise ConfigError(f"Unknown quadrant: {quadrant}")
    rows, cols = QUAD_MAP[quadrant]

    if start_corner == "TL":
        row_order = rows
        first_row_left_to_right = True
    elif start_corner == "TR":
        row_order = rows
        first_row_left_to_right = False
    elif start_corner == "BL":
        row_order = list(reversed(rows))
        first_row_left_to_right = True
    elif start_corner == "BR":
        row_order = list(reversed(rows))
        first_row_left_to_right = False
    else:
        raise ConfigError(f"Unknown start_corner: {start_corner}")

    wells = []
    for i, row in enumerate(row_order):
        if i % 2 == 0:
            col_order = cols if first_row_left_to_right else list(reversed(cols))
        else:
            col_order = list(reversed(cols)) if first_row_left_to_right else cols
        for col in col_order:
            wells.append(f"{row}{col}")
    return wells


def _nearest_corner(last_well):
    """Determine the nearest start corner from the previous quadrant's last well."""
    prev_row, prev_col = last_well[0], int(last_well[1:])
    top_rows = set("BCDEFGH")
    is_top = prev_row in top_rows
    is_left = prev_col <= 12

    if is_top and is_left:
        return "TL"
    elif is_top and not is_left:
        return "TR"
    elif not is_top and is_left:
        return "BL"
    else:
        return "BR"


def generate_plate_wells(n_wells, start_quadrant="B2"):
    """Generate wells for a single 384-well plate.

    Traverses quadrants in order (default B2 -> B3 -> C3 -> C2), starting
    each subsequent quadrant from the corner nearest the previous quadrant's
    last well.

    Parameters
    ----------
    n_wells : int
        Number of wells needed (max 308).
    start_quadrant : str
        First quadrant to fill (default 'B2').

    Returns
    -------
    list[str]
        Well addresses in serpentine order, truncated to *n_wells*.

    Raises
    ------
    ValueError
        If *n_wells* > 308.
    """
    if n_wells <= 0:
        return []
    if n_wells > WELLS_PER_PLATE:
        raise ConfigError(
            f"Requested {n_wells} wells but a single 384-well plate only has "
            f"{WELLS_PER_PLATE} usable wells (4 quadrants x 77). "
            f"Use generate_multiplate_wells() for overflow."
        )

    idx = QUADRANT_ORDER.index(start_quadrant)
    order = QUADRANT_ORDER[idx:] + QUADRANT_ORDER[:idx]

    all_wells = []
    for i, quad in enumerate(order):
        if i == 0:
            wells = generate_quadrant_serpentine(quad, start_corner="TL")
        else:
            wells = generate_quadrant_serpentine(quad, start_corner=_nearest_corner(all_wells[-1]))
        all_wells.extend(wells)
        if len(all_wells) >= n_wells:
            return all_wells[:n_wells]

    return all_wells[:n_wells]


def generate_multiplate_wells(n_wells, start_quadrant="B2"):
    """Generate well assignments across as many plates as needed.

    Each plate holds up to 308 wells. When a plate fills, the next plate
    starts fresh from TL of *start_quadrant*.

    Parameters
    ----------
    n_wells : int
        Total wells needed.
    start_quadrant : str
        First quadrant on each plate (default 'B2').

    Returns
    -------
    list[tuple[int, str]]
        (plate_number, well_address) tuples.  Plate numbers are 1-based.
    """
    if n_wells <= 0:
        return []

    result = []
    remaining = n_wells
    plate = 1
    while remaining > 0:
        batch = min(remaining, WELLS_PER_PLATE)
        wells = generate_plate_wells(batch, start_quadrant=start_quadrant)
        result.extend((plate, w) for w in wells)
        remaining -= batch
        plate += 1
    return result


def insert_empty_wells(plate_wells, n_samples, empty_pct=10.0, seed=42):
    """Insert empty (QC) wells evenly across the well sequence.

    Divides the total well sequence into *n_empty* equal segments and places
    one empty well randomly within each segment.  Collision retry guarantees
    exactly *n_empty* distinct positions.

    Parameters
    ----------
    plate_wells : list
        Well sequence (list of well-address strings or (plate, well) tuples).
    n_samples : int
        Number of actual samples (used to compute empty count).
    empty_pct : float
        Empty wells as percentage of *n_samples* (default 10%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list, set[int]]
        (plate_wells, empty_positions) — the same *plate_wells* list and the
        set of 0-based indices that should be left empty.
    """
    import numpy as np

    n_empty = max(1, math.ceil(n_samples * empty_pct / 100))
    n_total = len(plate_wells)

    if n_empty >= n_total:
        raise ValueError(f"Cannot insert {n_empty} empty wells into {n_total} total wells")

    rng = np.random.default_rng(seed)
    segment_size = n_total / n_empty
    empty_positions = set()
    for i in range(n_empty):
        seg_start = int(i * segment_size)
        seg_end = max(seg_start + 1, int((i + 1) * segment_size))
        pos = int(rng.integers(seg_start, seg_end))
        while pos in empty_positions:
            pos = int(rng.integers(0, n_total))
        empty_positions.add(pos)

    return plate_wells, empty_positions
