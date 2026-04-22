"""Group color assignment and palettes for spatial viewers."""

import colorsys

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Binary positive/negative
BINARY_COLORS = {"positive": "#ff4444", "negative": "#4488ff"}

# 4-group palette (multi-marker profiles)
QUAD_COLORS = ["#ff4444", "#4488ff", "#44cc44", "#ff8844"]

# 20-color maximally-distinct palette for N groups
AUTO_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabebe",
    "#469990",
    "#e6beff",
    "#9a6324",
    "#ffe119",
    "#aaffc3",
    "#800000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#808000",
    "#ff69b4",
]


def hsl_palette(n):
    """Generate n maximally-separated HSL colors as hex strings."""
    colors = []
    for i in range(n):
        h = (i * 360 / n) % 360
        s = 70 + (i % 3) * 10  # 70-90% saturation
        l = 55 + (i % 2) * 10  # 55-65% lightness
        colors.append(_hsl_to_hex(h, s, l))
    return colors


def _hsl_to_hex(h, s, l):
    """Convert HSL (h=0-360, s=0-100, l=0-100) to hex color string."""
    s /= 100
    l /= 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    ri = int((r + m) * 255)
    gi = int((g + m) * 255)
    bi = int((b + m) * 255)
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def assign_group_colors(slides_data):
    """Assign colors to groups across all slides.

    - 2 groups with positive/negative: red/blue
    - 2 arbitrary groups: red/blue
    - 4 groups: red/blue/green/orange
    - N groups (N <= 20): auto palette
    - N groups (N > 20): HSL-generated palette

    Args:
        slides_data: List of (name, data) tuples where data has 'groups' key.

    Returns:
        color_map: Dict of group_label -> hex color string.
    """
    all_groups = set()
    for _, data in slides_data:
        for g in data["groups"]:
            all_groups.add(g["label"])

    n = len(all_groups)
    sorted_groups = sorted(all_groups)

    if all_groups == {"positive", "negative"}:
        color_map = dict(BINARY_COLORS)
    elif n <= 2:
        palette = ["#ff4444", "#4488ff"]
        color_map = {lbl: palette[i] for i, lbl in enumerate(sorted_groups)}
    elif n <= 4:
        color_map = {lbl: QUAD_COLORS[i] for i, lbl in enumerate(sorted_groups)}
    elif n <= 20:
        color_map = {lbl: AUTO_COLORS[i] for i, lbl in enumerate(sorted_groups)}
    else:
        palette = hsl_palette(n)
        color_map = {lbl: palette[i] for i, lbl in enumerate(sorted_groups)}

    # Apply colors to group dicts
    for _, data in slides_data:
        for g in data["groups"]:
            g["color"] = color_map[g["label"]]

    return color_map


def shuffled_hsv_palette(
    k: int,
    *,
    seed: int = 0,
    saturation: float = 0.85,
    value: float = 0.95,
) -> np.ndarray:
    """Return (k, 3) uint8 RGB palette with shuffled HSV hues.

    Adjacent indices get dissimilar hues, so neighboring groups in any
    plot don't visually blur together. Used for manifold-sampling group
    coloring (1000 distinct groups on a single UMAP).

    Args:
        k: number of distinct colors to generate.
        seed: RNG seed for the hue permutation (deterministic).
        saturation: HSV saturation (0..1), default 0.85.
        value: HSV value/brightness (0..1), default 0.95.

    Returns:
        (k, 3) uint8 RGB array.
    """
    rng = np.random.default_rng(seed)
    hues = rng.permutation(k) / k
    rgb = np.empty((k, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(float(h), saturation, value)
        rgb[i, 0] = int(r * 255)
        rgb[i, 1] = int(g * 255)
        rgb[i, 2] = int(b * 255)
    return rgb
