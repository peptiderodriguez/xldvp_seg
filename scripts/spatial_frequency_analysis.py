#!/usr/bin/env python3
"""
Spatial frequency analysis of cell density distributions.

Reads detection JSON from run_segmentation.py and produces:
- Per-channel 1D density profiles along transects
- CWT scalograms (Morlet wavelet) showing spatial frequency vs position
- Boundary detection from scalogram gradients
- 2D intensity-weighted density heatmaps

Works with any cell type output (tissue_pattern, islet, NMJ, etc.).
Requires: pywt (PyWavelets), scipy, matplotlib, numpy.

Usage:
    python scripts/spatial_frequency_analysis.py \\
        --detections tissue_pattern_detections.json \\
        --channels 0,1,2,3 \\
        --corridor-width-um 200 \\
        --bin-width-um 10 \\
        --output-dir freq_output/

    # With manual transects:
    python scripts/spatial_frequency_analysis.py \\
        --detections tissue_pattern_detections.json \\
        --transects transects.json \\
        --output-dir freq_output/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


def load_detections(path):
    """Load detection JSON and extract positions + per-channel intensities."""
    with open(path) as f:
        data = json.load(f)

    # Handle both flat list and wrapped format
    if isinstance(data, dict):
        detections = data.get('detections', data.get('features', []))
    else:
        detections = data

    cells = []
    for det in detections:
        feats = det.get('features', det)
        center_um = det.get('global_center_um') or feats.get('global_center_um')
        if center_um is None:
            continue

        cell = {
            'x_um': float(center_um[0]),
            'y_um': float(center_um[1]),
            'area': float(feats.get('area', 0)),
        }

        # RF prediction score (for filtering)
        rf = det.get('rf_prediction')
        if rf is None:
            rf = feats.get('rf_prediction')
        if rf is not None:
            cell['rf_prediction'] = float(rf)

        # Collect per-channel mean intensities (ch0_mean, ch1_mean, ...)
        for key, val in feats.items():
            if key.startswith('ch') and key.endswith('_mean'):
                cell[key] = float(val) if val is not None else 0.0

        cells.append(cell)

    print(f"Loaded {len(cells)} cells with positions")
    return cells


def compute_tissue_bbox(cells):
    """Compute bounding box of all cell positions in um."""
    xs = np.array([c['x_um'] for c in cells])
    ys = np.array([c['y_um'] for c in cells])
    return {
        'x_min': xs.min(), 'x_max': xs.max(),
        'y_min': ys.min(), 'y_max': ys.max(),
        'width': xs.max() - xs.min(),
        'height': ys.max() - ys.min(),
    }


def generate_auto_transects(bbox, n_transects=5, margin_frac=0.05):
    """Generate grid of transects across tissue.

    Creates transects along the shorter axis (perpendicular to the longest
    dimension), evenly spaced along the longest axis.

    Args:
        bbox: Dict with x_min, x_max, y_min, y_max, width, height
        n_transects: Number of transects to generate
        margin_frac: Fraction of extent to skip at edges

    Returns:
        List of transect dicts with 'start_um', 'end_um', 'name'
    """
    transects = []
    w, h = bbox['width'], bbox['height']

    if w >= h:
        # Longest axis = X → transects run vertically (Y direction)
        margin = w * margin_frac
        positions = np.linspace(bbox['x_min'] + margin, bbox['x_max'] - margin, n_transects)
        for i, x_pos in enumerate(positions):
            transects.append({
                'name': f'transect_{i}',
                'start_um': [float(x_pos), float(bbox['y_min'])],
                'end_um': [float(x_pos), float(bbox['y_max'])],
            })
    else:
        # Longest axis = Y → transects run horizontally (X direction)
        margin = h * margin_frac
        positions = np.linspace(bbox['y_min'] + margin, bbox['y_max'] - margin, n_transects)
        for i, y_pos in enumerate(positions):
            transects.append({
                'name': f'transect_{i}',
                'start_um': [float(bbox['x_min']), float(y_pos)],
                'end_um': [float(bbox['x_max']), float(y_pos)],
            })

    return transects


def load_transects(path):
    """Load transects from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get('transects', data)


def project_cells_onto_transect(cells, transect, corridor_width_um):
    """Project cell positions onto a transect line.

    Args:
        cells: List of cell dicts with 'x_um', 'y_um' and channel means
        transect: Dict with 'start_um' [x,y] and 'end_um' [x,y]
        corridor_width_um: Full width of corridor around transect (cells outside excluded)

    Returns:
        Dict with:
            'distances': 1D array of projected distances along transect (um)
            'channel_intensities': Dict of channel_key -> intensity array
            'transect_length': Total transect length in um
    """
    start = np.array(transect['start_um'], dtype=float)
    end = np.array(transect['end_um'], dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1.0:
        return {'distances': np.array([]), 'channel_intensities': {}, 'transect_length': 0}

    unit = direction / length
    normal = np.array([-unit[1], unit[0]])  # perpendicular

    positions = np.array([[c['x_um'], c['y_um']] for c in cells])
    relative = positions - start

    # Project onto transect
    along = relative @ unit       # distance along transect
    across = relative @ normal    # distance from transect line

    # Filter: within corridor and within transect bounds
    half_corridor = corridor_width_um / 2.0
    mask = (np.abs(across) <= half_corridor) & (along >= 0) & (along <= length)

    distances = along[mask]

    # Collect channel intensities for selected cells
    selected_cells = [c for c, m in zip(cells, mask) if m]
    channel_keys = set()
    for c in selected_cells:
        for k in c:
            if k.startswith('ch') and k.endswith('_mean'):
                channel_keys.add(k)

    channel_intensities = {}
    for key in sorted(channel_keys):
        channel_intensities[key] = np.array([c.get(key, 0) for c in selected_cells])

    return {
        'distances': distances,
        'channel_intensities': channel_intensities,
        'transect_length': length,
    }


def build_density_profile(distances, transect_length, bin_width_um, weights=None):
    """Build 1D density profile from projected cell positions.

    Uses KDE for smooth density estimation, falling back to histogram
    if too few cells.

    Args:
        distances: 1D array of positions along transect (um)
        transect_length: Total transect length (um)
        bin_width_um: Spatial resolution of output profile
        weights: Optional intensity weights for each cell

    Returns:
        Tuple of (positions_um, density) arrays
    """
    n_bins = max(10, int(transect_length / bin_width_um))
    positions = np.linspace(0, transect_length, n_bins)

    if len(distances) < 5:
        return positions, np.zeros(n_bins)

    try:
        # Bandwidth: ~3 bin widths of smoothing in data units
        dist_std = np.std(distances)
        if dist_std > 0:
            bw = bin_width_um * 3.0 / dist_std
        else:
            bw = 0.1  # fallback
        kde = gaussian_kde(distances, bw_method=bw, weights=weights)
        density = kde(positions)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: simple histogram
        # histogram returns n_bins-1 values for n_bins edges, so use n_bins bins
        # to get n_bins values, then interpolate to match positions array length
        hist, edges = np.histogram(distances, bins=n_bins, range=(0, transect_length),
                                   weights=weights)
        # bin centers
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        density = np.interp(positions, bin_centers, hist.astype(float))

    return positions, density


def cwt_scalogram(density, positions_um, min_period_um=20, max_period_um=None):
    """Compute CWT scalogram using Morlet wavelet.

    Args:
        density: 1D density profile
        positions_um: Corresponding position values (um)
        min_period_um: Minimum spatial period to analyze (um)
        max_period_um: Maximum spatial period (default: half the profile length)

    Returns:
        Dict with 'coefficients', 'frequencies', 'periods_um', 'power'
    """
    total_length = positions_um[-1] - positions_um[0]
    dx = total_length / (len(positions_um) - 1)  # spacing in um

    if max_period_um is None:
        max_period_um = total_length / 2.0

    # Morlet wavelet: scale ≈ period / (2π * center_freq)
    # For pywt morl, center frequency ≈ 0.8125 Hz
    center_freq = pywt.central_frequency('morl')

    # Convert desired periods (um) to wavelet scales
    min_scale = max(1.0, center_freq * min_period_um / dx)
    max_scale = max(min_scale + 1.0, center_freq * max_period_um / dx)

    n_scales = min(200, int(max_scale - min_scale) + 1)
    if n_scales < 10:
        n_scales = 50
    scales = np.linspace(min_scale, max_scale, n_scales)

    # CWT
    coefficients, frequencies = pywt.cwt(density, scales, 'morl', sampling_period=dx)
    power = np.abs(coefficients) ** 2

    # Convert frequencies (cycles/um) to periods (um)
    periods_um = 1.0 / (frequencies + 1e-12)

    return {
        'coefficients': coefficients,
        'frequencies': frequencies,
        'periods_um': periods_um,
        'power': power,
        'scales': scales,
    }


def detect_boundaries(scalogram, positions_um, min_prominence_frac=0.1,
                       min_distance_um=50):
    """Detect spatial boundaries from scalogram gradient.

    Boundaries appear as sharp changes in the scalogram across multiple scales.

    Args:
        scalogram: Dict from cwt_scalogram()
        positions_um: Position array (um)
        min_prominence_frac: Minimum peak prominence as fraction of max
        min_distance_um: Minimum distance between boundaries (um)

    Returns:
        Dict with 'positions_um', 'strengths', 'boundary_signal'
    """
    power = scalogram['power']

    # Gradient of power along position axis, summed across scales
    grad = np.gradient(power, axis=1)
    boundary_signal = np.sum(np.abs(grad), axis=0)
    boundary_signal = gaussian_filter1d(boundary_signal, sigma=3)

    # Normalize
    if boundary_signal.max() > 0:
        boundary_signal_norm = boundary_signal / boundary_signal.max()
    else:
        return {'positions_um': np.array([]), 'strengths': np.array([]),
                'boundary_signal': boundary_signal}

    # Peak detection
    dx = positions_um[1] - positions_um[0] if len(positions_um) > 1 else 1.0
    min_distance_bins = max(1, int(min_distance_um / dx))

    peaks, props = find_peaks(
        boundary_signal_norm,
        distance=min_distance_bins,
        prominence=min_prominence_frac,
    )

    boundary_positions = positions_um[peaks]
    boundary_strengths = props['prominences']

    return {
        'positions_um': boundary_positions,
        'strengths': boundary_strengths,
        'boundary_signal': boundary_signal,
    }


def build_2d_heatmap(cells, channel_key, bin_size_um=50, sigma_bins=1.5):
    """Build 2D intensity-weighted density heatmap.

    Args:
        cells: List of cell dicts
        channel_key: Channel mean key (e.g., 'ch0_mean') or None for raw density
        bin_size_um: Spatial bin size in um
        sigma_bins: Gaussian smoothing sigma in bin units

    Returns:
        Dict with 'heatmap' (2D array), 'x_edges', 'y_edges', 'extent'
    """
    xs = np.array([c['x_um'] for c in cells])
    ys = np.array([c['y_um'] for c in cells])

    if channel_key is not None:
        weights = np.array([c.get(channel_key, 0) for c in cells])
    else:
        weights = None

    x_bins = max(10, int((xs.max() - xs.min()) / bin_size_um))
    y_bins = max(10, int((ys.max() - ys.min()) / bin_size_um))

    heatmap, x_edges, y_edges = np.histogram2d(
        xs, ys, bins=[x_bins, y_bins], weights=weights,
    )

    # Smooth
    if sigma_bins > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma_bins)

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    return {
        'heatmap': heatmap,
        'x_edges': x_edges,
        'y_edges': y_edges,
        'extent': extent,
    }


# ── Plotting ───────────────────────────────────────────────────────────

def plot_scalogram(scalogram, positions_um, channel_name, boundaries=None,
                   density=None, output_path=None):
    """Plot CWT scalogram with optional boundary lines and density profile."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 1]})

    # Top: density profile
    if density is not None:
        axes[0].plot(positions_um, density, 'k-', linewidth=1.0)
        axes[0].fill_between(positions_um, density, alpha=0.3, color='gray')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'CWT Scalogram — {channel_name}')

    # Middle: scalogram (pcolormesh for correct log-scale y-axis)
    periods = scalogram['periods_um']
    im = axes[1].pcolormesh(
        positions_um, periods, scalogram['power'],
        cmap='viridis', shading='auto',
    )
    axes[1].set_ylabel('Spatial period (um)')
    axes[1].set_yscale('log')
    axes[1].set_ylim(periods.min(), periods.max())
    plt.colorbar(im, ax=axes[1], label='Power', pad=0.01)

    # Bottom: boundary signal
    if boundaries is not None:
        axes[2].plot(positions_um, boundaries['boundary_signal'], 'b-', linewidth=0.8)
        axes[2].set_ylabel('Boundary\nstrength')

        for bp in boundaries['positions_um']:
            for ax in axes:
                ax.axvline(bp, color='red', linestyle='--', alpha=0.6, linewidth=0.8)

    axes[2].set_xlabel('Position along transect (um)')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_density_profiles(profiles, positions_um, transect_name, output_path=None):
    """Plot overlaid density profiles for all channels."""
    fig, ax = plt.subplots(figsize=(14, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(profiles), 1)))
    for (ch_name, density), color in zip(profiles.items(), colors):
        ax.plot(positions_um, density / (density.max() + 1e-12), label=ch_name,
                linewidth=1.2, color=color)

    ax.set_xlabel('Position along transect (um)')
    ax.set_ylabel('Normalized density')
    ax.set_title(f'Channel density profiles — {transect_name}')
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


def plot_2d_heatmap(heatmap_data, channel_name, output_path=None):
    """Plot 2D density heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        heatmap_data['heatmap'].T, origin='lower',
        extent=heatmap_data['extent'],
        aspect='equal', cmap='hot', interpolation='bilinear',
    )
    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'2D density — {channel_name}')
    plt.colorbar(im, ax=ax, label='Intensity-weighted density')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Spatial frequency analysis of cell density distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--detections', required=True,
                        help='Path to detection JSON from run_segmentation.py')
    parser.add_argument('--channels', type=str, default=None,
                        help='Comma-separated channel indices to analyze (default: all found)')
    parser.add_argument('--transects', type=str, default=None,
                        help='Path to transects JSON (default: auto-generate)')
    parser.add_argument('--n-transects', type=int, default=5,
                        help='Number of auto-generated transects (default: 5)')
    parser.add_argument('--corridor-width-um', type=float, default=200,
                        help='Corridor width around each transect in um (default: 200)')
    parser.add_argument('--bin-width-um', type=float, default=10,
                        help='Spatial bin width for density profiles in um (default: 10)')
    parser.add_argument('--min-period-um', type=float, default=20,
                        help='Minimum spatial period for CWT in um (default: 20)')
    parser.add_argument('--max-period-um', type=float, default=None,
                        help='Maximum spatial period for CWT in um (default: half transect length)')
    parser.add_argument('--heatmap-bin-um', type=float, default=50,
                        help='Bin size for 2D heatmaps in um (default: 50)')
    parser.add_argument('--min-score', type=float, default=0.0,
                        help='Minimum RF score to include (default: 0.0 = all)')
    parser.add_argument('--output-dir', type=str, default='freq_output',
                        help='Output directory (default: freq_output)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detections
    print(f"Loading detections from {args.detections}...")
    cells = load_detections(args.detections)

    if not cells:
        print("ERROR: No cells with positions found in detection JSON")
        sys.exit(1)

    # Filter by score
    if args.min_score > 0:
        before = len(cells)
        cells = [c for c in cells if c.get('rf_prediction', 1.0) >= args.min_score]
        print(f"Score filter (>= {args.min_score}): {before} -> {len(cells)} cells")

    # Determine channels to analyze
    all_ch_keys = set()
    for c in cells:
        for k in c:
            if k.startswith('ch') and k.endswith('_mean'):
                all_ch_keys.add(k)
    all_ch_keys = sorted(all_ch_keys)

    if args.channels:
        selected_indices = [int(x) for x in args.channels.split(',')]
        ch_keys = [f'ch{i}_mean' for i in selected_indices if f'ch{i}_mean' in all_ch_keys]
    else:
        ch_keys = all_ch_keys

    print(f"Channels to analyze: {ch_keys}")
    if not ch_keys:
        print("WARNING: No channel intensity features found. Will analyze raw cell density only.")
        ch_keys = [None]  # sentinel for unweighted density

    # Compute tissue bounding box
    bbox = compute_tissue_bbox(cells)
    print(f"Tissue extent: {bbox['width']:.0f} x {bbox['height']:.0f} um")

    # Load or generate transects
    if args.transects:
        transects = load_transects(args.transects)
        print(f"Loaded {len(transects)} transects from {args.transects}")
    else:
        transects = generate_auto_transects(bbox, n_transects=args.n_transects)
        print(f"Auto-generated {len(transects)} transects")

    # Save transects for reproducibility
    with open(output_dir / 'transects_used.json', 'w') as f:
        json.dump({'transects': transects}, f, indent=2)

    # ── Per-transect analysis ───────────────────────────────────────

    all_boundaries = {}

    for t_idx, transect in enumerate(transects):
        t_name = transect.get('name', f'transect_{t_idx}')
        print(f"\n--- {t_name} ---")

        projection = project_cells_onto_transect(
            cells, transect, args.corridor_width_um,
        )

        if len(projection['distances']) < 10:
            print(f"  Skipping: only {len(projection['distances'])} cells in corridor")
            continue

        print(f"  {len(projection['distances'])} cells in corridor "
              f"(length: {projection['transect_length']:.0f} um)")

        # Build density profiles per channel
        profiles = {}
        positions = None
        for ch_key in ch_keys:
            if ch_key is None:
                weights = None
                ch_label = 'cell_density'
            else:
                weights = projection['channel_intensities'].get(ch_key)
                ch_label = ch_key.replace('_mean', '')

            pos, density = build_density_profile(
                projection['distances'],
                projection['transect_length'],
                args.bin_width_um,
                weights=weights,
            )
            positions = pos  # same for all channels (same transect)
            profiles[ch_label] = density

            # CWT scalogram
            if len(density) > 10 and density.max() > 0:
                scalogram = cwt_scalogram(
                    density, positions,
                    min_period_um=args.min_period_um,
                    max_period_um=args.max_period_um,
                )

                boundaries = detect_boundaries(scalogram, positions)
                all_boundaries[f'{t_name}_{ch_label}'] = {
                    'positions_um': boundaries['positions_um'].tolist(),
                    'strengths': boundaries['strengths'].tolist(),
                }

                if len(boundaries['positions_um']) > 0:
                    print(f"  {ch_label}: {len(boundaries['positions_um'])} boundaries detected")

                plot_scalogram(
                    scalogram, positions, f'{ch_label} ({t_name})',
                    boundaries=boundaries,
                    density=density,
                    output_path=output_dir / f'scalogram_{ch_label}_{t_name}.png',
                )

        # Overlay density profiles
        if profiles and positions is not None:
            plot_density_profiles(
                profiles, positions, t_name,
                output_path=output_dir / f'density_profiles_{t_name}.png',
            )

    # ── 2D heatmaps (always generated) ─────────────────────────────

    print(f"\n--- 2D Heatmaps ---")

    # Raw cell density (unweighted)
    hm_data = build_2d_heatmap(cells, None, bin_size_um=args.heatmap_bin_um)
    plot_2d_heatmap(hm_data, 'cell_density',
                    output_path=output_dir / 'heatmap_2d_cell_density.png')
    np.save(output_dir / 'heatmap_2d_cell_density.npy', hm_data['heatmap'])

    # Per-channel intensity-weighted
    for ch_key in ch_keys:
        if ch_key is None:
            continue
        ch_label = ch_key.replace('_mean', '')
        hm_data = build_2d_heatmap(cells, ch_key, bin_size_um=args.heatmap_bin_um)
        plot_2d_heatmap(hm_data, ch_label,
                        output_path=output_dir / f'heatmap_2d_{ch_label}.png')
        np.save(output_dir / f'heatmap_2d_{ch_label}.npy', hm_data['heatmap'])

    # ── Save summary ────────────────────────────────────────────────

    summary = {
        'n_cells': len(cells),
        'tissue_extent_um': bbox,
        'n_transects': len(transects),
        'channels_analyzed': [k for k in ch_keys if k is not None],
        'parameters': {
            'corridor_width_um': args.corridor_width_um,
            'bin_width_um': args.bin_width_um,
            'min_period_um': args.min_period_um,
            'max_period_um': args.max_period_um,
            'heatmap_bin_um': args.heatmap_bin_um,
        },
        'boundaries': all_boundaries,
    }

    with open(output_dir / 'boundaries.json', 'w') as f:
        json.dump({'boundaries': all_boundaries}, f, indent=2)

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Outputs in {output_dir}/")
    print(f"  Total boundaries detected: {sum(len(b['positions_um']) for b in all_boundaries.values())}")


if __name__ == '__main__':
    main()
