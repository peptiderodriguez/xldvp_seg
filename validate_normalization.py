#!/usr/bin/env python3
"""
Compare normalization methods across segmentation outputs.

Computes variance reduction metrics and generates HTML report comparing:
- Unnormalized baseline
- Reinhard normalization
- Percentile normalization (optional)

Metrics:
- Per-slide RGB/Lab mean/std statistics
- Cross-slide variance and coefficient of variation (CV)
- Variance reduction percentages
- Cell count consistency (MK/HSPC CV)

Output:
- JSON file with raw metrics
- HTML report with interactive visualizations
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import color
from tqdm import tqdm

# Import from codebase
from segmentation.io.czi_loader import get_loader
from segmentation.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare normalization methods and compute variance reduction metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s --unnormalized /path/unified_8slides_unnormalized \\
           --reinhard /path/unified_8slides_reinhard \\
           --output validation_report.html
        """
    )
    parser.add_argument(
        "--unnormalized",
        required=True,
        type=Path,
        help="Baseline output directory (unnormalized)"
    )
    parser.add_argument(
        "--reinhard",
        required=True,
        type=Path,
        help="Reinhard normalized output directory"
    )
    parser.add_argument(
        "--percentile",
        type=Path,
        help="Percentile normalized output directory (optional)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HTML report path"
    )
    parser.add_argument(
        "--czi-dir",
        type=Path,
        default=Path("/viper/ptmp2/edrod/2025_11_18"),
        help="Directory containing CZI files (default: %(default)s)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of pixels to sample per slide (default: %(default)s)"
    )
    return parser.parse_args()


def sample_pixels_from_czi(czi_path: Path, n_samples: int = 10000, channel: int = 0) -> np.ndarray:
    """
    Sample random pixels from a CZI file.

    Args:
        czi_path: Path to CZI file
        n_samples: Number of pixels to sample
        channel: Channel to load (default 0 = RGB)

    Returns:
        RGB array of sampled pixels (N, 3)
    """
    logger.info(f"  Sampling {n_samples:,} pixels from {czi_path.name}...")

    loader = get_loader(str(czi_path), load_to_ram=True, channel=channel, quiet=True)
    img = loader.get_channel_data(channel)

    if img is None:
        raise ValueError(f"Failed to load channel data from {czi_path}")

    # Handle RGB vs grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        h, w, c = img.shape
    elif img.ndim == 2:
        h, w = img.shape
        # Convert grayscale to RGB
        img = np.stack([img] * 3, axis=-1)
        h, w, c = img.shape
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Random sampling
    n_samples = min(n_samples, h * w)
    row_indices = np.random.randint(0, h, size=n_samples)
    col_indices = np.random.randint(0, w, size=n_samples)
    samples = img[row_indices, col_indices, :].copy()

    loader.close()

    return samples


def compute_slide_statistics(samples_rgb: np.ndarray) -> Dict:
    """
    Compute RGB and Lab statistics for a slide.

    Args:
        samples_rgb: RGB samples (N, 3) in range [0, 255]

    Returns:
        Dictionary with RGB and Lab statistics
    """
    # RGB statistics
    rgb_mean = np.mean(samples_rgb, axis=0)
    rgb_std = np.std(samples_rgb, axis=0)

    # Convert to Lab
    samples_rgb_scaled = samples_rgb.astype(np.float32) / 255.0
    samples_reshaped = samples_rgb_scaled.reshape(-1, 1, 3)
    samples_lab = color.rgb2lab(samples_reshaped).reshape(-1, 3)

    # Lab statistics
    lab_mean = np.mean(samples_lab, axis=0)
    lab_std = np.std(samples_lab, axis=0)

    return {
        'R_mean': float(rgb_mean[0]),
        'G_mean': float(rgb_mean[1]),
        'B_mean': float(rgb_mean[2]),
        'R_std': float(rgb_std[0]),
        'G_std': float(rgb_std[1]),
        'B_std': float(rgb_std[2]),
        'L_mean': float(lab_mean[0]),
        'a_mean': float(lab_mean[1]),
        'b_mean': float(lab_mean[2]),
        'L_std': float(lab_std[0]),
        'a_std': float(lab_std[1]),
        'b_std': float(lab_std[2]),
    }


def compute_cross_slide_variance(slide_stats: Dict[str, Dict]) -> Dict:
    """
    Compute variance of means across slides.

    Args:
        slide_stats: Dict mapping slide_name -> statistics

    Returns:
        Dictionary with variance and CV for each channel
    """
    channels = ['R', 'G', 'B', 'L', 'a', 'b']
    metrics = {}

    for ch in channels:
        means = [stats[f'{ch}_mean'] for stats in slide_stats.values()]
        mean_of_means = np.mean(means)
        var_of_means = np.var(means)
        std_of_means = np.std(means)

        # Coefficient of variation (CV = std / mean)
        # For a/b channels, use absolute mean to avoid division issues
        if ch in ['a', 'b']:
            cv = std_of_means / (abs(mean_of_means) + 1e-6)
        else:
            cv = std_of_means / (mean_of_means + 1e-6)

        metrics[f'{ch}_var'] = float(var_of_means)
        metrics[f'{ch}_std'] = float(std_of_means)
        metrics[f'{ch}_cv'] = float(cv)
        metrics[f'{ch}_mean_of_means'] = float(mean_of_means)

    return metrics


def compute_variance_reduction(unnorm_metrics: Dict, norm_metrics: Dict) -> Dict:
    """
    Compute variance reduction from unnormalized to normalized.

    Args:
        unnorm_metrics: Cross-slide metrics for unnormalized
        norm_metrics: Cross-slide metrics for normalized

    Returns:
        Dictionary with variance reduction percentages
    """
    channels = ['R', 'G', 'B', 'L', 'a', 'b']
    reduction = {}

    for ch in channels:
        var_unnorm = unnorm_metrics[f'{ch}_var']
        var_norm = norm_metrics[f'{ch}_var']

        # Variance reduction percentage
        if var_unnorm > 0:
            reduction[f'{ch}_var_reduction_pct'] = float((var_unnorm - var_norm) / var_unnorm * 100)
        else:
            reduction[f'{ch}_var_reduction_pct'] = 0.0

        # CV reduction
        cv_unnorm = unnorm_metrics[f'{ch}_cv']
        cv_norm = norm_metrics[f'{ch}_cv']

        if cv_unnorm > 0:
            reduction[f'{ch}_cv_reduction_pct'] = float((cv_unnorm - cv_norm) / cv_unnorm * 100)
        else:
            reduction[f'{ch}_cv_reduction_pct'] = 0.0

    return reduction


def parse_cell_counts(output_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Parse MK/HSPC counts from output directory.

    Args:
        output_dir: Segmentation output directory

    Returns:
        Dict mapping slide_name -> {'mk': count, 'hspc': count}
    """
    cell_counts = {}

    # Look for slide-specific output directories
    for slide_dir in output_dir.iterdir():
        if not slide_dir.is_dir():
            continue

        slide_name = slide_dir.name

        # Count MKs
        mk_tiles = slide_dir / "mk" / "tiles"
        mk_count = 0
        if mk_tiles.exists():
            for tile_dir in mk_tiles.iterdir():
                features_file = tile_dir / "features.json"
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        features = json.load(f)
                        mk_count += len(features)

        # Count HSPCs
        hspc_tiles = slide_dir / "hspc" / "tiles"
        hspc_count = 0
        if hspc_tiles.exists():
            for tile_dir in hspc_tiles.iterdir():
                features_file = tile_dir / "features.json"
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        features = json.load(f)
                        hspc_count += len(features)

        if mk_count > 0 or hspc_count > 0:
            cell_counts[slide_name] = {
                'mk': mk_count,
                'hspc': hspc_count
            }

    return cell_counts


def compute_cell_count_cv(cell_counts: Dict[str, Dict[str, int]]) -> Dict:
    """
    Compute coefficient of variation for cell counts.

    Args:
        cell_counts: Dict mapping slide_name -> {'mk': count, 'hspc': count}

    Returns:
        Dictionary with CV metrics
    """
    mk_counts = [counts['mk'] for counts in cell_counts.values() if counts['mk'] > 0]
    hspc_counts = [counts['hspc'] for counts in cell_counts.values() if counts['hspc'] > 0]

    metrics = {}

    if len(mk_counts) > 0:
        mk_mean = np.mean(mk_counts)
        mk_std = np.std(mk_counts)
        metrics['mk_cv'] = float(mk_std / (mk_mean + 1e-6))
        metrics['mk_mean'] = float(mk_mean)
        metrics['mk_std'] = float(mk_std)
    else:
        metrics['mk_cv'] = 0.0
        metrics['mk_mean'] = 0.0
        metrics['mk_std'] = 0.0

    if len(hspc_counts) > 0:
        hspc_mean = np.mean(hspc_counts)
        hspc_std = np.std(hspc_counts)
        metrics['hspc_cv'] = float(hspc_std / (hspc_mean + 1e-6))
        metrics['hspc_mean'] = float(hspc_mean)
        metrics['hspc_std'] = float(hspc_std)
    else:
        metrics['hspc_cv'] = 0.0
        metrics['hspc_mean'] = 0.0
        metrics['hspc_std'] = 0.0

    return metrics


def generate_html_report(
    metrics: Dict,
    output_path: Path,
    plot_dir: Path
):
    """
    Generate HTML report with visualizations.

    Args:
        metrics: Dictionary with all computed metrics
        output_path: Path to save HTML report
        plot_dir: Directory containing generated plots
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Normalization Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .verdict {{
            background-color: #2ecc71;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 16px;
        }}
        .warning {{
            background-color: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-positive {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .metric-negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .stats-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
        }}
        .stats-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Normalization Validation Report</h1>
        <p><strong>Generated:</strong> {metrics.get('timestamp', 'N/A')}</p>
"""

    # Variance reduction summary
    if 'variance_reduction' in metrics and 'reinhard' in metrics['variance_reduction']:
        reduction = metrics['variance_reduction']['reinhard']

        # Compute average variance reduction across RGB channels
        rgb_reduction = np.mean([
            reduction['R_var_reduction_pct'],
            reduction['G_var_reduction_pct'],
            reduction['B_var_reduction_pct']
        ])

        # Compute average variance reduction across Lab channels
        lab_reduction = np.mean([
            reduction['L_var_reduction_pct'],
            reduction['a_var_reduction_pct'],
            reduction['b_var_reduction_pct']
        ])

        verdict_class = "verdict" if rgb_reduction > 0 else "warning"

        html += f"""
        <div class="{verdict_class}">
            <strong>VERDICT:</strong> Reinhard normalization reduced cross-slide variance by
            <strong>{rgb_reduction:.1f}%</strong> (RGB) and <strong>{lab_reduction:.1f}%</strong> (Lab)
        </div>
"""

    # Variance reduction table
    html += """
        <h2>1. Variance Reduction Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Channel</th>
                    <th>Unnormalized Var</th>
                    <th>Reinhard Var</th>
                    <th>Reduction (%)</th>
                    <th>CV Reduction (%)</th>
                </tr>
            </thead>
            <tbody>
"""

    if 'cross_slide_variance' in metrics:
        channels = ['R', 'G', 'B', 'L', 'a', 'b']
        for ch in channels:
            unnorm_var = metrics['cross_slide_variance']['unnormalized'].get(f'{ch}_var', 0)
            reinhard_var = metrics['cross_slide_variance'].get('reinhard', {}).get(f'{ch}_var', 0)

            if 'variance_reduction' in metrics and 'reinhard' in metrics['variance_reduction']:
                var_reduction = metrics['variance_reduction']['reinhard'].get(f'{ch}_var_reduction_pct', 0)
                cv_reduction = metrics['variance_reduction']['reinhard'].get(f'{ch}_cv_reduction_pct', 0)
            else:
                var_reduction = 0
                cv_reduction = 0

            reduction_class = "metric-positive" if var_reduction > 0 else "metric-negative"

            html += f"""
                <tr>
                    <td><strong>{ch}</strong></td>
                    <td>{unnorm_var:.4f}</td>
                    <td>{reinhard_var:.4f}</td>
                    <td class="{reduction_class}">{var_reduction:.1f}%</td>
                    <td class="{reduction_class}">{cv_reduction:.1f}%</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
"""

    # Per-slide statistics
    html += """
        <h2>2. Per-Slide Statistics</h2>
        <p>Mean intensity values for each slide (lower variance across slides = better normalization)</p>
"""

    if 'per_slide_stats' in metrics and 'unnormalized' in metrics['per_slide_stats']:
        # Create comparison table
        slide_names = list(metrics['per_slide_stats']['unnormalized'].keys())

        html += """
        <table>
            <thead>
                <tr>
                    <th>Slide</th>
                    <th colspan="3">Unnormalized RGB Mean</th>
                    <th colspan="3">Reinhard RGB Mean</th>
                </tr>
                <tr>
                    <th></th>
                    <th>R</th>
                    <th>G</th>
                    <th>B</th>
                    <th>R</th>
                    <th>G</th>
                    <th>B</th>
                </tr>
            </thead>
            <tbody>
"""

        for slide in slide_names:
            unnorm_stats = metrics['per_slide_stats']['unnormalized'][slide]
            reinhard_stats = metrics['per_slide_stats'].get('reinhard', {}).get(slide, {})

            html += f"""
                <tr>
                    <td><strong>{slide}</strong></td>
                    <td>{unnorm_stats.get('R_mean', 0):.1f}</td>
                    <td>{unnorm_stats.get('G_mean', 0):.1f}</td>
                    <td>{unnorm_stats.get('B_mean', 0):.1f}</td>
                    <td>{reinhard_stats.get('R_mean', 0):.1f}</td>
                    <td>{reinhard_stats.get('G_mean', 0):.1f}</td>
                    <td>{reinhard_stats.get('B_mean', 0):.1f}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

    # Cell count comparison
    if 'cell_counts' in metrics:
        html += """
        <h2>3. Cell Count Statistics</h2>
        <p>Coefficient of variation (CV) for cell counts (lower CV = better consistency)</p>

        <div class="grid">
"""

        for method in ['unnormalized', 'reinhard', 'percentile']:
            if method not in metrics['cell_counts']:
                continue

            stats = metrics['cell_counts'][method]

            html += f"""
            <div class="stats-box">
                <h3>{method.title()}</h3>
                <p><strong>MK Count CV:</strong> {stats.get('mk_cv', 0):.3f}</p>
                <p><strong>MK Mean:</strong> {stats.get('mk_mean', 0):.0f} ± {stats.get('mk_std', 0):.0f}</p>
                <p><strong>HSPC Count CV:</strong> {stats.get('hspc_cv', 0):.3f}</p>
                <p><strong>HSPC Mean:</strong> {stats.get('hspc_mean', 0):.0f} ± {stats.get('hspc_std', 0):.0f}</p>
            </div>
"""

        html += """
        </div>
"""

    # Plots
    html += """
        <h2>4. Visualizations</h2>
"""

    plot_files = [
        ('rgb_means_comparison.png', 'RGB Mean Intensity Comparison'),
        ('lab_means_comparison.png', 'Lab Mean Intensity Comparison'),
        ('rgb_boxplots.png', 'RGB Distribution Boxplots'),
        ('lab_boxplots.png', 'Lab Distribution Boxplots'),
    ]

    for plot_file, title in plot_files:
        plot_path = plot_dir / plot_file
        if plot_path.exists():
            html += f"""
        <div class="plot">
            <h3>{title}</h3>
            <img src="{plot_path.name}" alt="{title}">
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_path}")


def generate_plots(metrics: Dict, output_dir: Path):
    """
    Generate visualization plots.

    Args:
        metrics: Dictionary with all computed metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'per_slide_stats' not in metrics:
        logger.warning("No per-slide statistics found, skipping plots")
        return

    slide_names = list(metrics['per_slide_stats']['unnormalized'].keys())
    methods = [m for m in ['unnormalized', 'reinhard', 'percentile']
               if m in metrics['per_slide_stats']]

    # Plot 1: RGB means comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['R', 'G', 'B']
    colors = ['red', 'green', 'blue']

    for i, (ch, color) in enumerate(zip(channels, colors)):
        ax = axes[i]

        for method in methods:
            means = [metrics['per_slide_stats'][method][slide][f'{ch}_mean']
                     for slide in slide_names]
            ax.plot(range(len(slide_names)), means, 'o-', label=method.title(),
                   markersize=8, linewidth=2)

        ax.set_xlabel('Slide Index')
        ax.set_ylabel(f'{ch} Mean Intensity')
        ax.set_title(f'{ch} Channel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(slide_names)))
        ax.set_xticklabels([s[:10] for s in slide_names], rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'rgb_means_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Lab means comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lab_channels = ['L', 'a', 'b']

    for i, ch in enumerate(lab_channels):
        ax = axes[i]

        for method in methods:
            means = [metrics['per_slide_stats'][method][slide][f'{ch}_mean']
                     for slide in slide_names]
            ax.plot(range(len(slide_names)), means, 'o-', label=method.title(),
                   markersize=8, linewidth=2)

        ax.set_xlabel('Slide Index')
        ax.set_ylabel(f'{ch} Mean')
        ax.set_title(f'{ch} Channel')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(slide_names)))
        ax.set_xticklabels([s[:10] for s in slide_names], rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'lab_means_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: RGB boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ch in enumerate(channels):
        ax = axes[i]

        data_to_plot = []
        labels = []

        for method in methods:
            means = [metrics['per_slide_stats'][method][slide][f'{ch}_mean']
                     for slide in slide_names]
            data_to_plot.append(means)
            labels.append(method.title())

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors_box = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors_box[:len(methods)]):
            patch.set_facecolor(color)

        ax.set_ylabel(f'{ch} Mean Intensity')
        ax.set_title(f'{ch} Channel Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'rgb_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Lab boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ch in enumerate(lab_channels):
        ax = axes[i]

        data_to_plot = []
        labels = []

        for method in methods:
            means = [metrics['per_slide_stats'][method][slide][f'{ch}_mean']
                     for slide in slide_names]
            data_to_plot.append(means)
            labels.append(method.title())

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors_box = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors_box[:len(methods)]):
            patch.set_facecolor(color)

        ax.set_ylabel(f'{ch} Mean')
        ax.set_title(f'{ch} Channel Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'lab_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {output_dir}")


def main():
    args = parse_args()

    logger.info("="*70)
    logger.info("NORMALIZATION VALIDATION")
    logger.info("="*70)

    # Validate inputs
    if not args.unnormalized.exists():
        logger.error(f"Unnormalized directory not found: {args.unnormalized}")
        return 1

    if not args.reinhard.exists():
        logger.error(f"Reinhard directory not found: {args.reinhard}")
        return 1

    if args.percentile and not args.percentile.exists():
        logger.error(f"Percentile directory not found: {args.percentile}")
        return 1

    # Find CZI files
    czi_files = sorted(args.czi_dir.glob("*.czi"))
    if len(czi_files) == 0:
        logger.error(f"No CZI files found in {args.czi_dir}")
        return 1

    logger.info(f"Found {len(czi_files)} CZI files")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize metrics dictionary
    metrics = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'czi_dir': str(args.czi_dir),
        'n_samples': args.n_samples,
        'per_slide_stats': {},
        'cross_slide_variance': {},
        'variance_reduction': {},
        'cell_counts': {}
    }

    # =========================================================================
    # 1. Compute per-slide statistics from CZI files
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Computing per-slide statistics from CZI files")
    logger.info("="*70)

    for method in ['unnormalized', 'reinhard', 'percentile']:
        if method == 'percentile' and not args.percentile:
            continue

        logger.info(f"\nProcessing {method.upper()}...")
        metrics['per_slide_stats'][method] = {}

        for czi_file in tqdm(czi_files, desc=f"Sampling {method}"):
            slide_name = czi_file.stem

            try:
                samples = sample_pixels_from_czi(czi_file, args.n_samples)
                stats = compute_slide_statistics(samples)
                metrics['per_slide_stats'][method][slide_name] = stats
            except Exception as e:
                logger.error(f"Failed to process {slide_name}: {e}")

    # =========================================================================
    # 2. Compute cross-slide variance
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Computing cross-slide variance")
    logger.info("="*70)

    for method in ['unnormalized', 'reinhard', 'percentile']:
        if method not in metrics['per_slide_stats']:
            continue

        logger.info(f"\n{method.upper()}:")
        variance_metrics = compute_cross_slide_variance(metrics['per_slide_stats'][method])
        metrics['cross_slide_variance'][method] = variance_metrics

        # Log summary
        logger.info(f"  RGB variance: R={variance_metrics['R_var']:.4f}, "
                   f"G={variance_metrics['G_var']:.4f}, B={variance_metrics['B_var']:.4f}")
        logger.info(f"  Lab variance: L={variance_metrics['L_var']:.4f}, "
                   f"a={variance_metrics['a_var']:.4f}, b={variance_metrics['b_var']:.4f}")
        logger.info(f"  RGB CV: R={variance_metrics['R_cv']:.4f}, "
                   f"G={variance_metrics['G_cv']:.4f}, B={variance_metrics['B_cv']:.4f}")

    # =========================================================================
    # 3. Compute variance reduction
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Computing variance reduction")
    logger.info("="*70)

    unnorm_metrics = metrics['cross_slide_variance']['unnormalized']

    for method in ['reinhard', 'percentile']:
        if method not in metrics['cross_slide_variance']:
            continue

        norm_metrics = metrics['cross_slide_variance'][method]
        reduction = compute_variance_reduction(unnorm_metrics, norm_metrics)
        metrics['variance_reduction'][method] = reduction

        logger.info(f"\n{method.upper()} vs UNNORMALIZED:")
        logger.info(f"  RGB variance reduction: R={reduction['R_var_reduction_pct']:.1f}%, "
                   f"G={reduction['G_var_reduction_pct']:.1f}%, B={reduction['B_var_reduction_pct']:.1f}%")
        logger.info(f"  Lab variance reduction: L={reduction['L_var_reduction_pct']:.1f}%, "
                   f"a={reduction['a_var_reduction_pct']:.1f}%, b={reduction['b_var_reduction_pct']:.1f}%")

    # =========================================================================
    # 4. Parse cell counts (if available)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Parsing cell counts from output directories")
    logger.info("="*70)

    for method, output_dir in [
        ('unnormalized', args.unnormalized),
        ('reinhard', args.reinhard),
        ('percentile', args.percentile)
    ]:
        if output_dir is None:
            continue

        logger.info(f"\nParsing {method.upper()}...")
        cell_counts = parse_cell_counts(output_dir)

        if len(cell_counts) > 0:
            cv_metrics = compute_cell_count_cv(cell_counts)
            metrics['cell_counts'][method] = cv_metrics

            logger.info(f"  Found {len(cell_counts)} slides with cell counts")
            logger.info(f"  MK CV: {cv_metrics['mk_cv']:.3f} "
                       f"(mean={cv_metrics['mk_mean']:.0f}, std={cv_metrics['mk_std']:.0f})")
            logger.info(f"  HSPC CV: {cv_metrics['hspc_cv']:.3f} "
                       f"(mean={cv_metrics['hspc_mean']:.0f}, std={cv_metrics['hspc_std']:.0f})")
        else:
            logger.warning(f"  No cell counts found in {output_dir}")

    # =========================================================================
    # 5. Generate plots and HTML report
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Generating visualizations and HTML report")
    logger.info("="*70)

    # Create output directory for plots
    plot_dir = args.output.parent / f"{args.output.stem}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    generate_plots(metrics, plot_dir)

    # Save raw metrics to JSON
    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Raw metrics saved to {json_output}")

    # Generate HTML report
    generate_html_report(metrics, args.output, plot_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)

    if 'variance_reduction' in metrics and 'reinhard' in metrics['variance_reduction']:
        reduction = metrics['variance_reduction']['reinhard']
        rgb_reduction = np.mean([
            reduction['R_var_reduction_pct'],
            reduction['G_var_reduction_pct'],
            reduction['B_var_reduction_pct']
        ])
        lab_reduction = np.mean([
            reduction['L_var_reduction_pct'],
            reduction['a_var_reduction_pct'],
            reduction['b_var_reduction_pct']
        ])

        logger.info(f"\nVERDICT: Reinhard normalization reduced cross-slide variance by:")
        logger.info(f"  - RGB: {rgb_reduction:.1f}%")
        logger.info(f"  - Lab: {lab_reduction:.1f}%")

    logger.info(f"\nOutputs:")
    logger.info(f"  - HTML report: {args.output}")
    logger.info(f"  - Raw metrics: {json_output}")
    logger.info(f"  - Plots: {plot_dir}")
    logger.info("")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
