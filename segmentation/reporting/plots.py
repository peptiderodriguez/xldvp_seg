"""
Plotly visualization functions for vessel segmentation reports.

Creates interactive charts and plots for vessel detection analysis.
All functions return Plotly figure objects that can be embedded in HTML
or exported to static images.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None


# Color scheme matching existing codebase dark theme
DARK_THEME = {
    "paper_bgcolor": "#0a0a0a",
    "plot_bgcolor": "#111111",
    "font_color": "#dddddd",
    "grid_color": "#333333",
    "primary_color": "#4a4",  # Green
    "secondary_color": "#44a",  # Blue
    "accent_color": "#a44",  # Red
    "warning_color": "#aa4",  # Yellow
}

# Vessel type colors
VESSEL_TYPE_COLORS = {
    "capillary": "#66ff66",  # Light green
    "arteriole": "#6666ff",  # Light blue
    "artery": "#ff6666",  # Light red
    "unknown": "#888888",  # Gray
}

# Confidence colors
CONFIDENCE_COLORS = {
    "high": "#4a4",  # Green
    "medium": "#aa4",  # Yellow
    "low": "#a44",  # Red
    "unknown": "#888",  # Gray
}


def _check_plotly() -> None:
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: pip install plotly"
        )


def _apply_dark_theme(fig: "go.Figure") -> "go.Figure":
    """
    Apply dark theme to a Plotly figure.

    Args:
        fig: Plotly figure object

    Returns:
        Figure with dark theme applied
    """
    fig.update_layout(
        paper_bgcolor=DARK_THEME["paper_bgcolor"],
        plot_bgcolor=DARK_THEME["plot_bgcolor"],
        font=dict(color=DARK_THEME["font_color"], family="monospace"),
        xaxis=dict(
            gridcolor=DARK_THEME["grid_color"],
            zerolinecolor=DARK_THEME["grid_color"],
        ),
        yaxis=dict(
            gridcolor=DARK_THEME["grid_color"],
            zerolinecolor=DARK_THEME["grid_color"],
        ),
    )
    return fig


def create_diameter_histogram(
    features_list: List[Dict[str, Any]],
    bins: int = 30,
    title: str = "Vessel Diameter Distribution",
    show_stats: bool = True,
    range_um: Optional[Tuple[float, float]] = None,
) -> "go.Figure":
    """
    Create a histogram of vessel outer diameters.

    Args:
        features_list: List of feature dicts from detections
        bins: Number of histogram bins
        title: Plot title
        show_stats: Whether to show statistics annotation
        range_um: Optional (min, max) range in microns

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    diameters = [
        f.get("outer_diameter_um")
        for f in features_list
        if f.get("outer_diameter_um") is not None
    ]

    if not diameters:
        fig = go.Figure()
        fig.add_annotation(
            text="No diameter data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    diameters = np.array(diameters)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=diameters,
            nbinsx=bins,
            name="Diameter",
            marker_color=DARK_THEME["primary_color"],
            opacity=0.8,
        )
    )

    # Add statistics annotation
    if show_stats:
        stats_text = (
            f"n = {len(diameters)}<br>"
            f"Mean: {np.mean(diameters):.1f} um<br>"
            f"Median: {np.median(diameters):.1f} um<br>"
            f"Std: {np.std(diameters):.1f} um"
        )
        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            showarrow=False,
            align="right",
            bgcolor="rgba(17, 17, 17, 0.8)",
            bordercolor=DARK_THEME["grid_color"],
            font=dict(size=11),
        )

    # Add vertical lines for vessel type thresholds
    fig.add_vline(x=10, line_dash="dash", line_color="#666", annotation_text="10 um")
    fig.add_vline(x=100, line_dash="dash", line_color="#666", annotation_text="100 um")

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Outer Diameter (um)",
        yaxis_title="Count",
        showlegend=False,
        xaxis_range=range_um,
    )

    return _apply_dark_theme(fig)


def create_wall_thickness_histogram(
    features_list: List[Dict[str, Any]],
    bins: int = 25,
    title: str = "Wall Thickness Distribution",
    show_quantiles: bool = True,
) -> "go.Figure":
    """
    Create a histogram of vessel wall thickness.

    Args:
        features_list: List of feature dicts from detections
        bins: Number of histogram bins
        title: Plot title
        show_quantiles: Whether to show quantile lines

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    thicknesses = [
        f.get("wall_thickness_mean_um")
        for f in features_list
        if f.get("wall_thickness_mean_um") is not None
    ]

    if not thicknesses:
        fig = go.Figure()
        fig.add_annotation(
            text="No wall thickness data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    thicknesses = np.array(thicknesses)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=thicknesses,
            nbinsx=bins,
            name="Wall Thickness",
            marker_color=DARK_THEME["secondary_color"],
            opacity=0.8,
        )
    )

    # Add quantile lines
    if show_quantiles:
        q25, q50, q75 = np.percentile(thicknesses, [25, 50, 75])
        fig.add_vline(x=q25, line_dash="dot", line_color="#888", annotation_text="Q25")
        fig.add_vline(x=q50, line_dash="solid", line_color="#aaa", annotation_text="Median")
        fig.add_vline(x=q75, line_dash="dot", line_color="#888", annotation_text="Q75")

    # Statistics annotation
    stats_text = (
        f"n = {len(thicknesses)}<br>"
        f"Mean: {np.mean(thicknesses):.2f} um<br>"
        f"Median: {np.median(thicknesses):.2f} um<br>"
        f"Q25-Q75: {np.percentile(thicknesses, 25):.2f}-{np.percentile(thicknesses, 75):.2f} um"
    )
    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        showarrow=False,
        align="right",
        bgcolor="rgba(17, 17, 17, 0.8)",
        bordercolor=DARK_THEME["grid_color"],
        font=dict(size=11),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Wall Thickness (um)",
        yaxis_title="Count",
        showlegend=False,
    )

    return _apply_dark_theme(fig)


def create_diameter_vs_wall_scatter(
    features_list: List[Dict[str, Any]],
    title: str = "Diameter vs Wall Thickness",
    color_by: str = "confidence",
    show_regression: bool = True,
) -> "go.Figure":
    """
    Create a scatter plot of diameter vs wall thickness.

    Args:
        features_list: List of feature dicts from detections
        title: Plot title
        color_by: Field to color points by ("confidence", "vessel_type", or "circularity")
        show_regression: Whether to show regression line

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    # Extract data
    data = []
    for f in features_list:
        d = f.get("outer_diameter_um")
        w = f.get("wall_thickness_mean_um")
        if d is not None and w is not None:
            data.append({
                "diameter": d,
                "wall_thickness": w,
                "confidence": f.get("confidence", "unknown"),
                "vessel_type": f.get("vessel_type", "unknown"),
                "circularity": f.get("circularity", 0),
            })

    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for scatter plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    fig = go.Figure()

    # Group by color field
    if color_by == "confidence":
        colors = CONFIDENCE_COLORS
        groups = {}
        for d in data:
            key = d["confidence"]
            if key not in groups:
                groups[key] = {"x": [], "y": []}
            groups[key]["x"].append(d["diameter"])
            groups[key]["y"].append(d["wall_thickness"])

        for group_name, group_data in groups.items():
            fig.add_trace(
                go.Scatter(
                    x=group_data["x"],
                    y=group_data["y"],
                    mode="markers",
                    name=group_name.capitalize(),
                    marker=dict(
                        color=colors.get(group_name, "#888"),
                        size=6,
                        opacity=0.7,
                    ),
                )
            )

    elif color_by == "vessel_type":
        colors = VESSEL_TYPE_COLORS
        groups = {}
        for d in data:
            key = d["vessel_type"]
            if key not in groups:
                groups[key] = {"x": [], "y": []}
            groups[key]["x"].append(d["diameter"])
            groups[key]["y"].append(d["wall_thickness"])

        for group_name, group_data in groups.items():
            fig.add_trace(
                go.Scatter(
                    x=group_data["x"],
                    y=group_data["y"],
                    mode="markers",
                    name=group_name.capitalize(),
                    marker=dict(
                        color=colors.get(group_name, "#888"),
                        size=6,
                        opacity=0.7,
                    ),
                )
            )

    else:
        # Color by circularity (continuous)
        circularity = [d["circularity"] for d in data]
        fig.add_trace(
            go.Scatter(
                x=[d["diameter"] for d in data],
                y=[d["wall_thickness"] for d in data],
                mode="markers",
                marker=dict(
                    color=circularity,
                    colorscale="Viridis",
                    size=6,
                    opacity=0.7,
                    colorbar=dict(title="Circularity"),
                ),
            )
        )

    # Add regression line
    if show_regression:
        x_vals = np.array([d["diameter"] for d in data])
        y_vals = np.array([d["wall_thickness"] for d in data])
        if len(x_vals) > 2:
            # Simple linear regression
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = np.polyval(coeffs, x_line)

            # Correlation coefficient
            r = np.corrcoef(x_vals, y_vals)[0, 1]

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Fit (R={r:.3f})",
                    line=dict(color="#fff", dash="dash", width=1),
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Outer Diameter (um)",
        yaxis_title="Wall Thickness (um)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(17, 17, 17, 0.8)",
        ),
    )

    return _apply_dark_theme(fig)


def create_lumen_vs_wall_scatter(
    features_list: List[Dict[str, Any]],
    title: str = "Lumen Area vs Wall Area",
    color_by: str = "vessel_type",
    log_scale: bool = True,
) -> "go.Figure":
    """
    Create a scatter plot of lumen area vs wall area.

    Args:
        features_list: List of feature dicts from detections
        title: Plot title
        color_by: Field to color points by
        log_scale: Whether to use log scale for axes

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    # Extract data
    data = []
    for f in features_list:
        lumen = f.get("lumen_area_um2")
        wall = f.get("wall_area_um2")
        if lumen is not None and wall is not None and lumen > 0 and wall > 0:
            data.append({
                "lumen_area": lumen,
                "wall_area": wall,
                "vessel_type": f.get("vessel_type", "unknown"),
                "confidence": f.get("confidence", "unknown"),
            })

    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No area data available for scatter plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    fig = go.Figure()

    # Group by vessel type
    colors = VESSEL_TYPE_COLORS if color_by == "vessel_type" else CONFIDENCE_COLORS
    field = "vessel_type" if color_by == "vessel_type" else "confidence"

    groups: Dict[str, Dict[str, List[float]]] = {}
    for d in data:
        key = d[field]
        if key not in groups:
            groups[key] = {"x": [], "y": []}
        groups[key]["x"].append(d["lumen_area"])
        groups[key]["y"].append(d["wall_area"])

    for group_name, group_data in groups.items():
        fig.add_trace(
            go.Scatter(
                x=group_data["x"],
                y=group_data["y"],
                mode="markers",
                name=group_name.capitalize(),
                marker=dict(
                    color=colors.get(group_name, "#888"),
                    size=6,
                    opacity=0.7,
                ),
            )
        )

    # Add 1:1 line reference
    all_x = [d["lumen_area"] for d in data]
    all_y = [d["wall_area"] for d in data]
    max_val = max(max(all_x), max(all_y))
    min_val = min(min(all_x), min(all_y))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="1:1 Line",
            line=dict(color="#666", dash="dash", width=1),
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Lumen Area (um^2)",
        yaxis_title="Wall Area (um^2)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(17, 17, 17, 0.8)",
        ),
    )

    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    return _apply_dark_theme(fig)


def create_vessel_type_pie_chart(
    features_list: List[Dict[str, Any]],
    title: str = "Vessel Type Breakdown",
    diameter_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> "go.Figure":
    """
    Create a pie chart showing vessel type distribution.

    Args:
        features_list: List of feature dicts from detections
        title: Plot title
        diameter_thresholds: Optional custom thresholds for classification

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    if diameter_thresholds is None:
        diameter_thresholds = {
            "capillary": (0, 10),
            "arteriole": (10, 100),
            "artery": (100, float("inf")),
        }

    # Count vessel types
    counts: Dict[str, int] = {vtype: 0 for vtype in diameter_thresholds}
    counts["unknown"] = 0

    for f in features_list:
        vessel_type = f.get("vessel_type", "unknown")
        if vessel_type == "unknown":
            diameter = f.get("outer_diameter_um", 0)
            for vtype, (min_d, max_d) in diameter_thresholds.items():
                if min_d <= diameter < max_d:
                    vessel_type = vtype
                    break
        if vessel_type in counts:
            counts[vessel_type] += 1
        else:
            counts["unknown"] += 1

    # Remove zero counts
    counts = {k: v for k, v in counts.items() if v > 0}

    if not counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No vessel data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [VESSEL_TYPE_COLORS.get(l, "#888") for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[l.capitalize() for l in labels],
                values=values,
                marker_colors=colors,
                textinfo="label+percent+value",
                textposition="auto",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(17, 17, 17, 0.8)",
        ),
    )

    return _apply_dark_theme(fig)


def create_confidence_histogram(
    features_list: List[Dict[str, Any]],
    title: str = "Confidence Distribution",
) -> "go.Figure":
    """
    Create a bar chart showing confidence level distribution.

    Args:
        features_list: List of feature dicts from detections
        title: Plot title

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    # Count confidence levels
    counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for f in features_list:
        conf = f.get("confidence", "unknown")
        if conf in counts:
            counts[conf] += 1
        else:
            counts["unknown"] += 1

    labels = ["High", "Medium", "Low", "Unknown"]
    values = [counts["high"], counts["medium"], counts["low"], counts["unknown"]]
    colors = [
        CONFIDENCE_COLORS["high"],
        CONFIDENCE_COLORS["medium"],
        CONFIDENCE_COLORS["low"],
        CONFIDENCE_COLORS["unknown"],
    ]

    # Remove zeros
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    labels, values, colors = zip(*filtered)

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(labels),
                y=list(values),
                marker_color=list(colors),
                text=list(values),
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Confidence Level",
        yaxis_title="Count",
        showlegend=False,
    )

    return _apply_dark_theme(fig)


def create_ring_completeness_histogram(
    features_list: List[Dict[str, Any]],
    bins: int = 20,
    title: str = "Ring Completeness Distribution",
) -> "go.Figure":
    """
    Create a histogram of ring completeness values.

    Args:
        features_list: List of feature dicts from detections
        bins: Number of histogram bins
        title: Plot title

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    completeness = [
        f.get("ring_completeness")
        for f in features_list
        if f.get("ring_completeness") is not None
    ]

    if not completeness:
        fig = go.Figure()
        fig.add_annotation(
            text="No ring completeness data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    completeness = np.array(completeness)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=completeness,
            nbinsx=bins,
            marker_color=DARK_THEME["warning_color"],
            opacity=0.8,
        )
    )

    # Statistics
    stats_text = (
        f"n = {len(completeness)}<br>"
        f"Mean: {np.mean(completeness):.2f}<br>"
        f"Median: {np.median(completeness):.2f}<br>"
        f">0.8: {(completeness > 0.8).sum()} ({100 * (completeness > 0.8).sum() / len(completeness):.1f}%)"
    )
    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(17, 17, 17, 0.8)",
        bordercolor=DARK_THEME["grid_color"],
        font=dict(size=11),
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Ring Completeness",
        yaxis_title="Count",
        xaxis_range=[0, 1],
        showlegend=False,
    )

    return _apply_dark_theme(fig)


def create_batch_comparison_bar(
    slide_names: List[str],
    values: List[float],
    title: str = "Vessel Count by Slide",
    yaxis_title: str = "Count",
    color: Optional[str] = None,
) -> "go.Figure":
    """
    Create a bar chart comparing values across slides.

    Args:
        slide_names: List of slide names
        values: List of values for each slide
        title: Plot title
        yaxis_title: Y-axis label
        color: Optional bar color

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    if not slide_names or not values:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for comparison",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    if color is None:
        color = DARK_THEME["primary_color"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=slide_names,
                y=values,
                marker_color=color,
                text=[f"{v:.0f}" if isinstance(v, float) else str(v) for v in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Slide",
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
        showlegend=False,
    )

    return _apply_dark_theme(fig)


def create_batch_comparison_violin(
    slide_data: Dict[str, List[float]],
    title: str = "Distribution Comparison",
    yaxis_title: str = "Value",
) -> "go.Figure":
    """
    Create violin plots comparing distributions across slides.

    Args:
        slide_data: Dict mapping slide name to list of values
        title: Plot title
        yaxis_title: Y-axis label

    Returns:
        Plotly Figure object
    """
    _check_plotly()

    if not slide_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for comparison",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_dark_theme(fig)

    fig = go.Figure()

    colors = [
        DARK_THEME["primary_color"],
        DARK_THEME["secondary_color"],
        DARK_THEME["accent_color"],
        DARK_THEME["warning_color"],
        "#8a8",
        "#88a",
        "#a8a",
        "#aa8",
    ]

    for i, (name, values) in enumerate(slide_data.items()):
        if values:
            fig.add_trace(
                go.Violin(
                    y=values,
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)],
                    opacity=0.6,
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title=yaxis_title,
        violinmode="group",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(17, 17, 17, 0.8)",
        ),
    )

    return _apply_dark_theme(fig)


def create_multi_panel_summary(
    features_list: List[Dict[str, Any]],
    title: str = "Vessel Analysis Summary",
) -> "go.Figure":
    """
    Create a multi-panel figure with key visualizations.

    Includes:
    - Diameter histogram
    - Wall thickness histogram
    - Diameter vs Wall scatter
    - Vessel type pie chart

    Args:
        features_list: List of feature dicts from detections
        title: Overall figure title

    Returns:
        Plotly Figure object with 2x2 subplots
    """
    _check_plotly()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Diameter Distribution",
            "Wall Thickness Distribution",
            "Diameter vs Wall Thickness",
            "Vessel Type Breakdown",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
    )

    # Extract data
    diameters = [f.get("outer_diameter_um") for f in features_list if f.get("outer_diameter_um")]
    thicknesses = [f.get("wall_thickness_mean_um") for f in features_list if f.get("wall_thickness_mean_um")]

    # Diameter histogram
    if diameters:
        fig.add_trace(
            go.Histogram(x=diameters, marker_color=DARK_THEME["primary_color"], opacity=0.8),
            row=1,
            col=1,
        )

    # Wall thickness histogram
    if thicknesses:
        fig.add_trace(
            go.Histogram(x=thicknesses, marker_color=DARK_THEME["secondary_color"], opacity=0.8),
            row=1,
            col=2,
        )

    # Scatter plot
    scatter_data = [
        (f.get("outer_diameter_um"), f.get("wall_thickness_mean_um"))
        for f in features_list
        if f.get("outer_diameter_um") and f.get("wall_thickness_mean_um")
    ]
    if scatter_data:
        x_vals, y_vals = zip(*scatter_data)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(color=DARK_THEME["primary_color"], size=4, opacity=0.6),
            ),
            row=2,
            col=1,
        )

    # Pie chart - vessel types
    type_counts: Dict[str, int] = {"capillary": 0, "arteriole": 0, "artery": 0}
    for f in features_list:
        d = f.get("outer_diameter_um", 0)
        if d < 10:
            type_counts["capillary"] += 1
        elif d < 100:
            type_counts["arteriole"] += 1
        else:
            type_counts["artery"] += 1

    type_counts = {k: v for k, v in type_counts.items() if v > 0}
    if type_counts:
        fig.add_trace(
            go.Pie(
                labels=[k.capitalize() for k in type_counts.keys()],
                values=list(type_counts.values()),
                marker_colors=[VESSEL_TYPE_COLORS.get(k, "#888") for k in type_counts.keys()],
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        height=700,
    )

    # Update axes labels
    fig.update_xaxes(title_text="Diameter (um)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Wall Thickness (um)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Diameter (um)", row=2, col=1)
    fig.update_yaxes(title_text="Wall Thickness (um)", row=2, col=1)

    return _apply_dark_theme(fig)
