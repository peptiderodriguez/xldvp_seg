"""
Vessel segmentation report generation.

Generates comprehensive HTML and PDF reports with statistical summaries
and interactive visualizations for vessel detection results.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .stats import (
    VesselStatistics,
    compute_summary_statistics,
    compute_batch_comparison,
)
from .plots import (
    create_diameter_histogram,
    create_wall_thickness_histogram,
    create_diameter_vs_wall_scatter,
    create_lumen_vs_wall_scatter,
    create_vessel_type_pie_chart,
    create_confidence_histogram,
    create_ring_completeness_histogram,
    create_batch_comparison_bar,
    create_batch_comparison_violin,
    PLOTLY_AVAILABLE,
)


# CSS styles matching the existing codebase dark theme
REPORT_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: monospace;
    background: #0a0a0a;
    color: #ddd;
    line-height: 1.6;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
}

.header {
    background: #111;
    padding: 30px;
    border: 1px solid #333;
    margin-bottom: 30px;
    text-align: center;
}

h1 {
    font-size: 1.8em;
    font-weight: normal;
    margin-bottom: 10px;
    color: #4a4;
}

h2 {
    font-size: 1.3em;
    font-weight: normal;
    color: #aaa;
    margin: 30px 0 15px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #333;
}

h3 {
    font-size: 1.1em;
    font-weight: normal;
    color: #888;
    margin: 20px 0 10px 0;
}

.subtitle {
    color: #888;
    font-size: 0.9em;
}

.timestamp {
    color: #666;
    font-size: 0.8em;
    margin-top: 10px;
}

.section {
    background: #111;
    border: 1px solid #333;
    padding: 20px;
    margin-bottom: 20px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.stat-card {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 15px;
    text-align: center;
}

.stat-card .label {
    font-size: 0.85em;
    color: #888;
    margin-bottom: 5px;
}

.stat-card .value {
    font-size: 1.8em;
    color: #4a4;
}

.stat-card .unit {
    font-size: 0.7em;
    color: #666;
}

.stat-card.secondary .value {
    color: #44a;
}

.stat-card.accent .value {
    color: #aa4;
}

.plot-container {
    margin: 20px 0;
    background: #0a0a0a;
    border: 1px solid #333;
    padding: 10px;
    min-height: 400px;
}

.plot-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin: 20px 0;
}

@media (max-width: 900px) {
    .plot-row {
        grid-template-columns: 1fr;
    }
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}

th, td {
    padding: 10px 15px;
    text-align: left;
    border: 1px solid #333;
}

th {
    background: #1a1a1a;
    color: #aaa;
    font-weight: normal;
}

td {
    background: #111;
}

tr:hover td {
    background: #1a1a1a;
}

.quantile-table td.value {
    text-align: right;
    font-family: monospace;
}

.vessel-type-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 3px;
    font-size: 0.85em;
}

.vessel-type-badge.capillary { background: #1a3a1a; color: #6f6; }
.vessel-type-badge.arteriole { background: #1a1a3a; color: #66f; }
.vessel-type-badge.artery { background: #3a1a1a; color: #f66; }
.vessel-type-badge.unknown { background: #2a2a2a; color: #888; }

.confidence-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 3px;
    font-size: 0.85em;
}

.confidence-badge.high { background: #1a3a1a; color: #6f6; }
.confidence-badge.medium { background: #3a3a1a; color: #ff6; }
.confidence-badge.low { background: #3a1a1a; color: #f66; }

.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 0.8em;
    border-top: 1px solid #333;
    margin-top: 30px;
}

.batch-section {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #333;
}

.slide-summary {
    background: #111;
    border: 1px solid #333;
    padding: 15px;
    margin: 10px 0;
}

.slide-summary h4 {
    color: #4a4;
    margin-bottom: 10px;
}

.no-data {
    text-align: center;
    padding: 40px;
    color: #666;
    font-style: italic;
}
"""


@dataclass
class VesselReport:
    """
    Vessel segmentation report generator.

    Loads vessel detection data and generates comprehensive HTML/PDF reports
    with statistical summaries and interactive visualizations.

    Attributes:
        detections: List of vessel detection dicts
        statistics: Computed VesselStatistics object
        slide_name: Name of the slide
        metadata: Additional metadata (pixel_size, etc.)
    """

    detections: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Optional[VesselStatistics] = None
    slide_name: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls,
        json_path: Union[str, Path],
        slide_name: Optional[str] = None,
    ) -> "VesselReport":
        """
        Create a VesselReport from a vessel_detections.json file.

        Args:
            json_path: Path to vessel_detections.json
            slide_name: Optional slide name (inferred from path if not provided)

        Returns:
            VesselReport instance
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Detection file not found: {json_path}")

        with open(json_path, "r") as f:
            detections = json.load(f)

        # Try to infer slide name from path or data
        if slide_name is None:
            if detections and isinstance(detections[0], dict):
                slide_name = detections[0].get("slide_name", json_path.parent.name)
            else:
                slide_name = json_path.parent.name

        # Extract features from detection structure
        features_list = []
        for det in detections:
            if "features" in det:
                features_list.append(det["features"])
            else:
                features_list.append(det)

        # Compute statistics
        statistics = compute_summary_statistics(features_list, slide_name)

        # Try to load metadata from summary.json if available
        metadata = {}
        summary_path = json_path.parent / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                metadata = json.load(f)

        return cls(
            detections=detections,
            statistics=statistics,
            slide_name=slide_name,
            metadata=metadata,
        )

    @classmethod
    def from_detections(
        cls,
        detections: List[Dict[str, Any]],
        slide_name: str = "Unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "VesselReport":
        """
        Create a VesselReport from detection dicts directly.

        Args:
            detections: List of detection dicts
            slide_name: Slide identifier
            metadata: Optional additional metadata

        Returns:
            VesselReport instance
        """
        features_list = []
        for det in detections:
            if "features" in det:
                features_list.append(det["features"])
            else:
                features_list.append(det)

        statistics = compute_summary_statistics(features_list, slide_name)

        return cls(
            detections=detections,
            statistics=statistics,
            slide_name=slide_name,
            metadata=metadata or {},
        )

    def _get_features_list(self) -> List[Dict[str, Any]]:
        """Extract features list from detections."""
        features_list = []
        for det in self.detections:
            if "features" in det:
                features_list.append(det["features"])
            else:
                features_list.append(det)
        return features_list

    def generate_html(
        self,
        output_path: Union[str, Path],
        include_plots: bool = True,
        standalone: bool = True,
    ) -> Path:
        """
        Generate an HTML report.

        Args:
            output_path: Output file path
            include_plots: Whether to include interactive Plotly plots
            standalone: Whether to create a standalone HTML file

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features_list = self._get_features_list()
        stats = self.statistics or compute_summary_statistics(features_list, self.slide_name)

        html_parts = []

        # HTML header
        html_parts.append(self._generate_html_header(standalone))

        # Report header
        html_parts.append(self._generate_report_header())

        # Summary statistics section
        html_parts.append(self._generate_summary_section(stats))

        # Vessel type breakdown section
        html_parts.append(self._generate_vessel_type_section(stats))

        # Quality metrics section
        html_parts.append(self._generate_quality_section(stats))

        # Plots section
        if include_plots and PLOTLY_AVAILABLE:
            html_parts.append(self._generate_plots_section(features_list))

        # Quantile tables section
        html_parts.append(self._generate_quantile_section(stats))

        # Footer
        html_parts.append(self._generate_footer())

        # Close HTML
        html_parts.append("</body></html>")

        # Write to file
        html_content = "\n".join(html_parts)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def generate_pdf(
        self,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate a PDF report.

        Requires weasyprint to be installed.

        Args:
            output_path: Output file path

        Returns:
            Path to generated PDF file
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "weasyprint is required for PDF export. "
                "Install with: pip install weasyprint"
            )

        output_path = Path(output_path)

        # Generate HTML first (without interactive plots for PDF)
        html_path = output_path.with_suffix(".tmp.html")
        self.generate_html(html_path, include_plots=False, standalone=True)

        # Convert to PDF
        HTML(filename=str(html_path)).write_pdf(str(output_path))

        # Clean up temp file
        html_path.unlink()

        return output_path

    def _generate_html_header(self, standalone: bool) -> str:
        """Generate HTML head section."""
        plotly_script = ""
        if PLOTLY_AVAILABLE and standalone:
            plotly_script = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vessel Segmentation Report - {self.slide_name}</title>
    <style>{REPORT_CSS}</style>
    {plotly_script}
</head>
<body>"""

    def _generate_report_header(self) -> str:
        """Generate report header section."""
        pixel_size = self.metadata.get("pixel_size_um", "N/A")
        if isinstance(pixel_size, list):
            pixel_size = pixel_size[0] if pixel_size else "N/A"

        return f"""
<div class="header">
    <h1>Vessel Segmentation Report</h1>
    <div class="subtitle">{self.slide_name}</div>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <div class="subtitle" style="margin-top: 10px;">
        Pixel Size: {pixel_size} um | Total Vessels: {len(self.detections)}
    </div>
</div>"""

    def _generate_summary_section(self, stats: VesselStatistics) -> str:
        """Generate summary statistics section."""
        d_stats = stats.diameter_stats
        w_stats = stats.wall_thickness_stats
        a_stats = stats.area_stats

        return f"""
<div class="section">
    <h2>Summary Statistics</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Total Vessels</div>
            <div class="value">{stats.vessel_count}</div>
        </div>
        <div class="stat-card secondary">
            <div class="label">Mean Diameter</div>
            <div class="value">{d_stats.get('mean', 0):.1f}<span class="unit"> um</span></div>
        </div>
        <div class="stat-card secondary">
            <div class="label">Median Diameter</div>
            <div class="value">{d_stats.get('median', 0):.1f}<span class="unit"> um</span></div>
        </div>
        <div class="stat-card accent">
            <div class="label">Mean Wall Thickness</div>
            <div class="value">{w_stats.get('mean', 0):.2f}<span class="unit"> um</span></div>
        </div>
        <div class="stat-card accent">
            <div class="label">Median Wall Thickness</div>
            <div class="value">{w_stats.get('q50', w_stats.get('median', 0)):.2f}<span class="unit"> um</span></div>
        </div>
        <div class="stat-card">
            <div class="label">Mean Lumen Area</div>
            <div class="value">{a_stats.get('lumen_area_mean', 0):.0f}<span class="unit"> um^2</span></div>
        </div>
    </div>

    <h3>Diameter Range</h3>
    <table>
        <tr>
            <th>Statistic</th>
            <th>Value (um)</th>
        </tr>
        <tr><td>Minimum</td><td>{d_stats.get('min', 0):.2f}</td></tr>
        <tr><td>25th Percentile (Q1)</td><td>{d_stats.get('q25', 0):.2f}</td></tr>
        <tr><td>Median (Q2)</td><td>{d_stats.get('median', 0):.2f}</td></tr>
        <tr><td>75th Percentile (Q3)</td><td>{d_stats.get('q75', 0):.2f}</td></tr>
        <tr><td>Maximum</td><td>{d_stats.get('max', 0):.2f}</td></tr>
        <tr><td>Standard Deviation</td><td>{d_stats.get('std', 0):.2f}</td></tr>
    </table>
</div>"""

    def _generate_vessel_type_section(self, stats: VesselStatistics) -> str:
        """Generate vessel type breakdown section."""
        types = stats.vessel_types
        total = sum(types.values()) or 1  # Avoid div by zero

        rows = []
        for vtype in ["capillary", "arteriole", "artery", "unknown"]:
            count = types.get(vtype, 0)
            pct = 100 * count / total
            badge_class = vtype
            rows.append(
                f'<tr><td><span class="vessel-type-badge {badge_class}">{vtype.capitalize()}</span></td>'
                f'<td>{count}</td><td>{pct:.1f}%</td></tr>'
            )

        return f"""
<div class="section">
    <h2>Vessel Type Breakdown</h2>
    <p style="color: #888; font-size: 0.9em; margin-bottom: 15px;">
        Classification by diameter: Capillary (&lt;10 um), Arteriole (10-100 um), Artery (&gt;100 um)
    </p>
    <table>
        <tr>
            <th>Vessel Type</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
        {''.join(rows)}
        <tr style="font-weight: bold;">
            <td>Total</td>
            <td>{total}</td>
            <td>100%</td>
        </tr>
    </table>
</div>"""

    def _generate_quality_section(self, stats: VesselStatistics) -> str:
        """Generate quality metrics section."""
        q_stats = stats.quality_stats
        conf_dist = q_stats.get("confidence_distribution", {})
        ring_stats = q_stats.get("ring_completeness", {})
        circ_stats = q_stats.get("circularity", {})

        # Confidence distribution rows
        conf_rows = []
        total_conf = sum(conf_dist.values()) or 1
        for level in ["high", "medium", "low", "unknown"]:
            count = conf_dist.get(level, 0)
            pct = 100 * count / total_conf
            conf_rows.append(
                f'<tr><td><span class="confidence-badge {level}">{level.capitalize()}</span></td>'
                f'<td>{count}</td><td>{pct:.1f}%</td></tr>'
            )

        return f"""
<div class="section">
    <h2>Quality Metrics</h2>

    <h3>Detection Confidence</h3>
    <table>
        <tr>
            <th>Confidence Level</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
        {''.join(conf_rows)}
    </table>

    <h3>Ring Completeness</h3>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Mean Completeness</div>
            <div class="value">{ring_stats.get('mean', 0):.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Median Completeness</div>
            <div class="value">{ring_stats.get('median', 0):.2f}</div>
        </div>
        <div class="stat-card secondary">
            <div class="label">Min - Max</div>
            <div class="value" style="font-size: 1.2em;">
                {ring_stats.get('min', 0):.2f} - {ring_stats.get('max', 0):.2f}
            </div>
        </div>
    </div>

    <h3>Circularity</h3>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Mean Circularity</div>
            <div class="value">{circ_stats.get('mean', 0):.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Median Circularity</div>
            <div class="value">{circ_stats.get('median', 0):.2f}</div>
        </div>
        <div class="stat-card accent">
            <div class="label">Std Dev</div>
            <div class="value">{circ_stats.get('std', 0):.2f}</div>
        </div>
    </div>
</div>"""

    def _generate_plots_section(self, features_list: List[Dict[str, Any]]) -> str:
        """Generate interactive plots section."""
        if not PLOTLY_AVAILABLE:
            return '<div class="section"><h2>Visualizations</h2><p class="no-data">Plotly not available for interactive plots.</p></div>'

        if not features_list:
            return '<div class="section"><h2>Visualizations</h2><p class="no-data">No detection data available for visualization.</p></div>'

        # Generate plots
        diameter_hist = create_diameter_histogram(features_list)
        wall_hist = create_wall_thickness_histogram(features_list)
        scatter_plot = create_diameter_vs_wall_scatter(features_list)
        area_scatter = create_lumen_vs_wall_scatter(features_list)
        vessel_pie = create_vessel_type_pie_chart(features_list)
        conf_hist = create_confidence_histogram(features_list)
        ring_hist = create_ring_completeness_histogram(features_list)

        # Convert to HTML divs
        def fig_to_div(fig, div_id: str) -> str:
            """Convert Plotly figure to HTML div."""
            return fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=div_id,
            )

        return f"""
<div class="section">
    <h2>Visualizations</h2>

    <h3>Size Distributions</h3>
    <div class="plot-row">
        <div class="plot-container">
            {fig_to_div(diameter_hist, 'diameter-hist')}
        </div>
        <div class="plot-container">
            {fig_to_div(wall_hist, 'wall-hist')}
        </div>
    </div>

    <h3>Morphological Relationships</h3>
    <div class="plot-row">
        <div class="plot-container">
            {fig_to_div(scatter_plot, 'diameter-wall-scatter')}
        </div>
        <div class="plot-container">
            {fig_to_div(area_scatter, 'lumen-wall-scatter')}
        </div>
    </div>

    <h3>Classification and Quality</h3>
    <div class="plot-row">
        <div class="plot-container">
            {fig_to_div(vessel_pie, 'vessel-type-pie')}
        </div>
        <div class="plot-container">
            {fig_to_div(conf_hist, 'confidence-hist')}
        </div>
    </div>

    <div class="plot-container">
        {fig_to_div(ring_hist, 'ring-completeness-hist')}
    </div>
</div>"""

    def _generate_quantile_section(self, stats: VesselStatistics) -> str:
        """Generate wall thickness quantile table section."""
        w_stats = stats.wall_thickness_stats

        quantile_rows = []
        for q in [10, 25, 50, 75, 90]:
            key = f"q{q}"
            val = w_stats.get(key, 0)
            quantile_rows.append(
                f'<tr><td>{q}th Percentile</td><td class="value">{val:.3f} um</td></tr>'
            )

        return f"""
<div class="section">
    <h2>Wall Thickness Quantiles</h2>
    <table class="quantile-table">
        <tr>
            <th>Quantile</th>
            <th>Wall Thickness</th>
        </tr>
        {(''.join(quantile_rows))}
        <tr>
            <td>Mean</td>
            <td class="value">{w_stats.get('mean', 0):.3f} um</td>
        </tr>
        <tr>
            <td>Std Dev</td>
            <td class="value">{w_stats.get('std', 0):.3f} um</td>
        </tr>
    </table>
</div>"""

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
<div class="footer">
    <p>Generated by vessel_seg reporting module</p>
    <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>"""


@dataclass
class BatchVesselReport:
    """
    Batch vessel report generator for multiple slides.

    Generates comparative reports across multiple slides with
    batch-level statistics and comparisons.
    """

    slide_reports: List[VesselReport] = field(default_factory=list)
    batch_name: str = "Batch Analysis"

    @classmethod
    def from_json_files(
        cls,
        json_paths: List[Union[str, Path]],
        batch_name: str = "Batch Analysis",
    ) -> "BatchVesselReport":
        """
        Create a BatchVesselReport from multiple JSON files.

        Args:
            json_paths: List of paths to vessel_detections.json files
            batch_name: Name for the batch

        Returns:
            BatchVesselReport instance
        """
        slide_reports = []
        for path in json_paths:
            try:
                report = VesselReport.from_json(path)
                slide_reports.append(report)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

        return cls(slide_reports=slide_reports, batch_name=batch_name)

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        pattern: str = "**/vessel_detections.json",
        batch_name: Optional[str] = None,
    ) -> "BatchVesselReport":
        """
        Create a BatchVesselReport from all detection files in a directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern for finding detection files
            batch_name: Optional batch name (defaults to directory name)

        Returns:
            BatchVesselReport instance
        """
        directory = Path(directory)
        json_paths = list(directory.glob(pattern))

        if batch_name is None:
            batch_name = directory.name

        return cls.from_json_files(json_paths, batch_name)

    def generate_html(
        self,
        output_path: Union[str, Path],
        include_individual_reports: bool = False,
    ) -> Path:
        """
        Generate a batch comparison HTML report.

        Args:
            output_path: Output file path
            include_individual_reports: Whether to include per-slide details

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect statistics
        slide_stats = [r.statistics for r in self.slide_reports if r.statistics]
        batch_comparison = compute_batch_comparison(slide_stats)

        html_parts = []

        # HTML header
        plotly_script = ""
        if PLOTLY_AVAILABLE:
            plotly_script = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

        html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Vessel Report - {self.batch_name}</title>
    <style>{REPORT_CSS}</style>
    {plotly_script}
</head>
<body>""")

        # Header
        html_parts.append(f"""
<div class="header">
    <h1>Batch Vessel Analysis Report</h1>
    <div class="subtitle">{self.batch_name}</div>
    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    <div class="subtitle" style="margin-top: 10px;">
        Slides: {len(self.slide_reports)} | Total Vessels: {batch_comparison.get('total_vessels', 0)}
    </div>
</div>""")

        # Batch overview section
        html_parts.append(self._generate_batch_overview(batch_comparison))

        # Comparison plots
        if PLOTLY_AVAILABLE and batch_comparison:
            html_parts.append(self._generate_batch_plots(batch_comparison))

        # Per-slide summary table
        html_parts.append(self._generate_slide_summary_table())

        # Individual slide sections (optional)
        if include_individual_reports:
            html_parts.append(self._generate_individual_sections())

        # Footer
        html_parts.append(f"""
<div class="footer">
    <p>Generated by vessel_seg reporting module</p>
    <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>""")

        # Write to file
        html_content = "\n".join(html_parts)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def _generate_batch_overview(self, batch_comp: Dict[str, Any]) -> str:
        """Generate batch overview statistics."""
        return f"""
<div class="section">
    <h2>Batch Overview</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Total Slides</div>
            <div class="value">{len(self.slide_reports)}</div>
        </div>
        <div class="stat-card">
            <div class="label">Total Vessels</div>
            <div class="value">{batch_comp.get('total_vessels', 0)}</div>
        </div>
        <div class="stat-card secondary">
            <div class="label">Avg Vessels/Slide</div>
            <div class="value">{batch_comp.get('avg_vessels_per_slide', 0):.1f}</div>
        </div>
        <div class="stat-card accent">
            <div class="label">Std Dev</div>
            <div class="value">{batch_comp.get('std_vessels_per_slide', 0):.1f}</div>
        </div>
    </div>
</div>"""

    def _generate_batch_plots(self, batch_comp: Dict[str, Any]) -> str:
        """Generate batch comparison plots."""
        if not PLOTLY_AVAILABLE:
            return ""

        slide_names = batch_comp.get("slide_names", [])
        vessel_counts = batch_comp.get("vessel_counts", [])
        mean_diameters = batch_comp.get("mean_diameters", [])

        plots_html = []

        # Vessel count comparison
        if slide_names and vessel_counts:
            count_fig = create_batch_comparison_bar(
                slide_names,
                vessel_counts,
                title="Vessel Count by Slide",
                yaxis_title="Count",
            )
            plots_html.append(
                f'<div class="plot-container">{count_fig.to_html(full_html=False, include_plotlyjs=False, div_id="vessel-count-bar")}</div>'
            )

        # Mean diameter comparison
        if slide_names and mean_diameters:
            diameter_fig = create_batch_comparison_bar(
                slide_names,
                mean_diameters,
                title="Mean Diameter by Slide",
                yaxis_title="Diameter (um)",
                color="#44a",
            )
            plots_html.append(
                f'<div class="plot-container">{diameter_fig.to_html(full_html=False, include_plotlyjs=False, div_id="mean-diameter-bar")}</div>'
            )

        # Diameter distribution violin plot
        slide_diameters = {}
        for report in self.slide_reports:
            features = report._get_features_list()
            diameters = [f.get("outer_diameter_um") for f in features if f.get("outer_diameter_um")]
            if diameters:
                slide_diameters[report.slide_name] = diameters

        if slide_diameters:
            violin_fig = create_batch_comparison_violin(
                slide_diameters,
                title="Diameter Distribution by Slide",
                yaxis_title="Diameter (um)",
            )
            plots_html.append(
                f'<div class="plot-container">{violin_fig.to_html(full_html=False, include_plotlyjs=False, div_id="diameter-violin")}</div>'
            )

        return f"""
<div class="section">
    <h2>Batch Comparisons</h2>
    {''.join(plots_html)}
</div>"""

    def _generate_slide_summary_table(self) -> str:
        """Generate summary table for all slides."""
        rows = []
        for report in self.slide_reports:
            stats = report.statistics
            if not stats:
                continue

            d_mean = stats.diameter_stats.get("mean", 0)
            w_mean = stats.wall_thickness_stats.get("mean", 0)
            high_conf = stats.quality_stats.get("confidence_distribution", {}).get("high", 0)

            rows.append(f"""
<tr>
    <td>{report.slide_name}</td>
    <td>{stats.vessel_count}</td>
    <td>{d_mean:.1f}</td>
    <td>{w_mean:.2f}</td>
    <td>{high_conf}</td>
</tr>""")

        return f"""
<div class="section">
    <h2>Per-Slide Summary</h2>
    <table>
        <tr>
            <th>Slide</th>
            <th>Vessel Count</th>
            <th>Mean Diameter (um)</th>
            <th>Mean Wall (um)</th>
            <th>High Confidence</th>
        </tr>
        {''.join(rows)}
    </table>
</div>"""

    def _generate_individual_sections(self) -> str:
        """Generate detailed sections for each slide."""
        sections = []
        for report in self.slide_reports:
            stats = report.statistics
            if not stats:
                continue

            sections.append(f"""
<div class="slide-summary">
    <h4>{report.slide_name}</h4>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Vessels</div>
            <div class="value">{stats.vessel_count}</div>
        </div>
        <div class="stat-card secondary">
            <div class="label">Mean Diameter</div>
            <div class="value">{stats.diameter_stats.get('mean', 0):.1f}<span class="unit"> um</span></div>
        </div>
        <div class="stat-card accent">
            <div class="label">Mean Wall</div>
            <div class="value">{stats.wall_thickness_stats.get('mean', 0):.2f}<span class="unit"> um</span></div>
        </div>
    </div>
</div>""")

        return f"""
<div class="section batch-section">
    <h2>Individual Slide Details</h2>
    {''.join(sections)}
</div>"""


def generate_vessel_report(
    detection_source: Union[str, Path, List[Dict[str, Any]]],
    output_dir: Union[str, Path],
    formats: List[str] = ["html"],
    slide_name: Optional[str] = None,
    report_name: str = "vessel_report",
) -> Dict[str, Path]:
    """
    Convenience function to generate vessel reports.

    Args:
        detection_source: Path to JSON file or list of detection dicts
        output_dir: Directory for output files
        formats: List of formats to generate ("html", "pdf")
        slide_name: Optional slide name
        report_name: Base name for output files

    Returns:
        Dict mapping format to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create report
    if isinstance(detection_source, (str, Path)):
        report = VesselReport.from_json(detection_source, slide_name)
    else:
        report = VesselReport.from_detections(
            detection_source,
            slide_name=slide_name or "Unknown",
        )

    # Generate requested formats
    outputs: Dict[str, Path] = {}

    if "html" in formats:
        html_path = output_dir / f"{report_name}.html"
        report.generate_html(html_path)
        outputs["html"] = html_path

    if "pdf" in formats:
        pdf_path = output_dir / f"{report_name}.pdf"
        try:
            report.generate_pdf(pdf_path)
            outputs["pdf"] = pdf_path
        except ImportError as e:
            print(f"Warning: PDF generation skipped - {e}")

    return outputs
