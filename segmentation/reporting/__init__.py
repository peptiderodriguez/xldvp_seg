"""
Vessel segmentation reporting module.

Generates PDF and HTML reports with statistical analysis and visualizations
for vessel detection results.

Usage:
    from segmentation.reporting import VesselReport, generate_vessel_report

    # Load detections and generate report
    report = VesselReport.from_json("vessel_detections.json")
    report.generate_html("report.html")
    report.generate_pdf("report.pdf")  # Requires weasyprint

    # Or use the convenience function
    generate_vessel_report(
        "vessel_detections.json",
        output_dir="reports/",
        formats=["html", "pdf"]
    )
"""

from .stats import (
    VesselStatistics,
    compute_summary_statistics,
    compute_diameter_distribution,
    compute_wall_thickness_quantiles,
    compute_vessel_type_breakdown,
    compute_quality_metrics,
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
)

from .vessel_report import (
    VesselReport,
    generate_vessel_report,
    BatchVesselReport,
)

__all__ = [
    # Statistics
    "VesselStatistics",
    "compute_summary_statistics",
    "compute_diameter_distribution",
    "compute_wall_thickness_quantiles",
    "compute_vessel_type_breakdown",
    "compute_quality_metrics",
    "compute_batch_comparison",
    # Plots
    "create_diameter_histogram",
    "create_wall_thickness_histogram",
    "create_diameter_vs_wall_scatter",
    "create_lumen_vs_wall_scatter",
    "create_vessel_type_pie_chart",
    "create_confidence_histogram",
    "create_ring_completeness_histogram",
    "create_batch_comparison_bar",
    "create_batch_comparison_violin",
    # Report generation
    "VesselReport",
    "generate_vessel_report",
    "BatchVesselReport",
]
