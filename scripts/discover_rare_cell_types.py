#!/usr/bin/env python
"""CLI entry for rare-cell-population discovery (morph + SAM2 + spatial).

Runs :func:`xldvp_seg.analysis.rare_cell_discovery.discover_rare_cell_types`
over a detection JSON, writes augmented detections + cluster summary CSV +
Ward linkage on centroids + dendrogram PNG, then (optionally) a card-grid
annotation HTML of exemplar cells per cluster for manual review.

Typical usage (n45, GPU-accelerated on p.hpcl93)::

    xlseg discover-rare-cells \\
        --detections cell_detections_with_organs_pts64.json \\
        --feature-groups shape,color,sam2 \\
        --min-cluster-size 1000 \\
        --stability-sizes 500,1000,2000 \\
        --use-gpu \\
        --czi-path slide.czi \\
        --display-channels 2,4 \\
        --n-exemplars-per-cluster 100 \\
        --output-dir rare_cells_out/

Outputs:
    - ``detections_with_rare_labels.json``
    - ``cluster_summary.csv``
    - ``linkage.npy``   (scipy Ward linkage matrix)
    - ``dendrogram.png``
    - ``exemplars_annotation/index.html`` + cards (optional)
    - ``X_pca_<hash>.npz`` + ``W_delaunay_k10.npz``  (cached for re-runs)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

from xldvp_seg.analysis.rare_cell_discovery import (  # noqa: E402
    RareCellConfig,
    discover_rare_cell_types,
)
from xldvp_seg.exceptions import ExportError  # noqa: E402
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _atomic_np_save(path: Path, arr: np.ndarray) -> None:
    """``np.save`` via temp file + ``os.replace`` for crash-safe writes.

    Use an open file handle so ``np.save``'s ``.npy`` auto-append doesn't
    break the atomic rename.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Dendrogram PNG
# ---------------------------------------------------------------------------


def plot_dendrogram(
    linkage: np.ndarray,
    cluster_ids: np.ndarray,
    summary_rows: list[dict],
    output_path: Path,
    stable_only: bool = True,
) -> None:
    """Render Ward-linkage dendrogram on cluster centroids to PNG.

    Leaves colored by rarity tier: smallest 20% in bright red, middle 60% in
    olive, largest 20% in dim gray. Labels are cluster IDs.
    """
    from scipy.cluster.hierarchy import dendrogram

    if linkage.shape[0] < 2:
        logger.warning("Linkage has <3 clusters; skipping dendrogram")
        return

    # Build id→size / stable lookup
    size_by_id = {row["cluster_id"]: row["size"] for row in summary_rows}
    stable_by_id = {row["cluster_id"]: row["stable"] for row in summary_rows}

    # Optionally restrict to stable clusters (produce a sub-linkage). For now
    # we render all with stable flagged; filtering requires re-linkage which
    # we skip at this layer for simplicity.
    sizes = np.array([size_by_id.get(int(cid), 0) for cid in cluster_ids])
    # Rarity tiers: smallest 20% = rare, middle 60%, largest 20% = common
    p20, p80 = np.percentile(sizes, [20, 80]) if len(sizes) > 0 else (0, 0)

    def _leaf_color(idx: int) -> str:
        cid = int(cluster_ids[idx])
        # Missing-from-summary is abnormal — treat as exploratory (not stable).
        if not stable_by_id.get(cid, False):
            return "#888888"  # exploratory: dim
        sz = size_by_id.get(cid, 0)
        if sz <= p20:
            return "#e6194b"  # rare: bright red
        if sz >= p80:
            return "#5a5a5a"  # common: dim
        return "#bfef45"  # middle: olive

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(cluster_ids)), 6), dpi=120)
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#111")
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_color("#333")

    dendrogram(
        linkage,
        labels=[str(int(c)) for c in cluster_ids],
        leaf_font_size=9,
        color_threshold=0.0,
        above_threshold_color="#bbbbbb",
        ax=ax,
    )
    # Apply leaf colors
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(_leaf_color(i))

    ax.set_title(
        "Ward linkage on cluster centroids (red=rare, olive=typical, gray=common/exploratory)",
        color="#ddd",
        fontsize=11,
    )
    ax.set_ylabel("Ward distance", color="#ccc")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote dendrogram: %s", output_path)


# ---------------------------------------------------------------------------
# Cluster summary CSV
# ---------------------------------------------------------------------------


def write_cluster_summary_csv(summary: list[dict], path: Path) -> None:
    """Write per-cluster summary CSV atomically (tmp + os.replace).

    Noise (-1) is excluded by construction. ``moran_i = None`` (degenerate
    cluster) is emitted as an empty cell — the viewer's CSV loader handles
    both forms.
    """
    fieldnames = [
        "cluster_id",
        "size",
        "hdbscan_persistence",
        "moran_i",
        "stable",
        "noise_pct",
        "top_regions",
        "top_morph_features",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary:
            out = {}
            for k in fieldnames:
                v = row.get(k, "")
                out[k] = "" if v is None else v
            w.writerow(out)
    os.replace(tmp, path)
    logger.info(
        "Wrote cluster summary: %s (%d stable, %d total)",
        path,
        sum(1 for r in summary if r.get("stable")),
        len(summary),
    )


# ---------------------------------------------------------------------------
# Exemplar annotation HTML
# ---------------------------------------------------------------------------


def _pick_exemplars(
    kept_detections: list[dict],
    labels: np.ndarray,
    summary: list[dict],
    n_per_cluster: int,
    top_n_clusters: int = 20,
    stable_only: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Sample exemplars: uniform random per cluster, capped at cluster size.

    Picks the smallest N clusters by size (smallest = rarest), stable-only
    if requested. Uses ``np.random.default_rng`` per project convention.
    """
    rng = np.random.default_rng(seed)
    eligible = sorted(
        (row for row in summary if (not stable_only) or row["stable"]),
        key=lambda r: r["size"],
    )[:top_n_clusters]
    picked_cids = {row["cluster_id"] for row in eligible}

    buckets: dict[int, list[int]] = {cid: [] for cid in picked_cids}
    for i, lbl in enumerate(labels):
        cid = int(lbl)
        if cid in buckets:
            buckets[cid].append(i)

    exemplars = []
    for cid in picked_cids:
        indices = buckets[cid]
        n = min(n_per_cluster, len(indices))
        if n > 0:
            sampled = rng.choice(indices, size=n, replace=False).tolist()
        else:
            sampled = []
        for idx in sampled:
            det = dict(kept_detections[int(idx)])  # shallow copy
            det["_exemplar_cluster_id"] = cid
            exemplars.append(det)
    return exemplars


def _generate_exemplar_html(
    exemplars: list[dict],
    czi_path: Path | None,
    display_channels: str,
    output_dir: Path,
    title: str,
) -> None:
    """Generate paginated card-grid annotation HTML via regenerate_html.py."""
    if not exemplars:
        logger.warning("No exemplars to render — skipping HTML")
        return
    if czi_path is None:
        logger.warning("--czi-path not provided — skipping exemplar HTML (need fluorescence)")
        return

    exemplar_json = output_dir / "exemplar_detections.json"
    atomic_json_dump(exemplars, str(exemplar_json))

    import subprocess

    html_dir = output_dir / "exemplars_annotation"
    html_dir.mkdir(parents=True, exist_ok=True)
    regen = REPO / "scripts" / "regenerate_html.py"
    cmd = [
        sys.executable,
        str(regen),
        "--detections",
        str(exemplar_json),
        "--czi-path",
        str(czi_path),
        "--output-dir",
        str(output_dir),
        "--html-dir",
        str(html_dir),
        "--display-channels",
        display_channels,
        "--dashed-contour",
        "--max-samples",
        str(len(exemplars)),
        "--crop-context-factor",
        "2.5",
        "--contour-thickness",
        "2",
    ]
    logger.info("Generating exemplar annotation HTML: %s", html_dir)
    logger.debug("Command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ExportError(
            f"regenerate_html.py failed (exit {result.returncode}):\n" f"{result.stderr[-2000:]}"
        )
    logger.info("Exemplar HTML written to: %s/index.html", html_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--detections", type=Path, required=True, help="Input detection JSON")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--feature-groups",
        type=str,
        default="shape,color,sam2",
        help="Comma-separated feature groups (shape,color,sam2,channel,deep). "
        "Default: shape,color,sam2 (= 'morph'+SAM2, ~334D).",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=1000,
        help="HDBSCAN min_cluster_size. At 500K cells with default 1000, "
        "expect ~20-80 clusters total. (default: 1000)",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples — DENSITY SENSITIVITY, not cluster count. "
        "Higher → more noise, fewer marginal clusters. "
        "Lower → more clusters emerge. "
        "Default: min_cluster_size // 10. Most conservative: = min_cluster_size.",
    )
    p.add_argument(
        "--stability-sizes",
        type=str,
        default="500,1000,2000",
        help="Comma-separated min_cluster_size values for the stability check "
        "(clusters surviving ≥2 runs are 'stable'). Default: 500,1000,2000.",
    )
    p.add_argument(
        "--max-pcs",
        type=int,
        default=30,
        help="Max PCA components (HDBSCAN degrades above ~30 dims). Default: 30.",
    )
    p.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="PCA variance target (capped by --max-pcs). Default: 0.95.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cuML for PCA+HDBSCAN if available (default: on). "
        "Pass --no-use-gpu to force CPU.",
    )
    p.add_argument(
        "--feature-group-weights",
        type=str,
        default="equal",
        help="Per-group feature weighting: 'equal' (default, 1/sqrt(group_dim) "
        "so each group contributes ~equal squared distance) or 'raw' (no "
        "weighting — SAM2 dominates morph 3×).",
    )

    # Pre-filter thresholds
    p.add_argument("--filter-min-n-nuclei", type=int, default=1)
    p.add_argument("--filter-nc-min", type=float, default=0.02)
    p.add_argument("--filter-nc-max", type=float, default=0.95)
    p.add_argument("--filter-min-overlap", type=float, default=0.8)
    p.add_argument("--filter-area-min-um2", type=float, default=20.0)
    p.add_argument("--filter-area-max-um2", type=float, default=5000.0)

    # Exemplar HTML
    p.add_argument(
        "--n-exemplars-per-cluster",
        type=int,
        default=100,
        help="Uniform random exemplars per cluster for annotation HTML (default: 100).",
    )
    p.add_argument(
        "--top-n-clusters",
        type=int,
        default=20,
        help="Top-N smallest stable clusters to render as exemplars (default: 20).",
    )
    p.add_argument("--czi-path", type=Path, default=None, help="CZI for exemplar HTML background.")
    p.add_argument("--display-channels", type=str, default="2,4", help="RGB display channels.")
    p.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Also render exemplars for unstable (exploratory) clusters.",
    )
    p.add_argument("--no-exemplar-html", action="store_true", help="Skip exemplar HTML generation.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = RareCellConfig(
        feature_groups=tuple(s.strip() for s in args.feature_groups.split(",") if s.strip()),
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        stability_sizes=tuple(int(s) for s in args.stability_sizes.split(",") if s.strip()),
        max_pcs=args.max_pcs,
        pca_variance=args.pca_variance,
        feature_group_weights=args.feature_group_weights,
        nuc_filter_min_n_nuclei=args.filter_min_n_nuclei,
        nuc_filter_nc_min=args.filter_nc_min,
        nuc_filter_nc_max=args.filter_nc_max,
        nuc_filter_min_overlap=args.filter_min_overlap,
        area_filter_min_um2=args.filter_area_min_um2,
        area_filter_max_um2=args.filter_area_max_um2,
        use_gpu=args.use_gpu,
        seed=args.seed,
        cache_dir=args.output_dir,
    )

    logger.info("Loading detections: %s", args.detections)
    detections = fast_json_load(str(args.detections))
    logger.info("  %d detections loaded", len(detections))

    result = discover_rare_cell_types(detections, cfg)

    # Write outputs
    out_detections = args.output_dir / "detections_with_rare_labels.json"
    atomic_json_dump(result["detections"], str(out_detections))
    logger.info("Wrote augmented detections: %s", out_detections)

    write_cluster_summary_csv(result["cluster_summary"], args.output_dir / "cluster_summary.csv")

    _atomic_np_save(args.output_dir / "linkage.npy", result["linkage"])
    _atomic_np_save(args.output_dir / "cluster_ids.npy", result["cluster_ids"])
    logger.info(
        "Wrote linkage (shape=%s) + cluster_ids (%d IDs)",
        result["linkage"].shape,
        len(result["cluster_ids"]),
    )

    plot_dendrogram(
        result["linkage"],
        result["cluster_ids"],
        result["cluster_summary"],
        args.output_dir / "dendrogram.png",
    )

    # Report quick stats
    n_stable = sum(1 for r in result["cluster_summary"] if r.get("stable"))
    logger.info(
        "═══ Pipeline complete ═══ "
        "PCA: %d dims (%.1f%% var) | "
        "HDBSCAN: %d clusters (%d stable, %.1f%% noise) | "
        "kept %d / %d cells",
        result["pca_n_components"],
        100 * result["pca_variance"],
        len(result["cluster_summary"]),
        n_stable,
        100 * result["noise_pct"],
        len(result["kept_detections"]),
        result["prefilter_stats"]["input"],
    )

    # Exemplar HTML
    if not args.no_exemplar_html:
        exemplars = _pick_exemplars(
            result["kept_detections"],
            result["labels"],
            result["cluster_summary"],
            args.n_exemplars_per_cluster,
            top_n_clusters=args.top_n_clusters,
            stable_only=not args.include_exploratory,
            seed=args.seed,
        )
        logger.info(
            "Sampled %d exemplars across %d clusters for annotation HTML",
            len(exemplars),
            min(args.top_n_clusters, n_stable),
        )
        _generate_exemplar_html(
            exemplars,
            args.czi_path,
            args.display_channels,
            args.output_dir,
            f"Rare populations (top {args.top_n_clusters} stable)",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
