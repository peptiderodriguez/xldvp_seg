#!/usr/bin/env python
"""Plot per-region % multinucleated-cell distribution from region_nuc_stats.json.

For each region, computes % cells with n_nuclei >= 2 (multinucleation fraction).
Writes a PNG histogram + a sorted CSV of regions ranked by multinucleation.

Usage:
    python scripts/region_multinuc_plot.py \\
        --nuc-stats region_nuc_stats.json \\
        --min-cells 1000 \\
        --output-dir out/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nuc-stats", required=True, help="region_nuc_stats.json")
    p.add_argument("--min-cells", type=int, default=1000, help="Min nucleated cells per region")
    p.add_argument(
        "--threshold", type=int, default=2, help="n_nuclei >= this counts as multinucleated"
    )
    p.add_argument("--output-dir", default=".", help="Output directory")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.nuc_stats) as f:
        raw = json.load(f)
    stats = {int(k): v for k, v in raw.items()}

    rows = []
    for rid, d in stats.items():
        total = d.get("count", 0)
        if total < args.min_cells:
            continue
        dist = d.get("nuc_dist", {})
        multi = sum(int(v) for k, v in dist.items() if int(k) >= args.threshold)
        # Per-bin fractions (n=1, n=2, n=3, n=4+), normalized over total (n>=1)
        n1 = int(dist.get("1", 0))
        n2 = int(dist.get("2", 0))
        n3 = int(dist.get("3", 0))
        n4p = sum(int(v) for k, v in dist.items() if int(k) >= 4)
        rows.append(
            {
                "region": rid,
                "n_cells": total,
                "n_multi": multi,
                "pct_multi": 100.0 * multi / total,
                "pct_n1": 100.0 * n1 / total,
                "pct_n2": 100.0 * n2 / total,
                "pct_n3": 100.0 * n3 / total,
                "pct_n4p": 100.0 * n4p / total,
                "mean_nuc": d.get("mean_nuc", d.get("mean", 0.0)),
                "median_nuc": d.get("median_nuc", d.get("median", 0)),
            }
        )

    if not rows:
        print(f"No regions with >= {args.min_cells} nucleated cells")
        return

    print(f"Analyzed {len(rows)} regions (>= {args.min_cells} nucleated cells)")

    pcts = np.array([r["pct_multi"] for r in rows])
    print(
        f"%% multinucleated (n_nuclei >= {args.threshold}): "
        f"mean={pcts.mean():.2f} median={np.median(pcts):.2f} "
        f"min={pcts.min():.2f} max={pcts.max():.2f}"
    )

    # --- Tukey fences ---
    q1, q3 = np.percentile(pcts, [25, 75])
    iqr = q3 - q1
    tukey_inner_low = q1 - 1.5 * iqr
    tukey_inner_high = q3 + 1.5 * iqr
    tukey_outer_low = q1 - 3 * iqr
    tukey_outer_high = q3 + 3 * iqr
    tukey_mild = [
        r for r in rows if r["pct_multi"] < tukey_inner_low or r["pct_multi"] > tukey_inner_high
    ]
    tukey_extreme = [
        r for r in rows if r["pct_multi"] < tukey_outer_low or r["pct_multi"] > tukey_outer_high
    ]
    print(
        f"\nTukey fences: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f} | "
        f"inner=[{tukey_inner_low:.2f}, {tukey_inner_high:.2f}] outer=[{tukey_outer_low:.2f}, {tukey_outer_high:.2f}]"
    )
    print(f"Tukey mild outliers: {len(tukey_mild)} | extreme: {len(tukey_extreme)}")

    # --- GMM with BIC k=1 vs k=2 ---
    # GaussianMixture(n_components=2) needs at least 2 samples; skip cleanly
    # for degenerate inputs so callers still get Tukey results + a posterior=0
    # fallback on every row.
    gmm_outliers = []
    gmm_threshold = None
    if len(rows) < 2:
        print("\nGMM: skipped (need >= 2 regions). Use Tukey/MAD.")
        for r in rows:
            r["gmm_posterior"] = 0.0
            r["gmm_outlier"] = False
    else:
        from sklearn.mixture import GaussianMixture

        X = pcts.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5).fit(X)
        bic1, bic2 = gmm1.bic(X), gmm2.bic(X)
        print(f"\nGMM: BIC(k=1)={bic1:.1f}, BIC(k=2)={bic2:.1f}, delta={bic1-bic2:.1f}")

        if bic2 < bic1 - 6:  # k=2 preferred (delta >= 6 = strong evidence)
            # Minority component = outliers
            weights = gmm2.weights_
            means = gmm2.means_.flatten()
            stds = np.sqrt(gmm2.covariances_.flatten())
            minority_idx = int(np.argmin(weights))
            majority_idx = 1 - minority_idx
            print(
                f"GMM k=2 preferred (delta>=6): majority mean={means[majority_idx]:.2f}\u00b1{stds[majority_idx]:.2f} "
                f"(w={weights[majority_idx]:.2f}), minority mean={means[minority_idx]:.2f}\u00b1{stds[minority_idx]:.2f} "
                f"(w={weights[minority_idx]:.2f})"
            )
            # Posterior probability of minority component per region
            posteriors = gmm2.predict_proba(X)[:, minority_idx]
            for r, p in zip(rows, posteriors):
                r["gmm_posterior"] = float(p)
                r["gmm_outlier"] = bool(p > 0.5)
                if p > 0.5:
                    gmm_outliers.append(r)
            # Threshold: crossover point where posteriors are equal
            # For two Gaussians, this is computed analytically but simpler to scan
            x_scan = np.linspace(pcts.min(), pcts.max(), 1000).reshape(-1, 1)
            post_scan = gmm2.predict_proba(x_scan)[:, minority_idx]
            cross_idx = np.argmin(np.abs(post_scan - 0.5))
            gmm_threshold = float(x_scan[cross_idx, 0])
            print(
                f"GMM outliers: {len(gmm_outliers)} (posterior > 0.5, threshold at {gmm_threshold:.2f}%)"
            )
        else:
            print("GMM: k=1 preferred (single population, no clean split). Use Tukey/MAD.")
            for r in rows:
                r["gmm_posterior"] = 0.0
                r["gmm_outlier"] = False

    # Histogram + ranked %multi + stacked composition
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    fig.patch.set_facecolor("#0a0a0a")
    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="#bbb")
        for s in ax.spines.values():
            s.set_color("#333")

    ax = axes[0]
    # Histogram on density scale so the KDE overlay is comparable
    _, bin_edges, _ = ax.hist(
        pcts, bins=30, color="#4caf50", edgecolor="#0a0a0a", density=True, alpha=0.85
    )
    # Gaussian KDE clipped to the observed data range
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(pcts)
    x_kde = np.linspace(pcts.min(), pcts.max(), 400)
    y_kde = kde(x_kde)
    ax.plot(x_kde, y_kde, color="#42d4f4", linewidth=2, label="Gaussian KDE")
    ax.fill_between(x_kde, y_kde, color="#42d4f4", alpha=0.12)
    ax.axvline(
        np.median(pcts), color="#ff9800", linestyle="--", label=f"median={np.median(pcts):.1f}%"
    )
    ax.axvline(pcts.mean(), color="#e6194b", linestyle="--", label=f"mean={pcts.mean():.1f}%")
    # Tukey inner fences
    xr = (pcts.min(), pcts.max())
    if tukey_inner_high <= xr[1] + 0.5:
        ax.axvline(
            tukey_inner_high,
            color="#bfef45",
            linestyle=":",
            linewidth=1.5,
            label=f"Tukey high={tukey_inner_high:.1f}%",
        )
    if tukey_inner_low >= xr[0] - 0.5:
        ax.axvline(tukey_inner_low, color="#bfef45", linestyle=":", linewidth=1.5)
    # GMM threshold (if k=2 wins)
    if gmm_threshold is not None:
        ax.axvline(
            gmm_threshold,
            color="#f032e6",
            linestyle="-.",
            linewidth=1.5,
            label=f"GMM cutoff={gmm_threshold:.1f}%",
        )
    ax.set_xlabel(f"% cells with n_nuclei >= {args.threshold}", color="#ddd")
    ax.set_ylabel("density", color="#ddd")
    ax.set_title(
        f"Per-region multinucleation ({len(rows)} regions, >= {args.min_cells} cells)",
        color="#fff",
    )
    ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#ddd")
    ax.set_xlim(pcts.min(), pcts.max())

    # Ranked bar chart — color-coded by outlier status
    rows_sorted = sorted(rows, key=lambda r: -r["pct_multi"])
    xs = np.arange(len(rows_sorted))
    ys = np.array([r["pct_multi"] for r in rows_sorted])

    def _bar_color(r):
        tm = r["pct_multi"] < tukey_inner_low or r["pct_multi"] > tukey_inner_high
        te = r["pct_multi"] < tukey_outer_low or r["pct_multi"] > tukey_outer_high
        go = bool(r.get("gmm_outlier"))
        if te and go:
            return "#e6194b"  # red: extreme + GMM
        if tm and go:
            return "#ff9800"  # orange: Tukey-mild + GMM
        if tm:
            return "#bfef45"  # yellow-green: Tukey-mild only
        if go:
            return "#f032e6"  # magenta: GMM only
        return "#4caf50"  # green: normal

    colors = [_bar_color(r) for r in rows_sorted]
    ax = axes[1]
    ax.bar(xs, ys, color=colors, width=1.0, edgecolor="none")
    if gmm_threshold is not None:
        ax.axhline(gmm_threshold, color="#f032e6", linestyle="-.", linewidth=1, alpha=0.6)
    if tukey_inner_high < ys.max() + 1:
        ax.axhline(tukey_inner_high, color="#bfef45", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Region rank (sorted by % multi)", color="#ddd")
    ax.set_ylabel(f"% n_nuclei >= {args.threshold}", color="#ddd")
    ax.set_title("Ranked by %multi (color = outlier class)", color="#fff")
    # Legend (only include categories that appear in the data)
    from matplotlib.patches import Patch

    legend_items = [
        ("#e6194b", "Tukey extreme + GMM"),
        ("#ff9800", "Tukey mild + GMM"),
        ("#bfef45", "Tukey mild only"),
        ("#f032e6", "GMM only"),
        ("#4caf50", "normal"),
    ]
    used = set(colors)
    handles = [Patch(facecolor=c, label=lbl) for c, lbl in legend_items if c in used]
    ax.legend(
        handles=handles,
        facecolor="#111",
        edgecolor="#333",
        labelcolor="#ddd",
        loc="upper right",
        fontsize=8,
    )
    # Label top 5 and bottom 3
    for i in list(range(5)) + list(range(len(rows_sorted) - 3, len(rows_sorted))):
        if 0 <= i < len(rows_sorted):
            r = rows_sorted[i]
            ax.text(
                i, r["pct_multi"] + 0.5, str(r["region"]), ha="center", fontsize=7, color="#ccc"
            )

    # Stacked 100% bar chart: n=1 / n=2 / n=3 / n=4+, sorted by % multi desc
    ax = axes[2]
    n1 = np.array([r["pct_n1"] for r in rows_sorted])
    n2 = np.array([r["pct_n2"] for r in rows_sorted])
    n3 = np.array([r["pct_n3"] for r in rows_sorted])
    n4p = np.array([r["pct_n4p"] for r in rows_sorted])
    xs = np.arange(len(rows_sorted))

    ax.bar(xs, n1, color="#8dd98e", width=1.0, edgecolor="none", label="n=1")
    ax.bar(xs, n2, bottom=n1, color="#42d4f4", width=1.0, edgecolor="none", label="n=2")
    ax.bar(xs, n3, bottom=n1 + n2, color="#ff9800", width=1.0, edgecolor="none", label="n=3")
    ax.bar(xs, n4p, bottom=n1 + n2 + n3, color="#e6194b", width=1.0, edgecolor="none", label="n>=4")
    ax.set_xlim(-0.5, len(rows_sorted) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Region rank (sorted by % multi desc)", color="#ddd")
    ax.set_ylabel("% of nucleated cells", color="#ddd")
    ax.set_title("Per-region nuclear count composition", color="#fff")
    ax.legend(
        facecolor="#111", edgecolor="#333", labelcolor="#ddd", loc="lower right", ncol=4, fontsize=9
    )

    plt.tight_layout()
    png_path = out_dir / "region_multinuc_histogram.png"
    plt.savefig(png_path, dpi=150, facecolor="#0a0a0a")
    plt.close()
    print(f"Wrote {png_path}")

    # Ranked CSV
    csv_path = out_dir / "region_multinuc_ranked.csv"
    # Annotate rows with Tukey flags
    for r in rows_sorted:
        r["tukey_mild"] = bool(
            r["pct_multi"] < tukey_inner_low or r["pct_multi"] > tukey_inner_high
        )
        r["tukey_extreme"] = bool(
            r["pct_multi"] < tukey_outer_low or r["pct_multi"] > tukey_outer_high
        )

    with open(csv_path, "w") as f:
        f.write(
            "region,n_cells,n_multi,pct_multi,pct_n1,pct_n2,pct_n3,pct_n4p,"
            "mean_nuc,median_nuc,tukey_mild,tukey_extreme,gmm_posterior,gmm_outlier\n"
        )
        for r in rows_sorted:
            f.write(
                f"{r['region']},{r['n_cells']},{r['n_multi']},{r['pct_multi']:.2f},"
                f"{r['pct_n1']:.2f},{r['pct_n2']:.2f},{r['pct_n3']:.2f},{r['pct_n4p']:.2f},"
                f"{r['mean_nuc']},{r['median_nuc']},"
                f"{int(r['tukey_mild'])},{int(r['tukey_extreme'])},"
                f"{r['gmm_posterior']:.3f},{int(r['gmm_outlier'])}\n"
            )
    print(f"Wrote {csv_path}")

    # Top 10 / bottom 5 summary
    print("\n=== Top 10 most multinucleated ===")
    for r in rows_sorted[:10]:
        print(
            f"  Region {r['region']:>4}: {r['n_cells']:>5} cells | "
            f"{r['pct_multi']:>5.1f}% multi | mean_nuc={r['mean_nuc']}"
        )
    print("\n=== Bottom 5 (least multinucleated) ===")
    for r in rows_sorted[-5:]:
        print(
            f"  Region {r['region']:>4}: {r['n_cells']:>5} cells | "
            f"{r['pct_multi']:>5.1f}% multi | mean_nuc={r['mean_nuc']}"
        )

    # Outlier summary
    tukey_ext_ids = [r["region"] for r in rows_sorted if r["tukey_extreme"]]
    tukey_mild_only = [
        r["region"] for r in rows_sorted if r["tukey_mild"] and not r["tukey_extreme"]
    ]
    # Tukey "+": all regions above inner fence (includes both mild and extreme)
    tukey_plus_ids = sorted([r["region"] for r in rows_sorted if r["tukey_mild"]])
    gmm_ids = sorted([r["region"] for r in rows_sorted if r.get("gmm_outlier")])

    # Write outlier ID lists as JSON for downstream viewers
    outlier_json = {
        "thresholds": {
            "tukey_inner_low": float(tukey_inner_low),
            "tukey_inner_high": float(tukey_inner_high),
            "tukey_outer_low": float(tukey_outer_low),
            "tukey_outer_high": float(tukey_outer_high),
            "gmm_threshold": float(gmm_threshold) if gmm_threshold is not None else None,
        },
        "tukey_plus": tukey_plus_ids,
        "tukey_extreme": sorted(tukey_ext_ids),
        "gmm_outliers": gmm_ids,
        "tukey_plus_and_gmm": sorted(set(tukey_plus_ids) & set(gmm_ids)),
    }
    json_path = out_dir / "region_multinuc_outlier_ids.json"
    with open(json_path, "w") as f:
        json.dump(outlier_json, f, indent=2)
    print(f"Wrote {json_path}")
    # Also write flat ID lists (directly consumable by --highlight-regions)
    for key in ("tukey_plus", "tukey_extreme", "gmm_outliers"):
        p = out_dir / f"region_multinuc_{key}_ids.json"
        with open(p, "w") as f:
            json.dump(outlier_json[key], f)
    print("\n=== Outlier flags ===")
    print(
        f"  Tukey extreme (|x-Q|>3*IQR): {len(tukey_ext_ids):>3} regions: {tukey_ext_ids[:15]}"
        + (f" ... +{len(tukey_ext_ids) - 15} more" if len(tukey_ext_ids) > 15 else "")
    )
    print(
        f"  Tukey mild only (1.5-3*IQR): {len(tukey_mild_only):>3} regions: {tukey_mild_only[:15]}"
        + (f" ... +{len(tukey_mild_only) - 15} more" if len(tukey_mild_only) > 15 else "")
    )
    print(
        f"  GMM outliers (posterior>0.5): {len(gmm_ids):>3} regions: {gmm_ids[:15]}"
        + (f" ... +{len(gmm_ids) - 15} more" if len(gmm_ids) > 15 else "")
    )
    # Intersection = highest-confidence outliers
    ext_set = set(tukey_ext_ids)
    gmm_set = set(gmm_ids)
    both = sorted(ext_set & gmm_set)
    print(f"  Both Tukey-extreme AND GMM: {len(both):>3} regions: {both[:15]}")


if __name__ == "__main__":
    main()
