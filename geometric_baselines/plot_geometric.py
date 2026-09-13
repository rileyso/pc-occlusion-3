#!/usr/bin/env python
"""Figures for the classical geometric baselines (from evaluate_geometric.py outputs).

Analysis figures (220 dpi, self-documenting captions, workspace curve style):
  fig_range_detection.png        P(detected) vs ground range
  fig_point_count_detection.png  P(detected) vs LiDAR points in the GT box
  fig_density_detection.png      P(detected) vs volumetric and facing-surface density
  fig_kitti_coarse_labels.png    P(detected) per KITTI occlusion state / difficulty level

Report figure (300 dpi, minimal in-figure text):
  fig_geometric_baselines_summary.png  2x2: range, point count, facing-surface
                                       density, KITTI occlusion state
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DETECTORS = ("pointpillar", "second", "pointrcnn", "pv_rcnn")
DETECTOR_LABELS = {"pointpillar": "PointPillars", "second": "SECOND", "pointrcnn": "PointRCNN", "pv_rcnn": "PV-RCNN"}

METHOD_CAPTION = (
    "Method - per ground-truth pedestrian: detected = a confidence-ranked prediction covers the GT box at 3D IoU >= 0.50 "
    "with score >= 0.1; predictions matching other annotated pedestrians are ignored. "
    "n = 2,280 KITTI validation-split pedestrians per detector. Curves: cubic-logistic fits (log transform for counts and "
    "densities) with 95% bootstrap bands (200 resamples); dots: equal-population bins of >= 55 objects placed at bin means. "
)

DENSITY_NOTE = (
    "Reading note - the decline at the extreme high-density end is a proximity/cropping effect: the top density decile is "
    "dominated by very close pedestrians (mean ground range 6.5 m, vs 8.6-37 m in lower deciles) that are heavily "
    "image-truncated (27% with truncation > 0.3, vs <= 5% below), so detectors under-predict the cropped extent and fail "
    "the 3D IoU >= 0.50 match."
)

COVARIATE_AXES = {
    "ground_range_m": {"transform": lambda v: np.asarray(v, dtype=float),
                       "ticks": [5, 10, 15, 20, 25, 30, 40, 50, 60, 80],
                       "xlabel": "Ground range (m)"},
    "num_points_in_gt": {"transform": np.log1p,
                         "ticks": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
                         "xlabel": "LiDAR points in GT box (log scale)"},
    "point_density_m3": {"transform": np.log1p,
                         "ticks": [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
                         "xlabel": "Volumetric return density, points / m$^3$ (log scale)"},
    "point_density_m2_facing": {"transform": np.log1p,
                                "ticks": [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
                                "xlabel": "Facing-surface return density, points / m$^2$ (log scale)"},
}


def load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_curves(path, covariate):
    curves = defaultdict(lambda: ([], [], [], []))
    for row in load_csv(path):
        if row["covariate"] != covariate:
            continue
        xs, rates, lows, highs = curves[row["detector"]]
        xs.append(float(row["x"]))
        rates.append(float(row["rate_pct"]))
        lows.append(float(row["ci_low_pct"]))
        highs.append(float(row["ci_high_pct"]))
    result = {}
    for detector, (xs, rates, lows, highs) in curves.items():
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        result[detector] = (np.asarray(xs)[order], np.asarray(rates)[order],
                            np.asarray(lows)[order], np.asarray(highs)[order])
    return result


def load_bins(path, covariate):
    rows = [row for row in load_csv(path) if row["covariate"] == covariate and row["transform"] != "categorical"]
    bins = defaultdict(list)
    for row in rows:
        bins[row["detector"]].append(row)
    return bins


def overall_rates(path):
    rates = {}
    for row in load_csv(path):
        if row.get("overall_rate_pct"):
            rates[row["detector"]] = float(row["overall_rate_pct"])
    return rates


def covariate_ticks(covariate, curves):
    """Nice ticks in original units, filtered to the fitted range."""
    axis = COVARIATE_AXES[covariate]
    x_min = min(curve[0].min() for curve in curves.values())
    x_max = max(curve[0].max() for curve in curves.values())
    ticks = [tick for tick in axis["ticks"] if x_min <= tick <= x_max]
    if len(ticks) < 2:
        ticks = None
    return axis, ticks


def plot_covariate_curves(ax, curves, covariate, bins, rates_overall):
    axis, ticks = covariate_ticks(covariate, curves)
    transform = axis["transform"]
    for index, detector in enumerate(DETECTORS):
        if detector not in curves:
            continue
        xs, rates, lows, highs = curves[detector]
        color = f"C{index}"
        label = DETECTOR_LABELS[detector]
        if detector in rates_overall:
            label = f"{label} ({rates_overall[detector]:.1f}% overall)"
        ax.plot(transform(xs), rates, "-", color=color, label=label)
        ax.fill_between(transform(xs), lows, highs, color=color, alpha=0.12)
        if detector in bins:
            bin_x = transform(np.asarray([float(row["bin_mean"]) for row in bins[detector]]))
            bin_y = np.asarray([float(row["rate_pct"]) for row in bins[detector]])
            ax.plot(bin_x, bin_y, "o", color=color, markerfacecolor="none", markersize=4.5, alpha=0.85, linewidth=0)
    if ticks is not None:
        ax.set_xticks(transform(np.asarray(ticks, dtype=float)))
        ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set(xlabel=axis["xlabel"], ylabel="Pedestrians detected (%)", ylim=(0, 100))
    ax.grid(alpha=0.25)


def detector_bars(ax, rows, levels, level_labels, rates_overall):
    positions = np.arange(len(levels)) * 1.35
    width = 0.78 / len(DETECTORS)
    for index, detector in enumerate(DETECTORS):
        detector_rows = {row["bin_lo"]: row for row in rows if row["detector"] == detector}
        rates = np.asarray([float(detector_rows[level]["rate_pct"]) if level in detector_rows else np.nan
                            for level in levels])
        lows = np.asarray([float(detector_rows[level]["wilson_lo_pct"]) if level in detector_rows else np.nan
                           for level in levels])
        highs = np.asarray([float(detector_rows[level]["wilson_hi_pct"]) if level in detector_rows else np.nan
                            for level in levels])
        offset = (index - (len(DETECTORS) - 1) / 2) * width
        label = DETECTOR_LABELS[detector]
        if detector in rates_overall:
            label = f"{label} ({rates_overall[detector]:.1f}% overall)"
        error = np.stack([rates - lows, highs - rates], axis=0)
        bars = ax.bar(positions + offset, rates, width, label=label, yerr=error,
                      error_kw=dict(elinewidth=1.0, capsize=2.0), color=f"C{index}")
        ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)
    ax.set(xticks=positions, xticklabels=level_labels, ylabel="Pedestrians detected (%)", ylim=(0, 108))
    ax.grid(alpha=0.25, axis="y")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("geometric_baselines/results/analysis"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("geometric_baselines/results/plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = args.analysis_dir / "curve_fits.csv"
    bins_path = args.analysis_dir / "binned_detection_rate.csv"
    summary_path = args.analysis_dir / "covariate_summary.csv"
    rates_overall = overall_rates(summary_path)

    written = []

    # ---- single-covariate analysis figures
    for covariate, filename in (
        ("ground_range_m", "fig_range_detection.png"),
        ("num_points_in_gt", "fig_point_count_detection.png"),
    ):
        curves = load_curves(curves_path, covariate)
        bins = load_bins(bins_path, covariate)
        fig, ax = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
        if fig.get_layout_engine() is not None:
            fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
        plot_covariate_curves(ax, curves, covariate, bins, rates_overall)
        subject = {"ground_range_m": "ground range", "num_points_in_gt": "LiDAR point count"}[covariate]
        ax.set_title(f"Pedestrian detection probability vs {subject} - classical geometric baseline (KITTI val)")
        ax.legend(title="Detector", frameon=False, loc="lower left", fontsize=9)
        fig.text(0.01, 0.012, METHOD_CAPTION, fontsize=7.5, va="bottom", ha="left", wrap=True)
        path = args.output_dir / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path)

    # ---- density figure (two density definitions side by side)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    for ax, covariate, title in (
        (axes[0], "point_density_m3", "Volumetric: points / box volume"),
        (axes[1], "point_density_m2_facing", "Facing-surface: points / m$^2$ of LiDAR-facing surface"),
    ):
        plot_covariate_curves(ax, load_curves(curves_path, covariate), covariate, load_bins(bins_path, covariate),
                              rates_overall)
        ax.set_title(title, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, frameon=False, fontsize=9)
    fig.text(0.01, 0.012, METHOD_CAPTION + DENSITY_NOTE, fontsize=7.5, va="bottom", ha="left", wrap=True)
    path = args.output_dir / "fig_density_detection.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    # ---- coarse KITTI label figure
    binned_rows = load_csv(bins_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    detector_bars(axes[0], [row for row in binned_rows if row["covariate"] == "kitti_occlusion"],
                  ["0", "1", "2"], ["0\nfully visible", "1\npartly\noccluded", "2\nlargely\noccluded"],
                  rates_overall)
    detector_bars(axes[1], [row for row in binned_rows if row["covariate"] == "kitti_difficulty"],
                  ["Easy", "Moderate", "Hard"], ["Easy", "Moderate", "Hard"], rates_overall)
    axes[0].set_title("By KITTI occlusion annotation", fontsize=11)
    axes[1].set_title("By KITTI difficulty annotation", fontsize=11)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, frameon=False, fontsize=9)
    fig.text(0.01, 0.012,
             METHOD_CAPTION
             + "KITTI occlusion states: 0 = fully visible, 1 = partly occluded, 2 = largely occluded; difficulty combines "
             "occlusion, truncation and image height.",
             fontsize=7.5, va="bottom", ha="left", wrap=True)
    path = args.output_dir / "fig_kitti_coarse_labels.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    # ---- 2x2 report summary (300 dpi, minimal in-figure text)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.90))
    panels = (
        (axes[0][0], "ground_range_m", "(a) Ground range"),
        (axes[0][1], "num_points_in_gt", "(b) Points in GT box"),
        (axes[1][0], "point_density_m2_facing", "(c) Facing-surface density (points/m$^2$)"),
    )
    handles = None
    panel_labels = []
    for ax, covariate, title in panels:
        plot_covariate_curves(ax, load_curves(curves_path, covariate), covariate, load_bins(bins_path, covariate),
                              rates_overall)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        if handles is None:
            handles, panel_labels = ax.get_legend_handles_labels()
    detector_bars(axes[1][1], [row for row in binned_rows if row["covariate"] == "kitti_occlusion"],
                  ["0", "1", "2"], ["0", "1", "2"], rates_overall)
    axes[1][1].set_title("(d) KITTI occlusion state", fontsize=11)
    axes[1][1].set_xlabel("")
    axes[1][0].set_xlabel(COVARIATE_AXES["point_density_m2_facing"]["xlabel"], fontsize=9)
    axes[1][1].set_xlabel("KITTI occlusion annotation", fontsize=9)
    fig.legend(handles, panel_labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Classical geometric baselines: pedestrian detectability vs range, point count, density and coarse labels "
                 "(KITTI val)", fontsize=12, y=0.995)
    fig.text(0.01, 0.008,
             "KITTI validation split, n = 2,280 pedestrians per detector. Matching: 3D IoU >= 0.50, score >= 0.1, "
             "confidence-ranked, predictions matching other annotated pedestrians ignored. "
             "Curves: cubic-logistic fits, 95% bootstrap bands; dots: equal-population bins (>= 55); bars: Wilson 95% CIs. "
             "In (c) the extreme high-density end declines: the top decile is very close, image-truncated pedestrians.",
             fontsize=7, va="bottom", ha="left", wrap=True)
    path = args.output_dir / "fig_geometric_baselines_summary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    written.append(path)

    for path in written:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
