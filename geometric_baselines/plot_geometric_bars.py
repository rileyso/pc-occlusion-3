#!/usr/bin/env python
"""Bar-chart versions of the classical geometric baselines (regular linear scale).

Companion to plot_geometric.py: instead of fitted curves, plots P(detected)
per equal-population bin (>= 55 objects) as grouped bars for the four
detectors, with Wilson 95% error bars. Bin positions are equally spaced
categories on a plain linear scale, labelled with each bin's covariate
interval and count - no log transforms anywhere.

  fig_range_detection_bars.png        bars vs ground-range bin
  fig_point_count_detection_bars.png  bars vs points-in-box bin
  fig_density_detection_bars.png      bars vs volumetric / facing-surface density bins

Note: with equal-spaced categories, mid-bin differences are visually
over-weighted; the curve figures remain the analysis view and these bars the
direct, untransformed illustration.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_geometric import DETECTORS, DETECTOR_LABELS, METHOD_CAPTION, DENSITY_NOTE


def load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_bins(path, covariate):
    """{detector: [bin rows sorted by bin_index]} for one continuous covariate."""
    bins = defaultdict(list)
    for row in load_csv(path):
        if row["covariate"] == covariate and row["transform"] != "categorical":
            bins[row["detector"]].append(row)
    for detector in bins:
        bins[detector].sort(key=lambda row: int(row["bin_index"]))
    counts = {len(rows) for rows in bins.values()}
    if len(counts) != 1:
        raise ValueError(f"detectors see different bin sets for {covariate}: {counts}")
    return bins


def format_value(value):
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def grouped_bars(ax, bins, covariate):
    first = next(iter(bins.values()))
    labels = [f"[{format_value(float(row['bin_lo']))}, {format_value(float(row['bin_hi']))})\nn={row['n']}"
              for row in first]
    positions = np.arange(len(labels)) * 1.25
    width = 0.8 / len(DETECTORS)
    for index, detector in enumerate(DETECTORS):
        rows = bins[detector]
        rates = np.asarray([float(row["rate_pct"]) for row in rows])
        lows = np.asarray([float(row["wilson_lo_pct"]) for row in rows])
        highs = np.asarray([float(row["wilson_hi_pct"]) for row in rows])
        offset = (index - (len(DETECTORS) - 1) / 2) * width
        error = np.stack([rates - lows, highs - rates], axis=0)
        bars = ax.bar(positions + offset, rates, width, color=f"C{index}",
                      label=f"{DETECTOR_LABELS[detector]}", yerr=error,
                      error_kw=dict(elinewidth=1.0, capsize=2.0))
        ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=2)
    ax.set(xticks=positions, xticklabels=labels, ylabel="Pedestrians detected (%)", ylim=(0, 112))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(alpha=0.25, axis="y")
    return labels


BAR_SUBJECTS = {
    "ground_range_m": ("Ground-range bin (m)", "ground range"),
    "num_points_in_gt": ("LiDAR points in GT box bin", "LiDAR point count"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("geometric_baselines/results/analysis"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("geometric_baselines/results/plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bins_path = args.analysis_dir / "binned_detection_rate.csv"
    written = []

    # ---- single-covariate bar figures (distance, point count)
    for covariate, filename in (
        ("ground_range_m", "fig_range_detection_bars.png"),
        ("num_points_in_gt", "fig_point_count_detection_bars.png"),
    ):
        xlabel, subject = BAR_SUBJECTS[covariate]
        bins = load_bins(bins_path, covariate)
        fig, ax = plt.subplots(figsize=(12.8, 6.4), constrained_layout=True)
        if fig.get_layout_engine() is not None:
            fig.get_layout_engine().set(rect=(0, 0.10, 1, 0.90))
        grouped_bars(ax, bins, covariate)
        ax.set_xlabel(xlabel)
        ax.set_title(f"Pedestrian detection rate vs {subject} - classical geometric baseline (KITTI val)")
        ax.legend(title="Detector", frameon=False, loc="upper right", fontsize=9)
        fig.text(0.01, 0.012, METHOD_CAPTION + "Bars: per-bin detection rate (bin intervals labelled below), "
                 "Wilson 95% error bars. Bins are equally spaced categories; see the curve figure for the "
                 "analysis view.", fontsize=7.5, va="bottom", ha="left", wrap=True)
        path = args.output_dir / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path)

    # ---- density bar figure (two density definitions side by side)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    for ax, covariate, title in (
        (axes[0], "point_density_m3", "Volumetric: points / box volume"),
        (axes[1], "point_density_m2_facing", "Facing-surface: points / m$^2$ of LiDAR-facing surface"),
    ):
        bins = load_bins(bins_path, covariate)
        grouped_bars(ax, bins, covariate)
        ax.set_title(title, fontsize=11)
    axes[0].set_xlabel("Volumetric density bin (points / m$^3$)")
    axes[1].set_xlabel("Facing-surface density bin (points / m$^2$)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, frameon=False, fontsize=9)
    fig.text(0.01, 0.012,
             METHOD_CAPTION + "Bars: per-bin detection rate (bin intervals labelled below), Wilson 95% error bars. "
             "Bins are equally spaced categories on a regular scale. " + DENSITY_NOTE,
             fontsize=7.5, va="bottom", ha="left", wrap=True)
    path = args.output_dir / "fig_density_detection_bars.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    written.append(path)

    for path in written:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
