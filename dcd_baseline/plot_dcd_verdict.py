#!/usr/bin/env python
"""One-figure verdict on DCD as an occlusion baseline, from the per-object CSVs only.

Panel 1: detection rate vs DCD decile - an occlusion metric must slope DOWN
         (higher DCD = more hidden = harder); it slopes UP instead.
Panel 2: mean DCD vs LiDAR point-count decile - DCD mostly tracks point count
         (a density proxy), which is why the direction inverts.

CPU-only: reads results/analysis/per_object_outcomes_*.csv, needs no detectors.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_outcomes(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return (
        np.asarray([float(row["dcd"]) for row in rows]),
        np.asarray([int(row["detected"]) for row in rows]),
        np.asarray([float(row["num_points_in_gt"]) for row in rows]),
    )


def decile_profile(values, outcome, n_bins=10):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    centres, rates = [], []
    for b in range(n_bins):
        mask = index == b
        centres.append(float(values[mask].mean()))
        rates.append(100.0 * float(outcome[mask].mean()))
    return np.asarray(centres), np.asarray(rates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=Path("dcd_baseline/results/analysis/per_object_outcomes_pointpillar.csv"))
    parser.add_argument("--detector", default="PointPillars")
    parser.add_argument("--output", type=Path, default=Path("dcd_baseline/results/plots/fig_dcd_baseline_verdict.png"))
    args = parser.parse_args()

    dcd, detected, points = read_outcomes(args.outcomes)
    dcd_centres, det_rates = decile_profile(dcd, detected)
    pts_centres, dcd_by_pts = decile_profile(points, dcd)
    r_points = float(np.corrcoef(dcd, points)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.87))

    ax = axes[0]
    ax.plot(dcd_centres, det_rates, "o-", color="tab:red", linewidth=2.5, markersize=7)
    ax.annotate("an occlusion metric MUST fall here\n(more hidden = harder to detect)",
                xy=(dcd_centres[-1], det_rates[-1]), xytext=(0.62, 70),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10, bbox=dict(boxstyle="round", fc="mistyrose", ec="tab:red"))
    ax.set(xlabel="DCD (its intended meaning: higher = more occluded)",
           ylabel=f"Pedestrians detected (%)\n{args.detector}, 3D IoU >= 0.50, score >= 0.1",
           title="1. Detection goes UP as DCD rises\n-> the direction is wrong for occlusion", ylim=(0, 100))
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pts_centres, dcd_by_pts, "s-", color="tab:purple", linewidth=2.5, markersize=7)
    ax.annotate(f"DCD climbs almost in step with point count\nr = {r_points:+.2f} (with range: r = -0.71)",
                xy=(pts_centres[-2], dcd_by_pts[-2]), xytext=(150, 65.5),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=10,
                bbox=dict(boxstyle="round", fc="lavender", ec="tab:purple"))
    ax.set(xlabel="LiDAR points inside the box (density)",
           ylabel="Mean DCD",
           title="2. DCD mostly measures point count\n-> it is a density proxy, not an occlusion metric")
    ax.grid(alpha=0.3)

    fig.text(0.01, 0.012,
             "Why DCD fails as an occlusion baseline - KITTI val pedestrians, PointPillars. "
             "Per-object data: dcd_baseline/results/analysis/per_object_outcomes_pointpillar.csv (n=2,279). "
             "DCD = density-aware Chamfer Distance (Wu et al. NeurIPS 2021; alpha=1.0, raw metres) between the returns in the GT box "
             "and the ideal LiDAR-facing box surface. Both panels use 10 equal-population groups. "
             "A usable occlusion baseline must at minimum point the right way; DCD points the opposite way because its density "
             "weighting rewards the dense returns that close, unoccluded objects receive.",
             fontsize=7.5, va="bottom", ha="left", wrap=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
