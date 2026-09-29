#!/usr/bin/env python
"""Grouped bars of 3D mAP@[0.50:0.95] per DCD bin from the corrected evaluation CSVs.

Replaces the retired fig_dcd_map_bars.png (invalid: built before the
bin-assignment bug fix in evaluate_dcd.py).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_dcd import DETECTOR_LABELS


def load_result(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return [row for row in csv.DictReader(input_file) if row["dcd_bin"] != "all"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", action="append", required=True, help="DETECTOR=dcd_bin_metrics.csv (repeatable)")
    parser.add_argument("--output", type=Path, default=Path("dcd_baseline/results/plots/fig_dcd_binned_map.png"))
    args = parser.parse_args()

    results = []
    for specifier in args.metrics:
        name, path = specifier.split("=", 1)
        results.append((DETECTOR_LABELS.get(name, name), load_result(path)))

    labels = [f"{row['dcd_range']}\nn={row['instances']}" for row in results[0][1]]
    positions = np.arange(len(labels)) * 1.3
    width = 0.78 / len(results)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.97))
    for index, (name, rows) in enumerate(results):
        offset = (index - (len(results) - 1) / 2) * width
        bars = ax.bar(positions + offset, [100 * float(row["map_3d_50_95"]) for row in rows], width, label=name)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    tallest = max(100 * float(row["map_3d_50_95"]) for _, rows in results for row in rows)
    ax.set(
        xlabel="Per-object DCD bin (equal population) - empirically, higher DCD = closer / denser returns (see caption)",
        ylabel="3D mAP@0.50:0.95 (%)\nmean AP over 3D IoU thresholds 0.50-0.95",
        ylim=(0, tallest * 1.18),
        title="Pedestrian 3D mAP per DCD bin - DCD baseline (KITTI val, corrected binning)",
        xticks=positions,
        xticklabels=labels,
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(title="Detector", frameon=False, loc="upper left")
    fig.text(0.01, 0.012,
             "Bins: equal-population groups of per-object DCD (density-aware Chamfer Distance, Wu et al. NeurIPS 2021; "
             "alpha=1.0, raw metres; observed returns vs ideal LiDAR-facing box surface), >= 55 objects per bin. "
             "Matching: confidence-ranked predictions at 3D IoU >= 0.50, score >= 0.1; other-object matches ignored. "
             "Note the monotonic RISE: under the official formulation DCD acts as a return-density proxy in this regime, "
             "so the sparsest (lowest-DCD) pedestrians are the hardest to detect. KITTI validation split, n = 2,279.",
             fontsize=7.5, va="bottom", ha="left", wrap=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
