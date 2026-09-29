#!/usr/bin/env python
"""Plot Gilroy-style 3D mAP and detection outcomes by occlusion level."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_result(value):
    if "=" not in value:
        raise ValueError("--result expects DETECTOR=CSV")
    name, path = value.split("=", 1)
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = [row for row in csv.DictReader(input_file) if row["occlusion_bin"] != "all"]
    return name, rows


def plot(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{value}-{value + 9}" for value in range(0, 100, 10)]

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    positions = np.arange(len(labels)) * 1.25
    width = 0.72 / len(results)
    for index, (name, rows) in enumerate(results):
        offset = (index - (len(results) - 1) / 2) * width
        ax.bar(positions + offset, [100 * float(row["map_3d_50_95"]) for row in rows], width, label=name)
    ax.set(
        xlabel="Objective 3D occlusion (%)", ylabel="3D mAP@0.50:0.95 (%)",
        ylim=(0, None), title="Pedestrian detectability by 3D occlusion",
        xticks=positions, xticklabels=labels,
    )
    ax.grid(False)
    ax.legend(frameon=False)
    map_path = output_dir / "fig4_map_vs_occlusion.png"
    fig.savefig(map_path, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    fields = (("tp_percent", "TP (%)"), ("fp_per_gt", "FP per GT"), ("fn_percent", "FN (%)"))
    for ax, (field, label) in zip(axes, fields):
        for name, rows in results:
            ax.plot(labels, [float(row[field]) for row in rows], marker="o", label=name)
        ax.set(xlabel="Objective 3D occlusion (%)", ylabel=label)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Detection outcomes at 3D IoU ≥ 0.50")
    outcomes_path = output_dir / "fig6_tp_fp_fn_vs_occlusion.png"
    fig.savefig(outcomes_path, dpi=220)
    plt.close(fig)
    return map_path, outcomes_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="DETECTOR=metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gilroy_lidar_replication/plots"))
    args = parser.parse_args()
    paths = plot([load_result(value) for value in args.result], args.output_dir)
    for path in paths:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
