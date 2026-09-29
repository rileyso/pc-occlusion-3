#!/usr/bin/env python
"""Compare structured occlusion with equal-count random point removal."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return [row for row in csv.DictReader(input_file) if row["occlusion_bin"] != "all"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, help="NAME=STRUCTURED_CSV=RANDOM_CSV")
    parser.add_argument("--output", type=Path, default=Path("runs/gilroy_lidar_replication/controlled/plots/density_control_map.png"))
    args = parser.parse_args()
    pairs = []
    for value in args.pair:
        name, structured, random = value.split("=", 2)
        pairs.append((name, read(structured), read(random)))
    columns, rows = 2, (len(pairs) + 1) // 2
    fig, axes = plt.subplots(rows, columns, figsize=(11, 4.2 * rows), squeeze=False, constrained_layout=True)
    labels = [f"{value}-{value + 9}" for value in range(0, 100, 10)]
    for ax, (name, structured, random) in zip(axes.flat, pairs):
        ax.plot(labels, [100 * float(row["map_3d_50_95"]) for row in structured], marker="o", label="contiguous occlusion")
        ax.plot(labels, [100 * float(row["map_3d_50_95"]) for row in random], marker="o", linestyle="--", label="equal-count random removal")
        ax.set(title=name, xlabel="Target occlusion level (%)", ylabel="3D mAP@0.50:0.95 (%)", ylim=(0, None))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes.flat[len(pairs):]:
        ax.remove()
    fig.suptitle("Geometry control: spatially contiguous loss versus the same point count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
