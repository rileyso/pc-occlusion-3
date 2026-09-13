#!/usr/bin/env python
"""Minimal classical baseline: pedestrian AP@IoU 0.3 and AP@IoU 0.5 per bin of
ground range and point count, for the four detectors, as regular bar plots.

AP is a set metric - it ranks predictions against a pool of objects - so it is
computed per equal-population bin (workspace rule: target 10 bins, merged
until every bin holds >= 55 objects) rather than per object or on a continuous
sliding window. Within a bin, confidence-ranked predictions matching objects
outside the bin are ignored (workspace subset protocol); predictions with
score < 0.1 are excluded.

Outputs:
  results/analysis/ap_bin_metrics.csv   long table: covariate x detector x bin x IoU threshold
  results/plots/fig_ap_iou30_bars.png   AP@0.3 bars (ground range | points in box)
  results/plots/fig_ap_iou50_bars.png   AP@0.5 bars (ground range | points in box)
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gilroy_lidar_replication.evaluate import (  # noqa: E402
    average_precision,
    load_predictions,
    prepare_frames,
    ranked_outcomes,
)
from evaluate_geometric import assign_equal_population_bins  # noqa: E402
from plot_geometric import DETECTORS, DETECTOR_LABELS  # noqa: E402

BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
COVARIATES = {"ground_range_m": "Ground range (m)", "num_points_in_gt": "Points in GT box"}
METRICS = {"ap_iou_30": 0.3, "ap_iou_50": 0.5}


def load_master(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def evaluate_bins(rows, detections_dir, score_threshold):
    """{covariate: {detector: {metric: [ap_pct per bin]}}} plus shared bin labels."""
    binned = {}
    labels = {}
    for covariate in COVARIATES:
        values = np.asarray([float(row[covariate]) for row in rows], dtype=float)
        edges, index = assign_equal_population_bins(values, target_bins=10, minimum_per_bin=55)
        for row, bin_index in zip(rows, index):
            row[f"bin_{covariate}"] = int(bin_index)
        labels[covariate] = [
            (float(edges[i]), float(edges[i + 1]), int((index == i).sum())) for i in range(len(edges) - 1)
        ]

    results = {covariate: {} for covariate in COVARIATES}
    for detector in DETECTORS:
        prepared = prepare_frames(
            [{"frame_id": row["frame_id"], "label_index": int(row["label_index"]),
              **{f"bin_{covariate}": row[f"bin_{covariate}"] for covariate in COVARIATES},
              "box": np.asarray([row[field] for field in BOX_FIELDS], dtype=np.float32)}
             for row in rows],
            load_predictions(detections_dir / f"{detector}_val.json", "Pedestrian"),
        )
        for covariate in COVARIATES:
            per_metric = {metric: [] for metric in METRICS}
            for bin_index in range(len(labels[covariate])):
                selected = lambda gt_rows, value=bin_index: [  # noqa: E731
                    i for i, row in enumerate(gt_rows) if row[f"bin_{covariate}"] == value
                ]
                for metric, threshold in METRICS.items():
                    ranked, count = ranked_outcomes(prepared, selected, threshold)
                    per_metric[metric].append(100.0 * average_precision(ranked, count))
            results[covariate][detector] = per_metric
            print(f"{detector:>10s} {covariate:>16s}: " +
                  "  ".join(f"{metric} [" + ", ".join(f"{ap:.1f}" for ap in aps) + "]"
                            for metric, aps in per_metric.items()))
    return results, labels


def format_value(value):
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_metric_panel(ax, results, labels, covariate, metric, ylabel):
    per_detector = results[covariate]
    bin_labels = [f"[{format_value(lo)}, {format_value(hi)})\nn={n}" for lo, hi, n in labels[covariate]]
    positions = np.arange(len(bin_labels)) * 1.25
    width = 0.8 / len(DETECTORS)
    for index, detector in enumerate(DETECTORS):
        aps = np.asarray(per_detector[detector][metric])
        offset = (index - (len(DETECTORS) - 1) / 2) * width
        bars = ax.bar(positions + offset, aps, width, color=f"C{index}", label=DETECTOR_LABELS[detector])
        ax.bar_label(bars, fmt="%.0f", fontsize=6, padding=2)
    ax.set(xticks=positions, xticklabels=bin_labels, ylabel=ylabel,
           ylim=(0, 112), xlabel=COVARIATES[covariate])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(alpha=0.25, axis="y")


def plot_metric(results, labels, metric, title_metric, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.14, 1, 0.86))
    for ax, covariate, title in ((axes[0], "ground_range_m", "Ground range"),
                                 (axes[1], "num_points_in_gt", "Points in GT box")):
        draw_metric_panel(ax, results, labels, covariate, metric, ylabel=f"Pedestrian {title_metric} (%)")
        ax.set_title(title, fontsize=11)
    handles, det_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, det_labels, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4,
               frameon=False, fontsize=9)
    fig.text(0.01, 0.012,
             f"Pedestrian {title_metric} per equal-population bin (target 10, merged to >= 55 objects; bin interval and "
             "count below each group). Confidence-ranked predictions with score >= 0.1; predictions matching objects "
             "outside the bin are ignored. KITTI validation split, n = 2,280 pedestrians per detector.",
             fontsize=7.5, va="bottom", ha="left", wrap=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"WROTE {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path,
                        default=Path("research_workspace/geometric_baselines/results/master_per_object.csv"))
    parser.add_argument("--detections-dir", type=Path,
                        default=Path("research_workspace/dcd_baseline/results/detections"))
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("research_workspace/geometric_baselines/results/analysis"))
    parser.add_argument("--plots-dir", type=Path,
                        default=Path("research_workspace/geometric_baselines/results/plots"))
    args = parser.parse_args()

    rows = load_master(args.master)
    results, labels = evaluate_bins(rows, args.detections_dir, args.score_threshold)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.analysis_dir / "ap_bin_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file,
                                fieldnames=("covariate", "detector", "bin_index", "bin_lo", "bin_hi", "n",
                                            "ap_iou_30_pct", "ap_iou_50_pct"))
        writer.writeheader()
        for covariate in COVARIATES:
            for detector in DETECTORS:
                for bin_index, (lo, hi, n) in enumerate(labels[covariate]):
                    writer.writerow({
                        "covariate": covariate, "detector": detector, "bin_index": bin_index,
                        "bin_lo": f"{lo:.6g}", "bin_hi": f"{hi:.6g}", "n": n,
                        "ap_iou_30_pct": f"{results[covariate][detector]['ap_iou_30'][bin_index]:.3f}",
                        "ap_iou_50_pct": f"{results[covariate][detector]['ap_iou_50'][bin_index]:.3f}",
                    })
    print(f"WROTE {csv_path}")

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(results, labels, "ap_iou_30", "AP@IoU 0.3", args.plots_dir / "fig_ap_iou30_bars.png")
    plot_metric(results, labels, "ap_iou_50", "AP@IoU 0.5", args.plots_dir / "fig_ap_iou50_bars.png")


if __name__ == "__main__":
    main()
