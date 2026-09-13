#!/usr/bin/env python
"""Report figures for the KITTI difficulty baseline (no GPU needed).

Reads only the CSVs written by evaluate_kitti_difficulty.py and
kitti_official_ap.py:

1. fig_kitti_difficulty_bars.png - the headline comparison the baseline exists
   for: x = the KITTI difficulty labels (Easy / Moderate / Hard, plus the
   UnKnown band the official benchmark discards), grouped bars = the four
   detectors, y = 3D AP@0.50 (left) and detection rate at IoU >= 0.50,
   score >= 0.1 (right). Exclusive (disjoint) group membership.
2. fig_kitti_occlusion_bars.png - the same layout over the raw KITTI
   occlusion annotation levels (0 fully visible / 1 partly / 2 largely / 3
   unknown), i.e. the 'KITTI occlusion scores' proper, without the height and
   truncation criteria folded in.
3. fig_kitti_official_ap_bars.png - the official arm: AP_R40 under the
   benchmark's own protocol (cumulative ignore semantics), x = Easy /
   Moderate / Hard, panels = Car and Pedestrian.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from dcd_baseline.plot_dcd import DETECTOR_LABELS  # noqa: E402
from kitti_difficulty_baseline import groupings as grouping_registry  # noqa: E402

DETECTOR_ORDER = ("pointpillar", "second", "pointrcnn", "pv_rcnn")  # canonical thesis order
DETECTOR_COLORS = {"pointpillar": "#0072B2", "second": "#D55E00",
                   "pointrcnn": "#009E73", "pv_rcnn": "#CC79A7"}
EXCLUDED_GROUP_NOTE = "* group the official benchmark ignores at every difficulty"
BELOW_MINIMUM_NOTE = "\u2020 n below the Gilroy 55-instance per-group estimation minimum"


def canonical_order(detectors):
    known = [detector for detector in DETECTOR_ORDER if detector in detectors]
    return known + [detector for detector in detectors if detector not in known]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def grouped_bars(ax, groups, values, counts, ylabel, exclude_note=False):
    """groups: [(key, display)] left to right; values: {detector: {key: height}};
    counts: {key: n}. One bar per detector inside each group."""
    detectors = canonical_order(values.keys())
    width = 0.8 / len(detectors)
    positions = np.arange(len(groups))
    for offset, detector in enumerate(detectors):
        heights = [values[detector][key] for key, _ in groups]
        bars = ax.bar(positions + (offset - (len(detectors) - 1) / 2.0) * width, heights,
                      width * 0.92, color=DETECTOR_COLORS.get(detector, "#7f7f7f"),
                      label=DETECTOR_LABELS.get(detector, detector), zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{display}\nn={counts[key]}" for key, display in groups], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_ylim(0, 100)
    legend_note = []
    if exclude_note and any(display.endswith("*") for _, display in groups):
        legend_note.append(EXCLUDED_GROUP_NOTE)
    if any(display.endswith("\u2020") for _, display in groups):
        legend_note.append(BELOW_MINIMUM_NOTE)
    for order, note in enumerate(legend_note):
        ax.text(0.0, -0.20 - 0.06 * order, note, transform=ax.transAxes, fontsize=8, color="#555555")


def figure_group_bars(metrics_rows, grouping_name, output_path, title):
    """Two-panel figure (AP@0.50 | detection rate) for one grouping, exclusive view."""
    rows = [row for row in metrics_rows
            if row["grouping"] == grouping_name and row["mode"] == "exclusive"]
    # preserve the CSV's group order (Easy..UnKnown), de-duplicated
    seen, ordered_groups = set(), []
    for row in rows:
        if row["group"] not in seen:
            seen.add(row["group"])
            ordered_groups.append((row["group"], row["group_label"]))
    detectors = canonical_order({row["detector"] for row in rows})
    ignored = set(grouping_registry.get_grouping(grouping_name).ignored)
    counts = {key: int(next(row["instances"] for row in rows if row["group"] == key))
              for key, _ in ordered_groups}
    ordered_groups = [(key, display + ("*" if key in ignored else "")
                       + ("" if counts[key] >= 55 else "\u2020"))
                      for key, display in ordered_groups]
    values = {detector: {} for detector in detectors}
    for detector in detectors:
        for key, _ in ordered_groups:
            row = next(entry for entry in rows if entry["detector"] == detector and entry["group"] == key)
            values[detector][key] = {"ap": 100.0 * float(row["ap_3d_50"]),
                                     "rate": float(row["tp_percent"])}

    figure, (ap_axis, rate_axis) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    grouped_bars(ap_axis, ordered_groups, {d: {k: values[d][k]["ap"] for k, _ in ordered_groups}
                                           for d in detectors}, counts,
                 ylabel="3D AP@0.50 (%)", exclude_note=True)
    grouped_bars(rate_axis, ordered_groups, {d: {k: values[d][k]["rate"] for k, _ in ordered_groups}
                                             for d in detectors}, counts,
                 ylabel="Detection rate, IoU>=0.50 & score>=0.1 (%)", exclude_note=True)
    ap_axis.legend(loc="upper right", fontsize=9, framealpha=0.9)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0.02, 1, 0.96))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"WROTE {output_path}")


def figure_official_ap(official_rows, output_path):
    classes = ("Car", "Pedestrian")
    detectors = canonical_order({row["detector"] for row in official_rows})
    difficulties = ("easy", "moderate", "hard")
    figure, axes = plt.subplots(1, len(classes), figsize=(11.5, 4.4), sharey=True)
    for axis, class_name in zip(np.atleast_1d(axes), classes):
        width = 0.8 / len(detectors)
        positions = np.arange(len(difficulties))
        for offset, detector in enumerate(detectors):
            heights = [float(next(row["ap_r40"] for row in official_rows
                                  if row["detector"] == detector and row["class"] == class_name
                                  and row["difficulty"] == difficulty))
                       for difficulty in difficulties]
            axis.bar(positions + (offset - (len(detectors) - 1) / 2.0) * width, heights,
                     width * 0.92, color=DETECTOR_COLORS.get(detector, "#7f7f7f"),
                     label=DETECTOR_LABELS.get(detector, detector), zorder=3)
        axis.set_xticks(positions)
        axis.set_xticklabels([d.capitalize() for d in difficulties])
        axis.set_title(class_name)
        axis.grid(axis="y", alpha=0.25, zorder=0)
        axis.set_ylim(0, 100)
    axes[0].set_ylabel("Official KITTI 3D AP_R40 (%)")
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.9)
    figure.suptitle("Official KITTI protocol: AP_R40 by difficulty "
                    "(cumulative ignore semantics, nested GT sets)", fontsize=11)
    figure.tight_layout(rect=(0, 0.0, 1, 0.96))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"WROTE {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("kitti_difficulty_baseline/results"))
    parser.add_argument("--skip-official", action="store_true",
                        help="Only draw the thesis-protocol figures (official CSV absent).")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = args.results_dir / "evaluation" / "kitti_difficulty_metrics.csv"
    metrics_rows = read_csv(metrics_path)
    figure_group_bars(metrics_rows, "kitti_difficulty",
                      args.results_dir / "plots" / "fig_kitti_difficulty_bars.png",
                      "KITTI difficulty labels (exclusive bands), natural pedestrian benchmark, 4 detectors")
    figure_group_bars(metrics_rows, "kitti_occlusion",
                      args.results_dir / "plots" / "fig_kitti_occlusion_bars.png",
                      "KITTI occlusion annotation levels (exclusive bands), natural pedestrian benchmark, 4 detectors")
    if not args.skip_official:
        official_rows = read_csv(args.results_dir / "evaluation" / "kitti_official_ap.csv")
        figure_official_ap(official_rows,
                           args.results_dir / "plots" / "fig_kitti_official_ap_bars.png")


if __name__ == "__main__":
    main()
