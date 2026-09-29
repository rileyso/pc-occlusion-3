#!/usr/bin/env python
"""Figures for the Waymo point-count difficulty baseline (KITTI pedestrians, 4 detectors).

Reads only the CSVs written by evaluate_waymo.py:

1. fig_waymo_level_performance.png - the report comparison: 3D AP@0.50 and
   detection rate for each detector under (left) the literal Waymo rule
   (<= 5 points) and (right) the density-matched rule (<= 26 points, ~P10),
   each split into All / LEVEL_1 / LEVEL_2 (exclusive).
2. fig_waymo_rule_effectiveness.png - the verdict figure: (a) P(detected) vs
   point count with both thresholds marked; (b) LEVEL_1-LEVEL_2 gap across the
   threshold sweep vs the LEVEL_2 population share; (c) the detection-rate
   gradient inside literal LEVEL_1 that the one-bit rule hides; (d) the range
   confound (range distributions of the two levels, density-matched rule).
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

from dcd_baseline.plot_dcd import DETECTOR_LABELS, fit_logistic  # noqa: E402

LEVEL_COLORS = {"all": "#7f7f7f", "LEVEL_1": "#4477AA", "LEVEL_2": "#EE6677"}
LEVEL_LABELS = {"all": "All\n(cumulative L2)", "LEVEL_1": "LEVEL 1", "LEVEL_2": "LEVEL 2"}
DETECTOR_ORDER = ("pointpillar", "second", "pointrcnn", "pv_rcnn")  # canonical thesis order


def canonical_order(detectors):
    known = [detector for detector in DETECTOR_ORDER if detector in detectors]
    return known + [detector for detector in detectors if detector not in known]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def level_bars(ax, rows, value_field, detectors, ylabel):
    """Grouped bars per detector for the All/LEVEL_1/LEVEL_2 groups of one rule."""
    width = 0.26
    positions = np.arange(len(detectors))
    for offset, (group, color) in enumerate(zip(("all", "LEVEL_1", "LEVEL_2"),
                                                (LEVEL_COLORS["all"], LEVEL_COLORS["LEVEL_1"], LEVEL_COLORS["LEVEL_2"]))):
        heights, counts = [], []
        for detector in detectors:
            row = next(entry for entry in rows if entry["detector"] == detector and entry["group_key"] == group)
            heights.append(100.0 * float(row[value_field]) if value_field.startswith("ap") else float(row[value_field]))
            counts.append(int(row["instances"]))
        bars = ax.bar(positions + (offset - 1) * width, heights, width * 0.92, color=color,
                      label=LEVEL_LABELS[group], zorder=3)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5, str(count),
                    ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(positions)
    ax.set_xticklabels([DETECTOR_LABELS.get(detector, detector) for detector in detectors])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_ylim(0, 100)


def figure_level_performance(results_dir, output_dir, detectors, density_matched_threshold):
    literal = read_csv(results_dir / "evaluation" / "waymo_level_metrics.csv")
    sweep = read_csv(results_dir / "evaluation" / "waymo_threshold_sweep.csv")
    for row in literal:
        row["group_key"] = {"LEVEL_2_cumulative": "all"}.get(row["group"], row["group"].replace("_exclusive", ""))
    # "All (cumulative L2)" is threshold-independent: reuse the literal-rule cumulative rows.
    matched = [dict(row, group_key={"LEVEL_2_cumulative": "all"}.get(row["group"], row["group"]))
               for row in literal if row["group"] == "LEVEL_2_cumulative"]
    matched += [dict(row, group_key=row["group"].replace("_exclusive", "")) for row in sweep
                if int(row["threshold_points"]) == density_matched_threshold]
    literal_l2_n = next(int(row["instances"]) for row in literal if row["group"] == "LEVEL_2_exclusive")
    matched_l2_n = next(int(row["instances"]) for row in matched if row["group"] == "LEVEL_2_exclusive")
    matched_l2_share = 100.0 * matched_l2_n / sum(int(row["instances"]) for row in literal if row["group"] == "LEVEL_2_cumulative" and row["detector"] == detectors[0])

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.115, 1, 0.885))
    for column, (rows, title, subtitle) in enumerate((
        (literal, "Waymo rule transplanted literally (LEVEL 2 = ≤ 5 points)",
         f"LEVEL 2 holds {literal_l2_n} of 2,279 pedestrians ({100.0 * literal_l2_n / 2279:.1f}%) — rule degenerates on KITTI"),
        (matched, f"Density-matched rule (LEVEL 2 = ≤ {density_matched_threshold} points ≈ 10th percentile)",
         f"LEVEL 2 holds {matched_l2_n} pedestrians ({matched_l2_share:.1f}%) — sensor-appropriate operating point"),
    )):
        level_bars(axes[0, column], rows, "ap_3d_50", detectors, "3D AP @ IoU 0.50 (%)")
        level_bars(axes[1, column], rows, "tp_percent", detectors, "Pedestrians detected (%)\nIoU ≥ 0.50, score ≥ 0.1")
        axes[0, column].set_title(f"{title}\n{subtitle}", fontsize=10.5)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[0, 0].text(0.02, 0.93, "bars annotated with instance count n;\nleft LEVEL 2 n < 55 ⇒ unstable", transform=axes[0, 0].transAxes,
                    fontsize=7.5, va="top", color="#333333")
    caption = (
        "Method - the Waymo Open Dataset difficulty rule (Sun et al., CVPR 2020, §5.1) applied to the 2,280 KITTI validation pedestrians: labels with no LiDAR points ignored "
        "(1 object), LEVEL 2 = ≤ threshold points in the GT box, LEVEL 1 = the rest. Matching: greedy confidence-ranked, 3D IoU ≥ 0.50, score ≥ 0.1; other-object matches ignored. "
        "Left: the literal ≤ 5-point threshold, calibrated on Waymo's LiDAR, leaves 99.1% of KITTI pedestrians in LEVEL 1 (its LEVEL 2 estimates rest on n = 20). "
        "Right: the same one-bit rule at a density-matched threshold (≤ 26 points ≈ KITTI pedestrian P10). "
        "Reading: the heuristic separates difficulty where it bites (LEVEL 2 ≪ LEVEL 1 ≈ All), but the literal rule barely applies to this sensor - benchmark difficulty labels are sensor-relative."
    )
    fig.text(0.01, 0.012, caption, fontsize=7, va="bottom", ha="left", wrap=True)
    path = output_dir / "fig_waymo_level_performance.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def detection_probability_panel(ax, per_object, detectors, bootstrap, seed, thresholds):
    points_all = np.asarray(per_object["num_points_in_gt"], dtype=float)
    live = points_all > 0
    points = points_all[live]
    grid = np.logspace(np.log10(max(points.min(), 1)), np.log10(points.max()), 200)
    log_points = np.log1p(points)
    mean, sd = float(log_points.mean()), float(log_points.std())
    z = (log_points - mean) / sd

    def design(values_z):
        return np.stack([np.ones_like(values_z), values_z, values_z ** 2, values_z ** 3], axis=1)

    grid_z = (np.log1p(grid) - mean) / sd
    for index, detector in enumerate(detectors):
        detected = np.asarray(per_object[f"detected_{detector}"], dtype=float)[live]
        beta = fit_logistic(design(z), detected)
        point = 100.0 / (1.0 + np.exp(-(design(grid_z) @ beta)))
        sampler = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            sample = sampler.integers(0, len(points), len(points))
            boot_beta = fit_logistic(design(z[sample]), detected[sample])
            draws.append(100.0 / (1.0 + np.exp(-(design(grid_z) @ boot_beta))))
        low, high = np.percentile(draws, [2.5, 97.5], axis=0)
        color = f"C{index}"
        ax.fill_between(grid, low, high, color=color, alpha=0.12, zorder=2)
        ax.plot(grid, point, color=color, zorder=3, label=DETECTOR_LABELS.get(detector, detector))
    for threshold, style, label, label_y in (
        (thresholds["literal"], (0, (5, 4)), f"Waymo literal (≤ {thresholds['literal']} pts)", 98),
        (thresholds["density_matched"], (0, (1, 1)), f"Density-matched (≤ {thresholds['density_matched']} pts)", 86),
    ):
        ax.axvline(threshold, color="#222222", linestyle=style, linewidth=1.1, zorder=4)
        ax.text(threshold * 1.12, label_y, label, fontsize=7.5, ha="left", va="top", color="#222222")
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    share = 100.0 * (points <= thresholds["literal"]).mean()
    ax.annotate(f"{share:.1f}% of KITTI pedestrians\nlie right of the Waymo threshold",
                xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom", fontsize=8, color="#333333")
    ax.set_xscale("log")
    ax.set(xlabel="LiDAR points inside the GT box (log scale)",
           ylabel="Pedestrians detected (%)", ylim=(0, 100),
           title="(a) The continuous signal under the one-bit rule")
    ax.grid(alpha=0.25, which="both", zorder=0)


def sweep_panel(ax, sweep, detectors):
    thresholds = sorted({int(row["threshold_points"]) for row in sweep})
    shares = []
    for threshold in thresholds:
        level_2 = [int(row["instances"]) for row in sweep
                   if int(row["threshold_points"]) == threshold and row["group"] == "LEVEL_2_exclusive"]
        level_1 = [int(row["instances"]) for row in sweep
                   if int(row["threshold_points"]) == threshold and row["group"] == "LEVEL_1"]
        shares.append(100.0 * level_2[0] / (level_2[0] + level_1[0]))
    for index, detector in enumerate(detectors):
        gaps = []
        for threshold in thresholds:
            rows = [row for row in sweep if int(row["threshold_points"]) == threshold and row["detector"] == detector]
            level_2 = next(float(row["tp_percent"]) for row in rows if row["group"] == "LEVEL_2_exclusive")
            level_1 = next(float(row["tp_percent"]) for row in rows if row["group"] == "LEVEL_1")
            gaps.append(level_1 - level_2)
        ax.plot(thresholds, gaps, "o-", color=f"C{index}", label=DETECTOR_LABELS.get(detector, detector), zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(thresholds)
    ax.set_xticklabels([str(threshold) for threshold in thresholds])
    ax.axvline(5, color="#222222", linestyle=(0, (5, 4)), linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(thresholds, shares, "--", color="#7f7f7f", linewidth=1.3, zorder=2)
    ax2.set_ylabel("LEVEL 2 share of pedestrians (%)", color="#7f7f7f")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", colors="#7f7f7f")
    ax.set(xlabel="LEVEL 2 threshold (points in GT box)",
           ylabel="LEVEL 1 − LEVEL 2 detection-rate gap (pp)",
           title="(b) Stratification gap across thresholds\n(grey dashed, right axis: LEVEL 2 share)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both", zorder=0)


def tercile_panel(ax, summary, detectors):
    groups = [("LEVEL 2\n(literal ≤ 5 pts)", "literal_l2_rate_pct", LEVEL_COLORS["LEVEL_2"]),
              ("LEVEL 1 T1\n(sparsest third)", "literal_l1_tercile_1_rate_pct", "#B3C6E7"),
              ("LEVEL 1 T2", "literal_l1_tercile_2_rate_pct", "#6699CC"),
              ("LEVEL 1 T3\n(densest third)", "literal_l1_tercile_3_rate_pct", LEVEL_COLORS["LEVEL_1"])]
    width = 0.2
    positions = np.arange(len(detectors))
    for offset, (label, field, color) in enumerate(groups):
        values = [float(next(row for row in summary if row["detector"] == detector)[field]) for detector in detectors]
        ax.bar(positions + (offset - 1.5) * width, values, width * 0.9, color=color, label=label, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([DETECTOR_LABELS.get(detector, detector) for detector in detectors])
    ax.set_ylabel("Pedestrians detected (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(c) The gradient the one-bit rule hides:\nwithin literal LEVEL 1, detection still climbs with point count")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25, zorder=0)


def range_confound_panel(ax, per_object, density_matched_threshold):
    points = np.asarray(per_object["num_points_in_gt"], dtype=float)
    ranges = np.asarray(per_object["ground_range_m"], dtype=float)
    live = points > 0
    points, ranges = points[live], ranges[live]
    level_1, level_2 = points > density_matched_threshold, points <= density_matched_threshold
    for mask, color, label in ((level_1, LEVEL_COLORS["LEVEL_1"], "LEVEL 1 (density-matched rule)"),
                               (level_2, LEVEL_COLORS["LEVEL_2"], "LEVEL 2 (density-matched rule)")):
        ordered = np.sort(ranges[mask])
        ax.plot(ordered, np.linspace(0, 100, len(ordered)), color=color, linewidth=2.0, label=label)
    ax.axvline(np.median(ranges[level_1]), color=LEVEL_COLORS["LEVEL_1"], linestyle=":", linewidth=1.0)
    ax.axvline(np.median(ranges[level_2]), color=LEVEL_COLORS["LEVEL_2"], linestyle=":", linewidth=1.0)
    ax.text(np.median(ranges[level_1]) + 1, 6, f"median {np.median(ranges[level_1]):.0f} m", fontsize=8, color=LEVEL_COLORS["LEVEL_1"])
    ax.text(np.median(ranges[level_2]) + 1, 14, f"median {np.median(ranges[level_2]):.0f} m", fontsize=8, color=LEVEL_COLORS["LEVEL_2"])
    correlation = np.corrcoef(level_2.astype(float), ranges)[0, 1]
    ax.text(0.97, 0.55, f"r(LEVEL 2 membership, range) = {correlation:+.2f}\nr(points, range) = {np.corrcoef(points, ranges)[0, 1]:+.2f}",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#333333")
    ax.set(xlabel="Ground range (m)", ylabel="Cumulative share of pedestrians (%)", xlim=(0, 80),
           title="(d) The confound: LEVEL 2 is largely\na re-encoding of range")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, zorder=0)


def figure_effectiveness(results_dir, output_dir, detectors, bootstrap, seed, density_matched_threshold):
    rows = read_csv(results_dir / "analysis" / "per_object_waymo_levels.csv")
    per_object = {key: [row[key] for row in rows] for key in rows[0].keys()}
    sweep = read_csv(results_dir / "evaluation" / "waymo_threshold_sweep.csv")
    summary = read_csv(results_dir / "analysis" / "waymo_effectiveness_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.085, 1, 0.915))
    detection_probability_panel(axes[0, 0], per_object, detectors, bootstrap, seed,
                                {"literal": 5, "density_matched": density_matched_threshold})
    sweep_panel(axes[0, 1], sweep, detectors)
    tercile_panel(axes[1, 0], summary, detectors)
    range_confound_panel(axes[1, 1], per_object, density_matched_threshold)
    tercile_edges = summary[0]["literal_l1_tercile_edges"].replace("|", " and ")
    mean_spread = np.mean([float(row["literal_l1_tercile_spread_pp"]) for row in summary])
    level_range_r = float(summary[0]["r_level2_p10_range"])
    caption = (
        "Method - per-object detection status for four OpenPCDet detectors on 2,279 evaluable KITTI validation pedestrians; "
        "greedy confidence-ranked matching at 3D IoU ≥ 0.50 and score ≥ 0.1. "
        "(a) Cubic-logistic fits of P(detected) vs point count with 95% bootstrap bands over 200 resamples: the difficulty "
        "signal is continuous and monotone, while the Waymo rule keeps one bit of it. Vertical lines mark the literal Waymo "
        "threshold of 5 points and the density-matched threshold of 26 points. "
        "(b) The LEVEL 1 minus LEVEL 2 detection-rate gap grows as the threshold moves onto KITTI's density scale; at the "
        "literal threshold the LEVEL 2 group holds only 0.9% of pedestrians. Grey dashed line, right axis: LEVEL 2 share. "
        f"(c) Point-count terciles inside literal LEVEL 1, with edges at {tercile_edges} points: a {mean_spread:.0f} pp "
        "average detection-rate spread that the binary level discards. "
        f"(d) Ground-range ECDFs pooled over all objects: LEVEL 2 membership tracks range, r = {level_range_r:+.2f} under "
        "the density-matched rule. "
        "Verdict - the Waymo heuristic is directionally valid but sensor-relative, coarse, and range-confounded: the gap "
        "a geometric visibility metric must close."
    )
    fig.text(0.01, 0.010, caption, fontsize=7, va="bottom", ha="left", wrap=True)
    path = output_dir / "fig_waymo_rule_effectiveness.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_summary(results_dir, output_dir, detectors, density_matched_threshold):
    """Single-panel summary: detection rate per detector, All vs LEVEL_1 vs LEVEL_2 (density-matched rule)."""
    detectors = canonical_order(detectors)
    literal = read_csv(results_dir / "evaluation" / "waymo_level_metrics.csv")
    sweep = read_csv(results_dir / "evaluation" / "waymo_threshold_sweep.csv")
    all_rows = [row for row in literal if row["group"] == "LEVEL_2_cumulative"]
    matched = [row for row in sweep if int(row["threshold_points"]) == density_matched_threshold]
    literal_l2_n = next(int(row["instances"]) for row in literal if row["group"] == "LEVEL_2_exclusive")
    total_n = int(all_rows[0]["instances"])

    groups = (
        ("All pedestrians", all_rows, LEVEL_COLORS["all"]),
        (f"LEVEL 1, > {density_matched_threshold} pts", [r for r in matched if r["group"] == "LEVEL_1"], LEVEL_COLORS["LEVEL_1"]),
        (f"LEVEL 2, ≤ {density_matched_threshold} pts", [r for r in matched if r["group"] == "LEVEL_2_exclusive"], LEVEL_COLORS["LEVEL_2"]),
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.095, 1, 0.905))
    width = 0.26
    positions = np.arange(len(detectors))
    for offset, (label, rows, color) in enumerate(groups):
        values = [float(next(r for r in rows if r["detector"] == detector)["tp_percent"]) for detector in detectors]
        bars = ax.bar(positions + (offset - 1) * width, values, width * 0.92, color=color, zorder=3,
                      label=f"{label}  (n = {int(rows[0]['instances']):,})")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5, f"{value:.0f}",
                    ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(positions)
    ax.set_xticklabels([DETECTOR_LABELS.get(detector, detector) for detector in detectors])
    ax.set_ylabel("Pedestrians detected (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Pedestrian detection across Waymo difficulty levels — KITTI validation", fontsize=12)
    caption = (
        f"Waymo difficulty rule on 2,280 KITTI validation pedestrians ({total_n:,} evaluable), shown at a density-matched "
        f"threshold: LEVEL 2 = ≤ {density_matched_threshold} LiDAR points in the box ≈ the 10th percentile. Waymo's published "
        f"≤ 5-point threshold assigns 99.1% of these pedestrians to LEVEL 1 (LEVEL 2 n = {literal_l2_n}, omitted for "
        "stability). Detection = confidence-ranked prediction matching the GT box at 3D IoU ≥ 0.50, score ≥ 0.1; other-object "
        "matches ignored. LEVEL 2 pedestrians are detected far less often for every detector: the point-count heuristic ranks "
        "difficulty, but the one-bit label is sensor-relative and range-confounded."
    )
    fig.text(0.012, 0.012, caption, fontsize=7.5, va="bottom", ha="left", wrap=True)
    path = output_dir / "fig_waymo_detector_summary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("waymo_baseline/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("waymo_baseline/results/plots"))
    parser.add_argument("--density-matched-threshold", type=int, default=26)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detectors = sorted({row["detector"] for row in read_csv(args.results_dir / "analysis" / "waymo_effectiveness_summary.csv")})
    paths = [
        figure_level_performance(args.results_dir, args.output_dir, detectors, args.density_matched_threshold),
        figure_effectiveness(args.results_dir, args.output_dir, detectors, args.bootstrap, args.seed,
                             args.density_matched_threshold),
        figure_summary(args.results_dir, args.output_dir, detectors, args.density_matched_threshold),
    ]
    for path in paths:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
