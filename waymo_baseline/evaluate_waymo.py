#!/usr/bin/env python
"""Evaluate the Waymo point-count difficulty rule on the natural KITTI pedestrian benchmark.

The Waymo Open Dataset assigns detection difficulty from LiDAR observation richness
(Sun et al., CVPR 2020, section 5.1): 3D labels without any LiDAR points are ignored,
labels with <= 5 points (or flagged hard by a labeler, a criterion that does not
transfer to KITTI and is dropped here) are LEVEL_2, the rest are LEVEL_1, and
LEVEL_2 metrics are reported cumulatively (LEVEL_1 included).

This script transplants that rule onto the 2,280 KITTI validation pedestrians used
by the DCD baseline, reusing the Gilroy evaluation core unchanged (greedy
confidence-ranked matching, 3D IoU >= 0.50, score >= 0.1, other-object matches
ignored), for four pedestrian-capable detectors. It evaluates:

  1. the literal rule (<= 5 points) - LEVEL_1 / LEVEL_2 (exclusive) / LEVEL_2
     cumulative (= all non-ignored objects);
  2. a threshold sweep - because the literal threshold degenerates on KITTI's
     denser LiDAR, the same binary rule is evaluated at 10 / 26 / 50 / 100 points
     (26 is the ~10th percentile, a density-matched operating point);
  3. per-object detection status for the plotting/regression layer, written with
     the level assignments (results/analysis/per_object_waymo_levels.csv);
  4. an effectiveness summary - how much difficulty signal one point-count bit
     carries: AUC of the binary level vs continuous log point count and range,
     point-biserial level-range correlation, and the detection-rate gradient
     inside LEVEL_1 (point-count terciles) that the binary rule discards.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from dcd_baseline.plot_dcd import per_object_status  # noqa: E402
from gilroy_lidar_replication.evaluate import evaluate_subset, load_predictions, prepare_frames  # noqa: E402

BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
WAYMO_LEVEL_2_MAX_POINTS = 5  # Sun et al., CVPR 2020, section 5.1 (<= 5 points)
GILROY_MINIMUM = 55
DEFAULT_SWEEP = (5, 10, 26, 50, 100)
DENSITY_MATCHED_THRESHOLD = 26  # ~10th percentile of KITTI pedestrian point counts (density-matched rule)
METRIC_FIELDS = ("instances", "meets_gilroy_minimum", "map_3d_50_95", "ap_3d_50",
                 "tp_percent", "fp_count", "fp_per_gt", "fn_percent")


def load_instances(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        row["num_points_in_gt"] = float(row["num_points_in_gt"])
        row["ground_range_m"] = float(row["ground_range_m"])
        row["box"] = np.asarray([row[field] for field in BOX_FIELDS], dtype=np.float32)
        # Waymo convention: 3D labels without any LiDAR points are ignored.
        row["ignored"] = row["num_points_in_gt"] == 0
    return rows


def subset_metrics(prepared, predicate, score_threshold):
    """Evaluate one difficulty group; selectors run per frame over that frame's GT rows."""
    metrics = evaluate_subset(prepared, lambda rows: [i for i, row in enumerate(rows) if predicate(row)],
                              score_threshold)
    metrics["meets_gilroy_minimum"] = metrics["instances"] >= GILROY_MINIMUM
    return metrics


def auc(scores, outcomes):
    """Mann-Whitney AUC of outcomes=1 vs outcomes=0 ranked by scores (ties averaged)."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    positives, negatives = outcomes == 1, outcomes == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0  # 1-based average rank over the tie block
        start = end
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2.0)
                 / (positives.sum() * negatives.sum()))


def effectiveness_row(detector, table, density_matched_threshold):
    """Per-detector effectiveness statistics from the per-object table (list of dicts)."""
    live = np.asarray([not entry["ignored"] for entry in table])
    points = np.asarray([entry["num_points_in_gt"] for entry in table], dtype=float)[live]
    ranges = np.asarray([entry["ground_range_m"] for entry in table], dtype=float)[live]
    detected = np.asarray([entry["detected"] for entry in table], dtype=int)[live]
    level = (points > WAYMO_LEVEL_2_MAX_POINTS).astype(float)
    level_p10 = (points > density_matched_threshold).astype(float)
    log_points = np.log1p(points)

    literal_l2_mask = points <= WAYMO_LEVEL_2_MAX_POINTS
    literal_l1 = ~literal_l2_mask
    literal_l2 = literal_l2_mask
    row = {
        "detector": detector,
        "objects_evaluated": int(live.sum()),
        "objects_ignored_zero_points": int((~live).sum()),
        "overall_detection_rate_pct": round(100.0 * detected.mean(), 2),
        "auc_level_literal": round(auc(level, detected), 3),
        "auc_level_density_matched": round(auc(level_p10, detected), 3),
        "auc_log_points": round(auc(log_points, detected), 3),
        "auc_neg_range": round(auc(-ranges, detected), 3),
        "r_points_range": round(float(np.corrcoef(points, ranges)[0, 1]), 3),
        "r_level2_literal_range": round(float(np.corrcoef(literal_l2_mask.astype(float), ranges)[0, 1]), 3),
        "r_level2_p10_range": round(float(np.corrcoef((points <= density_matched_threshold).astype(float), ranges)[0, 1]), 3),
        "literal_l1_rate_pct": round(100.0 * detected[literal_l1].mean(), 2),
        "literal_l2_rate_pct": round(100.0 * detected[literal_l2].mean(), 2) if literal_l2.any() else float("nan"),
        "literal_gap_pp": round(100.0 * (detected[literal_l1].mean() - detected[literal_l2].mean()), 2),
    }

    # Detection-rate gradient inside LEVEL_1 (point-count terciles): the continuous
    # signal that the binary threshold collapses into a single bit. Correlations use
    # the LEVEL_2-membership convention: positive = LEVEL_2 skews to larger range.
    tercile_edges = np.quantile(points[literal_l1], [1 / 3, 2 / 3])
    tercile_rates = []
    lower = 0.0
    for edge in list(tercile_edges) + [np.inf]:
        member = literal_l1 & (points > lower) & (points <= edge)
        tercile_rates.append(100.0 * detected[member].mean())
        lower = edge
    for position, rate in enumerate(tercile_rates, start=1):
        row[f"literal_l1_tercile_{position}_rate_pct"] = round(rate, 2)
    row["literal_l1_tercile_spread_pp"] = round(max(tercile_rates) - min(tercile_rates), 2)
    row["literal_l1_tercile_edges"] = f"{tercile_edges[0]:.0f}|{tercile_edges[1]:.0f} points"
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path,
                        default=Path("dcd_baseline/results/natural/dcd_instances.csv"))
    parser.add_argument("--prediction", action="append", required=True, help="DETECTOR=predictions.json (repeatable)")
    parser.add_argument("--target-class", default="Pedestrian")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--sweep", type=int, nargs="+", default=list(DEFAULT_SWEEP),
                        help="LEVEL_2 point-count thresholds for the sweep; 5 is the literal Waymo rule.")
    parser.add_argument("--output-dir", type=Path, default=Path("waymo_baseline/results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    if min(args.sweep) != WAYMO_LEVEL_2_MAX_POINTS:
        parser.error(f"--sweep must include the literal Waymo threshold {WAYMO_LEVEL_2_MAX_POINTS}")
    evaluation_dir = args.output_dir / "evaluation"
    analysis_dir = args.output_dir / "analysis"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.instances)
    ignored_total = sum(row["ignored"] for row in instances)
    print(f"instances: {len(instances)} ({ignored_total} ignored per the Waymo zero-point convention)")

    level_rows, sweep_rows, summary_rows = [], [], []
    per_object = None
    for specifier in args.prediction:
        detector, prediction_path = specifier.split("=", 1)
        prepared = prepare_frames(instances, load_predictions(prediction_path, args.target_class))
        table = per_object_status(prepared, args.score_threshold)
        points = np.asarray([entry["num_points_in_gt"] for entry in table], dtype=float)
        detected = np.asarray([entry["detected"] for entry in table], dtype=int)
        ignored_mask = np.asarray([entry["ignored"] for entry in table], dtype=bool)

        # Literal Waymo rule: LEVEL_2 <= 5 points, LEVEL_1 > 5, cumulative LEVEL_2 = all non-ignored.
        groups = {
            "LEVEL_1": lambda row: not row["ignored"] and row["num_points_in_gt"] > WAYMO_LEVEL_2_MAX_POINTS,
            "LEVEL_2_exclusive": lambda row: not row["ignored"] and row["num_points_in_gt"] <= WAYMO_LEVEL_2_MAX_POINTS,
            "LEVEL_2_cumulative": lambda row: not row["ignored"],
        }
        for group, predicate in groups.items():
            metrics = subset_metrics(prepared, predicate, args.score_threshold)
            level_rows.append({"detector": detector, "rule": "waymo_literal_5pt", "group": group, **metrics})
        literal_l2_count = int((~ignored_mask & (points <= WAYMO_LEVEL_2_MAX_POINTS)).sum())
        print(f"{detector}: literal rule evaluated (LEVEL_2 n={literal_l2_count})")

        # Threshold sweep of the same binary rule (exclusive LEVEL_1/LEVEL_2).
        for threshold in args.sweep:
            for group, predicate in {
                "LEVEL_1": lambda row, t=threshold: not row["ignored"] and row["num_points_in_gt"] > t,
                "LEVEL_2_exclusive": lambda row, t=threshold: not row["ignored"] and row["num_points_in_gt"] <= t,
            }.items():
                metrics = subset_metrics(prepared, predicate, args.score_threshold)
                sweep_rows.append({"detector": detector, "threshold_points": threshold, "group": group, **metrics})
            level_2 = ~ignored_mask & (points <= threshold)
            level_1 = ~ignored_mask & (points > threshold)
            print(f"  sweep t={threshold}: LEVEL_2 n={int(level_2.sum())} "
                  f"rate={100.0 * detected[level_2].mean():.1f}% | LEVEL_1 n={int(level_1.sum())} "
                  f"rate={100.0 * detected[level_1].mean():.1f}%")

        summary_rows.append(effectiveness_row(detector, table, DENSITY_MATCHED_THRESHOLD))

        if per_object is None:
            per_object = {
                "frame_id": [entry["frame_id"] for entry in table],
                "label_index": [entry.get("label_index", "") for entry in table],
                "num_points_in_gt": points,
                "ground_range_m": np.asarray([entry["ground_range_m"] for entry in table], dtype=float),
                "ignored_zero_points": ignored_mask.astype(int),
                "level_literal": np.where(ignored_mask, "ignored", np.where(points <= 5, "LEVEL_2", "LEVEL_1")),
                "level_p10_t26": np.where(ignored_mask, "ignored", np.where(points <= 26, "LEVEL_2", "LEVEL_1")),
            }
        per_object[f"detected_{detector}"] = detected

    level_path = evaluation_dir / "waymo_level_metrics.csv"
    with level_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["detector", "rule", "group", *METRIC_FIELDS])
        writer.writeheader()
        writer.writerows(level_rows)
    sweep_path = evaluation_dir / "waymo_threshold_sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["detector", "threshold_points", "group", *METRIC_FIELDS])
        writer.writeheader()
        writer.writerows(sweep_rows)
    summary_path = analysis_dir / "waymo_effectiveness_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    per_object_path = analysis_dir / "per_object_waymo_levels.csv"
    with per_object_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        keys = list(per_object.keys())
        writer.writerow(keys)
        for position in range(len(per_object[keys[0]])):
            writer.writerow([per_object[key][position] for key in keys])

    protocol = {
        "baseline": "Waymo Open Dataset point-count difficulty rule transplanted to KITTI pedestrians",
        "rule_source": "Sun et al., CVPR 2020 (arXiv:1912.04838), section 5.1",
        "rule_as_published": {
            "ignored": "3D labels without any LiDAR points",
            "LEVEL_2": "labeler-flagged hard OR <= 5 LiDAR points in the box (only the point-count criterion transfers; the labeler flag is dropped)",
            "LEVEL_1": "the rest (> 5 points)",
            "reporting": "LEVEL_2 metrics are cumulative (include LEVEL_1); exclusive per-level subsets are computed here for the difficulty gap",
        },
        "threshold_sweep": {
            "thresholds": args.sweep,
            "rationale": {
                "5": "literal Waymo rule",
                "10": "first sweep threshold whose LEVEL_2 subset meets the Gilroy >= 55-instance minimum",
                "26": "~10th percentile of KITTI pedestrian point counts (density-matched operating point)",
                "50": "unbalanced-to-moderate split (LEVEL_2 = ~22%)",
                "100": "near-balanced split (LEVEL_2 = ~45%)",
            },
        },
        "instances": {"source": str(args.instances), "total": len(instances), "ignored_zero_points": int(ignored_total),
                      "note": "num_points_in_gt equals the DCD pipeline's n_observed (returns inside the GT box); verified identical on all rows"},
        "matching_protocol": "greedy confidence-ranked, 3D IoU >= 0.50, score >= 0.1, predictions matching other annotated objects ignored (Gilroy semantics)",
        "detectors": sorted(specifier.split("=")[0] for specifier in args.prediction),
        "score_threshold": args.score_threshold,
    }
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    for path in (level_path, sweep_path, summary_path, per_object_path, protocol_path):
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
