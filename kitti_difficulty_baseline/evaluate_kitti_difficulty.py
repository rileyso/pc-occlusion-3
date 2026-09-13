#!/usr/bin/env python
"""Evaluate the KITTI Easy/Moderate/Hard difficulty labels (and the raw KITTI
occlusion annotation levels) as a per-object metric on the natural KITTI
pedestrian benchmark.

KITTI's own coarse benchmark labels - Easy / Moderate / Hard, defined by the
devkit rule (Geiger et al., CVPR 2012) from the occlusion annotation, image
truncation and 2D bbox height - are the first 'established benchmark'
difficulty labels the thesis compares its fine-grained visibility metric
against. This script transplants nothing: the labels are native to KITTI. It
reuses the Gilroy evaluation core unchanged (greedy confidence-ranked matching,
3D IoU >= 0.50, score >= 0.1, predictions matching other annotated objects
ignored), so per-group numbers are directly comparable with the Gilroy
occlusion bins, the DCD bins and the Waymo LEVEL_1/2 baseline.

For every grouping (see groupings.py) it evaluates two group views:

- exclusive (disjoint bands): the view for the report bar figure;
- cumulative (official semantics): group d holds every object the official
  protocol counts at difficulty d (UnKnown objects ignored everywhere).

Outputs: per-group metrics CSV (all groupings/views), a per-object table with
the label assignments and per-detector detection status (the SS5.6 regression
join), a summary of what the coarse label does and does not resolve (AUCs,
per-group covariate means), and protocol.json.
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
from kitti_difficulty_baseline import groupings as grouping_registry  # noqa: E402
from waymo_baseline.evaluate_waymo import auc  # noqa: E402

BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
GILROY_MINIMUM = 55
METRIC_FIELDS = ("instances", "meets_gilroy_minimum", "map_3d_50_95", "ap_3d_50",
                 "tp_percent", "fp_count", "fp_per_gt", "fn_percent")
DIFFICULTY_ORDINAL = {"Easy": 0, "Moderate": 1, "Hard": 2}  # UnKnown excluded (official semantics)


def load_instances(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        for field in ("num_points_in_gt", "ground_range_m", "radial_range_m", "occlusion_percent",
                      "head_visible_percent", "kitti_truncation", "point_density_m3", "dcd"):
            if row.get(field):
                row[field] = float(row[field])
        row["box"] = np.asarray([row[name] for name in BOX_FIELDS], dtype=np.float32)
    return rows


def subset_metrics(prepared, predicate, score_threshold):
    metrics = evaluate_subset(prepared, lambda rows: [i for i, row in enumerate(rows) if predicate(row)],
                              score_threshold)
    metrics["meets_gilroy_minimum"] = metrics["instances"] >= GILROY_MINIMUM
    return metrics


def group_metrics_rows(detector, grouping, prepared, score_threshold):
    rows = []
    for mode, spec in (("exclusive", grouping.exclusive_groups()),
                       ("cumulative", grouping.cumulative_groups())):
        for key, display, members in spec:
            member_set = set(members)
            metrics = subset_metrics(prepared, lambda row, allowed=member_set: row[grouping.column] in allowed,
                                     score_threshold)
            rows.append({"detector": detector, "grouping": grouping.name, "mode": mode, "group": key,
                         "group_label": display, **metrics})
    return rows


def summary_rows(instances, per_object):
    """What the coarse label does and does not resolve: group covariate means
    (the conflation evidence) and per-detector AUC of the difficulty label."""
    covariate_fields = ("ground_range_m", "num_points_in_gt", "occlusion_percent",
                        "head_visible_percent", "kitti_truncation")
    rows = []
    for value in grouping_registry.get_grouping("kitti_difficulty").order:
        members = [row for row in instances if row["kitti_difficulty"] == value]
        row = {"statistic": "group_profile", "group": value, "instances": len(members)}
        for field in covariate_fields:
            values = np.asarray([entry[field] for entry in members], dtype=float)
            row[f"mean_{field}"] = round(float(values.mean()), 2) if len(values) else float("nan")
        rows.append(row)

    table = per_object["_rows"]
    live = [position for position, entry in enumerate(table)
            if entry["kitti_difficulty"] in DIFFICULTY_ORDINAL]
    easiness = np.asarray([2 - DIFFICULTY_ORDINAL[table[position]["kitti_difficulty"]] for position in live])
    ranges = np.asarray([table[position]["ground_range_m"] for position in live], dtype=float)
    points = np.asarray([table[position]["num_points_in_gt"] for position in live], dtype=float)
    ray_occlusion = np.asarray([table[position]["occlusion_percent"] for position in live], dtype=float)
    for key in per_object:
        if not key.startswith("detected_"):
            continue
        detected = np.asarray([per_object[key][position] for position in live], dtype=int)
        rows.append({"statistic": "detector_auc", "group": key[len("detected_"):], "instances": len(live),
                     "auc_easiness_label": round(auc(easiness, detected), 3),
                     "auc_ray_occlusion": round(auc(-ray_occlusion, detected), 3),
                     "auc_log_points": round(auc(np.log1p(points), detected), 3),
                     "auc_neg_range": round(auc(-ranges, detected), 3)})
    return rows


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with Path(path).open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path,
                        default=Path("research_workspace/dcd_baseline/results/natural/dcd_instances.csv"))
    parser.add_argument("--prediction", action="append", required=True,
                        help="DETECTOR=predictions.json (repeatable)")
    parser.add_argument("--target-class", default="Pedestrian")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--grouping", action="append", default=None,
                        help="Grouping(s) to evaluate (default: every registered grouping)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research_workspace/kitti_difficulty_baseline/results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    selected = ([grouping_registry.get_grouping(name) for name in args.grouping]
                if args.grouping else list(grouping_registry.GROUPINGS.values()))
    evaluation_dir = args.output_dir / "evaluation"
    analysis_dir = args.output_dir / "analysis"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.instances)
    counts = {value: sum(row["kitti_difficulty"] == value for row in instances)
              for value in ("Easy", "Moderate", "Hard", "UnKnown")}
    print(f"instances: {len(instances)} | Easy={counts['Easy']} Moderate={counts['Moderate']} "
          f"Hard={counts['Hard']} UnKnown={counts['UnKnown']}")

    metric_rows, per_object = [], {"_rows": None}
    detector_meta = {}
    prediction_paths = dict(specifier.split("=", 1) for specifier in args.prediction)
    for specifier in args.prediction:
        detector, prediction_path = specifier.split("=", 1)
        with Path(prediction_path).open("r", encoding="utf-8") as input_file:
            artifact = json.load(input_file)
        detector_meta[detector] = {key: artifact.get(key) for key in ("model_config", "checkpoint")}
        prepared = prepare_frames(instances, load_predictions(prediction_path, args.target_class))
        for grouping in selected:
            rows = group_metrics_rows(detector, grouping, prepared, args.score_threshold)
            metric_rows.extend(rows)
            for row in rows:
                if row["mode"] == "exclusive" and row["grouping"] == "kitti_difficulty":
                    print(f"  {detector} {row['group']:>8}: n={row['instances']:4d} "
                          f"AP@0.50={100.0 * row['ap_3d_50']:5.1f} TP%={row['tp_percent']:5.1f}")
        table = per_object_status(prepared, args.score_threshold)
        if per_object["_rows"] is None:
            per_object["_rows"] = table
        per_object[f"detected_{detector}"] = [entry["detected"] for entry in table]
        print(f"{detector}: evaluated")

    rows = per_object["_rows"]
    per_object_rows = [{
        "frame_id": row["frame_id"],
        "label_index": row.get("label_index", ""),
        "class_name": row.get("class_name", ""),
        "kitti_difficulty": row["kitti_difficulty"],
        "kitti_occlusion": row["kitti_occlusion"],
        "kitti_truncation": row["kitti_truncation"],
        "occlusion_percent": row["occlusion_percent"],
        "head_visible_percent": row["head_visible_percent"],
        "num_points_in_gt": int(row["num_points_in_gt"]),
        "ground_range_m": row["ground_range_m"],
        "dcd": row.get("dcd", ""),
    } for row in rows]
    for key, values in per_object.items():
        if key.startswith("detected_"):
            for row, value in zip(per_object_rows, values):
                row[key] = value

    write_csv(evaluation_dir / "kitti_difficulty_metrics.csv", metric_rows,
              fieldnames=["detector", "grouping", "mode", "group", "group_label", *METRIC_FIELDS])
    write_csv(analysis_dir / "per_object_kitti_difficulty.csv", per_object_rows)
    write_csv(analysis_dir / "kitti_difficulty_summary.csv", summary_rows(instances, per_object))

    protocol = {
        "baseline": "KITTI Easy/Moderate/Hard benchmark difficulty labels evaluated as a per-object metric",
        "rule_source": "Geiger et al., CVPR 2012 (KITTI object devkit) difficulty definition",
        "rule_as_published": grouping_registry.KITTI_DIFFICULTY_RULE,
        "occlusion_annotation_levels": grouping_registry.KITTI_OCCLUSION_RULE,
        "group_views": {
            "exclusive": "disjoint groups - each object in its own difficulty band (report bar figure)",
            "cumulative": "official benchmark reporting semantics - group d holds every object the "
                          "official protocol counts at difficulty d; UnKnown objects ignored everywhere",
        },
        "groupings": {grouping.name: grouping.description for grouping in selected},
        "instances": {"source": str(args.instances), "total": len(instances), "counts": counts},
        "matching_protocol": "greedy confidence-ranked, 3D IoU >= 0.50, score >= 0.1, predictions "
                             "matching other annotated objects ignored (Gilroy semantics)",
        "detectors": {name: {**meta, "predictions": prediction_paths[name]}
                      for name, meta in detector_meta.items()},
        "score_threshold": args.score_threshold,
        "gilroy_minimum_per_group": GILROY_MINIMUM,
    }
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {protocol_path}")


if __name__ == "__main__":
    main()
