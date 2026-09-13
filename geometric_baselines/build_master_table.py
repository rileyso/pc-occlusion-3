#!/usr/bin/env python
"""Build the master per-object table for the classical geometric baselines.

One row per ground-truth pedestrian in the KITTI val natural benchmark with:

  * the classical geometric covariates: ground range, radial range, bearing,
    LiDAR points inside the GT box, volumetric return density (points / m^3),
    LiDAR-facing box surface area and facing-surface return density
    (points / m^2), plus the KITTI annotations (occlusion state, truncation,
    difficulty) and oriented box fields;
  * a per-detector `detected` flag (0/1) for PointPillars, SECOND, PointRCNN
    and PV-RCNN.

Detected semantics (same for every detector): greedy confidence-ranked
matching at 3D IoU >= 0.50 with score >= 0.1; predictions matched to other
annotated pedestrians in the frame are ignored rather than counted as hits.

Flags are taken from the existing per-object outcome tables (dcd_baseline
analysis outputs, n=2,279) and the single instance they exclude - frame 002424
pedestrian 1, which has zero LiDAR returns and hence no outcome row - is
recomputed here with the identical matching protocol, giving n=2,280 for every
detector. Join key: (frame_id, label_index); joined rows are cross-checked on
ground range and point count.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from gilroy_lidar_replication.build_benchmark import surface_samples  # noqa: E402
from gilroy_lidar_replication.evaluate import load_predictions, prepare_frames  # noqa: E402

DETECTORS = ("pointpillar", "second", "pointrcnn", "pv_rcnn")
BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
INSTANCES_COLUMNS = (
    "frame_id", "label_index", "class_name", "ground_range_m", "radial_range_m", "bearing_deg",
    "num_points_in_gt", "point_density_m3", "x", "y", "z", "dx", "dy", "dz", "heading",
    "kitti_occlusion", "kitti_truncation", "kitti_difficulty",
)
NEW_COLUMNS = ("facing_area_m2", "point_density_m2_facing") + tuple(f"detected_{d}" for d in DETECTORS)

SCORE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
# Tolerances for the join-consistency check (ground_range is float32-rounded).
CHECK_FIELDS = {"ground_range_m": 1e-3, "num_points_in_gt": 0.5}


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_outcomes(path):
    """Map (frame_id, label_index) -> {detected, covariates} from one outcome table."""
    outcomes = {}
    for row in read_csv_rows(path):
        key = (row["frame_id"], int(row["label_index"]))
        if key in outcomes:
            raise ValueError(f"duplicate object key {key} in {path}")
        outcomes[key] = {"detected": int(row["detected"]),
                         **{field: float(row[field]) for field in CHECK_FIELDS}}
    return outcomes


def greedy_detected(prepared_frame, score_threshold=SCORE_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """Greedy score-ranked matching for one frame; returns one 0/1 per GT row.

    Identical rule to the existing per-object outcome tables: each prediction
    (descending score, >= score_threshold) covers the best unused GT at
    IoU >= iou_threshold; predictions covering no GT are simply not used.
    """
    matched = np.zeros(len(prepared_frame["gt"]), dtype=int)
    used = set()
    for pred_index, prediction in enumerate(prepared_frame["predictions"]):
        if prediction["score"] < score_threshold:
            break  # predictions are score-sorted descending
        overlaps = prepared_frame["iou"][pred_index]
        best_iou, best_index = max(
            ((overlaps[i], i) for i in range(len(matched)) if i not in used),
            default=(0.0, None),
        )
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matched[best_index] = 1
    return matched


def fill_missing_flags(rows, missing_keys, detections_dir, target_class):
    """Recompute detected flags for instances absent from the outcome tables."""
    if not missing_keys:
        return
    affected_frames = sorted({frame_id for frame_id, _label_index in missing_keys})
    for detector in DETECTORS:
        predictions = load_predictions(detections_dir / f"{detector}_val.json", target_class)
        # all pedestrians of the affected frames participate as GT so greedy
        # matching sees the same competition as in the full evaluation
        frame_rows = [row for row in rows if row["frame_id"] in affected_frames]
        prepared = prepare_frames(
            [{"frame_id": row["frame_id"], "label_index": int(row["label_index"]),
              "box": np.asarray([row[field] for field in BOX_FIELDS], dtype=np.float32)}
             for row in frame_rows],
            predictions,
        )
        flags = {}
        for frame_id, frame in prepared.items():
            for row, hit in zip(frame["gt"], greedy_detected(frame)):
                flags[(frame_id, int(row["label_index"]))] = int(hit)
        for row in rows:
            key = (row["frame_id"], int(row["label_index"]))
            if key in flags and row[f"detected_{detector}"] == "":
                row[f"detected_{detector}"] = flags[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path,
                        default=Path("research_workspace/dcd_baseline/results/natural/dcd_instances.csv"))
    parser.add_argument("--outcomes-dir", type=Path,
                        default=Path("research_workspace/dcd_baseline/results/analysis"))
    parser.add_argument("--detections-dir", type=Path,
                        default=Path("research_workspace/dcd_baseline/results/detections"),
                        help="Per-detector <detector>_val.json prediction files, used only to "
                             "recompute flags for instances the outcome tables exclude.")
    parser.add_argument("--surface-resolution", type=int, default=12,
                        help="Samples per face axis for the facing-area estimate.")
    parser.add_argument("--target-class", default="Pedestrian")
    parser.add_argument("--output-dir", type=Path, default=Path("research_workspace/geometric_baselines/results"))
    args = parser.parse_args()

    instances = read_csv_rows(args.instances)
    # keep only the classical geometric covariates; the source table's DCD and
    # ray-cast occlusion columns are out of scope for this baseline
    rows = []
    for source in instances:
        row = {column: source[column] for column in INSTANCES_COLUMNS}
        rows.append(row)

    outcomes = {detector: load_outcomes(args.outcomes_dir / f"per_object_outcomes_{detector}.csv")
                for detector in DETECTORS}

    matched = {detector: 0 for detector in DETECTORS}
    mismatches = 0
    missing_keys = set()
    for row in rows:
        box = np.asarray([float(row[field]) for field in BOX_FIELDS], dtype=float)
        _samples, weights, _head = surface_samples(box, args.surface_resolution)
        facing_area = float(weights.sum())
        num_points = float(row["num_points_in_gt"])
        row["facing_area_m2"] = f"{facing_area:.6f}"
        row["point_density_m2_facing"] = f"{num_points / facing_area:.6f}" if facing_area > 0 else ""
        key = (row["frame_id"], int(row["label_index"]))
        for detector in DETECTORS:
            outcome = outcomes[detector].get(key)
            if outcome is not None:
                matched[detector] += 1
                row[f"detected_{detector}"] = outcome["detected"]
                for field, tolerance in CHECK_FIELDS.items():
                    if abs(float(row[field]) - outcome[field]) > tolerance:
                        mismatches += 1
            else:
                row[f"detected_{detector}"] = ""
                missing_keys.add(key)
    if mismatches:
        raise ValueError(f"{mismatches} covariate mismatches between outcome tables and instances table")

    fill_missing_flags(rows, sorted(missing_keys), args.detections_dir, args.target_class)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(INSTANCES_COLUMNS) + list(NEW_COLUMNS)
    output_path = args.output_dir / "master_per_object.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    recomputed = sorted(missing_keys)
    protocol = {
        "source_instances": str(args.instances),
        "outcome_tables": {detector: str(args.outcomes_dir / f"per_object_outcomes_{detector}.csv")
                           for detector in DETECTORS},
        "detections": {detector: str(args.detections_dir / f"{detector}_val.json") for detector in DETECTORS},
        "join_key": ["frame_id", "label_index"],
        "matched_from_outcome_tables": matched,
        "flags_recomputed": [f"{frame_id}#{label_index}" for frame_id, label_index in recomputed],
        "instances_total": len(rows),
        "surface_resolution_per_face_axis": args.surface_resolution,
        "density_definitions": {
            "point_density_m3": "num_points_in_gt / (dx*dy*dz) box volume",
            "point_density_m2_facing": "num_points_in_gt / LiDAR-facing box surface area (m^2), "
                                       "back faces excluded",
        },
        "detected_semantics": "greedy confidence-ranked matching, 3D IoU >= 0.50, score >= 0.1, "
                              "predictions matching other annotated pedestrians ignored",
    }
    (args.output_dir / "master_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE {output_path} ({len(rows)} instances)")
    for detector in DETECTORS:
        print(f"  {detector}: {matched[detector]} from outcome tables + "
              f"{sum(1 for row in rows if row[f'detected_{detector}'] != '') - matched[detector]} recomputed, "
              f"overall {100.0 * sum(int(row[f'detected_{detector}']) for row in rows) / len(rows):.1f}% detected")


if __name__ == "__main__":
    main()
