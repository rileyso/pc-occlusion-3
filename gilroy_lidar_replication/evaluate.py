#!/usr/bin/env python
"""Evaluate OpenPCDet Pedestrian predictions by objective 3D occlusion bin."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils


IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


def load_instances(path, arm=None):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        source_rows = list(csv.DictReader(input_file))
    rows = []
    for source in source_rows:
        if arm and source.get("arm") != arm:
            continue
        source["frame_id"] = source.get("sample_id", source.get("frame_id", ""))
        source["occlusion_bin"] = int(source["occlusion_bin"])
        source["box"] = np.asarray([source[name] for name in ("x", "y", "z", "dx", "dy", "dz", "heading")], dtype=np.float32)
        rows.append(source)
        for box in json.loads(source.get("ignore_boxes_json") or "[]"):
            rows.append({"frame_id": source["frame_id"], "occlusion_bin": -1, "box": np.asarray(box, dtype=np.float32)})
    return rows


def load_predictions(path, target_class="Pedestrian", max_detections=100):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    records = value.get("detections", value) if isinstance(value, dict) else value
    grouped = defaultdict(list)
    for record in records:
        if record.get("class_name") == target_class:
            grouped[str(record["frame_id"]).zfill(6)].append({
                "score": float(record["score"]),
                "box": np.asarray(record["box_lidar"][:7], dtype=np.float32),
            })
    for frame_id in grouped:
        grouped[frame_id] = sorted(grouped[frame_id], key=lambda row: row["score"], reverse=True)[:max_detections]
    return grouped


def iou_matrix(predictions, ground_truth):
    if not len(predictions) or not len(ground_truth):
        return np.zeros((len(predictions), len(ground_truth)), dtype=np.float32)
    if not torch.cuda.is_available():
        raise RuntimeError("OpenPCDet 3D IoU evaluation requires CUDA")
    pred = torch.as_tensor(predictions, dtype=torch.float32, device="cuda")
    gt = torch.as_tensor(ground_truth, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        return iou3d_nms_utils.boxes_iou3d_gpu(pred, gt).cpu().numpy()


def prepare_frames(instances, predictions):
    ground_truth = defaultdict(list)
    for row in instances:
        ground_truth[row["frame_id"]].append(row)
    prepared = {}
    for frame_id, rows in ground_truth.items():
        frame_predictions = predictions.get(frame_id, [])
        prepared[frame_id] = {
            "gt": rows,
            "predictions": frame_predictions,
            "iou": iou_matrix(
                np.asarray([row["box"] for row in frame_predictions], dtype=np.float32),
                np.asarray([row["box"] for row in rows], dtype=np.float32),
            ),
        }
    return prepared


def ranked_outcomes(prepared, selected_indices, threshold):
    """Score selected GTs; predictions matching non-selected GTs are ignored."""
    ranked = []
    selected_total = 0
    for frame_id, frame in prepared.items():
        active = set(selected_indices(frame["gt"]))
        if not active:
            continue
        selected_total += len(active)
        used = set()
        for pred_index, prediction in enumerate(frame["predictions"]):
            overlaps = frame["iou"][pred_index]
            available = [(overlaps[index], index) for index in active - used]
            best_iou, best_index = max(available, default=(0.0, None))
            if best_index is not None and best_iou >= threshold:
                used.add(best_index)
                outcome = 1
            elif any(overlaps[index] >= threshold for index in range(len(frame["gt"])) if index not in active):
                outcome = None
            else:
                outcome = 0
            if outcome is not None:
                ranked.append((prediction["score"], outcome))
    ranked.sort(reverse=True)
    return ranked, selected_total


def average_precision(ranked, gt_count):
    if not gt_count:
        return float("nan")
    if not ranked:
        return 0.0
    outcomes = np.asarray([item[1] for item in ranked])
    true_positive = np.cumsum(outcomes == 1)
    false_positive = np.cumsum(outcomes == 0)
    recall = true_positive / gt_count
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    return float(np.mean([precision[recall >= level].max(initial=0.0) for level in np.linspace(0, 1, 101)]))


def evaluate_subset(prepared, selected_indices, score_threshold=0.1):
    aps = []
    ap50_ranked = None
    gt_count = 0
    for threshold in IOU_THRESHOLDS:
        ranked, count = ranked_outcomes(prepared, selected_indices, threshold)
        aps.append(average_precision(ranked, count))
        if np.isclose(threshold, 0.5):
            ap50_ranked, gt_count = ranked, count
    kept = [outcome for score, outcome in ap50_ranked if score >= score_threshold]
    true_positive, false_positive = kept.count(1), kept.count(0)
    false_negative = gt_count - true_positive
    return {
        "instances": gt_count,
        "map_3d_50_95": float(np.mean(aps)),
        "ap_3d_50": aps[0],
        "tp_percent": 100.0 * true_positive / gt_count if gt_count else float("nan"),
        "fp_count": false_positive,
        "fp_per_gt": false_positive / gt_count if gt_count else float("nan"),
        "fn_percent": 100.0 * false_negative / gt_count if gt_count else float("nan"),
    }


def evaluate(instances, predictions, score_threshold=0.1):
    prepared = prepare_frames(instances, predictions)
    results = []
    for bin_index in range(10):
        metrics = evaluate_subset(
            prepared,
            lambda rows, value=bin_index: [index for index, row in enumerate(rows) if row["occlusion_bin"] == value],
            score_threshold,
        )
        results.append({"occlusion_bin": bin_index, "occlusion_range": f"{10 * bin_index}-{10 * bin_index + 9}%", **metrics})
    overall = evaluate_subset(prepared, lambda rows: [index for index, row in enumerate(rows) if row["occlusion_bin"] >= 0], score_threshold)
    results.append({"occlusion_bin": "all", "occlusion_range": "0-99%", **overall})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--arm", choices=("structured", "random"), help="Controlled-manifest arm to evaluate.")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gilroy_lidar_replication/evaluation"))
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    results = evaluate(load_instances(args.instances, args.arm), load_predictions(args.predictions), args.score_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.arm}" if args.arm else ""
    output = args.output_dir / f"{args.detector}{suffix}_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
