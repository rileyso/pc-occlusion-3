#!/usr/bin/env python
"""Official KITTI AP_R40 per difficulty (Easy / Moderate / Hard) on the full
validation split, for the four thesis detectors.

This is the 'official arm' of the KITTI difficulty baseline: the benchmark's
own evaluation protocol (Geiger et al. devkit semantics as implemented by
OpenPCDet) with its difficulty-dependent ignore rules - for difficulty d,
ground-truth objects failing the d-criteria are ignored entirely and
predictions matching them are not false positives. The three difficulty
numbers are therefore computed on different (nested) ground-truth sets, which
is exactly how every published KITTI result is reported - including the
reference PointPillars numbers used to verify this script.

Predictions are the pedestrian-preserving full-val JSONs written by
dcd_baseline/infer_natural.py (all classes kept; the devkit class matching and
ignore rules handle other-class detections). The dt_annos are built with the
same camera-frame projection convention as KittiDataset.generate_prediction_dicts
(lidar->camera boxes, image-space 2D bboxes, alpha), so results are directly
comparable with a test.py evaluation of the same checkpoint.

Modularity note: unlike the generic grouping evaluator (Gilroy core), official
benchmark protocols are dataset-specific by design - a future dataset (Waymo,
nuScenes, Mixed Signals) gets its own official-arm module with the same
output schema (detector, class, difficulty, ap), while the portable
predictions-JSON format and instances-CSV format stay unchanged.
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # noqa: E402
from pcdet.utils import box_utils, calibration_kitti  # noqa: E402

DIFFICULTY_NAMES = ("easy", "moderate", "hard")
# Official min 3D IoU per class for the reported metric (overlap row 0 of
# get_official_eval_result): Car 0.70, Pedestrian 0.50, Cyclist 0.50.
CLASS_IOU_3D = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
# eval_class expects devkit integer class indices (0 car, 1 pedestrian, 2 cyclist).
CLASS_INDEX = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def load_detections(path):
    """frame_id -> {field: np.array} of raw detections (all classes kept)."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    per_frame = {}
    for record in artifact["detections"]:
        per_frame.setdefault(str(record["frame_id"]).zfill(6), []).append(record)
    return artifact, per_frame


def build_dt_annos(per_frame, frame_ids, calib_dir, image_shapes):
    """Per-frame devkit dt annotations, following
    KittiDataset.generate_prediction_dicts (camera-frame projection included,
    because the devkit height-ignores small 2D detection boxes)."""
    template_keys = ("name", "truncated", "occluded", "alpha", "bbox", "dimensions",
                     "location", "rotation_y", "score", "boxes_lidar")
    dt_annos = []
    for frame_id in frame_ids:
        records = per_frame.get(frame_id, [])
        anno = {key: np.zeros(len(records)) for key in template_keys}
        anno["name"] = np.zeros(len(records), dtype=object)
        anno["bbox"] = np.zeros((len(records), 4))
        anno["dimensions"] = np.zeros((len(records), 3))
        anno["location"] = np.zeros((len(records), 3))
        anno["boxes_lidar"] = np.zeros((len(records), 7))
        if records:
            boxes = np.asarray([record["box_lidar"][:7] for record in records], dtype=np.float32)
            names = [record["class_name"] for record in records]
            scores = np.asarray([record["score"] for record in records], dtype=np.float32)
            calib = calibration_kitti.Calibration(str(calib_dir / f"{frame_id}.txt"))
            boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(boxes, calib)
            boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                boxes_camera, calib, image_shape=image_shapes[frame_id])
            anno["name"] = np.asarray(names, dtype=object)
            anno["alpha"] = -np.arctan2(-boxes[:, 1], boxes[:, 0]) + boxes_camera[:, 6]
            anno["bbox"] = boxes_img
            anno["dimensions"] = boxes_camera[:, 3:6]
            anno["location"] = boxes_camera[:, 0:3]
            anno["rotation_y"] = boxes_camera[:, 6]
            anno["score"] = scores
            anno["boxes_lidar"] = boxes
        dt_annos.append(anno)
    return dt_annos


def evaluate_class(gt_annos, dt_annos, class_name):
    """AP_R40 for the three difficulties of one class, official semantics."""
    min_overlaps = np.zeros((1, 3, 1))  # [num_overlap, metric, class]; only metric=2 (3D) is used
    min_overlaps[0, 2, 0] = CLASS_IOU_3D[class_name]
    result = kitti_eval.eval_class(gt_annos, dt_annos, [CLASS_INDEX[class_name]], [0, 1, 2], 2, min_overlaps)
    ap_r40 = kitti_eval.get_mAP_R40(result["precision"])  # [num_class=1, num_difficulty=3, num_minoverlap=1]
    return {difficulty: float(ap_r40[0, position, 0]) for position, difficulty in enumerate(DIFFICULTY_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True,
                        help="DETECTOR=predictions.json (repeatable)")
    parser.add_argument("--infos", type=Path, default=Path("data/kitti/kitti_infos_val.pkl"))
    parser.add_argument("--calib-dir", type=Path, default=Path("data/kitti/training/calib"))
    parser.add_argument("--class-name", action="append", default=None,
                        choices=list(CLASS_IOU_3D), help="Classes to evaluate (default: all three)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("kitti_difficulty_baseline/results"))
    return parser.parse_args()


def main():
    args = parse_args()
    class_names = args.class_name or list(CLASS_IOU_3D)
    evaluation_dir = args.output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    with args.infos.open("rb") as input_file:
        infos = pickle.load(input_file)
    frame_ids = [str(info["point_cloud"]["lidar_idx"]).zfill(6) for info in infos]
    image_shapes = {frame_id: info["image"]["image_shape"] for frame_id, info in zip(frame_ids, infos)}
    gt_annos = [info["annos"] for info in infos]
    print(f"infos: {len(infos)} frames, classes: {class_names}")

    rows, detector_meta = [], {}
    for specifier in args.prediction:
        detector, prediction_path = specifier.split("=", 1)
        artifact, per_frame = load_detections(prediction_path)
        detector_meta[detector] = {key: artifact.get(key) for key in ("model_config", "checkpoint")}
        detector_meta[detector]["predictions"] = prediction_path
        detector_meta[detector]["num_frames_with_detections"] = len(per_frame)
        dt_annos = build_dt_annos(per_frame, frame_ids, args.calib_dir, image_shapes)
        for class_name in class_names:
            ap = evaluate_class(gt_annos, dt_annos, class_name)
            for difficulty in DIFFICULTY_NAMES:
                rows.append({"detector": detector, "class": class_name, "difficulty": difficulty,
                             "mode": "official_cumulative", "min_iou_3d": CLASS_IOU_3D[class_name],
                             "ap_r40": round(ap[difficulty], 2)})
            print(f"{detector} {class_name}: AP_R40 easy/moderate/hard = "
                  f"{ap['easy']:.2f} / {ap['moderate']:.2f} / {ap['hard']:.2f}")

    output_path = evaluation_dir / "kitti_official_ap.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["detector", "class", "difficulty", "mode",
                                                         "min_iou_3d", "ap_r40"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output_path}")

    protocol = {
        "baseline": "Official KITTI AP_R40 per difficulty (benchmark's own evaluation protocol)",
        "rule_source": "Geiger et al., CVPR 2012 devkit semantics, as implemented by OpenPCDet "
                       "(pcdet/datasets/kitti/kitti_object_eval_python/eval.py)",
        "evaluation_protocol": {
            "metric": "3D AP, 40 recall points (AP_R40), reported overlap row (Car 0.70 / Pedestrian 0.50 / Cyclist 0.50)",
            "ignore_semantics": "for difficulty d, GT objects failing the d-criteria are ignored and "
                                "predictions matching them are not false positives; the three difficulty "
                                "APs are therefore computed on nested, different GT sets (Easy subset of "
                                "Moderate subset of Hard); UnKnown objects ignored at every difficulty",
            "dt_annos_convention": "KittiDataset.generate_prediction_dicts projection (camera frame, "
                                   "image-space bbox, alpha) so results match a test.py evaluation",
            "predictions_note": "full-val JSONs from dcd_baseline/infer_natural.py; all classes kept; "
                                "raw model post-processing output (NMS + config score threshold, min "
                                "observed score 0.1)",
        },
        "detectors": detector_meta,
    }
    protocol_path = args.output_dir / "protocol_official_ap.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {protocol_path}")


if __name__ == "__main__":
    main()
