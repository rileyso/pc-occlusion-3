"""Ego PointPillars detections for the same car boxes that main_single.py scores.

Run from the repository root (openpcdet environment, CoAlign checkout):
    PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.main_single_inference

Uses the vehicle-trained single-agent PointPillars checkpoint
(data/checkpoints/pointpillars-late-fusion) through CoAlign's own
`inference_no_fusion`: vehicle LiDAR only, no collaboration. The GT boxes are
main_single.py's selection (cooperative world labels, type "car", same order
and indices), moved into the vehicle LiDAR frame and matched one-to-one to
the predictions by 3D IoU, highest confidence first.

All 46 example pairs are in CoAlign's *training* split, so this checkpoint
has seen these frames; detections here are optimistic.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from dairv2x.coalign import infer, iou_3d, match_boxes
from dairv2x.explorer import DAIRV2XExplorer
from dairv2x.main_single import (
    DATASET_ROOT, PAIR_INDEX, annotation_key, pascal_difficulty,
    visible_face_occupancy,
)
from dairv2x.visibility_frame import boxes_in_sensor_frame, yaw_rotation
from mixedsig.tools.visualize_agg_frame import points_inside_box

CHECKPOINT = Path("data/checkpoints/pointpillars-late-fusion")
COALIGN_ROOT = Path("data/coalign_example")
COALIGN_REPO = Path("/home/riley/CoAlign")
IOU_THRESHOLD = 0.5
DETECTOR_RANGE = (100.8, 40.0)  # |x|, |y| limits of the checkpoint's cav_lidar_range


def box_corners(box):
    """(8, 3) corners, bottom face first in polygon order (as iou_3d expects)."""
    l, w, h = np.asarray(box[3:6]) / 2
    local = np.array([[l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h],
                      [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h]])
    return local @ yaw_rotation(box[6]).T + box[:3]


def main_single_boxes(dataset, pair_index):
    """main_single.py's car boxes (world frame) with label and difficulty."""
    pair = dataset.pairs[pair_index]
    frame_index = dataset.return_labeled_sync_time_ids_of_seq(pair_index)[0]
    labels = dataset._json(pair["cooperative_label_path"])
    vehicle_frame = Path(pair["vehicle_image_path"]).stem
    vehicle_annotations = {annotation_key(label) for label in
                           dataset._json(f"vehicle-side/label/camera/{vehicle_frame}.json")}
    all_boxes = dataset.return_gt_boxes_in_map(pair_index, frame_index)
    cars = [i for i, label in enumerate(labels) if label["type"].lower() == "car"]
    difficulty = [pascal_difficulty(labels[i]) if annotation_key(labels[i]) in vehicle_annotations
                  else None for i in cars]
    return pair, frame_index, all_boxes, cars, [labels[i] for i in cars], difficulty


def occupancy_v(sensor_origin, box, points_inside):
    """main_single.py's visible-face occupancy V at 0.5 m resolution."""
    faces = visible_face_occupancy(sensor_origin, box, points_inside, resolution=0.5)
    total = sum(face["total_cells"] for face in faces)
    return sum(face["occupied_count"] for face in faces) / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=int, default=PAIR_INDEX)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--coalign-root", type=Path, default=COALIGN_ROOT)
    parser.add_argument("--coalign-repo", type=Path, default=COALIGN_REPO)
    parser.add_argument("--output", type=Path, help="optional JSON with one record per box")
    args = parser.parse_args()
    checkpoint, coalign_root = args.checkpoint.resolve(), args.coalign_root.resolve()
    output = args.output.resolve() if args.output else None

    dataset = DAIRV2XExplorer(DATASET_ROOT)
    pair, frame_index, all_boxes, cars, labels, difficulty = main_single_boxes(dataset, args.pair_index)
    frame_id = Path(pair["vehicle_pointcloud_path"]).stem
    pose = dataset.return_map_se3_agent(args.pair_index, "vehicle", frame_index)
    points, _ = dataset.return_agent_point_cloud(args.pair_index, "vehicle", frame_index)
    points_world = (pose @ np.c_[points[:, :3], np.ones(len(points))].T).T[:, :3]

    split = next(s for s in ("val", "train")
                 if frame_id in json.loads((coalign_root / f"{s}.json").read_text()))
    sys.path.insert(0, str(args.coalign_repo.resolve()))
    os.chdir(args.coalign_repo)  # CoAlign resolves some config paths from its checkout
    result = infer(checkpoint, coalign_root, split, frame_id, "single", None, args.iou)
    pred_corners, scores = result["pred_boxes"], result["scores"]

    # GT (world) -> vehicle LiDAR frame, where the predictions are.
    local = boxes_in_sensor_frame(all_boxes[:, :7], pose)
    gt_corners = np.array([box_corners(b) for b in local[cars]]).reshape(-1, 8, 3)
    iou = np.array(iou_3d(gt_corners, pred_corners)).reshape(len(cars), len(scores))
    matches = match_boxes(iou.tolist(), scores, args.iou)

    # Unmatched predictions: do they sit on a labelled non-car (van, truck, ...)?
    others = [i for i in range(len(all_boxes)) if i not in cars]
    other_iou = np.array(iou_3d(np.array([box_corners(b) for b in local[others]]).reshape(-1, 8, 3),
                                pred_corners)).reshape(len(others), len(scores))
    used = {m["prediction_index"] for m in matches if m}
    unmatched = [p for p in range(len(scores)) if p not in used]
    on_other = [p for p in unmatched if len(others) and other_iou[:, p].max() >= args.iou]

    print(f"Pair {args.pair_index}: vehicle frame {frame_id} (CoAlign '{split}' split"
          f"{' - seen in training, results optimistic' if split == 'train' else ''})")
    print(f"Checkpoint {checkpoint.name}, ego LiDAR only; {len(scores)} predictions; "
          f"match = 3D IoU >= {args.iou}, one-to-one, highest confidence first\n")
    header = (f"{'box':>4} {'label':<6}{'difficulty':<11}{'range m':>8}{'points':>7}{'V occ':>7}"
              f"  {'detected':<9}{'confidence':>10}{'IoU 3D':>8}  {'best IoU (conf)':<16}")
    print(header)
    print("-" * len(header))
    records = []
    for i, (label, box) in enumerate(zip(labels, all_boxes[cars])):
        inside = points_inside_box(points_world, box)
        v = occupancy_v(pose[:3, 3], box, points_world[inside])
        distance = float(np.linalg.norm(local[cars[i], :3]))
        in_range = abs(local[cars[i], 0]) <= DETECTOR_RANGE[0] and abs(local[cars[i], 1]) <= DETECTOR_RANGE[1]
        match = matches[i]
        best = int(np.argmax(iou[i])) if len(scores) else None
        best_text = f"{iou[i, best]:.2f} ({scores[best]:.2f})" if best is not None and iou[i, best] > 0 else "-"
        status = "yes" if match else ("no" if in_range else "out-range")
        confidence = f"{match['confidence']:.3f}" if match else "-"
        matched_iou = f"{match['iou_3d']:.2f}" if match else "-"
        print(f"{i:>4} {label['type']:<6}{str(difficulty[i]):<11}{distance:>8.1f}{int(inside.sum()):>7}{v:>7.2f}"
              f"  {status:<9}{confidence:>10}{matched_iou:>8}  {best_text:<16}")
        records.append({"box_index": i, "label": label["type"], "difficulty": difficulty[i],
                        "range_m": distance, "points": int(inside.sum()), "v_occupancy": v,
                        "in_detector_range": bool(in_range), "detected": bool(match),
                        "confidence": match["confidence"] if match else None,
                        "iou_3d": match["iou_3d"] if match else None,
                        "best_iou_3d": float(iou[i, best]) if best is not None else None})
    detected = sum(r["detected"] for r in records)
    print(f"\nDetected {detected}/{len(records)} car boxes; {len(unmatched)} unmatched predictions "
          f"({len(on_other)} of them on labelled non-car vehicles, {len(unmatched) - len(on_other)} on nothing labelled)")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"pair_index": args.pair_index, "vehicle_frame_id": frame_id,
                                      "split": split, "iou_threshold": args.iou, "boxes": records}, indent=2) + "\n")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
