#!/usr/bin/env python
"""Build the natural KITTI geometry baseline for a Gilroy-style LiDAR study.

Occlusion is the area-weighted fraction of the target's LiDAR-facing 3D box
surface whose rays hit another annotated 3D box first. This is objective but
only sees annotated object occluders; it does not infer occlusion from point
count or distance.
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from pcdet.utils import box_utils, calibration_kitti, object3d_kitti


COLUMNS = (
    "frame_id", "label_index", "class_name", "occlusion_bin", "occlusion_percent",
    "head_visible_percent", "ground_range_m", "radial_range_m", "bearing_deg",
    "num_points_in_gt", "point_density_m3", "x", "y", "z", "dx", "dy", "dz",
    "heading", "kitti_occlusion", "kitti_truncation", "kitti_difficulty",
)


def rotation(box):
    cosine, sine = math.cos(float(box[6])), math.sin(float(box[6]))
    return np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def surface_samples(box, resolution=12, head_fraction=0.2):
    """Sample LiDAR-facing box surfaces and return points, area weights, head mask."""
    if resolution < 2 or not 0 < head_fraction <= 1:
        raise ValueError("resolution must be >= 2 and head_fraction must be in (0, 1]")
    box = np.asarray(box, dtype=float)
    dimensions = box[3:6]
    if np.any(dimensions <= 0):
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=bool)
    rot = rotation(box)
    grid = (np.arange(resolution) + 0.5) / resolution - 0.5
    faces = []
    for axis in range(3):
        others = [value for value in range(3) if value != axis]
        area = dimensions[others[0]] * dimensions[others[1]]
        for sign in (-1, 1):
            normal = np.zeros(3)
            normal[axis] = sign
            if float((normal @ rot.T) @ -box[:3]) <= 0:
                continue
            local = np.zeros((resolution * resolution, 3))
            local[:, axis] = sign * dimensions[axis] / 2
            u, v = np.meshgrid(grid * dimensions[others[0]], grid * dimensions[others[1]])
            local[:, others[0]], local[:, others[1]] = u.ravel(), v.ravel()
            world = local @ rot.T + box[:3]
            weights = np.full(len(world), area / len(world))
            head = local[:, 2] >= dimensions[2] * (0.5 - head_fraction)
            faces.append((world, weights, head))
    if not faces:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=bool)
    return tuple(np.concatenate(values) for values in zip(*faces))


def rays_hit_box_before(points, box, epsilon=1e-3):
    """Return which origin-to-point rays enter an oriented box before the point."""
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points, axis=1)
    directions = points / np.maximum(distances[:, None], np.finfo(float).eps)
    rot = rotation(box)
    local_origin = -np.asarray(box[:3], dtype=float) @ rot
    local_directions = directions @ rot
    half = np.asarray(box[3:6], dtype=float) / 2
    near = np.empty_like(local_directions)
    far = np.empty_like(local_directions)
    for axis in range(3):
        direction = local_directions[:, axis]
        moving = np.abs(direction) > 1e-12
        near[:, axis], far[:, axis] = -np.inf, np.inf
        first = (-half[axis] - local_origin[axis]) / direction[moving]
        second = (half[axis] - local_origin[axis]) / direction[moving]
        near[moving, axis] = np.minimum(first, second)
        far[moving, axis] = np.maximum(first, second)
        if abs(local_origin[axis]) > half[axis]:
            near[~moving, axis], far[~moving, axis] = np.inf, -np.inf
    entry, exit_ = near.max(axis=1), far.min(axis=1)
    return (exit_ >= np.maximum(entry, 0.0)) & (entry < distances - epsilon) & (exit_ > epsilon)


def box_visibility(target_box, occluder_boxes, resolution=12, head_fraction=0.2):
    points, weights, head = surface_samples(target_box, resolution, head_fraction)
    occluded = np.zeros(len(points), dtype=bool)
    for box in occluder_boxes:
        occluded |= rays_hit_box_before(points, box)
    total = weights.sum()
    occlusion = 100.0 * weights[occluded].sum() / total if total else 0.0
    head_total = weights[head].sum()
    head_visible = 100.0 * weights[head & ~occluded].sum() / head_total if head_total else float("nan")
    return min(occlusion, 99.999), head_visible


def points_in_box(points, box):
    local = (np.asarray(points)[:, :3] - box[:3]) @ rotation(box)
    return np.all(np.abs(local) <= np.asarray(box[3:6]) / 2 + 1e-6, axis=1)


def load_frame(kitti_root, frame_id):
    base = Path(kitti_root) / "training"
    objects = object3d_kitti.get_objects_from_label(base / "label_2" / f"{frame_id}.txt")
    valid = [
        (index, obj) for index, obj in enumerate(objects)
        if obj.cls_type != "DontCare" and obj.l > 0 and obj.w > 0 and obj.h > 0
    ]
    calibration = calibration_kitti.Calibration(base / "calib" / f"{frame_id}.txt")
    camera_boxes = np.asarray(
        [[obj.loc[0], obj.loc[1], obj.loc[2], obj.l, obj.h, obj.w, obj.ry] for _, obj in valid],
        dtype=np.float32,
    )
    lidar_boxes = box_utils.boxes3d_kitti_camera_to_lidar(camera_boxes, calibration)
    points = np.fromfile(base / "velodyne" / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
    return valid, lidar_boxes, points


def build_rows(kitti_root, frame_ids, target_class="Pedestrian", resolution=12, head_fraction=0.2):
    rows = []
    for frame_id in frame_ids:
        valid, boxes, points = load_frame(kitti_root, frame_id)
        for target_position, ((label_index, obj), target_box) in enumerate(zip(valid, boxes)):
            if obj.cls_type != target_class:
                continue
            occluders = np.delete(boxes, target_position, axis=0)
            occlusion, head_visible = box_visibility(target_box, occluders, resolution, head_fraction)
            count = int(points_in_box(points, target_box).sum())
            volume = max(float(np.prod(target_box[3:6])), 1e-6)
            rows.append({
                "frame_id": frame_id,
                "label_index": label_index,
                "class_name": obj.cls_type,
                "occlusion_bin": min(int(occlusion // 10), 9),
                "occlusion_percent": occlusion,
                "head_visible_percent": head_visible,
                "ground_range_m": float(np.hypot(target_box[0], target_box[1])),
                "radial_range_m": float(np.linalg.norm(target_box[:3])),
                "bearing_deg": float(np.degrees(np.arctan2(target_box[1], target_box[0]))),
                "num_points_in_gt": count,
                "point_density_m3": count / volume,
                **dict(zip(("x", "y", "z", "dx", "dy", "dz", "heading"), map(float, target_box))),
                "kitti_occlusion": int(obj.occlusion),
                "kitti_truncation": float(obj.truncation),
                "kitti_difficulty": obj.level_str,
            })
    return rows


def save_outputs(rows, output_dir, split, minimum_per_bin, resolution, head_fraction):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instances_path = output_dir / "instances.csv"
    with instances_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["occlusion_bin"] for row in rows)
    summary = []
    for index in range(10):
        count = counts[index]
        summary.append({"bin": index, "range": f"{index * 10}-{index * 10 + 9}%", "instances": count, "meets_minimum": count >= minimum_per_bin})
    with (output_dir / "bin_summary.csv").open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("bin", "range", "instances", "meets_minimum"))
        writer.writeheader()
        writer.writerows(summary)
    protocol = {
        "split": split,
        "target_class": rows[0]["class_name"] if rows else "Pedestrian",
        "occlusion_definition": "area-weighted fraction of LiDAR-facing target 3D-box surface shadowed by another annotated 3D box",
        "bins": [item["range"] for item in summary],
        "minimum_instances_per_bin": minimum_per_bin,
        "surface_resolution_per_face_axis": resolution,
        "head_proxy": f"upper {head_fraction:.0%} of LiDAR-facing 3D-box surface",
        "all_bins_ready": all(item["meets_minimum"] for item in summary),
        "limitations": ["annotated 3D-box occluders only", "box surface is a proxy for pedestrian body surface"],
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return instances_path, summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--target-class", default="Pedestrian")
    parser.add_argument("--surface-resolution", type=int, default=12)
    parser.add_argument("--head-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-per-bin", type=int, default=55)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gilroy_lidar_replication/natural"))
    return parser.parse_args()


def main():
    args = parse_args()
    split_path = args.kitti_root / "ImageSets" / f"{args.split}.txt"
    frame_ids = split_path.read_text(encoding="utf-8").split()
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]
    rows = build_rows(args.kitti_root, frame_ids, args.target_class, args.surface_resolution, args.head_fraction)
    path, summary = save_outputs(rows, args.output_dir, args.split, args.minimum_per_bin, args.surface_resolution, args.head_fraction)
    print(f"WROTE {path} ({len(rows)} instances)")
    for item in summary:
        print(f"{item['range']}: {item['instances']} {'OK' if item['meets_minimum'] else 'NEEDS MORE'}")


if __name__ == "__main__":
    main()
