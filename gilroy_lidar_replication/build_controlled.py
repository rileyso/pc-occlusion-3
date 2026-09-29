#!/usr/bin/env python
"""Create the paired 820-incident controlled-occlusion manifest.

The 82 base Pedestrians are reused at ten occlusion levels. Structured and
random-drop arms retain exactly the same number of target-box points. Point
clouds are transformed on demand by infer_controlled.py; they are not copied.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gilroy_lidar_replication.build_benchmark import load_frame, points_in_box, rotation, surface_samples


DIRECTIONS = ("bottom", "top", "left", "right")
LEVELS = tuple(range(5, 100, 10))
BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
SAMPLE_FIELDS = (
    "sample_id", "source_frame_id", "label_index", "arm", "occlusion_bin",
    "target_occlusion_percent", "achieved_surface_occlusion_percent", "head_visible_percent",
    "direction", "seed", "original_target_points", "removed_target_points",
    "retained_target_points", "ground_range_m", "ignore_boxes_json", *BOX_FIELDS,
)


def scalar_coordinate(local_points, box, direction):
    """Coordinate increasing from the selected occluding edge across the target."""
    if direction in ("bottom", "top"):
        values = local_points[:, 2] / box[5] + 0.5
        return values if direction == "bottom" else 1.0 - values
    values = local_points[:, 1] / box[4] + 0.5
    return values if direction == "left" else 1.0 - values


def structured_cutoff(box, fraction, direction, resolution=30, head_fraction=0.2):
    surface, weights, head = surface_samples(box, resolution, head_fraction)
    local = (surface - box[:3]) @ rotation(box)
    values = scalar_coordinate(local, box, direction)
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    position = min(np.searchsorted(cumulative, fraction, side="left"), len(order) - 1)
    cutoff = float(values[order[position]])
    removed = values <= cutoff
    achieved = float(weights[removed].sum() / weights.sum())
    head_total = weights[head].sum()
    head_visible = float(weights[head & ~removed].sum() / head_total) if head_total else float("nan")
    return cutoff, achieved, head_visible


def apply_control(points, box, fraction, direction, arm, seed, resolution=30):
    """Apply structured removal or an equal-count random target-point removal."""
    points = np.asarray(points)
    target_indices = np.flatnonzero(points_in_box(points, box))
    local = (points[target_indices, :3] - box[:3]) @ rotation(box)
    cutoff, achieved, head_visible = structured_cutoff(box, fraction, direction, resolution)
    structured = target_indices[scalar_coordinate(local, box, direction) <= cutoff]
    remove_count = min(len(structured), len(target_indices))
    if arm == "structured":
        removed = structured
    elif arm == "random":
        removed = np.random.default_rng(seed).choice(target_indices, remove_count, replace=False)
    else:
        raise ValueError("arm must be structured or random")
    keep = np.ones(len(points), dtype=bool)
    keep[removed] = False
    return points[keep], {
        "original_target_points": len(target_indices),
        "removed_target_points": remove_count,
        "retained_target_points": len(target_indices) - remove_count,
        "achieved_surface_occlusion_percent": 100.0 * achieved,
        "head_visible_percent": 100.0 * head_visible,
    }


def load_candidates(path, minimum_points=30):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    candidates = []
    seen_frames = set()
    for row in sorted(rows, key=lambda item: (float(item["ground_range_m"]), item["frame_id"], int(item["label_index"]))):
        if float(row["occlusion_percent"]) >= 10 or int(row["num_points_in_gt"]) < minimum_points or row["frame_id"] in seen_frames:
            continue
        row["box"] = np.asarray([row[name] for name in BOX_FIELDS], dtype=np.float32)
        candidates.append(row)
        seen_frames.add(row["frame_id"])
    return candidates


def select_bases(candidates, count=82):
    if len(candidates) < count:
        raise ValueError(f"Need {count} clear, unique-frame base Pedestrians; found {len(candidates)}")
    return [candidates[index] for index in np.linspace(0, len(candidates) - 1, count, dtype=int)]


def build_samples(kitti_root, bases, resolution=30):
    samples = []
    for base_index, base in enumerate(bases):
        frame_id, box = base["frame_id"], base["box"]
        valid, boxes, points = load_frame(Path(kitti_root), frame_id)
        ignore_boxes = [
            candidate.tolist() for (label_index, obj), candidate in zip(valid, boxes)
            if obj.cls_type == "Pedestrian" and label_index != int(base["label_index"])
        ]
        direction = DIRECTIONS[base_index % len(DIRECTIONS)]
        for bin_index, percent in enumerate(LEVELS):
            seed = base_index * 100 + bin_index
            for arm in ("structured", "random"):
                _, details = apply_control(points, box, percent / 100.0, direction, arm, seed, resolution)
                sample_id = f"{frame_id}_{int(base['label_index']):02d}_{bin_index}_{arm[0]}"
                samples.append({
                    "sample_id": sample_id,
                    "source_frame_id": frame_id,
                    "label_index": int(base["label_index"]),
                    "arm": arm,
                    "occlusion_bin": bin_index,
                    "target_occlusion_percent": percent,
                    "direction": direction,
                    "seed": seed,
                    "ground_range_m": float(base["ground_range_m"]),
                    "ignore_boxes_json": json.dumps(ignore_boxes, separators=(",", ":")),
                    **details,
                    **{name: float(base[name]) for name in BOX_FIELDS},
                })
    return samples


def save_manifest(samples, output_dir, base_count, resolution):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "samples.csv"
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(samples)
    protocol = {
        "base_pedestrians": base_count,
        "levels_percent": list(LEVELS),
        "structured_incidents": base_count * len(LEVELS),
        "random_drop_controls": base_count * len(LEVELS),
        "surface_resolution_per_face_axis": resolution,
        "pairing": "same frame, target, distance, pose, and retained target-point count",
        "structured_directions": list(DIRECTIONS),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=Path("runs/gilroy_lidar_replication/natural/val/instances.csv"))
    parser.add_argument("--kitti-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--base-count", type=int, default=82)
    parser.add_argument("--minimum-points", type=int, default=30)
    parser.add_argument("--surface-resolution", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gilroy_lidar_replication/controlled"))
    args = parser.parse_args()
    bases = select_bases(load_candidates(args.instances, args.minimum_points), args.base_count)
    samples = build_samples(args.kitti_root, bases, args.surface_resolution)
    path = save_manifest(samples, args.output_dir, len(bases), args.surface_resolution)
    print(f"WROTE {path}: {len(bases) * 10} structured incidents + {len(bases) * 10} matched random controls")


if __name__ == "__main__":
    main()
