"""Score every DAIR box with Definition-3 visibility and print the AUC table.

Run: PYTHONPATH=. python -m dairv2x.visibility_auc

For each sensor side (vehicle, RSU):
  1. load that side's labelled boxes (sensor frame, sensor at the origin)
  2. compute visibility V per box (labelled boxes only, no point cloud)
  3. count LiDAR points inside each box (baseline only)
  4. AUC: how well each score separates occluded_state > 0 from = 0,
     overall and within 0-30 / 30-60 / 60+ m

AUC = chance a random occluded box scores higher than a random visible one
(0.5 = no better than chance). Scores: occlusion 1 - V, distance, and
-log1p(points) (fewer points = "more occluded").
"""
import json
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud
from sklearn.metrics import roc_auc_score

from dairv2x.visibility_frame import DATASET_ROOT, solid_angle_visibility
from mixedsig.tools.visualize_agg_frame import points_inside_box

SAMPLES_PER_EDGE = 32
RANGE_BINS = (("All", 0, np.inf), ("0-30 m", 0, 30), ("30-60 m", 30, 60), ("60+ m", 60, np.inf))
SIDES = {
    "Vehicle": ("vehicle-side/label/lidar", "vehicle-side/velodyne"),
    "RSU": ("infrastructure-side/label/virtuallidar", "infrastructure-side/velodyne"),
}


def iter_side_frames(root, label_dir, cloud_dir):
    """Yield (labels, boxes, cloud) per frame, all in this sensor's frame.

    Boxes are [x, y, z, l, w, h, yaw]; degenerate labels are dropped.
    """
    for label_path in sorted((root / label_dir).glob("*.json")):
        labels = json.loads(label_path.read_text())
        boxes = np.array([[l["3d_location"][k] for k in "xyz"]
                          + [l["3d_dimensions"][k] for k in "lwh"]
                          + [float(l["rotation"])] for l in labels], dtype=float).reshape(-1, 7)
        keep = np.isfinite(boxes).all(1) & (boxes[:, 3:6] > 0).all(1)
        boxes, labels = boxes[keep], [l for l, k in zip(labels, keep) if k]
        if not len(boxes):
            continue
        cloud = PointCloud.from_path(root / cloud_dir / f"{label_path.stem}.pcd").numpy(("x", "y", "z"))
        yield labels, boxes, cloud[np.isfinite(cloud).all(1)]


def score_side(root, label_dir, cloud_dir):
    """Return per-box arrays: occluded label, visibility, distance, points."""
    occluded, visibility, distance, points = [], [], [], []
    for labels, boxes, cloud in iter_side_frames(root, label_dir, cloud_dir):
        # Definition-3 visibility, sensor at the origin of its own frame.
        visibility += list(solid_angle_visibility(boxes, np.zeros(3), SAMPLES_PER_EDGE))
        # Point-count baseline from the matching point cloud.
        points += [int(points_inside_box(cloud, box).sum()) for box in boxes]
        distance += list(np.linalg.norm(boxes[:, :3], axis=1))
        occluded += [int(l["occluded_state"]) > 0 for l in labels]
    occluded, visibility, distance, points = map(np.array, (occluded, visibility, distance, points))
    valid = np.isfinite(visibility)  # NaN = sensor inside the box
    return occluded[valid], visibility[valid], distance[valid], points[valid]


def auc(y, score):
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan


def main():
    root = Path(DATASET_ROOT)
    for side, (label_dir, cloud_dir) in SIDES.items():
        occluded, visibility, distance, points = score_side(root, label_dir, cloud_dir)
        print(f"\n{side} side: {len(occluded)} boxes, {int(occluded.sum())} labelled occluded")
        print(f"{'Range':<9}{'n':>6}{'Visibility (1-V)':>18}{'Distance':>10}{'Points':>8}")
        print("-" * 51)
        # Step 4: AUC per range bin.
        for name, low, high in RANGE_BINS:
            k = (distance >= low) & (distance < high)
            y = occluded[k]
            print(f"{name:<9}{k.sum():>6}{auc(y, 1 - visibility[k]):>18.3f}"
                  f"{auc(y, distance[k]):>10.3f}{auc(y, -np.log1p(points[k])):>8.3f}")
    print("\nnan = bin has only one label class. RSU values depend on the "
          "unverified RSU origin height.")


if __name__ == "__main__":
    main()
