"""Convergent-validity check: Definition-3 visibility vs DAIR occluded_state.

Run: PYTHONPATH=. python -m dairv2x.validate_visibility_labels

Each side's own labels (sensor frame, sensor at the origin) are scored and
compared with that side's human occluded_state (0 none, 1 partial, 2 heavy).
occluded_state is image-annotated and coarse: agreement is evidence the metric
tracks perceived occlusion, not proof of physical correctness. Range and
log in-box point count are reported as nuisance baselines; the metric is only
interesting if it separates the labels *within* range strata.
"""
import csv
import json
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud
from sklearn.metrics import roc_auc_score

from dairv2x.visibility_frame import DATASET_ROOT, solid_angle_visibility
from mixedsig.tools.visualize_agg_frame import points_inside_box

OUTPUT = Path("data/visibility_validation")
SAMPLES_PER_EDGE = 32
RANGE_BINS = ((0, 30), (30, 60), (60, 1000))
BOOTSTRAP = 1000
# Unverified: the RSU "virtual LiDAR" origin is ~1.7 m above the labelled
# ground while the RSU camera sits ~4.7 m higher. RSU scores are very
# sensitive to this height, so they are reported but not trusted yet.
SIDES = {
    "vehicle": ("vehicle-side/label/lidar", "vehicle-side/velodyne"),
    "rsu": ("infrastructure-side/label/virtuallidar", "infrastructure-side/velodyne"),
}


def load_side(root, label_dir, cloud_dir):
    rows, dropped = [], 0
    for path in sorted((root / label_dir).glob("*.json")):
        labels = json.loads(path.read_text())
        boxes = np.array([[l["3d_location"][k] for k in "xyz"]
                          + [l["3d_dimensions"][k] for k in "lwh"]
                          + [float(l["rotation"])] for l in labels], dtype=float).reshape(-1, 7)
        valid = np.isfinite(boxes).all(1) & (boxes[:, 3:6] > 0).all(1)
        dropped += int((~valid).sum())
        labels = [l for l, ok in zip(labels, valid) if ok]
        boxes = boxes[valid]
        if not len(boxes):
            continue
        points = PointCloud.from_path(root / cloud_dir / f"{path.stem}.pcd").numpy(("x", "y", "z"))
        points = points[np.isfinite(points).all(1)]
        visibility = solid_angle_visibility(boxes, np.zeros(3), SAMPLES_PER_EDGE)
        for label, box, v in zip(labels, boxes, visibility):
            rows.append({
                "frame": path.stem, "class": label["type"],
                "occluded_state": int(label["occluded_state"]),
                "visibility": v, "range_m": float(np.linalg.norm(box[:3])),
                "points": int(points_inside_box(points, box).sum()),
            })
    return rows, dropped


def frame_bootstrap_auc(frames, y, score, rng):
    """95% CI resampling whole frames (objects in one frame are not independent)."""
    unique = np.unique(frames)
    index = {f: np.flatnonzero(frames == f) for f in unique}
    values = []
    for _ in range(BOOTSTRAP):
        take = np.concatenate([index[f] for f in rng.choice(unique, len(unique))])
        if 0 < y[take].sum() < len(take):
            values.append(roc_auc_score(y[take], score[take]))
    return np.percentile(values, [2.5, 97.5])


def main():
    root, rng = Path(DATASET_ROOT), np.random.default_rng(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Definition-3 visibility vs DAIR occluded_state", "",
             f"Dataset: `{DATASET_ROOT}`; {SAMPLES_PER_EDGE}x{SAMPLES_PER_EDGE} samples per face.",
             "Positive class: occluded_state > 0. AUC 0.5 = no separation.",
             f"CIs: {BOOTSTRAP} frame-level bootstrap resamples (adjacent frames are still correlated)."]
    for side, (label_dir, cloud_dir) in SIDES.items():
        rows, dropped = load_side(root, label_dir, cloud_dir)
        rows = [r for r in rows if np.isfinite(r["visibility"])]
        with (OUTPUT / f"{side}_objects.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        frames = np.array([r["frame"] for r in rows])
        state = np.array([r["occluded_state"] for r in rows])
        occlusion = 1 - np.array([r["visibility"] for r in rows])
        distance = np.array([r["range_m"] for r in rows])
        sparsity = -np.log1p([r["points"] for r in rows])
        y = (state > 0).astype(int)
        lines += ["", f"## {side.upper()}", "",
                  f"Objects: {len(rows)} in {len(np.unique(frames))} frames; "
                  f"degenerate boxes dropped: {dropped}.", "",
                  "| occluded_state | n | mean V | median V | median range (m) | median points |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in np.unique(state):
            k = state == s
            lines.append(f"| {s} | {k.sum()} | {1 - occlusion[k].mean():.3f} | "
                         f"{1 - np.median(occlusion[k]):.3f} | {np.median(distance[k]):.1f} | "
                         f"{np.median([r['points'] for r, m in zip(rows, k) if m]):.0f} |")
        lines += ["", "| Stratum | n | AUC 1-V [95% CI] | AUC range | AUC -log1p(points) |",
                  "|---|---:|---:|---:|---:|"]
        strata = [("all", np.ones(len(rows), bool))] + [
            (f"{lo}-{hi} m", (distance >= lo) & (distance < hi)) for lo, hi in RANGE_BINS]
        for name, k in strata:
            if not 0 < y[k].sum() < k.sum():
                continue
            low, high = frame_bootstrap_auc(frames[k], y[k], occlusion[k], rng)
            lines.append(f"| {name} | {k.sum()} | {roc_auc_score(y[k], occlusion[k]):.3f} "
                         f"[{low:.3f}, {high:.3f}] | {roc_auc_score(y[k], distance[k]):.3f} | "
                         f"{roc_auc_score(y[k], sparsity[k]):.3f} |")
    lines += ["", "RSU results depend on the unverified virtual-LiDAR origin height; "
              "do not interpret them until that is resolved."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
