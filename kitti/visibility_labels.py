"""Convergent-validity check: Definition-3 visibility vs KITTI `occluded` labels.

Run from the repository root:
    ~/miniconda3/envs/openpcdet/bin/python kitti/visibility_labels.py

Mirrors `dairv2x/validate_visibility_labels.py` with the same solid-angle
implementation (`dairv2x.visibility_frame`), so KITTI and DAIR numbers are
comparable. KITTI `occluded` (0 fully visible, 1 partly, 2 largely, 3 unknown)
is image-annotated; objects labelled 3 are dropped as targets but kept as
occluders. DontCare regions are not 3-D objects and are ignored entirely.
Range and log in-box point count are nuisance baselines; the metric is only
interesting if it separates the labels *within* range strata.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_frame import camera_box_corners, camera_to_velo, read_calibration, read_labels
from dairv2x.validate_visibility_labels import BOOTSTRAP, RANGE_BINS, frame_bootstrap_auc
from dairv2x.visibility_frame import solid_angle_visibility
from mixedsig.tools.visualize_agg_frame import points_inside_box

DATA_ROOT = Path("/home/riley/OpenPCDet/data/kitti")
SPLIT = "val"
SAMPLES_PER_EDGE = 32
OUTPUT = Path("kitti/data/visibility_validation")
CLASSES = ("all", "Car", "Pedestrian", "Cyclist")


def lidar_box(obj: dict, r0: np.ndarray, t_velo_cam: np.ndarray) -> np.ndarray:
    """KITTI camera label -> [x, y, z, l, w, h, yaw] in the Velodyne frame.

    Yaw is read from the transformed length edge (corners 0 -> 4) rather than
    the usual -ry - pi/2 shortcut, so calibration rotation is respected.
    """
    corners = camera_to_velo(camera_box_corners(obj), r0, t_velo_cam)
    length_axis = corners[4] - corners[0]
    h, w, length = obj["hwl"]
    return np.r_[corners.mean(0), length, w, h, np.arctan2(length_axis[1], length_axis[0])]


def score_frame(frame_id: str) -> list[dict]:
    root = DATA_ROOT / "training"
    r0, t_velo_cam = read_calibration(root / "calib" / f"{frame_id}.txt")
    objects = [o for o in read_labels(root / "label_2" / f"{frame_id}.txt")
               if o["class"] != "DontCare"]
    if not objects:
        return []
    boxes = np.array([lidar_box(o, r0, t_velo_cam) for o in objects])
    points = np.fromfile(root / "velodyne" / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
    visibility = solid_angle_visibility(boxes, np.zeros(3), SAMPLES_PER_EDGE)
    return [{
        "frame": frame_id, "class": obj["class"], "occluded": obj["occlusion"],
        "truncation": obj["truncation"], "visibility": v,
        "range_m": float(np.linalg.norm(box[:3])),
        "points": int(points_inside_box(points[:, :3], box).sum()),
    } for obj, box, v in zip(objects, boxes, visibility)]


def main() -> None:
    frames = (DATA_ROOT / "ImageSets" / f"{SPLIT}.txt").read_text().split()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, frame_id in enumerate(frames, 1):
        rows += score_frame(frame_id)
        if index % 500 == 0:
            print(f"{index}/{len(frames)} frames", flush=True)
    with (OUTPUT / f"{SPLIT}_objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unknown = sum(r["occluded"] == 3 for r in rows)
    rows = [r for r in rows if r["occluded"] != 3 and np.isfinite(r["visibility"])]
    rng = np.random.default_rng(0)
    lines = ["# Definition-3 visibility vs KITTI occluded", "",
             f"Split: {SPLIT} ({len(frames)} frames); {SAMPLES_PER_EDGE}x{SAMPLES_PER_EDGE} samples per face.",
             f"Targets: {len(rows)} non-DontCare objects; {unknown} with occluded=3 (unknown) dropped.",
             "Positive class: occluded > 0. AUC 0.5 = no separation.",
             f"CIs: {BOOTSTRAP} frame-level bootstrap resamples (frames from one drive are correlated)."]
    for name in CLASSES:
        subset = [r for r in rows if name == "all" or r["class"] == name]
        frame = np.array([r["frame"] for r in subset])
        state = np.array([r["occluded"] for r in subset])
        occlusion = 1 - np.array([r["visibility"] for r in subset])
        distance = np.array([r["range_m"] for r in subset])
        count = np.array([r["points"] for r in subset])
        y = (state > 0).astype(int)
        lines += ["", f"## {name}", "",
                  "| occluded | n | mean V | median V | median range (m) | median points |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in np.unique(state):
            k = state == s
            lines.append(f"| {s} | {k.sum()} | {1 - occlusion[k].mean():.3f} | "
                         f"{1 - np.median(occlusion[k]):.3f} | {np.median(distance[k]):.1f} | "
                         f"{np.median(count[k]):.0f} |")
        lines += ["", "| Stratum | n | AUC 1-V [95% CI] | AUC range | AUC -log1p(points) |",
                  "|---|---:|---:|---:|---:|"]
        strata = [("all", np.ones(len(subset), bool))] + [
            (f"{lo}-{hi} m", (distance >= lo) & (distance < hi)) for lo, hi in RANGE_BINS]
        for stratum, k in strata:
            if not 0 < y[k].sum() < k.sum():
                continue
            low, high = frame_bootstrap_auc(frame[k], y[k], occlusion[k], rng)
            lines.append(f"| {stratum} | {k.sum()} | {roc_auc_score(y[k], occlusion[k]):.3f} "
                         f"[{low:.3f}, {high:.3f}] | {roc_auc_score(y[k], distance[k]):.3f} | "
                         f"{roc_auc_score(y[k], -np.log1p(count[k])):.3f} |")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
