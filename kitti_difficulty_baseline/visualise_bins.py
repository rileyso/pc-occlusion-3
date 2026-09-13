#!/usr/bin/env python
"""Ego-POV BEV gallery: what each KITTI difficulty bin looks like in the LiDAR.

One row, one column per bin (Easy / Moderate / Hard / UnKnown from the
grouping registry). Each panel overlays, in ego-vehicle coordinates (forward
= up, left = left), the aggregate LiDAR scan of a seeded sample of val frames
in grey and the bin's objects in colour: rotated GT box outline plus the
LiDAR returns inside the box. The per-bin point clouds make the baseline's
quantitative claim visible - Easy objects are close and dense, Hard objects
sparse, and UnKnown objects are mostly image-truncated rather than
observation-poor.

Reuses gilroy_lidar_replication.build_benchmark.load_frame / points_in_box /
rotation (the same object-to-lidar projection and inside-box test as every
other thesis artifact), and groupings.py for the bin order and display names,
so the gallery follows any grouping the registry grows.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from gilroy_lidar_replication.build_benchmark import load_frame, points_in_box  # noqa: E402
from kitti_difficulty_baseline import groupings as grouping_registry  # noqa: E402

BIN_COLORS = {"Easy": "#009988", "Moderate": "#EE7733", "Hard": "#CC3311", "UnKnown": "#777777",
              "0": "#009988", "1": "#EE7733", "2": "#CC3311", "3": "#777777"}
EGO_COLOR = "#111111"


def box_corners_2d(box):
    """(4, 2) BEV corners of a lidar-frame box [x, y, z, dx, dy, dz, heading]."""
    rotation = np.asarray([[np.cos(box[6]), -np.sin(box[6])],
                           [np.sin(box[6]), np.cos(box[6])]])
    half = np.asarray([box[3], box[4]]) / 2.0
    signs = np.asarray([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    return box[:2] + (signs * half) @ rotation.T


def val_label_totals(kitti_root, split, target_class, valid_values):
    """Object counts per bin over the whole split (label files only, no points),
    plus the frames that contain at least one target object."""
    from pcdet.utils import object3d_kitti
    frame_ids = (Path(kitti_root) / "ImageSets" / f"{split}.txt").read_text(encoding="utf-8").split()
    totals = {value: 0 for value in valid_values}
    frames_with_target = []
    label_dir = Path(kitti_root) / "training" / "label_2"
    for frame_id in frame_ids:
        objects = object3d_kitti.get_objects_from_label(label_dir / f"{frame_id}.txt")
        has_target = False
        for obj in objects:
            if obj.cls_type == target_class and obj.level_str in totals:
                totals[obj.level_str] += 1
                has_target = True
        if has_target:
            frames_with_target.append(frame_id)
    return totals, frames_with_target


def collect(kitti_root, split, args, candidate_frames):
    """Sample frames once; return (background scan, per-bin object records)."""
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(candidate_frames, size=min(args.num_frames, len(candidate_frames)), replace=False))
    grouping = grouping_registry.get_grouping(args.grouping)
    order = [value for value in grouping.order]

    background, objects = [], {value: [] for value in order}
    for frame_id in chosen:
        valid, boxes, points = load_frame(args.kitti_root, frame_id)
        height = (points[:, 2] >= args.z_min) & (points[:, 2] <= args.z_max)
        background_scan = points[height][::args.point_stride, :2]
        background.append(background_scan[np.linalg.norm(background_scan, axis=1) >= args.ego_crop_radius])
        for (label_index, obj), box in zip(valid, boxes):
            if obj.cls_type != args.target_class or obj.level_str not in objects:
                continue
            if len(objects[obj.level_str]) >= args.max_objects_per_bin:
                continue
            box7 = np.asarray(box, dtype=np.float32)
            inside = points[points_in_box(points, box7)]
            objects[obj.level_str].append({
                "frame_id": frame_id,
                "corners": box_corners_2d(box7),
                "points": inside[:, :2],
                "num_points": int(len(inside)),
                "range_m": float(np.hypot(box7[0], box7[1])),
            })
    background, objects, num_frames = np.concatenate(background), objects, len(chosen)
    if len(background) > args.max_background_points:
        keep = rng.choice(len(background), size=args.max_background_points, replace=False)
        background = background[keep]
    return background, objects, len(chosen)


def draw_panel(ax, objects, total, color, forward_range, lateral_range):
    interior = [entry["points"] for entry in objects if len(entry["points"])]
    if interior:
        points = np.concatenate(interior)
        ax.scatter(points[:, 1], points[:, 0], s=0.5, color=color, alpha=0.85,
                   linewidths=0, rasterized=True, zorder=3)
    lines = [[(corner[1], corner[0]) for corner in entry["corners"]]
             + [(entry["corners"][0][1], entry["corners"][0][0])] for entry in objects]
    ax.add_collection(LineCollection(lines, colors=color, linewidths=0.9, alpha=0.9, zorder=4))

    ax.scatter([0], [0], marker="^", s=50, color=EGO_COLOR, zorder=6)
    bar_y, bar_x = -lateral_range * 0.88, -0.15 * forward_range + 3.0
    ax.plot([bar_y, bar_y + 10.0], [bar_x, bar_x], color=EGO_COLOR, linewidth=1.4, zorder=6)
    ax.text(bar_y + 5.0, bar_x + 1.6, "10 m", ha="center", fontsize=7, color=EGO_COLOR, zorder=6)

    shown = len(objects)
    mean_points = float(np.mean([entry["num_points"] for entry in objects])) if objects else float("nan")
    mean_range = float(np.mean([entry["range_m"] for entry in objects])) if objects else float("nan")
    ax.text(0.025, 0.975, f"{shown} shown of {total:,} in val\n"
                         f"mean {mean_points:.0f} points, {mean_range:.1f} m",
            transform=ax.transAxes, va="top", fontsize=7.5, color="#222222",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5}, zorder=7)
    ax.set_xlim(lateral_range, -lateral_range)  # inverted: left of scene on the left
    ax.set_ylim(-0.15 * forward_range, forward_range)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-class", default="Car")
    parser.add_argument("--grouping", default="kitti_difficulty")
    parser.add_argument("--kitti-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-frames", type=int, default=160)
    parser.add_argument("--max-objects-per-bin", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forward-range", type=float, default=60.0)
    parser.add_argument("--lateral-range", type=float, default=28.0)
    parser.add_argument("--z-min", type=float, default=-2.2)
    parser.add_argument("--z-max", type=float, default=0.6)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--max-background-points", type=int, default=400_000)
    parser.add_argument("--ego-crop-radius", type=float, default=3.5,
                        help="hide host-vehicle returns within this radius of the origin")
    parser.add_argument("--sample-all-frames", action="store_true",
                        help="sample from every split frame instead of only frames containing the target class")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research_workspace/kitti_difficulty_baseline/results/plots"))
    return parser.parse_args()


def main():
    args = parse_args()
    grouping = grouping_registry.get_grouping(args.grouping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"counting {args.target_class} objects in all {args.split} labels ...")
    totals, frames_with_target = val_label_totals(args.kitti_root, args.split, args.target_class, grouping.order)
    print("  totals:", {key: value for key, value in totals.items()})
    all_frames = (args.kitti_root / "ImageSets" / f"{args.split}.txt").read_text(encoding="utf-8").split()
    candidate_frames = all_frames if args.sample_all_frames or not frames_with_target else frames_with_target

    background, objects, num_frames = collect(args.kitti_root, args.split, args, candidate_frames)
    print(f"frames sampled: {num_frames} of {len(candidate_frames)} candidates | objects collected: "
          f"{ {key: len(value) for key, value in objects.items()} }")

    figure, axes = plt.subplots(1, len(grouping.order), figsize=(3.4 * len(grouping.order), 5.0),
                                constrained_layout=True)
    for ax, value in zip(np.atleast_1d(axes), grouping.order):
        color = BIN_COLORS.get(value, "#4477AA")
        ax.scatter(background[:, 1], background[:, 0], s=0.15, color="#9aa0a6",
                   alpha=0.35, linewidths=0, rasterized=True, zorder=1)
        draw_panel(ax, objects[value], totals[value], color, args.forward_range, args.lateral_range)
        ax.set_title(grouping.display.get(value, value), fontsize=12, fontweight="bold", pad=8)
    frame_note = ("frames containing the class" if candidate_frames is not all_frames
                  else "all frames")
    grouping_title = grouping.name.replace("_", " ").replace("kitti", "KITTI")
    figure.suptitle(f"KITTI val {args.target_class}s by {grouping_title} bin — "
                    f"aggregate BEV of {num_frames} seeded {frame_note} "
                    f"(ego at origin, forward = up, returns inside GT boxes in colour)", fontsize=11)
    output = args.output_dir / f"fig_difficulty_bev_gallery_{args.target_class.lower()}.png"
    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
