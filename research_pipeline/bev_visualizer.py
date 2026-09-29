#!/usr/bin/env python
"""Small, dependency-light bird's-eye-view LiDAR visualizer.

Boxes use OpenPCDet's canonical layout::

    [x, y, z, dx, dy, dz, heading, ...]

Example::

    from research_pipeline.bev_visualizer import save_bev

    save_bev("bev.png", points, gt_boxes=gt_boxes, pred_boxes=pred_boxes)

The command-line interface accepts ``.npy`` arrays and raw float32 ``.bin``
point clouds::

    python research_pipeline/bev_visualizer.py points.bin \
        --point-features 4 --pred-boxes predictions.npy --output bev.png
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openpcdet-matplotlib-cache")
import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


GT_COLOR = "#3182bd"
PRED_COLOR = "#31a354"


def as_array(value, columns, name):
    """Convert NumPy or Torch-like input into a validated two-dimensional array."""
    if value is None:
        return np.empty((0, columns), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < columns:
        raise ValueError(f"{name} must have shape (N, {columns}+), got {array.shape}")
    return array


def box_bev_corners(boxes):
    """Return ``(N, 4, 2)`` corners for OpenPCDet boxes."""
    boxes = as_array(boxes, 7, "boxes")
    if not len(boxes):
        return np.empty((0, 4, 2), dtype=np.float32)

    template = np.asarray(
        [[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]],
        dtype=np.float32,
    )
    local = boxes[:, None, 3:5] * template[None, :, :]
    cosine, sine = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    corners = np.empty_like(local)
    corners[:, :, 0] = local[:, :, 0] * cosine[:, None] - local[:, :, 1] * sine[:, None]
    corners[:, :, 1] = local[:, :, 0] * sine[:, None] + local[:, :, 1] * cosine[:, None]
    return corners + boxes[:, None, :2]


def draw_boxes(
    ax, boxes, color, linestyle="-", scores=None, show_scores=False, label=None, ego_view=False,
):
    """Draw rotated box footprints and short heading indicators."""
    boxes = as_array(boxes, 7, "boxes")
    if not len(boxes):
        return
    if scores is not None:
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(boxes):
            raise ValueError("scores must contain one value per box")

    corners = box_bev_corners(boxes)
    for index, (box, footprint) in enumerate(zip(boxes, corners)):
        closed = np.vstack([footprint, footprint[0]])
        horizontal, vertical = (closed[:, 1], closed[:, 0]) if ego_view else (closed[:, 0], closed[:, 1])
        ax.plot(
            horizontal, vertical, color=color, linewidth=1.6,
            linestyle=linestyle, label=label if index == 0 else None,
        )
        heading_end = box[:2] + 0.5 * box[3] * np.asarray([np.cos(box[6]), np.sin(box[6])])
        heading_x, heading_y = ([box[1], heading_end[1]], [box[0], heading_end[0]]) if ego_view else ([box[0], heading_end[0]], [box[1], heading_end[1]])
        ax.plot(heading_x, heading_y, color=color, linewidth=1.4)
        if show_scores and scores is not None:
            text_x, text_y = (box[1], box[0]) if ego_view else (box[0], box[1])
            ax.text(text_x, text_y, f"{scores[index]:.2f}", color=color, fontsize=7)


def apply_auto_limits(ax, points, box_groups, xlim, ylim, margin=3.0, ego_view=False):
    xy_values = []
    if len(points):
        xy_values.append(points[:, :2])
    for boxes in box_groups:
        corners = box_bev_corners(boxes)
        if len(corners):
            xy_values.append(corners.reshape(-1, 2))

    combined = np.concatenate(xy_values, axis=0) if xy_values else np.asarray([[-10.0, -10.0], [10.0, 10.0]])
    if xlim is None:
        xlim = (float(combined[:, 0].min() - margin), float(combined[:, 0].max() + margin))
    if ylim is None:
        ylim = (float(combined[:, 1].min() - margin), float(combined[:, 1].max() + margin))
    if ego_view:
        ax.set_xlim(*ylim)
        ax.set_ylim(*xlim)
    else:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


def point_color_values(points, color_by):
    """Return point colors and the corresponding colorbar label."""
    if color_by == "height":
        return (points[:, 2], "Height (m)") if points.shape[1] >= 3 else ("#4d4d4d", None)
    ground_range = np.hypot(points[:, 0], points[:, 1])
    if color_by == "ground_range":
        return ground_range, "Ground range (m)"
    if color_by == "radial_range":
        return (np.linalg.norm(points[:, :3], axis=1), "Radial range (m)") if points.shape[1] >= 3 else (ground_range, "Ground range (m)")
    raise ValueError("color_by must be height, ground_range, or radial_range")


def draw_range_rings(ax, ranges):
    for value in ranges:
        ax.add_patch(plt.Circle((0.0, 0.0), value, color="#6b7280", fill=False, linewidth=0.5, alpha=0.35))


def plot_bev(
    points,
    gt_boxes=None,
    pred_boxes=None,
    pred_scores=None,
    score_threshold=0.0,
    xlim=None,
    ylim=None,
    max_points=100_000,
    point_size=0.6,
    title="Bird's-eye view",
    show_scores=False,
    ax=None,
    color_by="height",
    color_limits=None,
    show_colorbar=True,
    show_legend=True,
    source_label=None,
    ego_view=False,
    show_range_rings=False,
):
    """Plot LiDAR points and optional GT/predicted boxes; return ``(figure, axes)``."""
    points = as_array(points, 2, "points")
    gt_boxes = as_array(gt_boxes, 7, "gt_boxes")
    pred_boxes = as_array(pred_boxes, 7, "pred_boxes")

    if pred_scores is not None:
        pred_scores = np.asarray(pred_scores).reshape(-1)
        if len(pred_scores) != len(pred_boxes):
            raise ValueError("pred_scores must contain one value per predicted box")
        keep = pred_scores >= float(score_threshold)
        pred_boxes, pred_scores = pred_boxes[keep], pred_scores[keep]
    elif score_threshold > 0 and len(pred_boxes):
        raise ValueError("score_threshold requires pred_scores")

    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be positive or None")
    if max_points is not None and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]

    if ego_view:
        xlim = (0.0, 70.0) if xlim is None else xlim
        ylim = (-35.0, 35.0) if ylim is None else ylim
    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    else:
        figure = ax.figure

    point_colors, color_label = point_color_values(points, color_by)
    scatter = None
    if len(points):
        horizontal, vertical = (points[:, 1], points[:, 0]) if ego_view else (points[:, 0], points[:, 1])
        scatter = ax.scatter(
            horizontal, vertical, c=point_colors, cmap="viridis", s=point_size, alpha=0.7,
            edgecolors="none", rasterized=True,
            **({"vmin": color_limits[0], "vmax": color_limits[1]} if color_limits else {}),
        )
    draw_boxes(ax, gt_boxes, GT_COLOR, label="Ground truth", ego_view=ego_view)
    draw_boxes(
        ax, pred_boxes, PRED_COLOR, linestyle="--", scores=pred_scores,
        show_scores=show_scores, label="Prediction", ego_view=ego_view,
    )
    ax.scatter([0.0], [0.0], marker="+", s=50, linewidths=1.5, color="#de2d26", label="LiDAR")
    apply_auto_limits(ax, points, (gt_boxes, pred_boxes), xlim, ylim, ego_view=ego_view)
    if show_range_rings:
        draw_range_rings(ax, (20, 40, 60))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y left (m)" if ego_view else "x forward (m)")
    ax.set_ylabel("x forward (m)" if ego_view else "y left (m)")
    ax.set_title(title)
    if source_label:
        ax.text(
            0.01, 0.01, f"Source: {source_label}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
    ax.grid(alpha=0.2)
    if show_colorbar and scatter is not None and color_label:
        figure.colorbar(scatter, ax=ax, label=color_label)
    handles, labels = ax.get_legend_handles_labels()
    if show_legend and handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper right")
    return figure, ax


def save_bev(output_path, points, **kwargs):
    """Render a BEV plot to disk and return its resolved path."""
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_bev(points, **kwargs)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path.resolve()


def load_kitti_geometry(path, target_class="Car"):
    """Load forward-facing GT objects with only raw box geometry and point counts."""
    required = {"frame_id", "gt_index", "gt_x", "gt_y", "gt_z", "gt_dx", "gt_dy", "gt_dz", "gt_heading"}
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Geometry CSV is missing columns: {', '.join(sorted(missing))}")
        objects = []
        for row in reader:
            if target_class and row.get("gt_class", target_class) != target_class:
                continue
            try:
                box = np.asarray([row[f"gt_{name}"] for name in ("x", "y", "z", "dx", "dy", "dz", "heading")], dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if box[0] < 0 or np.any(box[3:6] <= 0):
                continue
            objects.append({
                "frame_id": str(row["frame_id"]).zfill(6), "gt_index": int(row["gt_index"]),
                "box": box, "ground_range": float(np.hypot(box[0], box[1])),
                "bearing": float(np.degrees(np.arctan2(box[1], box[0]))),
                "point_count": row.get("num_points_in_gt", ""),
            })
    return objects


def select_geometry_gallery(objects, max_panels=64):
    """Select up to 64 raw-geometry examples, evenly spaced through sorted range."""
    if max_panels < 1:
        raise ValueError("max_panels must be positive")
    ordered = sorted(objects, key=lambda item: (item["ground_range"], item["frame_id"], item["gt_index"]))
    if len(ordered) <= max_panels:
        return ordered
    return [ordered[index] for index in np.linspace(0, len(ordered) - 1, max_panels, dtype=int)]


def save_kitti_geometry_gallery(output_path, kitti_root, geometry_csv, target_class="Car", max_panels=64):
    """Save a compact ego-BEV gallery with no inferred occlusion metric or empty cells."""
    root = Path(kitti_root).expanduser()
    objects = [
        item for item in load_kitti_geometry(geometry_csv, target_class=target_class)
        if (root / "training" / "velodyne" / f"{item['frame_id']}.bin").is_file()
    ]
    selected = select_geometry_gallery(objects, max_panels=max_panels)
    if not selected:
        raise FileNotFoundError("No selected object has a corresponding KITTI Velodyne file")
    columns = min(8, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.25 * columns, 2.45 * rows), squeeze=False)
    point_cache, used_axes = {}, []
    for index, item in enumerate(selected):
        ax = axes.flat[index]
        used_axes.append(ax)
        frame_id = item["frame_id"]
        velodyne_path = Path("training") / "velodyne" / f"{frame_id}.bin"
        if frame_id not in point_cache:
            point_cache[frame_id] = load_array(root / velodyne_path, 4)
        plot_bev(
            point_cache[frame_id], gt_boxes=item["box"].reshape(1, 7), ax=ax,
            xlim=(-5, 70), ylim=(-35, 35), max_points=30_000, point_size=0.28,
            title="", color_by="ground_range", color_limits=(0, 80), show_colorbar=False,
            show_legend=False, ego_view=True, show_range_rings=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        size = "×".join(f"{value:.1f}" for value in item["box"][3:6])
        ax.text(
            0.02, 0.98,
            f"Frame {frame_id} · Car {item['gt_index']}\n{velodyne_path}\nfile={len(point_cache[frame_id]):,} pts · car n={item['point_count']}\nd={item['ground_range']:.1f} m · bearing={item['bearing']:.0f}° · size={size} m",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )
    for ax in axes.flat[len(selected):]:
        ax.remove()
    figure.suptitle("KITTI ego-BEV geometry gallery", fontsize=13, y=0.995)
    figure.text(
        0.5, 0.976,
        "How to read: each panel is one KITTI training/velodyne frame and selected Car, ordered from near to far. "
        "Purple points are near; yellow points are far. Blue = target Car; red + = LiDAR; grey rings = 20/40/60 m.\n"
        "Labels show raw geometry only: source file, total file points, returns inside the Car, ground distance, lateral bearing, and box size. No occlusion score is inferred.",
        ha="center", va="top", fontsize=7, wrap=True,
    )
    figure.colorbar(
        ScalarMappable(norm=Normalize(0, 80), cmap="viridis"), ax=used_axes, shrink=0.55, pad=0.01,
        label="Point ground range (m)",
    )
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path.resolve(), len(selected)


def load_array(path, point_features=None):
    path = Path(path).expanduser()
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".bin":
        if point_features is None or point_features < 1:
            raise ValueError("Raw .bin input requires a positive --point-features value")
        flat = np.fromfile(path, dtype=np.float32)
        if flat.size % point_features:
            raise ValueError(f"{path} does not contain a whole number of {point_features}-feature points")
        return flat.reshape(-1, point_features)
    raise ValueError(f"Unsupported array format: {path}; use .npy or float32 .bin")


def load_predictions(path, frame_id=None):
    """Load boxes and optional scores from ``.npy`` or pipeline detection JSON."""
    path = Path(path).expanduser()
    if path.suffix == ".npy":
        return np.load(path), None
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value.get("detections", value) if isinstance(value, dict) else value
        if frame_id is not None:
            records = [record for record in records if str(record.get("frame_id")) == str(frame_id)]
            if not records:
                raise ValueError(f"No predictions for frame '{frame_id}' in {path}")
        boxes = np.asarray([record["box_lidar"] for record in records], dtype=np.float32)
        scores = np.asarray([record.get("score", np.nan) for record in records], dtype=np.float32)
        return boxes.reshape(-1, 7), scores
    raise ValueError(f"Unsupported prediction format: {path}; use .npy or .json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render a LiDAR BEV or a compact KITTI geometry gallery.")
    parser.add_argument("points", type=Path, nargs="?", help="Point cloud as .npy or raw float32 .bin.")
    parser.add_argument("--point-features", type=int, default=4, help="Features per point for raw .bin input.")
    parser.add_argument("--gt-boxes", type=Path, help="Ground-truth boxes as an (N, 7+) .npy array.")
    parser.add_argument("--pred-boxes", type=Path, help="Predictions as an (N, 7+) .npy array or pipeline JSON.")
    parser.add_argument("--frame-id", help="Frame to select from a multi-frame prediction JSON.")
    parser.add_argument("--pred-scores", type=Path, help="Optional prediction scores as a .npy array.")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--title", default="Bird's-eye view")
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument("--color-by", choices=("height", "ground_range", "radial_range"), default="height")
    parser.add_argument("--ego-view", action="store_true", help="Put ego at bottom-center with forward upward.")
    parser.add_argument("--range-rings", action="store_true", help="Draw 20, 40, and 60 m range rings.")
    parser.add_argument("--geometry-csv", type=Path, help="Make a compact KITTI geometry gallery from GT boxes.")
    parser.add_argument("--kitti-root", type=Path, help="KITTI root required by --geometry-csv.")
    parser.add_argument("--target-class", default="Car", help="GT class selected for --geometry-csv.")
    parser.add_argument("--output", type=Path, default=Path("bev.png"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.geometry_csv:
            if args.points or not args.kitti_root:
                raise ValueError("--geometry-csv requires --kitti-root and no point-cloud argument")
            output, selected = save_kitti_geometry_gallery(
                args.output, args.kitti_root, args.geometry_csv, target_class=args.target_class,
            )
            print(f"WROTE {output} ({selected} geometry panels)")
            return 0
        if args.points is None:
            raise ValueError("Supply a point cloud or --geometry-csv with --kitti-root")
        points = load_array(args.points, args.point_features)
        gt_boxes = load_array(args.gt_boxes) if args.gt_boxes else None
        pred_boxes, embedded_scores = (
            load_predictions(args.pred_boxes, args.frame_id) if args.pred_boxes else (None, None)
        )
        pred_scores = load_array(args.pred_scores) if args.pred_scores else embedded_scores
        output = save_bev(
            args.output,
            points,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            score_threshold=args.score_threshold,
            xlim=args.xlim,
            ylim=args.ylim,
            max_points=args.max_points,
            title=args.title,
            show_scores=args.show_scores,
            color_by=args.color_by,
            source_label=args.points.name,
            ego_view=args.ego_view,
            show_range_rings=args.range_rings,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 2
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
