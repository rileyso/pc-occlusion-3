"""One figure, one panel per Definition-3 pipeline step, on a real DAIR pair.

Run: PYTHONPATH=. python -m dairv2x.visibility_pipeline_figure
Writes data/visibility_example/pipeline.png (and .pdf for the report).

Panels: (a) labelled boxes + sensor poses in the world frame, (b) the same
scene after the inverse SE(3) into the vehicle LiDAR frame, (c) solid-angle
weighted samples on the target's sensor-facing faces, (d) sensor->sample rays
tested against the other boxes, (e) the projection onto the unit sphere seen
from the sensor, (f) the resulting visibility for every box and both sensors.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dairv2x.tools.visibility_frame import (
    DATASET_ROOT, PAIR_INDEX, SAMPLES_PER_EDGE, DAIRV2XExplorer, blocked_samples,
    boxes_in_sensor_frame, facing_face_samples, occluder_candidates,
    solid_angle_visibility, target_breakdown, yaw_rotation,
)

OUTPUT = Path("data/visibility_example/pipeline")
FIGURE_SAMPLES = 16          # coarser grid so individual samples are visible
RAYS_SHOWN = 60
TARGET, OCCLUDER, OTHER = "#2a78d6", "#eb6834", "#b4b2a9"
INK, MUTED = "#0b0b0b", "#52514e"


def footprint(box):
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]) * box[3:5] / 2
    return corners @ yaw_rotation(box[6])[:2, :2].T + box[:2]


def draw_boxes(ax, boxes, target, occluder, shift=np.zeros(3)):
    for i, box in enumerate(boxes):
        colour = TARGET if i == target else OCCLUDER if i == occluder else OTHER
        xy = footprint(box) - shift[:2]
        ax.fill(*xy.T, color=colour, alpha=0.25 if colour == OTHER else 0.45, linewidth=0)
        ax.plot(*xy.T, color=colour, linewidth=1.2)


def draw_sensor(ax, pose, label, marker, shift=np.zeros(3), arrow=8.0):
    origin = pose[:3, 3] - shift
    ax.plot(*origin[:2], marker=marker, color=INK, markersize=8, linestyle="none")
    for axis, name in ((0, "x"), (1, "y")):
        tip = pose[:3, axis] * arrow
        ax.annotate("", origin[:2] + tip[:2], origin[:2],
                    arrowprops=dict(arrowstyle="->", color=INK, linewidth=1))
        ax.text(*(origin[:2] + tip[:2] * 1.15), name, color=MUTED, fontsize=7,
                ha="center", va="center")
    ax.text(origin[0] - 2, origin[1] - 3, label, color=INK, fontsize=8, ha="right", va="top")


def azimuth_elevation(points, origin):
    rays = points - origin
    return (np.degrees(np.arctan2(rays[:, 1], rays[:, 0])),
            np.degrees(np.arctan2(rays[:, 2], np.linalg.norm(rays[:, :2], axis=1))))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(OTHER)
    ax.grid(color="#e8e7e2", linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    dataset = DAIRV2XExplorer(DATASET_ROOT)
    frame = dataset.return_labeled_sync_time_ids_of_seq(PAIR_INDEX)[0]
    boxes = dataset.return_gt_boxes_in_map(PAIR_INDEX, frame)
    poses = {a: dataset.return_map_se3_agent(PAIR_INDEX, a, frame)
             for a in ("vehicle", "infrastructure")}
    origin = poses["vehicle"][:3, 3]
    scores = {a: solid_angle_visibility(boxes, p[:3, 3], SAMPLES_PER_EDGE) for a, p in poses.items()}
    unavailable = np.isnan(scores["vehicle"])
    target = int(np.nanargmin(scores["vehicle"]))
    detail = target_breakdown(boxes, origin, target, unavailable, SAMPLES_PER_EDGE, "ray")
    occluder = max(detail["by_occluder"], key=detail["by_occluder"].get)

    fig = plt.figure(figsize=(15, 9.5), facecolor="white")
    axes = [fig.add_subplot(2, 3, k + 1, projection="3d" if k == 2 else None) for k in range(6)]

    # (a) world frame: boxes + both sensor poses (centred on the vehicle LiDAR for readability)
    ax = axes[0]
    draw_boxes(ax, boxes, target, occluder, shift=origin)
    draw_sensor(ax, poses["vehicle"], "vehicle LiDAR", "^", shift=origin)
    draw_sensor(ax, poses["infrastructure"], "RSU LiDAR", "s", shift=origin)
    ax.set_aspect("equal")
    style(ax, "(a) Labelled boxes and sensor poses, world frame",
          "world x − vehicle origin (m)", "world y − vehicle origin (m)")

    # (b) vehicle LiDAR frame after T_sensor<-world = inv(T_world<-sensor)
    ax = axes[1]
    local = boxes_in_sensor_frame(boxes, poses["vehicle"])
    draw_boxes(ax, local, target, occluder)
    draw_sensor(ax, np.eye(4), "vehicle LiDAR (origin)", "^")
    ax.set_aspect("equal")
    style(ax, "(b) Inverse SE(3): vehicle LiDAR frame", "x forward (m)", "y left (m)")

    # (c) solid-angle weighted samples on sensor-facing faces of the target
    ax = axes[2]
    points, weights = facing_face_samples(boxes[target], origin, FIGURE_SAMPLES)
    rel = points - boxes[target, :3]
    sc = ax.scatter(*rel.T, c=weights / weights.sum(), cmap="Blues", s=14,
                    vmin=0, edgecolors=TARGET, linewidths=0.3)
    toward = (origin - boxes[target, :3]) / np.linalg.norm(origin - boxes[target, :3]) * 3
    ax.quiver(0, 0, 0, *toward, color=INK, linewidth=1, arrow_length_ratio=0.15)
    ax.text(*toward * 1.1, "to sensor", color=INK, fontsize=7)
    ax.set_title("(c) Sensor-facing face samples", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("x (m)", fontsize=7, color=MUTED)
    ax.set_ylabel("y (m)", fontsize=7, color=MUTED)
    ax.set_zlabel("z (m)", fontsize=7, color=MUTED)
    ax.tick_params(labelsize=6, colors=MUTED)
    ax.set_box_aspect(np.ptp(rel, axis=0) + 0.2)  # equal metric scale on x, y, z
    bar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    bar.set_label("share of target Ω per sample", fontsize=7, color=MUTED)
    bar.ax.tick_params(labelsize=6, colors=MUTED)

    # (d) rays from the sensor to samples, tested against the other boxes
    ax = axes[3]
    blocked = np.zeros(len(points), bool)
    for j in occluder_candidates(boxes, origin, target, unavailable, "ray"):
        blocked |= blocked_samples(points, origin, boxes[j], "ray")
    pick = np.linspace(0, len(points) - 1, RAYS_SHOWN).astype(int)
    for k in pick:
        ax.plot([0, points[k, 0] - origin[0]], [0, points[k, 1] - origin[1]],
                color=OCCLUDER if blocked[k] else TARGET, linewidth=0.6, alpha=0.7)
    draw_boxes(ax, boxes, target, occluder, shift=origin)
    ax.plot(0, 0, marker="^", color=INK, markersize=8)
    span = np.r_[boxes[[target, occluder], :2] - origin[:2], [[0, 0]]]
    pad = 4
    ax.set_xlim(span[:, 0].min() - pad, span[:, 0].max() + pad)
    ax.set_ylim(span[:, 1].min() - pad, span[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.plot([], [], color=TARGET, label="ray reaches target")
    ax.plot([], [], color=OCCLUDER, label="ray hits another box first")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    style(ax, "(d) Ray test against other labelled boxes",
          "world x − vehicle origin (m)", "world y − vehicle origin (m)")

    # (e) unit-sphere view: azimuth/elevation of target and occluder silhouettes
    ax = axes[4]
    occ_points, _ = facing_face_samples(boxes[occluder], origin, 24)
    az, el = azimuth_elevation(occ_points, origin)
    ax.scatter(az, el, s=10, color=OCCLUDER, alpha=0.35, linewidths=0, label="occluder silhouette")
    az, el = azimuth_elevation(points, origin)
    ax.scatter(az[~blocked], el[~blocked], s=12, color=TARGET, linewidths=0, label="target, visible")
    ax.scatter(az[blocked], el[blocked], s=12, facecolors="none", edgecolors=TARGET,
               linewidths=0.8, label="target, hidden")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax, f"(e) Unit-sphere projection: V = {detail['visibility']:.2f}",
          "azimuth from vehicle LiDAR (°)", "elevation (°)")

    # (f) visibility for every box, both sensors
    ax = axes[5]
    order = np.argsort(np.nan_to_num(scores["vehicle"], nan=-1))
    rows = np.arange(len(order))
    for row, i in zip(rows, order):
        v, r = scores["vehicle"][i], scores["infrastructure"][i]
        if np.isfinite(v) and np.isfinite(r):
            ax.plot([v, r], [row, row], color=OTHER, linewidth=1.5, zorder=1)
    ax.scatter(scores["vehicle"][order], rows, marker="^", s=40, color=INK, zorder=2, label="vehicle LiDAR")
    ax.scatter(scores["infrastructure"][order], rows, marker="s", s=34, facecolors="white",
               edgecolors=INK, zorder=2, label="RSU LiDAR")
    fig.canvas.draw()
    ax.set_yticks(rows, [f"box {int(boxes[i, 8])}" for i in order], fontsize=6)
    for row, i in zip(rows, order):
        if i == target:
            ax.get_yticklabels()[row].set_color(TARGET)
        if i == occluder:
            ax.get_yticklabels()[row].set_color(OCCLUDER)
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax, "(f) Visibility per box and sensor", "V (0 hidden – 1 fully visible)", "")

    fig.suptitle(f"Definition-3 visibility pipeline — DAIR-V2X-C pair {PAIR_INDEX} "
                 f"(target box {target} blue, main occluder box {occluder} orange)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(OUTPUT.with_suffix(suffix), dpi=200)
    print(f"Saved {OUTPUT.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
