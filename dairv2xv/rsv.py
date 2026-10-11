"""Return-weighted solid-angle visibility (RSV) for vehicle-side DAIR boxes.

Run: python dairv2xv/rsv.py

Each box's expected LiDAR footprint is the set of ray-grid cells whose centre
ray passes through it. Each footprint cell is classified by its nearest
measured return r against the ray's box entry/exit distances (t_in, t_out):
H hit (t_in - delta <= r <= t_out + delta), B before (r < t_in - delta),
P pass-through (r > t_out + delta), M missing (no return). Every cell is
weighted by its solid angle, and

    visibility_score = Omega(H) / Omega(H + B + P)

M is excluded (unknown, not occluded). The score is None when no footprint
cell has a return, the box has all-zero dimensions, or the sensor is inside it.
All geometry uses vehicle LiDAR coordinates with the sensor at the origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from box import Box
from checkpoint import dump_boxes, load_checkpoint, save_checkpoint
from explorer import DAIRV2XVehicleExplorer
from log_tee import tee_to_log

# Example Dataset: /mnt/c/Datasets/dair-v2x-c-example/vehicle-side
# Pass the vehicle-side folder as the first argument to override the default.
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FRESH = "--fresh" in sys.argv  # ignore any saved checkpoint and start over
ROOT = args[0] if args else "/mnt/d/dairv2x_v_full/vehicle-side"
CHECKPOINT_EVERY = 200  # frames between partial-result saves
DELTA = 0.2          # metres of depth tolerance for H/B/P
GRID_FRAMES = 20     # captures pooled to estimate the ray grid
# Hesai Pandar40P nominal beam elevations (degrees). The DAIR vehicle LiDAR
# matches it (-25..+15 deg, 40 rings, 0.2 deg azimuth step).
PANDAR40P = np.r_[-25, -19, -14, -13, -12, -11, -10, -9, -8, -7,
                  np.linspace(-6, 2, 25), 3, 5, 8, 11, 15].astype(float)

COLOURS = {"Easy": "tab:green", "Moderate": "tab:orange", "Hard": "tab:red"}
def ring_bounds(elevations):
    """Band edges at midpoints between rings; outer edges mirror the half gap."""
    mid = (elevations[1:] + elevations[:-1]) / 2
    return np.r_[2 * elevations[0] - mid[0], mid, 2 * elevations[-1] - mid[-1]]


@dataclass
class RayGrid:
    """A spinning LiDAR's ray grid: ring elevations x azimuth columns (degrees)."""

    elevations: np.ndarray
    az_min: float = -180.0
    az_step: float = 0.2

    @property
    def bounds(self):
        return ring_bounds(self.elevations)

    @property
    def columns(self):
        return int(round(360 / self.az_step))

    def solid_angle(self, rows):
        """Omega_k = d_phi * (sin e_top - sin e_bottom) for each ring row."""
        top, bottom = np.radians(self.bounds[rows + 1]), np.radians(self.bounds[rows])
        return np.radians(self.az_step) * (np.sin(top) - np.sin(bottom))

    def directions(self, rows, cols):
        """Unit ray directions at ring elevation and cell-centre azimuth."""
        elevation = np.radians(self.elevations[rows])
        azimuth = np.radians(self.az_min + (cols + 0.5) * self.az_step)
        return np.c_[np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]

    def range_image(self, points):
        """(rings, columns) nearest return range; inf where a cell has no return."""
        points = np.asarray(points, dtype=float)[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        ranges = np.linalg.norm(points, axis=1)
        points, ranges = points[ranges > 0.5], ranges[ranges > 0.5]
        elevation = np.degrees(np.arcsin(points[:, 2] / ranges))
        azimuth = (np.degrees(np.arctan2(points[:, 1], points[:, 0])) - self.az_min) % 360
        rows = np.searchsorted(self.bounds, elevation) - 1
        cols = np.floor(azimuth / self.az_step).astype(int) % self.columns
        keep = (rows >= 0) & (rows < len(self.elevations))
        image = np.full((len(self.elevations), self.columns), np.inf)
        np.minimum.at(image, (rows[keep], cols[keep]), ranges[keep])
        return image


def estimate_ray_grid(point_clouds):
    """Pandar40P table with each ring moved to the median elevation of its returns."""
    points = np.concatenate([np.asarray(p, dtype=float)[:, :3] for p in point_clouds])
    points = points[np.isfinite(points).all(axis=1)]
    ranges = np.linalg.norm(points, axis=1)
    elevation = np.degrees(np.arcsin(points[ranges > 0.5, 2] / ranges[ranges > 0.5]))
    rows = np.searchsorted(ring_bounds(PANDAR40P), elevation) - 1
    refined = [np.median(elevation[rows == k]) if np.any(rows == k) else e
               for k, e in enumerate(PANDAR40P)]
    return RayGrid(np.sort(refined))


def _rotation(box):
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def is_available(box):
    """False for all-zero (unavailable) boxes and boxes containing the sensor."""
    dimensions = np.array([box.length, box.width, box.height])
    if np.all(dimensions == 0):
        return False
    sensor_local = -np.array([box.x, box.y, box.z]) @ _rotation(box)
    return not np.all(np.abs(sensor_local) <= dimensions / 2 + 1e-9)


def ray_box_intersections(directions, box):
    """Rays from the origin: (hit mask, entry distance, exit distance) per ray."""
    rotation = _rotation(box)
    origin = -np.array([box.x, box.y, box.z]) @ rotation
    local = np.asarray(directions, dtype=float) @ rotation
    half = np.array([box.length, box.width, box.height]) / 2
    parallel = np.abs(local) < 1e-12
    safe = np.where(parallel, 1.0, local)
    first, second = (-half - origin) / safe, (half - origin) / safe
    near = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
    far = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
    hit = ~np.any(parallel & (np.abs(origin) > half), axis=1) & (far >= np.maximum(near, 0))
    return hit, np.maximum(near, 0), far


def footprint_cells(box, grid):
    """(rows, cols) whose centre ray passes through the box."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = (signs * np.array([box.length, box.width, box.height]) / 2) @ _rotation(box).T \
        + [box.x, box.y, box.z]
    elevation = np.degrees(np.arctan2(corners[:, 2], np.hypot(corners[:, 0], corners[:, 1])))
    centre = np.degrees(np.arctan2(box.y, box.x))
    azimuth = centre + (np.degrees(np.arctan2(corners[:, 1], corners[:, 0])) - centre + 180) % 360 - 180
    bounds = grid.bounds
    rows = np.flatnonzero((bounds[1:] > elevation.min()) & (bounds[:-1] < elevation.max()))
    cols = np.arange(np.floor((azimuth.min() - grid.az_min) / grid.az_step),
                     np.ceil((azimuth.max() - grid.az_min) / grid.az_step) + 1).astype(int) % grid.columns
    rows, cols = (a.ravel() for a in np.meshgrid(rows, cols, indexing="ij"))
    hit, _, _ = ray_box_intersections(grid.directions(rows, cols), box)
    return rows[hit], cols[hit]


def compute_rsv(box, frame_boxes, image, grid, delta=DELTA):
    """Set box.visibility_score to RSV and return its solid-angle parts.

    frame_boxes are the other labelled boxes in the capture; one entered
    before the target on a B ray makes that ray B_lab, otherwise B_unlab.
    Returns None (score None) when the box is unavailable.
    """
    box.visibility_score = None
    if not is_available(box):
        return None
    rows, cols = footprint_cells(box, grid)
    directions = grid.directions(rows, cols)
    _, t_in, t_out = ray_box_intersections(directions, box)
    omega = grid.solid_angle(rows)
    r = image[rows, cols]
    empty = ~np.isfinite(r)
    hit = ~empty & (r >= t_in - delta) & (r <= t_out + delta)
    before = ~empty & (r < t_in - delta)
    passed = ~empty & (r > t_out + delta)
    labelled = np.zeros(len(r), dtype=bool)
    for other in frame_boxes:
        if other is not box and is_available(other):
            crosses, entry, _ = ray_box_intersections(directions, other)
            labelled |= crosses & (entry < t_in)
    parts = {"cells": len(r), "omega_expected": omega.sum(),
             "H": omega[hit].sum(), "B_lab": omega[before & labelled].sum(),
             "B_unlab": omega[before & ~labelled].sum(),
             "P": omega[passed].sum(), "M": omega[empty].sum()}
    measured = parts["H"] + parts["B_lab"] + parts["B_unlab"] + parts["P"]
    if measured > 0:
        box.visibility_score = parts["H"] / measured
    return parts


if __name__ == "__main__":
    tee_to_log(Path(__file__).with_name("rsv.log"))
    data = DAIRV2XVehicleExplorer(ROOT, skip_incomplete=False)
    captures = data.return_available_sequences()
    grid_clouds = []
    for idx in captures:
        if len(grid_clouds) == GRID_FRAMES:
            break
        try:
            grid_clouds.append(data.return_agent_point_cloud(idx)[0])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    print(f"estimating ray grid from {len(grid_clouds)} frames")
    grid = estimate_ray_grid(grid_clouds)
    total_frames = len(captures)
    CHECKPOINT = Path(__file__).with_name("rsv.ckpt.pkl")
    boxes, broken, start = (([], [], 0) if FRESH
                            else load_checkpoint(CHECKPOINT, ROOT, total_frames))
    for idx in range(start, total_frames):
        if idx > start and idx % CHECKPOINT_EVERY == 0:
            save_checkpoint(CHECKPOINT, ROOT, total_frames, idx, broken, boxes)
        try:
            frame_boxes = []
            pose = None  # pose is only stored on the Box; the official data has no NovAtel calib
            points, _ = data.return_agent_point_cloud(idx)
            labels = data.return_labels(idx)
            for row in data.return_gt_boxes_in_lidar(idx):
                box = Box(*row[:7], category=int(row[7]), object_index=int(row[8]),
                          frame=idx, sensor_pose=pose)
                box.compute_measurements(points, labels[box.object_index])
                frame_boxes.append(box)
            image = grid.range_image(points)
            for box in frame_boxes:
                compute_rsv(box, frame_boxes, image, grid)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            broken.append(idx)
            print(f"frame: {idx + 1}/{total_frames} skipped ({type(error).__name__}: {error})")
            continue
        boxes += frame_boxes
        print(f"frame: {idx + 1}/{total_frames} boxes: {len(frame_boxes)}")
    print(f"skipped {len(broken)} broken frames")
    dump_boxes(boxes, Path(__file__).with_name("rsv.pkl"))
    CHECKPOINT.unlink(missing_ok=True)  # finished: a rerun starts clean

for box in boxes:
    if box.class_name == "Car" and box.difficulty != "Unclassified" and box.difficulty == "Easy":
        score = f"{box.visibility_score:.4f}" if box.visibility_score is not None else "None"
        print(f"frame: {box.frame}, box: {box.object_index}, points: {box.num_points}, "
            f"difficulty label: {box.difficulty}, visibility score: {score}, class: {box.class_name}")

for box in boxes:
    if box.class_name == "Car" and box.difficulty != "Unclassified" and box.difficulty == "Moderate":
        score = f"{box.visibility_score:.4f}" if box.visibility_score is not None else "None"
        print(f"frame: {box.frame}, box: {box.object_index}, points: {box.num_points}, "
            f"difficulty label: {box.difficulty}, visibility score: {score}, class: {box.class_name}")

for box in boxes:
    if box.class_name == "Car" and box.difficulty != "Unclassified" and box.difficulty == "Hard":
        score = f"{box.visibility_score:.4f}" if box.visibility_score is not None else "None"
        print(f"frame: {box.frame}, box: {box.object_index}, points: {box.num_points}, "
            f"difficulty label: {box.difficulty}, visibility score: {score}, class: {box.class_name}")



cars = [b for b in boxes if b.class_name == "Car" and b.difficulty in COLOURS
        and b.visibility_score is not None]
cars.sort(key=lambda b: (list(COLOURS).index(b.difficulty), -b.visibility_score))

df = pd.DataFrame({
    "box": range(len(cars)),
    "difficulty": [b.difficulty for b in cars],
    "visibility_score": [b.visibility_score for b in cars],
})
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=df, x="box", y="visibility_score", hue="difficulty",
            hue_order=list(COLOURS), palette=COLOURS, dodge=False, width=1.0,
            ax=ax)
ax.set_xticks([])
ax.set_xlabel("Car box (grouped by difficulty)")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1)
ax.set_title("PASCAL Criteria")
ax.legend(title=None)
fig.savefig(Path(__file__).with_name("rsv_bins_full_set.png"), dpi=300, bbox_inches="tight")
# plt.show()  # blocks until the window is closed; the PNGs above are saved without it

fig, ax = plt.subplots(figsize=(6, 5))
BLUE, NAVY = "#9ECAE1", "#08306B"
sns.boxplot(data=df, x="difficulty", y="visibility_score", order=list(COLOURS),
            color=BLUE, linecolor=NAVY, linewidth=1.2,
            flierprops={"markeredgecolor": NAVY, "markersize": 4}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1.0)
ax.set_title("PASCAL Criteria")
fig.savefig(Path(__file__).with_name("rsv_whisker_full_set.png"), dpi=300, bbox_inches="tight")
# plt.show()  # blocks until the window is closed; the PNGs above are saved without it
