"""Return-weighted Solid-angle Visibility (RSV) for DAIR-V2X-C.

Run: PYTHONPATH=. python -m dairv2x.rsv

Candidate metric specified in THESIS_CONTEXT.md. Definition 3 asks how much of
a box's solid angle *should* be visible given the labelled boxes. RSV asks
how much of it the LiDAR *measured* as the box, using the actual returns:

  1. range image   bin this sensor's returns into its own ray grid
                   (rings x azimuth); each cell keeps its nearest return and
                   covers solid angle  Omega_k = dphi * (sin e_top - sin e_bot)
  2. footprint     K_i = cells whose centre ray passes through box i;
                   Omega_exp = sum of their Omega_k
  3. classify      each footprint cell, with t_in/t_out the ray's entry/exit
                   distance through the box and tolerance delta:
                     H hit      t_in - delta <= r <= t_out + delta
                     B before   r < t_in - delta  (B_lab if another labelled
                                box is entered first on that ray, else B_unlab)
                     P pass     r > t_out + delta
                     M missing  no return in the cell (not counted as occluded)
  4. score         V_rsv = Omega(H) / (Omega(H) + Omega(B) + Omega(P));
                   unknown (NaN) when no footprint cell has a return

Everything is in the sensor's own frame with the LiDAR at the origin.
Range, point count, incidence and intensity are not weights.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dairv2x.tools.visibility_auc import SIDES, iter_side_frames
from dairv2x.tools.visibility_frame import (
    DATASET_ROOT, facing_face_samples, solid_angle_visibility, yaw_rotation,
)
from mixedsig.tools.visualize_agg_frame import points_inside_box, ray_box_intersections

DELTA = 0.2          # m, depth tolerance for H/B/P
N_MIN = 10           # footprints with fewer cells are flagged low-resolution
GRID_FRAMES = 20     # frames pooled to estimate each sensor's ray grid

# Hesai Pandar40P nominal beam elevations (deg). The DAIR vehicle LiDAR matches
# it (-25..+15 deg, 40 rings, 0.2 deg azimuth step); each ring is then refined
# to the median elevation of the returns assigned to it.
PANDAR40P = np.r_[-25, -19, -14, -13, -12, -11, -10, -9, -8, -7,
                  np.linspace(-6, 2, 25), 3, 5, 8, 11, 15].astype(float)


@dataclass
class RayGrid:
    """One sensor's ray grid: ring elevations x azimuth columns (degrees)."""
    elevations: np.ndarray    # (R,) ascending beam elevations
    bounds: np.ndarray        # (R+1,) ring band edges
    az_min: float
    az_step: float
    columns: int
    full_circle: bool
    method: str

    def cell_solid_angle(self, rows):
        top, bottom = np.radians(self.bounds[rows + 1]), np.radians(self.bounds[rows])
        return np.radians(self.az_step) * (np.sin(top) - np.sin(bottom))

    def directions(self, rows, cols):
        el = np.radians(self.elevations[rows])
        az = np.radians(self.az_min + (cols + 0.5) * self.az_step)
        return np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]

    def range_image(self, cloud):
        """(R, C) nearest return range per cell; inf where the cell is empty."""
        r = np.linalg.norm(cloud, axis=1)
        cloud, r = cloud[r > 0.5], r[r > 0.5]
        el = np.degrees(np.arcsin(cloud[:, 2] / r))
        az = np.degrees(np.arctan2(cloud[:, 1], cloud[:, 0])) - self.az_min
        if self.full_circle:
            az %= 360
        rows = np.searchsorted(self.bounds, el) - 1
        cols = np.floor(az / self.az_step).astype(int)
        keep = (rows >= 0) & (rows < len(self.elevations)) & (cols >= 0) & (cols < self.columns)
        image = np.full((len(self.elevations), self.columns), np.inf)
        np.minimum.at(image, (rows[keep], cols[keep]), r[keep])
        return image


def ring_bounds(elevations):
    """Band edges at midpoints between rings; outer edges mirror the half gap."""
    mid = (elevations[1:] + elevations[:-1]) / 2
    return np.r_[2 * elevations[0] - mid[0], mid, 2 * elevations[-1] - mid[-1]]


def estimate_grid(clouds, side):
    """Estimate a sensor's ray grid from pooled returns (sensor frame)."""
    cloud = np.concatenate(clouds)
    r = np.linalg.norm(cloud, axis=1)
    cloud, r = cloud[r > 0.5], r[r > 0.5]
    el = np.degrees(np.arcsin(cloud[:, 2] / r))
    az = np.degrees(np.arctan2(cloud[:, 1], cloud[:, 0]))
    if side == "Vehicle":
        rows = np.searchsorted(ring_bounds(PANDAR40P), el) - 1
        refined = np.array([np.median(el[rows == k]) if np.any(rows == k) else e
                            for k, e in enumerate(PANDAR40P)])
        return RayGrid(refined, ring_bounds(refined), -180.0, 0.2, 1800, True,
                       "Pandar40P table, ring medians from data")
    # RSU: no stable ring structure in the virtual-LiDAR frame (ring elevations
    # drift with azimuth), so use a uniform grid over the observed field of view.
    step_el, step_az = 0.2, 0.25
    low, high = np.percentile(el, [0.1, 99.9])
    elevations = np.arange(low, high, step_el) + step_el / 2
    az_low, az_high = np.percentile(az, [0.05, 99.95])
    return RayGrid(elevations, ring_bounds(elevations), az_low, step_az,
                   int(np.ceil((az_high - az_low) / step_az)), False,
                   f"uniform {step_el} x {step_az} deg grid (approximate)")


def footprint_cells(box, grid):
    """Candidate (rows, cols) inside the box's azimuth/elevation bounds."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = (signs * box[3:6] / 2) @ yaw_rotation(box[6]).T + box[:3]
    el = np.degrees(np.arctan2(corners[:, 2], np.hypot(corners[:, 0], corners[:, 1])))
    centre_az = np.degrees(np.arctan2(box[1], box[0]))
    az = centre_az + (np.degrees(np.arctan2(corners[:, 1], corners[:, 0])) - centre_az + 180) % 360 - 180
    rows = np.flatnonzero((grid.bounds[1:] > el.min()) & (grid.bounds[:-1] < el.max()))
    cols = np.arange(np.floor((az.min() - grid.az_min) / grid.az_step),
                     np.ceil((az.max() - grid.az_min) / grid.az_step) + 1).astype(int)
    if grid.full_circle:
        cols %= grid.columns
    else:
        cols = cols[(cols >= 0) & (cols < grid.columns)]
    rows, cols = np.meshgrid(rows, cols, indexing="ij")
    return rows.ravel(), cols.ravel()


def rsv(boxes, image, grid, delta=DELTA):
    """RSV and its parts per box, from boxes and a range image (sensor frame).

    Returns a dict of (N,) arrays. Fractions are NaN when undefined; boxes
    containing the sensor are NaN and are not used as labelled occluders.
    """
    boxes = np.asarray(boxes, dtype=float)
    origin = np.zeros(3)
    contains = np.array([points_inside_box(origin[None], b)[0] for b in boxes])
    keys = ("v_rsv", "v_lo", "v_hi", "o_lab", "o_unlab", "p_frac", "m_frac",
            "omega_exp", "omega_def3", "cells", "low_res")
    out = {k: np.full(len(boxes), np.nan) for k in keys}
    for i, box in enumerate(boxes):
        if contains[i]:
            continue
        rows, cols = footprint_cells(box, grid)
        directions = grid.directions(rows, cols)
        inside, t_in, t_out = ray_box_intersections(origin, directions, box, margin=0.0)
        rows, cols, directions = rows[inside], cols[inside], directions[inside]
        t_in, t_out = t_in[inside], t_out[inside]
        omega = grid.cell_solid_angle(rows)
        r = image[rows, cols]
        empty = ~np.isfinite(r)
        hit = ~empty & (r >= t_in - delta) & (r <= t_out + delta)
        before = ~empty & (r < t_in - delta)
        passed = ~empty & (r > t_out + delta)
        labelled = np.zeros(len(r), dtype=bool)
        for j in np.flatnonzero(~contains):
            if j != i:
                crosses, entry, _ = ray_box_intersections(origin, directions, boxes[j], margin=0.0)
                labelled |= crosses & (entry < t_in)
        expected = omega.sum()
        measured = omega[hit | before | passed].sum()
        out["omega_exp"][i], out["cells"][i] = expected, len(r)
        out["low_res"][i] = len(r) < N_MIN
        out["omega_def3"][i] = facing_face_samples(box, origin, 32)[1].sum()
        if expected > 0:
            out["m_frac"][i] = omega[empty].sum() / expected
            out["v_lo"][i] = omega[hit].sum() / expected
            out["v_hi"][i] = omega[hit | empty].sum() / expected
        if measured > 0:
            out["v_rsv"][i] = omega[hit].sum() / measured
            out["o_lab"][i] = omega[before & labelled].sum() / measured
            out["o_unlab"][i] = omega[before & ~labelled].sum() / measured
            out["p_frac"][i] = omega[passed].sum() / measured
    return out


def sensor_grids(root):
    """Estimate each side's ray grid once from its first GRID_FRAMES clouds."""
    grids = {}
    for side, (label_dir, cloud_dir) in SIDES.items():
        clouds = [cloud for _, (_, _, cloud) in zip(range(GRID_FRAMES),
                                                    iter_side_frames(root, label_dir, cloud_dir))]
        grids[side] = estimate_grid(clouds, side)
    return grids


def main():
    root = Path(DATASET_ROOT)
    for side, grid in sensor_grids(root).items():
        label_dir, cloud_dir = SIDES[side]
        labels, boxes, cloud = next(iter_side_frames(root, label_dir, cloud_dir))
        full = grid.cell_solid_angle(np.arange(len(grid.elevations))).sum() * grid.columns
        print(f"\n{side}: {grid.method}; {len(grid.elevations)} rings x {grid.columns} columns, "
              f"grid solid angle {full:.3f} sr")
        parts = rsv(boxes, grid.range_image(cloud), grid)
        def3 = solid_angle_visibility(boxes, np.zeros(3), 32)
        print(f"{'box':>4}{'range':>7}{'cells':>7}{'V_def3':>8}{'V_rsv':>7}"
              f"{'O_lab':>7}{'O_unlab':>8}{'P':>6}{'M':>6}  occluded_state")
        for i, label in enumerate(labels):
            row = [def3[i], parts["v_rsv"][i], parts["o_lab"][i], parts["o_unlab"][i],
                   parts["p_frac"][i], parts["m_frac"][i]]
            print(f"{i:>4}{np.linalg.norm(boxes[i, :3]):>7.1f}{parts['cells'][i]:>7.0f}"
                  + "".join(f"{v:>{w}.2f}" for v, w in zip(row, (8, 7, 7, 8, 6, 6)))
                  + f"  {label['occluded_state']}")


if __name__ == "__main__":
    main()
