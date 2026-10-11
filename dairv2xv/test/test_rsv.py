"""Run: python dairv2xv/test/test_rsv.py (synthetic scans; no dataset or PCD reader needed)."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from box import Box
from rsv import RayGrid, compute_rsv, ray_box_intersections, ring_bounds


def make_box(x, y, z, length, width, height, yaw=0.0, index=0):
    return Box(x, y, z, length, width, height, yaw, category=0,
               object_index=index, frame=0, sensor_pose=np.eye(4))


def scan(grid, scene):
    """One return per ray at the nearest scene surface (perfect synthetic LiDAR)."""
    rows, cols = (a.ravel() for a in np.meshgrid(np.arange(len(grid.elevations)),
                                                 np.arange(grid.columns), indexing="ij"))
    directions = grid.directions(rows, cols)
    nearest = np.full(len(directions), np.inf)
    for obstacle in scene:
        hit, entry, _ = ray_box_intersections(directions, obstacle)
        nearest = np.where(hit, np.minimum(nearest, entry), nearest)
    keep = np.isfinite(nearest)
    return directions[keep] * nearest[keep, None]


grid = RayGrid(np.arange(-10, 10.01, 0.25), az_step=0.25)
np.testing.assert_allclose(ring_bounds(np.array([0.0, 1.0, 3.0])), [-0.5, 0.5, 2.0, 4.0])
# Solid angles sum to the band between the outer ring edges.
total = grid.solid_angle(np.arange(len(grid.elevations))).sum() * grid.columns
np.testing.assert_allclose(total, 2 * np.pi * (np.sin(np.radians(grid.bounds[-1]))
                                               - np.sin(np.radians(grid.bounds[0]))))

target = make_box(20, 0, 0, 4, 2, 1.6)

# (a) Isolated, fully returned box: every footprint ray is a hit.
parts = compute_rsv(target, [target], grid.range_image(scan(grid, [target])), grid)
assert target.visibility_score == 1.0 and parts["M"] == 0 and parts["cells"] > 50

# (b) Unlabelled wall covering the target's left half (y > 0): B_unlab, about 0.5.
wall = make_box(10, 1.0, 0, 0.2, 2.0, 4.0)
parts = compute_rsv(target, [target], grid.range_image(scan(grid, [target, wall])), grid)
assert 0.4 < target.visibility_score < 0.6
assert parts["B_unlab"] > 0 and parts["B_lab"] == 0

# (c) The same blocker labelled: identical score, occlusion moves to B_lab.
blocker = make_box(10, 1.0, 0, 0.2, 2.0, 4.0, index=1)
score_unlabelled = target.visibility_score
parts = compute_rsv(target, [target, blocker], grid.range_image(scan(grid, [target, blocker])), grid)
assert target.visibility_score == score_unlabelled
assert parts["B_lab"] > 0 and parts["B_unlab"] == 0

# (d) Rays with no return are missing, not occluded: score unchanged.
points = scan(grid, [target])
parts = compute_rsv(target, [target], grid.range_image(points[::2]), grid)
assert target.visibility_score == 1.0 and parts["M"] > 0

# (e) Beams passing behind the box (empty label) are pass-through, not hits.
background = make_box(40, 0, 0, 0.2, 20, 10)
parts = compute_rsv(target, [target], grid.range_image(scan(grid, [background])), grid)
assert target.visibility_score == 0.0 and parts["P"] > 0 and parts["H"] == 0

# Unavailable boxes keep a None score: no returns, zero size, sensor inside.
assert compute_rsv(target, [target], grid.range_image(np.empty((0, 3))), grid)["M"] > 0
assert target.visibility_score is None
for unavailable in (make_box(5, 0, 0, 0, 0, 0), make_box(0, 0, 0, 4, 2, 2)):
    assert compute_rsv(unavailable, [unavailable], grid.range_image(points), grid) is None
    assert unavailable.visibility_score is None

# Rotating and translating the scene about the sensor changes nothing.
yaw = 0.6
c, s = np.cos(yaw), np.sin(yaw)
turned = make_box(20 * c, 20 * s, 0, 4, 2, 1.6, yaw)
turned_wall = make_box(10 * c - 1.0 * s, 10 * s + 1.0 * c, 0, 0.2, 2.0, 4.0, yaw)
compute_rsv(turned, [turned], grid.range_image(scan(grid, [turned, turned_wall])), grid)
assert abs(turned.visibility_score - score_unlabelled) < 0.05

print("RSV ray grid, classification, labelled/unlabelled blockers and unavailable-box checks passed")
