"""Run: python dairv2xv/test/test_process_boxes.py (no dataset or PCD reader needed)."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explorer import DAIRV2XVehicleExplorer


def label(height, occlusion=0, truncation=0):
    return {"type": "Car", "2d_box": {"ymin": 10, "ymax": 10 + height},
            "occluded_state": occlusion, "truncated_state": truncation}


row = np.array([[10, 20, 30, 4, 2, 2, np.pi / 2, 0, 0]])
# Rotated length is along world Y; test center, faces, and outside points.
points = np.array([[10, 20, 30, 1], [10, 22, 30, 2], [11, 20, 31, 3],
                   [12, 20, 30, 4], [10, 22.01, 30, 5], [10, 20, 31.01, 6],
                   [np.nan, 20, 30, 7]])
with patch.object(DAIRV2XVehicleExplorer, "__init__", return_value=None), \
     patch.object(DAIRV2XVehicleExplorer, "return_available_sequences", return_value=(0, 1)), \
     patch.object(DAIRV2XVehicleExplorer, "return_gt_boxes_in_lidar", side_effect=[row, np.empty((0, 9))]), \
     patch.object(DAIRV2XVehicleExplorer, "return_map_se3_agent", return_value=np.eye(4)), \
     patch.object(DAIRV2XVehicleExplorer, "return_agent_point_cloud", return_value=(points, 0)) as read_cloud, \
     patch.object(DAIRV2XVehicleExplorer, "return_labels", side_effect=[[label(40)], []]), \
     contextlib.redirect_stdout(io.StringIO()) as output:
    result = runpy.run_path(str(Path(__file__).resolve().parents[1] / "process_boxes.py"))

box, = result["boxes"]
assert (box.x, box.y, box.z, box.length, box.width, box.height, box.yaw) == tuple(row[0, :7])
assert (box.category, box.object_index, box.frame) == (0, 0, 0)
assert isinstance(box.object_index, int)
np.testing.assert_array_equal(box.sensor_pose, np.eye(4))
np.testing.assert_array_equal(box.points_inside, points[:3])
assert box.num_points == 3 and box.pixel_height == 40 and box.difficulty == "Easy"
assert read_cloud.call_count == 2
assert box.class_name == "Car"
assert 0 < box.total_occupied_cells <= box.num_points
assert output.getvalue() == (
    "frame: 0 boxes: 1\nframe: 1 boxes: 0\n"
    f"frame: 0, box: 0, points: 3, difficulty label: Easy, "
    f"visibility score: {box.visibility_score:.4f}, class: Car\n"
)

for height, occlusion, truncation, expected in (
    (40, 0, 0, "Easy"), (39.99, 0, 0, "Moderate"),
    (40, 0, 1, "Moderate"), (40, 1, 0, "Moderate"),
    (25, 1, 2, "Moderate"), (25, 2, 2, "Hard"),
    (24.99, 0, 0, "Unclassified"), (100, 3, 0, "Unclassified"),
):
    box.compute_measurements(np.empty((0, 4)), label(height, occlusion, truncation))
    assert box.difficulty == expected, (height, occlusion, truncation)
    assert box.num_points == 0 and box.points_inside.shape == (0, 4)

# A rotated box seen from the LiDAR origin exposes three faces.
assert set(box.visible_surface_grid) == {"x-", "y+", "z-"}
assert box.total_cells == 20 * 20 + 40 * 20 + 40 * 20
c, s = np.cos(box.yaw), np.sin(box.yaw)
rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
for name, face in box.visible_surface_grid.items():
    local = (face["centers"] - [box.x, box.y, box.z]) @ rotation
    axis = "xyz".index(name[0])
    half = np.array([box.length, box.width, box.height]) / 2
    np.testing.assert_allclose(local[..., axis], half[axis] * (1 if name[1] == "+" else -1))
    assert np.all(np.abs(local) <= half + 1e-9)

box.x = box.y = box.z = box.yaw = 0
box.length, box.width, box.height = 2, 0.25, 0.15
box.compute_visible_surface_grid(sensor_origin=(3, 0, 0))
assert set(box.visible_surface_grid) == {"x+"}
assert box.total_cells == 6
face = box.visible_surface_grid["x+"]
assert face["centers"].shape == (3, 2, 3)
np.testing.assert_allclose(np.diff(face["edges"][0]), [0.1, 0.1, 0.05])
np.testing.assert_allclose(np.diff(face["edges"][1]), [0.1, 0.05])
for origin in ((0, 0, 0), (1, 0, 0)):
    box.compute_visible_surface_grid(sensor_origin=origin)
    assert box.visible_surface_grid == {} and box.total_cells == 0
for resolution in (0, -0.1, np.nan):
    try:
        box.compute_visible_surface_grid(resolution=resolution)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid grid resolution accepted")

# Duplicate rays, distinct entry faces, a corner tie, and an outside return.
box.length = box.width = box.height = 2
ray_points = np.array([[-0.8, 0.5, 0], [-0.8, 0.5, 0], [0.5, -0.8, 0],
                       [0, 0, 0], [3, 3, 0]])
box.compute_measurements(ray_points, label(40))
box.compute_surface_occupancy(sensor_origin=(-3, -3, 0), resolution=1)
assert box.num_points == 4
assert box.total_cells == 8 and box.total_occupied_cells == 3
assert box.visibility_score == 3 / 8
np.testing.assert_array_equal(box.occupied_cells["x-"], [[False, True], [False, True]])
np.testing.assert_array_equal(box.occupied_cells["y-"], [[False, False], [False, True]])

# Rotation/translation must not change which box-local cells are occupied.
box.x, box.y, box.z, box.yaw = 10, 20, 30, 0.7
c, s = np.cos(box.yaw), np.sin(box.yaw)
rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
center = np.array([box.x, box.y, box.z])
box.compute_measurements(ray_points @ rotation.T + center, label(40))
origin = np.array([-3, -3, 0]) @ rotation.T + center
box.compute_surface_occupancy(origin, resolution=1)
assert box.total_occupied_cells == 3 and box.visibility_score == 3 / 8
box.compute_measurements(np.empty((0, 4)), label(40))
box.compute_surface_occupancy(origin)
assert box.total_occupied_cells == 0 and box.visibility_score == 0
box.compute_surface_occupancy(center)
assert box.total_cells == 0 and box.visibility_score is None

box.length = box.width = box.height = 0
box.compute_measurements(points, label(40))
assert box.num_points is None and box.points_inside is None
assert box.difficulty == "Easy" and box.pixel_height == 40
box.compute_surface_occupancy()
assert box.visible_surface_grid is None and box.total_cells is None
assert box.occupied_cells is None and box.visibility_score is None

for dimensions in ((-1, 2, 2), (0, 2, 2), (np.nan, 2, 2)):
    box.length, box.width, box.height = dimensions
    try:
        box.compute_measurements(points, label(40))
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid dimensions accepted: {dimensions}")

print("Box measurements, difficulty, surface grids, and ray occupancy checks passed")
