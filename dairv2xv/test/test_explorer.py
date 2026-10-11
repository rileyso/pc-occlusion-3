"""Run: python dairv2xv/test/test_explorer.py (NumPy only)."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explorer import DAIRV2XVehicleExplorer


def test_explorer():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        frame = {
            "pointcloud_path": "002.pcd", "pointcloud_timestamp": "1234567890123456",
            "label_lidar_std_path": "labels.json",
            "calib_lidar_to_novatel_path": "lidar.json",
            "calib_novatel_to_world_path": "world.json",
        }
        files = {
            "data_info.json": [frame, dict(frame, pointcloud_path="001.pcd",
                                          label_lidar_std_path="empty.json")],
            "lidar.json": {"transform": {"rotation": np.eye(3).tolist(),
                                           "translation": [[1], [0], [0]]}},
            "world.json": {"rotation": rotation, "translation": [10, 20, 30]},
            "labels.json": [{"type": "Car", "3d_location": {"x": 2, "y": 3, "z": 4},
                             "3d_dimensions": {"l": 5, "w": 2, "h": 1}, "rotation": 0.2}],
            "empty.json": [],
        }
        for name, content in files.items():
            (root / name).write_text(json.dumps(content))
        explorer = DAIRV2XVehicleExplorer(root)
        assert explorer.return_available_sequences() == (0, 1)
        assert explorer.return_gt_boxes_in_lidar(0).shape == (0, 9)
        assert explorer.return_gt_boxes_in_map(0).shape == (0, 9)
        assert explorer.return_labels(1)[0]["type"] == "Car"
        assert explorer.return_name_cavs_in_seq(1) == ("vehicle",)
        assert explorer.name_rsu_lidars == ()
        assert explorer.return_labeled_sync_time_ids_of_seq(1) == (0,)
        pose = explorer.return_map_se3_agent(1)
        np.testing.assert_allclose(pose[:3, 3], [10, 21, 30])
        np.testing.assert_allclose(explorer.top_se3_map @ pose, np.eye(4))
        boxes = explorer.return_gt_boxes_in_map(1)
        np.testing.assert_allclose(boxes[0, :3], [7, 23, 34])
        np.testing.assert_allclose(boxes[0, 6], 0.2 + np.pi / 2)
        np.testing.assert_allclose(explorer.return_gt_boxes_in_lidar(1)[0],
                                   [2, 3, 4, 5, 2, 1, 0.2, 0, 0])
        for call in (lambda: explorer.return_labels(-1),
                     lambda: explorer.return_labels(2),
                     lambda: explorer.return_labels(1, 1),
                     lambda: explorer.return_map_se3_agent(1, "infrastructure"),
                     lambda: explorer.return_map_se3_agent(1, timestamp=0)):
            try:
                call()
            except (ValueError, IndexError):
                pass
            else:
                raise AssertionError("Invalid input accepted")

        # Exercise the reader boundary without requiring a PCD dependency.
        def read_cloud(path):
            assert Path(path) == root / "002.pcd"
            return SimpleNamespace(pc_data={"x": [1], "y": [2], "z": [3], "intensity": [42]})

        reader = SimpleNamespace(PointCloud=SimpleNamespace(from_path=read_cloud))
        with patch.dict("sys.modules", {"pypcd": SimpleNamespace(pypcd=reader)}):
            points, timestamp = explorer.return_agent_point_cloud(1)
        np.testing.assert_array_equal(points, [[1, 2, 3, 42]])
        assert timestamp == 1234567890123456
        explorer.return_map_se3_agent(1, timestamp=timestamp)


if __name__ == "__main__":
    test_explorer()
    print("Explorer checks passed")
