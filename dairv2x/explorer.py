"""DAIR-V2X-C pairs exposed in the frame contract used by visualize_agg_frame.

Boxes and poses start in DAIR world coordinates; the shared visualizer uses
vehicle LiDAR coordinates as its display frame. Pair indices are sorted by
vehicle point-cloud filename, not temporal track identifiers.
"""

import json
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud


class DAIRV2XExplorer:
    name_rsu_lidars = ("infrastructure",)
    top_se3_map = None

    def __init__(self, root):
        self.root = Path(root)
        pairs = json.loads((self.root / "cooperative/data_info.json").read_text())
        self.pairs = sorted(pairs, key=lambda pair: pair["vehicle_pointcloud_path"])

    def _pair(self, pair_index):
        return self.pairs[pair_index]

    def _json(self, path):
        return json.loads((self.root / path).read_text())

    @staticmethod
    def _pose(calibration):
        calibration = calibration.get("transform", calibration)
        pose = np.eye(4)
        pose[:3, :3] = calibration["rotation"]
        pose[:3, 3] = np.asarray(calibration["translation"]).reshape(3)
        return pose

    def _vehicle_pose(self, pair):
        stem = Path(pair["vehicle_pointcloud_path"]).stem
        lidar_to_novatel = self._json(f"vehicle-side/calib/lidar_to_novatel/{stem}.json")
        novatel_to_world = self._json(f"vehicle-side/calib/novatel_to_world/{stem}.json")
        return self._pose(novatel_to_world) @ self._pose(lidar_to_novatel)

    def _infrastructure_pose(self, pair):
        stem = Path(pair["infrastructure_pointcloud_path"]).stem
        calib = self._json(f"infrastructure-side/calib/virtuallidar_to_world/{stem}.json")
        return self._pose(calib)

    def return_labeled_sync_time_ids_of_seq(self, pair_index):
        self.top_se3_map = np.linalg.inv(self._vehicle_pose(self._pair(pair_index)))
        return (0,)

    def return_name_cavs_in_seq(self, pair_index):
        return ("vehicle",)

    def return_gt_boxes_in_map(self, pair_index, frame_index):
        labels = self._json(self._pair(pair_index)["cooperative_label_path"])
        boxes = np.empty((len(labels), 9), dtype=float)
        for index, label in enumerate(labels):
            loc, dim = label["3d_location"], label["3d_dimensions"]
            # DAIR's rotation is not consistently present or in world axes.
            # The 0->3 corner edge follows the box length in world coordinates.
            corners = np.asarray(label["world_8_points"])
            length_axis = corners[3] - corners[0]
            yaw = np.arctan2(length_axis[1], length_axis[0])
            boxes[index] = (loc["x"], loc["y"], loc["z"], dim["l"],
                            dim["w"], dim["h"], yaw, 0, index)
        return boxes

    def return_agent_point_cloud(self, pair_index, agent_name, frame_index):
        pair = self._pair(pair_index)
        path = pair["vehicle_pointcloud_path"] if agent_name == "vehicle" else pair["infrastructure_pointcloud_path"]
        if agent_name not in ("vehicle", "infrastructure"):
            raise ValueError(f"Unknown agent: {agent_name}")
        points = PointCloud.from_path(self.root / path).numpy(("x", "y", "z", "intensity"))
        return np.asarray(points, dtype=float), frame_index

    def return_map_se3_agent(self, pair_index, agent_name, timestamp):
        pair = self._pair(pair_index)
        if agent_name == "vehicle":
            return self._vehicle_pose(pair)
        if agent_name == "infrastructure":
            return self._infrastructure_pose(pair)
        raise ValueError(f"Unknown agent: {agent_name}")
