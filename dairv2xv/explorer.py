"""Vehicle-only DAIR-V2X explorer; root is the vehicle-side directory.

Each filename-sorted capture is a one-frame sequence (frame_index=0), not a
track or temporal batch. Raw points and LiDAR labels use vehicle LiDAR axes.
Map poses use DAIR world axes; top_se3_map means world -> selected vehicle
LiDAR for compatibility with the shared visualizer (there is no TOP sensor).
"""

import json
from pathlib import Path

import numpy as np


class DAIRV2XVehicleExplorer:
    name_rsu_lidars = ()
    name_vehicles = ("vehicle",)
    top_se3_map = None

    def __init__(self, root):
        self.root = Path(root)
        self.frames = sorted(self._json("data_info.json"),
                             key=lambda frame: frame["pointcloud_path"])

    def _json(self, path):
        return json.loads((self.root / path).read_text())

    def _frame(self, capture_index, frame_index=0):
        if frame_index != 0:
            raise IndexError("Each capture has only frame_index=0")
        if not 0 <= capture_index < len(self.frames):
            raise IndexError(f"Capture index out of range: {capture_index}")
        return self.frames[capture_index]

    @staticmethod
    def _check_agent(agent_name):
        if agent_name != "vehicle":
            raise ValueError(f"Unknown agent: {agent_name}")

    @staticmethod
    def _pose(calibration):
        calibration = calibration.get("transform", calibration)
        pose = np.eye(4)
        pose[:3, :3] = calibration["rotation"]
        pose[:3, 3] = np.asarray(calibration["translation"]).reshape(3)
        return pose

    def return_available_sequences(self):
        """Capture indices, sorted by point-cloud filename, not batch IDs."""
        return tuple(range(len(self.frames)))

    def return_name_cavs_in_seq(self, capture_index):
        self._frame(capture_index)
        return self.name_vehicles

    def return_labeled_sync_time_ids_of_seq(self, capture_index):
        """Select the display origin and return the capture's sole frame ID."""
        self.top_se3_map = np.linalg.inv(
            self.return_map_se3_agent(capture_index, "vehicle"))
        return (0,)

    def return_agent_point_cloud(self, capture_index, agent_name="vehicle", frame_index=0):
        """Return (N, 4) local XYZ/intensity and integer timestamp in microseconds.

        Intensities are unchanged. Requires the same pypcd reader as the
        cooperative DAIR explorer.
        """
        self._check_agent(agent_name)
        frame = self._frame(capture_index, frame_index)
        from pypcd import pypcd

        data = pypcd.PointCloud.from_path(str(self.root / frame["pointcloud_path"])).pc_data
        points = np.column_stack([data[key] for key in ("x", "y", "z", "intensity")])
        return np.asarray(points, dtype=float), int(frame["pointcloud_timestamp"])

    def return_map_se3_agent(self, capture_index, agent_name="vehicle", timestamp=None):
        """Vehicle LiDAR -> world, composed through NovAtel.

        Only this capture's timestamp is supported; no odometry interpolation.
        """
        self._check_agent(agent_name)
        frame = self._frame(capture_index)
        if timestamp is not None and timestamp != int(frame["pointcloud_timestamp"]):
            raise ValueError("Timestamp does not belong to this capture")
        return (self._pose(self._json(frame["calib_novatel_to_world_path"]))
                @ self._pose(self._json(frame["calib_lidar_to_novatel_path"])))

    def return_gt_boxes_in_lidar(self, capture_index, frame_index=0):
        """Local boxes: x,y,z,l,w,h,yaw,category_placeholder,object_index.

        Category is 0 (unspecified), and object_index is NOT a tracking ID.
        Original class names and other annotations remain in return_labels().
        """
        labels = self.return_labels(capture_index, frame_index)
        boxes = np.empty((len(labels), 9), dtype=float)
        for index, label in enumerate(labels):
            loc, dim = label["3d_location"], label["3d_dimensions"]
            boxes[index] = (loc["x"], loc["y"], loc["z"], dim["l"],
                            dim["w"], dim["h"], label["rotation"], 0, index)
        return boxes

    def return_labels(self, capture_index, frame_index=0):
        """Original vehicle-side LiDAR annotations, not camera/cooperative labels."""
        frame = self._frame(capture_index, frame_index)
        return self._json(frame["label_lidar_std_path"])

    def return_gt_boxes_in_map(self, capture_index, frame_index=0):
        """World boxes in the visualizer's upright, yaw-only representation."""
        boxes = self.return_gt_boxes_in_lidar(capture_index, frame_index)
        pose = self.return_map_se3_agent(capture_index)
        boxes[:, :3] = boxes[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        # ponytail: yaw-only contract drops roll/pitch; use transformed corners
        # if full 3D box orientation is needed. Matches shared apply_se3_.
        boxes[:, 6] += np.arctan2(pose[1, 0], pose[0, 0])
        boxes[:, 6] = np.arctan2(np.sin(boxes[:, 6]), np.cos(boxes[:, 6]))
        return boxes
