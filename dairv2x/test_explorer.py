import io
import json
import tempfile
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from dairv2x.coalign_visualisation import visualise
from dairv2x.explorer import DAIRV2XExplorer
from dairv2x.visualize_agg_frame import visualize
from mixedsig.utils.geometry import apply_se3
from mixedsig.utils.o3d_viz_utils import PointPainter


class DAIRV2XExplorerTest(unittest.TestCase):
    def test_world_boxes_and_sensor_poses_share_vehicle_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def save(path, data):
                file = root / path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(json.dumps(data))

            pair = {
                "vehicle_pointcloud_path": "vehicle-side/velodyne/1.pcd",
                "infrastructure_pointcloud_path": "infrastructure-side/velodyne/2.pcd",
                "cooperative_label_path": "cooperative/label_world/1.json",
            }
            save("cooperative/data_info.json", [pair])
            identity = np.eye(3).tolist()
            save("vehicle-side/calib/lidar_to_novatel/1.json", {
                "transform": {"rotation": identity, "translation": [[1], [0], [0]]}})
            save("vehicle-side/calib/novatel_to_world/1.json", {
                "rotation": identity, "translation": [[10], [0], [0]]})
            save("infrastructure-side/calib/virtuallidar_to_world/2.json", {
                "rotation": identity, "translation": [[13], [0], [0]]})
            # World label without a 'rotation' key: 0->3 is the length axis.
            corners = [[15, -2, -1], [13, -2, -1], [13, 2, -1],
                       [15, 2, -1], [15, -2, 1], [13, -2, 1],
                       [13, 2, 1], [15, 2, 1]]
            save("cooperative/label_world/1.json", [{
                "3d_location": {"x": 14, "y": 0, "z": 0},
                "3d_dimensions": {"l": 4, "w": 2, "h": 2},
                "world_8_points": corners,
            }])

            dataset = DAIRV2XExplorer(root)
            self.assertEqual(dataset.return_labeled_sync_time_ids_of_seq(0), (0,))
            box = apply_se3(dataset.top_se3_map, boxes_=dataset.return_gt_boxes_in_map(0, 0))
            np.testing.assert_allclose(box[0, :7], [3, 0, 0, 4, 2, 2, np.pi / 2])
            infra_in_vehicle = dataset.top_se3_map @ dataset.return_map_se3_agent(0, "infrastructure", 0)
            np.testing.assert_allclose(infra_in_vehicle[:3, 3], [2, 0, 0])
            vehicle_in_vehicle = dataset.top_se3_map @ dataset.return_map_se3_agent(0, "vehicle", 0)
            np.testing.assert_allclose(vehicle_in_vehicle, np.eye(4))


class VisualizationTest(unittest.TestCase):
    def test_coalign_colors_keep_all_gt_white(self):
        corners = np.zeros((8, 3)).tolist()
        results = {
            "pair_index": 0, "vehicle_frame_id": "1",
            "infrastructure_pointcloud_path": "infrastructure-side/velodyne/2.pcd",
            "iou_threshold": 0.5,
            "predictions": {"coalign": [
                {"corners_vehicle_lidar": corners},
                {"corners_vehicle_lidar": corners}]},
            "ground_truth": [
                {"corners_vehicle_lidar": corners,
                 "coalign": {"prediction_index": 1, "iou_3d": 0.5}},
                {"corners_vehicle_lidar": corners, "coalign": None}],
        }
        with patch("dairv2x.coalign_visualisation.DAIRV2XExplorer") as factory:
            dataset = factory.return_value
            dataset.pairs = [{"vehicle_pointcloud_path": "vehicle-side/velodyne/1.pcd",
                              "infrastructure_pointcloud_path": results["infrastructure_pointcloud_path"]}]
            dataset.top_se3_map = np.eye(4)
            dataset.return_labeled_sync_time_ids_of_seq.return_value = (0,)
            dataset.return_agent_point_cloud.return_value = (np.zeros((1, 4)), 0)
            dataset.return_map_se3_agent.return_value = np.eye(4)
            with redirect_stdout(io.StringIO()):
                painter = visualise(results, "unused", show=False)
        boxes = painter.objs_to_draw[-4:]
        for box, color in zip(boxes, ([1, 0, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1])):
            np.testing.assert_allclose(np.asarray(box.colors), np.tile(color, (12, 1)))

    def test_default_camera_uses_one_pixel_points(self):
        with patch("mixedsig.utils.o3d_viz_utils.o3d.visualization.Visualizer") as factory:
            PointPainter().show()
            self.assertEqual(factory.return_value.get_render_option.return_value.point_size, 1.0)
            factory.return_value.run.assert_called_once()
            camera = {"lookat": [0, 0, 0], "front": [0, 0, 1],
                      "up": [1, 0, 0], "zoom": 0.8}
            PointPainter().show(camera=camera)
            control = factory.return_value.get_view_control.return_value
            control.set_lookat.assert_called_once_with([0, 0, 0])
            control.set_front.assert_called_once_with([0, 0, 1])
            control.set_zoom.assert_called_once_with(0.8)

    def test_all_points_and_boxes_are_painted_in_vehicle_frame(self):
        class Dataset:
            top_se3_map = np.eye(4)

            def return_labeled_sync_time_ids_of_seq(self, pair_index):
                return (0,)

            def return_agent_point_cloud(self, pair_index, agent_name, frame_index):
                direction = 1 if agent_name == "vehicle" else -1
                return np.array([[1.0, 2.0, 3.0, 0.5],
                                 [2.0 * direction, 0, 0, 0.5],
                                 [5.0 * direction, 0, 0, 0.5],
                                 [8.0 * direction, 0, 0, 0.5]]), 0

            def return_map_se3_agent(self, pair_index, agent_name, timestamp):
                transform = np.eye(4)
                if agent_name == "infrastructure":
                    transform[0, 3] = 10
                return transform

            def return_gt_boxes_in_map(self, pair_index, frame_index):
                return np.array([[5.0, 0, 0, 4, 2, 2, 0, 0, 0]])

        class Painter:
            def __init__(self):
                self.clouds = []
                self.boxes = []

            def add_pointclouds_(self, points, color):
                self.clouds.append(points.copy())

            def add_boxes_(self, boxes, color):
                self.boxes.append(boxes.copy())

            def show(self, camera=None):
                self.camera = camera

        painter = Painter()
        output = io.StringIO()
        with redirect_stdout(output):
            visualize(Dataset(), 0, show=False, painter=painter)
        for agent in ("vehicle", "infrastructure"):
            self.assertIn(f"Box 0 | {agent}: before=1/3, inside=1/3, "
                          "beyond=1/3, before/(before+inside)=0.50", output.getvalue())
        self.assertEqual(len(painter.clouds), 2)
        np.testing.assert_allclose(painter.clouds[0][0, :3], [1, 2, 3])
        np.testing.assert_allclose(painter.clouds[1][0, :3], [11, 2, 3])
        self.assertEqual(len(painter.boxes), 1)
        np.testing.assert_allclose(painter.boxes[0][0, :7], [5, 0, 0, 4, 2, 2, 0])
        with redirect_stdout(io.StringIO()):
            visualize(Dataset(), 0, painter=painter, zoom=0.9)
        self.assertEqual(painter.camera["lookat"], [0, 0, 0])
        self.assertEqual(painter.camera["front"], [0, 0, 1])
        self.assertEqual(painter.camera["zoom"], 0.9)


if __name__ == "__main__":
    unittest.main()
