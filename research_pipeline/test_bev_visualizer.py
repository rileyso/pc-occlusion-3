import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_pipeline import bev_visualizer as bev


class BevVisualizerTest(unittest.TestCase):
    def test_axis_aligned_and_rotated_box_corners(self):
        boxes = np.asarray([
            [1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.0, np.pi / 2],
        ], dtype=np.float32)
        corners = bev.box_bev_corners(boxes)
        np.testing.assert_allclose(
            corners[0],
            np.asarray([[3.0, 3.0], [3.0, 1.0], [-1.0, 1.0], [-1.0, 3.0]]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.sort(np.abs(corners[1]), axis=0),
            np.sort(np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]), axis=0),
            atol=1e-6,
        )

    def test_save_bev_writes_nonempty_image(self):
        points = np.asarray([[0, 0, 0], [2, 1, 0.5], [4, -1, -0.5]], dtype=np.float32)
        gt_boxes = np.asarray([[2, 0, 0, 3, 2, 1, 0]], dtype=np.float32)
        pred_boxes = np.asarray([[2.2, 0.1, 0, 3, 2, 1, 0.1], [20, 20, 0, 2, 2, 1, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "bev.png"
            resolved = bev.save_bev(
                output, points, gt_boxes=gt_boxes, pred_boxes=pred_boxes,
                pred_scores=np.asarray([0.9, 0.1]), score_threshold=0.5,
                xlim=(-5, 10), ylim=(-5, 5), show_scores=True,
            )
            self.assertEqual(resolved, output.resolve())
            self.assertGreater(output.stat().st_size, 1_000)

    def test_ego_range_view_swaps_axes_and_uses_ground_range(self):
        points = np.asarray([[10, -2, 1], [30, 5, 2]], dtype=np.float32)
        figure, axis = bev.plot_bev(
            points, color_by="ground_range", ego_view=True, show_legend=False, show_range_rings=True,
            source_label="training/velodyne/000001.bin",
        )
        self.assertEqual(axis.get_xlabel(), "y left (m)")
        self.assertEqual(axis.get_ylabel(), "x forward (m)")
        np.testing.assert_allclose(axis.get_xlim(), [-35, 35])
        np.testing.assert_allclose(axis.get_ylim(), [0, 70])
        np.testing.assert_allclose(axis.collections[0].get_array(), [np.hypot(10, -2), np.hypot(30, 5)])
        self.assertIn("Source: training/velodyne/000001.bin", [text.get_text() for text in axis.texts])
        bev.plt.close(figure)

    def test_geometry_gallery_selects_range_ordered_examples_without_gaps(self):
        objects = [
            {"frame_id": str(index), "gt_index": 0, "ground_range": float(index)}
            for index in range(10)
        ]
        selected = bev.select_geometry_gallery(objects, max_panels=4)
        self.assertEqual([item["frame_id"] for item in selected], ["0", "3", "6", "9"])

    def test_cli_loaders_support_bin_npy_and_pipeline_json(self):
        points = np.asarray([[1, 2, 3, 0.5], [4, 5, 6, 0.7]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            binary_path = root / "points.bin"
            points.tofile(binary_path)
            np.testing.assert_array_equal(bev.load_array(binary_path, 4), points)

            boxes = np.asarray([[1, 2, 0, 4, 2, 1, 0]], dtype=np.float32)
            box_path = root / "boxes.npy"
            np.save(box_path, boxes)
            np.testing.assert_array_equal(bev.load_array(box_path), boxes)

            json_path = root / "predictions.json"
            json_path.write_text(json.dumps({"detections": [
                {"frame_id": "frame", "box_lidar": boxes[0].tolist(), "score": 0.8},
                {"frame_id": "other", "box_lidar": boxes[0].tolist(), "score": 0.2},
            ]}), encoding="utf-8")
            loaded_boxes, scores = bev.load_predictions(json_path, "frame")
            np.testing.assert_array_equal(loaded_boxes, boxes)
            np.testing.assert_allclose(scores, [0.8])

            output = root / "cli.png"
            exit_code = bev.main([
                str(binary_path), "--point-features", "4", "--pred-boxes", str(json_path),
                "--frame-id", "frame",
                "--output", str(output),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
