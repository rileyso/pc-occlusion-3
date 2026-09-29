import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gilroy_lidar_replication.evaluate import average_precision, load_instances, ranked_outcomes


class EvaluationTest(unittest.TestCase):
    def test_perfect_ranking_has_perfect_ap(self):
        self.assertEqual(average_precision([(0.9, 1), (0.8, 1)], 2), 1.0)

    def test_controlled_arm_loads_target_and_ignore_boxes(self):
        fields = ["sample_id", "arm", "occlusion_bin", "x", "y", "z", "dx", "dy", "dz", "heading", "ignore_boxes_json"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"sample_id": "a_s", "arm": "structured", "occlusion_bin": 2, "x": 1, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 2, "heading": 0, "ignore_boxes_json": json.dumps([[2, 0, 0, 1, 1, 2, 0]])})
                writer.writerow({"sample_id": "a_r", "arm": "random", "occlusion_bin": 2, "x": 1, "y": 0, "z": 0, "dx": 1, "dy": 1, "dz": 2, "heading": 0, "ignore_boxes_json": "[]"})
            rows = load_instances(path, "structured")
        self.assertEqual([row["frame_id"] for row in rows], ["a_s", "a_s"])
        self.assertEqual([row["occlusion_bin"] for row in rows], [2, -1])

    def test_non_selected_ground_truth_predictions_are_ignored(self):
        prepared = {
            "frame": {
                "gt": [{"occlusion_bin": 0}, {"occlusion_bin": 1}],
                "predictions": [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}],
                "iou": np.asarray([[0.8, 0.0], [0.0, 0.8], [0.0, 0.0]]),
            }
        }
        ranked, count = ranked_outcomes(
            prepared, lambda rows: [index for index, row in enumerate(rows) if row["occlusion_bin"] == 0], 0.5
        )
        self.assertEqual(count, 1)
        self.assertEqual(ranked, [(0.9, 1), (0.7, 0)])


if __name__ == "__main__":
    unittest.main()
