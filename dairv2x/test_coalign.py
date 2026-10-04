"""Pure matching check; no model or GPU required."""

import unittest

from dairv2x.coalign import match_boxes


class TestMatching(unittest.TestCase):
    def test_one_prediction_cannot_match_two_ground_truth_boxes(self):
        matches = match_boxes([[0.8, 0.1], [0.7, 0.6]], [0.9, 0.7], 0.5)
        self.assertEqual([match["prediction_index"] for match in matches], [0, 1])
        self.assertEqual(match_boxes([[0.4]], [0.9], 0.5), [None])
        self.assertEqual(match_boxes([[]], [], 0.5), [None])


if __name__ == "__main__":
    unittest.main()
