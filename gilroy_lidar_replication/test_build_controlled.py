import unittest

import numpy as np

from gilroy_lidar_replication.build_controlled import apply_control, structured_cutoff


class ControlledOcclusionTest(unittest.TestCase):
    def setUp(self):
        self.box = np.asarray([10, 0, 0, 2, 2, 2, 0], dtype=float)
        rng = np.random.default_rng(4)
        target = rng.uniform([-0.9, -0.9, -0.9], [0.9, 0.9, 0.9], size=(500, 3)) + self.box[:3]
        self.points = np.column_stack([target, np.ones(len(target))])

    def test_surface_cutoff_tracks_requested_level(self):
        _, achieved, _ = structured_cutoff(self.box, 0.55, "bottom", resolution=40)
        self.assertAlmostEqual(achieved, 0.55, delta=0.03)

    def test_structured_and_random_arms_retain_equal_counts(self):
        _, structured = apply_control(self.points, self.box, 0.65, "left", "structured", 7)
        _, random = apply_control(self.points, self.box, 0.65, "left", "random", 7)
        self.assertEqual(structured["removed_target_points"], random["removed_target_points"])
        self.assertEqual(structured["retained_target_points"], random["retained_target_points"])


if __name__ == "__main__":
    unittest.main()
