import unittest

import numpy as np

from gilroy_lidar_replication.build_benchmark import box_visibility, points_in_box


class BenchmarkGeometryTest(unittest.TestCase):
    def test_front_box_occludes_target_without_using_point_density(self):
        target = np.asarray([10, 0, 0, 1, 1, 2, 0], dtype=float)
        occluder = np.asarray([5, 0, 0, 2, 2, 3, 0], dtype=float)
        clear, clear_head = box_visibility(target, [], resolution=10)
        blocked, blocked_head = box_visibility(target, [occluder], resolution=10)
        self.assertEqual(clear, 0)
        self.assertEqual(clear_head, 100)
        self.assertGreater(blocked, 95)
        self.assertLess(blocked_head, 5)

    def test_points_in_rotated_box(self):
        box = np.asarray([2, 3, 0, 4, 2, 2, np.pi / 2])
        points = np.asarray([[2, 3, 0], [2, 5, 0], [4, 3, 0]])
        np.testing.assert_array_equal(points_in_box(points, box), [True, True, False])


if __name__ == "__main__":
    unittest.main()
