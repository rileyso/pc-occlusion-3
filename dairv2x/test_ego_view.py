import unittest

import numpy as np
import open3d as o3d

from dairv2x.coalign_ego_view import bev_camera, ego_camera, estimate_ground_z


class EgoCameraTest(unittest.TestCase):
    def test_bev_camera_top_down_and_zoom(self):
        camera = bev_camera(-1.8, 3, zoom=0.3)
        pose = np.linalg.inv(camera.extrinsic)
        np.testing.assert_allclose(pose[:3, 3], [0, 0, 1.2], atol=1e-10)
        np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-10)  # looking straight down
        self.assertAlmostEqual(np.linalg.det(camera.extrinsic[:3, :3]), 1)
        eye = pose[:3, 3]
        # Vehicle +X (forward) appears up on screen; +Y (left) appears left.
        forward = camera.extrinsic @ np.r_[eye + [10, 0, 0], 1]
        left = camera.extrinsic @ np.r_[eye + [0, 10, 0], 1]
        self.assertLess(forward[1], 0)
        self.assertLess(left[0], 0)
        base = ego_camera(-1.8).intrinsic.get_focal_length()
        np.testing.assert_allclose(np.asarray(camera.intrinsic.get_focal_length()),
                                   np.asarray(base) * 0.3)
        for ground, height, zoom in ((float('nan'), 2, 1), (-1.8, 0, 1), (-1.8, 2, 0)):
            with self.assertRaises(ValueError):
                bev_camera(ground, height, zoom=zoom)

    def test_ground_height_and_forward_projection(self):
        x, y = np.meshgrid(np.linspace(-8, 8, 20), np.linspace(-8, 8, 20))
        road = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.8)))
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(road))
        ground_z = estimate_ground_z(cloud)
        self.assertAlmostEqual(ground_z, -1.8)
        camera = ego_camera(ground_z)
        eye = np.linalg.inv(camera.extrinsic)[:3, 3]
        np.testing.assert_allclose(eye, [0, 0, 0.2], atol=1e-10)
        for offset, expected in (([10, 0, 0], [0, 0, 10]),
                                 ([10, 1, 0], [-1, 0, 10]),
                                 ([10, 0, 1], [0, -1, 10])):
            projected = camera.extrinsic @ np.r_[eye + offset, 1]
            np.testing.assert_allclose(projected[:3], expected, atol=1e-10)
        self.assertAlmostEqual(np.linalg.inv(ego_camera(-1.8, 3).extrinsic)[2, 3], 1.2)
        for tilt in (-90, -30, 0, 30, 90):
            extrinsic = ego_camera(-1.8, 3, tilt_down=tilt).extrinsic
            pose = np.linalg.inv(extrinsic)
            np.testing.assert_allclose(pose[:3, 3], [0, 0, 1.2], atol=1e-10)
            angle = np.deg2rad(tilt)
            np.testing.assert_allclose(pose[:3, 2], [np.cos(angle), 0, -np.sin(angle)], atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(extrinsic[:3, :3]), 1)
        for zoom in (0.5, 1, 2):
            zoomed = ego_camera(ground_z, zoom=zoom)
            np.testing.assert_allclose(zoomed.extrinsic, camera.extrinsic)
            np.testing.assert_allclose(zoomed.intrinsic.get_focal_length(),
                                       np.array(camera.intrinsic.get_focal_length()) * zoom)
        for zoom in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ego_camera(ground_z, zoom=zoom)
        for tilt in (float('nan'), float('inf'), -91, 91):
            with self.assertRaises(ValueError):
                ego_camera(-1.8, tilt_down=tilt)
        with self.assertRaises(ValueError):
            estimate_ground_z(o3d.geometry.PointCloud())
        for ground, height in ((float('nan'), 2), (-1.8, 0), (-1.8, float('inf'))):
            with self.assertRaises(ValueError):
                ego_camera(ground, height)


if __name__ == "__main__":
    unittest.main()
