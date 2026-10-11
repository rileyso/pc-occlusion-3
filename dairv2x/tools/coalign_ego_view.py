"""Show saved CoAlign results from the ego LiDAR's XY position, facing +X.

The camera starts 2 m above estimated local ground, not 2 m above the LiDAR.
DAIR vehicle LiDAR coordinates are X forward, Y left, Z up. Box colors and
1 px points match coalign_visualisation. Override uncertain ground with --ground-z.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from dairv2x.tools.coalign_visualisation import visualise


def estimate_ground_z(cloud):
    points = np.asarray(cloud.points)
    nearby = (np.linalg.norm(points[:, :2], axis=1) < 10) & (points[:, 2] < 0)
    local = cloud.select_by_index(np.flatnonzero(nearby))
    if len(local.points) < 30:
        raise ValueError("Too few nearby ground candidates; supply --ground-z in LiDAR coordinates")
    o3d.utility.random.seed(303)
    plane, inliers = local.segment_plane(distance_threshold=0.15, ransac_n=3,
                                        num_iterations=1000)
    # ponytail: dominant nearby plane approximates road; use --ground-z on curbs/uneven terrain.
    if abs(plane[2]) < 0.9 or len(inliers) < max(30, 0.3 * len(local.points)):
        raise ValueError("No reliable near-horizontal ground plane; supply --ground-z")
    return float(-plane[3] / plane[2])


def ego_camera(ground_z, height=2.0, width=1280, height_px=720, tilt_down=0.0, zoom=1.0):
    if not np.isfinite(ground_z) or not np.isfinite(height) or height <= 0:
        raise ValueError("Ground height must be finite and camera height must be positive")
    if not np.isfinite(tilt_down) or not -90 <= tilt_down <= 90:
        raise ValueError("Tilt must be finite and between -90 and 90 degrees")
    if not np.isfinite(zoom) or zoom <= 0:
        raise ValueError("Zoom must be finite and positive")
    eye = np.array([0.0, 0.0, ground_z + height])
    # Open3D pinhole axes: right, down, forward; vehicle axes: forward, left, up.
    angle = np.deg2rad(tilt_down)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[0., -1., 0.], [-s, 0., -c], [c, 0., -s]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    camera = o3d.camera.PinholeCameraParameters()
    focal = zoom * height_px / (2 * np.tan(np.deg2rad(60) / 2))
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height_px, focal, focal, (width - 1) / 2, (height_px - 1) / 2)
    camera.extrinsic = extrinsic
    return camera


def bev_camera(ground_z, height=2.0, width=1280, height_px=720, zoom=1.0):
    """Top-down camera directly above the ego, looking straight down at -Z."""
    if not np.isfinite(ground_z) or not np.isfinite(height) or height <= 0:
        raise ValueError("Ground height must be finite and camera height must be positive")
    if not np.isfinite(zoom) or zoom <= 0:
        raise ValueError("Zoom must be finite and positive")
    eye = np.array([0.0, 0.0, ground_z + height])
    # Camera right = world -Y, camera down = world -X, camera forward = world -Z,
    # so vehicle +X (forward) appears up on screen and +Y (left) appears left.
    rotation = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    camera = o3d.camera.PinholeCameraParameters()
    focal = zoom * height_px / (2 * np.tan(np.deg2rad(60) / 2))
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height_px, focal, focal, (width - 1) / 2, (height_px - 1) / 2)
    camera.extrinsic = extrinsic
    return camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/coalign_full"))
    parser.add_argument("--height", type=float, default=5.0, help="Camera height above ground in metres")
    parser.add_argument("--tilt-down", type=float, default=15.0,
                        help="Pitch in degrees: positive looks down, negative looks up (-90 to 90)")
    parser.add_argument("--zoom", type=float, default=0.3,
                        help="Focal-length multiplier: 1 normal, 2 closer, 0.5 wider; camera position stays fixed")
    parser.add_argument("--ground-z", type=float, help="Override ground Z at the ego origin in LiDAR coordinates")
    parser.add_argument("--no-show", action="store_true", help="Check geometry and camera without a window")
    args = parser.parse_args()
    if not np.isfinite(args.height) or args.height <= 0:
        parser.error("--height must be finite and positive")
    if not np.isfinite(args.tilt_down) or not -90 <= args.tilt_down <= 90:
        parser.error("--tilt-down must be finite and between -90 and 90 degrees")
    if not np.isfinite(args.zoom) or args.zoom <= 0:
        parser.error("--zoom must be finite and positive")
    if args.ground_z is not None and not np.isfinite(args.ground_z):
        parser.error("--ground-z must be finite")
    painter = visualise(json.loads(args.results.read_text()), args.root, show=False)
    # The existing visualizer adds vehicle points first, then infrastructure points.
    ground_z = args.ground_z if args.ground_z is not None else estimate_ground_z(painter.objs_to_draw[0])
    camera = ego_camera(ground_z, args.height, tilt_down=args.tilt_down, zoom=args.zoom)
    print(f"Ground Z: {ground_z:.3f} m; camera Z: {ground_z + args.height:.3f} m "
          f"({args.height:g} m above ground); heading: vehicle +X; tilt down: {args.tilt_down:g} degrees")
    if args.no_show:
        return
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name="CoAlign ego view", width=1280, height=720):
        raise RuntimeError("Could not create Open3D window")
    try:
        for geometry in painter.objs_to_draw:
            vis.add_geometry(geometry)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)
        control = vis.get_view_control()
        if not control.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True):
            raise RuntimeError("Could not set ego camera")
        control.set_constant_z_near(0.1)
        control.set_constant_z_far(500.0)
        vis.run()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
