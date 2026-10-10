"""Play cached CoAlign predictions as an ego-centric scene.

Geometry is added once and mutated in place using Open3D's non-blocking
visualization pattern (reassign buffers, then update_geometry) — no per-frame
clear/re-add and no camera reset. The camera is constant because every cached
frame is already in that frame's vehicle-LiDAR coordinates, so the ego vehicle
is always at the origin facing +X.

python -m dairv2x.play_sequence \
     --sequence data/coalign_sequences/batch_106 \
     --root data/coalign_full
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from dairv2x.coalign_ego_view import bev_camera, ego_camera, estimate_ground_z
from dairv2x.coalign_visualisation import EDGES, build_scene
from dairv2x.explorer import DAIRV2XExplorer

# GLFW key codes and action codes used by VisualizerWithKeyCallback.
KEY_SPACE, KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 32, 263, 262, 265, 264
KEY_ESC, KEY_Q, KEY_R = 256, 81, 82
PRESS = 1
VOXEL = 0.2  # matches PointPainter's downsampling in the single-frame views


def frame_index_at(timestamps_us, elapsed_us):
    """Largest index whose timestamp is at most first-timestamp + elapsed."""
    index = int(np.searchsorted(timestamps_us, timestamps_us[0] + elapsed_us, side="right")) - 1
    return int(max(0, min(index, len(timestamps_us) - 1)))


def downsample(points, color):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float)[:, :3])
    cloud = cloud.voxel_down_sample(voxel_size=VOXEL)
    cloud.paint_uniform_color(color)
    return cloud


class Player:
    def __init__(self, frames, speed=1.0):
        self.frames = frames
        self.timestamps = np.asarray([f["timestamp_us"] for f in frames], dtype=np.int64)
        self.index = 0
        self.speed = speed
        self.paused = True  # start paused so the first frame is visible before motion
        self.quit = False
        self.step = 0
        self.sim_elapsed_us = 0
        self._last = time.perf_counter()

    def _tick(self):
        now = time.perf_counter()
        if not self.paused:
            self.sim_elapsed_us += (now - self._last) * self.speed * 1e6
        self._last = now

    def current_index(self):
        if self.step:
            self.index += self.step
            self.step = 0
            self.index = int(np.clip(self.index, 0, len(self.frames) - 1))
            self.sim_elapsed_us = self.timestamps[self.index] - self.timestamps[0]
        elif not self.paused:
            self.index = frame_index_at(self.timestamps, self.sim_elapsed_us)
        return self.index

    def toggle_pause(self):
        self.paused = not self.paused

    def restart(self):
        self.index = 0
        self.sim_elapsed_us = 0

    def faster(self):
        self.speed *= 2.0

    def slower(self):
        self.speed /= 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=Path("data/coalign_sequences/batch_106"),
                        help="cache directory containing manifest.json and frame JSONs")
    parser.add_argument("--root", type=Path, default=Path("data/coalign_full"))
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--tilt-down", type=float, default=20.0)
    parser.add_argument("--zoom", type=float, default=0.4)
    parser.add_argument("--view", choices=("ego", "bev"), default="ego",
                        help="bev: top-down over the ego; ego: forward-facing from the ego position")
    parser.add_argument("--prediction", choices=("coalign", "vehicle_only"), default="coalign",
                        help="prediction set to display from each cached frame")
    parser.add_argument("--ground-z", type=float)
    parser.add_argument("--show-ego", action="store_true",
                        help="also draw boxes containing the ego LiDAR origin")
    parser.add_argument("--start", type=int, default=0, help="frame index to start at")
    parser.add_argument("--end", type=int, help="frame index to stop before")
    parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed multiplier")
    args = parser.parse_args()
    if args.view == "bev":
        args.tilt_down = 0.0  # tilt does not apply to a top-down camera
    for name, value in (("--height", args.height), ("--zoom", args.zoom), ("--speed", args.speed)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(args.tilt_down) or not -90 <= args.tilt_down <= 90:
        parser.error("--tilt-down must be between -90 and 90")

    manifest = json.loads((args.sequence / "manifest.json").read_text())
    frames = manifest["frames"][args.start:args.end]
    if not frames:
        parser.error("empty frame range")

    def load(frame):
        path = args.sequence / frame["prediction_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name="CoAlign scene playback", width=1280, height=720):
        raise RuntimeError("Could not create Open3D window")
    try:
        dataset = DAIRV2XExplorer(args.root)
        clouds = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        box_pool = []
        MAX_BOXES = 128

        def scene(frame):
            frame_clouds, boxes = build_scene(
                load(frame), args.root, dataset, hide_ego=not args.show_ego,
                prediction_key=args.prediction)
            return [downsample(points, color) for points, color in frame_clouds], boxes

        def update_clouds(rendered):
            for cloud, points in zip(clouds, rendered):
                cloud.points = points.points
                cloud.colors = points.colors
                vis.update_geometry(cloud)

        def update_boxes(boxes):
            if len(boxes) > MAX_BOXES:
                raise RuntimeError(f"Frame has {len(boxes)} boxes, pool is {MAX_BOXES}; raise MAX_BOXES")
            for index, (corners, color) in enumerate(boxes):
                geom = box_pool[index]
                geom.points = o3d.utility.Vector3dVector(np.asarray(corners, dtype=float))
                geom.paint_uniform_color(color)
                vis.update_geometry(geom)
            for index in range(len(boxes), MAX_BOXES):
                geom = box_pool[index]
                geom.points = o3d.utility.Vector3dVector(np.zeros((8, 3)))
                geom.paint_uniform_color([0, 0, 0])
                vis.update_geometry(geom)

        # Add every geometry before setting the camera: Open3D re-fits its view
        # on add_geometry, so add_geometry must never run during playback.
        rendered, boxes = scene(frames[0])
        for cloud, points in zip(clouds, rendered):
            cloud.points = points.points
            cloud.colors = points.colors
            vis.add_geometry(cloud)
        for _ in range(MAX_BOXES):
            geom = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.zeros((8, 3))),
                lines=o3d.utility.Vector2iVector(EDGES))
            geom.paint_uniform_color([0, 0, 0])
            vis.add_geometry(geom)
            box_pool.append(geom)
        update_boxes(boxes)

        player = Player(frames, args.speed)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

        ground_z = args.ground_z if args.ground_z is not None else estimate_ground_z(clouds[0])
        camera = (bev_camera(ground_z, args.height, zoom=args.zoom)
                  if args.view == "bev" else
                  ego_camera(ground_z, args.height, tilt_down=args.tilt_down, zoom=args.zoom))
        control = vis.get_view_control()
        if not control.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True):
            raise RuntimeError("Could not set ego camera")
        control.set_constant_z_near(0.1)
        control.set_constant_z_far(500.0)

        def callback(handler):
            def wrapped(*args):
                action = args[1] if len(args) >= 2 else PRESS
                if action != PRESS:
                    return False
                handler()
                return True
            return wrapped

        vis.register_key_callback(KEY_SPACE, callback(player.toggle_pause))
        vis.register_key_callback(KEY_LEFT, callback(lambda: setattr(player, "step", -1)))
        vis.register_key_callback(KEY_RIGHT, callback(lambda: setattr(player, "step", 1)))
        vis.register_key_callback(KEY_UP, callback(player.faster))
        vis.register_key_callback(KEY_DOWN, callback(player.slower))
        vis.register_key_callback(KEY_R, callback(player.restart))
        for key in (KEY_ESC, KEY_Q):
            vis.register_key_callback(key, callback(lambda: setattr(player, "quit", True)))

        shown = 0
        print(f"Ground Z {ground_z:.3f}; {args.view} view, {args.height:g} m above ground, zoom {args.zoom:g}.")
        print("Space: play/pause | Left/Right: step | Up/Down: speed | R: restart | Esc/Q: quit")
        while not player.quit:
            player._tick()
            index = player.current_index()
            if index != shown:
                rendered, boxes = scene(frames[index])
                update_clouds(rendered)
                update_boxes(boxes)
                shown = index
                frame = frames[index]
                print(f"[{index + 1}/{len(frames)}] {frame['vehicle_frame_id']} "
                      f"(t={frame['timestamp_us'] / 1e6:.3f}s) speed={player.speed:g} "
                      f"{'paused' if player.paused else 'playing'}")
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
