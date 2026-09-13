from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
#!/usr/bin/env python3
"""Small, dependency-light KITTI Velodyne/ground-truth inspector.

Example:
    python kitti/inspect_frame.py --data-root data/kitti --frame-id 000000

KITTI layout expected:
    training/velodyne/<frame>.bin   float32 [x, y, z, reflectance]
    training/label_2/<frame>.txt    one object per line
    training/calib/<frame>.txt      camera/LiDAR calibration
"""

"""
run with 

# python inspect_frame.py \
#      --data-root /home/riley/OpenPCDet/data/kitti \
#      --frame-id 000000


"""



# KITTI's label boxes are in camera coordinates.  This is deliberately kept
# here instead of importing OpenPCDet so the file explains the raw format.
def read_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        numbers = np.fromstring(raw, sep=" ")
        if name in {"R0_rect", "R_rect_00"}:
            values["R0"] = numbers.reshape(3, 3)
        elif name in {"Tr_velo_to_cam", "Tr_velo_cam"}:
            values["T"] = numbers.reshape(3, 4)
    if "R0" not in values or "T" not in values:
        raise ValueError(f"{path} needs R0_rect and Tr_velo_to_cam")
    return values["R0"], values["T"]


def velo_to_camera(points_xyz: np.ndarray, r0: np.ndarray, t_velo_cam: np.ndarray) -> np.ndarray:
    """Transform [N, 3] Velodyne points into KITTI camera coordinates."""
    xyz1 = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    return (r0 @ t_velo_cam[:, :3] @ xyz1[:, :3].T + r0 @ t_velo_cam[:, 3:]).T


def read_labels(path: Path) -> list[dict]:
    objects = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) < 15:
            continue
        # KITTI: type, truncation, occlusion, alpha, 2-D box, h, w, l, x, y, z, ry
        objects.append({
            "line": line_number,
            "class": fields[0],
            "truncation": float(fields[1]),
            "occlusion": int(fields[2]),
            "box_2d": tuple(map(float, fields[4:8])),
            "hwl": tuple(map(float, fields[8:11])),
            "location_camera": tuple(map(float, fields[11:14])),
            "rotation_y": float(fields[14]),
        })
    return objects


def points_in_camera_box(points_camera: np.ndarray, obj: dict) -> np.ndarray:
    """Return a boolean mask for one KITTI camera-coordinate box."""
    h, w, length = obj["hwl"]
    cx, cy, cz = obj["location_camera"]
    ry = obj["rotation_y"]
    dx, dy, dz = (points_camera - (cx, cy, cz)).T
    c, s = np.cos(ry), np.sin(ry)
    local_x = c * dx - s * dz
    local_z = s * dx + c * dz
    return (
        (np.abs(local_x) <= length / 2)
        & (np.abs(dy) <= h / 2)
        & (np.abs(local_z) <= w / 2)
    )


def plot_point_cloud(points: np.ndarray, frame_id: str, ground_truth_mask: np.ndarray) -> None:
    """Show a 3-D plot with ground-truth points highlighted in red."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    background = ~ground_truth_mask
    plot = axis.scatter(
        points[background, 0], points[background, 1], points[background, 2],
        c=points[background, 3], s=0.2, cmap="viridis", label="background",
    )
    if ground_truth_mask.any():
        axis.scatter(
            points[ground_truth_mask, 0], points[ground_truth_mask, 1], points[ground_truth_mask, 2],
            c="red", s=3, label="ground truth",
        )
    axis.set_xlabel("x: forward (m)")
    axis.set_ylabel("y: left (m)")
    axis.set_zlabel("z: up (m)")
    axis.set_title(f"KITTI frame {frame_id} — 3-D Velodyne point cloud")
    figure.colorbar(plot, ax=axis, label="background reflectance")
    axis.legend()
    plt.show()


def inspect_frame(data_root: Path, frame_id: str, show_plot: bool = False) -> None:
    frame_id = str(frame_id).zfill(6)
    root = data_root / "training"
    velo_path = root / "velodyne" / f"{frame_id}.bin"
    label_path = root / "label_2" / f"{frame_id}.txt"
    calib_path = root / "calib" / f"{frame_id}.txt"
    for path in (velo_path, label_path, calib_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file for frame {frame_id}: {path}")

    # Every row is one laser return: x, y, z in metres plus reflectance.
    points = np.fromfile(velo_path, dtype=np.float32)
    if points.size % 4:
        raise ValueError(f"{velo_path} has {points.size} floats; expected a multiple of 4")
    points = points.reshape(-1, 4)
    xyz, reflectance = points[:, :3], points[:, 3]
    r0, transform = read_calibration(calib_path)
    points_camera = velo_to_camera(xyz, r0, transform)
    objects = read_labels(label_path)
    masks = [points_in_camera_box(points_camera, obj) for obj in objects]
    ground_truth_mask = np.logical_or.reduce(masks) if masks else np.zeros(len(points), dtype=bool)
    if show_plot:
        plot_point_cloud(points, frame_id, ground_truth_mask)

    print(f"Frame {frame_id}")
    print(f"  Velodyne file: {velo_path}")
    print(f"  ground-truth file: {label_path}")
    print(f"  calibration file: {calib_path}")
    print(f"  points array: {points.shape} float32 = [x, y, z, reflectance]")
    print(f"  xyz array:    {xyz.shape}; reflectance array: {reflectance.shape}")
    print("  Velodyne: rotating multi-beam LiDAR; each return stores position and intensity.")
    print("  coordinates: LiDAR x=forward, y=left, z=up; labels use camera x=right, y=down, z=forward.")
    print(f"  labels: {len(objects)} objects")

    print("\nPoint-cloud -> ground-truth mapping (same frame id, then point-in-box):")
    if not objects:
        print("  no labels")
        return
    print(f"this many ground truths {len(objects)}")
    for index, (obj, mask) in enumerate(zip(objects, masks)):
        print(
            f"  {frame_id}.bin -> {frame_id}.txt -> box {index} "
            f"({obj['class']}, label line {obj['line']}): "
            f"{int(mask.sum()):,} points, "
            f"camera h/w/l={obj['hwl']}, location={obj['location_camera']}, ry={obj['rotation_y']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--frame-id", help="KITTI frame id, e.g. 000000; defaults to first .bin")
    parser.add_argument("--plot", action="store_true", help="show a top-down Matplotlib point-cloud plot")
    args = parser.parse_args()
    frame_id = args.frame_id
    if frame_id is None:
        files = sorted((args.data_root / "training" / "velodyne").glob("*.bin"))
        if not files:
            raise FileNotFoundError("No .bin files found; pass --data-root or --frame-id")
        frame_id = files[0].stem
    inspect_frame(args.data_root, frame_id, show_plot=args.plot)


if __name__ == "__main__":
    main()
