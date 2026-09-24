from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
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
DATA_ROOT = Path('/home/riso7312/OpenPCDet/data/kitti')
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
        # print(fields)
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
    """Return a boolean mask for one KITTI camera-coordinate box.

    KITTI object locations are the bottom centre of the box, not its centre.
    """
    h, w, length = obj["hwl"]
    cx, cy, cz = obj["location_camera"]
    ry = obj["rotation_y"]
    dx, dy, dz = (points_camera - (cx, cy - h / 2, cz)).T
    c, s = np.cos(ry), np.sin(ry)
    local_x = c * dx - s * dz
    local_z = s * dx + c * dz
    return (
        (np.abs(local_x) <= length / 2)
        & (np.abs(dy) <= h / 2)
        & (np.abs(local_z) <= w / 2)
    )


def camera_box_corners(obj: dict) -> np.ndarray:
    """Return the eight corners of a KITTI box in camera coordinates."""
    h, w, length = obj["hwl"]
    cx, cy, cz = obj["location_camera"]
    ry = obj["rotation_y"]
    local = np.array([
        [x, y, z]
        for x in (-length / 2, length / 2)
        for y in (-h, 0.0)
        for z in (-w / 2, w / 2)
    ])
    c, s = np.cos(ry), np.sin(ry)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return local @ rotation.T + np.array([cx, cy, cz])


def camera_to_velo(points_camera: np.ndarray, r0: np.ndarray, t_velo_cam: np.ndarray) -> np.ndarray:
    """Transform [N, 3] camera points into Velodyne coordinates."""
    rotation = r0 @ t_velo_cam[:, :3]
    translation = r0 @ t_velo_cam[:, 3]
    return (np.asarray(points_camera) - translation) @ rotation


def ray_box_entry_distance(direction: np.ndarray, obj: dict, r0: np.ndarray,
                           t_velo_cam: np.ndarray) -> float | None:
    """Return the forward entry distance for a LiDAR ray and camera box."""
    rotation = r0 @ t_velo_cam[:, :3]
    translation = r0 @ t_velo_cam[:, 3]
    origin_camera = translation - np.asarray(obj["location_camera"])
    ray_camera = rotation @ direction
    c, s = np.cos(obj["rotation_y"]), np.sin(obj["rotation_y"])
    box_rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    origin_local = box_rotation.T @ origin_camera
    direction_local = box_rotation.T @ ray_camera
    h, w, length = obj["hwl"]
    lower = np.array([-length / 2, -h, -w / 2])
    upper = np.array([length / 2, 0.0, w / 2])
    near, far = 0.0, np.inf
    for origin, direction, lo, hi in zip(origin_local, direction_local, lower, upper):
        if abs(direction) < 1e-12:
            if origin < lo or origin > hi:
                return None
            continue
        first, last = sorted(((lo - origin) / direction, (hi - origin) / direction))
        near, far = max(near, first), min(far, last)
        if near > far:
            return None
    return near if far >= near else None


def object_ray_directions(obj: dict, r0: np.ndarray, t_velo_cam: np.ndarray,
                          azimuth_resolution: float, elevation_resolution: float) -> list[np.ndarray]:
    """Create angularly binned rays whose centres intersect a camera box."""
    corners = camera_to_velo(camera_box_corners(obj), r0, t_velo_cam)
    azimuth = np.arctan2(corners[:, 1], corners[:, 0])
    reference = np.arctan2(np.mean(np.sin(azimuth)), np.mean(np.cos(azimuth)))
    unwrapped = reference + np.arctan2(np.sin(azimuth - reference), np.cos(azimuth - reference))
    horizontal = np.hypot(corners[:, 0], corners[:, 1])
    elevation = np.arctan2(corners[:, 2], horizontal)
    centre = camera_to_velo(np.asarray([obj["location_camera"]]), r0, t_velo_cam)[0]
    centre_azimuth = np.arctan2(centre[1], centre[0])
    centre_elevation = np.arctan2(centre[2], np.hypot(centre[0], centre[1]))
    azimuths = np.unique(np.r_[
        np.arange(unwrapped.min(), unwrapped.max() + azimuth_resolution / 2, azimuth_resolution),
        centre_azimuth,
    ])
    elevations = np.unique(np.r_[
        np.arange(elevation.min(), elevation.max() + elevation_resolution / 2, elevation_resolution),
        centre_elevation,
    ])
    directions = []
    for az in azimuths:
        for el in elevations:
            direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            if ray_box_entry_distance(direction, obj, r0, t_velo_cam) is not None:
                directions.append(direction)
    return directions


def compute_ray_occlusion(objects: list[dict], r0: np.ndarray, t_velo_cam: np.ndarray,
                           azimuth_resolution: float = np.deg2rad(0.1),
                           elevation_resolution: float = np.deg2rad(0.1)) -> list[dict]:
    """Estimate per-object geometric occlusion using rays from the LiDAR origin."""
    objects = [obj for obj in objects if obj["class"] != "DontCare"]
    results = []
    for target_index, target in enumerate(objects):
        directions = object_ray_directions(target, r0, t_velo_cam,
                                           azimuth_resolution, elevation_resolution)
        occluded = 0
        blockers = {}
        for direction in directions:
            distances = [ray_box_entry_distance(direction, obj, r0, t_velo_cam)
                         for obj in objects]
            hit = [distance if distance is not None else np.inf for distance in distances]
            nearest = int(np.argmin(hit))
            if nearest != target_index:
                occluded += 1
                blockers[objects[nearest]["class"]] = blockers.get(objects[nearest]["class"], 0) + 1
        total = len(directions)
        results.append({
            "object": target,
            "target_rays": total,
            "visible_rays": total - occluded,
            "occluded_rays": occluded,
            "occlusion_percent": 100.0 * occluded / total if total else 0.0,
            "blockers": blockers,
        })
    return results


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


def inspect_frame(data_root: Path, frame_id: str, show_plot: bool = False,
                  ray_trace: bool = False, ray_resolution: float = 0.1) -> None:
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
    # print(f"  ground-truth file: {label_path}")
    # print(f"  calibration file: {calib_path}")
    # # print(f"  points array: {points.shape} float32 = [x, y, z, reflectance]")
    # # print(f"  xyz array:    {xyz.shape}; reflectance array: {reflectance.shape}")
    # # print("  Velodyne: rotating multi-beam LiDAR; each return stores position and intensity.")
    # # print("  coordinates: LiDAR x=forward, y=left, z=up; labels use camera x=right, y=down, z=forward.")
    print(f"  labels: {len(objects)} objects")

    print("\nPoint-cloud -> ground-truth mapping (same frame id, then point-in-box):")
    if not objects:
        print("  no labels")
        return

    
    for index, (obj, mask) in enumerate(zip(objects, masks)):
        print(
            f"  {frame_id}.bin -> {frame_id}.txt -> box {index} "
            f"({obj['class']}, label line {obj['line']}): "
            f"{int(mask.sum()):,} points, "
            f"camera h/w/l={obj['hwl']}, location={obj['location_camera']}, ry={obj['rotation_y']:.3f}"
        )

    if ray_trace:
        ray_results = compute_ray_occlusion(
            objects, r0, transform,
            azimuth_resolution=np.deg2rad(ray_resolution),
            elevation_resolution=np.deg2rad(ray_resolution),
        )
        print("\nGeometric ray-tracing occlusion:")
        for result in ray_results:
            obj = result["object"]
            blockers = ", ".join(f"{name}={count}" for name, count in result["blockers"].items()) or "none"
            print(
                f"  {obj['class']} (label line {obj['line']}): "
                f"{result['occlusion_percent']:.1f}% occluded "
                f"({result['occluded_rays']}/{result['target_rays']} rays); "
                f"blockers: {blockers}"
            )

def inspect(data_root: Path, frame_id: str, show_plot: bool = False) -> None:
    """Print the number of labels and DontCare regions for the first 3 KITTI frames."""
    N = 5
    root = data_root / "training"
    velo_files = sorted((root / "velodyne").glob("*.bin"))
    if not velo_files:
        raise FileNotFoundError(f"No Velodyne files found in {root / 'velodyne'}")

    for velo_path in velo_files[:10]:
        frame_id = velo_path.stem
        label_path = root / "label_2" / f"{frame_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file for frame {frame_id}: {label_path}")

        labels = read_labels(label_path)
        # dontcare_count = sum(label["class"] == "DontCare" for label in labels)
        print(f"Frame {frame_id}: {len(labels)} labels; ")
        # for label in labels:
        #     print(label['class'])
        # print(labels[0]['class'])


def inspect_dont_care(data_root=DATA_ROOT):
    root = data_root / 'training'
    velo_files = sorted((root / 'velodyne').glob('*.bin'))
    if not velo_files:
        raise FileNotFoundError(f'No Velodyne files found in {root / "velodyne"}')

    frame_ids = []
    dontcare_counts = []
    class_names = set()

    for velo_path in velo_files[:]:
        frame_id = velo_path.stem
        label_path = root / 'label_2' / f'{frame_id}.txt'
        if not label_path.is_file():
            raise FileNotFoundError(f'Missing label file for frame {frame_id}: {label_path}')

        labels = read_labels(label_path)
        # print(f"Frame {frame_id}: {len(labels)} labels; ")
        # for label in labels:
        #     print(label['class'])
        class_names.update(label['class'] for label in labels)
        frame_ids.append(int(frame_id))
        dontcare_counts.append(sum(label['class'] == 'DontCare' for label in labels))
    print('Classes found:', sorted(class_names))

    # print(f'Frames with DontCare: {sum(count > 0 for count in dontcare_counts)}')
    # print(f'Total DontCare labels: {sum(dontcare_counts)}')
    plt.figure(figsize=(16, 5))
    plt.bar(frame_ids, dontcare_counts, width=1.0)
    plt.xlabel('Frame ID')
    plt.ylabel('Number of DontCare labels')
    plt.title('DontCare labels per KITTI frame')
    plt.xticks(frame_ids[::500], rotation=90)
    plt.tight_layout()
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/riley/OpenPCDet/data/kitti"))
    parser.add_argument("--frame-id", help="KITTI frame id, e.g. 000000; defaults to first .bin")
    parser.add_argument("--plot", action="store_true", help="show a top-down Matplotlib point-cloud plot")
    parser.add_argument("--ray-trace", action="store_true", help="compute geometric occlusion from LiDAR-origin rays")
    parser.add_argument("--ray-resolution", type=float, default=0.1, help="ray angular resolution in degrees")
    args = parser.parse_args()
    frame_id = args.frame_id

    if frame_id is None:
        files = sorted((args.data_root / "training" / "velodyne").glob("*.bin"))
        if not files:
            raise FileNotFoundError("No .bin files found; pass --data-root or --frame-id")
        frame_id = files[0].stem

    if args.frame_id is not None or args.ray_trace:
        inspect_frame(
            args.data_root, frame_id, show_plot=args.plot,
            ray_trace=args.ray_trace, ray_resolution=args.ray_resolution,
        )
    else:
        inspect_dont_care(args.data_root)



if __name__ == "__main__":
    main()
