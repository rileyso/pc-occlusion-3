from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from inspect_frame import camera_to_velo, ray_box_entry_distance, read_calibration, read_labels

MAX_FRAMES = 7481
SURFACE_SAMPLE_SPACING = 0.5
SURFACE_TOLERANCE = 0.02
RESULTS_PATH = Path("kitti/data/visible_surface_area_results.pkl")


def _surface_samples(obj: dict, spacing: float) -> list[tuple[np.ndarray, float]]:
    """Return local surface sample points and the area represented by each."""
    height, width, length = obj["hwl"]
    faces = [
        # (varying dimensions, varying axes, fixed axis, fixed coordinate)
        (length, width, "x", "z", "y", -height),
        (length, width, "x", "z", "y", 0.0),
        (height, width, "y", "z", "x", -length / 2),
        (height, width, "y", "z", "x", length / 2),
        (height, length, "y", "x", "z", -width / 2),
        (height, length, "y", "x", "z", width / 2),
    ]
    samples = []
    for first_size, second_size, first_axis, second_axis, fixed_axis, fixed in faces:
        first_count = max(1, int(np.ceil(first_size / spacing)))
        second_count = max(1, int(np.ceil(second_size / spacing)))
        first_values = np.linspace(
            -first_size / 2 + first_size / (2 * first_count),
            first_size / 2 - first_size / (2 * first_count),
            first_count,
        )
        second_values = np.linspace(
            -second_size / 2 + second_size / (2 * second_count),
            second_size / 2 - second_size / (2 * second_count),
            second_count,
        )
        area_per_sample = first_size * second_size / (first_count * second_count)
        for first in first_values:
            for second in second_values:
                point = {"x": 0.0, "y": 0.0, "z": 0.0}
                point[first_axis] = first
                point[second_axis] = second
                point[fixed_axis] = fixed
                samples.append((np.array([point["x"], point["y"], point["z"]]), area_per_sample))
    return samples


def visible_surface_area_frame(
    data_root: str | Path,
    frame_id: str | int,
    spacing: float = SURFACE_SAMPLE_SPACING,
) -> dict[int, dict]:
    """Estimate each ground-truth box's LiDAR-visible surface area."""
    frame_id = str(frame_id).zfill(6)
    root = Path(data_root) / "training"
    label_path = root / "label_2" / f"{frame_id}.txt"
    calib_path = root / "calib" / f"{frame_id}.txt"
    for path in (label_path, calib_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file for frame {frame_id}: {path}")

    objects = read_labels(label_path)
    traceable = [
        (index, obj) for index, obj in enumerate(objects)
        if obj["class"] != "DontCare"
    ]
    r0, transform = read_calibration(calib_path)
    results = {}

    for original_index, target in traceable:
        height, width, length = target["hwl"]
        total_area = 2.0 * (height * width + height * length + width * length)
        viewable_area = 0.0
        visible_area = 0.0
        blocked_area = 0.0
        visible_samples = 0
        blocked_samples = 0
        blockers = {}
        target_rotation = target["rotation_y"]
        c, s = np.cos(target_rotation), np.sin(target_rotation)
        rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        centre = np.asarray(target["location_camera"])

        for local_point, sample_area in _surface_samples(target, spacing):
            point_camera = rotation @ local_point + centre
            point_velo = camera_to_velo(point_camera[None], r0, transform)[0]
            distance = float(np.linalg.norm(point_velo))
            if distance <= 0.0:
                continue
            direction = point_velo / distance
            target_entry = ray_box_entry_distance(direction, target, r0, transform)
            if target_entry is None or abs(target_entry - distance) > SURFACE_TOLERANCE:
                continue

            viewable_area += sample_area
            blocker = None
            blocker_distance = np.inf
            for _, other in traceable:
                if other is target:
                    continue
                entry = ray_box_entry_distance(direction, other, r0, transform)
                if entry is not None and entry < blocker_distance:
                    blocker = other
                    blocker_distance = entry
            if blocker is not None and blocker_distance < distance - SURFACE_TOLERANCE:
                blocked_area += sample_area
                blocked_samples += 1
                blockers[blocker["class"]] = blockers.get(blocker["class"], 0) + 1
            else:
                visible_area += sample_area
                visible_samples += 1

        visible_percent = 100.0 * visible_area / viewable_area if viewable_area else 0.0
        occluded_percent = 100.0 * blocked_area / viewable_area if viewable_area else 0.0
        results[original_index] = {
            "index": original_index,
            "class": target["class"],
            "object": target,
            "surface_area": total_area,
            "viewable_surface_area": viewable_area,
            "visible_surface_area": visible_area,
            "occluded_surface_area": blocked_area,
            "visible_surface_percent": visible_percent,
            "occluded_surface_percent": occluded_percent,
            # Keeps this pickle compatible with q1.py's current plotter.
            "occlusion_percent": occluded_percent,
            "surface_samples": visible_samples + blocked_samples,
            "visible_samples": visible_samples,
            "blocked_samples": blocked_samples,
            "blockers": blockers,
        }
    return results


def visible_surface_area_all_frames(
    data_root: str | Path,
    spacing: float = SURFACE_SAMPLE_SPACING,
    max_frames: int = MAX_FRAMES,
) -> dict[str, dict[int, dict]]:
    """Compute visible surface metrics for every available KITTI frame."""
    root = Path(data_root) / "training"
    label_dir = root / "label_2"
    calib_dir = root / "calib"
    frame_ids = sorted(
        path.stem for path in label_dir.glob("*.txt")
        if (calib_dir / path.name).is_file()
    )[:max_frames]
    results = {}
    for number, frame_id in enumerate(frame_ids, 1):
        results[frame_id] = visible_surface_area_frame(data_root, frame_id, spacing)
        print(f"Processed {number}/{len(frame_ids)}: {frame_id}")
    return results


def save_results(results: dict[str, dict[int, dict]], path: str | Path = RESULTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    results = visible_surface_area_all_frames("/home/riley/OpenPCDet/data/kitti")
    save_results(results)
    print(f"Processed {len(results)} frames")
    print(f"Saved results to {RESULTS_PATH}")
