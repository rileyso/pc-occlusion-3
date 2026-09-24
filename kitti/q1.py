from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from inspect_frame import compute_ray_occlusion, read_calibration, read_labels

MAX_FRAMES = 7481
MATCH_IOU_THRESHOLD = 0.5
PLOT_OUTPUT_PATH = Path("kitti/images/confidence_distance_occlusion.png")
RAY_RESULTS_PATH = Path("kitti/data/ray_results_7481.pkl")
DETECTOR_RESULTS_PATH = Path(
    "/home/riley/OpenPCDet/output/kitti_models/pointpillar/default/"
    "eval/epoch_no_number/val/baseline_check/final_result/data"
)


def load_detector_results(
    results_path: str | Path = DETECTOR_RESULTS_PATH,
    max_frames: int = MAX_FRAMES,
) -> dict[str, list[dict]]:
    """Load KITTI detector result files, keyed by frame ID."""
    results_path = Path(results_path)
    if not results_path.is_dir():
        raise FileNotFoundError(f"Detector results directory not found: {results_path}")
    loaded = {}
    frame_paths = sorted(results_path.glob("*.txt"))[:max_frames]

    for frame_path in frame_paths:
        detections = []
        for index, line in enumerate(frame_path.read_text().splitlines()):
            fields = line.split()
            if len(fields) < 16:
                continue
            detections.append({
                "index": index,
                "class": fields[0],
                "truncation": float(fields[1]),
                "occlusion": int(float(fields[2])),
                "alpha": float(fields[3]),
                "box_2d": np.array(fields[4:8], dtype=float),
                "hwl": np.array(fields[8:11], dtype=float),
                "location_camera": np.array(fields[11:14], dtype=float),
                "rotation_y": float(fields[14]),
                "score": float(fields[15]),
            })
        loaded[frame_path.stem] = detections

    return loaded


def box_iou_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Return IoU for two KITTI [left, top, right, bottom] boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_detector_to_ray_metrics(
    detector_results: dict[str, list[dict]],
    ray_results: dict[str, dict[int, dict]],
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> list[dict]:
    """Match detections to same-class ground truth boxes using greedy 2-D IoU."""
    matches = []
    for frame_id, detections in detector_results.items():
        ground_truth = ray_results.get(frame_id, {})
        used_ground_truth = set()
        for detection in detections:
            candidates = []
            for index, metric in ground_truth.items():
                if index in used_ground_truth or metric["class"] != detection["class"]:
                    continue
                iou = box_iou_2d(
                    detection["box_2d"],
                    np.asarray(metric["object"]["box_2d"]),
                )
                if iou >= iou_threshold:
                    candidates.append((iou, index, metric))
            if not candidates:
                continue
            iou, ground_truth_index, metric = max(candidates)
            used_ground_truth.add(ground_truth_index)
            location = np.asarray(metric["object"]["location_camera"])
            matches.append({
                "frame_id": frame_id,
                "detection_index": detection["index"],
                "ground_truth_index": ground_truth_index,
                "class": detection["class"],
                "distance": float(np.linalg.norm(location)),
                "confidence": detection["score"],
                "occlusion_percent": metric["occlusion_percent"],
                "kitti_occlusion_label": int(metric["object"]["occlusion"]),
                "iou_2d": float(iou),
            })
    return matches


def plot_confidence_distance_occlusion(
    matches: list[dict],
    output_path: str | Path = PLOT_OUTPUT_PATH,
) -> None:
    """Save a 3-D plot: distance, confidence, and ray occlusion percentage."""
    import matplotlib.pyplot as plt

    if not matches:
        raise ValueError("No detector/ground-truth matches available to plot")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.lines import Line2D

    distances = np.asarray([match["distance"] for match in matches])
    confidence = np.asarray([match["confidence"] for match in matches])
    occlusion = np.asarray([match["occlusion_percent"] for match in matches])
    kitti_labels = np.asarray([
        match["kitti_occlusion_label"] for match in matches
    ])
    colors = {0: "tab:green", 1: "tab:orange", 2: "tab:red", 3: "tab:gray"}
    names = {
        0: "0: fully visible",
        1: "1: partly occluded",
        2: "2: largely occluded",
        3: "3: unknown",
    }
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    for label, color in colors.items():
        selected = kitti_labels == label
        if selected.any():
            axis.scatter(
                distances[selected], occlusion[selected], confidence[selected],
                color=color, s=18, alpha=0.7,
            )
    axis.set_xlabel("Ground-truth distance (m)")
    axis.set_ylabel("Ray occlusion (%)")
    axis.set_zlabel("Detector confidence")
    axis.set_title("Detector confidence, distance, and geometric occlusion")
    legend = [
        Line2D([0], [0], marker="o", color="w", label=names[label],
               markerfacecolor=color, markersize=8)
        for label, color in colors.items()
        if (kitti_labels == label).any()
    ]
    axis.legend(handles=legend, title="KITTI occlusion label")
    plt.show()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_ray_results(
    results: dict[str, dict[int, dict]],
    output_path: str | Path = RAY_RESULTS_PATH,
) -> None:
    """Save computed ray-tracing results for reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_ray_results(
    input_path: str | Path = RAY_RESULTS_PATH,
) -> dict[str, dict[int, dict]]:
    """Load previously saved ray-tracing results."""
    with Path(input_path).open("rb") as file:
        return pickle.load(file)


def ray_trace_frame(
    data_root: str | Path,
    frame_id: str | int,
    ray_resolution: float = 0.1,
) -> dict[int, dict]:
    """Return ray-tracing metrics for every ground-truth object in one KITTI frame.

    The result is keyed by the object's zero-based position in ``label_2``.
    Each value contains the object index, class, KITTI metadata, and geometric
    ray-occlusion metrics from ``compute_ray_occlusion``.
    """
    frame_id = str(frame_id).zfill(6)
    root = Path(data_root) / "training"
    label_path = root / "label_2" / f"{frame_id}.txt"
    calib_path = root / "calib" / f"{frame_id}.txt"

    for path in (label_path, calib_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file for frame {frame_id}: {path}")

    objects = read_labels(label_path)
    r0, transform = read_calibration(calib_path)
    metrics = compute_ray_occlusion(
        objects,
        r0,
        transform,
        azimuth_resolution=np.deg2rad(ray_resolution),
        elevation_resolution=np.deg2rad(ray_resolution),
    )
    traceable_indices = [
        index for index, obj in enumerate(objects) if obj["class"] != "DontCare"
    ]

    return {
        index: {
            "index": index,
            "class": result["object"]["class"],
            "object": result["object"],
            "target_rays": result["target_rays"],
            "visible_rays": result["visible_rays"],
            "occluded_rays": result["occluded_rays"],
            "occlusion_percent": result["occlusion_percent"],
            "blockers": result["blockers"],
        }
        for index, result in zip(traceable_indices, metrics)
    }


def ray_trace_all_frames(
    data_root: str | Path,
    ray_resolution: float = 0.5,
) -> dict[str, dict[int, dict]]:
    """Return ray-tracing metrics for every labelled KITTI frame."""
    root = Path(data_root) / "training"
    label_dir = root / "label_2"
    calib_dir = root / "calib"
    frame_ids = sorted(
        path.stem
        for path in label_dir.glob("*.txt")
        if (calib_dir / path.name).is_file()
    )[:MAX_FRAMES]

    results = {}
    for number, frame_id in enumerate(frame_ids, 1):
        results[frame_id] = ray_trace_frame(data_root, frame_id, ray_resolution)
        print(f"Processed {number}/{len(frame_ids)}: {frame_id}")
    return results


if __name__ == "__main__":
    # results = ray_trace_all_frames("/home/riley/OpenPCDet/data/kitti")
    # save_ray_results(results)  # Comment out after the first run.
    results = load_ray_results()  # Use this after saving the results.
    detector_results = load_detector_results()
    matches = match_detector_to_ray_metrics(detector_results, results)
    plot_confidence_distance_occlusion(matches)
    print(f"Processed {len(results)} frames")
    print(f"Loaded detector results for {len(detector_results)} frames")
    print(f"Matched {len(matches)} detections")
    print(f"Saved plot to {PLOT_OUTPUT_PATH}")
