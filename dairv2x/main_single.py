from pathlib import Path
from dairv2x.explorer import DAIRV2XExplorer
from mixedsig.tools.visualize_agg_frame import points_inside_box
import numpy as np

DATASET_ROOT = "/mnt/c/Datasets/dair-v2x-c-example"
PAIR_INDEX = 0
AGENTS = ("vehicle", "infrastructure")


def annotation_key(label):
    """Identify a source-camera annotation copied into cooperative labels."""
    box = label.get("2d_box") or {}
    return (label.get("type", "").lower(),
            *(box.get(name) for name in ("xmin", "ymin", "xmax", "ymax")))


def pascal_difficulty(label):
    """Return the easiest PASCAL difficulty met by a DAIR ground truth."""
    box = label.get("2d_box") or {}
    try:
        height = float(box["ymax"]) - float(box["ymin"])
        occlusion = int(label["occluded_state"])
    except (KeyError, TypeError, ValueError):
        return None

    try:
        untruncated = int(label["truncated_state"]) == 0
    except (KeyError, TypeError, ValueError):
        untruncated = False

    if height >= 40 and occlusion == 0 and untruncated:
        return "easy"
    if height >= 25 and occlusion <= 1:
        return "moderate"
    if height >= 25 and occlusion <= 2:
        return "hard"
    return None


def box_corners_in_map(box):
    """Return the eight world/map-frame corners of a yawed box."""
    box = np.asarray(box, dtype=float)
    if box.size < 7 or np.any(box[3:6] <= 0):
        raise ValueError("box must contain positive dx, dy, dz dimensions")

    half = box[3:6] / 2
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = signs * half
    yaw = box[6]
    # Rotation around Z
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    return local @ rotation.T + box[:3]

def visible_face_areas(sensor_origin, box):
    """Return the areas of box faces visible from a map-frame sensor origin."""
    sensor_origin = np.asarray(sensor_origin, dtype=float)
    box = np.asarray(box, dtype=float)
    if sensor_origin.shape != (3,):
        raise ValueError("sensor_origin must have shape (3,)")

    yaw = box[6]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    # converts sensor position from map coordinates into box local coordiantes
    sensor_local = rotation.T @ (sensor_origin - box[:3])
    half = box[3:6] / 2
    areas = []
    for axis, axis_name in enumerate(("x", "y", "z")):
        if abs(sensor_local[axis]) > half[axis]:
            side = "+" if sensor_local[axis] > 0 else "-"
            areas.append({
                "face": f"{axis_name}{side}",
                "area": float(np.prod(box[3:6]) / box[3 + axis]),
            })
    return areas

def visible_face_occupancy(sensor_origin, box, lidar_points, resolution=0.01):
    """Return occupied grid cells for each sensor-facing box face.

    Points are projected onto each visible face in the box's local frame.
    A set of ``(row, column)`` indices makes multiple LiDAR returns in one
    cell count once.
    """
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    box = np.asarray(box, dtype=float)
    points = np.asarray(lidar_points, dtype=float)
    if box.size < 7 or np.any(box[3:6] <= 0):
        raise ValueError("box must contain positive dimensions")
    if points.size == 0:
        points = np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("lidar_points must have shape (N, 3+)")

    yaw = box[6]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    local_points = (points[:, :3] - box[:3]) @ rotation
    half = box[3:6] / 2
    inside = np.all(np.abs(local_points) <= half + 1e-9, axis=1)
    result = []

    for face in visible_face_areas(sensor_origin, box):
        axis = "xyz".index(face["face"][0])
        tangent_axes = [i for i in range(3) if i != axis]
        spans = box[3 + np.array(tangent_axes)]
        cells_across = np.ceil(spans / resolution - 1e-9).astype(int)
        projected = local_points[inside][:, tangent_axes]
        indices = np.floor((projected + spans / 2) / resolution).astype(int)
        indices = np.minimum(indices, cells_across - 1)
        occupied = {tuple(index) for index in indices}
        total_cells = int(np.prod(cells_across))
        result.append({
            "face": face["face"],
            "cells_across": cells_across.tolist(),
            "occupied_cells": sorted(occupied),
            "occupied_count": len(occupied),
            "total_cells": total_cells,
            "occupancy": len(occupied) / total_cells,
        })
    return result

def main():


    dataset = DAIRV2XExplorer(DATASET_ROOT)
    # Load single frame
    
    pair = dataset.pairs[PAIR_INDEX]
    frame_index = dataset.return_labeled_sync_time_ids_of_seq(PAIR_INDEX)[0]

    # Keep only car boxes; labels stay aligned with the filtered geometry.
    labels = dataset._json(pair["cooperative_label_path"])
    vehicle_frame = Path(pair["vehicle_image_path"]).stem
    vehicle_labels = dataset._json(
        f"vehicle-side/label/camera/{vehicle_frame}.json"
    )
    vehicle_annotations = {annotation_key(label) for label in vehicle_labels}
    all_boxes = dataset.return_gt_boxes_in_map(PAIR_INDEX, frame_index)
    car_indices = [i for i, label in enumerate(labels)
                   if label["type"].lower() == "car"]
    boxes = all_boxes[car_indices]
    labels = [labels[i] for i in car_indices]
    # for i, (box, label) in enumerate(zip(boxes, labels)):
    #     print(f"box {i} label={label['type']} "
    #           f"{np.round(box[:7], 2).tolist()}")

    # SE(3) sensor pose -> origin world
    poses = {agent: dataset.return_map_se3_agent(PAIR_INDEX, agent, frame_index)
                for agent in AGENTS}

    sensor_pose = poses["vehicle"]
    print(f"sensor origin {np.round(sensor_pose[:3, 3], 2)}")

    # Point clouds are sensor-local; transform them into the world frame so they
    # can be tested against the world-frame boxes.
    points, _ = dataset.return_agent_point_cloud(PAIR_INDEX, "vehicle", frame_index)
    xyz1 = np.c_[points[:, :3], np.ones(len(points))]
    points_world = (poses["vehicle"] @ xyz1.T).T[:, :3]
    points_world_with_intensity = np.column_stack((points_world, points[:, 3]))

    # Each entry contains one box and all vehicle points inside it. Point
    # coordinates and intensity are stored in the world/map frame.
    box_points = {}
    for box_index, (box, label) in enumerate(zip(boxes, labels)):
        inside = points_inside_box(points_world, box)
        box_points[box_index] = {
            "label": label["type"],
            "difficulty": (pascal_difficulty(label)
                           if annotation_key(label) in vehicle_annotations else None),
            "box": box[:7].tolist(),
            "points": [
                {"x": float(x), "y": float(y), "z": float(z),
                 "I": float(intensity)}
                for x, y, z, intensity in points_world_with_intensity[inside]
            ],
        }
        print(f"box {box_index} label={label['type']} "
              f"difficulty={box_points[box_index]['difficulty']} "
              f"points={int(inside.sum())}")
        areas = visible_face_areas(sensor_pose[:3, 3], box)
        occupancy = visible_face_occupancy(
            sensor_pose[:3, 3], box, points_world[inside], resolution=0.5
        )
        occupied_cells = 0
        total_cells = 0
        for face, cells in zip(areas, occupancy):
            # print(f"  face {face['face']} ={face['area']:.2f}")
            # print(f"  occupancy={cells}")
            # print(f"total cells {cells["total_cells"]}")
            # print(f"occ cells {cells["occupied_count"]}")
            occupied_cells += cells["occupied_count"]
            total_cells += cells["total_cells"]
        V = 0.0
        if total_cells:
            V = occupied_cells / total_cells
        print(f"{V:.2f}")

if __name__ == '__main__':
    main()