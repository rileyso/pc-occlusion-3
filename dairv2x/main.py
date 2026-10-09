import numpy as np
from dairv2x.explorer import DAIRV2XExplorer
from mixedsig.tools.visualize_agg_frame import points_inside_box

DATASET_ROOT = "/mnt/c/Datasets/dair-v2x-c-example"
PAIR_INDEX = 0
AGENTS = ("vehicle",)


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


def visible_face_cell_counts(sensor_origin, box, resolution=0.1):
    """Count resolution-sized cells on each face exposed to the sensor."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    dimensions = np.asarray(box[3:6], dtype=float)
    counts = []
    for face in visible_face_areas(sensor_origin, box):
        axis = "xyz".index(face["face"][0])
        spans = [dimensions[i] for i in range(3) if i != axis]
        cells_across = [int(np.ceil(span / resolution - 1e-9))
                        for span in spans]
        counts.append({
            "face": face["face"],
            "cells_across": cells_across,
            "cell_count": cells_across[0] * cells_across[1],
        })
    return counts


def far_edges_and_corners(sensor_origin, box):
    """Return far edges and farthest corners on faces visible to the sensor."""
    sensor_origin = np.asarray(sensor_origin, dtype=float)
    box = np.asarray(box, dtype=float)
    corners = box_corners_in_map(box)
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    yaw = box[6]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    sensor_local = rotation.T @ (sensor_origin - box[:3])
    half = box[3:6] / 2
    result = []

    for axis, axis_name in enumerate(("x", "y", "z")):
        if abs(sensor_local[axis]) <= half[axis]:
            continue

        side = 1 if sensor_local[axis] > 0 else -1
        face_indices = np.flatnonzero(signs[:, axis] == side)
        face_corners = corners[face_indices]
        edges = []
        for i in range(4):
            for j in range(i + 1, 4):
                if np.sum(signs[face_indices[i]] != signs[face_indices[j]]) == 1:
                    edges.append(face_corners[[i, j]])

        far_edge = max(edges, key=lambda edge: np.linalg.norm(
            edge.mean(axis=0) - sensor_origin))
        farthest_corner = max(face_corners, key=lambda point: np.linalg.norm(
            point - sensor_origin))
        result.append({
            "face": f"{axis_name}{'+' if side > 0 else '-'}",
            "far_edge": far_edge.tolist(),
            "farthest_corner": farthest_corner.tolist(),
        })

    return result

def main():

    dataset = DAIRV2XExplorer(DATASET_ROOT)
    # Load single frame
    pair = dataset.pairs[PAIR_INDEX]
    frame_index = dataset.return_labeled_sync_time_ids_of_seq(PAIR_INDEX)[0]

    # Keep only car boxes; labels stay aligned with the filtered geometry.
    labels = dataset._json(pair["cooperative_label_path"])
    all_boxes = dataset.return_gt_boxes_in_map(PAIR_INDEX, frame_index)
    car_indices = [i for i, label in enumerate(labels)
                   if label["type"].lower() == "car"]
    boxes = all_boxes[car_indices]
    labels = [labels[i] for i in car_indices]
    for i, (box, label) in enumerate(zip(boxes, labels)):
        print(f"box {i} label={label['type']} "
              f"{np.round(box[:7], 2).tolist()}")

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
            "box": box[:7].tolist(),
            "points": [
                {"x": float(x), "y": float(y), "z": float(z),
                 "I": float(intensity)}
                for x, y, z, intensity in points_world_with_intensity[inside]
            ],
        }
        print(f"box {box_index} label={label['type']} "
              f"points={int(inside.sum())}")
        areas = visible_face_areas(sensor_pose[:3, 3], box)
        for face in areas:
            print(f"  face {face['face']}={face['area']:.2f}")
            # print("  visible face cells:", visible_face_cell_counts(sensor_pose[:3, 3], box))
            # print("  far edges/corners:", far_edges_and_corners(sensor_pose[:3, 3], box))

if __name__ == '__main__':
    main()






