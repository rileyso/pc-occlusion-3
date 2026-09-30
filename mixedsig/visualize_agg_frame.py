"""Barebones single-frame aggregate point-cloud visualizer.

Edit the constants below while developing a thesis application.
"""
import numpy as np

from mixedsignals.mixed_signals import MixedSignalsExplorer
from mixedsignals.utils.geometry import apply_se3, apply_se3_
from mixedsignals.utils.mixed_signals_utils import AGENT_COLOR, VIEW_POINT
from mixedsignals.utils.o3d_viz_utils import PointPainter

DATASET_ROOT = "data/mixed-signals-mini"
SEQUENCE_INDEX = 20
LABELED_FRAME_INDEX = 0

# AGENTS = ("top", "dome", "003", "004", "laser")
AGENTS = ("top", "dome", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
VISIBLE_AGENTS = ["laser"]

def points_inside_box(points, box):
    offset = points[:, :3] - box[:3]

    # Rotate points into the box's local coordinate frame.
    cos_yaw = np.cos(box[6])
    sin_yaw = np.sin(box[6])

    local = offset.copy()
    local[:, 0] = cos_yaw * offset[:, 0] + sin_yaw * offset[:, 1]
    local[:, 1] = -sin_yaw * offset[:, 0] + cos_yaw * offset[:, 1]

    return np.all(np.abs(local) <= box[3:6] / 2, axis=1)

def ray_box_intersections(sensor_origin, points, box, margin=0.1):
    """Return intersection mask and entry/exit distances for measured rays.

    ``sensor_origin``, ``points``, and ``box`` must use the same coordinate
    frame. Distances are measured from the sensor in metres; rays that miss
    the box receive ``np.inf`` for both distances.
    """
    sensor_origin = np.asarray(sensor_origin, dtype=float)
    points = np.asarray(points)
    box = np.asarray(box, dtype=float)

    if sensor_origin.shape != (3,):
        raise ValueError("sensor_origin must have shape (3,)")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+)" )
    if box.ndim != 1 or box.size < 7:
        raise ValueError("box must contain x, y, z, dx, dy, dz, yaw")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if np.any(box[3:6] <= 0):
        raise ValueError("box dimensions must be positive")

    ray_vectors = points[:, :3] - sensor_origin
    ray_lengths = np.linalg.norm(ray_vectors, axis=1)
    valid_rays = np.isfinite(ray_lengths) & (ray_lengths > 1e-9)
    ray_directions = np.zeros_like(ray_vectors, dtype=float)
    ray_directions[valid_rays] = (
        ray_vectors[valid_rays] / ray_lengths[valid_rays, None]
    )

    # Transform the sensor and ray directions into the box's local frame.
    cos_yaw = np.cos(box[6])
    sin_yaw = np.sin(box[6])
    box_rotation = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ])
    local_origin = (sensor_origin - box[:3]) @ box_rotation
    local_directions = ray_directions @ box_rotation
    half_size = box[3:6] / 2 + margin

    # Slab intersection: overlap the ray interval for the box's x, y and z.
    parallel = np.abs(local_directions) < 1e-12
    safe_directions = np.where(parallel, 1.0, local_directions)
    first = (-half_size - local_origin) / safe_directions
    second = (half_size - local_origin) / safe_directions
    near = np.where(parallel, -np.inf, np.minimum(first, second))
    far = np.where(parallel, np.inf, np.maximum(first, second))

    entry = near.max(axis=1)
    exit = far.min(axis=1)
    parallel_outside = np.any(
        parallel & (np.abs(local_origin) > half_size), axis=1
    )
    intersects = (
        valid_rays
        & ~parallel_outside
        & (exit >= np.maximum(entry, 0.0))
    )

    entry = np.where(intersects, np.maximum(entry, 0.0), np.inf)
    exit = np.where(intersects, exit, np.inf)
    return intersects, entry, exit

def classify_box_returns(sensor_origin, points, box, margin=0.1):
    """Classify measured returns relative to one ground-truth box.

    Returns masks and distances so callers can define different metrics
    without changing the ray/box geometry.
    """
    sensor_origin = np.asarray(sensor_origin, dtype=float)
    points = np.asarray(points)
    measured_range = np.linalg.norm(points[:, :3] - sensor_origin, axis=1)
    valid = np.isfinite(measured_range) & (measured_range > 1e-9)

    intersects, entry, exit = ray_box_intersections(
        sensor_origin, points, box, margin
    )
    hit = intersects & (measured_range >= entry) & (measured_range <= exit)
    occluded = intersects & (measured_range < entry)
    beyond = intersects & (measured_range > exit)

    return {
        "valid": valid,
        "miss": valid & ~intersects,
        "intersects": intersects,
        "hit": hit,
        "occluded": occluded,
        "beyond": beyond,
        "measured_range": measured_range,
        "entry": entry,
        "exit": exit,
    }


def main():
    msig = MixedSignalsExplorer(DATASET_ROOT, verbose=False)

    labeled_time_ids = msig.return_labeled_sync_time_ids_of_seq(SEQUENCE_INDEX)
    sync_time_idx = labeled_time_ids[LABELED_FRAME_INDEX]

    # Display every available sensor in the TOP coordinate frame.
    agent_names = msig.name_rsu_lidars + tuple(
        msig.return_name_cavs_in_seq(SEQUENCE_INDEX)
    )

    painter = PointPainter()

    all_points = []
    for agent_name in AGENTS:
        points, timestamp = msig.return_agent_point_cloud(
            SEQUENCE_INDEX,
            agent_name,
            sync_time_idx,
        )

        map_se3_agent = msig.return_map_se3_agent(
            SEQUENCE_INDEX,
            agent_name,
            timestamp,
        )

        # Convert the agent point cloud into the TOP frame.
        apply_se3_(msig.top_se3_map @ map_se3_agent, points)
        all_points.append(points)
        # painter.add_pointclouds_(points, AGENT_COLOR[agent_name])
    points = np.concatenate(all_points)

    # Ground truth is returned in MAP coordinates; convert it to TOP.
    gt_boxes = msig.return_gt_boxes_in_map(SEQUENCE_INDEX, sync_time_idx)
    gt_boxes = apply_se3(msig.top_se3_map, boxes_=gt_boxes)

    # ray box intersections
    for agent_name in AGENTS:
        points, timestamp = msig.return_agent_point_cloud(
            SEQUENCE_INDEX,
            agent_name,
            sync_time_idx,
        )
        map_se3_agent = msig.return_map_se3_agent(
            SEQUENCE_INDEX,
            agent_name,
            timestamp,
        )
        # special euclidian 3
        top_se3_agent = msig.top_se3_map @ map_se3_agent
        sensor_origin = top_se3_agent[:3, 3]
        apply_se3_(top_se3_agent, points)

        for box in gt_boxes:
            # intersects, entry, exit = ray_box_intersections(sensor_origin, points, box)
            result = classify_box_returns(sensor_origin, points, box)

            selected_mask = result["hit"] | result["occluded"]
            selected_points = points[selected_mask]
            hit_points = points[result["hit"]]
            occluded_points = points[result["occluded"]]

            if len(hit_points) > 0:
                painter.add_pointclouds_(
                    hit_points,
                    np.array([0.0, 1.0, 0.0]),  # green
                )

            if len(occluded_points) > 0:
                painter.add_pointclouds_(
                    occluded_points,
                    np.array([1.0, 0.0, 0.0]),  # red
                )

            # if len(selected_points) > 0:
            #     painter.add_pointclouds_(selected_points, AGENT_COLOR[agent_name])


    # inside_mask = np.zeros(len(points), dtype=bool)
    # for box in gt_boxes:
    #     inside_mask |= points_inside_box(points, box)
    # points_in_gt = points[inside_mask]

    # painter.add_pointclouds_(
    #     points_in_gt,
    #     np.array([0.0, 1.0, 0.0]),)

    painter.add_boxes_(gt_boxes, np.ones(3))
    painter.show(view_points=VIEW_POINT)


if __name__ == "__main__":
    main()

'''pose shape
    [ R11 R12 R13 tx ]
    [ R21 R22 R23 ty ]
    [ R31 R32 R33 tz ]
    [  0   0   0   1 ]'''