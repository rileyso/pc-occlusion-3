from dataclasses import dataclass, field

import numpy as np

from mixedsig.mixed_signals import MixedSignalsExplorer
from mixedsig.utils.geometry import apply_se3, apply_se3_
from mixedsig.utils.mixed_signals_utils import AGENT_COLOR, VIEW_POINT
from mixedsig.utils.o3d_viz_utils import PointPainter

DATASET_ROOT = "/mnt/c/Datasets/mixed-signals-mini"
SEQUENCE_INDEX = 20
LABELED_FRAME_INDEX = 0

AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
# AGENTS = ("top", "dome", "003", "004", "laser")
EGO_AGENT = "laser"
HIT_FRACTION_THRESHOLD = 0.5
VISIBLE_AGENTS = ["laser"]

@dataclass
class ObjectObservation:
    """One object's per-sensor measurements for a single frame."""

    track_id: int
    sensors: dict = field(default_factory=dict)

    def add_sensor(self, agent_name, sensor_origin, box, result, before_fraction):
        """Summarize one sensor's measured returns for this object."""
        hit_count = int(result["hit"].sum())
        before_count = int(result["occluded"].sum())
        beyond_count = int(result["beyond"].sum())
        candidate_rays = hit_count + before_count + beyond_count
        self.sensors[agent_name] = {
            "hit_fraction": hit_count / candidate_rays if candidate_rays else None,
            "range_m": float(np.linalg.norm(box[:3] - sensor_origin)),
            "hit_count": hit_count,
            "before_count": before_count,
            "beyond_count": beyond_count,
            "candidate_rays": candidate_rays,
            "metrics": {"before_return_fraction": before_fraction},
        }

    def __str__(self):
        lines = [f"Object {self.track_id}"]
        for agent, stats in self.sensors.items():
            label = (f"RSU {agent.upper()}" if agent in ("top", "dome", "infrastructure")
                     else f"Vehicle {agent}")
            before_fraction = stats["metrics"]["before_return_fraction"]
            score = "unknown" if before_fraction is None else f"{before_fraction:.2f}"
            lines.append(f"  {label}: before-box returns fraction = {score}")
            # Previous detailed output:
            # hit_fraction = stats["hit_fraction"]
            # score = "unknown" if hit_fraction is None else f"{hit_fraction:.2f}"
            # lines.append(
            #     f"  {label}: V proxy = {score}, range = {stats['range_m']:.2f} m, "
            #     f"points = {stats['hit_count']}, before = {stats['before_count']}, "
            #     f"beyond = {stats['beyond_count']}, candidate rays = {stats['candidate_rays']}"
            # )
        return "\n".join(lines)


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

    # Occluded rays
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


# -----------------------------------------------------------------------------
# ADD NEW PER-SENSOR/OBJECT METRICS HERE.
#
# Metric functions receive the classified returns for one sensor/object pair.
# Return one float, or None when the metric has insufficient evidence.
# Values are stored under observation.sensors[agent]["metrics"].
# -----------------------------------------------------------------------------

def before_return_fraction(returns):
    """Fraction stopping before rather than inside the box."""
    hit_count = int(returns["hit"].sum())
    before_count = int(returns["occluded"].sum())
    denominator = hit_count + before_count
    return before_count / denominator if denominator else None


def main(msig=None, sequence_index=SEQUENCE_INDEX, frame_index=LABELED_FRAME_INDEX,
         agents=AGENTS, ego_agent=EGO_AGENT, view_point=VIEW_POINT, show=True):
    if msig is None:
        msig = MixedSignalsExplorer(DATASET_ROOT, verbose=False)

    labeled_time_ids = msig.return_labeled_sync_time_ids_of_seq(sequence_index)
    sync_time_idx = labeled_time_ids[frame_index]

    # Display every available sensor in the TOP coordinate frame.
    agent_names = msig.name_rsu_lidars + tuple(
        msig.return_name_cavs_in_seq(sequence_index)
    )

    painter = PointPainter()
    sensor_results = []

    # Ground truth is returned in MAP coordinates; convert it to TOP.
    gt_boxes = msig.return_gt_boxes_in_map(sequence_index, sync_time_idx)
    gt_boxes = apply_se3(msig.top_se3_map, boxes_=gt_boxes)
    objects = {
        int(box[8]): ObjectObservation(int(box[8]))
        for box in gt_boxes
    }

    # Process configured sensors that are available in this sequence.
    for agent_name in agents:
        if agent_name not in agent_names:
            print(f"Skipping unavailable sensor: {agent_name}")
            continue
        points, timestamp = msig.return_agent_point_cloud(
            sequence_index,
            agent_name,
            sync_time_idx,
        )
        map_se3_agent = msig.return_map_se3_agent(
            sequence_index,
            agent_name,
            timestamp,
        )
        # se3
        top_se3_agent = msig.top_se3_map @ map_se3_agent
        sensor_origin = top_se3_agent[:3, 3]
        apply_se3_(top_se3_agent, points)

        hit_mask = np.zeros(len(points), dtype=bool)
        before_mask = np.zeros(len(points), dtype=bool)
        box_results = {}

        for box in gt_boxes:
            result = classify_box_returns(sensor_origin, points, box)
            box_results[int(box[8])] = result

            before_fraction = before_return_fraction(result)
            objects[int(box[8])].add_sensor(
                agent_name, sensor_origin, box, result, before_fraction
            )
            hit_mask |= result["hit"]
            before_mask |= result["occluded"]

        # A return inside any box takes priority over being before another box.
        before_mask &= ~hit_mask
        sensor_results.append((points, box_results, hit_mask, before_mask))
    # inside_mask = np.zeros(len(points), dtype=bool)
    # for box in gt_boxes:
    #     inside_mask |= points_inside_box(points, box)
    # points_in_gt = points[inside_mask]

    # painter.add_pointclouds_(
    #     points_in_gt,
    #     np.array([0.0, 1.0, 0.0]),)

    ranked_objects = []
    for observation in objects.values():
        ego = observation.sensors.get(ego_agent, {}).get("hit_fraction")
        partner_scores = [
            stats["hit_fraction"]
            for agent, stats in observation.sensors.items()
            if agent != ego_agent and stats["hit_fraction"] is not None
        ]
        best_partner = max(partner_scores, default=None)
        if ego is not None and best_partner is not None:
            ranked_objects.append((ego, -best_partner, observation, best_partner))

    ranked_objects.sort(key=lambda row: (row[0], row[1]))
    rank_by_track = {
        observation.track_id: (rank, ego, best_partner)
        for rank, (ego, _, observation, best_partner) in enumerate(ranked_objects)
    }

    for observation in objects.values():
        print(observation)

    total_ranked = len(ranked_objects)
    for points, box_results, hit_mask, before_mask in sensor_results:
        for track_id, result in box_results.items():
            rank_info = rank_by_track.get(track_id)
            if rank_info is None:
                color = np.array([0.5, 0.5, 0.5])
            else:
                rank, ego, best_partner = rank_info
                collaboration_rich = (
                    ego < HIT_FRACTION_THRESHOLD
                    and best_partner >= HIT_FRACTION_THRESHOLD
                )
                if collaboration_rich or total_ranked <= 1:
                    color = np.array([0.0, 1.0, 0.0])
                else:
                    position = rank / (total_ranked - 1)
                    color = np.array([
                        min(1.0, 2.0 * position),
                        min(1.0, 2.0 * (1.0 - position)),
                        0.0,
                    ])

            hit = result["hit"]
            before = result["occluded"] & ~hit_mask
            if hit.any():
                painter.add_pointclouds_(points[hit], color)
            if before.any():
                painter.add_pointclouds_(points[before], color)

    painter.add_boxes_(gt_boxes, np.ones(3))
    if show:
        painter.show(view_points=view_point)
    return objects


if __name__ == "__main__":
    main()

'''pose shape
    [ R11 R12 R13 tx ]
    [ R21 R22 R23 ty ]
    [ R31 R32 R33 tz ]
    [  0   0   0   1 ]'''