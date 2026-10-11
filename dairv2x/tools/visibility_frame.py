"""Definition-3 3D object visibility (Luo et al., arXiv:2403.03681) for DAIR-V2X-C.

Run: PYTHONPATH=. python -m dairv2x.visibility_frame

Pipeline (each step is a function below; `main` prints every stage):

  1. frame       pick a cooperative pair -> vehicle/RSU frame ids, label file
  2. boxes       load labelled 3D boxes in the DAIR world frame
  3. poses       calibration chain -> SE(3) pose of each LiDAR in world;
                 the sensor origin is the translation column
  4. frames      (check) move boxes into each sensor frame with the inverse
                 pose; visibility must not change, it is frame-invariant
  5. samples     per target box, sample its sensor-facing faces; each sample
                 carries its solid angle dOmega = cos(incidence) dA / r^2
  6. occlusion   cast a geometric ray from the sensor origin to each sample
                 and test it against the *other labelled boxes*
  7. visibility  V = unblocked solid angle / total solid angle

Only boxes and sensor positions are used. No LiDAR point cloud is loaded, so
measured returns (e.g. returns stopping before a box) play no part; the rays
in step 6 are numerical-integration rays, not LiDAR beams.
"""
import numpy as np

from dairv2x.explorer import DAIRV2XExplorer
from mixedsig.tools.visualize_agg_frame import points_inside_box, ray_box_intersections

DATASET_ROOT = "/mnt/c/Datasets/dair-v2x-c-example"
PAIR_INDEX = 0
SAMPLES_PER_EDGE = 64
METRIC_NAME = "3D object visibility (Definition 3: solid-angle visibility)"
AGENTS = ("vehicle", "infrastructure")


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Step 4 ---------------------------------------------------------------------
def boxes_in_sensor_frame(boxes, sensor_pose):
    """Apply T_sensor<-world = inv(T_world<-sensor) to box centres and yaw.

    Boxes are yaw-only, so any sensor roll/pitch cannot be represented and is
    dropped; this causes small differences versus world-frame results.
    """
    world_to_sensor = np.linalg.inv(sensor_pose)
    rotation = world_to_sensor[:3, :3]
    moved = np.array(boxes, dtype=float)
    moved[:, :3] = boxes[:, :3] @ rotation.T + world_to_sensor[:3, 3]
    moved[:, 6] += np.arctan2(rotation[1, 0], rotation[0, 0])
    return moved


# Step 5 ---------------------------------------------------------------------
def facing_face_samples(box, origin, samples_per_edge):
    """Midpoint samples on the faces of `box` that face `origin`.

    Returns (points (M, 3), solid_angle_weights (M,)). The weights sum to the
    box's solid angle Omega in steradians: sensor-facing faces cover the
    box's silhouette on the unit sphere exactly once.
    """
    rotation = yaw_rotation(box[6])
    local_origin = (origin - box[:3]) @ rotation
    half = box[3:6] / 2
    grid = (np.arange(samples_per_edge) + 0.5) / samples_per_edge * 2 - 1
    u, v = (g.ravel() for g in np.meshgrid(grid, grid))
    all_points, all_weights = [], []
    for axis in range(3):
        # At most one face per axis faces an external observer.
        if abs(local_origin[axis]) <= half[axis]:
            continue
        a, b = [k for k in range(3) if k != axis]
        points = np.empty((u.size, 3))
        points[:, axis] = np.sign(local_origin[axis]) * half[axis]
        points[:, a] = u * half[a]
        points[:, b] = v * half[b]
        points = points @ rotation.T + box[:3]
        ranges = np.linalg.norm(points - origin, axis=1)
        face_distance = abs(local_origin[axis]) - half[axis]  # = r cos(incidence)
        cell_area = box[3 + a] * box[3 + b] / u.size
        all_points.append(points)
        all_weights.append(face_distance * cell_area / ranges**3)
    return np.concatenate(all_points), np.concatenate(all_weights)


# Step 6 ---------------------------------------------------------------------
def occluder_candidates(boxes, origin, target, unavailable, ordering):
    """Indices of boxes allowed to hide `target`.

    "centre" is the paper's literal rule: boxes whose centre is nearer the
    sensor. "ray" keeps any box that could reach in front of the target and
    lets `blocked_samples` decide per ray which surface is hit first.
    """
    distances = np.linalg.norm(boxes[:, :3] - origin, axis=1)
    if ordering == "centre":
        allowed = distances < distances[target]
    else:
        radii = np.linalg.norm(boxes[:, 3:6], axis=1) / 2
        allowed = distances - radii < distances[target] + radii[target]
    allowed &= ~unavailable
    allowed[target] = False
    return np.flatnonzero(allowed)


def blocked_samples(points, origin, occluder, ordering):
    """True where the sensor->sample ray passes through `occluder`.

    With ordering="ray" the occluder must also be entered before the sample
    (depth test), so a box behind the target never hides it.
    """
    intersects, entry, _ = ray_box_intersections(origin, points, occluder, margin=0.0)
    if ordering == "centre":
        return intersects
    return intersects & (entry < np.linalg.norm(points - origin, axis=1) - 1e-6)


# Steps 5-7 for one target ---------------------------------------------------
def target_breakdown(boxes, origin, target, unavailable, samples_per_edge, ordering):
    """Solid angle, visible solid angle and per-occluder blocked solid angle.

    Blocked samples are a boolean union across occluders, so two overlapping
    occluders never count the same direction twice; per-occluder values are
    attribution only and may overlap.
    """
    points, weights = facing_face_samples(boxes[target], origin, samples_per_edge)
    blocked = np.zeros(len(points), dtype=bool)
    by_occluder = {}
    for j in occluder_candidates(boxes, origin, target, unavailable, ordering):
        hit = blocked_samples(points, origin, boxes[j], ordering)
        if hit.any():
            by_occluder[int(j)] = weights[hit].sum()
        blocked |= hit
    omega = weights.sum()
    visible = weights[~blocked].sum()
    return {"omega": omega, "visible": visible, "visibility": visible / omega,
            "samples": len(points), "blocked_samples": int(blocked.sum()),
            "by_occluder": by_occluder}


def solid_angle_visibility(boxes, sensor_origin, samples_per_edge=64, ordering="ray"):
    """Return one visibility in [0, 1] per box; NaN if sensor is in/on it.

    boxes: (N, 7+) rows [x, y, z, length, width, height, yaw, ...] (box
    centre, yaw-only, radians). All geometry must share a coordinate frame.

    Implements V_i = area(S_i minus union(S_j for nearer j)) / area(S_i) on
    the unit sphere. ordering="ray" (default) decides 'nearer' per ray; the
    paper's ordering="centre" compares box centres, which fails for long
    boxes (a truck whose centre is farther can still be in front).
    Touching/coincident surfaces do not occlude each other. Sensor-containing
    boxes are NaN and excluded as occluders, so an ego-vehicle label does not
    hide the entire scene. No point clouds, background occluders, sensor FOV
    or range limits are used.
    """
    boxes = np.asarray(boxes, dtype=float)
    origin = np.asarray(sensor_origin, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError("boxes must have shape (N, 7+)")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("sensor_origin must be a finite (3,) vector")
    if not np.isfinite(boxes[:, :7]).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("box geometry must be finite with positive dimensions")
    if not isinstance(samples_per_edge, (int, np.integer)) or samples_per_edge < 1:
        raise ValueError("samples_per_edge must be a positive integer")
    if ordering not in ("ray", "centre"):
        raise ValueError("ordering must be 'ray' or 'centre'")

    contains_sensor = np.array([
        points_inside_box(origin[None, :], box)[0] for box in boxes
    ], dtype=bool)
    scores = np.full(len(boxes), np.nan)
    # ponytail: O(N^2 * samples_per_edge^2) quadrature for one frame;
    # increase sampling for precision, add angular culling for large scenes.
    for i in np.flatnonzero(~contains_sensor):
        scores[i] = target_breakdown(boxes, origin, i, contains_sensor,
                                     samples_per_edge, ordering)["visibility"]
    return scores


def main():
    def show(value):
        return "NaN (sensor inside box)" if np.isnan(value) else f"{value:.3f}"

    print(f"{METRIC_NAME}\nInputs: labelled boxes + sensor poses only (no point cloud).\n")
    dataset = DAIRV2XExplorer(DATASET_ROOT)

    # Step 1: frame
    pair = dataset.pairs[PAIR_INDEX]
    frame_index = dataset.return_labeled_sync_time_ids_of_seq(PAIR_INDEX)[0]
    print(f"[1 frame] pair {PAIR_INDEX}: vehicle {pair['vehicle_pointcloud_path']}, "
          f"RSU {pair['infrastructure_pointcloud_path']}\n"
          f"          labels {pair['cooperative_label_path']}")

    # Step 2: boxes in world frame
    boxes = dataset.return_gt_boxes_in_map(PAIR_INDEX, frame_index)
    print(f"[2 boxes] {len(boxes)} cooperative boxes, world frame, rows "
          f"[x y z l w h yaw]; e.g. box 0 = {np.round(boxes[0, :7], 2).tolist()}")

    # Step 3: SE(3) sensor poses -> origins
    poses = {agent: dataset.return_map_se3_agent(PAIR_INDEX, agent, frame_index)
             for agent in AGENTS}
    print("[3 poses] vehicle: T_world<-lidar = T_world<-novatel @ T_novatel<-lidar\n"
          "          RSU:     T_world<-lidar = T_world<-virtuallidar")
    for agent, pose in poses.items():
        print(f"          {agent:14s} origin (world) = {np.round(pose[:3, 3], 2)}")

    # Step 7 for every box (steps 5-6 run inside), in the world frame
    results = {agent: solid_angle_visibility(boxes, poses[agent][:3, 3], SAMPLES_PER_EDGE)
               for agent in AGENTS}

    # Step 4: same answer from inside each sensor's own frame (origin = 0)
    for agent, pose in poses.items():
        local = solid_angle_visibility(boxes_in_sensor_frame(boxes, pose), np.zeros(3), 16)
        coarse = solid_angle_visibility(boxes, pose[:3, 3], 16)
        finite = np.isfinite(local) & np.isfinite(coarse)
        print(f"[4 frames] {agent:14s} max |V_world - V_sensor| = "
              f"{np.abs(local - coarse)[finite].max():.4f} (yaw-only boxes drop roll/pitch)")

    # Steps 5-6 in detail for the most occluded vehicle-view target
    origin = poses["vehicle"][:3, 3]
    unavailable = np.isnan(results["vehicle"])
    target = int(np.nanargmin(results["vehicle"]))
    detail = target_breakdown(boxes, origin, target, unavailable, SAMPLES_PER_EDGE, "ray")
    print(f"[5 samples] vehicle view, box {target} at "
          f"{np.linalg.norm(boxes[target, :3] - origin):.1f} m: {detail['samples']} face samples, "
          f"Omega = {detail['omega']:.2e} sr")
    print(f"[6 occlusion] {detail['blocked_samples']} samples blocked; blocked solid angle by box: "
          + ", ".join(f"box {j}: {w / detail['omega']:.0%}" for j, w in detail["by_occluder"].items()))
    print(f"[7 visibility] V = {detail['visible']:.2e} / {detail['omega']:.2e} = "
          f"{detail['visibility']:.3f}\n")

    print("Per box, 0 = fully occluded, 1 = fully visible")
    for i, box in enumerate(boxes):
        print(f"  box {int(box[8]):3d} | vehicle {show(results['vehicle'][i]):>24s} | "
              f"RSU {show(results['infrastructure'][i]):>24s}")
    return results


if __name__ == "__main__":
    main()
