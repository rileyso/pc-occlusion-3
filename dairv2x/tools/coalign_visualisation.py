"""Show CoAlign predictions and all GT boxes over the DAIR-V2X-C LiDAR pair.

White = all displayed GT, green = matched predictions, red = unmatched predictions.
The ego box is hidden by default when its footprint contains the LiDAR origin
or its centre is within 2 m of it. Use `--show-ego` to display it. Colors use
saved one-to-one IoU matches.
Cyan/orange points = vehicle/infrastructure LiDAR.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from dairv2x.explorer import DAIRV2XExplorer
from mixedsig.utils.geometry import apply_se3_
from mixedsig.utils.o3d_viz_utils import PointPainter


EDGES = [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]

AGENT_COLORS = (("vehicle", [0.0, 0.8, 1.0]), ("infrastructure", [1.0, 0.6, 0.0]))


def box_contains_origin(corners):
    """Return whether a box footprint contains the ego LiDAR origin."""
    polygon = np.asarray(corners, dtype=float)[:4, :2]
    signs = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        signs.append(edge[0] * (-start[1]) - edge[1] * (-start[0]))
    return all(sign >= -1e-6 for sign in signs) or all(sign <= 1e-6 for sign in signs)


def is_ego_box(corners):
    corners = np.asarray(corners, dtype=float)
    return box_contains_origin(corners) or np.linalg.norm(corners[:, :2].mean(axis=0)) <= 2.0


def build_scene(results, root, dataset=None, hide_ego=True, prediction_key="coalign"):
    """Return one frame as ((points, color), ...) and ((corners, color), ...).

    Points are in vehicle-LiDAR coordinates, ready to hand to any renderer;
    this keeps the per-frame computation separate from Open3D object creation.
    """
    dataset = dataset if dataset is not None else DAIRV2XExplorer(root)
    pair = dataset.pairs[results["pair_index"]]
    if (Path(pair["vehicle_pointcloud_path"]).stem != results["vehicle_frame_id"]
            or pair["infrastructure_pointcloud_path"] != results["infrastructure_pointcloud_path"]):
        raise ValueError("Result frame does not match this dataset's pair index")
    frame = dataset.return_labeled_sync_time_ids_of_seq(results["pair_index"])[0]
    clouds = []
    for agent, color in AGENT_COLORS:
        points, timestamp = dataset.return_agent_point_cloud(results["pair_index"], agent, frame)
        vehicle_from_agent = dataset.top_se3_map @ dataset.return_map_se3_agent(
            results["pair_index"], agent, timestamp)
        apply_se3_(vehicle_from_agent, points_=points)
        clouds.append((points, np.array(color)))
    if prediction_key not in results["predictions"]:
        raise ValueError(f"Result has no {prediction_key!r} predictions")
    predictions = [(index, prediction) for index, prediction in
                   enumerate(results["predictions"][prediction_key])
                   if not hide_ego or not is_ego_box(prediction["corners_vehicle_lidar"])]
    ground_truth = [box for box in results["ground_truth"]
                    if not hide_ego or not is_ego_box(box["corners_vehicle_lidar"])]
    matched = {box[prediction_key]["prediction_index"] for box in results["ground_truth"]
               if box.get(prediction_key) is not None}
    boxes = []
    for index, prediction in predictions:
        boxes.append((prediction["corners_vehicle_lidar"],
                      [0.0, 1.0, 0.0] if index in matched else [1.0, 0.0, 0.0]))
    for box in ground_truth:
        boxes.append((box["corners_vehicle_lidar"], [1.0, 1.0, 1.0]))
    return clouds, boxes


def add_corners(painter, corners, color):
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3):
        raise ValueError(f"Expected (8, 3) box corners, got {corners.shape}")
    box = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(EDGES))
    box.paint_uniform_color(color)
    painter.objs_to_draw.append(box)


def visualise(results, root, show=True, painter=None, zoom=0.5, hide_ego=True,
             prediction_key="coalign"):
    painter = painter if painter is not None else PointPainter()
    clouds, boxes = build_scene(results, root, hide_ego=hide_ego, prediction_key=prediction_key)
    for points, color in clouds:
        painter.add_pointclouds_(points, color)
    for corners, color in boxes:
        add_corners(painter, corners, color)
    all_predictions = results["predictions"][prediction_key]
    all_ground_truth = results["ground_truth"]
    hidden_predictions = sum(is_ego_box(p["corners_vehicle_lidar"])
                             for p in all_predictions) if hide_ego else 0
    hidden_ground_truth = sum(is_ego_box(g["corners_vehicle_lidar"])
                              for g in all_ground_truth) if hide_ego else 0
    print(f"Frame {results['vehicle_frame_id']}: {len(all_predictions) - hidden_predictions} displayed predictions, "
          f"{len(all_ground_truth) - hidden_ground_truth} displayed GT boxes, "
          f"{hidden_predictions + hidden_ground_truth} ego boxes hidden, "
          f"(3D IoU threshold {results['iou_threshold']})")
    print("Cyan/orange: vehicle/roadside points; white: all GT; "
          "green: matched predictions; red: unmatched predictions (one-to-one IoU matching)")
    if show:
        painter.show(camera={"lookat": [0, 0, 0], "front": [0, 0, 1],
                             "up": [1, 0, 0], "zoom": zoom})
    return painter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="JSON produced by dairv2x.coalign")
    parser.add_argument("--root", default="/mnt/c/Datasets/dair-v2x-c-example", type=Path,
                        help="DAIR root with cooperative/data_info.json and both point clouds")
    parser.add_argument("--no-show", action="store_true", help="load geometry without opening Open3D")
    parser.add_argument("--show-ego", action="store_true", help="also draw boxes containing the ego LiDAR origin")
    parser.add_argument("--zoom", type=float, default=0.5)
    parser.add_argument("--prediction", choices=("coalign", "vehicle_only"), default="coalign")
    args = parser.parse_args()
    if not 0 < args.zoom <= 2:
        parser.error("--zoom must be between 0 and 2")
    visualise(json.loads(args.results.read_text()), args.root,
              show=not args.no_show, zoom=args.zoom, hide_ego=not args.show_ego,
              prediction_key=args.prediction)


if __name__ == "__main__":
    main()
