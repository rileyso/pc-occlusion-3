"""Show CoAlign predictions and all GT boxes over the DAIR-V2X-C LiDAR pair.

White = all GT, green = matched predictions, red = unmatched predictions.
Colors use the saved one-to-one IoU matches, not a confidence threshold.
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


def add_corners(painter, corners, color):
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (8, 3):
        raise ValueError(f"Expected (8, 3) box corners, got {corners.shape}")
    box = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(EDGES))
    box.paint_uniform_color(color)
    painter.objs_to_draw.append(box)


def visualise(results, root, show=True, painter=None, zoom=0.5):
    dataset = DAIRV2XExplorer(root)
    pair_index = results["pair_index"]
    pair = dataset.pairs[pair_index]
    if (Path(pair["vehicle_pointcloud_path"]).stem != results["vehicle_frame_id"]
            or pair["infrastructure_pointcloud_path"] != results["infrastructure_pointcloud_path"]):
        raise ValueError("Result frame does not match this dataset's pair index")
    painter = painter if painter is not None else PointPainter()
    frame = dataset.return_labeled_sync_time_ids_of_seq(pair_index)[0]
    for agent, color in (("vehicle", [0.0, 0.8, 1.0]),
                         ("infrastructure", [1.0, 0.6, 0.0])):
        points, timestamp = dataset.return_agent_point_cloud(pair_index, agent, frame)
        vehicle_from_agent = dataset.top_se3_map @ dataset.return_map_se3_agent(
            pair_index, agent, timestamp)
        apply_se3_(vehicle_from_agent, points_=points)
        painter.add_pointclouds_(points, np.array(color))

    predictions = results["predictions"]["coalign"]
    ground_truth = results["ground_truth"]
    matched = {box["coalign"]["prediction_index"] for box in ground_truth
               if box["coalign"] is not None}
    for index, prediction in enumerate(predictions):
        add_corners(painter, prediction["corners_vehicle_lidar"],
                    [0.0, 1.0, 0.0] if index in matched else [1.0, 0.0, 0.0])
    for box in ground_truth:
        add_corners(painter, box["corners_vehicle_lidar"], [1.0, 1.0, 1.0])
    print(f"Frame {results['vehicle_frame_id']}: {len(predictions)} predictions, "
          f"{len(ground_truth)} GT boxes, "
          f"{sum(box['coalign'] is None for box in ground_truth)} unmatched "
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
    parser.add_argument("--zoom", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.zoom <= 2:
        parser.error("--zoom must be between 0 and 2")
    visualise(json.loads(args.results.read_text()), args.root,
              show=not args.no_show, zoom=args.zoom)


if __name__ == "__main__":
    main()
