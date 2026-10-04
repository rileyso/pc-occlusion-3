"""Visualize both DAIR-V2X-C LiDAR clouds and cooperative GT boxes."""

import argparse

import numpy as np

from dairv2x.explorer import DAIRV2XExplorer
from mixedsig.tools.visualize_agg_frame import (
    before_return_fraction, classify_box_returns, points_inside_box,
)
from mixedsig.utils.geometry import apply_se3, apply_se3_
from mixedsig.utils.o3d_viz_utils import PointPainter


COLORS = {"vehicle": np.array([0.0, 0.8, 1.0]),
          "infrastructure": np.array([1.0, 0.6, 0.0])}

def visualize(dataset, pair_index, show=True, painter=None, zoom=0.8):
    # All geometries must share the vehicle LiDAR frame.
    frame_index = dataset.return_labeled_sync_time_ids_of_seq(pair_index)[0]
    painter = painter if painter is not None else PointPainter()
    sensor_data = {}
    for agent_name in ("vehicle", "infrastructure"):
        points, timestamp = dataset.return_agent_point_cloud(pair_index, agent_name, frame_index)
        vehicle_se3_agent = dataset.top_se3_map @ dataset.return_map_se3_agent(
            pair_index, agent_name, timestamp)
        apply_se3_(vehicle_se3_agent, points_=points)
        sensor_data[agent_name] = (points, vehicle_se3_agent[:3, 3])
        painter.add_pointclouds_(points, COLORS[agent_name])
        print(f"{agent_name}: {len(points)} points")

    boxes_world = dataset.return_gt_boxes_in_map(pair_index, frame_index)
    boxes_vehicle = apply_se3(dataset.top_se3_map, boxes_=boxes_world)
    painter.add_boxes_(boxes_vehicle, np.array([0.9, 0.9, 0.9]))
    print(f"Ground truth: {len(boxes_vehicle)} boxes")
    for index, box in enumerate(boxes_vehicle):
        for agent_name, (points, origin) in sensor_data.items():
            if points_inside_box(origin[None, :], box)[0]:
                print(f"Box {index} | {agent_name}: sensor inside box; metrics unavailable")
                continue
            returns = classify_box_returns(origin, points, box)
            before = int(returns["occluded"].sum())
            inside = int(returns["hit"].sum())
            beyond = int(returns["beyond"].sum())
            total = before + inside + beyond
            fraction = before_return_fraction(returns)
            score = "unknown" if fraction is None else f"{fraction:.2f}"
            print(f"Box {index} | {agent_name}: before={before}/{total}, "
                  f"inside={inside}/{total}, beyond={beyond}/{total}, "
                  f"before/(before+inside)={score}")
    if show:
        painter.show(camera={"lookat": [0, 0, 0], "front": [0, 0, 1],
                             "up": [1, 0, 0], "zoom": zoom})
    return boxes_vehicle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/c/Datasets/dair-v2x-c-example")
    parser.add_argument("--pair-index", type=int, default=0,
                        help="index in vehicle-filename-sorted cooperative/data_info.json")
    parser.add_argument("--no-show", action="store_true", help="load and count geometry without opening Open3D")
    parser.add_argument("--zoom", type=float, default=0.5, help="top-down camera zoom (default: 0.5)")
    args = parser.parse_args()
    if not 0 < args.zoom <= 2:
        parser.error("--zoom must be between 0 and 2")

    dataset = DAIRV2XExplorer(args.root)
    pair = dataset.pairs[args.pair_index]
    print(f"Pair {args.pair_index}: {pair['vehicle_pointcloud_path']} + "
          f"{pair['infrastructure_pointcloud_path']}")
    return visualize(dataset, args.pair_index, show=not args.no_show, zoom=args.zoom)


if __name__ == "__main__":
    main()
