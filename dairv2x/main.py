import numpy as np
from dairv2x.explorer import DAIRV2XExplorer
from mixedsig.tools.visualize_agg_frame import points_inside_box

DATASET_ROOT = "/mnt/c/Datasets/dair-v2x-c-example"
PAIR_INDEX = 0
AGENTS = ("vehicle",)

dataset = DAIRV2XExplorer(DATASET_ROOT)
# Load single frame
pair = dataset.pairs[PAIR_INDEX]
frame_index = dataset.return_labeled_sync_time_ids_of_seq(PAIR_INDEX)[0]

# Box in world frame
boxes = dataset.return_gt_boxes_in_map(PAIR_INDEX, frame_index)
for i in range(len(boxes)): 
    print(f"box {i} {np.round(boxes[i, :7], 2).tolist()}")

# SE(3) sensor pose -> origin world
poses = {agent: dataset.return_map_se3_agent(PAIR_INDEX, agent, frame_index)
            for agent in AGENTS}

vehicle_pose = poses["vehicle"]
print(f"vehicle pose, origin worl/map frame {np.round(vehicle_pose[:3, 3], 2)}")

# Point clouds are sensor-local; transform them into the world frame so they
# can be tested against the world-frame boxes.
points, _ = dataset.return_agent_point_cloud(PAIR_INDEX, "vehicle", frame_index)
xyz1 = np.c_[points[:, :3], np.ones(len(points))]
points_world = (poses["vehicle"] @ xyz1.T).T[:, :3]
points_world_with_intensity = np.column_stack((points_world, points[:, 3]))

# Each entry contains one box and all vehicle points inside it. Point
# coordinates and intensity are stored in the world/map frame.
box_points = {}
for box_index, box in enumerate(boxes):
    inside = points_inside_box(points_world, box)
    box_points[box_index] = {
        "box": box[:7].tolist(),
        "points": [
            {"x": float(x), "y": float(y), "z": float(z),
             "I": float(intensity)}
            for x, y, z, intensity in points_world_with_intensity[inside]
        ],
    }






