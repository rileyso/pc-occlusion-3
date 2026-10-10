import numpy as np
from box import Box
import matplotlib.pyplot as plt
from explorer import DAIRV2XVehicleExplorer

ROOT = "/mnt/c/Datasets/dair-v2x-c-example/vehicle-side"

data = DAIRV2XVehicleExplorer(ROOT)
boxes = []
for idx in data.return_available_sequences():
    frame_boxes = data.return_gt_boxes_in_lidar(idx)
    pose = data.return_map_se3_agent(idx)
    points, _ = data.return_agent_point_cloud(idx)
    labels = data.return_labels(idx)
    for row in frame_boxes:
        box = Box(*row[:7], category=int(row[7]), object_index=int(row[8]),
                  frame=idx, sensor_pose=pose)
        box.compute_measurements(points, labels[box.object_index])
        box.compute_surface_occupancy(resolution=0.05)
        boxes.append(box)
    print(f"frame: {idx} boxes: {len(frame_boxes)}")

for box in boxes:
    if box.class_name == "Car" and box.difficulty != "Unclassified" and box.difficulty == "Easy":
        score = f"{box.visibility_score:.4f}" if box.visibility_score is not None else "None"
        print(f"frame: {box.frame}, box: {box.object_index}, points: {box.num_points}, "
            f"difficulty label: {box.difficulty}, visibility score: {score}, class: {box.class_name}")

for box in boxes:
    if box.class_name == "Car" and box.difficulty != "Unclassified" and box.difficulty == "Moderate":
        score = f"{box.visibility_score:.4f}" if box.visibility_score is not None else "None"
        print(f"frame: {box.frame}, box: {box.object_index}, points: {box.num_points}, "
            f"difficulty label: {box.difficulty}, visibility score: {score}, class: {box.class_name}")

for box in boxes:
    if box.class_name == "Car" and box.difficulty != "Unclassified" and box.difficulty == "Hard":
        score = f"{box.visibility_score:.4f}" if box.visibility_score is not None else "None"
        print(f"frame: {box.frame}, box: {box.object_index}, points: {box.num_points}, "
            f"difficulty label: {box.difficulty}, visibility score: {score}, class: {box.class_name}")



COLOURS = {"Easy": "tab:green", "Moderate": "tab:orange", "Hard": "tab:red"}

cars = [b for b in boxes if b.class_name == "Car" and b.difficulty in COLOURS
        and b.visibility_score is not None]
cars.sort(key=lambda b: (list(COLOURS).index(b.difficulty), -b.visibility_score))

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(cars)), [b.visibility_score for b in cars],
       color=[COLOURS[b.difficulty] for b in cars], width=1.0)
for name, colour in COLOURS.items():
    ax.bar(0, 0, color=colour, label=name)
ax.legend()
ax.set_xlabel("Car box (grouped by difficulty)")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1)
ax.set_title("PASCAL Criteria")
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
groups = [[b.visibility_score for b in cars if b.difficulty == name] for name in COLOURS]
bp = ax.boxplot(groups, patch_artist=True)
ax.set_xticklabels(list(COLOURS))
for patch, colour in zip(bp["boxes"], COLOURS.values()):
    patch.set_facecolor(colour)
for median in bp["medians"]:
    median.set_color("black")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1)
ax.set_title("Car visibility score by KITTI difficulty")
plt.show()
