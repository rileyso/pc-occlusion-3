import pickle
from pathlib import Path

import numpy as np
from box import Box
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from explorer import DAIRV2XVehicleExplorer

# Example Dataset
ROOT = "/mnt/c/Datasets/dair-v2x-c-example/vehicle-side"

# Full
# ROOT = "/mnt/d/dairv2x_v_full/vehicle-side"

data = DAIRV2XVehicleExplorer(ROOT, skip_incomplete=True)
boxes = []
broken = []
for idx in data.return_available_sequences():
    try:
        frame_boxes = data.return_gt_boxes_in_lidar(idx)
        pose = data.return_map_se3_agent(idx)
        points, _ = data.return_agent_point_cloud(idx)
        labels = data.return_labels(idx)
        new_boxes = []
        for row in frame_boxes:
            box = Box(*row[:7], category=int(row[7]), object_index=int(row[8]),
                      frame=idx, sensor_pose=pose)
            box.compute_measurements(points, labels[box.object_index])
            box.compute_surface_occupancy(resolution=0.05)
            new_boxes.append(box)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        broken.append(idx)
        print(f"frame: {idx} skipped ({type(error).__name__}: {error})")
        continue
    boxes += new_boxes
    print(f"frame: {idx} boxes: {len(frame_boxes)}")
print(f"skipped {len(broken)} broken frames")

RECORD_FIELDS = ("frame", "object_index", "class_name", "difficulty", "num_points",
                 "pixel_height", "visibility_score", "total_cells",
                 "total_occupied_cells", "x", "y", "z", "length", "width", "height", "yaw")


def dump_boxes(boxes, path):
    """Pickle per-box scalars only; the per-box cell grids are far too large."""
    records = [{name: getattr(box, name) for name in RECORD_FIELDS} for box in boxes]
    with open(path, "wb") as file:
        pickle.dump(records, file)
    print(f"saved {len(records)} boxes to {path}")

dump_boxes(boxes, Path(__file__).with_name("process_boxes.pkl"))

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

df = pd.DataFrame({
    "box": range(len(cars)),
    "difficulty": [b.difficulty for b in cars],
    "visibility_score": [b.visibility_score for b in cars],
})
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=df, x="box", y="visibility_score", hue="difficulty",
            hue_order=list(COLOURS), palette=COLOURS, dodge=False, width=1.0,
            ax=ax)
ax.set_xticks([])
ax.set_xlabel("Car box (grouped by difficulty)")
ax.set_ylabel("Visibility score")
ax.set_title("PASCAL Criteria")
ax.legend(title=None)
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
BLUE, NAVY = "#9ECAE1", "#08306B"
sns.boxplot(data=df, x="difficulty", y="visibility_score", order=list(COLOURS),
            color=BLUE, linecolor=NAVY, linewidth=1.2,
            flierprops={"markeredgecolor": NAVY, "markersize": 4}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1.2)
ax.set_title("PASCAL Criteria")
plt.show()
