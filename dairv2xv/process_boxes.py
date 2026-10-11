import sys
from pathlib import Path

import numpy as np
from box import Box
from checkpoint import dump_boxes, load_checkpoint, save_checkpoint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from explorer import DAIRV2XVehicleExplorer
from log_tee import tee_to_log

# Example Dataset: /mnt/c/Datasets/dair-v2x-c-example/vehicle-side
# Pass the vehicle-side folder as the first argument to override the default.
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FRESH = "--fresh" in sys.argv  # ignore any saved checkpoint and start over
ROOT = args[0] if args else "/mnt/d/dairv2x_v_full/vehicle-side"
CHECKPOINT_EVERY = 200  # frames between partial-result saves
COLOURS = {"Easy": "tab:green", "Moderate": "tab:orange", "Hard": "tab:red"}

tee_to_log(Path(__file__).with_name("process_boxes.log"))
data = DAIRV2XVehicleExplorer(ROOT, skip_incomplete=False)
total_frames = len(data.return_available_sequences())
CHECKPOINT = Path(__file__).with_name("process_boxes.ckpt.pkl")
boxes, broken, start = ([], [], 0) if FRESH else load_checkpoint(CHECKPOINT, ROOT, total_frames)
for idx in range(start, total_frames):
    if idx > start and idx % CHECKPOINT_EVERY == 0:
        save_checkpoint(CHECKPOINT, ROOT, total_frames, idx, broken, boxes)
    try:
        frame_boxes = data.return_gt_boxes_in_lidar(idx)
        pose = None  # pose is only stored on the Box; the official data has no NovAtel calib
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
        print(f"frame: {idx + 1}/{total_frames} skipped ({type(error).__name__}: {error})")
        continue
    boxes += new_boxes
    print(f"frame: {idx + 1}/{total_frames} boxes: {len(frame_boxes)}")
print(f"skipped {len(broken)} broken frames")
dump_boxes(boxes, Path(__file__).with_name("process_boxes.pkl"))
CHECKPOINT.unlink(missing_ok=True)  # finished: a rerun starts clean

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
fig.savefig(Path(__file__).with_name("process_boxes_bins_full_set.png"), dpi=300, bbox_inches="tight")
# plt.show()  # blocks until the window is closed; the PNGs above are saved without it

fig, ax = plt.subplots(figsize=(6, 5))
BLUE, NAVY = "#9ECAE1", "#08306B"
sns.boxplot(data=df, x="difficulty", y="visibility_score", order=list(COLOURS),
            color=BLUE, linecolor=NAVY, linewidth=1.2,
            flierprops={"markeredgecolor": NAVY, "markersize": 4}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1.0)
ax.set_title("PASCAL Criteria")
fig.savefig(Path(__file__).with_name("process_boxes_whisker_full_set.png"), dpi=300, bbox_inches="tight")
# plt.show()  # blocks until the window is closed; the PNGs above are saved without it
