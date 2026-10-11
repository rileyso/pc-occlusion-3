"""Box-and-whisker plot of Car visibility score by difficulty, filtered.

Usage: python dairv2xv/plot_filtered.py [records.pkl] [--min-points N] [--max-dist M]
Reads the pickle written by rsv.py / process_boxes.py and saves a PNG next to it.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("pkl", nargs="?", default=Path(__file__).with_name("rsv.pkl"), type=Path)
parser.add_argument("--min-points", type=int, default=20)
parser.add_argument("--max-dist", type=float, default=80.0)
args = parser.parse_args()

DIFFICULTIES = ["Easy", "Moderate", "Hard"]
BLUE, NAVY = "#9ECAE1", "#08306B"

df = pd.DataFrame(pickle.load(open(args.pkl, "rb")))
cars = df[(df.class_name == "Car") & df.difficulty.isin(DIFFICULTIES)
          & df.visibility_score.notna() & df.num_points.notna()].copy()
cars["dist"] = np.hypot(cars.x, cars.y)
kept = cars[(cars.num_points >= args.min_points) & (cars.dist <= args.max_dist)]

print(f"{args.pkl.name}: {len(cars)} labelled Cars -> {len(kept)} kept "
      f"(>= {args.min_points} points, <= {args.max_dist:g} m)")
for name in DIFFICULTIES:
    before, after = (cars.difficulty == name).sum(), (kept.difficulty == name).sum()
    scores = kept.visibility_score[kept.difficulty == name]
    print(f"{name:9s} {before:6d} -> {after:6d}  median {scores.median():.3f}  "
          f"Q1 {scores.quantile(.25):.3f}  Q3 {scores.quantile(.75):.3f}")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=kept, x="difficulty", y="visibility_score", order=DIFFICULTIES,
            color=BLUE, linecolor=NAVY, linewidth=1.2,
            flierprops={"markeredgecolor": NAVY, "markersize": 4}, ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Visibility score")
ax.set_ylim(0, 1.0)
ax.set_title(f"PASCAL Criteria (>= {args.min_points} points, <= {args.max_dist:g} m)")
out = args.pkl.with_name(f"{args.pkl.stem}_whisker_filtered.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved {out}")
