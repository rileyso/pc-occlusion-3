"""Compare Definition-3 and RSV against DAIR occluded_state, printed as AUC tables.

Run: PYTHONPATH=. python -m dairv2x.compare_visibility_auc

Same protocol as `visibility_auc.py`, with an RSV column added. All columns use
the same boxes: those where both Def-3 and RSV are defined (RSV is unknown
when no footprint ray has a return). AUC = chance a random occluded box
scores higher than a random visible one (0.5 = chance). Scores: occlusion
1 - V for each metric, distance, and -log1p(points).
"""
from pathlib import Path

import numpy as np

from dairv2x.tools.rsv import rsv, sensor_grids
from dairv2x.visibility_auc import (
    RANGE_BINS, SAMPLES_PER_EDGE, SIDES, auc, iter_side_frames,
)
from dairv2x.visibility_frame import DATASET_ROOT, solid_angle_visibility
from mixedsig.tools.visualize_agg_frame import points_inside_box


def score_side(root, label_dir, cloud_dir, grid):
    columns = {k: [] for k in ("occluded", "def3", "rsv", "distance", "points")}
    for labels, boxes, cloud in iter_side_frames(root, label_dir, cloud_dir):
        columns["def3"] += list(solid_angle_visibility(boxes, np.zeros(3), SAMPLES_PER_EDGE))
        columns["rsv"] += list(rsv(boxes, grid.range_image(cloud), grid)["v_rsv"])
        columns["distance"] += list(np.linalg.norm(boxes[:, :3], axis=1))
        columns["points"] += [int(points_inside_box(cloud, box).sum()) for box in boxes]
        columns["occluded"] += [int(l["occluded_state"]) > 0 for l in labels]
    return {k: np.array(v) for k, v in columns.items()}


def main():
    root = Path(DATASET_ROOT)
    grids = sensor_grids(root)
    for side, (label_dir, cloud_dir) in SIDES.items():
        c = score_side(root, label_dir, cloud_dir, grids[side])
        valid = np.isfinite(c["def3"]) & np.isfinite(c["rsv"])
        dropped = int(np.isfinite(c["def3"]).sum() - valid.sum())
        c = {k: v[valid] for k, v in c.items()}
        print(f"\n{side} side: {valid.sum()} boxes, {int(c['occluded'].sum())} labelled occluded "
              f"({dropped} with RSV unknown excluded)")
        print(f"ray grid: {grids[side].method}")
        print(f"{'Range':<9}{'n':>6}{'Def-3 (1-V)':>13}{'RSV (1-V)':>11}{'Distance':>10}{'Points':>8}")
        print("-" * 57)
        for name, low, high in RANGE_BINS:
            k = (c["distance"] >= low) & (c["distance"] < high)
            y = c["occluded"][k]
            print(f"{name:<9}{k.sum():>6}{auc(y, 1 - c['def3'][k]):>13.3f}"
                  f"{auc(y, 1 - c['rsv'][k]):>11.3f}{auc(y, c['distance'][k]):>10.3f}"
                  f"{auc(y, -np.log1p(c['points'][k])):>8.3f}")
    print("\nnan = bin has only one label class. RSU values depend on the unverified "
          "RSU origin height and an approximate ray grid.")


if __name__ == "__main__":
    main()
