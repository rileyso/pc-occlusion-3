"""Reproduce the RSV decomposition tables in research_notes/rsv_dairv2x_frozen_2026-10-09.md.

Run after dairv2x.detection_vs_metrics has cached the vehicle-GT metrics:
    PYTHONPATH=. python -m dairv2x.rsv_decomposition_analysis

Part A (human labels, DAIR example): AUC of each RSV component vs
occluded_state > 0, overall and by range.
Part B (ego PointPillars misses, full DAIR val, vehicle GT, IoU 0.5):
held-out gains of each component beyond range + log points and beyond
range + log points + V_lo; AUC within point-count quintiles.

Components: V_lo = Omega(H)/Omega_exp (observed visibility, evidence);
O_ext = Omega(B)/Omega(H+B+P) (occlusion); V_rsv = Omega(H)/Omega(H+B+P).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dairv2x.detection_vs_metrics import heldout
from dairv2x.rsv import rsv, sensor_grids
from dairv2x.visibility_auc import SIDES, iter_side_frames
from dairv2x.visibility_frame import DATASET_ROOT, solid_angle_visibility
from mixedsig.tools.visualize_agg_frame import points_inside_box

DETECTION_METRICS = Path("data/pointpillar_ego/val_vehicle_gt_metrics.csv")
LABEL_BINS = (("all", 0, np.inf), ("0-30", 0, 30), ("30-60", 30, 60), ("60+", 60, np.inf))


def label_table():
    root = Path(DATASET_ROOT)
    grid = sensor_grids(root)["Vehicle"]
    rows = []
    for labels, boxes, cloud in iter_side_frames(root, *SIDES["Vehicle"]):
        parts = rsv(boxes, grid.range_image(cloud), grid)
        def3 = solid_angle_visibility(boxes, np.zeros(3), 32)
        for i, label in enumerate(labels):
            rows.append({"occluded": int(label["occluded_state"]) > 0,
                         "range_m": float(np.linalg.norm(boxes[i, :3])), "def3": def3[i],
                         "v_lo": parts["v_lo"][i], "v_rsv": parts["v_rsv"][i],
                         "o_ext": parts["o_lab"][i] + parts["o_unlab"][i],
                         "points": int(points_inside_box(cloud, boxes[i]).sum())})
    table = pd.DataFrame(rows).dropna(subset=["def3", "v_rsv"])
    print(f"A. Human occluded_state (DAIR example, vehicle side, n={len(table)}): AUC")
    scores = (("1 - V_def3", 1 - table.def3), ("1 - V_lo", 1 - table.v_lo),
              ("1 - V_rsv", 1 - table.v_rsv), ("O_ext", table.o_ext),
              ("-log1p(points)", -np.log1p(table.points)))
    print(f"{'score':<18}" + "".join(f"{name:>8}" for name, *_ in LABEL_BINS))
    for name, score in scores:
        cells = []
        for _, low, high in LABEL_BINS:
            k = (table.range_m >= low) & (table.range_m < high)
            cells.append(roc_auc_score(table.occluded[k], score[k]))
        print(f"{name:<18}" + "".join(f"{c:>8.3f}" for c in cells))


def detection_table():
    d = pd.read_csv(DETECTION_METRICS, dtype={"frame": str})
    d = d[d.def3.notna() & d.v_rsv.notna()].copy()
    d["log_points"], d["o_ext"] = np.log1p(d.points), d.o_lab + d.o_unlab
    y = d["missed@0.5"].astype(int)
    print(f"\nB. Ego PointPillars misses (full val, vehicle GT, IoU 0.5, n={len(d)})")
    print("Single-score AUC: " + ", ".join(
        f"{n} {roc_auc_score(y, s):.3f}" for n, s in (("1-V_lo", 1 - d.v_lo), ("1-V_rsv", 1 - d.v_rsv),
                                                     ("O_ext", d.o_ext), ("-log points", -d.log_points))))
    for base_features, label in (([], "range + log points"), (["v_lo"], "range + log points + V_lo")):
        base = heldout(d, y, base_features)[0]
        print(f"Held-out AUC, base = {label}: {base:.3f}")
        for extra in (["def3"], ["v_lo"], ["v_rsv"], ["o_ext"], ["o_ext", "p_frac", "m_frac"]):
            if extra[0] in base_features:
                continue
            score = heldout(d, y, base_features + extra)[0]
            print(f"   + {' + '.join(extra):<24}{score:.3f} ({score - base:+.3f})")
    print("AUC within log-point quintiles")
    print(f"{'points':>12}{'n':>6}{'miss %':>8}{'Def-3':>8}{'V_lo':>8}{'V_rsv':>8}{'O_ext':>8}{'P_frac':>8}")
    for interval, g in d.groupby(pd.qcut(d.log_points, 5), observed=True):
        yy = g["missed@0.5"]
        cells = [roc_auc_score(yy, s) for s in (1 - g.def3, 1 - g.v_lo, 1 - g.v_rsv, g.o_ext, g.p_frac)]
        span = f"{np.expm1(interval.left):.0f}-{np.expm1(interval.right):.0f}"
        print(f"{span:>12}{len(g):>6}{100 * yy.mean():>8.1f}" + "".join(f"{c:>8.3f}" for c in cells))


if __name__ == "__main__":
    label_table()
    detection_table()
