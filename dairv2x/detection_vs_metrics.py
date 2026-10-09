"""Do Def-3, RSV, distance or point count predict ego PointPillars misses?

Run after dairv2x.pointpillar_inference:
    PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.detection_vs_metrics

For every GT box in the inference output (vehicle LiDAR frame, sensor at the
origin) compute, from the ego viewpoint only:
  Def-3   solid-angle visibility from labelled boxes (dairv2x.visibility_frame)
  RSV     return-weighted solid-angle visibility (dairv2x.rsv)
  range   distance to the box centre
  points  ego LiDAR returns inside the box
Outcome: missed = no prediction matched at the IoU threshold (one-to-one,
confidence-ordered). Every GT box is kept, including ones with zero returns.

Prints, per IoU threshold: AUC for predicting a miss overall and within
0-30 / 30-60 / 60+ m; recall by visibility band; and held-out logistic models
testing whether each metric adds to range + log point count; confidence of
detected boxes vs each metric; and a one-table summary. Confidence exists only
for detected boxes, so those correlations are conditional on detection.
Per-box values are cached as CSV next to the detections.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pypcd4 import PointCloud
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from dairv2x.rsv import GRID_FRAMES, estimate_grid, rsv
from dairv2x.visibility_auc import RANGE_BINS
from dairv2x.visibility_frame import solid_angle_visibility
from mixedsig.tools.visualize_agg_frame import points_inside_box

DETECTIONS = Path("data/pointpillar_ego/val_cooperative_gt.jsonl")
DATA_ROOT = Path("data/coalign_full")
SAMPLES_PER_EDGE = 32
IOU_THRESHOLDS = ("0.5", "0.7")
V_BANDS = ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.999), (0.999, 1.01))
BLOCK = 20  # frames per cross-validation group (neighbouring frames are correlated)


def corners_to_boxes(corners):
    """(N, 8, 3) corners -> (N, 7) [x, y, z, l, w, h, yaw]; long edge = length."""
    boxes = np.empty((len(corners), 7))
    for i, c in enumerate(np.asarray(corners, dtype=float)):
        bottom = c[np.argsort(c[:, 2])[:4]]
        start = bottom[0]
        edges = sorted((np.linalg.norm(p[:2] - start[:2]), p) for p in bottom[1:])[:2]
        (width, _), (length, far) = edges
        boxes[i] = (*c.mean(0), length, width, np.ptp(c[:, 2]),
                    np.arctan2(far[1] - start[1], far[0] - start[0]))
    return boxes


def load_cloud(root, frame_id):
    cloud = PointCloud.from_path(root / "vehicle-side/velodyne" / f"{frame_id}.pcd").numpy(("x", "y", "z"))
    return cloud[np.isfinite(cloud).all(1)]


def score_frame(task):
    record, root, grid = task
    corners = np.asarray(record["gt_corners"], dtype=float).reshape(-1, 8, 3)
    if not len(corners):
        return []
    boxes = corners_to_boxes(corners)
    cloud = load_cloud(root, record["vehicle_frame_id"])
    def3 = solid_angle_visibility(boxes, np.zeros(3), SAMPLES_PER_EDGE)
    parts = rsv(boxes, grid.range_image(cloud), grid)
    rows = []
    for i, box in enumerate(boxes):
        row = {"frame": record["vehicle_frame_id"], "gt_index": i,
               "range_m": float(np.linalg.norm(box[:3])),
               "points": int(points_inside_box(cloud, box).sum()), "def3": def3[i],
               **{k: parts[k][i] for k in ("v_rsv", "v_lo", "o_lab", "o_unlab", "p_frac", "m_frac", "cells")}}
        for t in IOU_THRESHOLDS:
            match = record["matches"][t][i]
            row[f"missed@{t}"] = match is None
            row[f"confidence@{t}"] = match["confidence"] if match else np.nan
        rows.append(row)
    return rows


def build_table(detections, root, workers):
    records = [json.loads(line) for line in detections.read_text().splitlines()]
    clouds = [load_cloud(root, r["vehicle_frame_id"]) for r in records[:GRID_FRAMES]]
    grid = estimate_grid(clouds, "Vehicle")
    print(f"{len(records)} frames; ray grid: {grid.method}")
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for n, frame_rows in enumerate(pool.map(score_frame, [(r, root, grid) for r in records],
                                                chunksize=8), 1):
            rows += frame_rows
            if n % 200 == 0:
                print(f"  scored {n}/{len(records)} frames", flush=True)
    return pd.DataFrame(rows)


def auc(y, score):
    return roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan


METRICS = (("Def-3 V", "def3"), ("RSV V", "v_rsv"), ("Distance", "range_m"), ("log points", "log_points"))


def heldout(table, y, features, regression=False):
    """Cross-validated fit on range spline + log points (+ features).

    Logistic (miss) -> (AUC, log-loss); linear (confidence) -> (R^2, None).
    """
    columns = ColumnTransformer([("r", SplineTransformer(n_knots=5), ["range_m"]),
                                 ("x", StandardScaler(), ["log_points"] + features)])
    groups = table["frame"].astype("category").cat.codes // BLOCK
    if regression:
        model = make_pipeline(columns, LinearRegression())
        p = cross_val_predict(model, table, y, cv=GroupKFold(5), groups=groups)
        return r2_score(y, p), None
    model = make_pipeline(columns, LogisticRegression(max_iter=3000))
    p = cross_val_predict(model, table, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), log_loss(y, p)


def confidence_report(d, t, summary):
    """Spearman(confidence, metric) for detected boxes, plus held-out R^2 gains."""
    hit = d[~d[f"missed@{t}"].astype(bool)]
    confidence = hit[f"confidence@{t}"]
    print(f"\nDetection confidence vs metrics (detected boxes only, n={len(hit)}; "
          f"Spearman rho, + = higher value -> more confident)")
    print(f"{'Range':<9}{'n':>7}{'mean conf':>10}" + "".join(f"{name:>12}" for name, _ in METRICS))
    print("-" * 74)
    for name, low, high in RANGE_BINS:
        k = (hit["range_m"] >= low) & (hit["range_m"] < high)
        rhos = [spearmanr(confidence[k], hit[col][k])[0] for _, col in METRICS]
        print(f"{name:<9}{k.sum():>7}{confidence[k].mean():>10.3f}" + "".join(f"{r:>+12.3f}" for r in rhos))
        if name == "All":
            summary[f"rho conf@{t}"] = dict(zip((n for n, _ in METRICS), rhos))
    print("Held-out linear model of confidence (R^2)")
    base = heldout(hit, confidence, [], regression=True)[0]
    print(f"{'range + log points':<22}R^2 {base:.3f}")
    gains = {}
    for name, col in (("Def-3 V", "def3"), ("RSV V", "v_rsv")):
        r2 = heldout(hit, confidence, [col], regression=True)[0]
        gains[name] = r2 - base
        print(f"{'  + ' + name:<22}R^2 {r2:.3f}  ({r2 - base:+.3f})")
    summary[f"dR2 conf@{t}"] = gains


def report(table):
    table = table.copy()
    table["log_points"] = np.log1p(table["points"])
    unknown_def3 = table["def3"].isna()
    unknown_rsv = table["v_rsv"].isna() & ~unknown_def3
    print(f"\nGT boxes: {len(table)} in {table['frame'].nunique()} frames; "
          f"zero ego returns: {int((table['points'] == 0).sum())}; "
          f"Def-3 undefined (ego box): {int(unknown_def3.sum())}; "
          f"RSV unknown (no return in footprint): {int(unknown_rsv.sum())}")
    summary = {}
    for t in IOU_THRESHOLDS:
        missed = table[f"missed@{t}"].astype(bool)
        print(f"\n===== IoU {t}: recall {1 - missed.mean():.3f} "
              f"({int((~missed).sum())}/{len(table)} GT detected)")
        if unknown_rsv.any():
            print(f"miss rate where RSV is unknown: {missed[unknown_rsv].mean():.3f} "
                  f"(n={int(unknown_rsv.sum())}); these boxes are excluded from the tables below")
        d = table[~unknown_def3 & ~unknown_rsv]
        y = d[f"missed@{t}"].astype(int)
        print("\nAUC for predicting a MISS (0.5 = chance)")
        print(f"{'Range':<9}{'n':>7}{'miss %':>8}{'Def-3 (1-V)':>13}{'RSV (1-V)':>11}"
              f"{'Distance':>10}{'Points':>8}")
        print("-" * 66)
        for name, low, high in RANGE_BINS:
            k = ((d["range_m"] >= low) & (d["range_m"] < high)).values
            yk = y.values[k]
            aucs = [auc(yk, 1 - d['def3'].values[k]), auc(yk, 1 - d['v_rsv'].values[k]),
                    auc(yk, d['range_m'].values[k]), auc(yk, -d['log_points'].values[k])]
            print(f"{name:<9}{k.sum():>7}{100 * yk.mean():>8.1f}{aucs[0]:>13.3f}{aucs[1]:>11.3f}"
                  f"{aucs[2]:>10.3f}{aucs[3]:>8.3f}")
            if name == "All":
                summary[f"AUC miss@{t}"] = dict(zip((n for n, _ in METRICS), aucs))
        print("\nRecall by visibility band")
        print(f"{'V band':<12}{'Def-3 n':>9}{'recall':>8}{'RSV n':>9}{'recall':>8}")
        for low, high in V_BANDS:
            label = "1.0" if low >= 0.999 else f"{low:.2f}-{min(high, 1):.2f}"
            cells = []
            for col in ("def3", "v_rsv"):
                k = (d[col] >= low) & (d[col] < high)
                cells.append(f"{int(k.sum()):>9}{1 - y[k].mean() if k.any() else np.nan:>8.3f}")
            print(f"{label:<12}{''.join(cells)}")
        print("\nHeld-out logistic model of a miss (5-fold, grouped by blocks of frames)")
        base = heldout(d, y, [])[0]
        gains = {}
        for name, features in (("range + log points", []), ("  + Def-3", ["def3"]),
                               ("  + RSV", ["v_rsv"]), ("  + Def-3 + RSV", ["def3", "v_rsv"])):
            score, loss = heldout(d, y, features)
            print(f"{name:<22}AUC {score:.3f}  log-loss {loss:.4f}")
            gains[name.strip("+ ")] = score - base
        summary[f"dAUC miss@{t}"] = {"Def-3 V": gains["Def-3"], "RSV V": gains["RSV"]}
        confidence_report(d, t, summary)

    print("\n===== Summary (all ranges; misses: AUC with 1-V / distance / fewer points as the "
          "miss score; confidence: Spearman rho on detected boxes; gains: held-out, beyond range + log points)")
    print(f"{'':<24}" + "".join(f"{name:>12}" for name, _ in METRICS))
    for key, values in summary.items():
        print(f"{key:<24}" + "".join(
            f"{values[name]:>+12.3f}" if key.startswith(("d", "rho")) and name in values
            else f"{values[name]:>12.3f}" if name in values else f"{'base':>12}"
            for name, _ in METRICS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=DETECTIONS)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--recompute", action="store_true", help="ignore the cached per-box CSV")
    args = parser.parse_args()
    cache = args.detections.with_name(args.detections.stem + "_metrics.csv")
    if cache.is_file() and not args.recompute:
        table = pd.read_csv(cache, dtype={"frame": str})
        print(f"Loaded cached per-box metrics from {cache}")
    else:
        table = build_table(args.detections, args.root.resolve(), args.workers)
        table.to_csv(cache, index=False)
        print(f"Saved per-box metrics to {cache}")
    report(table)


if __name__ == "__main__":
    main()
