#!/usr/bin/env python
"""Plot binning-free occlusion-difficulty curves for the DCD baseline.

Two self-documenting figures replace the earlier binned bar charts (quantile
bins with equal spacing visually exaggerated noise between nearly identical
DCD values):

1. fig_dcd_detection_probability.png - P(pedestrian detected) as a smooth
   function of DCD for each detector (cubic-logistic fit, 95% bootstrap band).
2. fig_dcd_adjusted_effect.png - the same curves after adjusting for ground
   range and LiDAR point count (both held at their medians), drawn over the
   unadjusted curves. A preserved top-end drop means DCD carries occlusion
   information beyond sparsity; a flattened curve means DCD was proxying range.

Also writes the per-object outcome table used for fitting (one row per
object per detector with detected 0/1) to the analysis directory for reuse
in the thesis regression work.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from gilroy_lidar_replication.evaluate import load_predictions, prepare_frames  # noqa: E402

BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
DETECTOR_LABELS = {"pointpillar": "PointPillars", "second": "SECOND", "pointrcnn": "PointRCNN", "pv_rcnn": "PV-RCNN"}


def load_instances(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        row["dcd"] = float(row["dcd"])
        row["ground_range_m"] = float(row["ground_range_m"])
        row["num_points_in_gt"] = float(row["num_points_in_gt"])
        row["box"] = np.asarray([row[field] for field in BOX_FIELDS], dtype=np.float32)
    return [row for row in rows if np.isfinite(row["dcd"])]


def per_object_status(prepared, score_threshold, iou_threshold=0.5):
    """Greedy score-ranked matching: detected=1 when a prediction covers the GT box."""
    table = []
    for frame_id, frame in prepared.items():
        matched = np.zeros(len(frame["gt"]), dtype=int)
        used = set()
        for pred_index, prediction in enumerate(frame["predictions"]):
            if prediction["score"] < score_threshold:
                break  # predictions are score-sorted descending
            overlaps = frame["iou"][pred_index]
            best_iou, best_index = max(
                ((overlaps[i], i) for i in range(len(frame["gt"])) if i not in used),
                default=(0.0, None),
            )
            if best_index is not None and best_iou >= iou_threshold:
                used.add(best_index)
                matched[best_index] = 1
        for gt_row, hit in zip(frame["gt"], matched):
            table.append({**gt_row, "detected": int(hit)})
    return table


def fit_logistic(design, outcome, max_iter=100, ridge=1e-6):
    """Newton-IRLS logistic regression with a small ridge for separable bootstraps."""
    beta = np.zeros(design.shape[1])
    for _ in range(max_iter):
        prob = 1.0 / (1.0 + np.exp(-(design @ beta)))
        weights = prob * (1.0 - prob) + 1e-9
        gradient = design.T @ (outcome - prob)
        hessian = (design * weights[:, None]).T @ design + ridge * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


class CurveModel:
    """Cubic-logistic P(detected | DCD), optionally adjusted for range and point count."""

    def __init__(self, dcd, ranges, counts, adjusted):
        self.adjusted = adjusted
        self.d_mean, self.d_sd = float(np.mean(dcd)), float(np.std(dcd))
        self.r_mean, self.r_sd = float(np.mean(ranges)), float(np.std(ranges))
        log_counts = np.log1p(counts)
        self.c_mean, self.c_sd = float(np.mean(log_counts)), float(np.std(log_counts))
        self.r_median = float(np.median(ranges))
        self.c_median = float(np.median(counts))

    def design(self, dcd, ranges=None, counts=None):
        dcd = np.atleast_1d(np.asarray(dcd, dtype=float))
        dz = (dcd - self.d_mean) / self.d_sd
        columns = [np.ones_like(dz), dz, dz ** 2, dz ** 3]
        if self.adjusted:
            rz = (np.atleast_1d(np.asarray(ranges, dtype=float)) - self.r_mean) / self.r_sd
            cz = (np.log1p(np.atleast_1d(np.asarray(counts, dtype=float))) - self.c_mean) / self.c_sd
            columns += [rz, cz]
        return np.stack(columns, axis=1)

    def predict(self, beta, dcd):
        dcd = np.atleast_1d(np.asarray(dcd, dtype=float))
        if self.adjusted:
            ranges = np.full_like(dcd, self.r_median)
            counts = np.full_like(dcd, self.c_median)
            return 1.0 / (1.0 + np.exp(-(self.design(dcd, ranges, counts) @ beta)))
        return 1.0 / (1.0 + np.exp(-(self.design(dcd) @ beta)))


def detector_curves(table, grid, n_boot, seed):
    """Unadjusted and covariate-adjusted curves with 95% bootstrap bands."""
    dcd = table["dcd"]
    curves = {}
    for adjusted in (False, True):
        model = CurveModel(dcd, table["ground_range_m"], table["num_points_in_gt"], adjusted)
        beta = fit_logistic(model.design(dcd, table["ground_range_m"], table["num_points_in_gt"]), table["detected"])
        point = 100.0 * model.predict(beta, grid)
        sampler = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            index = sampler.integers(0, len(dcd), len(dcd))
            try:
                boot_beta = fit_logistic(model.design(dcd[index], table["ground_range_m"][index], table["num_points_in_gt"][index]), table["detected"][index])
            except np.linalg.LinAlgError:
                continue
            draws.append(model.predict(boot_beta, grid))
        band_low = 100.0 * np.percentile(draws, 2.5, axis=0)
        band_high = 100.0 * np.percentile(draws, 97.5, axis=0)
        p10 = float(model.predict(beta, np.quantile(dcd, 0.10)))
        p95 = float(model.predict(beta, np.quantile(dcd, 0.95)))
        curves["adj" if adjusted else "unadj"] = {
            "point": point, "low": band_low, "high": band_high,
            "drop_10_to_95": 100.0 * (p10 - p95),
        }
    return curves


def plot_curves(grid, results, adjusted_overlay, title, caption, output_dir, filename):
    fig, ax = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.17, 1, 0.83))
    for index, (detector, curves) in enumerate(sorted(results.items())):
        color = f"C{index}"
        if adjusted_overlay:
            ax.plot(grid, curves["unadj"]["point"], "--", color=color, alpha=0.5)
            ax.plot(grid, curves["adj"]["point"], "-", color=color, label=detector)
            ax.fill_between(grid, curves["adj"]["low"], curves["adj"]["high"], color=color, alpha=0.12)
        else:
            ax.plot(grid, curves["unadj"]["point"], "-", color=color, label=detector)
            ax.fill_between(grid, curves["unadj"]["low"], curves["unadj"]["high"], color=color, alpha=0.12)
    ax.set(
        xlabel="Per-object DCD - density-aware Chamfer Distance (empirically rises with return density; see caption)",
        ylabel="Pedestrians detected (%)\nmatched at 3D IoU >= 0.50, score >= 0.1",
        ylim=(0, 100),
        title=title,
    )
    ax.legend(title="Detector", frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    ax.text(0.01, 0.015, caption, transform=fig.transFigure, fontsize=7, va="bottom", ha="left", wrap=True)
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_report_comparison(grid, curves, output_dir, filename="fig_report_dcd_comparison.png"):
    """Report-ready single-panel detector comparison (300 dpi, minimal in-figure text)."""
    fig, ax = plt.subplots(figsize=(9.5, 5.9), constrained_layout=True)
    if fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(rect=(0, 0.02, 1, 0.93))
    for index, (detector, res) in enumerate(sorted(curves.items())):
        color = f"C{index}"
        label = DETECTOR_LABELS.get(detector, detector.replace("_", " ").title())
        ax.plot(grid, res["unadj"]["point"], "--", color=color, alpha=0.55)
        ax.plot(grid, res["adj"]["point"], "-", color=color, label=label)
        ax.fill_between(grid, res["adj"]["low"], res["adj"]["high"], color=color, alpha=0.10)
    ax.set(
        xlabel="Per-object DCD (density-aware Chamfer Distance)",
        ylabel="Pedestrians detected (%)",
        ylim=(0, 100),
        title="Pedestrian detectability vs DCD, KITTI val\n(solid: adjusted for range and point count; dashed: unadjusted; bands: 95% bootstrap CI)",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def method_caption(parameters, n_objects, adjusted, correlations=None):
    alpha = parameters.get("alpha", 1.0)
    n_lambda = parameters.get("n_lambda", 1)
    mode = "RMS-normalised sets" if parameters.get("normalize") else "raw metre coordinates"
    caption = (
        "Method - per ground-truth pedestrian, DCD (density-aware Chamfer Distance, Wu et al. NeurIPS 2021; "
        f"alpha={alpha}, n_lambda={n_lambda}, {mode}) compares the LiDAR returns inside the box with uniformly area-weighted "
        "samples of the box's LiDAR-facing faces (back faces excluded: self-occlusion). "
        "Detected = a confidence-ranked prediction covers the GT box at 3D IoU >= 0.50 with score >= 0.1 (other-object matches "
        f"ignored). Curves: logistic fits cubic in DCD with a 95% bootstrap confidence band over n={n_objects} pedestrians "
        "(KITTI validation split). "
    )
    if correlations:
        caption += (
            f"Empirical direction - under the official formulation with a variable observed-set size, DCD rises with point count "
            f"(r = {correlations['points']:+.2f}) and falls with ground range (r = {correlations['range']:+.2f}), so detection "
            "probability RISES with DCD here: the metric acts as a return-density proxy rather than an occlusion metric in this "
            "regime. This is the baseline result the thesis visibility metric is measured against. "
        )
    if adjusted:
        caption += (
            " Solid = adjusted for ground range and LiDAR point count (held at their medians); dashed = unadjusted. "
            "A solid curve that stays separated means DCD carries detection-relevant information beyond density; "
            "one that collapses onto flat means DCD only proxied range or point count."
        )
    else:
        caption += " Reading: the slope and direction show how detection probability varies across the DCD scale."
    return caption


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=Path("dcd_baseline/results/natural/dcd_instances.csv"))
    parser.add_argument("--prediction", action="append", required=True, help="DETECTOR=predictions.json (repeatable)")
    parser.add_argument("--target-class", default="Pedestrian")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("dcd_baseline/results/plots"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("dcd_baseline/results/analysis"))
    return parser.parse_args()


def main():
    args = parse_args()
    instances = load_instances(args.instances)
    protocol_path = args.instances.parent / "protocol.json"
    parameters = {}
    if protocol_path.exists():
        parameters = json.loads(protocol_path.read_text(encoding="utf-8")).get("parameters", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    grids = []
    results = {}
    for specifier in args.prediction:
        detector, path = specifier.split("=", 1)
        prepared = prepare_frames(instances, load_predictions(path, args.target_class))
        table = per_object_status(prepared, args.score_threshold)
        out_path = args.analysis_dir / f"per_object_outcomes_{detector}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=("frame_id", "label_index", "class_name", "dcd", "ground_range_m", "num_points_in_gt", "detected"))
            writer.writeheader()
            for row in table:
                writer.writerow({key: row[key] for key in writer.fieldnames})
        results[detector] = {
            "dcd": np.asarray([row["dcd"] for row in table]),
            "ground_range_m": np.asarray([row["ground_range_m"] for row in table]),
            "num_points_in_gt": np.asarray([row["num_points_in_gt"] for row in table]),
            "detected": np.asarray([row["detected"] for row in table]),
        }
        print(f"{detector}: {len(table)} objects, {100.0 * results[detector]['detected'].mean():.1f}% detected -> {out_path}")
        grids.append((results[detector]["dcd"].min(), results[detector]["dcd"].max()))

    grid = np.linspace(min(low for low, _ in grids), max(high for _, high in grids), 200)
    curves = {detector: detector_curves(table, grid, args.bootstrap, args.seed) for detector, table in results.items()}

    all_dcd = np.concatenate([t["dcd"] for t in results.values()])
    all_points = np.concatenate([t["num_points_in_gt"] for t in results.values()])
    all_ranges = np.concatenate([t["ground_range_m"] for t in results.values()])
    correlations = {
        "points": float(np.corrcoef(all_dcd, all_points)[0, 1]),
        "range": float(np.corrcoef(all_dcd, all_ranges)[0, 1]),
    }
    n_objects = len(next(iter(results.values()))["dcd"])

    paths = [
        plot_curves(grid, curves, adjusted_overlay=False,
                    title="Pedestrian detection probability vs DCD - DCD baseline (KITTI val)",
                    caption=method_caption(parameters, n_objects, adjusted=False, correlations=correlations),
                    output_dir=args.output_dir, filename="fig_dcd_detection_probability.png"),
        plot_curves(grid, curves, adjusted_overlay=True,
                    title="DCD effect beyond range and point count - DCD baseline (KITTI val)",
                    caption=method_caption(parameters, n_objects, adjusted=True, correlations=correlations),
                    output_dir=args.output_dir, filename="fig_dcd_adjusted_effect.png"),
        plot_report_comparison(grid, curves, args.output_dir),
    ]

    summary_path = args.analysis_dir / "dcd_effect_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("detector", "dcd_points_correlation", "dcd_range_correlation", "unadjusted_change_p10_to_p95_pp", "adjusted_change_p10_to_p95_pp"))
        writer.writeheader()
        for detector in sorted(curves):
            writer.writerow({
                "detector": detector,
                "dcd_points_correlation": round(correlations["points"], 3),
                "dcd_range_correlation": round(correlations["range"], 3),
                "unadjusted_change_p10_to_p95_pp": round(-curves[detector]["unadj"]["drop_10_to_95"], 2),
                "adjusted_change_p10_to_p95_pp": round(-curves[detector]["adj"]["drop_10_to_95"], 2),
            })
            print(f"{detector}: detection-rate change from DCD P10 to P95 = "
                  f"{-curves[detector]['unadj']['drop_10_to_95']:+.1f} pp unadjusted, "
                  f"{-curves[detector]['adj']['drop_10_to_95']:+.1f} pp adjusted "
                  f"(corr dcd-points {correlations['points']:+.2f}, dcd-range {correlations['range']:+.2f})")
    for path in paths + [summary_path]:
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
