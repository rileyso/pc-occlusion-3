#!/usr/bin/env python
"""Classical geometric baselines: detection outcomes per geometric covariate bin.

For each continuous covariate (ground range, points in box, volumetric and
facing-surface return density) this script:

  1. bins the per-object values into equal-population groups (merging adjacent
     bins until every bin holds at least 55 instances, the workspace benchmark
     minimum);
  2. reports P(detected) per bin with Wilson 95% intervals, per detector;
  3. fits cubic-logistic detection-probability curves with a 95% bootstrap
     band over the covariate range;

and for the coarse benchmark labels (KITTI occlusion state 0/1/2, KITTI
difficulty) reports P(detected) per level. Also writes a close-range slice
(ground range <= 10 m) by occlusion state: if detection still drops with
occlusion among close, dense pedestrians, the coarse label carries information
that range and density cannot.

Matching semantics (identical for every detector): greedy confidence-ranked
matching at 3D IoU >= 0.50, score >= 0.1, predictions matching other annotated
pedestrians ignored.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]

DETECTORS = ("pointpillar", "second", "pointrcnn", "pv_rcnn")
DETECTOR_LABELS = {"pointpillar": "PointPillars", "second": "SECOND", "pointrcnn": "PointRCNN", "pv_rcnn": "PV-RCNN"}

# transform: applied before fitting (skewed counts/densities); identity for range.
CONTINUOUS_COVARIATES = {
    "ground_range_m": {"transform": "identity"},
    "num_points_in_gt": {"transform": "log1p"},
    "point_density_m3": {"transform": "log1p"},
    "point_density_m2_facing": {"transform": "log1p"},
}
CATEGORICAL_COVARIATES = {
    "kitti_occlusion": ["0", "1", "2"],
    "kitti_difficulty": ["Easy", "Moderate", "Hard"],
}


def wilson_interval(successes, total, z=1.959963984540054):
    """Wilson score interval on the proportion; returns (low, high) as percentages."""
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def assign_equal_population_bins(values, target_bins=10, minimum_per_bin=55):
    """Equal-population bins, merged until every bin has >= minimum_per_bin rows.

    Returns (edges, index) with len(edges) == number of bins + 1.
    """
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, target_bins + 1))).tolist()

    def bin_counts(current_edges):
        index = np.clip(np.searchsorted(current_edges, values, side="right") - 1, 0, len(current_edges) - 2)
        return np.bincount(index, minlength=len(current_edges) - 1)

    while len(edges) > 2:
        counts = bin_counts(edges)
        if counts.min() >= minimum_per_bin:
            break
        position = int(np.argmax(counts < minimum_per_bin))
        if position == 0:
            del edges[1]
        elif position == len(counts) - 1:
            del edges[-2]
        elif counts[position - 1] <= counts[position + 1]:
            del edges[position]
        else:
            del edges[position + 1]
    index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return edges, index


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


class CovariateCurve:
    """Cubic-logistic P(detected | covariate) on a standardised transform scale."""

    def __init__(self, values, transform):
        transformed = transform(values)
        self.transform = transform
        self.mean, self.sd = float(np.mean(transformed)), float(np.std(transformed))
        self.lo_t, self.hi_t = float(np.min(transformed)), float(np.max(transformed))

    def design(self, transformed):
        transformed = np.atleast_1d(np.asarray(transformed, dtype=float))
        z = (transformed - self.mean) / self.sd
        return np.stack([np.ones_like(z), z, z ** 2, z ** 3], axis=1)

    def predict(self, beta, transformed):
        return 1.0 / (1.0 + np.exp(-(self.design(transformed) @ beta)))


TRANSFORMS = {"identity": (lambda v: np.asarray(v, dtype=float), lambda t: t),
              "log1p": (np.log1p, np.expm1)}


def detector_table(master_rows, detector):
    """Per-detector arrays; rows without a detected flag (NaN-DCD instance) drop out."""
    keep, detected = [], []
    for row in master_rows:
        flag = row[f"detected_{detector}"]
        if flag != "":
            keep.append(row)
            detected.append(int(flag))
    return keep, np.asarray(detected, dtype=float)


def curve_for_covariate(rows, detected, covariate, transform_name, n_boot, seed):
    transform, inverse = TRANSFORMS[transform_name]
    values = np.asarray([float(row[covariate]) for row in rows], dtype=float)
    model = CovariateCurve(values, transform)
    transformed = transform(values)
    beta = fit_logistic(model.design(transformed), detected)
    grid_t = np.linspace(model.lo_t, model.hi_t, 200)
    point = 100.0 * model.predict(beta, grid_t)
    sampler = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        index = sampler.integers(0, len(values), len(values))
        try:
            boot_beta = fit_logistic(model.design(transformed[index]), detected[index])
        except np.linalg.LinAlgError:
            continue
        draws.append(model.predict(boot_beta, grid_t))
    band_low = 100.0 * np.percentile(draws, 2.5, axis=0)
    band_high = 100.0 * np.percentile(draws, 97.5, axis=0)
    grid_x = inverse(grid_t)
    p10 = 100.0 * float(model.predict(beta, transform(np.quantile(values, 0.10))))
    p95 = 100.0 * float(model.predict(beta, transform(np.quantile(values, 0.95))))
    return {
        "grid_x": grid_x, "grid_t": grid_t, "rate": point,
        "low": band_low, "high": band_high,
        "rate_p10": p10, "rate_p95": p95,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path,
                        default=Path("research_workspace/geometric_baselines/results/master_per_object.csv"))
    parser.add_argument("--target-bins", type=int, default=10)
    parser.add_argument("--minimum-per-bin", type=int, default=55)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-range", type=float, default=10.0,
                        help="Ground-range threshold (m) for the close-range occlusion slice.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research_workspace/geometric_baselines/results/analysis"))
    args = parser.parse_args()

    with args.master.open("r", encoding="utf-8", newline="") as input_file:
        master_rows = list(csv.DictReader(input_file))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    binned_rows, curve_rows, summary_rows = [], [], []

    # ---- continuous covariates: shared equal-population bins + cubic-logistic curves
    for covariate, spec in CONTINUOUS_COVARIATES.items():
        # identical 2,279-row subset for every detector, so bins are shared
        shared = [row for row in master_rows if all(row[f"detected_{d}"] != "" for d in DETECTORS)]
        values = np.asarray([float(row[covariate]) for row in shared], dtype=float)
        edges, _index = assign_equal_population_bins(values, args.target_bins, args.minimum_per_bin)
        for detector in DETECTORS:
            rows, detected = detector_table(master_rows, detector)
            per_object_values = np.asarray([float(row[covariate]) for row in rows], dtype=float)
            transform_name = spec["transform"]
            transform, _inverse = TRANSFORMS[transform_name]

            curve = curve_for_covariate(rows, detected, covariate, transform_name, args.bootstrap, args.seed)
            for x, rate, low, high in zip(curve["grid_x"], curve["rate"], curve["low"], curve["high"]):
                curve_rows.append({"covariate": covariate, "detector": detector,
                                   "x": f"{x:.6g}", "rate_pct": f"{rate:.3f}",
                                   "ci_low_pct": f"{low:.3f}", "ci_high_pct": f"{high:.3f}"})

            _, bin_index = assign_equal_population_bins(per_object_values, args.target_bins, args.minimum_per_bin)
            transformed = transform(per_object_values)
            for bin_number in range(len(edges) - 1):
                members = bin_index == bin_number
                n = int(members.sum())
                hits = int(detected[members].sum())
                low_ci, high_ci = wilson_interval(hits, n)
                binned_rows.append({
                    "covariate": covariate, "detector": detector, "bin_index": bin_number,
                    "bin_lo": f"{edges[bin_number]:.6g}", "bin_hi": f"{edges[bin_number + 1]:.6g}",
                    "bin_mean": f"{per_object_values[members].mean():.6g}",
                    "bin_mean_transformed": f"{transformed[members].mean():.6g}",
                    "transform": transform_name, "n": n, "n_detected": hits,
                    "rate_pct": f"{100.0 * hits / n:.3f}" if n else "",
                    "wilson_lo_pct": f"{low_ci:.3f}", "wilson_hi_pct": f"{high_ci:.3f}",
                })

            change = curve["rate_p95"] - curve["rate_p10"]
            summary_rows.append({
                "detector": detector, "covariate": covariate,
                "rate_p10_pct": f"{curve['rate_p10']:.2f}", "rate_p95_pct": f"{curve['rate_p95']:.2f}",
                "change_p10_to_p95_pp": f"{change:.2f}",
                "overall_rate_pct": f"{100.0 * detected.mean():.2f}", "n": len(rows),
            })
            print(f"{covariate:>24s} {detector:>10s}: P10 {curve['rate_p10']:.1f}% -> P95 {curve['rate_p95']:.1f}% "
                  f"({change:+.1f} pp), overall {100.0 * detected.mean():.1f}% (n={len(rows)})")

    # ---- covariate-level correlations with range and point count (detector-independent)
    shared = [row for row in master_rows if all(row[f"detected_{d}"] != "" for d in DETECTORS)]
    shared_range = np.asarray([float(row["ground_range_m"]) for row in shared], dtype=float)
    shared_points = np.asarray([float(row["num_points_in_gt"]) for row in shared], dtype=float)
    for covariate in CONTINUOUS_COVARIATES:
        values = np.asarray([float(row[covariate]) for row in shared], dtype=float)
        summary_rows.append({
            "detector": "all", "covariate": covariate,
            "corr_with_ground_range": f"{np.corrcoef(values, shared_range)[0, 1]:.3f}",
            "corr_with_num_points": f"{np.corrcoef(values, shared_points)[0, 1]:.3f}",
            "n": len(shared),
        })

    # ---- coarse benchmark labels: P(detected) per level
    for covariate, levels in CATEGORICAL_COVARIATES.items():
        observed = sorted({row[covariate] for row in master_rows if row[f"detected_{DETECTORS[0]}"] != ""})
        levels = [level for level in levels if level in observed] or observed
        for detector in DETECTORS:
            rows, detected = detector_table(master_rows, detector)
            labels = np.asarray([row[covariate] for row in rows])
            for level_index, level in enumerate(levels):
                members = labels == level
                n = int(members.sum())
                hits = int(detected[members].sum())
                low_ci, high_ci = wilson_interval(hits, n)
                binned_rows.append({
                    "covariate": covariate, "detector": detector, "bin_index": level_index,
                    "bin_lo": level, "bin_hi": level, "bin_mean": "", "bin_mean_transformed": "",
                    "transform": "categorical", "n": n, "n_detected": hits,
                    "rate_pct": f"{100.0 * hits / n:.3f}" if n else "",
                    "wilson_lo_pct": f"{low_ci:.3f}", "wilson_hi_pct": f"{high_ci:.3f}",
                })
                print(f"{covariate:>24s} {detector:>10s} level {level}: {hits}/{n} = "
                      f"{100.0 * hits / n:.1f}%" if n else f"level {level}: n=0")

    # ---- close-range slice: does the coarse occlusion label matter when density is high?
    close_rows = []
    for detector in DETECTORS:
        rows, detected = detector_table(master_rows, detector)
        ranges = np.asarray([float(row["ground_range_m"]) for row in rows], dtype=float)
        labels = np.asarray([row["kitti_occlusion"] for row in rows])
        close = ranges <= args.close_range
        for level in CATEGORICAL_COVARIATES["kitti_occlusion"]:
            members = close & (labels == level)
            n = int(members.sum())
            hits = int(detected[members].sum())
            low_ci, high_ci = wilson_interval(hits, n)
            close_rows.append({
                "detector": detector, "max_ground_range_m": args.close_range, "kitti_occlusion": level,
                "n": n, "n_detected": hits,
                "rate_pct": f"{100.0 * hits / n:.3f}" if n else "",
                "wilson_lo_pct": f"{low_ci:.3f}", "wilson_hi_pct": f"{high_ci:.3f}",
            })
            if n:
                print(f"close-range (<= {args.close_range:.0f} m) occlusion {level} {detector:>10s}: "
                      f"{hits}/{n} = {100.0 * hits / n:.1f}%")

    outputs = {
        "binned_detection_rate.csv": binned_rows,
        "curve_fits.csv": curve_rows,
        "covariate_summary.csv": summary_rows,
        "close_range_by_occlusion.csv": close_rows,
    }
    for name, rows in outputs.items():
        path = args.output_dir / name
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"WROTE {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
