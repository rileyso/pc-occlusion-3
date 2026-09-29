#!/usr/bin/env python
"""Evaluate detector outcomes per Density-aware Chamfer Distance bin.

Bins per-object DCD values into equal-population groups (merging adjacent bins
until every bin holds at least the Gilroy minimum of 55 instances), then reuses
the Gilroy evaluation core unchanged: 3D mAP@[0.50:0.95] over confidence-ranked
predictions, plus TP%/FP-per-GT/FN% at 3D IoU >= 0.50 and score >= 0.1. This
makes DCD-bin curves directly comparable with the existing occlusion-bin curves.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from gilroy_lidar_replication.evaluate import evaluate_subset, load_predictions, prepare_frames  # noqa: E402

BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")


def load_dcd_instances(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        row["dcd"] = float(row["dcd"])
        row["box"] = np.asarray([row[field] for field in BOX_FIELDS], dtype=np.float32)
    return rows


def assign_dcd_bins(rows, target_bins=10, minimum_per_bin=55):
    """Equal-population DCD bins, merged until every bin has >= minimum_per_bin rows."""
    finite = [row for row in rows if math.isfinite(row["dcd"])]
    excluded = len(rows) - len(finite)
    values = np.asarray([row["dcd"] for row in finite])
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
    for row, bin_index in zip(finite, index):
        row["dcd_bin"] = int(bin_index)
    for row in rows:
        row.setdefault("dcd_bin", -1)
    labels = [f"{edges[i]:.3f}-{edges[i + 1]:.3f}" for i in range(len(edges) - 1)]
    counts = bin_counts(edges)
    return edges, labels, counts, excluded


def evaluate(instances, predictions, score_threshold=0.1):
    prepared = prepare_frames(instances, predictions)
    bins = sorted({row["dcd_bin"] for row in instances if row["dcd_bin"] >= 0})
    labels = {row["dcd_bin"]: row["dcd_label"] for row in instances if row["dcd_bin"] >= 0}
    results = []
    for bin_index in bins:
        metrics = evaluate_subset(
            prepared,
            lambda rows, value=bin_index: [i for i, row in enumerate(rows) if row["dcd_bin"] == value],
            score_threshold,
        )
        results.append({"dcd_bin": bin_index, "dcd_range": labels[bin_index], **metrics})
    overall = evaluate_subset(
        prepared,
        lambda rows: [i for i, row in enumerate(rows) if row["dcd_bin"] >= 0],
        score_threshold,
    )
    results.append({"dcd_bin": "all", "dcd_range": "all", **overall})
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=Path("dcd_baseline/results/natural/dcd_instances.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--target-class", default="Pedestrian")
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--target-bins", type=int, default=10)
    parser.add_argument("--minimum-per-bin", type=int, default=55)
    parser.add_argument("--output-dir", type=Path, default=Path("dcd_baseline/results/evaluation"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0 and 1")
    instances = load_dcd_instances(args.instances)
    edges, labels, counts, excluded = assign_dcd_bins(instances, args.target_bins, args.minimum_per_bin)
    for row in instances:
        row["dcd_label"] = labels[row["dcd_bin"]] if row["dcd_bin"] >= 0 else "excluded"

    protocol_path = args.instances.parent / "protocol.json"
    dcd_parameters = {}
    if protocol_path.exists():
        dcd_parameters = json.loads(protocol_path.read_text(encoding="utf-8")).get("parameters", {})

    results = evaluate(instances, load_predictions(args.predictions, args.target_class), args.score_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.detector}_dcd_bin_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    binning = {
        "detector": args.detector,
        "target_class": args.target_class,
        "score_threshold": args.score_threshold,
        "dcd_parameters": dcd_parameters,
        "bin_edges_dcd": edges,
        "bin_labels": labels,
        "bin_counts": [int(count) for count in counts],
        "minimum_per_bin": args.minimum_per_bin,
        "excluded_non_finite_dcd": excluded,
        "binning_rule": f"equal-population target of {args.target_bins} bins, adjacent bins merged until each holds >= {args.minimum_per_bin}",
    }
    (args.output_dir / f"{args.detector}_binning.json").write_text(json.dumps(binning, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {metrics_path}")
    for row in results:
        if row["dcd_bin"] != "all":
            print(f"bin {row['dcd_bin']} ({row['dcd_range']}): n={row['instances']} mAP={row['map_3d_50_95']:.4f}")


if __name__ == "__main__":
    main()
