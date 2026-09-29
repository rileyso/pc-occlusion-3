#!/usr/bin/env python
"""Compute per-object DCD: observed LiDAR returns vs the ideal LiDAR-facing box surface.

For every row of an instances CSV (frame_id + oriented 3D box fields, the same
schema the Gilroy natural benchmark produces) this script:
  1. loads the frame's LiDAR scan once and takes the returns inside the GT box;
  2. samples the LiDAR-facing faces of the box (self-occluded back faces are
     excluded by construction, matching the thesis external/self-occlusion split);
  3. normalises both point sets to the ideal surface's centroid and RMS radius;
  4. records DCD plus plain Chamfer Distance (cd_t squared, cd_p sqrt) as the
     comparison geometric baselines.

Input CSV is never modified; a superset is written to dcd_instances.csv.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from dcd_baseline.dcd_metric import calc_dcd, normalise_to_reference  # noqa: E402
from gilroy_lidar_replication.build_benchmark import points_in_box, surface_samples  # noqa: E402

BOX_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading")
NEW_COLUMNS = ("dcd", "cd_p", "cd_t", "n_observed", "n_ideal")


def read_instances(path):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def load_points(kitti_root, frame_id):
    path = Path(kitti_root) / "training" / "velodyne" / f"{frame_id}.bin"
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def object_metrics(observed, ideal, alpha, n_lambda, device, normalise):
    """DCD and CD for one object; returns (nan, nan, nan) pairs when undefined."""
    if len(observed) == 0 or len(ideal) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = torch.as_tensor(observed, dtype=torch.float32, device=device).unsqueeze(0)
    ref = torch.as_tensor(ideal, dtype=torch.float32, device=device).unsqueeze(0)
    if normalise:
        obs, ref = normalise_to_reference(obs, ref)
    with torch.no_grad():
        dcd, cd_p, cd_t = calc_dcd(obs, ref, alpha=alpha, n_lambda=n_lambda)
    return float(dcd[0]), float(cd_p[0]), float(cd_t[0])


def compute_rows(rows, kitti_root, resolution, alpha, n_lambda, device, normalise):
    by_frame = defaultdict(list)
    for position, row in enumerate(rows):
        by_frame[row["frame_id"]].append(position)
    for frame_number, (frame_id, positions) in enumerate(sorted(by_frame.items()), start=1):
        points = load_points(kitti_root, frame_id)
        for position in positions:
            row = rows[position]
            box = np.asarray([float(row[field]) for field in BOX_FIELDS], dtype=float)
            ideal, _weights, _head = surface_samples(box, resolution)
            observed = points[points_in_box(points, box)]
            dcd, cd_p, cd_t = object_metrics(observed, ideal, alpha, n_lambda, device, normalise)
            row.update({
                "dcd": f"{dcd:.6f}", "cd_p": f"{cd_p:.6f}", "cd_t": f"{cd_t:.6f}",
                "n_observed": len(observed), "n_ideal": len(ideal),
            })
        if frame_number % 50 == 0:
            print(f"frame {frame_number}/{len(by_frame)}", flush=True)
    return rows


def save_outputs(rows, source_path, output_dir, parameters):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for column in NEW_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    output_path = output_dir / "dcd_instances.csv"
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    protocol = {
        "source_instances": str(source_path),
        "parameters": {
            "alpha": parameters["alpha"],
            "n_lambda": parameters["n_lambda"],
            "normalize": bool(parameters["normalize"]),
            "surface_resolution": parameters["resolution"],
        },
        "dcd_definition": (
            "Density-aware Chamfer Distance (Wu et al., NeurIPS 2021; n_lambda={n_lambda}, "
            "non_reg=False, alpha={alpha}) between the observed LiDAR returns inside the GT box "
            "and uniformly area-weighted samples of the box's LiDAR-facing faces. Default alpha=1.0 "
            "on raw metre coordinates places alpha*d^2 near 1 at the relevant surface-distance scale "
            "(the official code reaches the same regime via unit-normalised shapes with alpha=1000)."
            + (" Both sets are centred and scaled by the ideal surface's centroid and RMS radius "
               "before matching." if parameters.get("normalize") else "")
        ),
        "surface_resolution_per_face_axis": parameters["resolution"],
        "self_occlusion": "excluded (back-facing surfaces are not sampled)",
        "columns_added": list(NEW_COLUMNS),
        "limitations": [
            "bounding-box surface is the proxy for the true object surface",
            "DCD responds to range-driven sparsity as well as occlusion; range and point "
            "count stay available as separate covariate columns",
        ],
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path,
                        default=Path("runs/gilroy_lidar_replication/natural/val/instances.csv"))
    parser.add_argument("--kitti-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--surface-resolution", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="DCD scale parameter; alpha * typical nearest-surface distance^2 should be O(1). "
                             "Raw-metre default 1.0 matches the official operating point (unit-normalised "
                             "shapes with alpha=1000).")
    parser.add_argument("--n-lambda", type=int, default=1)
    parser.add_argument("--normalize", action="store_true",
                        help="Centre and scale both sets by the ideal surface before matching (scale-invariant mode).")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("dcd_baseline/results/natural"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_instances(args.instances)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rows = compute_rows(rows, args.kitti_root, args.surface_resolution, args.alpha, args.n_lambda, device, args.normalize)
    path = save_outputs(rows, args.instances, args.output_dir, dict(vars(args), resolution=args.surface_resolution))
    print(f"WROTE {path} ({len(rows)} instances)")


if __name__ == "__main__":
    main()
