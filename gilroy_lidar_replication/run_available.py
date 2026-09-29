#!/usr/bin/env python
"""Run controlled inference/evaluation and Fig. 4/6 plots for usable models."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("gilroy_lidar_replication/models.json"))
    parser.add_argument("--samples", type=Path, default=Path("runs/gilroy_lidar_replication/controlled/samples.csv"))
    parser.add_argument("--max-samples", type=int, help="Smoke-test limit; omit for all 1,640 samples.")
    parser.add_argument("--require-all-seven", action="store_true")
    args = parser.parse_args()
    models = json.loads((ROOT / args.models).read_text(encoding="utf-8"))
    ready = {name: model for name, model in models.items() if model["local_status"] == "ready"}
    blocked = {name: model["local_status"] for name, model in models.items() if name not in ready}
    if blocked:
        print("BLOCKED MODELS:")
        for name, reason in blocked.items():
            print(f"- {name}: {reason}")
        if args.require_all_seven:
            raise SystemExit("Seven-model run refused: compatible artifacts are missing")

    result_arguments, pair_arguments = [], []
    for name, model in ready.items():
        slug = name.lower().replace("-", "_").replace(" ", "_")
        folder = "smoke" if args.max_samples else "detections"
        prediction = Path(f"runs/gilroy_lidar_replication/controlled/{folder}/{slug}.json")
        infer = [
            sys.executable, "gilroy_lidar_replication/infer_controlled.py",
            "--detector", slug, "--model-config", model["config"], "--checkpoint", model["checkpoint"],
            "--samples", str(args.samples), "--output", str(prediction),
        ]
        if args.max_samples:
            infer += ["--max-samples", str(args.max_samples)]
        run(infer)
        if args.max_samples:
            continue
        for arm in ("structured", "random"):
            run([
                sys.executable, "gilroy_lidar_replication/evaluate.py",
                "--instances", str(args.samples), "--predictions", str(prediction),
                "--detector", slug, "--arm", arm,
                "--output-dir", "runs/gilroy_lidar_replication/controlled/evaluation",
            ])
        result_arguments += [
            "--result",
            f"{name}=runs/gilroy_lidar_replication/controlled/evaluation/{slug}_structured_metrics.csv",
        ]
        pair_arguments += [
            "--pair",
            f"{name}=runs/gilroy_lidar_replication/controlled/evaluation/{slug}_structured_metrics.csv=runs/gilroy_lidar_replication/controlled/evaluation/{slug}_random_metrics.csv",
        ]
    if args.max_samples:
        print("Smoke inference complete; evaluation intentionally skipped for incomplete sample coverage")
        return
    run([
        sys.executable, "gilroy_lidar_replication/plot_results.py", *result_arguments,
        "--output-dir", "runs/gilroy_lidar_replication/controlled/plots",
    ])
    run([
        sys.executable, "gilroy_lidar_replication/plot_controls.py", *pair_arguments,
        "--output", "runs/gilroy_lidar_replication/controlled/plots/density_control_map.png",
    ])


if __name__ == "__main__":
    main()
