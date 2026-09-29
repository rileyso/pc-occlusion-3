#!/usr/bin/env python
"""Run an OpenPCDet model on controlled samples without copying point clouds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gilroy_lidar_replication.build_controlled import BOX_FIELDS, apply_control
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from research_pipeline.main_pipeline import load_yaml_config


class ControlledDataset(DatasetTemplate):
    def __init__(self, samples, kitti_root, dataset_cfg, class_names, resolution, logger):
        super().__init__(dataset_cfg, class_names, training=False, root_path=Path(kitti_root), logger=logger)
        self.samples = samples
        self.velodyne = Path(kitti_root) / "training" / "velodyne"
        self.resolution = resolution

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        points = np.fromfile(self.velodyne / f"{row['source_frame_id']}.bin", dtype=np.float32).reshape(-1, 4)
        box = np.asarray([row[name] for name in BOX_FIELDS], dtype=np.float32)
        transformed, details = apply_control(
            points, box, float(row["target_occlusion_percent"]) / 100.0,
            row["direction"], row["arm"], int(row["seed"]), self.resolution,
        )
        if details["retained_target_points"] != int(row["retained_target_points"]):
            raise ValueError(f"Controlled transform does not reproduce manifest sample {row['sample_id']}")
        return self.prepare_data({"points": transformed, "frame_id": row["sample_id"]})


def load_samples(path, arm="both", max_samples=None):
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if arm != "both":
        rows = [row for row in rows if row["arm"] == arm]
    return rows[:max_samples] if max_samples else rows


def run(detector, model_config, checkpoint, samples_path, kitti_root, output, arm="both", max_samples=None, resolution=30):
    logger = common_utils.create_logger()
    loaded = load_yaml_config(Path(model_config))
    if "Pedestrian" not in loaded.CLASS_NAMES:
        raise ValueError(f"{detector} config cannot detect Pedestrian: {loaded.CLASS_NAMES}")
    samples = load_samples(samples_path, arm, max_samples)
    dataset = ControlledDataset(samples, kitti_root, loaded.DATA_CONFIG, loaded.CLASS_NAMES, resolution, logger)
    model = build_network(loaded.MODEL, len(loaded.CLASS_NAMES), dataset)
    model.load_params_from_file(str(checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()
    detections = []
    with torch.no_grad():
        for index, sample in enumerate(dataset):
            batch = dataset.collate_batch([sample])
            load_data_to_gpu(batch)
            predictions, _ = model(batch)
            values = zip(
                predictions[0]["pred_boxes"].cpu().numpy(),
                predictions[0]["pred_scores"].cpu().numpy(),
                predictions[0]["pred_labels"].cpu().numpy(),
            )
            for detection_index, (box, score, label) in enumerate(values):
                detections.append({
                    "frame_id": samples[index]["sample_id"],
                    "detection_index": detection_index,
                    "class_name": loaded.CLASS_NAMES[int(label) - 1],
                    "score": float(score),
                    "box_lidar": [float(value) for value in box[:7]],
                })
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"{detector}: {index + 1}/{len(dataset)}")
    artifact = {
        "detector": detector,
        "model_config": str(model_config),
        "checkpoint": str(checkpoint),
        "samples": str(samples_path),
        "arm": arm,
        "num_samples": len(samples),
        "class_names": list(loaded.CLASS_NAMES),
        "detections": detections,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    print(f"WROTE {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=Path("runs/gilroy_lidar_replication/controlled/samples.csv"))
    parser.add_argument("--kitti-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--arm", choices=("both", "structured", "random"), default="both")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--surface-resolution", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"runs/gilroy_lidar_replication/controlled/detections/{args.detector}.json")
    run(args.detector, args.model_config, args.checkpoint, args.samples, args.kitti_root, output, args.arm, args.max_samples, args.surface_resolution)


if __name__ == "__main__":
    main()
