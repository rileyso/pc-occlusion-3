#!/usr/bin/env python
"""Run an OpenPCDet detector over natural dataset frames (no point-cloud edits).

Writes the research-pipeline prediction artifact ({frame_id, class_name, score,
box_lidar}) so DCD-binned evaluation shares its format with the Gilroy
controlled/natural runs. Dataset-agnostic: any split list under
<kitti-root>/ImageSets/<split>.txt with frames in <kitti-root>/training/velodyne.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))

from gilroy_lidar_replication.infer_controlled import load_yaml_config  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


class NaturalDataset(DatasetTemplate):
    def __init__(self, frame_ids, kitti_root, dataset_cfg, class_names, logger):
        super().__init__(dataset_cfg, class_names, training=False, root_path=Path(kitti_root), logger=logger)
        self.frame_ids = frame_ids
        self.velodyne = Path(kitti_root) / "training" / "velodyne"

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        points = np.fromfile(self.velodyne / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
        return self.prepare_data({"points": points, "frame_id": frame_id})


def load_frame_ids(kitti_root, split, max_frames=None):
    frame_ids = (Path(kitti_root) / "ImageSets" / f"{split}.txt").read_text(encoding="utf-8").split()
    return frame_ids[:max_frames] if max_frames else frame_ids


def run(detector, model_config, checkpoint, frame_ids, kitti_root, output, split="val", expect_class=None):
    logger = common_utils.create_logger()
    loaded = load_yaml_config(Path(model_config))
    if expect_class and expect_class not in loaded.CLASS_NAMES:
        raise ValueError(f"{detector} config cannot detect {expect_class}: {loaded.CLASS_NAMES}")
    dataset = NaturalDataset(frame_ids, kitti_root, loaded.DATA_CONFIG, loaded.CLASS_NAMES, logger)
    model = build_network(loaded.MODEL, len(loaded.CLASS_NAMES), dataset)
    model.load_params_from_file(str(checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()
    detections = []
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = dataset.collate_batch([dataset[index]])
            load_data_to_gpu(batch)
            predictions, _ = model(batch)
            values = zip(
                predictions[0]["pred_boxes"].cpu().numpy(),
                predictions[0]["pred_scores"].cpu().numpy(),
                predictions[0]["pred_labels"].cpu().numpy(),
            )
            for detection_index, (box, score, label) in enumerate(values):
                detections.append({
                    "frame_id": dataset.frame_ids[index],
                    "detection_index": detection_index,
                    "class_name": loaded.CLASS_NAMES[int(label) - 1],
                    "score": float(score),
                    "box_lidar": [float(value) for value in box[:7]],
                })
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"{detector}: {index + 1}/{len(dataset)}", flush=True)
    artifact = {
        "detector": detector,
        "model_config": str(model_config),
        "checkpoint": str(checkpoint),
        "split": split,
        "class_names": list(loaded.CLASS_NAMES),
        "num_frames": len(frame_ids),
        "detections": detections,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    print(f"WROTE {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--kitti-root", type=Path, default=Path("data/kitti"))
    parser.add_argument("--expect-class", default=None, help="Fail early if the config cannot detect this class (e.g. Pedestrian).")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    frame_ids = load_frame_ids(args.kitti_root, args.split, args.max_frames)
    output = args.output or Path(f"dcd_baseline/results/detections/{args.detector}_{args.split}.json")
    run(args.detector, args.model_config, args.checkpoint, frame_ids, args.kitti_root, output, args.split, args.expect_class)


if __name__ == "__main__":
    main()
