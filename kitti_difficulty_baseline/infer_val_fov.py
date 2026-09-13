#!/usr/bin/env python
"""Run an OpenPCDet detector over the full KITTI val split with the benchmark's
standard camera-FOV point crop, writing the portable predictions JSON.

This is dcd_baseline/infer_natural.py with one material difference: KITTI's
own evaluation path (KittiDataset, FOV_POINTS_ONLY) removes LiDAR returns that
project outside the camera image before inference. The natural-benchmark
JSONs used by the Gilroy/DCD/Waymo baselines deliberately skip that crop, and
those extra out-of-FOV detections depress official AP by several points
(verified against test.py: PointPillars Car AP_R40 87.75/78.39/75.18 with the
crop vs ~71/66/63 without). The KITTI-difficulty baseline's official arm needs
benchmark-faithful predictions, so this script reproduces the KittiDataset
behaviour exactly (same calib projection, same fov_flag test, image shapes
from kitti_infos_val.pkl) while emitting the shared JSON artifact format.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RESEARCH_WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_WORKSPACE))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dcd_baseline.infer_natural import load_frame_ids, load_yaml_config  # noqa: E402
from dcd_baseline.infer_natural import NaturalDataset  # noqa: E402
from pcdet.datasets.dataset import DatasetTemplate  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.utils import calibration_kitti, common_utils  # noqa: E402


class FovDataset(NaturalDataset):
    """NaturalDataset + KittiDataset's FOV_POINTS_ONLY crop."""

    def __init__(self, frame_ids, kitti_root, dataset_cfg, class_names, logger, image_shapes):
        super().__init__(frame_ids, kitti_root, dataset_cfg, class_names, logger)
        self.calib_dir = Path(kitti_root) / "training" / "calib"
        self.image_shapes = image_shapes
        self.fov_only = bool(dataset_cfg.get("FOV_POINTS_ONLY", False))

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        return np.logical_and(np.logical_and(val_flag_1, val_flag_2), pts_rect_depth > 0)

    def __getitem__(self, index):
        frame_id = self.frame_ids[index]
        points = np.fromfile(self.velodyne / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
        if self.fov_only:
            calib = calibration_kitti.Calibration(str(self.calib_dir / f"{frame_id}.txt"))
            pts_rect = calib.lidar_to_rect(points[:, :3])
            points = points[self.get_fov_flag(pts_rect, self.image_shapes[frame_id], calib)]
        return self.prepare_data({"points": points, "frame_id": frame_id})


def run(detector, model_config, checkpoint, frame_ids, kitti_root, output, split="val", infos=None):
    logger = common_utils.create_logger()
    loaded = load_yaml_config(Path(model_config))
    image_shapes = {str(info["point_cloud"]["lidar_idx"]).zfill(6): info["image"]["image_shape"]
                    for info in infos}
    dataset = FovDataset(frame_ids, kitti_root, loaded.DATA_CONFIG, loaded.CLASS_NAMES, logger, image_shapes)
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
            if (index + 1) % 250 == 0 or index + 1 == len(dataset):
                print(f"{detector}: {index + 1}/{len(dataset)}", flush=True)
    artifact = {
        "detector": detector,
        "model_config": str(model_config),
        "checkpoint": str(checkpoint),
        "split": split,
        "class_names": list(loaded.CLASS_NAMES),
        "num_frames": len(frame_ids),
        "fov_points_only": True,
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
    parser.add_argument("--infos", type=Path, default=Path("data/kitti/kitti_infos_val.pkl"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    import pickle

    frame_ids = load_frame_ids(args.kitti_root, args.split, args.max_frames)
    with args.infos.open("rb") as input_file:
        infos = pickle.load(input_file)
    output = args.output or Path(
        f"research_workspace/kitti_difficulty_baseline/results/detections/{args.detector}_{args.split}.json")
    run(args.detector, args.model_config, args.checkpoint, frame_ids, args.kitti_root, output, args.split, infos)


if __name__ == "__main__":
    main()
