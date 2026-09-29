#!/usr/bin/env python
"""Standalone, extensible detector research pipeline.

The pipeline deliberately keeps its orchestration and processing in this file.
OpenPCDet supplies model construction and CUDA operators, but no code is
imported from the older ``research`` package.

Typical runs::

    conda run -n openpcdet python research_pipeline/main_pipeline.py \
      --dataset kitti --detectors pvrcnn,second --split val_500

    conda run -n openpcdet python research_pipeline/main_pipeline.py \
      --dataset kitti --detectors all --stages metrics,report

    conda run -n openpcdet python research_pipeline/main_pipeline.py \
      --dataset nuscenes --detectors cbgs_voxel01_res3d_centerpoint --split val

    conda run -n openpcdet python research_pipeline/main_pipeline.py \
      --dataset pandaset --stages metrics --split val

    conda run -n openpcdet python research_pipeline/main_pipeline.py \
      --dataset pandaset --detectors experiment \
      --model-config experiment=path/to/model.yaml \
      --checkpoint experiment=path/to/model.pth

The registry covers KITTI, nuScenes, Waymo, PandaSet, Lyft, ONCE, Custom, and
Argoverse 2 through their native OpenPCDet loaders. Dataset-specific model
configs are discovered from ``tools/cfgs/*_models``. A model not bundled with
OpenPCDet can be supplied with matching ``--model-config NAME=PATH`` and
``--checkpoint NAME=PATH`` options.

Add a metric by defining one callback and adding a ``MetricSpec`` to
``METRIC_REGISTRY``.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openpcdet-matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_numba_cuda_driver(environ=None, platform_release=None, path_exists=None):
    """Prefer WSL's CUDA proxy when a native Linux driver shadows it."""
    environ = os.environ if environ is None else environ
    if environ.get("NUMBA_CUDA_DRIVER"):
        return None
    platform_release = platform.release() if platform_release is None else platform_release
    path_exists = Path.exists if path_exists is None else path_exists
    wsl_driver = Path("/usr/lib/wsl/lib/libcuda.so")
    if "microsoft" in platform_release.lower() and path_exists(wsl_driver):
        environ["NUMBA_CUDA_DRIVER"] = str(wsl_driver)
        return wsl_driver
    return None


# Must run before a dataset evaluator imports numba.cuda.
configure_numba_cuda_driver()

from pcdet.config import cfg_from_yaml_file
from pcdet.utils import box_utils, common_utils


STAGE_ORDER = ("infer", "benchmark", "match", "metrics", "report")
DETECTION_FIELDS = ("frame_id", "detection_index", "class_name", "score", "box_lidar")
MATCH_FIELDS = (
    "run_signature", "dataset", "split", "frame_id", "gt_index", "gt_class", "difficulty",
    "occlusion", "truncation", "gt_x", "gt_y", "gt_z", "gt_dx", "gt_dy",
    "gt_dz", "gt_heading", "matched", "matched_detection_index", "score",
    "iou_3d", "best_iou_3d", "best_detection_index", "best_score",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Native OpenPCDet dataset and normalization callbacks."""

    name: str
    root: Path
    config_file: Path
    splits: tuple
    split_info_paths: dict
    class_names: tuple
    target_class: str
    info_attribute: str
    frame_id_from_info: Callable
    annotations_from_info: Callable
    points_from_info: Callable
    split_versions: Optional[dict] = None


@dataclass(frozen=True)
class DetectorSpec:
    """An OpenPCDet model config and conventional checkpoint locations."""

    name: str
    cfg_file: Path
    checkpoint_candidates: tuple


@dataclass(frozen=True)
class MetricContext:
    """Dataset-neutral inputs passed to every object metric callback."""

    ground_truth: dict
    points_in_box: np.ndarray


@dataclass(frozen=True)
class MetricSpec:
    """A scalar object metric and optional bins used by the report plot."""

    name: str
    compute: Callable[[MetricContext], float]
    label: str
    bin_edges: Optional[tuple] = None
    bin_labels: Optional[tuple] = None


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Official or explicitly surrogate evaluation protocol for one dataset."""

    name: str
    local_evaluation: bool
    paper_comparable_splits: tuple
    primary_metric: str
    reference_title: str
    reference_url: str
    note: str = ""


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetSpec
    detectors: tuple
    detector_specs: dict
    split: str
    stages: tuple
    metrics: tuple
    target_class: str
    min_iou: float
    data_root: Path
    frame_ids: tuple
    output_dir: Path
    run_signature: str
    force: bool
    dry_run: bool


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def standard_frame_id(info):
    return str(info["point_cloud"]["lidar_idx"])


def path_frame_id(info):
    return Path(info["lidar_path"]).stem


def waymo_frame_id(info):
    return str(info["frame_id"])


def once_frame_id(info):
    return f'{info["sequence_id"]}/{info["frame_id"]}'


def pandaset_frame_id(info):
    return f'{info["sequence"]}/{int(info["frame_idx"]):02d}'


def argo2_frame_id(info):
    path = info["point_cloud"]["velodyne_path"]
    return Path(path).stem


def filter_aligned_metadata(metadata, valid):
    filtered = {}
    for key, value in metadata.items():
        array = np.asarray(value)
        filtered[key] = array[valid] if array.ndim > 0 and array.shape[0] == len(valid) else value
    return filtered


def standard_annotations(dataset, info):
    annos = info.get("annos", {})
    boxes = annos.get("gt_boxes_lidar", np.empty((0, 7), dtype=np.float32))
    names = np.asarray(annos.get("name", ()))
    metadata = annos
    if len(names) != len(boxes):
        annotation_indices = np.asarray(annos.get("index", ()))
        if len(annotation_indices) != len(names):
            raise ValueError("Annotation names and lidar boxes have incompatible lengths")
        valid = annotation_indices >= 0
        if int(valid.sum()) != len(boxes):
            raise ValueError("Annotation index mask does not align with lidar boxes")
        names = names[valid]
        metadata = filter_aligned_metadata(annos, valid)
    if dataset.dataset_cfg.get("INFO_WITH_FAKELIDAR", False) and len(boxes):
        boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(boxes)
    return names, boxes, metadata


def top_level_annotations(dataset, info):
    return info.get("gt_names", ()), info.get("gt_boxes", np.empty((0, 7), dtype=np.float32)), info


def nuscenes_annotations(dataset, info):
    names = np.asarray(info.get("gt_names", ()))
    boxes = np.asarray(info.get("gt_boxes", np.empty((0, 7), dtype=np.float32)))
    metadata = info
    minimum = dataset.dataset_cfg.get("FILTER_MIN_POINTS_IN_GT", False)
    if minimum and "num_lidar_pts" in info:
        valid = np.asarray(info["num_lidar_pts"]) >= int(minimum)
        names, boxes = names[valid], boxes[valid]
        metadata = filter_aligned_metadata(info, valid)
    return names, boxes, metadata


def once_annotations(dataset, info):
    annos = info.get("annos", {})
    return annos.get("name", ()), annos.get("boxes_3d", np.empty((0, 7), dtype=np.float32)), annos


def argo2_annotations(dataset, info):
    annos = info.get("annos", {})
    if not annos:
        return (), np.empty((0, 7), dtype=np.float32), annos
    boxes = np.concatenate(
        [annos["location"], annos["dimensions"], np.asarray(annos["rotation_y"])[:, None]], axis=1
    )
    return annos.get("name", ()), boxes, annos


def pandaset_annotations(dataset, info):
    pose = dataset._get_pose(info)
    boxes, names, _ = dataset._get_annotations(info, pose)
    return names, boxes, info


def standard_points(dataset, info, index):
    return dataset.get_lidar(info["point_cloud"]["lidar_idx"])


def sweep_points(dataset, info, index):
    return dataset.get_lidar_with_sweeps(index, max_sweeps=dataset.dataset_cfg.MAX_SWEEPS)


def waymo_points(dataset, info, index):
    pc_info = info["point_cloud"]
    return dataset.get_lidar(pc_info["lidar_sequence"], pc_info["sample_idx"])


def once_points(dataset, info, index):
    return dataset.get_lidar(info["sequence_id"], info["frame_id"])


def pandaset_points(dataset, info, index):
    pose = dataset._get_pose(info)
    return dataset._get_lidar_points(info, pose)


def argo2_points(dataset, info, index):
    return dataset.get_lidar(argo2_frame_id(info))


DATASET_REGISTRY = {
    "kitti": DatasetSpec(
        "kitti", Path("data/kitti"), Path("tools/cfgs/dataset_configs/kitti_dataset.yaml"),
        ("train", "val", "val_500", "test"),
        {"train": ("kitti_infos_train.pkl",), "val": ("kitti_infos_val.pkl",),
         "val_500": ("kitti_infos_val_500.pkl",), "test": ("kitti_infos_test.pkl",)},
        ("Car", "Pedestrian", "Cyclist"), "Car", "kitti_infos",
        standard_frame_id, standard_annotations, standard_points,
    ),
    "nuscenes": DatasetSpec(
        "nuscenes", Path("data/nuscenes"), Path("tools/cfgs/dataset_configs/nuscenes_dataset.yaml"),
        ("train", "val", "test"),
        {"train": ("nuscenes_infos_10sweeps_train.pkl",), "val": ("nuscenes_infos_10sweeps_val.pkl",),
         "test": ("nuscenes_infos_10sweeps_test.pkl",)},
        ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle",
         "bicycle", "pedestrian", "traffic_cone"), "car", "infos",
        path_frame_id, nuscenes_annotations, sweep_points, {"test": "v1.0-test"},
    ),
    "waymo": DatasetSpec(
        "waymo", Path("data/waymo"), Path("tools/cfgs/dataset_configs/waymo_dataset.yaml"),
        ("train", "val", "test"), {}, ("Vehicle", "Pedestrian", "Cyclist"), "Vehicle", "infos",
        waymo_frame_id, standard_annotations, waymo_points,
    ),
    "pandaset": DatasetSpec(
        "pandaset", Path("data/pandaset"), Path("tools/cfgs/dataset_configs/pandaset_dataset.yaml"),
        ("train", "val", "test"),
        {"train": ("pandaset_infos_train.pkl",), "val": ("pandaset_infos_val.pkl",),
         "test": ("pandaset_infos_test.pkl",)},
        ("Car", "Truck", "Other Vehicle", "Motorcycle", "Bus", "Bicycle", "Pedestrian", "Animal"),
        "Car", "pandaset_infos", pandaset_frame_id, pandaset_annotations, pandaset_points,
    ),
    "lyft": DatasetSpec(
        "lyft", Path("data/lyft"), Path("tools/cfgs/dataset_configs/lyft_dataset.yaml"),
        ("train", "val", "test"),
        {"train": ("lyft_infos_train.pkl",), "val": ("lyft_infos_val.pkl",),
         "test": ("lyft_infos_test.pkl",)},
        ("car", "truck", "bus", "emergency_vehicle", "other_vehicle", "motorcycle", "bicycle",
         "pedestrian", "animal"), "car", "infos", path_frame_id, top_level_annotations, sweep_points,
        {"test": "test"},
    ),
    "once": DatasetSpec(
        "once", Path("data/once"), Path("tools/cfgs/dataset_configs/once_dataset.yaml"),
        ("train", "val", "test"),
        {"train": ("once_infos_train.pkl",), "val": ("once_infos_val.pkl",),
         "test": ("once_infos_test.pkl",)},
        ("Car", "Bus", "Truck", "Pedestrian", "Cyclist"), "Car", "once_infos",
        once_frame_id, once_annotations, once_points,
    ),
    "custom": DatasetSpec(
        "custom", Path("data/custom"), Path("tools/cfgs/dataset_configs/custom_dataset.yaml"),
        ("train", "val"),
        {"train": ("custom_infos_train.pkl",), "val": ("custom_infos_val.pkl",)},
        ("Vehicle", "Pedestrian", "Cyclist"), "Vehicle", "custom_infos",
        standard_frame_id, standard_annotations, standard_points,
    ),
    "argo2": DatasetSpec(
        "argo2", Path("data/argo2"), Path("tools/cfgs/dataset_configs/argo2_dataset.yaml"),
        ("train", "val"),
        {"train": ("argo2_infos_train.pkl",), "val": ("argo2_infos_val.pkl",)},
        ("Regular_vehicle", "Pedestrian", "Bicyclist", "Motorcyclist", "Wheeled_rider", "Bollard",
         "Construction_cone", "Sign", "Construction_barrel", "Stop_sign", "Mobile_pedestrian_crossing_sign",
         "Large_vehicle", "Bus", "Box_truck", "Truck", "Vehicular_trailer", "Truck_cab", "School_bus",
         "Articulated_bus", "Message_board_trailer", "Bicycle", "Motorcycle", "Wheeled_device", "Wheelchair",
         "Stroller", "Dog"),
        "Regular_vehicle", "argo2_infos", argo2_frame_id, argo2_annotations, argo2_points,
    ),
}


def checkpoint_candidates(dataset_name, model_name):
    # Dataset-qualified paths avoid accidentally loading, for example, a
    # KITTI SECOND checkpoint into the Custom or ONCE SECOND config.
    return (Path("checkpoints") / dataset_name / f"{model_name}.pth",)


def discover_detector_registry():
    registry = {}
    for dataset_name in DATASET_REGISTRY:
        model_dir = REPO_ROOT / "tools" / "cfgs" / f"{dataset_name}_models"
        models = {}
        for config_path in sorted(model_dir.glob("*.yaml")):
            name = config_path.stem.lower()
            models[name] = DetectorSpec(
                name, config_path.relative_to(REPO_ROOT), checkpoint_candidates(dataset_name, name)
            )
        registry[dataset_name] = models

    # Preserve the established KITTI command names and downloaded checkpoint names.
    kitti = registry["kitti"]
    aliases = {
        "pvrcnn": ("pv_rcnn.yaml", "pv_rcnn_8369.pth"),
        "pointpillar": ("pointpillar.yaml", "pointpillar.pth"),
        "second": ("second.yaml", "second.pth"),
        "pointrcnn": ("pointrcnn.yaml", "pointrcnn.pth"),
        "voxel_rcnn_car": ("voxel_rcnn_car.yaml", "voxel_rcnn_car.pth"),
        "parta2": ("PartA2.yaml", "parta2.pth"),
        "pvrcnn_plusplus": ("pv_rcnn_plusplus_reproduced_by_community.yaml", "pvrcnn_plusplus.pth"),
    }
    for name, (config_name, checkpoint_name) in aliases.items():
        config_path = Path("tools/cfgs/kitti_models") / config_name
        kitti[name] = DetectorSpec(name, config_path, (Path("checkpoints") / checkpoint_name,))
    kitti.pop("pv_rcnn", None)
    kitti.pop("pv_rcnn_plusplus_reproduced_by_community", None)
    return registry


DETECTOR_REGISTRY = discover_detector_registry()


def metric_distance(context):
    return float(np.linalg.norm(context.ground_truth["box_lidar"][:3]))


def metric_point_count(context):
    return int(len(context.points_in_box))


def metric_point_density(context):
    box = context.ground_truth["box_lidar"]
    volume = max(float(box[3] * box[4] * box[5]), 1e-6)
    return float(len(context.points_in_box) / volume)


METRIC_REGISTRY = {
    "distance_m": MetricSpec(
        "distance_m", metric_distance, "Distance (m)",
        (0.0, 20.0, 40.0, 60.0, 80.0, float("inf")),
        ("0-20", "20-40", "40-60", "60-80", "80+"),
    ),
    "num_points_in_gt": MetricSpec(
        "num_points_in_gt", metric_point_count, "Points in GT box",
        (-0.5, 0.5, 10.5, 50.5, 200.5, float("inf")),
        ("0", "1-10", "11-50", "51-200", "201+"),
    ),
    "point_density": MetricSpec("point_density", metric_point_density, "Point density"),
}


BENCHMARK_PROTOCOLS = {
    "kitti": BenchmarkProtocol(
        "KITTI official object detection (AP_R40)", True, ("val",), "3D AP_R40",
        "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite",
        "https://www.cvlibs.net/publications/Geiger2012CVPR.pdf",
        "Validation is the reproducible local comparison; test numbers require the KITTI server.",
    ),
    "nuscenes": BenchmarkProtocol(
        "nuScenes detection_cvpr_2019", True, ("val",), "NDS / mAP",
        "nuScenes: A Multimodal Dataset for Autonomous Driving",
        "https://openaccess.thecvf.com/content_CVPR_2020/papers/"
        "Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf",
    ),
    "waymo": BenchmarkProtocol(
        "Waymo Open Dataset 3D detection", True, ("val",), "mAPH L2",
        "Scalability in Perception for Autonomous Driving: Waymo Open Dataset",
        "https://openaccess.thecvf.com/content_CVPR_2020/papers/"
        "Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf",
    ),
    "pandaset": BenchmarkProtocol(
        "No official PandaSet detection protocol", False, (), "N/A",
        "PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving",
        "https://arxiv.org/abs/2112.12610",
        "The bundled OpenPCDet PandaSet adapter explicitly returns no evaluation metrics.",
    ),
    "lyft": BenchmarkProtocol(
        "Lyft Level 5 3D detection mAP", True, ("val",), "mAP",
        "Lyft Level 5 AV Dataset devkit",
        "https://github.com/lyft/nuscenes-devkit",
        "This is the Lyft challenge/devkit protocol, not a universally standardized paper split.",
    ),
    "once": BenchmarkProtocol(
        "ONCE orientation-aware 3D AP", True, ("val",), "mAP",
        "One Million Scenes for Autonomous Driving: ONCE Dataset",
        "https://arxiv.org/abs/2106.11037",
    ),
    "custom": BenchmarkProtocol(
        "KITTI-compatible surrogate", True, (), "3D AP_R40",
        "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite",
        "https://www.cvlibs.net/publications/Geiger2012CVPR.pdf",
        "Custom data has no source benchmark; its class mapping makes this a KITTI-style surrogate only.",
    ),
    "argo2": BenchmarkProtocol(
        "Argoverse 2 3D object detection", True, ("val",), "CDS / mAP",
        "Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting",
        "https://openreview.net/pdf?id=vKQGe36av4k",
    ),
}


def configure_dataset_split(dataset_cfg, spec, split, data_root):
    """Return an inference config targeting one logical dataset split."""
    configured = copy.deepcopy(dataset_cfg)
    configured.DATA_PATH = str(data_root)
    configured.DATA_SPLIT["test"] = split
    if spec.split_versions and split in spec.split_versions:
        configured.VERSION = spec.split_versions[split]
    info_paths = spec.split_info_paths.get(split)
    if info_paths:
        configured.INFO_PATH["test"] = list(info_paths)
        # ONCE indexes INFO_PATH by the split name rather than DatasetTemplate.mode.
        configured.INFO_PATH[split] = list(info_paths)
    if "SAMPLED_INTERVAL" in configured:
        configured.SAMPLED_INTERVAL["test"] = 1
    return configured


def load_yaml_config(path):
    full_path = path if Path(path).is_absolute() else REPO_ROOT / path
    loaded = EasyDict(ROOT_DIR=REPO_ROOT, LOCAL_RANK=0)
    try:
        with working_directory(REPO_ROOT / "tools"):
            cfg_from_yaml_file(str(full_path), loaded)
    except FileNotFoundError:
        # Most bundled configs resolve bases from tools/, but a few and many
        # external configs use paths relative to the model config itself.
        loaded = EasyDict(ROOT_DIR=REPO_ROOT, LOCAL_RANK=0)
        with working_directory(full_path.parent):
            cfg_from_yaml_file(str(full_path), loaded)
    return loaded


def instantiate_native_dataset(spec, dataset_cfg, class_names, data_root, logger):
    from pcdet.datasets import __all__ as dataset_classes

    dataset_class = dataset_classes.get(dataset_cfg.DATASET)
    if dataset_class is None:
        raise ImportError(
            f"{dataset_cfg.DATASET} is unavailable; install the optional dependencies for {spec.name}"
        )
    try:
        return dataset_class(
            dataset_cfg=dataset_cfg,
            class_names=list(class_names),
            training=False,
            root_path=data_root,
            logger=logger,
        )
    except (NameError, ModuleNotFoundError) as error:
        raise ImportError(
            f"Could not initialize {spec.name}; install its optional dataset dependencies ({error})"
        ) from error


def metadata_at(metadata, names, index, default=""):
    for name in names:
        if name not in metadata:
            continue
        values = metadata[name]
        try:
            value = values[index]
        except (IndexError, KeyError, TypeError):
            value = values
        if isinstance(value, np.generic):
            value = value.item()
        return value
    return default


class NativeDatasetRuntime:
    """Dataset-neutral view over one of OpenPCDet's native dataset classes."""

    def __init__(self, spec, native_dataset):
        self.spec = spec
        self.native_dataset = native_dataset
        self.infos = list(getattr(native_dataset, spec.info_attribute))
        self.frame_ids = tuple(spec.frame_id_from_info(info) for info in self.infos)
        duplicates = sorted(frame_id for frame_id, count in Counter(self.frame_ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"{spec.name} produced duplicate frame ids: {', '.join(duplicates[:5])}")
        self.index_by_frame = {frame_id: index for index, frame_id in enumerate(self.frame_ids)}

    @classmethod
    def create(cls, spec, dataset_cfg, class_names, split, data_root, logger):
        configured = configure_dataset_split(dataset_cfg, spec, split, data_root)
        native = instantiate_native_dataset(spec, configured, class_names, data_root, logger)
        runtime = cls(spec, native)
        if not runtime.frame_ids:
            raise FileNotFoundError(
                f"No {spec.name} samples loaded for split '{split}' under {data_root}; "
                "generate the OpenPCDet info files first"
            )
        return runtime

    def ground_truth(self, frame_id, target_class):
        index = self.index_by_frame[str(frame_id)]
        info = self.infos[index]
        names, boxes, metadata = self.spec.annotations_from_info(self.native_dataset, info)
        names = np.asarray(names)
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = np.empty((0, 7), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] < 7 or len(names) != len(boxes):
            raise ValueError(f"Invalid annotations for {self.spec.name} frame {frame_id}")

        records = []
        for source_index in np.flatnonzero(names.astype(str) == target_class):
            box = boxes[source_index, :7]
            if not np.all(np.isfinite(box)) or np.any(box[3:6] <= 0):
                raise ValueError(f"Invalid 3D box for {self.spec.name} frame {frame_id}, GT {source_index}")
            records.append(
                {
                    "frame_id": str(frame_id),
                    "gt_index": len(records),
                    "class_name": str(names[source_index]),
                    "difficulty": metadata_at(metadata, ("difficulty", "difficulty_level"), source_index),
                    "occlusion": metadata_at(metadata, ("occluded", "occlusion"), source_index),
                    "truncation": metadata_at(metadata, ("truncated", "truncation"), source_index),
                    "box_lidar": box.astype(np.float32, copy=False),
                }
            )
        return records

    def points(self, frame_id):
        index = self.index_by_frame[str(frame_id)]
        points = np.asarray(self.spec.points_from_info(self.native_dataset, self.infos[index], index))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Invalid point cloud for {self.spec.name} frame {frame_id}")
        return points


class SelectedNativeDataset:
    """Subset a native dataset while preserving the attributes models inspect."""

    def __init__(self, runtime, frame_ids):
        self.runtime = runtime
        self.native_dataset = runtime.native_dataset
        self.frame_ids = tuple(map(str, frame_ids))
        missing = [frame_id for frame_id in self.frame_ids if frame_id not in runtime.index_by_frame]
        if missing:
            raise ValueError(f"Model dataset is missing selected frames: {', '.join(missing[:5])}")
        self.indices = tuple(runtime.index_by_frame[frame_id] for frame_id in self.frame_ids)

    def __getattr__(self, name):
        return getattr(self.native_dataset, name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.native_dataset[self.indices[index]]
        sample["frame_id"] = self.frame_ids[index]
        return sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the standalone detector research pipeline.")
    parser.add_argument("--dataset", default="kitti", help="Dataset registry name.")
    parser.add_argument("--detectors", default="all", help="Comma-separated detector names, or 'all'.")
    parser.add_argument(
        "--model-config", action="append", default=[], metavar="NAME=PATH",
        help="Register or override a detector model config (repeatable; pair new names with --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH",
        help="Override a detector checkpoint path (repeatable).",
    )
    parser.add_argument("--split", help="Dataset split name (defaults to val_500 for KITTI, val otherwise).")
    parser.add_argument(
        "--stages", default="all",
        help="Comma-separated infer,benchmark,match,metrics,report, or 'all'.",
    )
    parser.add_argument("--metrics", default="all", help="Comma-separated metric names, or 'all'.")
    parser.add_argument("--target-class", help="Target class override; defaults to the dataset preset.")
    parser.add_argument("--min-iou", type=float, default=0.7, help="3D IoU threshold for a match.")
    parser.add_argument("--data-root", type=Path, help="Dataset root override.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-frames", type=int, help="Use only the first N split frames for a smoke run.")
    parser.add_argument("--force", action="store_true", help="Replace artifacts for selected stages.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the run plan only.")
    parser.add_argument("--list-options", action="store_true", help="List registered datasets, detectors, metrics, and stages.")
    return parser.parse_args(argv)


def split_selection(value, choices, label):
    if value.strip().lower() == "all":
        return tuple(choices)
    selected = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = [item for item in selected if item not in choices]
    if not selected or unknown:
        detail = f" Unknown: {', '.join(unknown)}." if unknown else ""
        raise ValueError(f"Select at least one valid {label}.{detail} Choices: {', '.join(choices)}")
    return selected


def detector_checkpoint(detector):
    return next((path for path in detector.checkpoint_candidates if (REPO_ROOT / path).exists()), None)


def detector_is_available(detector):
    return (REPO_ROOT / detector.cfg_file).exists() and detector_checkpoint(detector) is not None


def parse_named_paths(values, option_name):
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} expects NAME=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        name, raw_path = name.strip().lower(), raw_path.strip()
        if not name or not raw_path or name in parsed:
            raise ValueError(f"Invalid or duplicate {option_name} entry: {value}")
        path = Path(raw_path).expanduser()
        parsed[name] = path if path.is_absolute() else REPO_ROOT / path
    return parsed


def configured_detector_registry(dataset_name, model_config_values, checkpoint_values):
    registry = dict(DETECTOR_REGISTRY[dataset_name])
    model_configs = parse_named_paths(model_config_values, "--model-config")
    checkpoints = parse_named_paths(checkpoint_values, "--checkpoint")
    unknown_checkpoints = set(checkpoints) - set(registry) - set(model_configs)
    if unknown_checkpoints:
        raise ValueError(
            "Checkpoint override has no bundled or supplied model config: " + ", ".join(sorted(unknown_checkpoints))
        )
    for name in set(model_configs) | set(checkpoints):
        existing = registry.get(name)
        config_path = model_configs.get(name, existing.cfg_file if existing else None)
        if config_path is None:
            raise ValueError(f"Detector '{name}' needs --model-config")
        if name in checkpoints:
            candidates = (checkpoints[name],)
        elif existing is not None:
            candidates = existing.checkpoint_candidates
        else:
            raise ValueError(f"New detector '{name}' needs --checkpoint")
        registry[name] = DetectorSpec(name, Path(config_path), tuple(map(Path, candidates)))
    return registry


def resolve_detectors(value, dataset_name, compatible_registry):
    compatible = tuple(compatible_registry)
    if value.strip().lower() == "all":
        available = tuple(name for name in compatible if detector_is_available(compatible_registry[name]))
        skipped = [name for name in compatible if name not in available]
        for name in skipped:
            print(f"SKIP unavailable detector {name}: config or checkpoint is missing")
        if not available:
            raise FileNotFoundError(f"No runnable detectors are registered for dataset '{dataset_name}'")
        return available
    selected = split_selection(value, compatible, "detector")
    for name in selected:
        detector = compatible_registry[name]
        if not (REPO_ROOT / detector.cfg_file).exists():
            raise FileNotFoundError(f"Detector '{name}' is missing config: {detector.cfg_file}")
        if detector_checkpoint(detector) is None:
            expected = ", ".join(map(str, detector.checkpoint_candidates))
            raise FileNotFoundError(f"Detector '{name}' is missing a checkpoint; expected one of: {expected}")
    return selected


def select_frames(runtime, max_frames=None):
    frame_ids = runtime.frame_ids
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive")
        frame_ids = frame_ids[:max_frames]
    if not frame_ids:
        raise ValueError("Dataset split contains no frames")
    return frame_ids


def resolve_run_config(args):
    if args.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choices: {', '.join(DATASET_REGISTRY)}")
    dataset = DATASET_REGISTRY[args.dataset]
    split = args.split or ("val_500" if args.dataset == "kitti" else "val")
    if split not in dataset.splits:
        raise ValueError(f"Unknown split '{split}' for {args.dataset}. Choices: {', '.join(dataset.splits)}")
    if not 0.0 <= args.min_iou <= 1.0:
        raise ValueError("--min-iou must be between 0 and 1")
    target_class = args.target_class or dataset.target_class
    if target_class not in dataset.class_names:
        raise ValueError(
            f"Unknown target class '{target_class}' for {dataset.name}. Choices: {', '.join(dataset.class_names)}"
        )
    data_root = (args.data_root or dataset.root).expanduser()
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    stages = split_selection(args.stages, STAGE_ORDER, "stage")
    detector_registry = configured_detector_registry(args.dataset, args.model_config, args.checkpoint)
    needs_detectors = any(stage in stages for stage in ("infer", "benchmark", "match", "report"))
    detectors = resolve_detectors(args.detectors, dataset.name, detector_registry) if needs_detectors else ()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    dataset_config = load_yaml_config(dataset.config_file)
    runtime = NativeDatasetRuntime.create(
        dataset, dataset_config, dataset.class_names, split, data_root, logging.getLogger("research_pipeline")
    )
    frame_ids = select_frames(runtime, args.max_frames)
    signature_values = [dataset.name, split, target_class, str(args.min_iou), *frame_ids]
    run_signature = hashlib.sha256("\n".join(signature_values).encode("utf-8")).hexdigest()[:16]
    return RunConfig(
        dataset=dataset,
        detectors=detectors,
        detector_specs={name: detector_registry[name] for name in detectors},
        split=split,
        stages=stages,
        metrics=split_selection(args.metrics, tuple(METRIC_REGISTRY), "metric"),
        target_class=target_class,
        min_iou=args.min_iou,
        data_root=data_root,
        frame_ids=frame_ids,
        output_dir=output_dir / dataset.name / split,
        run_signature=run_signature,
        force=args.force,
        dry_run=args.dry_run,
    )


def artifact_paths(config, detector=None):
    paths = {
        "metrics": config.output_dir / "metrics" / "geometry.csv",
        "summary": config.output_dir / "reports" / "detector_summary.csv",
        "plot": config.output_dir / "reports" / "geometry_match_rates.png",
        "benchmark_table": config.output_dir / "benchmarks" / "paper_table.csv",
        "benchmark_markdown": config.output_dir / "benchmarks" / "paper_table.md",
        "benchmark_long": config.output_dir / "benchmarks" / "official_metrics_long.csv",
    }
    if detector:
        paths["detections"] = config.output_dir / "detections" / f"{detector}.json"
        paths["evaluation_predictions"] = config.output_dir / "detections" / f"{detector}_evaluation.pkl"
        paths["matches"] = config.output_dir / "matches" / f"{detector}.csv"
        paths["benchmark_metrics"] = config.output_dir / "benchmarks" / f"{detector}_metrics.json"
        paths["benchmark_text"] = config.output_dir / "benchmarks" / f"{detector}_official.txt"
    return paths


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        pickle.dump(value, output_file)


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_config(detector):
    return load_yaml_config(detector.cfg_file)


def run_inference(config, detector_name, logger):
    from pcdet.models import build_network, load_data_to_gpu

    detector = config.detector_specs[detector_name]
    loaded = load_model_config(detector)
    if config.target_class not in loaded.CLASS_NAMES:
        raise ValueError(
            f"Detector '{detector_name}' does not predict target class '{config.target_class}'; "
            f"its classes are: {', '.join(loaded.CLASS_NAMES)}"
        )
    runtime = NativeDatasetRuntime.create(
        config.dataset, loaded.DATA_CONFIG, loaded.CLASS_NAMES, config.split, config.data_root, logger
    )
    dataset = SelectedNativeDataset(runtime, config.frame_ids)
    model = build_network(loaded.MODEL, len(loaded.CLASS_NAMES), dataset)
    checkpoint = detector_checkpoint(detector)
    if checkpoint is None:
        raise FileNotFoundError(f"Checkpoint disappeared for detector '{detector_name}'")
    model.load_params_from_file(str(REPO_ROOT / checkpoint), logger=logger, to_cpu=True)
    model.cuda().eval()
    detections = []
    evaluation_annos = []
    with torch.no_grad():
        for index, sample in enumerate(dataset):
            frame_id = str(sample["frame_id"])
            logger.info("%s inference %d/%d: %s", detector_name, index + 1, len(dataset), frame_id)
            batch = dataset.collate_batch([sample])
            load_data_to_gpu(batch)
            predictions, _ = model(batch)
            evaluation_annos.extend(
                dataset.generate_prediction_dicts(batch, predictions, loaded.CLASS_NAMES)
            )
            prediction_values = zip(
                predictions[0]["pred_boxes"].detach().cpu().numpy(),
                predictions[0]["pred_scores"].detach().cpu().numpy(),
                predictions[0]["pred_labels"].detach().cpu().numpy(),
            )
            for detection_index, (box, score, label) in enumerate(prediction_values):
                class_name = loaded.CLASS_NAMES[int(label) - 1]
                full_box = [float(value) for value in box]
                detections.append(
                    {
                        "frame_id": frame_id,
                        "detection_index": detection_index,
                        "class_name": class_name,
                        "score": float(score),
                        "box_lidar": full_box[:7],
                        "box_lidar_full": full_box,
                    }
                )
    del model, dataset
    gc.collect()
    torch.cuda.empty_cache()
    detection_artifact = {
        "dataset": config.dataset.name,
        "split": config.split,
        "detector": detector_name,
        "target_class": config.target_class,
        "class_names": list(loaded.CLASS_NAMES),
        "frame_ids": list(config.frame_ids),
        "detections": detections,
    }
    evaluation_artifact = {
        "dataset": config.dataset.name,
        "split": config.split,
        "detector": detector_name,
        "class_names": list(loaded.CLASS_NAMES),
        "eval_metric": loaded.MODEL.POST_PROCESSING.get("EVAL_METRIC", ""),
        "frame_ids": list(config.frame_ids),
        "annos": evaluation_annos,
    }
    return detection_artifact, evaluation_artifact


def load_detection_artifact(path, config, detector_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing detections for {detector_name}: {path}. Run the infer stage first.")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    expected = (config.dataset.name, config.split, detector_name, config.target_class, list(config.frame_ids))
    actual = (
        artifact.get("dataset"), artifact.get("split"), artifact.get("detector"),
        artifact.get("target_class"), artifact.get("frame_ids"),
    )
    if actual != expected:
        raise ValueError(f"Cached detection metadata does not match this run: {path}")
    for record in artifact.get("detections", []):
        missing = [field for field in DETECTION_FIELDS if field not in record]
        if missing or len(record.get("box_lidar", [])) != 7:
            raise ValueError(f"Invalid detection record in {path}; missing={missing}")
    return artifact


def load_evaluation_artifact(path, config, detector_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing native evaluator predictions for {detector_name}: {path}. "
            "Re-run the infer stage with --force."
        )
    with path.open("rb") as input_file:
        artifact = pickle.load(input_file)
    expected = (config.dataset.name, config.split, detector_name, list(config.frame_ids))
    actual = (
        artifact.get("dataset"), artifact.get("split"), artifact.get("detector"),
        artifact.get("frame_ids"),
    )
    if actual != expected:
        raise ValueError(f"Cached evaluator prediction metadata does not match this run: {path}")
    if len(artifact.get("annos", ())) != len(config.frame_ids):
        raise ValueError(f"Evaluator predictions do not contain one annotation per frame: {path}")
    if not artifact.get("class_names"):
        raise ValueError(f"Evaluator prediction artifact has no class list: {path}")
    return artifact


def group_detections(records, target_class):
    grouped = defaultdict(list)
    for record in records:
        if record["class_name"] == target_class:
            grouped[str(record["frame_id"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["detection_index"]))
    return grouped


def to_jsonable(value):
    """Convert evaluator outputs containing NumPy/Pandas values into JSON data."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return {
            str(index): {str(key): to_jsonable(item) for key, item in row.items()}
            for index, row in value.to_dict(orient="index").items()
        }
    if isinstance(value, pd.Series):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def flatten_benchmark_metrics(value, prefix=""):
    """Flatten heterogeneous native-evaluator dictionaries to scalar table cells."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    value = to_jsonable(value)
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_benchmark_metrics(item, child))
        return flattened
    if isinstance(value, list):
        if len(value) == 1:
            return flatten_benchmark_metrics(value[0], prefix)
        if all(isinstance(item, (int, float, bool)) or item is None for item in value):
            return {prefix: value}
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(flatten_benchmark_metrics(item, f"{prefix}/{index}"))
        return flattened
    return {prefix: value}


def run_official_benchmark(config, detector_name, logger):
    """Run the dataset's native evaluator and preserve its unmodified output."""
    protocol = BENCHMARK_PROTOCOLS[config.dataset.name]
    base = {
        "run_signature": config.run_signature,
        "dataset": config.dataset.name,
        "split": config.split,
        "detector": detector_name,
        "protocol": protocol.name,
        "primary_metric": protocol.primary_metric,
        "reference_title": protocol.reference_title,
        "reference_url": protocol.reference_url,
        "protocol_note": protocol.note,
        "paper_comparable": config.split in protocol.paper_comparable_splits,
    }
    if not protocol.local_evaluation:
        return {
            **base, "status": "unsupported", "metrics": {},
            "message": protocol.note or "No local official evaluator is available.",
        }, ""
    if config.split == "test":
        return {
            **base, "status": "submission_required", "metrics": {},
            "message": "The test split has no local ground truth; use the dataset evaluation server.",
        }, ""

    paths = artifact_paths(config, detector_name)
    predictions = load_evaluation_artifact(
        paths["evaluation_predictions"], config, detector_name
    )
    loaded = load_model_config(config.detector_specs[detector_name])
    runtime = NativeDatasetRuntime.create(
        config.dataset, loaded.DATA_CONFIG, predictions["class_names"],
        config.split, config.data_root, logger,
    )
    if tuple(config.frame_ids) != runtime.frame_ids:
        return {
            **base, "status": "incomplete_split", "paper_comparable": False, "metrics": {},
            "message": (
                f"Official evaluation requires the complete {config.split} split "
                f"({len(runtime.frame_ids)} frames); this run contains {len(config.frame_ids)}."
            ),
        }, ""

    output_path = paths["benchmark_metrics"].parent / f"{detector_name}_native"
    output_path.mkdir(parents=True, exist_ok=True)
    result_text, result_dict = runtime.native_dataset.evaluation(
        predictions["annos"], predictions["class_names"],
        eval_metric=predictions.get("eval_metric", ""), output_path=output_path,
    )
    raw_metrics = to_jsonable(result_dict)
    # The nuScenes devkit writes richer per-class details than OpenPCDet returns.
    metrics_summary = output_path / "metrics_summary.json"
    if metrics_summary.exists():
        raw_metrics = json.loads(metrics_summary.read_text(encoding="utf-8"))
    flattened = flatten_benchmark_metrics(raw_metrics)
    return {
        **base,
        "status": "ok",
        "class_names": list(predictions["class_names"]),
        "num_frames": len(config.frame_ids),
        "metrics": flattened,
        "raw_metrics": raw_metrics,
    }, result_text.to_string() if isinstance(result_text, pd.DataFrame) else str(result_text or "")


def metric_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def kitti_paper_rows(artifacts):
    pattern = re.compile(
        r"^(?P<class>.+)_(?P<view>3d|bev|image|aos)/"
        r"(?P<difficulty>easy|moderate|hard)_(?P<recall>R11|R40)$"
    )
    grouped = {}
    for artifact in artifacts:
        for key, value in artifact.get("metrics", {}).items():
            match = pattern.match(key)
            if not match:
                continue
            group_key = (
                artifact["detector"], match.group("class"), match.group("view"),
                match.group("recall"),
            )
            row = grouped.setdefault(group_key, {
                "Detector": artifact["detector"], "Class": match.group("class"),
                "Metric": f"{match.group('view').upper()} AP_{match.group('recall')}",
            })
            row[match.group("difficulty").title()] = metric_number(value)
    return list(grouped.values())


def once_paper_rows(artifacts):
    pattern = re.compile(r"^AP_(?P<class>[^/]+)/(?P<distance>.+)$")
    rows = []
    for artifact in artifacts:
        grouped = {}
        for key, value in artifact.get("metrics", {}).items():
            match = pattern.match(key)
            if not match:
                continue
            row = grouped.setdefault(match.group("class"), {
                "Detector": artifact["detector"], "Class": match.group("class")
            })
            row[match.group("distance")] = metric_number(value)
        rows.extend(grouped.values())
    return rows


def paper_metric_cell(dataset_name, key, value):
    number = metric_number(value)
    if number is None:
        return value
    percentage = (
        (dataset_name == "nuscenes" and key in ("nd_score", "mean_ap", "NDS", "mAP"))
        or dataset_name == "lyft"
        or (dataset_name == "waymo" and key.rsplit("/", 1)[-1] in ("AP", "APH"))
        or (dataset_name == "argo2" and key.rsplit("/", 1)[-1] in ("AP", "CDS"))
    )
    return number * 100.0 if percentage and abs(number) <= 1.0 else number


def paper_metric_label(dataset_name, key):
    aliases = {
        "nd_score": "NDS (%)", "NDS": "NDS (%)",
        "mean_ap": "mAP (%)", "mAP": "mAP (%)",
        "tp_errors/trans_err": "mATE", "trans_err": "mATE",
        "tp_errors/scale_err": "mASE", "scale_err": "mASE",
        "tp_errors/orient_err": "mAOE", "orient_err": "mAOE",
        "tp_errors/vel_err": "mAVE", "vel_err": "mAVE",
        "tp_errors/attr_err": "mAAE", "attr_err": "mAAE",
    }
    label = aliases.get(key, key)
    if dataset_name in ("lyft", "waymo", "argo2") and key.rsplit("/", 1)[-1] in ("AP", "APH", "CDS"):
        label += " (%)"
    return label


def generic_paper_rows(dataset_name, artifacts, preferred_metrics=()):
    rows = []
    for artifact in artifacts:
        row = {"Detector": artifact["detector"], "Status": artifact["status"]}
        metrics = artifact.get("metrics", {})
        keys = [key for key in preferred_metrics if key in metrics]
        if not keys:
            keys = [key for key, value in metrics.items() if metric_number(value) is not None]
        for key in keys:
            row[paper_metric_label(dataset_name, key)] = paper_metric_cell(
                dataset_name, key, metrics[key]
            )
        if artifact["status"] != "ok":
            row["Message"] = artifact.get("message", "")
        rows.append(row)
    return rows


def paper_table_rows(dataset_name, artifacts):
    successful = [artifact for artifact in artifacts if artifact.get("status") == "ok"]
    if dataset_name in ("kitti", "custom") and successful:
        return kitti_paper_rows(successful)
    if dataset_name == "once" and successful:
        return once_paper_rows(successful)
    preferred = {
        "nuscenes": ("nd_score", "mean_ap", "tp_errors/trans_err", "tp_errors/scale_err",
                      "tp_errors/orient_err", "tp_errors/vel_err", "tp_errors/attr_err",
                      "NDS", "mAP", "trans_err", "scale_err", "orient_err", "vel_err", "attr_err"),
        "lyft": tuple(
            dict.fromkeys(
                ["mAP"]
                + [name for artifact in artifacts for name in artifact.get("class_names", [])]
            )
        ),
        "argo2": ("AVERAGE_METRICS/AP", "AVERAGE_METRICS/CDS", "mAP", "CDS"),
    }.get(dataset_name, ())
    return generic_paper_rows(dataset_name, artifacts, preferred)


def markdown_table(rows):
    if not rows:
        return "_No metrics were produced._\n"
    columns = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)

    def cell(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_benchmark_tables(config, artifacts):
    paths = artifact_paths(config)
    rows = paper_table_rows(config.dataset.name, artifacts)
    paths["benchmark_table"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths["benchmark_table"], index=False)

    protocol = BENCHMARK_PROTOCOLS[config.dataset.name]
    comparable = all(artifact.get("paper_comparable", False) for artifact in artifacts)
    heading = [
        f"# {config.dataset.name.upper()} {config.split} benchmark",
        "",
        f"Protocol: {protocol.name}",
        "",
        f"Primary reference: [{protocol.reference_title}]({protocol.reference_url})",
        "",
        f"Paper-comparable split: {'yes' if comparable else 'no'}",
        "",
    ]
    if protocol.note:
        heading.extend([f"Note: {protocol.note}", ""])
    paths["benchmark_markdown"].write_text(
        "\n".join(heading) + markdown_table(rows), encoding="utf-8"
    )

    long_rows = []
    for artifact in artifacts:
        if not artifact.get("metrics"):
            long_rows.append({
                "dataset": config.dataset.name, "split": config.split,
                "detector": artifact["detector"], "status": artifact["status"],
                "metric": "", "value": "", "paper_comparable": artifact["paper_comparable"],
            })
        for metric, value in artifact.get("metrics", {}).items():
            long_rows.append({
                "dataset": config.dataset.name, "split": config.split,
                "detector": artifact["detector"], "status": artifact["status"],
                "metric": metric, "value": value,
                "paper_comparable": artifact["paper_comparable"],
            })
    pd.DataFrame(long_rows).to_csv(paths["benchmark_long"], index=False)
    return paths["benchmark_table"], paths["benchmark_markdown"], paths["benchmark_long"]


def compute_iou_matrix(gt_boxes, prediction_boxes):
    from pcdet.ops.iou3d_nms import iou3d_nms_utils

    if not len(gt_boxes) or not len(prediction_boxes):
        return np.zeros((len(gt_boxes), len(prediction_boxes)), dtype=np.float32)
    if not torch.cuda.is_available():
        raise RuntimeError("The match stage requires CUDA for OpenPCDet 3D IoU")
    gt = torch.as_tensor(gt_boxes, dtype=torch.float32, device="cuda")
    predictions = torch.as_tensor(prediction_boxes, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        return iou3d_nms_utils.boxes_iou3d_gpu(gt, predictions).detach().cpu().numpy()


def greedy_matches(iou_matrix, min_iou):
    candidates = [
        (float(iou_matrix[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(iou_matrix.shape[0])
        for pred_index in range(iou_matrix.shape[1])
        if float(iou_matrix[gt_index, pred_index]) >= min_iou
    ]
    matches, used_gt, used_predictions = {}, set(), set()
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_predictions:
            continue
        matches[gt_index] = (pred_index, iou)
        used_gt.add(gt_index)
        used_predictions.add(pred_index)
    return matches


def build_match_rows(config, detection_artifact, runtime):
    detections_by_frame = group_detections(detection_artifact["detections"], config.target_class)
    rows = []
    for frame_id in config.frame_ids:
        ground_truth = runtime.ground_truth(frame_id, config.target_class)
        predictions = detections_by_frame.get(frame_id, [])
        gt_boxes = np.asarray([row["box_lidar"] for row in ground_truth], dtype=np.float32)
        prediction_boxes = np.asarray([row["box_lidar"] for row in predictions], dtype=np.float32)
        iou_matrix = compute_iou_matrix(gt_boxes, prediction_boxes)
        matches = greedy_matches(iou_matrix, config.min_iou)
        for gt_index, gt in enumerate(ground_truth):
            box = gt["box_lidar"]
            best_index = int(np.argmax(iou_matrix[gt_index])) if iou_matrix.shape[1] else None
            best_iou = float(iou_matrix[gt_index, best_index]) if best_index is not None else 0.0
            best_prediction = predictions[best_index] if best_index is not None else None
            matched = gt_index in matches
            prediction = predictions[matches[gt_index][0]] if matched else None
            rows.append(
                {
                    "run_signature": config.run_signature,
                    "dataset": config.dataset.name,
                    "split": config.split,
                    "frame_id": frame_id,
                    "gt_index": gt["gt_index"],
                    "gt_class": gt["class_name"],
                    "difficulty": gt.get("difficulty", ""),
                    "occlusion": gt.get("occlusion", ""),
                    "truncation": gt.get("truncation", ""),
                    "gt_x": float(box[0]), "gt_y": float(box[1]), "gt_z": float(box[2]),
                    "gt_dx": float(box[3]), "gt_dy": float(box[4]), "gt_dz": float(box[5]),
                    "gt_heading": float(box[6]),
                    "matched": "TRUE" if matched else "FALSE",
                    "matched_detection_index": prediction["detection_index"] if prediction else "",
                    "score": prediction["score"] if prediction else "",
                    "iou_3d": matches[gt_index][1] if matched else 0.0,
                    "best_iou_3d": best_iou,
                    "best_detection_index": best_prediction["detection_index"] if best_prediction else "",
                    "best_score": best_prediction["score"] if best_prediction else "",
                }
            )
    return rows


def points_in_box(points_xyz, box):
    centered = points_xyz - box[:3]
    cosine, sine = math.cos(float(box[6])), math.sin(float(box[6]))
    rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    local = centered @ rotation
    half_dimensions = np.asarray(box[3:6], dtype=np.float32) / 2.0
    return points_xyz[np.all(np.abs(local) <= half_dimensions + 1e-6, axis=1)]


def build_metric_rows(config, runtime):
    rows = []
    for frame_index, frame_id in enumerate(config.frame_ids):
        print(f"metrics {frame_index + 1}/{len(config.frame_ids)}: {frame_id}")
        points = runtime.points(frame_id)
        ground_truth = runtime.ground_truth(frame_id, config.target_class)
        for gt in ground_truth:
            object_points = points_in_box(points[:, :3], gt["box_lidar"])
            context = MetricContext(gt, object_points)
            row = {
                "run_signature": config.run_signature,
                "dataset": config.dataset.name,
                "split": config.split,
                "frame_id": frame_id,
                "gt_index": gt["gt_index"],
                "gt_class": gt["class_name"],
            }
            row.update({name: METRIC_REGISTRY[name].compute(context) for name in config.metrics})
            rows.append(row)
    return rows


def load_table(path, required_columns, expected_signature=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    table = pd.read_csv(path, dtype={"frame_id": str})
    missing = [column for column in required_columns if column not in table]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    if table.duplicated(["frame_id", "gt_index"]).any():
        raise ValueError(f"{path} contains duplicate (frame_id, gt_index) keys")
    if expected_signature is not None:
        signatures = set(table["run_signature"].astype(str)) if "run_signature" in table else set()
        if signatures != {expected_signature}:
            raise ValueError(f"{path} belongs to a different pipeline run")
    return table


def build_report(config):
    metric_path = artifact_paths(config)["metrics"]
    metrics = load_table(
        metric_path,
        ["run_signature", "frame_id", "gt_index", *config.metrics],
        config.run_signature,
    )
    metric_keys = set(zip(metrics.frame_id, metrics.gt_index))
    joined, summaries = {}, []
    for detector_name in config.detectors:
        match_path = artifact_paths(config, detector_name)["matches"]
        matches = load_table(
            match_path,
            ["run_signature", "frame_id", "gt_index", "matched", "iou_3d", "score"],
            config.run_signature,
        )
        match_keys = set(zip(matches.frame_id, matches.gt_index))
        if match_keys != metric_keys:
            raise ValueError(f"GT keys differ between metrics and {detector_name} matches")
        matches["matched"] = matches["matched"].astype(str).str.upper().eq("TRUE")
        rows = matches.merge(metrics, on=["frame_id", "gt_index"], validate="one_to_one")
        joined[detector_name] = rows
        matched_rows = rows[rows.matched]
        summaries.append(
            {
                "run_signature": config.run_signature,
                "dataset": config.dataset.name,
                "split": config.split,
                "detector": detector_name,
                "num_frames": len(config.frame_ids),
                "num_gt": len(rows),
                "num_matched": int(rows.matched.sum()),
                "num_missed": int((~rows.matched).sum()),
                "match_rate": float(rows.matched.mean()) if len(rows) else 0.0,
                "mean_matched_iou": float(matched_rows.iou_3d.mean()) if len(matched_rows) else float("nan"),
                "mean_matched_score": float(pd.to_numeric(matched_rows.score, errors="coerce").mean()) if len(matched_rows) else float("nan"),
            }
        )
    summary = pd.DataFrame(summaries)
    paths = artifact_paths(config)
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(paths["summary"], index=False)
    plot_report(config, joined, summary, paths["plot"])
    return paths["summary"], paths["plot"]


def plot_report(config, joined, summary, output_path):
    plot_metrics = [METRIC_REGISTRY[name] for name in config.metrics if METRIC_REGISTRY[name].bin_edges]
    if not plot_metrics:
        fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        axis.bar(summary.detector, summary.match_rate, color="#356b8c")
        axis.set_ylabel("Match rate")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{config.dataset.name.upper()} {config.split}: detector match rate")
    else:
        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(7 * len(plot_metrics), 4.8), squeeze=False, constrained_layout=True)
        for axis, metric in zip(axes[0], plot_metrics):
            for detector_name, rows in joined.items():
                buckets = pd.cut(
                    pd.to_numeric(rows[metric.name], errors="coerce"),
                    bins=metric.bin_edges,
                    labels=metric.bin_labels,
                    right=False,
                    include_lowest=True,
                )
                rates = rows.assign(bucket=buckets).groupby("bucket", observed=False).matched.mean()
                axis.plot(metric.bin_labels, rates.reindex(metric.bin_labels), marker="o", label=detector_name)
            axis.set_xlabel(metric.label)
            axis.set_ylabel("Match rate")
            axis.set_ylim(0.0, 1.0)
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=25)
            axis.legend(frameon=False)
        fig.suptitle(f"{config.dataset.name.upper()} {config.split}: geometry and detector outcomes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_options():
    print("Datasets: " + ", ".join(DATASET_REGISTRY))
    for dataset_name, dataset in DATASET_REGISTRY.items():
        print(f"Splits ({dataset_name}): " + ", ".join(dataset.splits))
    for dataset_name, detectors in DETECTOR_REGISTRY.items():
        print(f"Detectors ({dataset_name}): " + (", ".join(detectors) or "none registered"))
    print("Metrics: " + ", ".join(METRIC_REGISTRY))
    print("Stages: " + ", ".join(STAGE_ORDER))


def print_plan(config):
    print(f"Dataset: {config.dataset.name} ({config.split}, {len(config.frame_ids)} frames)")
    print("Detectors: " + (", ".join(config.detectors) or "(not required for selected stages)"))
    print("Stages: " + ", ".join(stage for stage in STAGE_ORDER if stage in config.stages))
    print("Metrics: " + ", ".join(config.metrics))
    print(f"Run signature: {config.run_signature}")
    print(f"Output: {config.output_dir}")


def run_pipeline(config):
    print_plan(config)
    if config.dry_run:
        return
    logger = common_utils.create_logger()
    analysis_runtime = None

    def get_analysis_runtime():
        nonlocal analysis_runtime
        if analysis_runtime is None:
            dataset_config = load_yaml_config(config.dataset.config_file)
            analysis_runtime = NativeDatasetRuntime.create(
                config.dataset, dataset_config, config.dataset.class_names,
                config.split, config.data_root, logger,
            )
        return analysis_runtime

    for stage in STAGE_ORDER:
        if stage not in config.stages:
            continue
        if stage == "infer":
            for detector_name in config.detectors:
                paths = artifact_paths(config, detector_name)
                path = paths["detections"]
                if path.exists() and paths["evaluation_predictions"].exists() and not config.force:
                    try:
                        load_detection_artifact(path, config, detector_name)
                        load_evaluation_artifact(
                            paths["evaluation_predictions"], config, detector_name
                        )
                    except ValueError as error:
                        print(f"REBUILD infer {detector_name}: {error}")
                    else:
                        print(
                            f"SKIP infer {detector_name}: {path} and "
                            f"{paths['evaluation_predictions']}"
                        )
                        continue
                detections, evaluator_predictions = run_inference(config, detector_name, logger)
                save_json(path, detections)
                save_pickle(paths["evaluation_predictions"], evaluator_predictions)
                print(f"WROTE {path}")
                print(f"WROTE {paths['evaluation_predictions']}")
        elif stage == "benchmark":
            benchmark_artifacts = []
            for detector_name in config.detectors:
                paths = artifact_paths(config, detector_name)
                artifact = None
                if paths["benchmark_metrics"].exists() and not config.force:
                    try:
                        cached = json.loads(paths["benchmark_metrics"].read_text(encoding="utf-8"))
                        expected = (
                            config.run_signature, config.dataset.name, config.split, detector_name
                        )
                        actual = (
                            cached.get("run_signature"), cached.get("dataset"),
                            cached.get("split"), cached.get("detector"),
                        )
                        if actual != expected:
                            raise ValueError("benchmark metadata does not match this run")
                        artifact = cached
                    except (ValueError, json.JSONDecodeError) as error:
                        print(f"REBUILD benchmark {detector_name}: {error}")
                    else:
                        print(f"SKIP benchmark {detector_name}: {paths['benchmark_metrics']}")
                if artifact is None:
                    try:
                        artifact, result_text = run_official_benchmark(
                            config, detector_name, logger
                        )
                    except (ImportError, ModuleNotFoundError) as error:
                        protocol = BENCHMARK_PROTOCOLS[config.dataset.name]
                        artifact = {
                            "run_signature": config.run_signature,
                            "dataset": config.dataset.name,
                            "split": config.split,
                            "detector": detector_name,
                            "protocol": protocol.name,
                            "primary_metric": protocol.primary_metric,
                            "reference_title": protocol.reference_title,
                            "reference_url": protocol.reference_url,
                            "protocol_note": protocol.note,
                            "paper_comparable": False,
                            "status": "missing_dependency",
                            "metrics": {},
                            "message": str(error),
                        }
                        result_text = ""
                    save_json(paths["benchmark_metrics"], artifact)
                    paths["benchmark_text"].write_text(result_text, encoding="utf-8")
                    print(f"WROTE {paths['benchmark_metrics']}")
                    print(f"WROTE {paths['benchmark_text']}")
                benchmark_artifacts.append(artifact)
            for path in write_benchmark_tables(config, benchmark_artifacts):
                print(f"WROTE {path}")
        elif stage == "match":
            for detector_name in config.detectors:
                paths = artifact_paths(config, detector_name)
                if paths["matches"].exists() and not config.force:
                    try:
                        load_table(
                            paths["matches"],
                            ["run_signature", "frame_id", "gt_index", "matched"],
                            config.run_signature,
                        )
                    except ValueError as error:
                        print(f"REBUILD match {detector_name}: {error}")
                    else:
                        print(f"SKIP match {detector_name}: {paths['matches']}")
                        continue
                detections = load_detection_artifact(paths["detections"], config, detector_name)
                save_csv(paths["matches"], build_match_rows(config, detections, get_analysis_runtime()), MATCH_FIELDS)
                print(f"WROTE {paths['matches']}")
        elif stage == "metrics":
            path = artifact_paths(config)["metrics"]
            if path.exists() and not config.force:
                try:
                    load_table(
                        path,
                        ["run_signature", "frame_id", "gt_index", *config.metrics],
                        config.run_signature,
                    )
                except ValueError as error:
                    print(f"REBUILD metrics: {error}")
                else:
                    print(f"SKIP metrics: {path}")
                    continue
            fields = ("run_signature", "dataset", "split", "frame_id", "gt_index", "gt_class", *config.metrics)
            save_csv(path, build_metric_rows(config, get_analysis_runtime()), fields)
            print(f"WROTE {path}")
        else:
            summary, plot = artifact_paths(config)["summary"], artifact_paths(config)["plot"]
            if summary.exists() and plot.exists() and not config.force:
                try:
                    cached = pd.read_csv(summary, dtype={"run_signature": str})
                    valid = (
                        set(cached.get("run_signature", [])) == {config.run_signature}
                        and set(cached.get("detector", [])) == set(config.detectors)
                    )
                    if not valid:
                        raise ValueError("summary metadata does not match this run")
                except (ValueError, pd.errors.ParserError) as error:
                    print(f"REBUILD report: {error}")
                else:
                    print(f"SKIP report: {summary} and {plot}")
                    continue
            summary, plot = build_report(config)
            print(f"WROTE {summary}")
            print(f"WROTE {plot}")


def main(argv=None):
    args = parse_args(argv)
    if args.list_options:
        print_options()
        return 0
    try:
        run_pipeline(resolve_run_config(args))
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
