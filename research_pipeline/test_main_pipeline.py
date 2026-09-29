import importlib
import logging
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from easydict import EasyDict

from research_pipeline import main_pipeline as pipeline


class FakeNativeDataset:
    def __init__(self, spec, info):
        setattr(self, spec.info_attribute, [info])
        self.dataset_cfg = EasyDict(INFO_WITH_FAKELIDAR=False, MAX_SWEEPS=1)

    def get_lidar(self, *args):
        return np.asarray([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)

    def get_lidar_with_sweeps(self, index, max_sweeps):
        return self.get_lidar(index, max_sweeps)

    def _get_pose(self, info):
        return {"pose": True}

    def _get_annotations(self, info, pose):
        return np.asarray([[1, 2, 3, 4, 2, 2, 0.1]], dtype=np.float32), np.asarray(["Car"]), 0.0

    def _get_lidar_points(self, info, pose):
        return self.get_lidar(info)


class DatasetCompatibilityTest(unittest.TestCase):
    @staticmethod
    def write_pickle(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output_file:
            pickle.dump(value, output_file)

    def assert_native_runtime(self, dataset_name, root, expected_frame_id):
        spec = pipeline.DATASET_REGISTRY[dataset_name]
        config = pipeline.load_yaml_config(spec.config_file)
        runtime = pipeline.NativeDatasetRuntime.create(
            spec, config, spec.class_names, "val", root, logging.getLogger("compatibility-test")
        )
        self.assertEqual(runtime.frame_ids, (expected_frame_id,))
        self.assertEqual(len(runtime.ground_truth(expected_frame_id, spec.target_class)), 1)
        self.assertGreaterEqual(runtime.points(expected_frame_id).shape[1], 3)
        selected = pipeline.SelectedNativeDataset(runtime, (expected_frame_id,))
        self.assertEqual(selected[0]["frame_id"], expected_frame_id)

    def test_registry_covers_every_openpcdet_dataset_config(self):
        config_names = {
            path.stem[:-len("_dataset")]
            for path in (pipeline.REPO_ROOT / "tools/cfgs/dataset_configs").glob("*_dataset.yaml")
            if path.name != "waymo_dataset_multiframe.yaml"
        }
        self.assertEqual(config_names, set(pipeline.DATASET_REGISTRY))
        for spec in pipeline.DATASET_REGISTRY.values():
            loaded = pipeline.load_yaml_config(spec.config_file)
            self.assertTrue(loaded.DATASET)
            self.assertIn(spec.target_class, spec.class_names)

    def test_every_bundled_model_config_is_registered_and_parseable(self):
        for dataset_name, registry in pipeline.DETECTOR_REGISTRY.items():
            model_dir = pipeline.REPO_ROOT / f"tools/cfgs/{dataset_name}_models"
            bundled = {path.resolve() for path in model_dir.glob("*.yaml")}
            registered = {(pipeline.REPO_ROOT / spec.cfg_file).resolve() for spec in registry.values()}
            self.assertEqual(bundled, registered, dataset_name)
            for detector in registry.values():
                loaded = pipeline.load_model_config(detector)
                self.assertIn("MODEL", loaded)
                self.assertIn("DATA_CONFIG", loaded)

    def test_all_adapters_normalize_annotations_and_points(self):
        box = np.asarray([[1, 2, 3, 4, 2, 2, 0.1]], dtype=np.float32)
        fixtures = {
            "kitti": {
                "point_cloud": {"lidar_idx": "000001"},
                "annos": {
                    "name": np.asarray(["Car", "DontCare"]),
                    "gt_boxes_lidar": box,
                    "index": np.asarray([0, -1]),
                    "difficulty": np.asarray([1, -1]),
                },
            },
            "nuscenes": {"lidar_path": "samples/LIDAR_TOP/token.bin", "gt_names": np.asarray(["car"]), "gt_boxes": box},
            "waymo": {
                "frame_id": "segment_001",
                "point_cloud": {"lidar_sequence": "segment", "sample_idx": 1},
                "annos": {"name": np.asarray(["Vehicle"]), "gt_boxes_lidar": box},
            },
            "pandaset": {"sequence": "001", "frame_idx": 2},
            "lyft": {"lidar_path": "lidar/token.bin", "gt_names": np.asarray(["car"]), "gt_boxes": box},
            "once": {
                "sequence_id": "seq", "frame_id": "100",
                "annos": {"name": np.asarray(["Car"]), "boxes_3d": box},
            },
            "custom": {
                "point_cloud": {"lidar_idx": "sample"},
                "annos": {"name": np.asarray(["Vehicle"]), "gt_boxes_lidar": box},
            },
            "argo2": {
                "point_cloud": {"velodyne_path": "training/velodyne/100.bin"},
                "annos": {
                    "name": np.asarray(["Regular_vehicle"]),
                    "location": box[:, :3], "dimensions": box[:, 3:6], "rotation_y": box[:, 6],
                },
            },
        }
        for dataset_name, info in fixtures.items():
            with self.subTest(dataset=dataset_name):
                spec = pipeline.DATASET_REGISTRY[dataset_name]
                runtime = pipeline.NativeDatasetRuntime(spec, FakeNativeDataset(spec, info))
                frame_id = runtime.frame_ids[0]
                records = runtime.ground_truth(frame_id, spec.target_class)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["class_name"], spec.target_class)
                self.assertEqual(records[0]["box_lidar"].shape, (7,))
                self.assertEqual(runtime.points(frame_id).shape, (1, 4))

    def test_external_detector_registration_supports_datasets_without_presets(self):
        config = pipeline.REPO_ROOT / "tools/cfgs/custom_models/second.yaml"
        checkpoint = pipeline.REPO_ROOT / "checkpoints/second.pth"
        registry = pipeline.configured_detector_registry(
            "pandaset", [f"experiment={config}"], [f"experiment={checkpoint}"]
        )
        self.assertEqual(registry["experiment"].cfg_file, config)
        self.assertEqual(pipeline.detector_checkpoint(registry["experiment"]), checkpoint)

    def test_generated_test_splits_are_routed_to_native_layouts(self):
        expected_test_datasets = {"kitti", "nuscenes", "waymo", "pandaset", "lyft", "once"}
        for dataset_name in expected_test_datasets:
            with self.subTest(dataset=dataset_name):
                spec = pipeline.DATASET_REGISTRY[dataset_name]
                self.assertIn("test", spec.splits)
                configured = pipeline.configure_dataset_split(
                    pipeline.load_yaml_config(spec.config_file), spec, "test", Path("/dataset")
                )
                self.assertEqual(configured.DATA_SPLIT.test, "test")
        nuscenes = pipeline.configure_dataset_split(
            pipeline.load_yaml_config(pipeline.DATASET_REGISTRY["nuscenes"].config_file),
            pipeline.DATASET_REGISTRY["nuscenes"], "test", Path("/dataset"),
        )
        lyft = pipeline.configure_dataset_split(
            pipeline.load_yaml_config(pipeline.DATASET_REGISTRY["lyft"].config_file),
            pipeline.DATASET_REGISTRY["lyft"], "test", Path("/dataset"),
        )
        self.assertEqual(nuscenes.VERSION, "v1.0-test")
        self.assertEqual(nuscenes.INFO_PATH.test, ["nuscenes_infos_10sweeps_test.pkl"])
        self.assertEqual(lyft.VERSION, "test")
        self.assertEqual(lyft.INFO_PATH.test, ["lyft_infos_test.pkl"])

    def test_native_constructors_for_file_backed_dataset_families(self):
        box7 = np.asarray([[5.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.1]], dtype=np.float32)
        points4 = np.asarray([[5.0, 0.0, 0.0, 0.5], [5.2, 0.1, 0.0, 0.4]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)

            custom_root = base / "custom"
            (custom_root / "ImageSets").mkdir(parents=True)
            (custom_root / "ImageSets/val.txt").write_text("sample\n", encoding="utf-8")
            (custom_root / "points").mkdir()
            np.save(custom_root / "points/sample.npy", points4)
            self.write_pickle(custom_root / "custom_infos_val.pkl", [{
                "point_cloud": {"lidar_idx": "sample"},
                "annos": {"name": np.asarray(["Vehicle"]), "gt_boxes_lidar": box7},
            }])
            self.assert_native_runtime("custom", custom_root, "sample")

            nuscenes_root = base / "nuscenes"
            nuscenes_version = nuscenes_root / "v1.0-trainval"
            (nuscenes_version / "samples/LIDAR_TOP").mkdir(parents=True)
            point_path = nuscenes_version / "samples/LIDAR_TOP/token.bin"
            np.column_stack([points4, np.zeros(len(points4), dtype=np.float32)]).astype(np.float32).tofile(point_path)
            sweeps = [{"lidar_path": "samples/LIDAR_TOP/token.bin", "transform_matrix": None, "time_lag": 0.1}] * 9
            self.write_pickle(nuscenes_version / "nuscenes_infos_10sweeps_val.pkl", [{
                "lidar_path": "samples/LIDAR_TOP/token.bin", "token": "token", "sweeps": sweeps,
                "gt_names": np.asarray(["car"]),
                "gt_boxes": np.column_stack([box7, np.zeros((1, 2), dtype=np.float32)]),
                "num_lidar_pts": np.asarray([2]),
            }])
            self.assert_native_runtime("nuscenes", nuscenes_root, "token")

            lyft_root = base / "lyft"
            lyft_version = lyft_root / "trainval"
            (lyft_version / "lidar").mkdir(parents=True)
            lyft_points = np.column_stack([points4, np.zeros(len(points4), dtype=np.float32)]).astype(np.float32)
            lyft_points.tofile(lyft_version / "lidar/token.bin")
            lyft_sweeps = [{"lidar_path": "lidar/token.bin", "transform_matrix": None, "time_lag": 0.1}] * 4
            self.write_pickle(lyft_version / "lyft_infos_val.pkl", [{
                "lidar_path": "lidar/token.bin", "token": "token", "sweeps": lyft_sweeps,
                "gt_names": np.asarray(["car"]), "gt_boxes": box7,
            }])
            self.assert_native_runtime("lyft", lyft_root, "token")

            waymo_root = base / "waymo"
            (waymo_root / "ImageSets").mkdir(parents=True)
            (waymo_root / "ImageSets/val.txt").write_text("segment-test.tfrecord\n", encoding="utf-8")
            waymo_sequence = waymo_root / "waymo_processed_data_v0_5_0/segment-test"
            waymo_sequence.mkdir(parents=True)
            waymo_points = np.column_stack([
                points4, np.zeros((len(points4), 2), dtype=np.float32)
            ]).astype(np.float32)
            np.save(waymo_sequence / "0000.npy", waymo_points)
            self.write_pickle(waymo_sequence / "segment-test.pkl", [{
                "frame_id": "segment-test_000", "pose": np.eye(4, dtype=np.float32),
                "point_cloud": {"lidar_sequence": "segment-test", "sample_idx": 0},
                "annos": {"name": np.asarray(["Vehicle"]), "gt_boxes_lidar": box7,
                          "num_points_in_gt": np.asarray([2])},
            }])
            self.assert_native_runtime("waymo", waymo_root, "segment-test_000")

            once_root = base / "once"
            (once_root / "ImageSets").mkdir(parents=True)
            (once_root / "ImageSets/val.txt").write_text("seq\n", encoding="utf-8")
            once_lidar = once_root / "data/seq/lidar_roof"
            once_lidar.mkdir(parents=True)
            points4.tofile(once_lidar / "100.bin")
            self.write_pickle(once_root / "once_infos_val.pkl", [{
                "sequence_id": "seq", "frame_id": "100",
                "annos": {"name": np.asarray(["Car"]), "boxes_3d": box7,
                          "num_points_in_gt": np.asarray([2])},
            }])
            self.assert_native_runtime("once", once_root, "seq/100")

    def test_optional_sdk_dataset_constructors(self):
        box7 = np.asarray([[5.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.1]], dtype=np.float32)
        points4 = np.asarray([[5.0, 0.0, 0.0, 0.5], [5.2, 0.1, 0.0, 0.4]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)

            pandaset_root = base / "pandaset"
            self.write_pickle(pandaset_root / "pandaset_infos_val.pkl", [{
                "sequence": "001", "frame_idx": 2,
                "lidar_path": "unused-lidar.pkl.gz", "cuboids_path": "unused-cuboids.pkl.gz",
            }])
            pandaset_module = importlib.import_module("pcdet.datasets.pandaset.pandaset_dataset")
            fake_sdk = types.SimpleNamespace(DataSet=lambda path: object())
            with mock.patch.object(pandaset_module, "ps", fake_sdk, create=True):
                spec = pipeline.DATASET_REGISTRY["pandaset"]
                config = pipeline.load_yaml_config(spec.config_file)
                runtime = pipeline.NativeDatasetRuntime.create(
                    spec, config, spec.class_names, "val", pandaset_root,
                    logging.getLogger("compatibility-test"),
                )
                pose = {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "heading": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                }
                runtime.native_dataset._get_pose = lambda info: pose
                runtime.native_dataset._get_lidar_points = lambda info, current_pose: points4
                runtime.native_dataset._get_annotations = lambda info, current_pose: (
                    box7, np.asarray(["Car"]), 0.0
                )
                frame_id = "001/02"
                self.assertEqual(len(runtime.ground_truth(frame_id, "Car")), 1)
                self.assertEqual(runtime.points(frame_id).shape, (2, 4))
                self.assertEqual(pipeline.SelectedNativeDataset(runtime, (frame_id,))[0]["frame_id"], frame_id)

            argo2_root = base / "argo2"
            (argo2_root / "ImageSets").mkdir(parents=True)
            (argo2_root / "ImageSets/val.txt").write_text("100\n", encoding="utf-8")
            argo2_lidar = argo2_root / "training/velodyne"
            argo2_lidar.mkdir(parents=True)
            points4.tofile(argo2_lidar / "100.bin")
            self.write_pickle(argo2_root / "argo2_infos_val.pkl", [{
                "point_cloud": {"velodyne_path": "training/velodyne/100.bin"},
                "annos": {
                    "name": np.asarray(["Regular_vehicle"]),
                    "location": box7[:, :3], "dimensions": box7[:, 3:6], "rotation_y": box7[:, 6],
                },
            }])
            av2 = types.ModuleType("av2")
            av2_utils = types.ModuleType("av2.utils")
            av2_io = types.ModuleType("av2.utils.io")
            av2_io.read_feather = lambda path: None
            argo2_so3 = types.ModuleType("pcdet.datasets.argo2.argo2_utils.so3")
            argo2_so3.yaw_to_quat = lambda value: value
            argo2_so3.quat_to_yaw = lambda value: value
            stub_modules = {
                "av2": av2, "av2.utils": av2_utils, "av2.utils.io": av2_io,
                "pcdet.datasets.argo2.argo2_utils.so3": argo2_so3,
            }
            with mock.patch.dict(sys.modules, stub_modules):
                argo2_module = importlib.import_module("pcdet.datasets.argo2.argo2_dataset")
                from pcdet.datasets import __all__ as dataset_classes

                with mock.patch.dict(dataset_classes, {"Argo2Dataset": argo2_module.Argo2Dataset}):
                    self.assert_native_runtime("argo2", argo2_root, "100")


class BenchmarkTableTest(unittest.TestCase):
    def test_numba_driver_selection_is_wsl_only_and_respects_override(self):
        environment = {}
        selected = pipeline.configure_numba_cuda_driver(
            environment, platform_release="6.1.21-microsoft-standard-WSL2",
            path_exists=lambda path: str(path) == "/usr/lib/wsl/lib/libcuda.so",
        )
        self.assertEqual(str(selected), "/usr/lib/wsl/lib/libcuda.so")
        self.assertEqual(environment["NUMBA_CUDA_DRIVER"], "/usr/lib/wsl/lib/libcuda.so")

        existing = {"NUMBA_CUDA_DRIVER": "/custom/libcuda.so"}
        self.assertIsNone(pipeline.configure_numba_cuda_driver(
            existing, platform_release="microsoft WSL", path_exists=lambda path: True
        ))
        self.assertEqual(existing["NUMBA_CUDA_DRIVER"], "/custom/libcuda.so")

        linux = {}
        self.assertIsNone(pipeline.configure_numba_cuda_driver(
            linux, platform_release="6.8.0-generic", path_exists=lambda path: True
        ))
        self.assertNotIn("NUMBA_CUDA_DRIVER", linux)

    def test_protocol_registry_covers_every_dataset(self):
        self.assertEqual(set(pipeline.DATASET_REGISTRY), set(pipeline.BENCHMARK_PROTOCOLS))
        self.assertFalse(pipeline.BENCHMARK_PROTOCOLS["pandaset"].local_evaluation)
        self.assertIn("benchmark", pipeline.STAGE_ORDER)

    def test_flatten_metrics_handles_numpy_arrays_and_json_rows(self):
        metrics = pipeline.flatten_benchmark_metrics({
            "mAP": np.float32(0.5),
            "Vehicle": '{"AP":0.75,"ATE":0.2}',
            "nested": {"values": np.asarray([0.1])},
        })
        self.assertAlmostEqual(metrics["mAP"], 0.5)
        self.assertEqual(metrics["Vehicle/AP"], 0.75)
        self.assertEqual(metrics["Vehicle/ATE"], 0.2)
        self.assertEqual(metrics["nested/values"], 0.1)

    def test_kitti_table_matches_paper_layout(self):
        artifacts = [{
            "detector": "pvrcnn", "status": "ok",
            "metrics": {
                "Car_3d/easy_R40": 92.1,
                "Car_3d/moderate_R40": 84.5,
                "Car_3d/hard_R40": 82.0,
                "Car_bev/easy_R40": 95.0,
                "Car_bev/moderate_R40": 91.0,
                "Car_bev/hard_R40": 88.0,
            },
        }]
        rows = pipeline.paper_table_rows("kitti", artifacts)
        self.assertEqual(len(rows), 2)
        row = next(value for value in rows if value["Metric"] == "3D AP_R40")
        self.assertEqual(row["Detector"], "pvrcnn")
        self.assertEqual(row["Class"], "Car")
        self.assertEqual(row["Moderate"], 84.5)
        rendered = pipeline.markdown_table(rows)
        self.assertIn("| Detector | Class | Metric | Easy | Moderate | Hard |", rendered)

    def test_nuscenes_table_scales_scores_but_not_errors(self):
        artifacts = [{
            "detector": "centerpoint", "status": "ok",
            "metrics": {
                "nd_score": 0.673,
                "mean_ap": 0.592,
                "tp_errors/trans_err": 0.31,
                "tp_errors/scale_err": 0.25,
            },
        }]
        row = pipeline.paper_table_rows("nuscenes", artifacts)[0]
        self.assertAlmostEqual(row["NDS (%)"], 67.3)
        self.assertAlmostEqual(row["mAP (%)"], 59.2)
        self.assertAlmostEqual(row["mATE"], 0.31)

    def test_once_table_uses_distance_columns(self):
        artifacts = [{
            "detector": "second", "status": "ok",
            "metrics": {
                "AP_Vehicle/overall": 71.19,
                "AP_Vehicle/0-30m": 84.0,
                "AP_mean/overall": 51.89,
            },
        }]
        rows = pipeline.paper_table_rows("once", artifacts)
        vehicle = next(row for row in rows if row["Class"] == "Vehicle")
        self.assertEqual(vehicle["overall"], 71.19)
        self.assertEqual(vehicle["0-30m"], 84.0)


if __name__ == "__main__":
    unittest.main()
