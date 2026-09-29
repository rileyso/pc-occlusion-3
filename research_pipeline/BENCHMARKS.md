# Reproducing classical 3D detection metrics

The pipeline has two different evaluation layers:

- `benchmark` calls the dataset-native OpenPCDet evaluator and writes the
  classical benchmark metrics used by papers.
- `match`, `metrics`, and `report` produce project-specific object matching,
  geometry, and diagnostic analysis. Those outputs are useful research
  diagnostics, but they are not replacements for official AP, NDS, mAPH, or
  CDS.

## Run an official validation benchmark

Use the complete validation split. A shortened run such as `--max-frames 10`
is useful for smoke testing but is deliberately labelled non-comparable.

```bash
conda run -n openpcdet python research_pipeline/main_pipeline.py \
  --dataset kitti \
  --detectors pointpillar,pointrcnn,pvrcnn,second,voxel_rcnn_car \
  --split val \
  --stages infer,benchmark
```

The benchmark stage writes:

```text
runs/<study>/<run_id>/benchmarks/paper_table.md
runs/<study>/<run_id>/benchmarks/paper_table.csv
runs/<study>/<run_id>/benchmarks/official_metrics_long.csv
runs/<study>/<run_id>/benchmarks/<detector>_metrics.json
runs/<study>/<run_id>/benchmarks/<detector>_official.txt
```

Inference also stores the native prediction dictionaries used by OpenPCDet's
standard `tools/test.py` evaluator. This avoids losing velocity fields,
dataset metadata, or non-target classes.

## Dataset protocols and verification references

| Dataset | Local paper-style metrics | Primary verification reference | Important constraint |
| --- | --- | --- | --- |
| KITTI | 3D, BEV, image AP at R11 and R40; easy/moderate/hard | [KITTI paper](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf), [official object benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php) | Local `val` results are reproducible. Official `test` results require server submission. Older OpenPCDet model-zoo numbers use R11; current KITTI ranking uses R40. |
| nuScenes | NDS, mAP, mATE, mASE, mAOE, mAVE, mAAE, plus devkit details | [nuScenes paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf), [detection task](https://www.nuscenes.org/object-detection) | Match the devkit/config version (`detection_cvpr_2019`), dataset version, modality, sweeps, and external-data policy. |
| Waymo | Per-class 3D mAP/mAPH at LEVEL_1 and LEVEL_2 | [Waymo Open Dataset paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf), [official evaluator repository](https://github.com/waymo-research/waymo-open-dataset) | Match dataset version, full validation set, training fraction, frame count, and LEVEL definition. The optional Waymo/TensorFlow evaluator packages are required. |
| Lyft Level 5 | Per-class AP and mAP over configured IoU thresholds | [Lyft dataset devkit](https://github.com/lyft/nuscenes-devkit) | Treat the challenge/devkit definition as authoritative. There is no single detector-paper table shared by all Lyft research. |
| ONCE | Orientation-aware class AP and mAP overall and at 0–30 m, 30–50 m, 50 m+ | [ONCE paper](https://arxiv.org/abs/2106.11037), [ONCE benchmark](https://github.com/PointsCoder/ONCE_Benchmark) | Uses 50 recall positions and class-specific IoU thresholds. Compare the same labelled training split and superclass setting. |
| Argoverse 2 | mAP, CDS, ATE, ASE, and AOE by category and average | [Argoverse 2 paper](https://openreview.net/pdf?id=vKQGe36av4k), [official task guide](https://argoverse.github.io/user-guide/tasks/3d_object_detection.html) | Match the 26-category taxonomy, 150 m range, ROI filtering, AV2 API version, and validation split. |
| PandaSet | None in the bundled adapter | [PandaSet paper](https://arxiv.org/abs/2112.12610) | OpenPCDet explicitly implements no official PandaSet detection evaluator. A study must first declare and version a surrogate protocol; it must not be called official PandaSet mAP. |
| Custom | KITTI-style surrogate AP | [KITTI paper](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | There is no source-paper score. Record the class mapping, IoU thresholds, split construction, and difficulty rules. |

## What score should match?

Use this order when checking a run:

1. The same checkpoint evaluated by OpenPCDet's standard `tools/test.py` path.
2. The matching row in this repository's model zoo, with the same config,
   split, dataset version, and metric generation (for example R11 versus R40).
3. The detector paper only when its data split, training schedule, input
   sweeps/frames, augmentations, ensemble/test-time augmentation, and
   checkpoint provenance are also identical.

An original detector paper is usually the right source for understanding the
model, but not automatically the right numeric oracle for a downloaded
OpenPCDet checkpoint. The repository model-zoo row is generally the closer
numeric comparison.

## Reproducibility metadata to retain

For every table, retain the git commit, config file, checkpoint checksum,
dataset release, split/frame list, evaluator package version, CUDA/operator
versions, number of sweeps or frames, score/NMS settings, and whether test-time
augmentation or external data was used. A table without those fields can look
like a paper table while still being impossible to reproduce.
