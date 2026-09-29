# Gilroy LiDAR Replication

A KITTI/OpenPCDet adaptation of Gilroy et al., *The Impact of Partial Occlusion on Pedestrian Detectability* (DOI: `10.1016/j.birob.2023.100115`). Outputs are kept separately under `runs/gilroy_lidar_replication/`.

## Faithful parts of the protocol

- Target class: KITTI `Pedestrian`.
- Ten objective bins: 0–9%, 10–19%, …, 90–99% occluded.
- At least 55 target instances in every bin.
- Seven detector slots, with their architecture families recorded in `models.json`.
- 3D COCO-style mAP: mean AP at 3D IoU 0.50, 0.55, …, 0.95.
- TP%, FP count/rate, and FN% at each occlusion level.
- A head-visibility analogue and comparison with KITTI's coarse human annotation.

The paper calls the evaluation package “Voxel51”; the current product name is FiftyOne. FiftyOne is optional here because OpenPCDet supplies the 3D IoU operation and the small evaluator is reproducible without another dependency.

## Objective 3D adaptation

Gilroy et al. estimate visible/full 2D pedestrian body surface. KITTI has no pedestrian surface mesh, so this baseline uses the nearest objective geometric equivalent available in its labels:

> **3D occlusion percentage = area-weighted fraction of the target's LiDAR-facing 3D bounding-box surface for which a ray from the LiDAR intersects another annotated 3D box first.**

This value does **not** use distance, return count, density, detector output, or KITTI's coarse occlusion label. The head ablation uses the upper 20% of the LiDAR-facing box surface as an explicit proxy, not a claim that KITTI supplies head annotations.

Limit: unlabelled poles, vegetation, walls, and other scene surfaces cannot act as occluders in this natural baseline. A later controlled benchmark should add ray-consistent virtual occluders.

## Current benchmark

The full KITTI validation split produced 2,280 Pedestrian instances and already meets the paper's minimum in every objective bin:

| Occlusion | Instances |
|---|---:|
| 0–9% | 1,250 |
| 10–19% | 115 |
| 20–29% | 108 |
| 30–39% | 85 |
| 40–49% | 117 |
| 50–59% | 102 |
| 60–69% | 118 |
| 70–79% | 129 |
| 80–89% | 94 |
| 90–99% | 162 |

Build it with:

```bash
conda run -n openpcdet python gilroy_lidar_replication/build_benchmark.py \
  --split val \
  --output-dir runs/gilroy_lidar_replication/natural/val
```

Outputs:

- `natural/val/instances.csv`: one target Pedestrian per row, objective visibility and raw confounders;
- `natural/val/bin_summary.csv`: the ten-bin ≥55 check;
- `natural/val/protocol.json`: machine-readable definition and limitations.

## Evaluation

```bash
conda run -n openpcdet python gilroy_lidar_replication/evaluate.py \
  --instances runs/gilroy_lidar_replication/natural/val/instances.csv \
  --predictions path/to/all-class-val-predictions.json \
  --detector pointpillar \
  --output-dir runs/gilroy_lidar_replication/evaluation/natural_val
```

Prediction JSON is the OpenPCDet research-pipeline format with `frame_id`, `class_name`, `score`, and `box_lidar`. Predictions matching Pedestrians outside the active occlusion bin are ignored rather than counted as false positives.

The first PointPillars baseline is at `evaluation/natural_val/pointpillar_metrics.csv`.

## Controlled 820-incident design

`build_controlled.py` selects 82 clear Pedestrians from 82 distinct validation frames and reuses each at the ten bin midpoints (5%, 15%, …, 95%): exactly **820 structured incidents**. It also creates 820 matched random-drop controls. Each pair has the same source frame, target, distance, pose, and retained target-point count; only the spatial removal pattern differs. The manifest transforms original Velodyne files on demand, avoiding 1,640 duplicate point-cloud files.

```bash
conda run -n openpcdet python gilroy_lidar_replication/build_controlled.py
```

This writes `runs/gilroy_lidar_replication/controlled/samples.csv`. `infer_controlled.py` reconstructs each sample during model inference. Run all locally usable models, evaluate both arms, and produce Fig. 4/6-style structured-occlusion plots with:

```bash
conda run -n openpcdet python gilroy_lidar_replication/run_available.py
```

A safe two-sample model-loading check is available with `--max-samples 2`; incomplete smoke runs deliberately skip evaluation. All four usable models passed both smoke and complete 1,640-sample inference.

Current complete controlled outputs are under `controlled/evaluation/` and `controlled/plots/`. The structured-arm overall 3D mAP values are PointPillars 0.041, SECOND 0.023, PointRCNN 0.021, and PV-RCNN 0.047; equal-count random removal is higher for every model (0.067, 0.047, 0.033, and 0.076 respectively). This supports the planned hypothesis descriptively, but paired bootstrap confidence intervals are still required before calling the structured-minus-random differences statistically supported.

## Detector availability audit

Only PointPillars, SECOND, PointRCNN, and PV-RCNN currently have local multiclass KITTI configs/checkpoints. Part-A2 has no checkpoint; CenterPoint has no KITTI config/checkpoint; the local Voxel R-CNN checkpoint is Car-only. These three must be obtained or trained on the same KITTI training split before a seven-model comparison is valid. No substitute models will be silently used.

Existing older detection JSON files retain only Cars. New inference must preserve Pedestrian predictions as the research-pipeline PointPillars artifact does.

## Confounder isolation before claims

Natural KITTI results are descriptive until the controlled analysis is complete:

1. Report every natural curve within fixed ground-range strata and with distance-adjusted estimates.
2. The controlled manifest now holds distance, pose, and scene fixed by reusing every base Pedestrian at all ten levels.
3. Every structured sample now has a random point-drop control with exactly the same retained target-point count.
4. Remaining analysis work: paired frame/base-instance bootstrap confidence intervals and structured-minus-random effect plots.
5. Keep the natural benchmark separate from the controlled benchmark.

Planned paper-figure mapping:

- Fig. 4 → 3D mAP versus objective occlusion.
- Fig. 6 → TP%, FP per GT, and FN% versus objective occlusion.
- Fig. 8 → objective 3D bins versus KITTI's coarse occlusion labels, with distance/point-count distributions shown.
- Fig. 9 → all targets versus upper-box/head-proxy visible and occluded subsets.

Do not interpret the current PointPillars curve causally until the paired and density-matched controls above exist.

## Checks

```bash
conda run -n openpcdet python -m unittest discover \
  -s gilroy_lidar_replication -p 'test_*.py'
```
