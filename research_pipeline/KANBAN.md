# Research Pipeline Kanban

Last updated: 2026-08-30

## Done

- [x] Establish `main_pipeline.py` as the standalone pipeline entry point.
- [x] Add dataset, detector, stage, and metric registries.
- [x] Support resumable inference, matching, geometry metrics, and reporting.
- [x] Keep new artifacts isolated under `runs/<study>/<run_id>`. 
- [x] Validate every stage with a one-frame KITTI PV-RCNN smoke run.
- [x] Support every OpenPCDet dataset adapter through native loaders: KITTI,
  nuScenes, Waymo, PandaSet, Lyft, ONCE, Custom, and Argoverse 2.
- [x] Discover every bundled dataset-specific model config and support external
  model-config/checkpoint pairs for datasets without bundled presets.
- [x] Add compatibility tests for registry coverage, config parsing, annotation
  normalization, point loading, native dataset construction/preprocessing,
  generated test-split routing, and external detector registration.
- [x] Add a lightweight Matplotlib BEV module for points, GT boxes, predictions,
  score filtering, pipeline JSON input, and headless image export.

## Doing

- [ ] Compare standalone summary counts with the established research artifacts.

## Next

- [ ] Register a visibility or occlusion metric after its definition is fixed.
- [ ] Expand reports only when a research question requires another figure or table.

## Blocked

- [ ] Non-KITTI end-to-end smoke runs require the corresponding datasets and
  checkpoints to be installed locally.
- [ ] Inference and 3D IoU matching require working OpenPCDet CUDA operators.
