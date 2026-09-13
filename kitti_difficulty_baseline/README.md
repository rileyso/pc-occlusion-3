# KITTI Difficulty Baseline

The last of the thesis's coarse-annotation baselines: the **KITTI benchmark's own
Easy / Moderate / Hard difficulty labels** (Geiger et al., CVPR 2012 devkit —
occlusion annotation + image truncation + 2D bbox height) evaluated as a
per-object difficulty metric, alongside the benchmark's **official AP_R40
protocol** as the external anchor. Sibling to `waymo_baseline` (Waymo
point-count LEVEL rule) and `dcd_baseline` (DCD geometric metric); everything
on the natural pedestrian benchmark reuses the Gilroy evaluation core
unchanged, so the numbers are directly comparable with the occlusion-bin, DCD-bin
and Waymo-level tables.

It answers two questions:

1. **Thesis-protocol arm** — on the 2,280 natural validation pedestrians, how
   well do the native KITTI difficulty labels stratify detection difficulty for
   the four pedestrian-capable detectors, and what do the labels conflate?
   (Same matching protocol and per-group metrics as every other thesis baseline.)
2. **Official arm** — what do the four checkpoints score under the benchmark's
   own protocol (3D AP_R40 per difficulty, full 3,769-frame val split, official
   ignore semantics)? This is the "published numbers" anchor; it also verifies
   the checkpoints against the recorded reference values.

## Key result (one paragraph)

_On the natural pedestrian benchmark (thesis protocol), the KITTI difficulty
bands stratify detection sharply and monotonically — Easy → Moderate → Hard
3D AP@0.50 falls (PointPillars 45.4 → 21.0 → 8.4, SECOND 45.4 → 19.9 → 6.2,
PointRCNN 30.5 → 11.8 → 3.5, PV-RCNN 55.8 → 21.4 → 8.7) — but the labels are a
conflated, lossy summary: the ordinal 3-level label predicts detection at AUC
0.64–0.68 while continuous log point count reaches 0.71–0.82 for every
detector, and the UnKnown band the benchmark discards (n = 102) is **not** the
hardest band — its members are chiefly image-truncated (mean truncation 0.41)
rather than occluded (mean ray-cast occlusion 19%, 475 LiDAR points on
average), and PointRCNN detects 49% of them vs only 15% of its Hard band.
Under the official AP_R40 protocol (full val), the checkpoints anchor to the
literature — PointPillars reproduces the recorded reference exactly on all
three classes (Car 87.76/78.39/75.19, Pedestrian 57.33/51.50/46.92, Cyclist
81.57/62.95/58.96); PV-RCNN scores Car 92.11/84.37/82.50 — and the official
Easy→Hard gradient looks far gentler than the exclusive bands above, because
the nested ignore semantics mean even the official Hard set is still mostly
Easy and Moderate objects._

## Files

| File | Purpose |
|---|---|
| `groupings.py` | Grouping registry — label column → ordered groups (exclusive + official cumulative views). The extension seam for future occlusion metrics. |
| `evaluate_kitti_difficulty.py` | Thesis-protocol arm: per-group metrics via the Gilroy core, per-object table, AUC/summary, protocol.json |
| `infer_val_fov.py` | Full-val inference with the benchmark's standard FOV crop (writes the shared predictions-JSON format) |
| `kitti_official_ap.py` | Official arm: devkit AP_R40 per difficulty from OpenPCDet's own eval code, per class |
| `visualise_bins.py` | Ego-POV BEV gallery: LiDAR scan + the bins' objects, one panel per difficulty bin (any registry grouping, any class) |
| `plot_kitti.py` | The report figures from the evaluation CSVs (CPU only) |
| `report_text.md` | Thesis-ready prose, method, analysis, limitations, figure captions |
| `results/protocol.json` / `protocol_official_ap.json` | Exact rule definitions, citations, conventions |
| `results/evaluation/kitti_difficulty_metrics.csv` | Per-group metrics: detector × grouping × view (exclusive/cumulative) × group |
| `results/evaluation/kitti_official_ap.csv` | Official AP_R40: detector × class × difficulty |
| `results/analysis/per_object_kitti_difficulty.csv` | Per-object table (labels + covariates + detected per detector) for the §5.6 regression |
| `results/analysis/kitti_difficulty_summary.csv` | Group covariate profiles (the conflation evidence) + per-detector AUCs |
| `results/plots/fig_kitti_difficulty_bars.png` | **Report main figure**: x = Easy/Moderate/Hard (+ UnKnown), grouped bars = 4 detectors, y = AP@0.50 and detection rate |
| `results/plots/fig_kitti_occlusion_bars.png` | Same layout over the raw KITTI occlusion annotation levels 0/1/2/3 |
| `results/plots/fig_kitti_official_ap_bars.png` | Official AP_R40 by difficulty, Car and Pedestrian panels |
| `results/plots/fig_difficulty_bev_gallery_car.png` | **BEV gallery**: what each difficulty bin looks like from the ego vehicle (aggregate scans, GT boxes + returns) |
| `results/plots/fig_difficulty_bev_gallery_pedestrian.png` | Same for pedestrians (500 seeded frames containing pedestrians) |

## Quickstart (from the repo root, openpcdet env)

```bash
PY=~/miniconda3/envs/openpcdet/bin/python

# 1. thesis-protocol arm on the natural pedestrian benchmark (~4 min, CUDA)
$PY research_workspace/kitti_difficulty_baseline/evaluate_kitti_difficulty.py \
  --prediction pointpillar=research_workspace/dcd_baseline/results/detections/pointpillar_val.json \
  --prediction second=research_workspace/dcd_baseline/results/detections/second_val.json \
  --prediction pointrcnn=research_workspace/dcd_baseline/results/detections/pointrcnn_val.json \
  --prediction pv_rcnn=research_workspace/dcd_baseline/results/detections/pv_rcnn_val.json

# 2. benchmark-faithful full-val detections (FOV-cropped), once per detector (~10-25 min each)
$PY research_workspace/kitti_difficulty_baseline/infer_val_fov.py \
  --detector pointpillar --model-config tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint checkpoints/pointpillar.pth   # repeat for second / pointrcnn / pv_rcnn

# 3. official arm: AP_R40 per difficulty from those detections (~2 min per detector)
LD_LIBRARY_PATH=/usr/lib/wsl/lib $PY research_workspace/kitti_difficulty_baseline/kitti_official_ap.py \
  --prediction pointpillar=research_workspace/kitti_difficulty_baseline/results/detections/pointpillar_val.json \
  --prediction second=research_workspace/kitti_difficulty_baseline/results/detections/second_val.json \
  --prediction pointrcnn=research_workspace/kitti_difficulty_baseline/results/detections/pointrcnn_val.json \
  --prediction pv_rcnn=research_workspace/kitti_difficulty_baseline/results/detections/pv_rcnn_val.json

# 4. figures from the CSVs (~1 min, CPU only)
$PY research_workspace/kitti_difficulty_baseline/plot_kitti.py
```

Note the `LD_LIBRARY_PATH=/usr/lib/wsl/lib` on the official arm: the numba CUDA
runtime inside the KITTI devkit eval picks up a stale `libcuda.so` from
`/usr/lib/x86_64-linux-gnu` in this WSL install and fails with
`CUDA_ERROR_NO_DEVICE` unless the WSL driver directory comes first (PyTorch is
unaffected, so only the devkit-eval steps need it).

## Design decisions and limits

- **Two arms, two prediction sets, stated openly.** The thesis-protocol arm
  evaluates the same pedestrian-preserving JSONs as the DCD/Gilroy/Waymo
  baselines (generated by `infer_natural.py`, which does not apply the camera-FOV
  point crop) so every per-group number stays cross-baseline comparable. The
  official arm regenerates predictions with the benchmark's own FOV crop
  (`infer_val_fov.py`) because out-of-image-FOV detections otherwise depress
  official AP by 5-16 points (verified against `tools/test.py`: PointPillars Car
  AP_R40 87.75/78.39/75.18 with the crop vs 71.4/65.8/63.2 without).
- **Exclusive vs cumulative groups.** Official KITTI difficulty AP is computed
  on nested ground-truth sets (for difficulty d, objects failing the d-criteria
  are ignored entirely), so the three published numbers are not three slices of
  one population. The bar figures therefore use **exclusive (disjoint) bands** —
  each object counted once, in its own band — while the cumulative
  (official-semantics) rows are in `kitti_difficulty_metrics.csv` and the
  official arm reproduces the published convention exactly.
- **Use the KITTI AP?** Both. The official AP_R40 is the externally comparable
  anchor (and what the literature sees); the thesis-protocol per-group metrics
  are what the cross-baseline comparison and the §5.6 regression consume.
  Rationale in `report_text.md`.
- **UnKnown is reported, not hidden.** Objects failing the Hard criteria are
  ignored by the official benchmark at every difficulty; this baseline keeps
  them as their own band (marked `*` in the figures) because they are
  demonstrably not the hardest objects — see the key result.
- **Score floor.** All prediction sets inherit each model config's
  post-processing (`SCORE_THRESH: 0.1` + NMS). AP integrals therefore start
  from score 0.1; identical for all four detectors and all baselines.

## Adding a new dataset or occlusion metric (the modularity contract)

- **New metric (thesis visibility `v_i`, nuScenes camera-visibility bins, a
  binned continuous score):** compute it per object into a column of the
  instances CSV, then either add a `Grouping` entry in `groupings.py`
  (order + display + which values the official protocol ignores) or build one
  ad hoc with `groupings.from_column(...)`. `evaluate_kitti_difficulty.py
  --grouping <name>` and `plot_kitti.py` pick it up unchanged. Continuous
  metrics can reuse `dcd_baseline/evaluate_dcd.py::assign_dcd_bins`
  (equal-population, ≥ 55/bin) to make the column.
- **New dataset (Mixed Signals etc.):** produce (a) an instances CSV —
  `frame_id, label_index, class_name, x/y/z/dx/dy/dz/heading` plus the label
  columns (see `gilroy_lidar_replication/build_benchmark.py` for the KITTI
  generator) — and (b) predictions JSONs in the shared format
  (`{frame_id, class_name, score, box_lidar}`, one file per detector, all
  classes kept). Then run steps 1 and 4 above with `--instances` pointing at
  the new CSV. Official-arm equivalents (Waymo AP/APH, nuScenes mAP) are
  dataset-specific by design: add one evaluator module per benchmark with the
  same output schema (`detector, class, difficulty, ap`).
- The matching protocol (Gilroy core), the portable predictions format and the
  grouping registry are deliberately dataset-agnostic; only label provenance
  and official protocols are per-dataset.
