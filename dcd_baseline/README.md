# DCD Baseline

A repeatable geometric baseline for the thesis: the **Density-aware Chamfer Distance
(DCD)** of Wu et al. (NeurIPS 2021) computed per ground-truth object between the
observed LiDAR returns and the object's ideal LiDAR-facing box surface. DCD is the
geometric-similarity baseline flagged in thesis §2.2.1; it competes with range, point
count, point density, and the thesis occlusion metric when explaining detection
difficulty. Everything reuses the `gilroy_lidar_replication` evaluation core, so
DCD-bin curves are directly comparable with the existing occlusion-bin curves.

## Method (one paragraph)

For each labelled object: take the LiDAR returns inside its GT 3D box (**observed**
set), and sample the box's LiDAR-facing faces at 12x12 per face axis (**ideal** set;
back faces are excluded by construction, so self-occlusion is never penalised).
DCD is computed with the official formula (n_lambda=1, non_reg=False) at
**alpha=1.0 on raw metre coordinates**: the official code uses alpha=1000 on
unit-normalized completion shapes, and the operating-point requirement
`alpha * typical_nearest_distance^2 ~ O(1)` maps to alpha=1 in metres for
driving-scale objects (verified by an alpha sweep; alpha=1000 on raw metres
saturates the exponential and destroys dynamic range). Plain Chamfer Distance
(cd_p sqrt-form, cd_t squared-form) is recorded alongside as the cheaper
geometric baseline. Higher DCD = coverage holes / sparser observation.

## Files

| File | Purpose |
|---|---|
| `dcd_metric.py` | Pure-PyTorch port of the official `calc_dcd` (no compiled kernels needed) |
| `infer_natural.py` | Run one detector over natural frames, writing the research-pipeline prediction JSON |
| `compute_dcd.py` | instances.csv -> per-object DCD/CD table (`dcd_instances.csv`) |
| `evaluate_dcd.py` | Bin objects by DCD (equal population, >= 55/bin) and evaluate per bin (thesis tables) |
| `plot_dcd.py` | Binning-free detection-probability curves + covariate-adjusted effects; writes the per-object outcome table |
| `test_dcd_metric.py` | Metric property tests incl. a hand-computed verification case |

## Quickstart (from the repo root, openpcdet env)

```bash
# 1. per-object DCD for the natural KITTI pedestrian benchmark (~2 min)
conda run -n openpcdet python dcd_baseline/compute_dcd.py \
  --instances runs/gilroy_lidar_replication/natural/val/instances.csv

# 2. detector predictions (once per detector; keeps Pedestrian predictions)
conda run -n openpcdet python dcd_baseline/infer_natural.py \
  --detector pointpillar --model-config tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint checkpoints/pointpillar.pth --expect-class Pedestrian

# 3. DCD-binned evaluation
conda run -n openpcdet python dcd_baseline/evaluate_dcd.py \
  --instances dcd_baseline/results/natural/dcd_instances.csv \
  --predictions dcd_baseline/results/detections/pointpillar_val.json \
  --detector pointpillar

# 4. binning-free curve figures (also writes results/analysis/per_object_outcomes_*.csv)
conda run -n openpcdet python dcd_baseline/plot_dcd.py \
  --prediction pointpillar=dcd_baseline/results/detections/pointpillar_val.json \
  --prediction second=dcd_baseline/results/detections/second_val.json \
  --prediction pointrcnn=dcd_baseline/results/detections/pointrcnn_val.json \
  --prediction pv_rcnn=dcd_baseline/results/detections/pv_rcnn_val.json
```

Ready detectors (same four as the Gilroy study): `pointpillar`, `second`,
`pointrcnn`, `pv_rcnn` (configs in `tools/cfgs/kitti_models/`, checkpoints in
`checkpoints/`).

## Outputs

- `results/natural/dcd_instances.csv` - one row per object: source columns plus
  `dcd`, `cd_p`, `cd_t`, `n_observed`, `n_ideal`.
- `results/natural/protocol.json` - exact metric definition and parameters.
- `results/detections/<detector>_val.json` - research-pipeline prediction artifact.
- `results/evaluation/<detector>_dcd_bin_metrics.csv` - per-bin metrics
  (instances, map_3d_50_95, ap_3d_50, tp_percent, fp_count, fp_per_gt, fn_percent).
- `results/evaluation/<detector>_binning.json` - bin edges, labels, counts, rules.
- `results/plots/fig_dcd_detection_probability.png` - P(detected) vs DCD per
  detector: cubic-logistic smooth with a 95% bootstrap band; axes, matching
  protocol, operating point, and empirical direction are captioned on the figure.
- `results/plots/fig_dcd_adjusted_effect.png` - the same curves after adjusting
  for ground range and LiDAR point count (solid) overlaid on unadjusted (dashed):
  the test of whether DCD carries detection-relevant information beyond density.
- `results/analysis/per_object_outcomes_<detector>.csv` - per-object table
  (frame_id, dcd, ground_range_m, num_points_in_gt, detected 0/1) for the
  thesis regression analysis.
- `results/analysis/dcd_effect_summary.csv` - detection-rate change from DCD
  P10 to P95 per detector, unadjusted and adjusted, with correlations.

## Porting to other datasets (Mixed Signals etc.)

The pipeline is dataset-agnostic at three seams:

1. **Instances CSV**: any CSV with `frame_id`, `x, y, z, dx, dy, dz, heading`
   (oriented GT box in the LiDAR frame) and optionally `label_index`/`class_name`.
   `compute_dcd.py --instances <file> --kitti-root <root>` only needs
   `<root>/training/velodyne/<frame_id>.bin`-style scans; adapt `load_points` for
   other layouts (one function).
2. **Predictions JSON**: `{"frame_id", "class_name", "score", "box_lidar"}` records;
   produced for other datasets by the same OpenPCDet inference pattern.
3. **Bins and plots**: unchanged; evaluation and plotting only read the two file
   schemas above.

Normalisation by the ideal surface keeps DCD comparable across different LiDARs and
object scales.

## Design decisions and limits

- **Baseline finding (important for the thesis)**: on the natural KITTI pedestrian
  benchmark the published DCD formulation (variable observed-set size vs a fixed
  ideal-surface reference) behaves as a *return-density proxy*: DCD rises with
  point count (r = +0.63) and falls with range (r = -0.71), so detection
  probability RISES with DCD for every detector. After adjusting for range and
  point count a reduced but non-zero effect remains (+7 to +29 pp from DCD P10
  to P95 depending on detector). The published metric therefore does not act as
  an occlusion metric in this regime - exactly the gap the thesis visibility
  metric addresses. Cite this as an empirical result of the baseline, not a bug;
  the earlier binned figures that suggested the opposite direction were invalid
  (a bin-assignment bug, fixed in `evaluate_dcd.py`).
- **Box surface, not a mesh**: KITTI has no object meshes; the box proxy matches
  thesis §5.8 and is documented as a limitation.
- **Facing faces only**: DCD measures external observability; self-occlusion is
  excluded by construction rather than estimated.
- **Operating point**: `--alpha` (default 1.0) and `--normalize` (default off) exist
  for sensitivity analysis (thesis §5.6). An alpha sweep shows DCD's ranking of a
  contiguous gap vs equal-count uniform thinning crosses near alpha~10; the
  reported baseline therefore states its operating point explicitly rather than
  inheriting one tuned for a different task.
- **DCD responds to sparsity as well as occlusion** (it is density-aware by design);
  range and point count remain available as separate columns for covariate analysis,
  which is exactly the comparison the thesis validation (§5.6) requires.
- **Equal-population bins** (not the nominal 5-95% bands): DCD's distribution is
  skewed, so bins are quantile-based, merged to the >= 55-instance Gilroy minimum.
- The official DCD implementation ships compiled CUDA kernels; this port replaces
  them with `torch.cdist` and is verified against a hand-computed case plus
  structural property tests (`python dcd_baseline/test_dcd_metric.py`).

## Checks

```bash
conda run -n openpcdet python dcd_baseline/test_dcd_metric.py
```
