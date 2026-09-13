# Classical Geometric Baselines — KITTI Val Pedestrians

**The baseline (headline figures): pedestrian AP@IoU 0.3 and AP@IoU 0.5 as a
function of ground range and point count, for the four pedestrian-capable
OpenPCDet detectors — PointPillars, SECOND, PointRCNN, PV-RCNN — as minimal
regular bar plots.**

- `results/plots/fig_ap_iou30_bars.png`
- `results/plots/fig_ap_iou50_bars.png`

AP is a set metric (it ranks predictions against a pool of objects), so it is
computed per equal-population bin of the covariate (target 10 bins, merged to
≥ 55 objects; n ≈ 228 per bin) rather than per object or on a continuous
sliding window. Within a bin, confidence-ranked predictions with score ≥ 0.1
are evaluated; predictions matching objects outside the bin are ignored
(workspace subset protocol). Exact numbers: `results/analysis/ap_bin_metrics.csv`.

Regenerate everything from the repo root with `~/miniconda3/envs/openpcdet/bin/python`:

```bash
python geometric_baselines/build_master_table.py   # covariates + detected flags (n = 2,280)
python geometric_baselines/ap_bars.py              # AP@0.3 / AP@0.5 bins + bar figures
```

## Headline numbers (AP@IoU 0.5, first → last bin)

| | PointPillars | SECOND | PointRCNN | PV-RCNN |
| --- | --- | --- | --- | --- |
| Ground range [2.9, 7.5) → [29, 72) m | 37.7 → 5.9 | 33.0 → 8.9 | 44.3 → 0.8 | 35.9 → 13.2 |
| Points in box [0, 30) → [467, 4495) | 1.0 → 43.0 | 1.1 → 38.4 | 0.2 → 48.2 | 2.4 → 42.9 |

- Detection quality degrades severely and near-monotonically with range; with
  the looser AP@IoU 0.3 the same shape holds at higher absolute values
  (e.g. PointPillars 51.4 → 27.7).
- AP rises steeply with point count; the small dip in the last count bin is
  the known top-decile effect (very close, image-truncated pedestrians).
- PointRCNN is the strongest at close range but by far the most fragile as
  range grows or points vanish.

## Supplementary analyses (kept for reference)

The master table and earlier detection-rate analyses also cover return
density (volumetric and facing-surface), KITTI coarse occlusion/difficulty
labels, a close-range occlusion slice, and fitted detection-probability
curves: see `results/analysis/` (`binned_detection_rate.csv`,
`curve_fits.csv`, `covariate_summary.csv`, `close_range_by_occlusion.csv`),
`plot_geometric.py` (curves), `plot_geometric_bars.py` (detection-rate bars),
and the corresponding `fig_*_detection*.png` / `fig_kitti_coarse_labels.png`
/ `fig_geometric_baselines_summary.png` figures. Key context from those:

- Occlusion matters even at ≤ 10 m (KITTI state 0 vs 2: e.g. PV-RCNN
  89.8% vs 53.7% detection) — range and density cannot capture this.
- Classical covariates are mutually entangled (r(range, points) = −0.61;
  r(points/m², points) = +0.93), so single-covariate views are marginal, not
  causal — the gap the thesis visibility metric targets.

## Covariate definitions & matching protocol

| Column | Definition |
| --- | --- |
| `ground_range_m` | ground-plane distance from sensor to box centre |
| `num_points_in_gt` | LiDAR returns inside the GT box |
| `point_density_m3` / `point_density_m2_facing` | points / box volume; points / m² of LiDAR-facing box surface (back faces excluded) |
| `kitti_occlusion` / `kitti_difficulty` | KITTI annotations (0 fully visible … 2 largely occluded; Easy/Moderate/Hard) |

Master table `results/master_per_object.csv`: one row per pedestrian
(n = 2,280; provenance in `results/master_protocol.json`), including per-
detector `detected_*` flags from the shared protocol: greedy
confidence-ranked matching at 3D IoU ≥ 0.50, score ≥ 0.1, predictions
matching other annotated pedestrians ignored.

## Conventions & limitations

- Equal-population bins (≥ 55 objects) everywhere; bars on regular,
  equally spaced category axes (mid-bin noise is visually over-weighted —
  the curve figures are the smoother analysis view).
- Subset AP within a bin is not identical to AP over the whole set (per-bin
  score calibration differs); compare shapes across bins, not absolute means.
- AP@IoU 0.3 is a loose geometric match; KITTI's official pedestrian AP uses
  0.5 — the two figures bound the effect of the matching strictness.
- Box surface proxies the true object surface (KITTI has no meshes).
