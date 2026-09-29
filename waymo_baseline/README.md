# Waymo Difficulty Baseline

A coarse-annotation baseline for the thesis: the **Waymo Open Dataset detection
difficulty rule** (Sun et al., CVPR 2020, §5.1) transplanted onto the natural
KITTI pedestrian benchmark. Waymo grades every 3D label by LiDAR observation
richness — labels with **≤ 5 LiDAR points** in the box are **LEVEL_2** (plus a
labeler-flagged-hard criterion that does not transfer to KITTI and is dropped),
the rest are **LEVEL_1**, zero-point labels are ignored, and LEVEL_2 metrics are
reported **cumulatively**. This baseline asks how *effective* that heuristic is:
does it stratify pedestrian detection difficulty, for four pedestrian-capable
detectors, and what does one point-count bit miss?

Everything reuses the DCD-baseline assets (per-object point counts are the DCD
pipeline's `n_observed`; predictions are the same pedestrian-preserving JSONs)
and the Gilroy evaluation core unchanged, so all numbers are directly comparable
with the occlusion-bin and DCD-bin tables.

## Key result (one paragraph)

On 2,279 evaluable KITTI validation pedestrians the literal ≤ 5-point threshold
is **sensor-relative to the point of degeneracy**: 0.9% of pedestrians fall in
LEVEL_2 (n = 20, below every stable-estimate minimum), the cumulative LEVEL_2
AP equals the LEVEL_1 AP within 0.5 pp, and the binary label predicts detection
at AUC 0.51 for all four detectors — a coin flip. Given a **density-matched
threshold** (≤ 26 points ≈ the KITTI pedestrian P10), the same rule stratifies
sharply: LEVEL_2 AP@0.50 drops to 0.2–3.2% vs 26.4–48.3% in LEVEL_1 and
detection rates by 42–52 pp. But one bit discards most of the signal: detection
probability is continuous and monotone in point count, the within-LEVEL_1
tercile spread is 32–64 pp, continuous log point count predicts detection at
AUC 0.71–0.82 vs 0.58–0.62 for the binary level, and LEVEL_2 membership is
range-confounded (r = +0.64 with range; median 36 m vs 13 m). Verdict: directionally
valid, coarse, not portable, and unable to separate occlusion from range — the
anchor the thesis visibility metric is measured against. Hypotheses, method and
report-ready prose: `report_text.md`.

## Files

| File | Purpose |
|---|---|
| `evaluate_waymo.py` | Assign levels (literal + sweep), evaluate per level with the Gilroy core, write the effectiveness summary and per-object table |
| `plot_waymo.py` | The two report figures from the evaluation CSVs (no GPU needed) |
| `report_text.md` | Hypotheses, design-rationale methodology, analysis, limitations, thesis linkage and figure captions for the thesis |
| `results/protocol.json` | Exact rule definition, citation, sweep rationale, matching protocol |
| `results/evaluation/waymo_level_metrics.csv` | Literal-rule metrics: LEVEL_1 / LEVEL_2 (exclusive) / LEVEL_2 cumulative per detector |
| `results/evaluation/waymo_threshold_sweep.csv` | Same metrics for thresholds 5 / 10 / 26 / 50 / 100 points |
| `results/analysis/waymo_effectiveness_summary.csv` | AUCs (binary level vs continuous log points vs range), level-range correlations, within-LEVEL_1 tercile rates |
| `results/analysis/per_object_waymo_levels.csv` | Per-object table (frame_id, point count, range, level assignments, detected per detector) for the §5.6 regression |
| `results/plots/fig_waymo_detector_summary.png` | **Recommended single figure for the report main text**: one clean panel — detection rate per detector, All / LEVEL_1 / LEVEL_2 at the density-matched rule |
| `results/plots/fig_waymo_level_performance.png` | Appendix figure: per-detector AP and detection rate, All/LEVEL_1/LEVEL_2, literal vs density-matched rule |
| `results/plots/fig_waymo_rule_effectiveness.png` | Appendix figure: continuous signal, threshold sweep, hidden within-LEVEL_1 gradient, range confound |

## Quickstart (from the repo root, openpcdet env)

```bash
# 1. level assignment + per-level evaluation + effectiveness summary (~5 min, CUDA)
~/miniconda3/envs/openpcdet/bin/python waymo_baseline/evaluate_waymo.py \
  --prediction pointpillar=dcd_baseline/results/detections/pointpillar_val.json \
  --prediction second=dcd_baseline/results/detections/second_val.json \
  --prediction pointrcnn=dcd_baseline/results/detections/pointrcnn_val.json \
  --prediction pv_rcnn=dcd_baseline/results/detections/pv_rcnn_val.json

# 2. figures from the CSVs (~1 min, CPU only)
~/miniconda3/envs/openpcdet/bin/python waymo_baseline/plot_waymo.py
```

Predictions are the DCD baseline's pedestrian-preserving validation JSONs; if
they are ever regenerated, rerun `dcd_baseline/infer_natural.py` per detector
first. The point-count column comes from
`dcd_baseline/results/natural/dcd_instances.csv` (`num_points_in_gt`), verified
identical to the DCD pipeline's `n_observed` on all 2,280 rows.

## Design decisions and limits

- **The rule transplanted, not the metric.** Waymo's AP/APH (heading-weighted,
  recall-normalised, class-specific IoU) is not reproduced; the level *heuristic*
  is, and groups are evaluated with the thesis's matching protocol so numbers
  are comparable with the Gilroy occlusion bins and DCD bins.
- **Labeler-flag criterion dropped.** Waymo's LEVEL_2 also includes
  labeler-flagged-hard labels; KITTI has no equivalent flag, so only the
  point-count criterion transfers (KITTI's own E/M/H labels play that role and
  remain available in `dcd_instances.csv` for a cross-benchmark comparison).
- **Cumulative convention respected.** Waymo reports LEVEL_2 cumulatively
  (LEVEL_1 included); exclusive per-level subsets are computed *in addition*
  because the difficulty-gap analysis needs like-for-like strata.
- **Literal rule reported as-is, flagged as unstable.** LEVEL_2 (literal) has
  n = 20 < the 55-instance Gilroy minimum; its metrics are anecdotal and the
  density-matched 26-point cut is the primary quantitative comparison. The sweep
  (10 / 26 / 50 / 100) shows the trade-off between stratification gap and group
  balance; 26 is the 10th percentile, chosen before evaluation.
- **Sensor-relative thresholds are the point, not a bug.** The degeneracy of the
  literal rule on KITTI (99.1% LEVEL_1) is a finding about the portability of
  benchmark difficulty annotations, consistent with H1 in `report_text.md`.
- **Zero-point objects ignored** per the Waymo convention (1 object).
- **Panel (a) caveat of the effectiveness figure**: below ~15 points there is little
  data (the 5th-percentile pedestrian has 16 points), so the steep left tail of the
  cubic-logistic curves is model extrapolation, not observation — cite the summary
  figure or the sweep (panel b) rather than the far left of panel (a).
- **Detector order**: figures use the canonical thesis order PointPillars, SECOND,
  PointRCNN, PV-RCNN (`DETECTOR_ORDER` in `plot_waymo.py`).
- **Point-count threshold semantics**: Waymo counts LiDAR points inside the GT
  box on its sensors; here the same count is taken from KITTI Velodyne returns
  inside the box (10 Hz HDL-64E scans, ~2× denser object sampling at pedestrian
  ranges than Waymo's top/mid sensors — the source of the transfer failure).
