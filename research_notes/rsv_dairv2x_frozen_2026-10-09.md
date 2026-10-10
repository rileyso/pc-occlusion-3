# RSV vs Definition-3 on DAIR-V2X-C: frozen analysis

_Frozen 2026-10-09. This records the state of the analysis at this date. Later work should add a new dated note rather than edit these numbers. Status: exploratory and post-hoc. Nothing here is a validated metric._

## 1. Summary

We compared Luo et al.'s Definition-3 solid-angle visibility (Def-3) with the candidate return-weighted solid-angle visibility (RSV) on DAIR-V2X-C, against two references: human `occluded_state` labels and ego-only PointPillars detection outcomes.

RSV splits each object's expected LiDAR ray footprint by what each ray's first return actually did. This makes it more than a weighted point return, but its two parts behave differently.

- **Observed visibility** `V_lo = Ω(H)/Ω_exp` is the share of expected rays that returned from the object: a point return normalised by geometry. It is the best predictor of this detector's misses. It adds +0.048 held-out AUC beyond range and log point count, and once it is known, no other RSV component, and not Def-3, adds anything.
- **Occlusion share** `O_ext = Ω(B)/Ω(H+B+P)` is the share of measured rays stopped in front of the object. It best matches human occlusion labels: AUC 0.836 overall and 0.963 within 30 m, against 0.675 and 0.904 for Def-3. It adds nothing to detection prediction beyond `V_lo`.
- **Def-3** agrees with human labels at close range but adds nothing to detection prediction beyond range and point count.

Interpretation: for this detector, occlusion matters through how much evidence it removes. Once the remaining evidence is known, its cause adds no predictive value. The analysis supports a two-part, LiDAR-only metric: observed visibility as the evidence term, and a decomposition of the unobserved part into labelled occlusion, unlabelled occlusion, pass-through and missing returns as the explanatory term.

## 2. Data, detector and metrics

**Human-label analysis**
- DAIR-V2X-C example: `/mnt/c/Datasets/dair-v2x-c-example`, 46 cooperative pairs.
- Vehicle side: 91 frames, 1,419 boxes after dropping degenerate labels (181 dropped) and boxes with no RSV value.
- Reference: per-side `occluded_state` (0 none, 1 partial, 2 heavy), treated as occluded when > 0. The labels are annotated from camera images and are coarse.

**Detection analysis**
- DAIR-V2X-C CoAlign overlay `data/coalign_full`, validation split, 1,789 frames.
- Detector: PointPillars `data/checkpoints/pointpillars-late-fusion/net_epoch_bestval_at15.pth` (sha256 prefix `db3eb9ce4ab71daa`), trained with CoAlign's late-fusion recipe. Run **ego vehicle LiDAR only**, through CoAlign `inference_no_fusion`. CoAlign HEAD `64d1987`; pose noise off; score threshold 0.20; NMS 0.15.
- Primary ground truth: the vehicle's own labels (`--gt vehicle`), 23,659 boxes, of which 5 have no ego returns.
- Secondary ground truth: cooperative labels (`--gt cooperative`), 27,851 boxes, of which 2,428 have no ego returns.
- Outcome: missed if no prediction matched at 3D IoU ≥ 0.5 (also reported at 0.7), with one-to-one, confidence-ordered matching. Every GT box is kept.

**Metrics.** All are computed from the ego LiDAR, with the sensor at the origin of its own frame.

| Symbol | Definition | Source |
|---|---|---|
| `V_def3` | Unblocked fraction of the box's solid angle, where blockers are other labelled boxes, decided per ray. 32×32 samples per face. | `dairv2x/visibility_frame.py` |
| `Ω_exp` | Solid angle of the ray-grid cells whose centre ray passes through the box | `dairv2x/rsv.py` |
| H / B / P / M | Per footprint cell, nearest return r against the box's entry and exit distances, with δ = 0.2 m: on the box / in front of it / beyond it / no return. B is split into B_lab and B_unlab by whether another labelled box lies in front. | `dairv2x/rsv.py` |
| `V_rsv` | Ω(H) / Ω(H+B+P) | `dairv2x/rsv.py` |
| `V_lo` | Ω(H) / Ω_exp (observed visibility) | `dairv2x/rsv.py` |
| `O_ext` | Ω(B_lab+B_unlab) / Ω(H+B+P) | `dairv2x/rsv.py` |
| `P_frac`, `M_frac` | Ω(P) / Ω(H+B+P), Ω(M) / Ω_exp | `dairv2x/rsv.py` |
| range, points | Distance to box centre; ego returns inside the box | |

Vehicle ray grid: Hesai Pandar40P nominal beam table (40 rings, −25° to +15°), each ring refined to the median elevation of its returns, with a 0.2° azimuth step. Solid-angle weighting is close to uniform near the horizon, so the weighting itself has little effect. What separates RSV from point counts is the expected-ray denominator and the ray classification.

## 3. Results

### 3.1 Human occlusion labels (DAIR example, vehicle side, n = 1,419; AUC, occluded = positive)

| Score | All | 0–30 m | 30–60 m | 60+ m |
|---|---|---|---|---|
| 1 − V_def3 | 0.675 | 0.904 | 0.608 | 0.639 |
| 1 − V_lo | 0.779 | 0.709 | 0.773 | 0.723 |
| 1 − V_rsv | 0.795 | 0.720 | 0.801 | 0.735 |
| **O_ext** | **0.836** | **0.963** | **0.832** | **0.740** |
| −log1p(points) | 0.590 | 0.504 | 0.490 | 0.402 |

V_rsv minus Def-3, with a 95% frame-bootstrap interval:

| Range | Difference [95% interval] |
|---|---|
| All | +0.120 [+0.091, +0.151] |
| 0–30 m | −0.185 [−0.277, −0.087] |
| 30–60 m | +0.192 [+0.155, +0.232] |
| 60+ m | +0.096 [+0.044, +0.142] |

### 3.2 Ego PointPillars misses (full val, vehicle GT, IoU 0.5; recall 0.731)

AUC for predicting a miss:

| Range | n | miss % | Def-3 | V_rsv | Distance | Points |
|---|---|---|---|---|---|---|
| All | 23,659 | 26.9 | 0.610 | 0.734 | 0.631 | 0.726 |
| 0–30 m | 7,665 | 18.3 | 0.634 | 0.657 | 0.584 | 0.613 |
| 30–60 m | 11,213 | 26.1 | 0.636 | 0.809 | 0.560 | 0.768 |
| 60+ m | 4,781 | 42.5 | 0.531 | 0.674 | 0.510 | 0.629 |

Single-score AUC, all ranges:

| 1 − V_lo | 1 − V_rsv | −log points | O_ext |
|---|---|---|---|
| **0.758** | 0.734 | 0.726 | 0.715 |

Held-out logistic models (5-fold, grouped by blocks of 20 frames):

| Added to range + log points (base 0.730) | AUC | Gain |
|---|---|---|
| + Def-3 | 0.728 | −0.002 |
| **+ V_lo** | **0.778** | **+0.048** |
| + V_rsv | 0.762 | +0.032 |
| + O_ext | 0.750 | +0.020 |
| + O_ext + P_frac + M_frac | 0.771 | +0.041 |

| Added to range + log points + V_lo (base 0.778) | AUC | Gain |
|---|---|---|
| + Def-3 | 0.778 | +0.000 |
| + V_rsv | 0.776 | −0.001 |
| + O_ext | 0.777 | −0.000 |
| + O_ext + P_frac + M_frac | 0.777 | −0.000 |

AUC within log-point quintiles (objects with comparable point counts):

| Points | n | miss % | Def-3 | V_lo | V_rsv | O_ext | P_frac |
|---|---|---|---|---|---|---|---|
| 0–23 | 4,877 | 57.8 | 0.572 | 0.764 | 0.754 | 0.699 | 0.515 |
| 23–57 | 4,600 | 32.7 | 0.549 | 0.706 | 0.686 | 0.647 | 0.469 |
| 57–124 | 4,746 | 15.7 | 0.572 | 0.724 | 0.715 | 0.702 | 0.382 |
| 124–323 | 4,715 | 11.7 | 0.572 | 0.709 | 0.689 | 0.706 | 0.345 |
| 323+ | 4,721 | 15.6 | 0.581 | 0.421 | 0.424 | 0.534 | 0.277 |

Recall by visibility band:

| V band | Def-3 n | Def-3 recall | RSV n | RSV recall |
|---|---|---|---|---|
| 0.00–0.25 | 2,281 | 0.493 | 4,437 | 0.361 |
| 0.25–0.50 | 2,660 | 0.654 | 5,359 | 0.716 |
| 0.50–0.75 | 2,654 | 0.700 | 8,711 | 0.868 |
| 0.75–1.00 | 2,465 | 0.742 | 5,019 | 0.840 |
| 1.0 | 13,599 | 0.790 | 133 | 0.669 |

### 3.3 Detection confidence (detected boxes only; Spearman ρ; vehicle GT)

| | Def-3 V | V_rsv | Distance | log points |
|---|---|---|---|---|
| ρ, IoU 0.5 (n = 17,303) | +0.266 | +0.382 | −0.313 | **+0.401** |
| Held-out R² gain beyond range + log points (base 0.171) | +0.014 | **+0.061** | base | base |
| ρ, IoU 0.7 (n = 10,451) | +0.228 | +0.318 | −0.331 | **+0.363** |
| Held-out R² gain (base 0.135) | +0.020 | **+0.066** | base | base |

Within each range band, V_rsv correlates with confidence more strongly than point count does. For example, at 30–60 m: +0.417 vs +0.346.

### 3.4 IoU 0.7 and cooperative GT (summary)

| | Def-3 | V_rsv | Distance | Points | Def-3 gain | V_rsv gain |
|---|---|---|---|---|---|---|
| Vehicle GT, IoU 0.7 (recall 0.442) | 0.617 | 0.710 | 0.684 | 0.746 | +0.002 | +0.025 |
| Cooperative GT, IoU 0.5 (recall 0.624) | 0.645 | 0.815 | 0.644 | 0.807 | −0.001 | +0.018 |
| Cooperative GT, IoU 0.7 (recall 0.376) | 0.629 | 0.766 | 0.693 | 0.794 | +0.000 | +0.019 |

The first four columns are miss AUCs; the gains are beyond range + log points. Cooperative GT includes objects the ego cannot observe at all. These are trivially missed and inflate every AUC, so vehicle GT is the primary result.

### 3.5 Reliability (exploratory; DAIR example, vehicle side)

| Check | Def-3 | V_rsv |
|---|---|---|
| Rank correlation between consecutive frames (0.1 s apart, 577 matched objects) | 0.784 | 0.865 |
| Median change between consecutive frames | 0.029 | 0.054 |
| 90th-percentile change between consecutive frames | 0.330 | 0.171 |

- **Depth tolerance:** changing δ between 0.1, 0.2 and 0.5 m gives rank correlations of 0.998–0.999 and AUC between 0.793 and 0.797.
- **Footprint size:** footprints under 30 cells change by a median of 0.107 between frames, against 0.045 for footprints over 100 cells. 31 of 1,419 boxes have fewer than 10 cells.
- **Noise floor:** boxes that both Def-3 and the annotators call fully visible have median V_rsv 0.77 (0.72 within 30 m), mainly from P (mean 0.22).
- **Road returns:** for boxes labelled unoccluded within 60 m, 44–58% of the O_unlab solid angle comes from returns within 0.3 m of the box base height. For boxes labelled occluded, it is 5–11%.

Sections 3.1 to 3.4 are reproduced by the repo scripts in §6. Section 3.5, the bootstrap intervals in 3.1, and the road-return check came from one-off exploratory scripts that were not committed, so treat them as indicative.

## 4. Weaknesses

**Def-3**
- Only labelled boxes can block. It is near chance for misses beyond 60 m (0.531).
- It inherits label errors and flicker, which causes large frame-to-frame jumps.
- It has no sensor model: field of view, range, beam pattern and missing returns are all ignored.
- It is a pure ratio, so it says nothing about apparent size. Most boxes score exactly 1.0.

**RSV (V_rsv)**
- Pass-through (P) lowers visibility without occlusion. P comes from glass and from the car being smaller than its box. It runs against occlusion (P_frac AUC 0.28–0.52 within point-count groups), and it inverts V_rsv for close, dense objects.
- Road returns are classed as unlabelled blockers.
- It depends on the per-sensor ray grid. The RSU grid is approximate, and the RSU origin height is unverified, so the RSU side is not interpretable.
- It is circular with the detector input: it uses the same scan.
- Labels are still needed for the footprint and the labelled/unlabelled split. It is an analysis tool, not a runtime metric.

**Both**
- The human references are coarse, image-based labels that agree with themselves across consecutive frames only 82% of the time.
- One detector, one dataset, one split; neighbouring frames are correlated.
- Self-occlusion is not measured. Viewpoints are not combined.

## 5. Decisions frozen at this date

1. **Metric structure.** Treat the LiDAR-only metric as a decomposition of the expected footprint, not a single blended score:
   ```
   Ω_exp = Ω(H) + Ω(B_lab) + Ω(B_unlab) + Ω(P) + Ω(M)
   V_obs = Ω(H)/Ω_exp                      evidence term (predicts detection)
   O     = Ω(B_lab + B_unlab)/Ω_exp        occlusion term (cause: blocked)
   Ω(P)/Ω_exp, Ω(M)/Ω_exp                  non-occlusion loss (shape/glass, no return)
   ```
   `V_obs` is the same quantity as `V_lo`: the lower bound already defined in THESIS_CONTEXT.md RSV Step 4, which treats missing rays as not visible. It is renamed here for its role. The three versions differ only in how missing rays (M) are handled: `V_lo = Ω(H)/Ω_exp` counts them as not visible; `V_rsv = Ω(H)/Ω(H+B+P)` drops them as unknown; `V_hi = Ω(H+M)/Ω_exp` counts them as visible. Hence `V_lo = V_rsv·(1 − M_frac)` and `V_lo ≤ V_rsv ≤ V_hi`. `V_rsv` is kept as a reported comparison, not as the headline score. This structure was chosen after seeing these results. It is justified by construction (P and M are not occlusion; the spec already excluded M), but it must be confirmed on data not used here.
2. **Primary detection setting.** Ego-only PointPillars, vehicle GT, IoU 0.5, with every GT box kept.
3. **Planned corrections, not yet applied.** Exclude returns at or below the box base height + 0.3 m from B. Report P separately.
4. **Wording.** "Candidate LiDAR-only visibility proxy", until the confirmatory tests below pass.

**Pre-specified confirmatory tests**
- **Second detector, held-out data.** On a second detector (e.g. CoAlign SECOND) and on data not used here, test:
  - held-out gain of `V_obs` and `O` beyond range + log points
  - gain of `O` beyond `V_obs`
  - AUC within point-count quintiles
  - `O` against human labels by range band
- **Controlled intervention.** Inject synthetic blocker returns in front of real objects, compared with removing the same number of points at random.
- **Uncertainty.** Frame- or sequence-level bootstrap intervals on every AUC and gain.
- **Novelty check.** Compare against Hu et al. (WYSIWYG, CVPR 2020) §3, Xu et al. (BtcDet, AAAI 2022) §3, and Luo et al.'s citing papers.

## 6. Reproduction

Environment: `~/miniconda3/envs/openpcdet` (Python 3.8, PyTorch 2.4.1 + CUDA 12.1). The DAIR drive must be mounted at `/mnt/d`. Run from the repo root.

Label comparison (DAIR example):
```bash
PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.compare_visibility_auc
```

Ego PointPillars inference (about 20 minutes, resumable):
```bash
PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.pointpillar_inference --gt vehicle
```
```bash
PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.pointpillar_inference --gt cooperative
```

Detection vs metrics (§3.2–3.4):
```bash
PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.detection_vs_metrics --detections data/pointpillar_ego/val_vehicle_gt.jsonl
```
```bash
PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.detection_vs_metrics --detections data/pointpillar_ego/val_cooperative_gt.jsonl
```

Decomposition tables (§3.1 first table, §3.2 V_lo, O_ext and quintile tables):
```bash
PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.rsv_decomposition_analysis
```

Outputs: `data/pointpillar_ego/val_{vehicle,cooperative}_gt.jsonl` (detections and matches) and `*_metrics.csv` (one row per GT box).

Code state: the scripts above are committed together with this note, on branch `visibility-metrics-rsv`. Use that commit to reproduce the analysis.
