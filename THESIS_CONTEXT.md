# THESIS CONTEXT — LiDAR-Based Occlusion Metrics for 3D Object Detection

_Purpose: attach/paste this file into any chat to give complete project context. Last updated: 2026-10-08._

## Project and motivation

- **Thesis**: "LiDAR-Based Occlusion Metrics for 3D Object Detection" — Riley So.
- **Core question**: at comparable object range and LiDAR point count, does a fine-grained, geometry-based measure of partial occlusion explain vehicle-detection difficulty better than the coarse difficulty/visibility annotations used by established benchmarks?
- The work began as a Mixed Signals study of 3D detection and cooperative perception. Early discussions with Mao Shan and Steffany Berrioz identified a more useful thesis contribution: quantify what each LiDAR viewpoint can and cannot observe, rather than build another detector or claim a complete fusion system.
- Mixed Signals is valuable because a single scene is observed by heterogeneous, synchronised vehicle and roadside LiDARs. The same object can therefore be analysed from multiple viewpoints, exposing when an ego vehicle has weak evidence and whether another agent supplies complementary geometry.
- The practical motivation is to replace vague statements such as “occluded” or “hard” with measurable object-level evidence. This can support fairer detector analysis and dataset-quality assessment. Sensor deployment, operational fusion, planning, and detection recovery are motivations/future applications, not claims validated by the current work.

## Visibility metrics and experimental logic

The previously discussed sampled-surface visibility metric was an exploratory proxy/placeholder, not a proposed thesis metric or novel contribution. Do not describe it as "the proposed surface visibility metric" or assume it remains a planned primary metric. Existing surface-based experiments are exploratory work only.

The current DAIR-V2X experiment evaluates the existing Definition-3 solid-angle visibility metric from Luo et al., "3D Object Visibility Prediction in Autonomous Driving" (arXiv:2403.03681). This is a literature metric, not a newly proposed metric. As of 2026-10-08, one **candidate** metric is specified below (Return-Weighted Solid-Angle Visibility, RSV), implemented on DAIR-V2X but not validated, and must not be described as an established primary metric until the validation steps below are complete. External occlusion and self-occlusion should remain separate where possible. Range, point count/density, incidence angle, sensor identity, and detector architecture are covariates or comparison baselines, not part of the definition of occlusion.

Current Mixed Signals return-based proxies are exploratory rather than validated physical visibility metrics:

- `H/C`: measured-return hit fraction, where `H` is returns inside the target box and `C = H+B+P` is the measured candidate-ray count.
- `B/(H+B)`: before-box return fraction; `P/C`: pass-through fraction; `H`: raw return support.
- Vertical-band occupancy, height coverage, and facing-surface occupancy for spatial support.
- Labelled-box visible/occluded surface fraction from sampled target faces.
- Collaborative measures: partner complementary gain, viewpoint redundancy, visibility opportunity, shared-blindness proxy, multi-agent union coverage, and marginal agent gain.

Always retain each score's numerator/denominator, range, projected box size, class, sensor identity, and `H/B/P/C`. Undefined fractions are `unknown`, not zero. These measures indicate geometric evidence or opportunity for collaboration; they do not prove detection recovery.

## Candidate metric: Return-Weighted Solid-Angle Visibility (RSV) — implementation instructions

_Status: specified 2026-10-08; implemented for DAIR-V2X only (`dairv2x/rsv.py`, 2026-10-08); not validated. RSV is not planned for KITTI. Novelty is unconfirmed — a literature check (2026-10-08) found no prior per-object metric combining these parts, but the Google Scholar "cited by" list for Luo et al. still needs checking before any novelty claim._

### Idea in one paragraph

Luo et al. Definition 3 gives the *expected* visibility of a box from geometry alone (exact solid angle on the unit sphere, occluded only by closer labelled boxes, object-level depth ordering, no points). RSV keeps Definition 3's solid-angle denominator but fills the numerator from the *measured* first returns, ray by ray, weighting each ray by the solid angle it covers. It is the solid-angle-weighted generalisation of the existing `H/B/P/C` proxies, with an explicit fourth class for rays with no return. Because each LiDAR ray covers a fixed solid angle set by the scan pattern, weighting by ray solid angle removes the range/density bias of raw point counts and the bias from non-uniform ring spacing.

### Positioning against literature (for related-work prose)

- Luo et al. 2024 (arXiv:2403.03681), Def. 3: exact box solid angle, box-only occluders — RSV's denominator and baseline.
- Luo et al. Def. 2 / Waymo-style point counts: point-based, density-biased — baseline.
- Hu et al., "What You See is What You Get" (CVPR 2020): ray-cast free/occupied/unknown volume from returns, used as detector input, not an object score.
- Xu et al., BtcDet (AAAI 2022): separates external occlusion from signal miss in spherical coordinates; no scalar metric.
- MIDAR (arXiv:2508.02858) and cited Cao/Xiong/Liu 2022, Zhang/Feng 2024: 2D azimuth-interval/shadow models, box-only.
- Cohen-Or et al. 2003 survey: Coorg–Teller solid-angle heuristic (§5.2), hierarchical occlusion maps / opacity (§6.3), supporting/separating planes for full vs partial occlusion (§3, Fig. 9).

### Step 1 — Per-sensor ray grid (range image)

1. Work in each sensor's own frame with origin at the LiDAR optical centre (not the ego/vehicle origin). For DAIR-V2X, use each side's own LiDAR frame (vehicle LiDAR; RSU virtual-LiDAR frame, whose origin height is unverified). For Mixed Signals, use per-agent sensor extrinsics.
2. Build a range image: rows = laser rings (elevation), columns = azimuth bins.
   - Elevation per ring: use the sensor's beam table if available; otherwise estimate from data and record the method. DAIR PCDs have no ring field. Vehicle: Hesai Pandar40P nominal table (40 rings, −25° to +15°), each ring refined to the median elevation of its assigned returns. RSU: ring elevations drift with azimuth in the virtual-LiDAR frame, so a uniform 0.2° × 0.25° grid over the observed field of view is used as an approximation.
   - Azimuth bin width: sensor horizontal resolution (DAIR vehicle 0.2°, measured; RSU 0.25° uniform grid). Record the value used per sensor.
3. Each cell `k` stores: ray direction `d_k` (cell-centre azimuth/elevation), first-return range `r_k` (minimum range in the cell), or `EMPTY` if no return.
4. Cell solid angle, exact for an elevation band (elevation measured from the horizontal plane):
   `Ω_k = Δφ_k · (sin(ε_top,k) − sin(ε_bot,k))`, with ring boundaries at midpoints between adjacent ring elevations. Check: `Σ_k Ω_k` over the full grid equals the sensor's vertical-FOV band solid angle.
5. Motion: use the dataset's provided (de-skewed) points; note in limitations if the dataset is not motion-compensated.

### Step 2 — Target footprint (expected solid angle)

1. For target box `b_i` (optionally inflated by a small margin, e.g. 0.1 m, for hit assignment only — record the margin), find candidate cells: restrict to cells within the box's azimuth/elevation bounding range, then run a vectorised ray–OBB slab test from the origin along `d_k`. Keep the entry/exit distances `t_in,k`, `t_out,k`.
2. `K_i` = cells whose ray intersects `b_i`. Expected solid angle `Ω_i^exp = Σ_{k∈K_i} Ω_k`.
3. Sanity check: compare `Ω_i^exp` against the continuous Def-3 solid angle `Ω_i^def3` (union of six projected faces, `∬ sin θ dθ dφ`). The ratio measures discretisation error; report it and flag objects with `|K_i| < N_min` (start with `N_min = 10`) as low angular resolution. Keep flagged objects, do not drop them silently.
4. The footprint is the box's silhouette, so self-occlusion does not enter RSV. Keep self-occlusion/facing-surface measures separate, as above.

### Step 3 — Classify each footprint ray

With depth tolerance `δ` (start with 0.2 m; run a sensitivity sweep over {0.1, 0.2, 0.5} m):

| Class | Condition | Meaning |
|---|---|---|
| `H` hit | `t_in − δ ≤ r_k ≤ t_out + δ` | return from the target |
| `B` before | `r_k < t_in − δ` | external occluder in front |
| `P` pass-through | `r_k > t_out + δ` | beam passed the box (transparency, gaps, loose label, label error) |
| `M` missing | cell `EMPTY` | no return: dropout, absorption, specular surface, out of range — **not** counted as occluded |

Further split `B` by whether the ray also intersects another labelled box closer than `t_in`:
- `B_lab`: blocked by a labelled object (what Def 3 can see);
- `B_unlab`: blocked by something unlabelled (walls, vegetation, poles, unlabelled vehicles, `DontCare`) — what Def 3 cannot see.

Counts `H, B, P` must reproduce the existing `H/B/P/C` proxies when every cell has weight 1; use this as a regression test against `mixedsig/PARTIAL_OCCLUSION_METRICS.md`.

### Step 4 — Scores (all solid-angle weighted; `Ω(X) = Σ_{k∈X} Ω_k`)

- Measured denominator: `Ω_meas = Ω(H) + Ω(B) + Ω(P)`.
- **RSV visibility**: `V_rsv = Ω(H) / Ω_meas` (unknown if `Ω_meas = 0`).
- External occlusion: `O_ext = Ω(B) / Ω_meas`, split into `O_lab = Ω(B_lab)/Ω_meas` and `O_unlab = Ω(B_unlab)/Ω_meas`.
- Pass-through fraction: `P_frac = Ω(P) / Ω_meas`.
- Missing fraction: `M_frac = Ω(M) / Ω_i^exp`.
- Bounds over the missing rays: `V_lo = Ω(H)/Ω_i^exp` (missing = not visible), `V_hi = Ω(H ∪ M)/Ω_i^exp` (missing = visible).
- Gap to the literature baseline: `Δ_def3 = V_def3 − V_rsv`. Expect `O_unlab`, `P_frac` and `M_frac` to account for most of the gap; report this decomposition.

Undefined fractions are `unknown`, not zero.

### Step 5 — What is *not* in the definition

Per the rule above, incidence angle, intensity, range and point count are **covariates, not weights**. RSV uses only ray solid angle. If an incidence- or intensity-weighted variant is tried, label it a sensitivity analysis and keep the unweighted RSV as the reported metric.

### Step 6 — Outputs to cache (one row per object × sensor × frame)

`frame, sensor, object_id, class, range, projected_box_size, |K_i|, low_res_flag, Ω_exp, Ω_def3, Ω_H, Ω_B_lab, Ω_B_unlab, Ω_P, Ω_M, H, B, P, M, C, V_rsv, V_lo, V_hi, V_def3, O_ext, O_lab, O_unlab, P_frac, M_frac, Δ_def3, δ, box_margin, azimuth_res, ring_estimation_method`.

### Step 7 — Verification before using results

1. **Synthetic unit tests** (no dataset): (a) isolated box → `V_rsv ≈ 1`, `Ω_exp ≈ Ω_def3`; (b) planar wall covering half the box → `V_rsv ≈ 0.5`, all occlusion in `B_unlab`; (c) closer labelled box covering half → occlusion in `B_lab`, `V_rsv ≈ V_def3`; (d) random cells emptied → `M_frac` rises, `V_rsv` unchanged in expectation; (e) box placed past range → `Ω_meas = 0` → unknown.
2. **Regression test**: unit weights reproduce existing `H/B/P/C`.
3. **Resolution check**: distribution of `Ω_exp/Ω_def3` and `|K_i|` vs range and class.

### Step 8 — Experiments

RSV is evaluated on DAIR-V2X, not KITTI.

1. **DAIR-V2X natural population**: compare `V_rsv` with `V_def3`, the per-side `occluded_state` label, log point count and range, overall and within 0–30/30–60/60+ m (`dairv2x/compare_visibility_auc.py`). Report the `Δ_def3` decomposition.
2. **Detector outcome (DAIR-V2X)**: on matched CoAlign / single-agent detections, model detection/IoU/confidence ~ `V_rsv` + log range + log point count + class; compare against the same model with `V_def3`. Report sample sizes, denominators and uncertainty.
3. **Mixed Signals**: compute RSV per sensor (vehicle `laser`, `003`, `004`; roadside `top`, `dome`) using each sensor's own ray grid. Compare per-sensor RSV for the same object. Do **not** sum solid angles across sensors (they are different spheres); multi-agent union coverage needs a shared surface domain and stays on the existing proxy path until that definition is fixed.

First DAIR-V2X result (2026-10-08, 46-pair example, δ = 0.2 m, AUC vs `occluded_state` > 0; preliminary, unvalidated): vehicle side, n = 1,419, RSV 0.795 vs Def-3 0.675 overall; RSV higher at 30–60 m (0.801 vs 0.608) and 60+ m (0.735 vs 0.639) but lower at 0–30 m (0.720 vs 0.904). Near fully visible boxes have large `P` (rays through windows, cabin above a loose box, open cyclist frames), which lowers `V_rsv` without occlusion. RSU results (RSV 0.681 vs Def-3 0.685) are not interpretable until the RSU origin height and ray grid are resolved.

### Writing guidance for RSV

- Call it a "candidate visibility proxy" until Step 8.3 shows it explains detector outcomes beyond range and point count.
- Present it as combining Luo et al.'s Definitions 2 and 3: Def-3's density-free solid-angle normalisation with return-level evidence that captures unlabelled occluders and separates missing returns.

## Mixed Signals collaborative study

- **Dataset**: Mixed Signals V2X, Sydney; approximately 45.1k point clouds and 240.6k boxes, from three connected vehicles (`laser`, `003`, `004`) and two roadside LiDAR views (`top`, `dome`) as represented by the current explorer. It is LiDAR-only/heterogeneous and provides synchronised labels and poses.
- **Work completed**: `mixedsig/MixedSignalsExplorer` loads labels, synchronized point clouds, vehicle odometry, fixed RSU poses, map/agent transforms, categories, tracks, and per-agent frames. Visualization utilities now aggregate viewpoints, colour points by collaboration rank, and inspect ray/box relationships.
- **Current collaborative analysis**: compare per-sensor object evidence, then measure how much supported target geometry a partner adds beyond the ego sensor. Start with raw support, before/pass-through returns, four-band vertical occupancy, and partner complementary gain. Use union coverage and marginal gain only after the spatial-support definition is fixed.
- **Status/limit**: the dataset is not yet converted to OpenPCDet format and no Mixed Signals detector validation or fusion result is complete. Do not describe the current visualization/proxies as a validated occlusion metric, a communication/fusion method, or proof that collaboration improves detection.

## KITTI work and insights

KITTI is the current working/validation dataset because its tooling, labels, detector checkpoints, and point clouds are immediately available. It is a single-Velodyne benchmark, not a substitute for the multi-view Mixed Signals experiment.

Completed KITTI work:

- Ray tracing from the Velodyne origin through labelled 3-D boxes estimates geometric occlusion by another labelled box. It uses KITTI calibration and oriented-box intersections; `DontCare`, unlabelled scenery, sparse sampling, and missing returns are not counted as blockers.
- A sampled, area-weighted visible-surface metric was added and cached alongside ray results. It is a geometric label-derived measure, not detector output or the original laser firing pattern.
- A natural validation population contains all 3,769 KITTI validation frames, 824 frames with pedestrians, and 2,280 pedestrian instances. A controlled benchmark creates 820 structured-occlusion incidents plus 820 equal-count random-removal controls from 82 base pedestrians; the independent unit remains the 82 base pedestrians.
- PointPillars baseline 3-D AP_R40 on the complete validation split: Car 87.76/78.39/75.19, Pedestrian 57.33/51.50/46.92, Cyclist 81.57/62.95/58.96 for Easy/Moderate/Hard.
- The transplanted Waymo point-count rule is a useful negative/control result: literal `<=5` points leaves only 20 of 2,279 evaluable pedestrians in LEVEL_2, gives AUC about 0.51, and is effectively a coin flip on KITTI. A density-matched `<=26` threshold stratifies detection, but still conflates range with visibility; continuous log point count is more informative than the one-bit level.

Interpretation: KITTI confirms that coarse benchmark difficulty and point-count rules are useful baselines but do not isolate geometric occlusion. Natural KITTI relationships remain confounded by range, density, pose, and scene composition. The controlled equal-count comparison is the stronger test of whether spatial arrangement/coverage matters beyond point quantity. KITTI ray/surface estimates also miss walls, vegetation, poles, unlabelled vehicles, and sensor-specific return effects — the RSV `B_unlab` and `M` classes are intended to measure exactly these.

## Validation plan

1. Establish reproducible detector outputs and object matching on KITTI; retain point count, range, KITTI labels, geometric ray/surface scores, and detector outcomes.
2. Use controlled KITTI thinning and natural KITTI analysis to test whether spatially structured visibility explains detector degradation beyond equal-count random removal and coarse labels.
3. RSV is implemented for DAIR-V2X (`dairv2x/rsv.py`) and compared with Def-3 there (`dairv2x/compare_visibility_auc.py`). RSV is not planned for KITTI.
4. Prepare Mixed Signals data in a common coordinate frame and compute per-sensor and multi-agent support metrics on synchronized object observations.
5. Validate on Mixed Signals with detection status/confidence/localisation outcomes when the detector pipeline is available; compare the proposed score against range, density, point count, and benchmark categories.
6. Report uncertainty, sample counts, sensor identity, surface/ray resolution, calibration assumptions, and limitations. Avoid claiming causal detection recovery from a geometric score alone.

## Environment

- All compute runs in WSL Ubuntu; Windows client drives it via `wsl.exe` passthrough.
- Repo: `\\wsl.localhost\\Ubuntu\\home\\riley\\pc-occlusion-3`; OpenPCDet is separate at `~/OpenPCDet` and has uncommitted changes — do not commit without asking.
- Conda is not on the non-interactive PATH. Use `~/miniconda3/envs/openpcdet/bin/python` for OpenPCDet and the `mixsig` environment for Mixed Signals utilities.
- GPU: RTX 4060 Laptop 8 GB; nvcc 12.0.
- Thesis LaTeX is on Overleaf; local report material is under `live-report/`.

## Writing constraints

- Say “geometric evidence”, “visibility proxy”, or “opportunity for collaboration” unless a detector result supports a stronger claim.
- Do not repeatedly say the metric complements rather than replaces performance metrics.
- Avoid unsupported claims about robustness, benchmarking, planning, risk assessment, annotation standards, or “intrinsic observability”.
- State sample sizes and denominators; separate completed results from planned validation.

## Key files

- `mixedsig/mixed_signals.py` — Mixed Signals dataset explorer and transforms.
- `mixedsig/PARTIAL_OCCLUSION_METRICS.md` — current proxy definitions and recommended first experiment.
- `mixedsig/tools/visualize_agg_frame.py` — synchronized multi-agent visualization and per-object return evidence.
- `kitti/inspect_frame.py`, `kitti/q1.py` — KITTI ray/surface metrics and detector matching.
- `kitti/RAY_TRACING.md` — ray metric definition and limitations.
- `waymo_baseline/README.md`, `waymo_baseline/report_text.md` — point-count baseline results and thesis linkage.
- `live-report/main.tex`, `live-report/methodology.tex` — current thesis prose and experimental design.
- `dairv2x/rsv.py` — RSV implementation (per-sensor ray grid, footprint, H/B/P/M classification, scores) for DAIR-V2X.
- `dairv2x/compare_visibility_auc.py` — Def-3 vs RSV vs distance vs point count AUC tables against DAIR `occluded_state`.

## Next steps / open items

- RSV on DAIR-V2X: add the Step 7 synthetic tests and `H/B/P/C` regression check, and report the `Δ_def3` decomposition (`O_unlab`, `P_frac`, `M_frac`).
- Resolve the DAIR RSU origin height and RSU ray grid before interpreting RSU RSV.
- Check Google Scholar "cited by" for Luo et al. (arXiv:2403.03681) and re-read WYSIWYG §3 and BtcDet §3 before making any novelty claim for RSV.
- Convert Mixed Signals labels, poses, and multi-agent point clouds into an OpenPCDet-compatible representation, or define a minimal inference interface before adding a detector.
- Evaluate literature visibility metrics and their numerical resolution; do not promote the surface-visibility placeholder or other exploratory proxies to proposed thesis metrics.
- Finish the KITTI controlled/natural analysis and report matched-count results with uncertainty.
- Run detector validation on Mixed Signals, then test whether partner coverage predicts outcomes beyond range and point count.
- Keep external-occluder reconstruction, full multi-agent fusion, sensor deployment optimization, and operational claims out of scope unless separately implemented and validated.
