# THESIS CONTEXT — LiDAR-Based Occlusion Metrics for 3D Object Detection

_Purpose: attach/paste this file into any chat to give complete project context. Last updated: 2026-09-06._

## Project

- **Thesis**: "LiDAR-Based Occlusion Metrics for 3D Object Detection" — Riley So, supervisor Mao Shan, University of Sydney, AMME4112 (2026).
- **Research question**: at comparable object range and LiDAR point count, does a fine-grained, geometry-based measure of partial occlusion explain vehicle-detection difficulty better than the coarse difficulty/visibility annotations used in established benchmarks (KITTI Easy/Moderate/Hard, Waymo Level 1/2 point-count rule, nuScenes camera-visibility bins)?
- **Proposed metric**: object-level visibility score `v_i = A(V_i)/A(S_i)` (visible surface area over observable surface area) estimated by ray tracing from the LiDAR origin to sampled box-surface points; external occlusion and self-occlusion reported separately. Range, point count and density are covariates/comparison baselines, NOT part of the metric definition.
- **Primary dataset**: Mixed Signals V2X (Sydney; 45.1k point clouds, 240.6k boxes; 3 vehicles + 1 roadside unit; heterogeneous LiDARs; LiDAR-only). NOT yet ingested into OpenPCDet format. KITTI is the current working dataset.
- **Validation plan** (§5.6): logistic regression for detection status, binned degradation curves, bootstrap CIs; compare metric vs point count, density, range, coarse labels, and DCD.

## Environment (all compute runs in WSL Ubuntu; Windows client drives it via wsl.exe passthrough)

- Repo: `\\wsl.localhost\Ubuntu\home\riley\OpenPCDet` (git; has uncommitted changes — don't commit without asking).
- Conda at `~/miniconda3`; **conda is NOT on the non-interactive PATH** — invoke `~/miniconda3/envs/openpcdet/bin/python` directly. Env `openpcdet`: torch 2.4.1+cu121, pcdet working. Env `mixsig`: Mixed Signals devkit (numpy/scipy, no torch). Python 3.8 in openpcdet env (no dict `|` union).
- GPU: RTX 4060 Laptop 8 GB; nvcc 12.0; pdflatex available in WSL. Thesis LaTeX lives on **Overleaf** (no local .tex); a `report` folder in `C:\Users\Riley\Uni\AMME4112` is empty.
- Windows-side assets: `C:\Users\Riley\Uni\AMME4112\papers-library\lidar-occlusion-baselines` (8 baseline PDFs), thesis PDF in `C:\Users\Riley\Downloads`.

## Data & detectors

- KITTI at `data/kitti` (val split = 3,769 frames). Natural pedestrian benchmark: 824 frames, 2,280 instances. Also present: argo2, lyft, once, waymo dirs.
- Pedestrian-capable detectors (OpenPCDet configs + local checkpoints): **PointPillars, SECOND, PointRCNN, PV-RCNN**. Voxel R-CNN checkpoint is Car-only; CenterPoint/Part-A2 unavailable (no local KITTI checkpoint).
- KITTI 3D AP_R40 val reference (PointPillars): Car 87.76/78.39/75.19, Pedestrian 57.33/51.50/46.92, Cyclist 81.57/62.95/58.96.
- Archived `*_val_detections.json` in `archive/20260904-135049/outputs/research/` are **Car-only** — pedestrian-preserving predictions live in `dcd_baseline/results/detections/`.

## Completed work

1. **Gilroy replication** (`gilroy_lidar_replication`): KITTI adaptation of Gilroy et al. 2023 pedestrian-occlusion benchmark. (a) Natural benchmark: all 2,280 val pedestrians, ray-cast occlusion from annotated boxes, 10 bands ≥55. (b) Controlled benchmark: 82 pedestrians × 10 levels = 820 structured occlusions + 820 equal-count random-removal controls (repeated measures; manifest-based on-the-fly transforms). Structured-arm 3D mAP@[.5:.95]: PP 0.041, SECOND 0.023, PointRCNN 0.021, PV-RCNN 0.047; random controls higher for all. Pending: paired bootstrap CIs.
2. **DCD baseline** (`dcd_baseline`): pure-PyTorch port of official DCD (Wu et al., NeurIPS 2021) validated against a hand-computed case; α=1.0 raw metres (α-sweep justified; official α=1000 assumes unit-normalised shapes); ideal set = area-weighted 12×12 samples of LiDAR-facing box faces (self-occlusion excluded by construction); observed set = returns in GT box. Pipeline: `compute_dcd.py` → `infer_natural.py` → `evaluate_dcd.py` → `plot_dcd.py` (curves) / `plot_dcd_binned.py` (bars) / `plot_dcd_verdict.py` (summary figure).
   - **KEY FINDING**: with a fixed ideal surface vs variable-size observed returns, the official DCD formulation acts as a **return-density proxy**, not an occlusion metric: corr(DCD, points) = **+0.63**, corr(DCD, range) = **−0.71**; detection probability RISES monotonically with DCD for all four detectors (change DCD P10→P95: +46.9 PP, +41.8 SECOND, +67.6 PointRCNN, +42.3 PV-RCNN pp unadjusted; +29.0/+9.1/+7.5/+18.0 after adjusting for range + log point count). This is the falsification result motivating the thesis metric. Overall detection (IoU ≥ 0.5, score ≥ 0.1): PP 67.0%, SECOND 65.9%, PointRCNN 41.6%, PV-RCNN 70.9%; AP@0.50 overall 40.8/37.3/23.5/44.1%.
   - Per-object data (ready for the §5.6 regression): `results/analysis/per_object_outcomes_<detector>.csv` (frame_id, dcd, ground_range_m, num_points_in_gt, detected) + `dcd_effect_summary.csv`.
   - Key figures: `fig_report_dcd_comparison.png` (report, 300 dpi), `fig_dcd_detection_probability.png`, `fig_dcd_adjusted_effect.png`, `fig_dcd_binned_map.png` (corrected), `fig_dcd_baseline_verdict.png` (two-panel "why DCD fails").
   - Report prose ready in `dcd_baseline/report_text.md` (already pasted into Overleaf §"Density-Aware Chamfer Distance Baseline", \cite{bib_ref_33}).
   - **BUG HISTORY**: early binned figures/tables were invalid — `evaluate_dcd.py` zipped sorted DCD values against unsorted rows, scrambling bin assignment (fixed 2026-09-05). Corrected binned results rise monotonically with DCD. Do not reuse any pre-fix binned artefacts.
3. **Literature baselines selected** (5): Waymo point-count Level rule + range bins (Sun et al. CVPR 2020); WYSIWYG ray-cast visibility (Hu et al. CVPR 2020); 3D Corruptions AD / KITTI-C (**Dong** et al. CVPR 2023 — not "Zou"); DCD (Wu et al. NeurIPS 2021) — implemented above; PE-VGOP occupancy entropy (He et al. 2024). Runners-up: PDV (CVPR 2022), Robo3D (ICCV 2023), Luo 2024 visibility prediction. Note: WYSIWYG is nuScenes-based voxel visibility maps, not per-object surface fraction.

## Conventions & decisions

- Matching protocol everywhere: greedy confidence-ranked matching, 3D IoU ≥ 0.50, score ≥ 0.1; predictions matching other annotated objects are ignored, not counted FP (Gilroy semantics).
- Binning: equal-population bins merged to ≥55 instances (Gilroy minimum). Beware: equal-spaced categorical axes over-weigh mid-bin noise — prefer curves for analysis, bars only for illustration.
- Box surface is the object-surface proxy (KITTI has no meshes); facing faces only; natural benchmark is confounded by unlabelled occluders — controlled benchmark is primary for causal claims.
- DCD port replaces compiled CUDA kernels with `torch.cdist` (nvcc 12.0 vs torch cu121 made extension builds risky); math identical, tests in `test_dcd_metric.py`.
- Operating point documented everywhere: α=1.0 raw metres, n_λ=1, non_reg=False; `--alpha`/`--normalize` flags exist for sensitivity analysis.

## Writing style — avoid in thesis prose

- Repeated explanations that our measure complements rather than replaces performance metrics.
- Abstract phrases such as "intrinsic observability" and "wide geometric configuration space".
- Broad claims about robustness, benchmarking, planning, risk assessment, and annotation standards.
- Similar concluding sentences repeatedly explaining why each paper motivates the thesis.
- Confident interpretations without equally specific supporting tests.

## Next steps / open items

1. Build the thesis visibility metric `v_i` (ray tracing per §5.4) and run the §5.6 regression: detected ~ v_i + range + point count + density (+ DCD), per-object tables already prepared.
2. Ingest Mixed Signals into OpenPCDet format (mixsig env; pipeline is dataset-portable: needs only an instances CSV with `frame_id` + 7 box fields and predictions JSON `{frame_id, class_name, score, box_lidar}`).
3. Thesis LaTeX: `images/dcd-baseline/fig_dcd_map_bars.png` reference must be swapped for the corrected `fig_dcd_binned_map.png` (old file retired with the bug).
4. Optional: Car-class DCD baseline (archived Car predictions reusable); pedestrian-compatible CenterPoint/Part-A2/Voxel R-CNN; paired bootstrap CIs for the Gilroy controlled experiment; DCD α/normalize sensitivity appendix.
