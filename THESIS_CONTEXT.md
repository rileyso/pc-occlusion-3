# THESIS CONTEXT — LiDAR-Based Occlusion Metrics for 3D Object Detection

_Purpose: attach/paste this file into any chat to give complete project context. Last updated: 2026-10-01._

## Project and motivation

- **Thesis**: "LiDAR-Based Occlusion Metrics for 3D Object Detection" — Riley So.
- **Core question**: at comparable object range and LiDAR point count, does a fine-grained, geometry-based measure of partial occlusion explain vehicle-detection difficulty better than the coarse difficulty/visibility annotations used by established benchmarks?
- The work began as a Mixed Signals study of 3D detection and cooperative perception. Early discussions with Mao Shan and Steffany Berrioz identified a more useful thesis contribution: quantify what each LiDAR viewpoint can and cannot observe, rather than build another detector or claim a complete fusion system.
- Mixed Signals is valuable because a single scene is observed by heterogeneous, synchronised vehicle and roadside LiDARs. The same object can therefore be analysed from multiple viewpoints, exposing when an ego vehicle has weak evidence and whether another agent supplies complementary geometry.
- The practical motivation is to replace vague statements such as “occluded” or “hard” with measurable object-level evidence. This can support fairer detector analysis and dataset-quality assessment. Sensor deployment, operational fusion, planning, and detection recovery are motivations/future applications, not claims validated by the current work.

## Proposed metric and experimental logic

The intended primary score is an object-level, sensor-viewpoint visibility/occlusion measure based on sampled target-box surfaces and line-of-sight geometry. External occlusion and self-occlusion should remain separate where possible. Range, point count/density, incidence angle, sensor identity, and detector architecture are covariates or comparison baselines, not part of the definition of occlusion.

Current Mixed Signals return-based proxies are exploratory rather than validated physical visibility metrics:

- `H/C`: measured-return hit fraction, where `H` is returns inside the target box and `C = H+B+P` is the measured candidate-ray count.
- `B/(H+B)`: before-box return fraction; `P/C`: pass-through fraction; `H`: raw return support.
- Vertical-band occupancy, height coverage, and facing-surface occupancy for spatial support.
- Labelled-box visible/occluded surface fraction from sampled target faces.
- Collaborative measures: partner complementary gain, viewpoint redundancy, visibility opportunity, shared-blindness proxy, multi-agent union coverage, and marginal agent gain.

Always retain each score's numerator/denominator, range, projected box size, class, sensor identity, and `H/B/P/C`. Undefined fractions are `unknown`, not zero. These measures indicate geometric evidence or opportunity for collaboration; they do not prove detection recovery.

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

Interpretation: KITTI confirms that coarse benchmark difficulty and point-count rules are useful baselines but do not isolate geometric occlusion. Natural KITTI relationships remain confounded by range, density, pose, and scene composition. The controlled equal-count comparison is the stronger test of whether spatial arrangement/coverage matters beyond point quantity. KITTI ray/surface estimates also miss walls, vegetation, poles, unlabelled vehicles, and sensor-specific return effects.

## Validation plan

1. Establish reproducible detector outputs and object matching on KITTI; retain point count, range, KITTI labels, geometric ray/surface scores, and detector outcomes.
2. Use controlled KITTI thinning and natural KITTI analysis to test whether spatially structured visibility explains detector degradation beyond equal-count random removal and coarse labels.
3. Prepare Mixed Signals data in a common coordinate frame and compute per-sensor and multi-agent support metrics on synchronized object observations.
4. Validate on Mixed Signals with detection status/confidence/localisation outcomes when the detector pipeline is available; compare the proposed score against range, density, point count, and benchmark categories.
5. Report uncertainty, sample counts, sensor identity, surface/ray resolution, calibration assumptions, and limitations. Avoid claiming causal detection recovery from a geometric score alone.

## Environment

- All compute runs in WSL Ubuntu; Windows client drives it via `wsl.exe` passthrough.
- Repo: `\\wsl.localhost\\Ubuntu\\home\\riley\\pc-occlusion-3`; OpenPCDet is separate at `~/OpenPCDet` and has uncommitted changes — do not commit without asking.
- Conda is not on the non-interactive PATH. Use `~/miniconda3/envs/openpcdet/bin/python` for OpenPCDet and the `mixsig` environment for Mixed Signals utilities.
- GPU: RTX 4060 Laptop 8 GB; nvcc 12.0. KITTI is under `OpenPCDet/data/kitti`; validation split has 3,769 frames.
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

## Next steps / open items

- Convert Mixed Signals labels, poses, and multi-agent point clouds into an OpenPCDet-compatible representation, or define a minimal inference interface before adding a detector.
- Finalize one primary spatial-support metric and its surface/ray resolution; do not promote every exploratory proxy to a thesis metric.
- Finish the KITTI controlled/natural analysis and report matched-count results with uncertainty.
- Run detector validation on Mixed Signals, then test whether partner coverage predicts outcomes beyond range and point count.
- Keep external-occluder reconstruction, full multi-agent fusion, sensor deployment optimization, and operational claims out of scope unless separately implemented and validated.
