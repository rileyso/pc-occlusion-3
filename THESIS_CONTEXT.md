# THESIS CONTEXT — LiDAR-Based Occlusion Metrics for 3D Object Detection

_Purpose: attach/paste this file into any chat to give complete project context. Last updated: 2026-09-06._

## Project

- **Thesis**: "LiDAR-Based Occlusion Metrics for 3D Object Detection" — Riley So
- **Research question**: at comparable object range and LiDAR point count, does a fine-grained, geometry-based measure of partial occlusion explain vehicle-detection difficulty better than the coarse difficulty/visibility annotations used in established benchmarks (KITTI Easy/Moderate/Hard, Waymo Level 1/2 point-count rule, nuScenes camera-visibility bins)?
- **Proposed metric**:
- **Primary dataset**: Mixed Signals V2X (Sydney; 45.1k point clouds, 240.6k boxes; 3 vehicles + 1 roadside unit; heterogeneous LiDARs; LiDAR-only). NOT yet ingested into OpenPCDet format. KITTI is the current working dataset.
- **Validation plan** 

## Environment (all compute runs in WSL Ubuntu; Windows client drives it via wsl.exe passthrough)

- Repo: `\\wsl.localhost\Ubuntu\home\riley\OpenPCDet` (git; has uncommitted changes — don't commit without asking).
- Conda at `~/miniconda3`; **conda is NOT on the non-interactive PATH** — invoke `~/miniconda3/envs/openpcdet/bin/python` directly. Env `openpcdet`: torch 2.4.1+cu121, pcdet working. Env `mixsig`: Mixed Signals devkit (numpy/scipy, no torch). Python 3.8 in openpcdet env (no dict `|` union).
- GPU: RTX 4060 Laptop 8 GB; nvcc 12.0; pdflatex available in WSL. Thesis LaTeX lives on **Overleaf** (no local .tex); a `report` folder in `C:\Users\Riley\Uni\AMME4112` is empty.
- Windows-side assets: `C:\Users\Riley\Uni\AMME4112\papers-library\lidar-occlusion-baselines` (8 baseline PDFs), thesis PDF in `C:\Users\Riley\Downloads`.

## Data & detectors
- KITTI at `data/kitti` (val split = 3,769 frames). Natural pedestrian benchmark: 824 frames, 2,280 instances. Also present: argo2, lyft, once, waymo dirs.

## Conventions & decisions

## Writing style — avoid in thesis prose

- Repeated explanations that our measure complements rather than replaces performance metrics.
- Abstract phrases such as "intrinsic observability" and "wide geometric configuration space".
- Broad claims about robustness, benchmarking, planning, risk assessment, and annotation standards.
- Similar concluding sentences repeatedly explaining why each paper motivates the thesis.
- Confident interpretations without equally specific supporting tests.

## Next steps / open items