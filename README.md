# lidar visibility metrics

Vehicle-side LiDAR analysis for the DAIR-V2X dataset. Computes box difficulty labels and a surface-coverage score: unique grid cells entered by rays with in-box returns, divided by all sensor-facing cells. Scores depend on return, distance and occlusion.

## Setup

Use Python 3.10+ and install the analysis dependencies:

```bash
pip install numpy matplotlib pypcd-imp
```

Download the DAIR-V2X vehicle-side data separately. Set `ROOT` in `dairv2xv/process_boxes.py` (and `dairv2xv/rsv.py` when using RSV) to your `vehicle-side` directory, containing `data_info.json`, `velodyne/`, `label/`, `calib/`, and `image/`.

## Run

From the repository root:

```bash
python dairv2xv/process_boxes.py
```

Computes per-box point counts, difficulty, and surface occupancy; prints car statistics and plots scores by difficulty. Currently uses a **0.05 m** grid, configurable in `compute_surface_occupancy()`.

For the alternative ray-grid RSV analysis:

```bash
python dairv2xv/rsv.py
```
