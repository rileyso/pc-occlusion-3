# DAIR-V2X-C per-frame visualization

Run from the repository root with the Mixed Signals visualization dependencies installed:

```bash
PYTHONPATH=. python -m dairv2x.visualize_agg_frame \
  --root /mnt/c/Datasets/dair-v2x-c-example --pair-index 0
# Headless geometry / sanity check:
PYTHONPATH=. python -m dairv2x.visualize_agg_frame --pair-index 0 --no-show
PYTHONPATH=. python -m unittest dairv2x.test_explorer
```

`--pair-index` selects an entry in `cooperative/data_info.json` sorted by vehicle point-cloud filename; the script prints both selected filenames. Requires `numpy`, `pypcd4`, `pyquaternion`, and `open3d` (not all are listed in the repository's general `requirements.txt`). The existing `mixsig` conda environment has them all. No OpenPCDet/CUDA is used.

The DAIR visualization has its own rendering flow (it does not call the Mixed Signals `main`). It reuses `PointPainter`, SE(3) geometry utilities, and `classify_box_returns`/`before_return_fraction` from `mixedsig/tools/visualize_agg_frame.py`, adding **both complete LiDAR clouds** (vehicle cyan, infrastructure orange) and all cooperative GT boxes (gray) in the vehicle LiDAR frame. DAIR pairs replace Mixed Signals sequences/timestamps; the vehicle LiDAR is the display frame instead of TOP. PCDs supply points, the two calibration chains supply sensor poses, and `cooperative/label_world` supplies world-frame GT boxes. Box yaw comes from `world_8_points` because DAIR's `rotation` may be missing or not in world axes. No Mixed Signals car-height/intensity adjustment applies. DAIR `system_error_offset` is not applied: these are calibrated/world-frame points, not a deliberately noisy cooperative transform.

For each GT box and sensor, stdout reports before/inside/beyond counts over the candidate measured rays (directions that intersect the box), plus before/(before+inside). `unknown` means there are no before-or-inside returns; a sensor inside a box is excluded. These are simple geometric return proxies, **not detector performance or an estimate of all emitted rays**. `--no-show` prints the same metrics without opening a window. The default camera looks down over the ego vehicle (vehicle LiDAR origin), with vehicle-forward pointing up; use `--zoom 1.0` to move closer or a smaller value to zoom out. DAIR's labels have no track IDs; box indices are frame-local.

## One-frame CoAlign detector baseline

`coalign.py` wraps the **official** [CoAlign](https://github.com/yifanlu0227/CoAlign) inference path; it does not reimplement fusion. Clone/install CoAlign following its README, obtain its [DAIR-V2X checkpoint](https://drive.google.com/drive/folders/1zsgEMTGpB_Llz66SqeegXXKdaduIegXE?usp=sharing), and prepare its `my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure` data layout with complemented annotations. This needs a split JSON, `cooperative/data_info.json`, the point clouds, calibration, the original labels in `*_backup`, and the checkpoint's `config.yaml`. The official box-alignment config also needs its **precomputed stage-1 prediction JSON**; download it from the [coalign_precalc folder](https://drive.google.com/drive/folders/1otDzESlepuhRBE4ZgJQfpArnpG1TG8uu) and pass `--stage1` if it is not at the checkpoint config's path. Checkpoints and datasets are not bundled here.

For the small original DAIR example, `prepare_coalign.py` builds a separate overlay: symlinks for LiDAR/calibration and original backup labels, complemented labels from the [annotation ZIP](https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view), and the official train/val split files. It leaves the source dataset untouched. From the repository root, in a Python environment with CoAlign installed:

```bash
python -m dairv2x.prepare_coalign \
  --root /path/to/dair-v2x-c-example \
  --annotations /path/to/DAIR-V2X-C_Complemented_Anno.zip \
  --output data/coalign_example
python -m dairv2x.coalign \
  --coalign-repo /path/to/CoAlign \
  --coalign /path/to/DAIR_coalign_checkpoint \
  --root data/coalign_example --split train --pair-index 0 \
  --stage1 /path/to/coalign_precalc/dairv2x/train/stage1_boxes.json \
  --output data/frame0.json
```

On this machine, `openpcdet` (Python 3.8, PyTorch 2.4.1, spconv 2.3.6) successfully ran frame `015372` after cloning CoAlign, building its `opencood/utils/box_overlaps` extension, and installing `h5py`, `shapely==2.0.0`, `g2o-python`, `icecream`, `timm`, `efficientnet_pytorch`, `pypcd` (the CoAlign requirements Git fork), and `cython`. For a fresh setup, follow CoAlign's installation guide rather than assuming these additions cover every environment.

Optionally add `--single /path/to/DAIR_pointpillar_single_checkpoint` for the vehicle-only comparison (requires a separately trained, compatible checkpoint). Check matching logic with `python -m unittest dairv2x.test_coalign`.

Visualize saved detections (uses the existing Mixed Signals/Open3D visualization dependencies, e.g. the `mixsig` conda environment; no CoAlign model needed for this step):

```bash
conda run -n mixsig python -m dairv2x.coalign_visualisation \
  --results data/frame0.json --root /path/to/dair-v2x-c-example
# Geometry-only check without opening a window: add --no-show
```

For an ego-position perspective, facing vehicle +X and starting **2 m above estimated local ground**:

```bash
~/miniconda3/envs/mixsig/bin/python -m dairv2x.coalign_ego_view \
  --results data/coalign_assets/val_000211.json --root data/coalign_full
```

This retains the box colors and 1 px points. Ground is approximated by a near-horizontal plane fitted to nearby vehicle returns below the LiDAR. Use `--ground-z VALUE` (vehicle-LiDAR coordinates) if this estimate is unsuitable, or `--height VALUE` to change height above ground. `--no-show` checks the geometry/camera without opening a window. Check camera geometry with `python -m unittest dairv2x.test_ego_view`.

Vehicle points are cyan, infrastructure points orange, and all GT boxes remain white. CoAlign predictions are green when matched to GT and red when unmatched under the JSON's one-to-one 3D IoU matching (including duplicate predictions). Colors do not use a confidence threshold. A GT box without a prediction stays white; **a miss does not necessarily mean occlusion**. GT boxes are those retained by CoAlign's evaluation range/filtering, not necessarily every label in the original example.

The pair index uses the same sorted list as the visualizer, but the selected vehicle frame **must be in the specified CoAlign split**. All 46 pairs in the local `/mnt/c/Datasets/dair-v2x-c-example` are in CoAlign's **train** split; they are useful for a smoke test, not for held-out evaluation. Official complemented annotations and precomputed stage-1 train/val/test JSONs are linked in the [CoAlign README](https://github.com/yifanlu0227/CoAlign#checkpoints). The downloaded archive supplies `train.json`, `val.json` and replacement labels; retain the original labels in `cooperative/label_world_backup` and `vehicle-side/label/lidar_backup` before replacing them. `frame.json` contains predicted 3D box corners and confidence for both runs (when `--single` is set), plus one-to-one confidence-ordered **3D IoU** matches against each CoAlign cooperative GT box; `null` means no prediction reaches `--iou` (default 0.5). Boxes are in the vehicle LiDAR frame. The vehicle-only comparison is optional because its **separate** trained checkpoint may be unavailable. This is a detector-recovery baseline, not an occlusion measurement; the geometric before-return score is computed by `visualize_agg_frame.py` separately. Do not compare scores across incompatible checkpoint/config or annotation versions without checking the data conversion.
