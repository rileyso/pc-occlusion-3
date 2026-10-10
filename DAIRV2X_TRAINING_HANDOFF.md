# DAIR-V2X single-agent PointPillars training handoff

This file is intended for the next computer/LLM.

## Current storage layout

The dataset is on the external Windows `D:` drive, mounted in WSL as `/mnt/d`:

```text
/mnt/d/dair_coalign/raw/cooperative-vehicle-infrastructure
/mnt/d/dair_coalign/coalign_assets
```

The drive is currently disconnected (`/mnt/d: No such device`). No data was deleted.

The code and Python environment are on the Linux machine, not the drive:

```text
/home/riley/CoAlign                         # CoAlign/OpenCOOD code
/home/riley/pc-occlusion-3                  # project code
/home/riley/miniconda3/envs/openpcdet       # Python environment
```

The current DAIR dataset overlay is in:

```text
/home/riley/pc-occlusion-3/data/coalign_full
```

Its large directories are symlinks to `/mnt/d/dair_coalign/...`; they must be recreated or copied when moving to another computer.

## Training status

The attempted single-agent run stopped during epoch 0 because this vehicle point cloud is corrupt:

```text
vehicle-side/velodyne/006451.pcd
```

The error is an LZF compressed-P​​CD decompression failure:

```text
ValueError: error in compressed data
```

The run reached batch 1085/1202 and produced no usable model checkpoint. Prefer restoring/re-downloading this file. If that is not possible, explicitly exclude it from a copied training split; do not silently catch all loader errors.

## Required code/config

Use the existing PointPillars single-agent config:

```text
/home/riley/CoAlign/opencood/hypes_yaml/dairv2x/lidar_only_with_noise/pointpillar_single.yaml
```

The project created a path-adjusted version with batch size 4 and 15 epochs:

```text
/home/riley/pc-occlusion-3/data/dairv2x_pointpillar_single_train.yaml
```

The training command uses the existing CoAlign script; no new training script is required:

```bash
cd /path/to/CoAlign
PYTHONPATH=/path/to/CoAlign \
/path/to/conda/envs/openpcdet/bin/python -u \
/path/to/CoAlign/opencood/tools/train.py \
-y /path/to/dairv2x_pointpillar_single_train.yaml \
-f late
```

The config must use absolute paths for the new computer's dataset root, `train.json`, and `val.json`.

## Make a filtered training split

Run this from the CoAlign environment after setting `ROOT` to the local DAIR overlay. It validates vehicle PCDs referenced by the training split and writes a new split that excludes only unreadable files.

```bash
ROOT=/path/to/coalign_full python - <<'PY'
import json, os
from pathlib import Path
from pypcd import pypcd

root = Path(os.environ['ROOT'])
ids = json.loads((root / 'train.json').read_text())
rows = json.loads((root / 'cooperative/data_info.json').read_text())
by_id = {Path(r['vehicle_pointcloud_path']).stem: r for r in rows}
bad = []
for frame_id in ids:
    row = by_id[frame_id]
    path = root / row['vehicle_pointcloud_path']
    try:
        pypcd.PointCloud.from_path(str(path))
    except Exception as exc:
        bad.append((frame_id, str(path), repr(exc)))

filtered = [frame_id for frame_id in ids if frame_id not in {x[0] for x in bad}]
(root / 'train_single_filtered.json').write_text(json.dumps(filtered, indent=2) + '\n')
print('original:', len(ids))
print('bad:', len(bad))
for item in bad:
    print('BAD:', item)
print('filtered:', len(filtered))
PY
```

Set the single-training config's `root_dir` to:

```text
/path/to/coalign_full/train_single_filtered.json
```

Keep the official `train.json` unchanged. Record the excluded IDs in the experiment notes.

## Moving the setup to the external drive

When `D:` is reconnected, copy the code/config and export the environment. A conda environment is not guaranteed to be portable across computers with different CUDA/GPU drivers, so prefer an environment export over copying it byte-for-byte:

```bash
# On the current computer, after reconnecting the drive
conda env export -n openpcdet --no-builds > /mnt/d/dair_coalign/openpcdet-environment.yml
rsync -a --exclude='__pycache__' /home/riley/CoAlign/ /mnt/d/dair_coalign/CoAlign/
rsync -a /home/riley/pc-occlusion-3/dairv2x/ /mnt/d/dair_coalign/pc-occlusion-3/dairv2x/
cp /home/riley/pc-occlusion-3/data/dairv2x_pointpillar_single_train.yaml \
   /mnt/d/dair_coalign/dairv2x_pointpillar_single_train.yaml
cp /home/riley/pc-occlusion-3/DAIRV2X_TRAINING_HANDOFF.md \
   /mnt/d/dair_coalign/DAIRV2X_TRAINING_HANDOFF.md
```

On the other computer, install a compatible Python/PyTorch/CUDA/spconv environment, clone/copy CoAlign, recreate the overlay symlinks or use direct paths, then follow the filtered-split and training commands above.

Do not copy the existing `/home/riley/miniconda3/envs/openpcdet` directory blindly unless the other computer has a compatible Linux/CUDA setup. The current environment was:

```text
Python 3.8
PyTorch 2.4.1 + CUDA 12.1
spconv 2.3.6
GPU tested: RTX 4060 Laptop, 8 GB VRAM
```
