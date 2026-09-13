# PC Occlusion 3

Research code for point-cloud occlusion analysis built on
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This repository contains the analysis code, experiment runners, tests, and
research documentation. OpenPCDet is a separate dependency; datasets and
model checkpoints are not included.

## Layout

```text
/path/to/
├── OpenPCDet/          # OpenPCDet clone or fork
└── pc-occlusion-3/     # this repository
```

## Setup

Set the OpenPCDet checkout explicitly, then install the combined dependencies:

```bash
export OPENPCDET_ROOT=/path/to/OpenPCDet
export ANALYSIS_ROOT=/path/to/pc-occlusion-3
cd "$OPENPCDET_ROOT"
pip install -r "$ANALYSIS_ROOT/requirements.txt"
export PYTHONPATH="$ANALYSIS_ROOT"
```

Prepare datasets using OpenPCDet's instructions. KITTI should be available at:

```text
$OPENPCDET_ROOT/data/kitti/
├── training/
├── testing/
└── ImageSets/
```

Datasets and checkpoints are intentionally excluded from Git. Generate
OpenPCDet's info/database files as described in
`$OPENPCDET_ROOT/docs/GETTING_STARTED.md`.

## Run an example

Run commands from the OpenPCDet root so its configs, loaders, and CUDA
extensions are available:

```bash
PYTHONPATH="$ANALYSIS_ROOT" python "$ANALYSIS_ROOT/research_pipeline/main_pipeline.py" \
  --dataset kitti --split val_500 \
  --output-dir "$ANALYSIS_ROOT/runs/kitti_baseline"
```

Outputs belong in `runs/`, which is ignored. Record the OpenPCDet commit,
configuration, checkpoint, dataset preparation, and environment when sharing
results.
