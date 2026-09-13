# PC Occlusion Analysis

Research code for the point-cloud occlusion analysis experiments built on
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

This repository contains the analysis and experiment code only. OpenPCDet is a
separate dependency, and datasets/checkpoints are not included.

## Layout

```text
/home/riley/
├── OpenPCDet/              # OpenPCDet clone/fork
└── pc-occlusion-analysis/  # this repository
```

The scripts currently expect to be run from the OpenPCDet root and to find the
research package at `research_workspace/`. Create a symlink once after cloning:

```bash
cd /path/to/OpenPCDet
ln -s ../pc-occlusion-analysis research_workspace
```

If the directory already exists, remove or rename it first. On Windows, use a
junction or run from an elevated shell:

```powershell
cmd /c mklink /J research_workspace ..\pc-occlusion-analysis
```

## Setup

Use the Python/PyTorch environment required by your OpenPCDet fork:

```bash
cd /path/to/OpenPCDet
pip install -r research_workspace/requirements.txt
export PYTHONPATH=research_workspace  # PowerShell: $env:PYTHONPATH='research_workspace'
```

Prepare the datasets using OpenPCDet's instructions. For KITTI, the expected
layout is:

```text
data/kitti/
├── training/
├── testing/
└── ImageSets/
```

The dataset files are intentionally excluded from Git. Download them from the
official source and generate OpenPCDet's info/database files as described in
`OpenPCDet/docs/GETTING_STARTED.md`.

## Run an example

From the OpenPCDet root:

```bash
PYTHONPATH=research_workspace python research_workspace/research_pipeline/main_pipeline.py \
  --dataset kitti --split val_500 \
  --output-dir research_workspace/runs/kitti_baseline
```

Outputs belong in `runs/`, which is ignored. Record the OpenPCDet commit,
configuration, checkpoint, dataset preparation, and environment when sharing
results.
