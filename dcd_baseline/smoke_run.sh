#!/usr/bin/env bash
# End-to-end smoke run of the DCD baseline on the first 30 KITTI val frames.
set -euo pipefail
cd ~/OpenPCDet
PY=~/miniconda3/envs/openpcdet/bin/python
BASE=dcd_baseline/results/smoke

$PY - <<'EOF'
import csv, os
frames = set(open('data/kitti/ImageSets/val.txt').read().split()[:30])
rows = list(csv.DictReader(open('runs/gilroy_lidar_replication/natural/val/instances.csv')))
keep = [r for r in rows if r['frame_id'] in frames]
os.makedirs('dcd_baseline/results/smoke', exist_ok=True)
with open('dcd_baseline/results/smoke/instances_smoke.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
print('smoke instances:', len(keep))
EOF

$PY dcd_baseline/compute_dcd.py \
  --instances $BASE/instances_smoke.csv \
  --output-dir $BASE/natural

$PY dcd_baseline/infer_natural.py \
  --detector pointpillar \
  --model-config tools/cfgs/kitti_models/pointpillar.yaml \
  --checkpoint checkpoints/pointpillar.pth \
  --max-frames 30 --expect-class Pedestrian \
  --output $BASE/detections/pointpillar.json

$PY dcd_baseline/evaluate_dcd.py \
  --instances $BASE/natural/dcd_instances.csv \
  --predictions $BASE/detections/pointpillar.json \
  --detector pointpillar \
  --output-dir $BASE/evaluation

$PY dcd_baseline/plot_dcd.py \
  --result pointpillar=$BASE/evaluation/pointpillar_dcd_bin_metrics.csv \
  --binning pointpillar=$BASE/evaluation/pointpillar_binning.json \
  --output-dir $BASE/plots

echo "SMOKE OK"
