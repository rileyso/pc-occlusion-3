#!/usr/bin/env bash
# Regenerate binned CSVs (post bin-assignment fix) and the curve figures.
set -euo pipefail
cd ~/OpenPCDet
PY=~/miniconda3/envs/openpcdet/bin/python
BASE=dcd_baseline/results

for d in pointpillar second pointrcnn pv_rcnn; do
  $PY dcd_baseline/evaluate_dcd.py \
    --instances $BASE/natural/dcd_instances.csv \
    --predictions $BASE/detections/${d}_val.json \
    --detector "$d" \
    --output-dir $BASE/evaluation 2>/dev/null | grep -E '^(bin|WROTE)'
done

$PY dcd_baseline/plot_dcd.py \
  --prediction pointpillar=$BASE/detections/pointpillar_val.json \
  --prediction second=$BASE/detections/second_val.json \
  --prediction pointrcnn=$BASE/detections/pointrcnn_val.json \
  --prediction pv_rcnn=$BASE/detections/pv_rcnn_val.json 2>&1 | grep -v -i warning | tail -12
echo "REGEN COMPLETE"
