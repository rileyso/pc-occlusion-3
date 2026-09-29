#!/usr/bin/env bash
# Regenerate artifacts so every file is self-describing: protocol.json carries
# the structured parameters block, binning.json carries dcd_parameters, and the
# outcome figure uses the shortened panel titles. Reuses cached detections.
set -euo pipefail
cd ~/OpenPCDet
PY=~/miniconda3/envs/openpcdet/bin/python
BASE=dcd_baseline/results

$PY dcd_baseline/compute_dcd.py \
  --instances runs/gilroy_lidar_replication/natural/val/instances.csv \
  --output-dir $BASE/natural

for d in pointpillar second pointrcnn pv_rcnn; do
  $PY dcd_baseline/evaluate_dcd.py \
    --instances $BASE/natural/dcd_instances.csv \
    --predictions $BASE/detections/${d}_val.json \
    --detector "$d" \
    --output-dir $BASE/evaluation
done

PLOT_ARGS=()
for d in pointpillar second pointrcnn pv_rcnn; do
  PLOT_ARGS+=(--result "$d=$BASE/evaluation/${d}_dcd_bin_metrics.csv" --binning "$d=$BASE/evaluation/${d}_binning.json")
done
$PY dcd_baseline/plot_dcd.py "${PLOT_ARGS[@]}" --output-dir $BASE/plots
echo "FIXUP COMPLETE"
