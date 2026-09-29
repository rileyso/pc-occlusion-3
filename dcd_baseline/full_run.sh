#!/usr/bin/env bash
# Full DCD baseline: per-object DCD on the natural KITTI pedestrian benchmark,
# then inference + DCD-binned evaluation for each ready detector, then plots.
set -euo pipefail
cd ~/OpenPCDet
PY=~/miniconda3/envs/openpcdet/bin/python
BASE=dcd_baseline/results

echo "=== [1/3] per-object DCD over the full natural benchmark ==="
$PY dcd_baseline/compute_dcd.py \
  --instances runs/gilroy_lidar_replication/natural/val/instances.csv \
  --output-dir $BASE/natural

echo "=== [2/3] inference + evaluation per detector ==="
detectors=(pointpillar second pointrcnn pv_rcnn)
configs=(pointpillar.yaml second.yaml pointrcnn.yaml pv_rcnn.yaml)
ckpts=(pointpillar.pth second.pth pointrcnn.pth pv_rcnn_8369.pth)
for i in "${!detectors[@]}"; do
  d=${detectors[$i]}
  echo "--- $d inference ---"
  $PY dcd_baseline/infer_natural.py --detector "$d" \
    --model-config "tools/cfgs/kitti_models/${configs[$i]}" \
    --checkpoint "checkpoints/${ckpts[$i]}" \
    --expect-class Pedestrian \
    --output "$BASE/detections/${d}_val.json"
  echo "--- $d evaluation ---"
  $PY dcd_baseline/evaluate_dcd.py \
    --instances "$BASE/natural/dcd_instances.csv" \
    --predictions "$BASE/detections/${d}_val.json" \
    --detector "$d" \
    --output-dir "$BASE/evaluation"
done

echo "=== [3/3] plots ==="
PLOT_ARGS=()
for d in "${detectors[@]}"; do
  PLOT_ARGS+=(--result "$d=$BASE/evaluation/${d}_dcd_bin_metrics.csv" --binning "$d=$BASE/evaluation/${d}_binning.json")
done
$PY dcd_baseline/plot_dcd.py "${PLOT_ARGS[@]}" --output-dir "$BASE/plots"

echo "FULL RUN COMPLETE"
