"""Ego-only PointPillars baseline over a DAIR-V2X CoAlign split.

Run (CoAlign environment, external DAIR drive mounted):
    PYTHONPATH=. ~/miniconda3/envs/openpcdet/bin/python -m dairv2x.pointpillar_inference

The checkpoint (trained with CoAlign's late-fusion recipe) is a single-agent
PointPillars network. Only the vehicle LiDAR is run, through CoAlign's own
`inference_no_fusion`; there is no collaboration.

Ground truth (--gt):
  cooperative  CoAlign's merged vehicle+RSU labels in the evaluation range
               (default; includes objects the ego cannot see at all)
  vehicle      the vehicle's own labels only (CoAlign's "single" setting)

Every GT box gets a detected/missed outcome at each IoU threshold. One JSON
line per frame is appended, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from dairv2x.coalign import iou_3d, match_boxes

COALIGN_REPO = Path("/home/riley/CoAlign")
CHECKPOINT = Path("data/checkpoints/pointpillars-late-fusion")
DATA_ROOT = Path("data/coalign_full")
OUTPUT = Path("data/pointpillar_ego/val_{gt}_gt.jsonl")
IOU_THRESHOLDS = (0.5, 0.7)


def array(tensor, shape):
    return tensor.detach().cpu().numpy() if tensor is not None else np.empty(shape)


def frame_record(frame_id, gt, boxes, scores):
    """GT and predicted corners (vehicle LiDAR frame) and per-GT matches per IoU."""
    iou = iou_3d(gt, boxes)
    return {"vehicle_frame_id": frame_id, "gt_corners": gt.tolist(),
            "pred_corners": boxes.tolist(), "pred_scores": scores.tolist(),
            "matches": {str(t): match_boxes(iou, scores, t) for t in IOU_THRESHOLDS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--coalign-repo", type=Path, default=COALIGN_REPO)
    parser.add_argument("--gt", choices=("cooperative", "vehicle"), default="cooperative")
    parser.add_argument("--output", type=Path, help=f"default {OUTPUT}")
    parser.add_argument("--limit", type=int, help="only the first N frames (smoke test)")
    args = parser.parse_args()
    checkpoint, root = args.checkpoint.resolve(), args.root.resolve()
    output = (args.output or Path(str(OUTPUT).format(gt=args.gt))).resolve()
    sys.path.insert(0, str(args.coalign_repo.resolve()))

    import torch
    from torch.utils.data import DataLoader, Subset
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils

    hypes = yaml_utils.load_yaml(str(checkpoint / "config.yaml"))
    if hypes["fusion"]["core_method"] != "late" or hypes["fusion"]["dataset"] != "dairv2x":
        raise ValueError("Expected a DAIR-V2X single-agent (late-recipe) checkpoint config")
    hypes["data_dir"] = str(root)
    hypes["validate_dir"] = str(root / f"{args.split}.json")

    np.random.seed(303)  # CoAlign's inference seed (pose noise is off in this config)
    dataset = build_dataset(hypes, visualize=False, train=False)
    done = set()
    if output.is_file():
        done = {json.loads(line)["vehicle_frame_id"] for line in output.read_text().splitlines()}
    todo = [i for i, frame in enumerate(dataset.split_info) if frame not in done]
    if args.limit is not None:
        todo = [i for i in todo if i < args.limit]
    print(f"{len(dataset.split_info)} frames in {args.split}; {len(done)} done; {len(todo)} to run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    epoch, model = train_utils.load_saved_model(str(checkpoint), model)
    model.to(device).eval()
    print(f"Loaded epoch {epoch} on {device}")

    loader = DataLoader(Subset(dataset, todo), batch_size=1, num_workers=4, shuffle=False,
                        collate_fn=dataset.collate_batch_test)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.coalign_repo)  # CoAlign resolves some relative paths from its checkout
    with output.open("a") as stream:
        for n, (index, batch) in enumerate(zip(todo, loader), 1):
            frame_id = dataset.split_info[index]
            if batch is None:
                print(f"skip {frame_id}: CoAlign returned no sample")
                continue
            batch = train_utils.to_device(batch, device)
            with torch.no_grad():
                out = inference_utils.inference_no_fusion(
                    batch, model, dataset, single_gt=args.gt == "vehicle")
            record = frame_record(frame_id, array(out["gt_box_tensor"], (0, 8, 3)),
                                  array(out["pred_box_tensor"], (0, 8, 3)),
                                  array(out["pred_score"], (0,)))
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} frames", flush=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
