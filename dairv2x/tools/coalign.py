"""Run one DAIR-V2X-C pair through the official CoAlign checkpoint.

Run ``python -m dairv2x.coalign --help`` from this repository's root.
CoAlign and its checkpoints/data are external; see dairv2x/README.md.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def match_boxes(iou, scores, threshold):
    """One-to-one, confidence-ordered matches; one result per ground-truth box."""
    matches = [None] * len(iou)
    used = set()
    for pred in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
        candidates = [(row[pred], gt) for gt, row in enumerate(iou)
                      if gt not in used and row[pred] >= threshold]
        if candidates:
            overlap, gt = max(candidates)
            matches[gt] = {"prediction_index": pred, "confidence": float(scores[pred]),
                           "iou_3d": float(overlap)}
            used.add(gt)
    return matches


def iou_3d(gt_boxes, pred_boxes):
    """Oriented 3D IoU from CoAlign's box corners (N, 8, 3)."""
    from shapely.geometry import Polygon

    result = []
    for gt in gt_boxes:
        a = Polygon(gt[:4, :2])
        a_low, a_high = gt[:, 2].min(), gt[:, 2].max()
        row = []
        for pred in pred_boxes:
            b = Polygon(pred[:4, :2])
            b_low, b_high = pred[:, 2].min(), pred[:, 2].max()
            intersection = a.intersection(b).area * max(0, min(a_high, b_high) - max(a_low, b_low))
            union = a.area * (a_high - a_low) + b.area * (b_high - b_low) - intersection
            row.append(float(intersection / union) if union > 0 else 0.0)
        result.append(row)
    return result


def infer(model_dir, root, split, frame_id, method, stage1_path, threshold):
    return next(infer_frames(model_dir, root, split, [frame_id], method, stage1_path, threshold))


def infer_frames(model_dir, root, split, frame_ids, method, stage1_path, threshold):
    """Yield per-frame results while loading the dataset and model only once."""
    import numpy as np
    import torch
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import inference_utils, train_utils

    model_dir = Path(model_dir).resolve()
    if not (model_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"CoAlign checkpoint directory needs config.yaml: {model_dir}")
    hypes = yaml_utils.load_yaml(str(model_dir / "config.yaml"))
    hypes["data_dir"] = str(root)
    hypes["validate_dir"] = str(root / f"{split}.json")
    if not Path(hypes["validate_dir"]).is_file():
        raise FileNotFoundError(hypes["validate_dir"])
    if hypes["fusion"]["dataset"] != "dairv2x":
        raise ValueError("Checkpoint config must target DAIR-V2X")
    expected = "intermediate" if method == "coalign" else "late"
    if hypes["fusion"]["core_method"] != expected:
        raise ValueError(f"{method} requires a {expected}-fusion checkpoint, not {hypes['fusion']['core_method']}")

    if "box_align" in hypes:
        config_path = hypes["box_align"][f"{split}_result"]
        alignment = Path(stage1_path or config_path).resolve()
        if not alignment.is_file():
            raise FileNotFoundError(f"First-stage box-alignment predictions missing: {alignment}. "
                                    "Supply --stage1 or use a matching no-box-alignment checkpoint.")
        hypes["box_align"]["val_result"] = str(alignment)

    np.random.seed(303)  # match CoAlign's inference seed when pose noise is enabled
    dataset = build_dataset(hypes, visualize=False, train=False)
    indices = {frame_id: index for index, frame_id in enumerate(dataset.split_info)}
    for frame_id in frame_ids:
        if frame_id not in indices:
            raise ValueError(f"Vehicle frame {frame_id} is not in {split}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model.to(device).eval()
    def array(tensor, shape):
        return tensor.detach().cpu().numpy() if tensor is not None else np.empty(shape)

    for frame_id in frame_ids:
        np.random.seed(303)  # Match independent one-frame runs; also makes resumes reproducible.
        sample = dataset[indices[frame_id]]  # preserve the full split's stage-one indexing
        if sample is None:
            raise ValueError(f"CoAlign could not load frame {frame_id}")
        batch = train_utils.to_device(dataset.collate_batch_test([sample]), device)
        with torch.no_grad():
            result = (inference_utils.inference_intermediate_fusion(batch, model, dataset)
                      if method == "coalign" else
                      inference_utils.inference_no_fusion(batch, model, dataset))
        boxes = array(result["pred_box_tensor"], (0, 8, 3))
        scores = array(result["pred_score"], (0,))
        gt = array(result["gt_box_tensor"], (0, 8, 3))
        yield {"gt_boxes": gt, "pred_boxes": boxes, "scores": scores,
               "matches": match_boxes(iou_3d(gt, boxes), scores, threshold)}
        del batch, result


def result_record(pair_index, pair, split, threshold, fused, single=None, prediction_key="coalign"):
    """Build a result record for collaborative, vehicle-only, or both runs."""
    if prediction_key not in ("coalign", "vehicle_only"):
        raise ValueError(f"Unknown prediction key: {prediction_key}")
    if single is not None:
        single["matches"] = match_boxes(iou_3d(fused["gt_boxes"], single["pred_boxes"]),
                                         single["scores"], threshold)
    primary_key = prediction_key
    primary = fused
    output = {
        "pair_index": pair_index,
        "vehicle_frame_id": Path(pair["vehicle_pointcloud_path"]).stem,
        "infrastructure_pointcloud_path": pair["infrastructure_pointcloud_path"],
        "split": split, "iou_threshold": threshold,
        "ground_truth": [
            {"index": i, "corners_vehicle_lidar": box.tolist(),
             primary_key: primary["matches"][i],
             **({"vehicle_only": single["matches"][i]} if single is not None else {})}
            for i, box in enumerate(primary["gt_boxes"])
        ],
        "predictions": {primary_key: [
            {"corners_vehicle_lidar": box.tolist(), "confidence": float(score)}
            for box, score in zip(primary["pred_boxes"], primary["scores"])]},
    }
    if single is not None:
        output["predictions"]["vehicle_only"] = [
            {"corners_vehicle_lidar": box.tolist(), "confidence": float(score)}
            for box, score in zip(single["pred_boxes"], single["scores"])]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coalign", required=True, type=Path, help="downloaded CoAlign checkpoint directory (config.yaml + weights)")
    parser.add_argument("--single", type=Path, help="optional matching DAIR vehicle-only PointPillars checkpoint directory")
    parser.add_argument("--root", required=True, type=Path, help="CoAlign-formatted cooperative-vehicle-infrastructure directory")
    parser.add_argument("--coalign-repo", required=True, type=Path, help="clone of yifanlu0227/CoAlign")
    parser.add_argument("--pair-index", type=int, default=0, help="index in cooperative/data_info.json sorted by vehicle pointcloud path")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val",
                        help="CoAlign split containing the frame; test requires a test.json")
    parser.add_argument("--stage1", type=Path, help="precomputed first-stage box-alignment JSON (if not at checkpoint config path)")
    parser.add_argument("--iou", type=float, default=0.5, help="3D IoU needed to match a GT box")
    parser.add_argument("--output", type=Path, help="write JSON (otherwise print to stdout)")
    args = parser.parse_args()
    if not 0 <= args.iou <= 1:
        parser.error("--iou must be between 0 and 1")
    root, repo = args.root.resolve(), args.coalign_repo.resolve()
    if not (repo / "opencood").is_dir():
        parser.error(f"Not a CoAlign checkout: {repo}")
    pairs = sorted(json.loads((root / "cooperative/data_info.json").read_text()),
                   key=lambda pair: pair["vehicle_pointcloud_path"])
    if not 0 <= args.pair_index < len(pairs):
        parser.error(f"--pair-index must be between 0 and {len(pairs) - 1}")
    pair = pairs[args.pair_index]
    frame_id = Path(pair["vehicle_pointcloud_path"]).stem
    coalign_dir = args.coalign.resolve()
    single_dir = args.single.resolve() if args.single else None
    stage1_path = args.stage1.resolve() if args.stage1 else None
    output_path = args.output.resolve() if args.output else None
    sys.path.insert(0, str(repo))
    os.chdir(repo)  # CoAlign resolves its config's stage-1 paths relative to its checkout
    fused = infer(coalign_dir, root, args.split, frame_id, "coalign", stage1_path, args.iou)
    single = (infer(single_dir, root, args.split, frame_id, "single", None, args.iou)
              if single_dir else None)
    output = result_record(args.pair_index, pair, args.split, args.iou, fused, single)
    text = json.dumps(output, indent=2)
    if output_path:
        output_path.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
