"""PointPillars inference over a DAIR-V2X-V split (default: val).

Run in the mmdet3d environment:
    python dairv2xv/inference.py [vehicle-side-root] [--split val] [--fresh] [--limit N]

Uses the official single-vehicle PointPillars checkpoint in dairv2xv/checkpoints
(trained with mmdet3d 0.14) through mmdet3d 1.x's matching KITTI 3-class config.
Detections are boxes in vehicle LiDAR axes with z at the BOTTOM face (mmdet3d
convention); the DAIR labels may use the box centre. The detector only sees
x in [0, 69.12] m and |y| <= 39.68 m, the range it was trained on.

Saves inference_<split>.pkl next to the script: one dict per detection with
frame_id (the file stem, e.g. "000012"), x, y, z, length, width, height, yaw,
score and label. Frames with no detections have no rows. An interrupted run
resumes from inference_<split>.ckpt.pkl; --fresh starts over.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from explorer import DAIRV2XVehicleExplorer
from log_tee import tee_to_log

parser = argparse.ArgumentParser()
parser.add_argument("root", nargs="?", default="/mnt/d/dairv2x_v_full/vehicle-side",
                    help="vehicle-side folder")
parser.add_argument("--split", default="val")
parser.add_argument("--limit", type=int, help="only the first N frames (smoke test)")
parser.add_argument("--fresh", action="store_true",
                    help="ignore any saved checkpoint and start over")
cli = parser.parse_args()
ROOT, SPLIT, LIMIT, FRESH = cli.root, cli.split, cli.limit, cli.fresh
CHECKPOINT_EVERY = 200  # frames between partial-result saves
INTENSITY_SCALE = 1 / 255  # the DAIR->KITTI conversion stores intensity in [0, 1]

HERE = Path(__file__).parent
WEIGHTS = next((HERE / "checkpoints").glob("sv3d_veh_pointpillars_*.pth"))
CONFIG = "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
CLASSES = ("Pedestrian", "Cyclist", "Car")  # label index order of the checkpoint
TAG = f"{SPLIT}_limit{LIMIT}" if LIMIT else SPLIT  # a smoke test never overwrites a full run
OUTPUT = HERE / f"inference_{TAG}.pkl"
PARTIAL = HERE / f"inference_{TAG}.ckpt.pkl"


def save_partial(next_i, total, records):
    """Atomic, so a kill mid-write never leaves a corrupt checkpoint."""
    state = {"root": ROOT, "total": total, "next_i": next_i, "records": records}
    with open(f"{PARTIAL}.tmp", "wb") as file:
        pickle.dump(state, file)
    os.replace(f"{PARTIAL}.tmp", PARTIAL)
    print(f"checkpoint: {next_i}/{total} frames, {len(records)} detections")


def load_partial(total):
    if FRESH or not PARTIAL.exists():
        return [], 0
    try:
        with open(PARTIAL, "rb") as file:
            state = pickle.load(file)
        if state["root"] != ROOT or state["total"] != total:
            print(f"ignoring checkpoint {PARTIAL.name}: different root or frame count")
            return [], 0
        print(f"resuming from frame {state['next_i']}/{total} "
              f"({len(state['records'])} detections)")
        return state["records"], state["next_i"]
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError) as error:
        print(f"ignoring unreadable checkpoint {PARTIAL.name}: {type(error).__name__}: {error}")
        return [], 0


def find_config():
    import mmdet3d
    root = Path(mmdet3d.__file__).parent
    for base in (root / ".mim" / "configs", root.parent / "configs"):
        found = list(base.rglob(CONFIG))
        if found:
            return found[0]
    raise FileNotFoundError(f"{CONFIG} not found in the installed mmdet3d")


def load_legacy_first_layer(model):
    """Adapt the 9-feature first layer of the mmdet3d 0.14 checkpoint.

    mmdet3d 1.x feeds the pillar net 10 features (a z offset is added), and its
    legacy path also rewrites the raw z column as z - z_offset. The checkpoint
    expects raw z and no z offset, so: add a zero weight column for the extra
    feature, and fold the constant z shift into the batch-norm running mean.
    The result equals the original 9-feature layer.
    """
    import torch
    layer = model.voxel_encoder.pfn_layers[0]
    weight = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["state_dict"][
        "voxel_encoder.pfn_layers.0.linear.weight"]
    encoder = model.voxel_encoder
    z_shift = -encoder.z_offset  # the raw z column becomes z + z_shift
    with torch.no_grad():
        layer.linear.weight.copy_(torch.cat([weight, torch.zeros_like(weight[:, :1])], dim=1))
        layer.norm.running_mean += weight[:, 2].to(layer.norm.running_mean) * z_shift


def main():
    tee_to_log(HERE / "inference.log")
    data = DAIRV2XVehicleExplorer(ROOT)
    split_ids = json.loads((Path(ROOT) / "split_data.json").read_text())[SPLIT]
    index_of = {Path(f["pointcloud_path"]).stem: i for i, f in enumerate(data.frames)}
    frames = [(fid, index_of[fid]) for fid in split_ids if fid in index_of]
    if LIMIT is not None:
        frames = frames[:LIMIT]
    total = len(frames)
    print(f"{SPLIT} split: {total} of {len(split_ids)} frames in data_info.json")

    from mmdet3d.apis import inference_detector, init_model
    model = init_model(str(find_config()), str(WEIGHTS), device="cuda:0")
    load_legacy_first_layer(model)
    print(f"loaded {WEIGHTS.name}")

    records, start = load_partial(total)
    broken = []
    for i in range(start, total):
        if i > start and i % CHECKPOINT_EVERY == 0:
            save_partial(i, total, records)
        frame_id, idx = frames[i]
        try:
            points, _ = data.return_agent_point_cloud(idx)
            points = points.astype(np.float32)
            points[:, 3] *= INTENSITY_SCALE
            result, _ = inference_detector(model, points)
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.cpu().numpy()
            scores = pred.scores_3d.cpu().numpy()
            labels = pred.labels_3d.cpu().numpy()
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            broken.append(frame_id)
            print(f"frame: {i + 1}/{total} skipped ({type(error).__name__}: {error})")
            continue
        for box, score, label in zip(boxes, scores, labels):
            records.append({"frame_id": frame_id, "x": float(box[0]), "y": float(box[1]),
                            "z": float(box[2]), "length": float(box[3]),
                            "width": float(box[4]), "height": float(box[5]),
                            "yaw": float(box[6]), "score": float(score),
                            "label": CLASSES[int(label)]})
        print(f"frame: {i + 1}/{total} detections: {len(boxes)}")
    print(f"skipped {len(broken)} broken frames")
    with open(OUTPUT, "wb") as file:
        pickle.dump(records, file)
    print(f"saved {len(records)} detections to {OUTPUT}")
    PARTIAL.unlink(missing_ok=True)  # finished: a rerun starts clean


if __name__ == "__main__":
    main()
