"""Per-box record dumps and partial-result checkpoints for resumable runs."""

import os
import pickle
from types import SimpleNamespace

RECORD_FIELDS = ("frame", "object_index", "class_name", "difficulty", "num_points",
                 "pixel_height", "visibility_score", "total_cells",
                 "total_occupied_cells", "x", "y", "z", "length", "width", "height", "yaw")


def to_records(boxes):
    """Per-box scalars only; the per-box cell grids are far too large to pickle."""
    return [{name: getattr(box, name) for name in RECORD_FIELDS} for box in boxes]


def dump_boxes(boxes, path):
    with open(path, "wb") as file:
        pickle.dump(to_records(boxes), file)
    print(f"saved {len(boxes)} boxes to {path}")


def save_checkpoint(path, root, total_frames, next_idx, broken, boxes):
    """Write atomically so a kill mid-write never leaves a corrupt checkpoint."""
    state = {"root": str(root), "total_frames": total_frames, "next_idx": next_idx,
             "broken": list(broken), "records": to_records(boxes)}
    with open(f"{path}.tmp", "wb") as file:
        pickle.dump(state, file)
    os.replace(f"{path}.tmp", path)
    print(f"checkpoint: {next_idx}/{total_frames} frames, {len(boxes)} boxes")


def load_checkpoint(path, root, total_frames):
    """Return (boxes, broken, next_idx); boxes are attribute-style records.

    A checkpoint from a different root or frame count is ignored.
    """
    if not os.path.exists(path):
        return [], [], 0
    try:
        with open(path, "rb") as file:
            state = pickle.load(file)
        if state["root"] != str(root) or state["total_frames"] != total_frames:
            print(f"ignoring checkpoint {path}: different root or frame count")
            return [], [], 0
        boxes = [SimpleNamespace(**record) for record in state["records"]]
        print(f"resuming from frame {state['next_idx']}/{total_frames} "
              f"({len(boxes)} boxes, {len(state['broken'])} broken so far)")
        return boxes, state["broken"], state["next_idx"]
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, TypeError) as error:
        print(f"ignoring unreadable checkpoint {path}: {type(error).__name__}: {error}")
        return [], [], 0
