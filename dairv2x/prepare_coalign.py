"""Build a CoAlign-readable overlay for the DAIR-V2X-C example, without changing the source.

python -m dairv2x.prepare_coalign --root /path/to/dair-v2x-c-example \
  --annotations /path/to/DAIR-V2X-C_Complemented_Anno.zip --output data/coalign_example
"""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile


def prepare(root, archive, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("Output must be outside the original dataset directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    pairs = json.loads((root / "cooperative/data_info.json").read_text())
    with ZipFile(archive) as zipped:
        prefix = "DAIR-V2X-C_Complemented_Anno/"
        train = json.loads(zipped.read(prefix + "train.json"))
        frame_ids = [Path(p["vehicle_pointcloud_path"]).stem for p in pairs]
        if not set(frame_ids).issubset(train):
            raise ValueError("Example frame IDs are not all in CoAlign's train split")
        output.mkdir(parents=True, exist_ok=True)
        (output / "train.json").write_bytes(zipped.read(prefix + "train.json"))
        (output / "val.json").write_bytes(zipped.read(prefix + "val.json"))
        for side in ("vehicle-side", "infrastructure-side"):
            target = output / side
            target.mkdir()
            for folder in ("calib", "velodyne", "image"):
                (target / folder).symlink_to(root / side / folder, target_is_directory=True)
        (output / "infrastructure-side/label").symlink_to(
            root / "infrastructure-side/label", target_is_directory=True)
        (output / "vehicle-side/label").mkdir()
        (output / "vehicle-side/label/camera").symlink_to(
            root / "vehicle-side/label/camera", target_is_directory=True)
        (output / "vehicle-side/label/lidar_backup").symlink_to(
            root / "vehicle-side/label/lidar", target_is_directory=True)
        (output / "vehicle-side/label/lidar").mkdir()
        (output / "cooperative").mkdir()
        (output / "cooperative/data_info.json").symlink_to(root / "cooperative/data_info.json")
        (output / "cooperative/label_world_backup").symlink_to(
            root / "cooperative/label_world", target_is_directory=True)
        (output / "cooperative/label_world").mkdir()
        for frame in frame_ids:
            for source, dest in (
                ("new_labels/cooperative_label/label_world", "cooperative/label_world"),
                ("new_labels/vehicle-side_label/lidar", "vehicle-side/label/lidar"),
            ):
                (output / dest / f"{frame}.json").write_bytes(
                    zipped.read(f"{prefix}{source}/{frame}.json"))
    return len(frame_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.root, args.annotations, args.output)} pairs at {args.output.resolve()}")


if __name__ == "__main__":
    main()
