"""Evaluate the example dataset: python -m dairv2x.visibility_dataset."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dairv2x.visibility_frame import (
    DATASET_ROOT, METRIC_NAME, SAMPLES_PER_EDGE,
    DAIRV2XExplorer, solid_angle_visibility,
)

OUTPUT = Path("data/visibility_example")


def main():
    dataset = DAIRV2XExplorer(DATASET_ROOT)
    rows = []
    for pair_index, pair in enumerate(dataset.pairs):
        frame = dataset.return_labeled_sync_time_ids_of_seq(pair_index)[0]
        boxes = dataset.return_gt_boxes_in_map(pair_index, frame)
        scores, ranges = {}, {}
        for agent in ("infrastructure", "vehicle"):
            origin = dataset.return_map_se3_agent(pair_index, agent, frame)[:3, 3]
            scores[agent] = solid_angle_visibility(boxes, origin, SAMPLES_PER_EDGE)
            ranges[agent] = np.linalg.norm(boxes[:, :3] - origin, axis=1)
        for i, box in enumerate(boxes):
            row = {
                "pair_index": pair_index,
                "vehicle_frame": Path(pair["vehicle_pointcloud_path"]).stem,
                "rsu_frame": Path(pair["infrastructure_pointcloud_path"]).stem,
                "object_index": int(box[8]),
            }
            for agent, name in (("infrastructure", "rsu"), ("vehicle", "vehicle")):
                visibility = scores[agent][i]
                row[f"{name}_visibility"] = visibility
                row[f"{name}_occlusion"] = 1 - visibility
                row[f"{name}_range_m"] = ranges[agent][i]
            rows.append(row)
        print(f"Pair {pair_index + 1}/{len(dataset.pairs)}: {len(boxes)} objects", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "objects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rsu = np.array([r["rsu_occlusion"] for r in rows])
    vehicle = np.array([r["vehicle_occlusion"] for r in rows])
    valid = np.isfinite(rsu) & np.isfinite(vehicle)
    assert np.all((rsu[np.isfinite(rsu)] >= 0) & (rsu[np.isfinite(rsu)] <= 1))
    assert np.all((vehicle[np.isfinite(vehicle)] >= 0) & (vehicle[np.isfinite(vehicle)] <= 1))
    lines = [
        "# DAIR-V2X example: solid-angle occlusion", "",
        f"Metric: {METRIC_NAME}; occlusion = 1 - visibility.",
        f"Dataset: `{DATASET_ROOT}`.",
        f"Pairs: {len(dataset.pairs)}; object-frame observations: {len(rows)}; "
        f"sensor-object observations: {2 * len(rows)}.",
        f"Sampling: {SAMPLES_PER_EDGE} x {SAMPLES_PER_EDGE} midpoints per facing face.",
        "", "| Sensor | Valid observations | Mean occlusion | Median occlusion | Occlusion >= 0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in (("RSU", rsu), ("Vehicle", vehicle)):
        finite = values[np.isfinite(values)]
        lines.append(f"| {name} | {len(finite)} | {finite.mean():.4f} | "
                     f"{np.median(finite):.4f} | {int((finite >= 0.5).sum())} |")
    lines += ["", f"Both viewpoints valid: {int(valid.sum())}.",
              "Illustrative categories (0.5 is a descriptive threshold, not validated):"]
    for name, mask in (
        ("Vehicle occlusion >= 0.5, RSU occlusion < 0.5", (vehicle >= 0.5) & (rsu < 0.5)),
        ("RSU occlusion >= 0.5, vehicle occlusion < 0.5", (rsu >= 0.5) & (vehicle < 0.5)),
        ("Both occlusion >= 0.5", (rsu >= 0.5) & (vehicle >= 0.5)),
        ("Both occlusion < 0.5", (rsu < 0.5) & (vehicle < 0.5)),
    ):
        lines.append(f"- {name}: {int((mask & valid).sum())}")
    lines += ["", "## Interpretation and limitations",
              "These are repeated object-frame observations, not unique objects or independent scenes.",
              "Object indices refer to each frame's cooperative label order, not tracking IDs.",
              "Scores use labelled boxes only, with centre-distance ordering and no background, FOV, or range filtering.",
              "Sensor-containing boxes are NaN and excluded as blockers. All geometry is in world coordinates.",
              "Scores approximate spherical area by numerical quadrature, not measured LiDAR return coverage.",
              "No detector inference was run; viewpoint differences indicate geometric opportunity, not detection recovery.",
              "No point counts or class-stratified/confounder-controlled analysis is included in this initial sweep.",
              "", "Reproduce: `conda activate openpcdet && python -m dairv2x.visibility_dataset`."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, 1, 21)
    axes[0].hist([rsu[np.isfinite(rsu)], vehicle[np.isfinite(vehicle)]],
                 bins=bins, label=["RSU", "Vehicle"])
    axes[0].set(xlabel="Occlusion (1 - visibility)", ylabel="Object-frame observations")
    axes[0].legend()
    axes[1].scatter(vehicle[valid], rsu[valid], s=12, alpha=0.3)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].axhline(0.5, color="grey", linewidth=0.5)
    axes[1].axvline(0.5, color="grey", linewidth=0.5)
    axes[1].set(xlabel="Vehicle occlusion", ylabel="RSU occlusion",
                xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    fig.suptitle(f"DAIR-V2X example: {len(dataset.pairs)} pairs, {len(rows)} object-frame observations")
    fig.tight_layout()
    fig.savefig(OUTPUT / "occlusion.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines))
    print(f"\nSaved CSV, summary and plot to {OUTPUT}")


if __name__ == "__main__":
    main()
