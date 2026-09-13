"""Grouping registry for coarse occlusion/difficulty baselines.

A Grouping maps the per-object label column of an instances CSV onto the
ordered evaluation groups used by the bar figures and per-group metrics. This
module is the extension seam for adding new occlusion metrics: a metric that
already exists as a per-object instances-CSV column needs only a new entry
here (or an ad-hoc Grouping built with :func:`from_column`), and the evaluator
(`evaluate_kitti_difficulty.py`) and plotter (`plot_kitti.py`) pick it up
unchanged. A new dataset needs only an instances CSV in the portable format
(frame_id, label_index, class_name, x/y/z/dx/dy/dz/heading + label columns)
and predictions JSONs ({frame_id, class_name, score, box_lidar}).

Two group views are evaluated for every grouping:

- exclusive: disjoint groups - each object belongs to exactly one group (its
  own intrinsic label). This is the view for the bar figures: "how do objects
  of this difficulty band actually behave".
- cumulative: the official benchmark reporting semantics - group d contains
  every object the official protocol counts at difficulty d (KITTI ignores
  objects failing the difficulty-d criteria, so Hard = Easy + Moderate + Hard
  and UnKnown objects are ignored everywhere).
"""

from dataclasses import dataclass, field
from typing import Dict, Sequence


@dataclass(frozen=True)
class Grouping:
    name: str                       # registry key / CSV discriminator
    column: str                     # instances-CSV column holding the label
    order: Sequence[str]            # canonical group order, values exactly as stored in the CSV
    display: Dict[str, str]         # value -> figure/label text
    ignored: Sequence[str] = field(default_factory=tuple)  # values excluded by the official protocol at every difficulty
    description: str = ""

    @property
    def countable(self):
        """Values the official protocol counts at some difficulty (order minus ignored)."""
        return tuple(value for value in self.order if value not in self.ignored)

    def exclusive_groups(self):
        """(key, display, member_values) for the exclusive view, 'All' last."""
        groups = [(value, self.display.get(value, value), (value,)) for value in self.order]
        groups.append(("All", "All", tuple(self.order)))
        return groups

    def cumulative_groups(self):
        """(key, display, member_values) for the official cumulative view, 'All' last."""
        groups = []
        for position, value in enumerate(self.countable):
            groups.append((value, self.display.get(value, value), tuple(self.countable[:position + 1])))
        groups.append(("All", "All", tuple(self.order)))
        return groups

    def is_ignored(self, value):
        return value in self.ignored


KITTI_DIFFICULTY_RULE = {
    "Easy": "occlusion == 0, truncation <= 15%, 2D bbox height >= 40 px",
    "Moderate": "occlusion <= 1, truncation <= 30%, 2D bbox height >= 25 px",
    "Hard": "occlusion <= 2, truncation <= 50%, 2D bbox height >= 25 px",
    "UnKnown": "fails the Hard criteria; ignored by the official benchmark at every difficulty",
}

KITTI_OCCLUSION_RULE = {
    "0": "fully visible",
    "1": "partly occluded",
    "2": "largely occluded",
    "3": "unknown (set for distant/small or heavily truncated objects); ignored by the official benchmark",
}

GROUPINGS = {
    "kitti_difficulty": Grouping(
        name="kitti_difficulty",
        column="kitti_difficulty",
        order=("Easy", "Moderate", "Hard", "UnKnown"),
        display={"Easy": "Easy", "Moderate": "Moderate", "Hard": "Hard", "UnKnown": "UnKnown"},
        ignored=("UnKnown",),
        description="KITTI Easy/Moderate/Hard benchmark difficulty "
                    "(Geiger et al., CVPR 2012 devkit): " + "; ".join(f"{k} = {v}" for k, v in KITTI_DIFFICULTY_RULE.items()),
    ),
    "kitti_occlusion": Grouping(
        name="kitti_occlusion",
        column="kitti_occlusion",
        order=("0", "1", "2", "3"),
        display={"0": "Level 0", "1": "Level 1", "2": "Level 2", "3": "Level 3"},
        ignored=("3",),
        description="KITTI occlusion annotation levels: " + "; ".join(f"{k} = {v}" for k, v in KITTI_OCCLUSION_RULE.items()),
    ),
}


def from_column(name, column, order, display=None, ignored=(), description=""):
    """Ad-hoc Grouping for any pre-computed instances-CSV column (e.g. a future
    metric already binned by the caller)."""
    display = display or {value: str(value) for value in order}
    return Grouping(name=name, column=column, order=tuple(order), display=display,
                    ignored=tuple(ignored), description=description)


def get_grouping(name):
    if name not in GROUPINGS:
        raise KeyError(f"unknown grouping {name!r}; known: {sorted(GROUPINGS)}")
    return GROUPINGS[name]
