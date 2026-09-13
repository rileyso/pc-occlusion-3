# KITTI Difficulty Baseline — thesis text

Method, analysis and figure captions for the report chapter on coarse
benchmark annotations. Numbers from `results/` (thesis-protocol arm: natural
KITTI pedestrian benchmark, 2,280 pedestrians, Gilroy matching protocol,
IoU >= 0.50, score >= 0.1; official arm: full 3,769-frame val split, devkit
AP_R40).

## Why this baseline

KITTI's Easy / Moderate / Hard triple is *the* coarse difficulty annotation the
thesis positions its fine-grained visibility metric against: it is the label
every detection reader knows, it is native to the working dataset (unlike the
transplanted Waymo LEVEL rule), and it is entangled with the benchmark's own
evaluation protocol. The baseline therefore runs two arms:

1. **Thesis-protocol arm** — the labels as a per-object metric on the natural
   pedestrian benchmark, evaluated with the same Gilroy core as the occlusion
   bins, DCD bins and Waymo levels, so all coarse baselines are directly
   comparable.
2. **Official arm** — the benchmark's own protocol (3D AP_R40, 40 recall
   points, official ignore semantics) on the full val split, as the external
   anchor and checkpoint verification.

## The rule, as published

Devkit difficulty for a labelled object (Geiger et al., CVPR 2012; object
development kit):

| Label | Occlusion annotation | Truncation | 2D bbox height |
|---|---|---|---|
| Easy | = 0 (fully visible) | <= 15% | >= 40 px |
| Moderate | <= 1 (partly occluded) | <= 30% | >= 25 px |
| Hard | <= 2 (largely occluded) | <= 50% | >= 25 px |
| UnKnown | fails the Hard criteria | — | — |

Two structural properties matter for the thesis. First, the label **conflates
three signals** — an occlusion *annotation* (a human judgement, not a
measurement), image truncation, and pixel height (a proxy for distance given a
pedestrian's physical height). Second, the official protocol **ignores**
UnKnown objects at every difficulty, so they appear in no published number.

Under the official protocol the three difficulty metrics are also computed on
*nested* ground-truth sets (Easy ⊂ Moderate ⊂ Hard), not three disjoint slices
of one population — an Easy-labelled miss is not even a false positive under
Moderate. The bar figures therefore use disjoint (exclusive) bands, each object
counted once; the cumulative (official-semantics) rows are also computed and
written to `kitti_difficulty_metrics.csv`.

## Thesis-protocol results (natural pedestrian benchmark)

**The bands stratify sharply and monotonically.** 3D AP@0.50 by exclusive band
(n = 1,139 / 605 / 434 / 102):

| Detector | Easy | Moderate | Hard | UnKnown | All (n=2,280) |
|---|---|---|---|---|---|
| PointPillars | 45.4 | 21.0 | 8.4 | 11.6 | 40.8 |
| SECOND | 45.4 | 19.9 | 6.2 | 14.3 | 37.3 |
| PointRCNN | 30.5 | 11.8 | 3.5 | 20.0 | 23.5 |
| PV-RCNN | 55.8 | 21.4 | 8.7 | 16.1 | 44.1 |

Detection rate (TP%) falls the same way (PointPillars 78.0 → 62.6 → 49.8;
PointRCNN 56.0 → 32.4 → 14.7). The "All" row reproduces the DCD baseline's
recorded overall values exactly, confirming the shared core.

**But the label is lossy and conflated.** Band profiles (means):
Easy = 15.1 m, 242 points, 12% ray-cast occlusion; Moderate = 19.9 m, 167
points, 34%; Hard = 17.7 m, 114 points, 56%. The ordinal 3-level label
predicts detection at AUC 0.635–0.683 across the four detectors — while
continuous log point count reaches AUC 0.711–0.823, and negated range
0.628–0.775. As with the Waymo LEVEL rule, a one-dimensional coarse label
discards most of the difficulty signal that its own ingredients carry; unlike
the Waymo rule (AUC ≈ 0.51–0.58), KITTI's three levels are at least
directionally reliable.

**The discarded UnKnown band is not the hardest.** Its members fail the Hard
criteria mostly through *image truncation* (mean truncation 0.41 vs 0.00–0.03
elsewhere), not through occlusion (mean ray-cast occlusion 18.9%) or poverty
of returns (mean 475 LiDAR points — the *richest* band, because truncation
removes mostly near-image-edge, hence near, objects). Every detector finds a
larger share of UnKnown than of Hard pedestrians (e.g. PointRCNN 49.0% vs
14.7% TP). The benchmark throws away well-observed objects because they cross
an image boundary — an annotation artefact, not a perception difficulty.

## Official-protocol results (full val, AP_R40)

3D AP_R40 (40 recall points, official ignore semantics — Car IoU 0.70, Pedestrian/Cyclist 0.50):

| Detector | Car (E/M/H) | Pedestrian (E/M/H) | Cyclist (E/M/H) |
|---|---|---|---|
| PointPillars | 87.76 / 78.39 / 75.19 | 57.33 / 51.50 / 46.92 | 81.57 / 62.95 / 58.96 |
| SECOND | 90.55 / 81.61 / 78.61 | 55.94 / 51.15 / 46.17 | 82.97 / 66.73 / 62.78 |
| PointRCNN | 91.36 / 80.55 / 78.04 | 61.41 / 54.85 / 48.31 | 91.67 / 71.35 / 68.15 |
| PV-RCNN | 92.11 / 84.37 / 82.50 | 62.70 / 54.50 / 49.88 | 89.10 / 70.38 / 66.01 |

Note how gentle the official Easy→Hard gradient looks (PointPillars Car
87.8 → 75.2; Pedestrian 57.3 → 46.9) compared with the exclusive bands above
(AP@0.50 45.4 → 8.4): because the official Hard set still consists mostly of
Easy and Moderate objects (nested ignore semantics), the published Hard number
masks how catastrophic difficulty actually is on the objects the label calls
hard. The exclusive-band figure is the honest view; the official triple is the
comparable-with-literature view.

Validation: PointPillars Car reproduces the recorded reference exactly
(87.76 / 78.39 / 75.19), and matches an independent `tools/test.py` evaluation
of the same checkpoint to within rounding (87.7505 / 78.3860 / 75.1811). This
required regenerating predictions with the benchmark's standard camera-FOV
point crop: the natural-benchmark prediction JSONs (shared with the DCD,
Gilroy and Waymo baselines) deliberately omit it, and the out-of-image-FOV
false positives depress official AP by 5–16 points (PointPillars Car
71.4 / 65.8 / 63.2 without the crop). The published convention for every KITTI
number assumes the crop.

## Use the KITTI AP or not? (recommendation)

Both, with clearly separated roles:

- **Use the official AP_R40 as the anchor, not the analysis metric.** It is
  what the literature sees, it is the only way to sanity-check the checkpoints
  against known numbers, and the difficulty triple is the canonical statement
  of "actual detector performance". But it cannot serve the thesis comparison
  directly: its difficulty numbers are computed on nested GT sets under
  difficulty-dependent ignore rules, AP is a ranking metric over the whole
  split (no per-object detection status), and itsignore semantics mean the
  three bars do not slice one population.
- **Use the thesis-protocol per-group metrics for the cross-baseline story.**
  Disjoint bands, per-object detection status, identical matching protocol
  across occlusion bins / DCD bins / Waymo levels / KITTI labels — this is the
  arm that feeds the §5.6 regression (per-object table:
  `results/analysis/per_object_kitti_difficulty.csv`) and the bar figure the
  chapter is built around.

The chapter can then make the thesis's point with the benchmark's own numbers:
even the official triple conflates occlusion with distance and truncation by
construction, and the UnKnown objects it silently discards are demonstrably
not the hardest ones.

## Limitations

- Prediction sets inherit each config's post-processing (`SCORE_THRESH: 0.1`,
  NMS); AP integrals start at score 0.1, identically for all detectors.
- Thesis-protocol arm shares the natural-benchmark prediction JSONs (no FOV
  crop) for cross-baseline comparability; absolute AP@0.50 values there are
  slightly depressed by out-of-FOV false positives relative to a benchmark-
  faithful pipeline. Within-arm comparisons (the thesis use) are unaffected.
- KITTI labels exist only where a human annotated with the image in view; the
  metric definition is 2D-image-bound (pixel height, truncation), which does
  not transfer to MultiSig's roadside-unit geometry — one more reason the
  thesis metric is geometry-based.
- Occlusion Level 3 (n = 38) and UnKnown (n = 102) subgroups are small; only
  UnKnown clears the Gilroy 55-instance minimum (Level 3 is marked †).

## Figure captions

**fig_kitti_difficulty_bars.png** — KITTI difficulty labels as a detector
difficulty metric (natural pedestrian benchmark, 2,280 pedestrians, exclusive
bands). Left: 3D AP@0.50; right: detection rate at IoU >= 0.50, score >= 0.1;
grouped bars: PointPillars, SECOND, PointRCNN, PV-RCNN. All four detectors
degrade monotonically Easy → Moderate → Hard, but the UnKnown band that the
official benchmark ignores (*) is detected at rates comparable to Moderate —
its members are image-truncated, not perceptually hard.

**fig_kitti_occlusion_bars.png** — the same evaluation over the raw KITTI
occlusion annotation levels (0 fully visible / 1 partly / 2 largely / 3
unknown), i.e. the occlusion score proper without the height and truncation
criteria. The pure annotation stratifies less sharply than the full difficulty
label (ray-cast-occlusion AUC 0.60–0.64 vs label AUC 0.64–0.68), and Level 3
(n = 38, † below the estimation minimum) again contains well-observed objects.

**fig_kitti_official_ap_bars.png** — official devkit AP_R40 by difficulty
(full val split, benchmark ignore semantics: nested GT sets), Car and
Pedestrian panels, four detectors. PointPillars Car reproduces the recorded
reference 87.76 / 78.39 / 75.19, verifying the checkpoints and this pipeline
end to end.

**fig_difficulty_bev_gallery_car.png / _pedestrian.png** — the bins seen from
the sensor. Aggregate BEV of seeded val frames in ego coordinates (forward =
up; grey = scan, colour = GT boxes and the LiDAR returns inside them).
Easy objects sit near the ego and are dense (Cars: 18.3 m / 564 points on
average), Moderate and Hard push outward and sparser (Cars: 32.3 m / 200 and
27.9 m / 164), while UnKnown splits into image-edge-truncated cars far to the
sides and a dense cluster adjacent to the host vehicle (mean 1,060 points) —
well-observed objects the benchmark discards. Regenerate with
`visualise_bins.py --target-class Car|Pedestrian` (seeded, any grouping in the
registry).
