# KITTI difficulty baseline — concise analysis (for fig_kitti_difficulty_bars.png)

Thesis-ready prose centred on the one report figure. Everything here comes from
`results/evaluation/kitti_difficulty_metrics.csv` and
`results/analysis/kitti_difficulty_summary.csv`; the fuller method notes live in
`report_text.md`.

---

The KITTI Easy/Moderate/Hard labels are the coarse difficulty annotation the
thesis benchmarks its visibility metric against: native to the working dataset,
and the label every detection result is quoted by. They were evaluated as a
per-object metric on the natural KITTI validation pedestrians (n = 2,280) for
the four pedestrian-capable detectors, under the thesis matching protocol
(greedy confidence-ranked matching, 3D IoU >= 0.50, score >= 0.1). Because the
official protocol computes its three difficulty numbers on nested ground-truth
sets with difficulty-dependent ignore rules, objects were instead grouped into
disjoint bands by their own intrinsic label (Easy 1,139 / Moderate 605 /
Hard 434 / UnKnown 102), so each band reports performance on one population.

Figure X shows the result. The labels stratify detection sharply and
monotonically: 3D AP@0.50 collapses from Easy to Hard for every detector
(PointPillars 45.4 -> 21.0 -> 8.4; SECOND 45.4 -> 19.9 -> 6.2; PointRCNN
30.5 -> 11.8 -> 3.5; PV-RCNN 55.8 -> 21.4 -> 8.7), and detection rate falls in
step (PointPillars 78.0 -> 62.6 -> 49.8%; PointRCNN 56.0 -> 32.4 -> 14.7%).
Directionally, then, the benchmark labels are reliable: an object the
annotations call hard is, overwhelmingly, an object the detectors miss. The
bands also preserve the detector ranking (PV-RCNN strongest, PointRCNN weakest,
at every difficulty), with detector spread widening as difficulty increases.

The label is nevertheless a conflated and lossy summary of difficulty. Its
three criteria mix a human occlusion judgement with image truncation and 2D
bbox height (a distance proxy), and the band profiles inherit that mixture:
Easy pedestrians average 15.1 m range, 242 LiDAR points and 12% ray-cast
occlusion, while Hard pedestrians average 114 points and 56% occlusion. As an
ordinal signal the 3-level label predicts detection at AUC 0.64-0.68 across the
four detectors, whereas continuous log point count reaches AUC 0.71-0.82: the
label discards most of the difficulty signal carried by its own ingredients,
echoing the Waymo point-count rule, whose single bit was close to
uninformative (AUC ~ 0.51-0.58).

The discarded UnKnown band makes the coarseness concrete. Objects failing the
Hard criteria are ignored by the official benchmark at every difficulty, yet
they fail mainly through image truncation (mean truncation 0.41) rather than
poor observation: they average 475 LiDAR points and 19% occlusion, and every
detector finds them at rates comparable to its Moderate band (PointRCNN 49.0%
vs 14.7% in Hard). The benchmark therefore discards well-observed objects for
crossing an image boundary - an annotation artefact, not a perception
difficulty - while its Hard band mixes genuinely occluded objects with merely
distant ones. This is the gap the thesis's geometry-based visibility metric is
designed to close.

For completeness, the same checkpoints were also scored under the official
protocol itself (3D AP_R40 over the full validation split); PointPillars
reproduced the recorded reference values exactly (Car 87.76/78.39/75.19),
verifying checkpoints and pipeline. The official AP remains the
externally comparable anchor, but its nested ignore semantics and ranking-based
formulation make the disjoint-band per-object evaluation above the right
instrument for comparing occlusion metrics, and it supplies the per-object
detection labels used in the SS5.6 regression.

**Figure caption.** KITTI difficulty labels as a detector difficulty metric on
the natural KITTI validation pedestrians (2,280 objects, disjoint bands).
Left: 3D AP@0.50. Right: detection rate at IoU >= 0.50 and score >= 0.1.
Bars: PointPillars, SECOND, PointRCNN, PV-RCNN. Performance degrades
monotonically Easy -> Moderate -> Hard for all detectors, but the UnKnown band
the official benchmark ignores (*) is detected at rates comparable to
Moderate: its members are image-truncated rather than perceptually hard.
