# Validating LiDAR object observability with CoAlign and DAIR-V2X

## Recommendation

Validate observability primarily against **detection and localization outcomes for every eligible ground-truth (GT) object**, not detector confidence. Use paired vehicle-only/cooperative experiments on a checkpoint-compatible DAIR-V2X-C validation subset. Confidence correlation is a secondary diagnostic. The protocol below is a proposed evaluation design, not an established result of the cited papers.

## What the sources establish

- **CoAlign** combines uncertainty-aware single-agent box predictions, agent–object pose-graph optimization, and multiscale feature fusion. Its graph uses predicted box-position/yaw variances as information weights. These regression uncertainties are not classification confidence or ground-truth observability. The paper evaluates detection AP at IoU 0.5/0.7 under pose perturbations. On DAIR-V2X it complements missing annotations for 360-degree detection and uses a detection region of x ∈ [−100,100] m, y ∈ [−40,40] m. [1]
- **Its repository** explicitly distinguishes single detection with uncertainty and cooperative fusion resources, and supplies complemented vehicle-side/cooperative labels. A generic DAIR checkpoint or original label package is therefore not automatically compatible with a CoAlign experiment. [2]
- **DAIR-V2X** separates cooperative C data from additional vehicle-only V and infrastructure-only I collections; V/I are not interchangeable with the paired C benchmark. The paper describes 9,311 synchronized cooperative pairs and a 5:2:3 train/validation/test protocol. “Synchronized” does not imply identical timestamps: triggering differs, and VIC-Async deliberately substitutes earlier infrastructure frames. Infrastructure and vehicle LiDARs also have substantially different sampling/FOV specifications. [3]
- **Official files matter more than headline counts:** the inspected official `cooperative-split-data.json` contains 4,813/1,783/2,688 cooperative train/val/test IDs, which do not total the paper’s 9,311. It also contains separate vehicle/infrastructure/batch splits and `test_A`; do not concatenate these blindly or infer membership from a ratio. The repository distinguishes V2X-Seq-SPD from original DAIR-V2X-C. [4]
- **Calibration** means predicted confidence agrees with empirical correctness probability; Guo et al. show modern classifiers can be miscalibrated and demonstrate held-out temperature scaling. Their classification results do not establish calibration of a 3D detector’s scores or localization uncertainty. [5]

## Strongest practical validation

### 1. Define the population and outcome before computing correlations

Enumerate **all eligible GT boxes**, including zero-return and undetected objects, using a declared class, evaluation region, label version, and ignore policy. Do not require an emitted prediction or minimum point count: that would select away precisely the hard cases being studied.

Fix a detector operating threshold using separate development data. Match predictions to GT one-to-one, class-consistently, with declared 3D IoU thresholds (0.5 and 0.7 facilitate comparison with CoAlign [1]). For each GT object record detected/not detected. For matches record IoU, center error and yaw error; report these conditional errors together with miss rate, not as an apparent improvement caused by dropping difficult objects. A joint “detected within localization tolerance” outcome retains all GT objects. Separately report standard AP/precision to account for false positives.

A confidence score exists only for an emitted prediction. Correlation between observability and matched-prediction confidence is consequently selection-biased as primary validation. Do not assign confidence zero to missed GT and call the result confidence calibration. Secondary prediction-level reliability analysis must define correctness through GT matching and include unmatched predictions as false positives. [5 motivates calibration, not this detection-specific protocol.]

### 2. Baselines and tests that can falsify the metric

Compare against raw in-box return count (also `log1p(count)`), range alone, apparent angular extent/expected beam opportunities, and return count normalized by expected opportunities. Add a simple held-out predictor using count + range + dimensions/apparent size: the metric should improve on this strong nuisance-variable baseline, not merely beat a constant.

A missing return is **not automatically an occlusion**. Expected misses require a defensible scan/FOV model and separation of nearer blocking returns, out-of-FOV rays, unknown/no-return rays, and object hits. Bounding-box intersections approximate object surfaces; label resulting expectations model-based, not ground-truth visibility. Different DAIR sensor sampling patterns require per-sensor normalization. [3]

Report detection probability versus metric bins, AUROC/AUPRC for detection success or failure (state polarity), and held-out Brier/log loss only after fitting a probability mapping. Stratify by range, class, size, sensor and occlusion annotation. Official labels expose occlusion/truncation fields, but these are not documented per-beam LiDAR visibility truth; use them as auxiliary labels. [4]

Use controlled point thinning and angular-sector removal as sensitivity checks, distinguishing sampling loss from actual occlusion. Use separate detectors/checkpoints when feasible; bootstrap by collection batch rather than treating neighboring frames as independent.

### 3. Cooperative value, not merely more points

Evaluate the same GT population under vehicle-only and cooperative inference; add infrastructure-only, early fusion and late fusion when comparable trained checkpoints exist. DAIR’s paper supplies these fusion baselines. [3] Label per-object cooperative rescue, cooperative regression, and localization improvement. Compare cooperative observability gain against added point count and max/sum single-agent scores; test whether complementary angular coverage predicts benefit beyond those baselines.

Vary pose error and temporal offset separately from geometric visibility: CoAlign specifically addresses alignment failures, and DAIR contains timing differences. [1,3] More visible evidence need not improve a misaligned fusion system.

## Dataset acquisition and split discipline

**Do not start by downloading every DAIR archive.** First identify the exact checkpoint/configuration, preprocessing, split IDs, label release, coordinate conventions and evaluation range. A compatible validation experiment needs both sides’ selected point clouds, corresponding GT labels (including CoAlign complements where required), cooperative pair metadata/timestamps, and calibrations/relative transforms used by the loader. Images are needed only if that pipeline actually uses them. [2,4]

Downloading a complete archive may be the practical distribution mechanism, but it does not make all its frames eligible for evaluation. Keep an explicit subset manifest. Preserve official evaluation membership; audit collection batches/intersections and checkpoint training provenance before fitting observability mappings. If adding grouped/site holdouts, report them separately from the official benchmark. Do not mix original C, complemented labels, and Seq-SPD splits. No current archive size or download availability is asserted here.

## Primary sources

1. [Lu et al., *Robust Collaborative 3D Object Detection in Presence of Pose Errors*, ICRA 2023](https://arxiv.org/html/2211.07214), especially IV–V.
2. [Official CoAlign repository](https://github.com/yifanlu0227/CoAlign), README: complemented annotations and model resources.
3. [Yu et al., *DAIR-V2X*, CVPR 2022](https://arxiv.org/html/2204.05575), especially §§3,5.
4. [Official DAIR-V2X repository](https://github.com/AIR-THU/DAIR-V2X), specifically [data documentation](https://github.com/AIR-THU/DAIR-V2X/blob/main/data/README.md) and [cooperative split manifest](https://github.com/AIR-THU/DAIR-V2X/blob/main/data/split_datas/cooperative-split-data.json). Pin revisions when reproducing; repository URLs are mutable.
5. [Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
