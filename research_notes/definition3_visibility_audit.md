# Audit: Definition-3 visibility and how to validate it

_2026-10-07. Covers `dairv2x/visibility_frame.py`, the KITTI ray tracer, and the Mixed Signals proxies. All numbers are from the 46-pair DAIR-V2X-C example and are preliminary._

## 1. Where the thesis stands

- **Novelty.** Definition 3 (Luo et al., arXiv:2403.03681) is a published metric. The paper's contribution is predicting it with an MLP head. It never validates the metric itself: Table II scores the head against the algorithm's own output, and the paper calls itself "sensor-agnostic". Implementing Def-3 is therefore not novel. The gap you can fill is **validating Def-3 for LiDAR, and extending it with the sensor-specific terms the paper leaves out**. Name the extension explicitly. Otherwise `THESIS_CONTEXT.md` ("not novel") and the stated goal ("novel metric") contradict each other.
- **Duplicate implementations.** `kitti/inspect_frame.py::compute_ray_occlusion` already computes a Def-3-style quantity: a per-ray nearest-box test on a uniform azimuth/elevation grid. Because the grid is uniform in angle rather than weighted by solid angle, it counts rays instead of measuring solid angle; the difference is negligible near the horizon. Use one implementation across both datasets so the reported numbers match.
- **Dataset roles.** Each dataset should answer a different question:
  - **KITTI: primary predictive and interventional validation.** It already has detector outputs, `occluded` labels and the 820-incident controlled benchmark.
  - **DAIR-V2X: cross-view and cooperative validation.** It has per-side `occluded_state` labels, two viewpoints and CoAlign.
  - **Mixed Signals: demonstration only.** It has no occlusion labels and no detector pipeline, so the metric can't be validated there. Treat it as the application chapter unless time permits.

## 2. Issues in `visibility_frame.py`

| # | Issue | Evidence | Status |
|---|---|---|---|
| 1 | **Depth ordering by box centre** (the paper's convention) is wrong for long boxes. | A truck centred at 10 m hid a target centred at 9 m, but the code scored the target V=1.0 and the truck V=0.89, where a per-ray depth test gives 0 and 1. On the example data the effect is rare: 0.1% of observations change by more than 0.05, with a maximum change of 0.23. | **Fixed:** blocking now uses a per-ray depth test. Regression test added. |
| 2 | **RSU origin height is unverified.** The virtual-LiDAR origin sits about 1.7 m above the labelled box bottoms, and the RSU camera calibration puts the camera about 4.7 m above that origin. | Raising the RSU origin from 0 to 6 m raises the mean V of `occluded_state=0` objects from 0.60 to 0.96. The earlier finding that the RSU has more occlusion than the vehicle (0.38 vs 0.25 mean occlusion) may be an artefact of this. A ring-sharpness test confirmed the vehicle origin but was inconclusive for the tilted RSU. | **Open:** confirm from DAIR documentation or by fitting the origin from returns. This blocks any RSU claim. |
| 3 | **No field-of-view term.** The RSU LiDAR covers about ±47° azimuth. | 462 of 1,231 cooperative boxes (38%) are outside the RSU's horizontal FOV, and 183 are more than 120 m from it. They are currently scored as visible. | **Open:** report the out-of-FOV fraction as a separate truncation term, as KITTI does. Do not fold it into occlusion. |
| 4 | **Ratio only.** V says nothing about apparent size: a car at 80 m can have V=1 and still be tiny. | n/a | Keep the visible solid angle Ω_vis = V·Ω beside V. Ω_vis divided by the beam spacing gives the *expected* number of unoccluded returns. |
| 5 | **Labelled occluders only.** Walls, trees and poles are ignored, an omission the paper concedes. | n/a | This is the natural place for the extension (§3). |
| 6 | **Degenerate labels.** | The vehicle-side single-agent labels contain 181 boxes with non-positive dimensions. | The validation script drops and counts them. |
| 7 | **Quadrature resolution.** | Maximum error versus 128×128 sampling: 0.025 at 16×16 and 0.007 at 64×64. | The current default of 64 is adequate. Report the resolution with results. |

## 3. Recommended novel extension: decomposing Def-3 with measured returns

For each target surface sample (the code already generates these) and each sensor, assign the sample's solid-angle weight to exactly one category:

1. **Out of FOV.** Outside the sensor's angular coverage or maximum range.
2. **Blocked by a labelled box.** This is Def-3 with the per-ray depth test.
3. **Blocked by unlabelled structure.** The sensor's range image has a return clearly in front of the target along this direction. This is measured evidence of a wall, vegetation or a pole.
4. **Visible with a return.** The return lies on or inside the box.
5. **Visible with no return.** Dropout, low reflectivity or beyond range.

The categories partition Ω. Def-3 is recovered as 1 − (2)/(Ω − (1)). Categories 3 and 5 are what label-only Def-3 cannot see. The decomposition turns the existing H/B/P return proxies into solid-angle measure, so all three datasets share one definition. It also stays geometric: it does not include range or point count. For the cooperative case, take the union over agents on the *object surface*, because solid angles from different viewpoints cannot be added. This reuses the same surface samples.

## 4. Validation ladder (weakest to strongest)

1. **Verification.** Does the code compute what it claims?
   - Compare an unoccluded face against the closed-form solid angle of a rectangle.
   - Show convergence as sampling increases, and invariance under rotation and translation (already tested).
   - Cross-check against the KITTI ray tracer on identical boxes.
2. **Convergent validity against human labels.** `dairv2x/validate_visibility_labels.py` (DAIR `occluded_state`); do the same with KITTI's `occluded` field.
   - Require separation *within* range bins, and better than range or log point count alone.
   - Then inspect the largest disagreements. They reveal unlabelled occluders (motivating §3) or annotator conventions.
3. **Predictive validity.** Per ground-truth object, detected ~ log range + log1p(points) + class + size, then add V.
   - Test held-out log-loss or AUC gain and a likelihood-ratio test.
   - Bootstrap by sequence or batch.
   - Include every eligible GT object; never filter to matched predictions only.
4. **Interventional validity.** Insert a virtual box occluder into the scan by deleting points behind it. V is then known by construction.
   - Compare against equal-count random thinning, reusing the KITTI controlled benchmark design.
   - This is the closest the thesis gets to a causal claim.
5. **Cooperative validity (DAIR).** For objects with low V on the vehicle side but high V on the RSU side, cooperative detection should rescue more objects than vehicle-only detection.
   - Needs the single-agent checkpoint. Training is blocked by the corrupt `006451.pcd`; see `DAIRV2X_TRAINING_HANDOFF.md`.

## 5. Preliminary convergent-validity result (DAIR example)

Vehicle side: 1,420 objects in 91 frames. Positive class: `occluded_state` > 0.

| Range | n | AUC 1−V [95% CI] | AUC range | AUC −log1p(points) |
|---|---:|---:|---:|---:|
| all | 1420 | 0.675 [0.648, 0.701] | 0.642 | 0.589 |
| 0–30 m | 208 | 0.904 [0.842, 0.955] | 0.604 | 0.504 |
| 30–60 m | 499 | 0.608 [0.568, 0.645] | 0.511 | 0.490 |
| 60+ m | 713 | 0.640 [0.601, 0.680] | 0.395 | 0.400 |

Def-3 separates the human occlusion labels within every range bin, and better than point count. It is strongest at short range. The labels are camera-annotated and coarse, so this shows only agreement with perceived occlusion. The RSU results (overall AUC 0.68, below range alone at 0.81) are not interpretable until issue 2 is resolved. CIs resample frames, and adjacent frames are correlated, so the CIs are optimistic.
