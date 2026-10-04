# LiDAR 360° angular coverage, partial occlusion, and visibility metrics

**Research note — 2026-10-04**

## Executive conclusions

1. The strongest established object-level visibility definition found is **solid-angle visibility on a unit sphere**: project each 3-D box onto the sphere centred at the LiDAR/ego origin, subtract the angular area covered by closer boxes, and divide the remaining solid angle by the target's solid angle. Luo et al. give this explicitly as a visibility score. It naturally handles a 360° azimuth domain and is a better geometric denominator than an unweighted count of azimuth/elevation samples.
2. For this repository, report **three separate quantities**, not one opaque score: (a) geometric visibility opportunity, (b) observed angular return support, and (c) visible target-surface support. They answer different questions and prevent sparse returns from being mistaken for physical occlusion.
3. A true “360° coverage” score must state its domain. `azimuth span / 2π` is a sensor/scan field-of-view measure; a target's angular footprint is a solid angle in steradians; a target's surface coverage is an area fraction. These are not interchangeable.
4. The most useful partial-occlusion baseline here is a **solid-angle-weighted nearest-geometry score**, paired with a **solid-angle-weighted target-return occupancy score** and the existing area-weighted surface score. Keep numerator, denominator, angular resolution, range, point count, and sensor identity beside every result.
5. No source found establishes a universal LiDAR “effective coverage” composite for 3-D detection. Any product or weighted combination below is explicitly proposed for this repository, not a published metric.

## Repository context inspected

- `README.md`: this is OpenPCDet-based point-cloud analysis; datasets and checkpoints are external.
- `THESIS_CONTEXT.md`: the intended question is whether fine-grained geometry explains detection difficulty beyond range and point count. It recommends geometric evidence/visibility proxies rather than claims of detection recovery.
- `mixedsig/PARTIAL_OCCLUSION_METRICS.md`: current exploratory quantities include `H/C`, before-return and pass-through fractions, vertical/surface support, labelled-box visibility, and partner complementary gain.
- `kitti/RAY_TRACING.md`: current KITTI ray tracing samples box-intersecting angular rays and calls a ray occluded when another labelled box is nearer. It correctly warns that this is not the original laser firing pattern and misses unlabelled blockers.
- Current code: `kitti/inspect_frame.py` uses an approximately uniform azimuth/elevation grid; `kitti/visible_surface_area.py` samples box faces and computes visible/blocked area. The former is a useful baseline but is not solid-angle weighted; the latter is already the right general shape for a surface-support measure.
- `THESIS_CONTEXT.md` was present but already modified in the working tree; it was inspected and not changed.

## Terminology and units

Let the sensor origin be `o`, and let a direction be represented in spherical coordinates by azimuth `φ` and latitude/elevation `θ`. The differential solid angle is

```text
dΩ = cos(θ) dθ dφ
```

when `θ` is latitude (or `dΩ = sin(ϑ)dϑdφ` when `ϑ` is colatitude). Solid angle has units **steradians (sr)**; the full sphere is `4π sr`, and a horizontal 360° sweep spans `2π rad` in azimuth. A spherical grid must therefore weight elevation rows by `cos(θ)` (or use exact cell solid angles). Counting grid cells equally overweights high-elevation rows.

“Angular coverage” can mean:

- **Sensor/scan coverage:** the angular interval actually scanned, e.g. `(φ_max - φ_min)/(2π)` for a horizontal 360° claim.
- **Target angular footprint:** the target's subtended solid angle `Ω_T` at the sensor.
- **Target return support:** what fraction of the target footprint has observed returns.
- **Geometric visibility:** what fraction of the target footprint is not blocked by modelled nearer geometry.
- **Surface support:** what fraction of a target surface, usually the sensor-facing/viewable surface, is supported or unblocked.

Always label which one is being reported.

## Established or source-backed metrics

### 1. Solid-angle visibility of a 3-D box — strongest direct match

Luo, Cheng, Ren, Jiang, Chen, Wang, and Li, *3D Object Visibility Prediction in Autonomous Driving*, arXiv:2403.03681 (2024), define visibility by projecting each 3-D bounding box onto a unit sphere centred at the ego-vehicle origin. They define the projected solid angle as

```text
Ω = A / r²
A = ∬_S sin(θ) dθ dφ
```

in their colatitude convention, where `S` is the spherical projection. If `C_i` is the set of boxes closer to the ego vehicle than target box `i`, their visibility is

```text
V_i = [Ω_i - Ω_i ∩ (⋃_{j∈C_i} Ω_j)] / Ω_i.
```

Equivalently, `V_i = Ω_visible / Ω_target`, in `[0,1]` or `[0,100%]`. A target with no closer labelled box has 100% visibility under their definition. They note `O(N²)` box-pair complexity and explicitly state the limitation: only positive/labelled boxes are blockers; buildings, vegetation, walls, and other negative/unlabelled objects are ignored.

**Why it is useful here:** it is viewpoint-specific, naturally covers wrap-around azimuth, measures partial occlusion rather than point density, and gives an analytic interpretation to the existing ray tracer. It is not a detector metric and does not model whether the real sensor emitted a ray in every sampled direction.

**Implementation:** project the eight box corners and box faces to the unit sphere, rasterize or polygon-clip in `(φ, θ)` with wrap-around handling, and integrate `cos(θ)`-weighted cell area. A simpler approximation is the current angular grid with per-cell solid-angle weights. For overlapping blockers, keep the closest intersection per cell rather than subtracting each overlap independently.

**Citation:** Chuanyu Luo et al., “3D Object Visibility Prediction in Autonomous Driving,” arXiv:2403.03681, 2024. DOI-style arXiv record: <https://doi.org/10.48550/arXiv.2403.03681>; paper HTML and equations: <https://arxiv.org/html/2403.03681v1>.

### 2. Ray-cast free/occupied/unknown visibility

Hu, Ziglar, Held, and Ramanan, *What You See is What You Get: Exploiting Visibility for 3D Object Detection*, arXiv:1912.04986 (2019), treat a LiDAR return as evidence not only of an occupied endpoint but also of free space along the sensor-to-return segment. Their implementation traverses a voxel grid: voxels before the final point are free and the endpoint voxel is occupied. They use this visibility representation as an additional network input and for ray-cast-consistent object augmentation; they describe “culling” occluded inserted objects and “drilling” out original blockers.

This is an established **visibility representation**, not an object-level 360° score. It is useful for constructing `observed`, `free`, and `unknown` angular/voxel masks. It also makes clear why a point-only count is insufficient: a far return proves free space along its ray but says nothing about unmeasured directions.

**Citation:** Peiyun Hu, Jason Ziglar, David Held, and Deva Ramanan, “What You See is What You Get: Exploiting Visibility for 3D Object Detection,” arXiv:1912.04986, 2019. <https://arxiv.org/abs/1912.04986>; source implementation/paper text: <https://arxiv.org/html/1912.04986>.

### 3. Visibility-field / retinal-area occlusion

Zhang, Wang, Cheng, and Li, “Automated Visibility Field Evaluation of Traffic Sign Based on 3D LiDAR Point Clouds,” *ISPRS Archives* XLII-2/W13, 1185–1190 (2019), DOI `10.5194/isprs-archives-xlii-2-w13-1185-2019`, construct a hemispherical visibility field over viewpoints around a sign. For each viewpoint, surrounding points are tested by rays against the projected sign panel. The sign and occluder point clouds are projected into a viewpoint-normalized plane; alpha shapes and polygon areas estimate sign and occluding retinal image areas. Their reported occlusion ratio is an area ratio, validated against registered imagery.

This is useful precedent for area-weighted, viewpoint-dependent visibility and for distinguishing a target's projected area from the area blocked by foreground points. It is not a vehicle-box LiDAR benchmark and is not a 360° azimuth metric by itself; its application uses a hemisphere of road viewpoints and a human-vision projection.

**Citation:** S. Zhang, C. Wang, M. Cheng, and J. Li, “Automated Visibility Field Evaluation of Traffic Sign Based on 3D LiDAR Point Clouds,” *International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, XLII-2/W13, 1185–1190, 2019. DOI: <https://doi.org/10.5194/isprs-archives-xlii-2-w13-1185-2019>; official full text: <https://isprs-archives.copernicus.org/articles/XLII-2-W13/1185/2019/>.

### 4. Point-density / box-fill visibility is only a proxy

Luo et al. explicitly discuss the common LiDAR approach of measuring how comprehensively points fill a 3-D box. They warn that near occluders reduce the count, but distant unobstructed objects also have lower point density. Therefore `points_in_box / box volume`, raw point count, or a range-normalized count is evidence of measurement support, not pure occlusion. This supports retaining the current repository quantities `H`, `H/C`, and range/area-normalized support as covariates or observation scores, not naming them physical visibility.

### 5. Dataset visibility/occlusion labels are not 360° LiDAR metrics

- **KITTI:** the official object devkit defines `occluded` as an image annotation with integer states `0` fully visible, `1` partly occluded, `2` largely occluded, and `3` unknown. It also defines truncation by image boundaries. These are useful validation strata, but they are camera/image labels, not solid-angle LiDAR visibility.
  - Official devkit: <https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt>
  - KITTI paper: Andreas Geiger, Philip Lenz, and Raquel Urtasun, “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite,” CVPR 2012, DOI `10.1109/CVPR.2012.6248074`, <https://doi.org/10.1109/CVPR.2012.6248074>.
- **nuScenes:** the official schema has a camera-derived `visibility_token` with levels based on the fraction of an object visible in at least one camera: `1` (0–40%), `2` (40–60%), `3` (60–80%), `4` (80–100%); the devkit also evaluates LiDAR detection separately. Do not treat this token as a LiDAR angular-coverage ground truth.
  - Official devkit: <https://github.com/nutonomy/nuscenes-devkit>
  - Geometry utilities/source: <https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/geometry_utils.py>
  - Dataset paper: Holger Caesar et al., “nuScenes: A Multimodal Dataset for Autonomous Driving,” CVPR 2020, DOI `10.1109/CVPR42600.2020.01149`, <https://doi.org/10.1109/CVPR42600.2020.01149>.
- **Waymo Open Dataset:** difficulty levels use point-count and box/label conditions. That is a useful benchmark baseline, but a threshold on returns cannot distinguish range, incidence, and spatially structured occlusion. This agrees with the negative/control result recorded in `THESIS_CONTEXT.md`.

## 360° scan and angular-domain conventions from official tooling

ROS `sensor_msgs/LaserScan` defines `angle_min`, `angle_max`, and `angle_increment` in radians, with angles around the positive Z axis and zero along +X. The valid range is also explicitly defined in metres. This provides a clean sensor-level convention:

```text
scan_azimuth_fraction = measured_azimuth_span / (2π)
beam_valid_fraction   = valid_ranges / expected_beams
```

Do not infer a complete 360° scan from point-cloud extrema alone; preserve the sensor's configured angular limits and distinguish missing returns from unsampled angles.

- Official ROS message: <https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html>

PCL's official `RangeImage` API similarly represents angular resolution in radians per pixel and defaults its empty range image to `angle_width = 360°` and `angle_height = 180°`. This supports a spherical range-image implementation, but the default image dimensions are a representation choice, not proof that a particular LiDAR sampled the full sphere.

- Official PCL API: <https://pointclouds.org/documentation/classpcl_1_1_range_image.html>

## Recommended metrics for this repository

### A. Geometric target visibility (primary)

Use Luo et al.'s score, implemented with the repository's oriented-box ray intersections:

```text
V_geom = Σ_{k∈target, nearest object is target} w_k
        / Σ_{k∈target} w_k,

w_k ≈ cos(θ_k) Δθ Δφ  [sr]
```

A sample is target-visible when the target is the nearest labelled box along that direction. Store `visible_sr`, `target_sr`, `V_geom`, `occluded_sr`, the blocker class counts, and the angular resolutions. This is a published geometric visibility definition when expressed as the unit-sphere box projection above; the grid approximation is an implementation choice.

Use `unknown` rather than zero when the target footprint has no valid samples. Report labelled-box visibility separately from all-scene visibility because unlabelled blockers are absent from the current geometry.

### B. Observed target angular-return coverage (secondary)

For each target, define its angular footprint `Ω_T` from the box geometry. Bin actual returns or ray endpoints in the same spherical cells and compute

```text
C_return = Σ_{k∈Ω_T and target-supported} w_k
           / Σ_{k∈Ω_T and sensor-sampled} w_k.
```

“Target-supported” should mean a return inside the target box, or a validated target-surface return; it should not include every return in the angular footprint. The denominator must be the sensor-sampled/candidate domain when known. If the original firing pattern is unavailable, use the synthetic grid but name the result `grid_return_support`, not measured beam coverage.

Retain the existing `H/C`, `B/(H+B)`, `P/C`, and raw `H` beside `C_return`. `C_return` is affected by reflectivity, incidence angle, range, beam divergence, motion distortion, and annotation error; it is not a pure occlusion score.

### C. Visible target-surface support (secondary/validation)

The current `kitti/visible_surface_area.py` pattern is appropriate: sample each box face, cast from sensor to the sample, reject samples not on the first target intersection, and classify nearer labelled blockers. Use area weights and report

```text
S_visible = visible_viewable_surface_area / viewable_surface_area
S_blocked  = blocked_viewable_surface_area / viewable_surface_area.
```

The denominator should be `viewable_surface_area`, not the total six-face area, unless the experiment explicitly wants “fraction of the entire cuboid surface visible from this viewpoint.” Keep the two interpretations separate. Surface support is geometrically interpretable but depends on sample spacing, mesh/box approximation, and whether the object is represented by a cuboid rather than its true surface.

### D. Sensor-level 360° coverage

Use only when comparing sensor configurations or scan completeness:

```text
FOV_360 = Δφ / (2π)
```

For a 360° spinning LiDAR, this should normally be near 1 for azimuth, but it says nothing about whether a particular target received useful returns. Pair it with vertical limits, angular resolution, valid-return fraction, and temporal motion/scan duration.

## Proposed/new composites — not published metrics

The following are deliberately proposed for this repo and must be labelled as such if used:

### View-conditioned effective angular coverage (proposed)

```text
EAC = V_geom × C_return
```

This is a conservative “geometric opportunity times observed target support” score. It is easy to interpret, but the product is not established in the literature and can double-count missing returns. Treat it as an exploratory composite only; prefer reporting the two components and their interaction in a regression.

### Collaborative effective coverage (proposed)

For sensor set `A`, define a target support mask on the same solid-angle or surface-cell domain and compute

```text
C_union(A) = weighted_union_supported_cells / weighted_target_cells
ΔC(a | A)  = C_union(A ∪ {a}) - C_union(A).
```

This is the angular analogue of the repository's existing partner complementary gain and marginal agent gain. It quantifies complementary geometric evidence, not detector recovery. A sensor can add angular coverage while adding no usable features, and several partial views can be useful jointly even if each individual `V_geom` is low.

### Entropy of angular support (proposed, diagnostic)

For supported-cell weights `p_k = w_k / Σw`, report

```text
H = -Σ p_k log(p_k),       H_norm = H / log(K).
```

`H_norm` distinguishes concentrated support from broad support at similar total coverage. It is not an occlusion score: a few strong returns can have low entropy while still being correctly visible. Do not confuse this exploratory quantity with the published **Entropy-Guided Visibility Score (EGVS)** of Jiang et al. EGVS is a 2025 roadside-LiDAR deployment surrogate using traffic probabilistic occupancy grids and entropy/information gain; it is related to sensor placement, not this repo's object-level partial-occlusion label.

- Yuze Jiang, Ehsan Javanmardi, Manabu Tsukada, and Hiroshi Esaki, “Towards Efficient Roadside LiDAR Deployment: A Fast Surrogate Metric Based on Entropy-Guided Visibility,” IEEE IV 2025, DOI `10.1109/IV64158.2025.11097554`, <https://doi.org/10.1109/IV64158.2025.11097554>.

## Implementation recommendations

1. Keep the current unweighted `0.1°` ray result as a reproducibility baseline, but add a solid-angle-weighted result before interpreting elevation-dependent differences.
2. Handle azimuth wrap-around explicitly. A box crossing `-π/π` must not produce a nearly 360° footprint by taking raw min/max.
3. Use exact spherical cell weights or `cos(latitude)` weights. Equal cell counts are acceptable only as a named discretization baseline.
4. For every object/sensor result retain: `target_sr`, `visible_sr`, `blocked_sr`, `target_rays`, `visible_rays`, angular resolution, vertical limits, range, box projected area/solid angle, point count, `H/B/P/C`, sensor ID, and calibration/source frame.
5. Separate three failure modes: no angular opportunity (`Ω_T` outside sensor FOV), geometric blocking (`V_geom` low), and weak/absent measured support (`C_return` low). A low point count alone should not be called occlusion.
6. Validate against controlled equal-count point removal and detector outcomes, as planned in `THESIS_CONTEXT.md`. Use KITTI image occlusion labels only as coarse strata, not ground truth for LiDAR visibility.
7. For Mixed Signals, transform every sensor and box into the same frame, compute each sensor's own origin-based mask, then compute union and marginal gain on an explicitly fixed domain. Do not compare raw point counts across vehicle and roadside LiDAR without range, beam pattern, and sensor identity covariates.
8. If unlabelled-scene occlusion matters, replace box-only blockers with a reconstructed surface/occupancy map or use ray-cast free/occupied evidence. State that this is a different metric from labelled-box visibility.

## Bottom line

Use **`V_geom` (solid-angle-weighted labelled-geometry visibility)** as the primary partial-occlusion metric, **`C_return` (solid-angle-weighted measured target support)** as the observation metric, and **`S_visible` (area-weighted visible target surface)** as a complementary geometric check. Report `FOV_360` only as a sensor/scan property. Treat `EAC`, entropy support, and collaborative union scores as proposed repository metrics, not published standards.

## Bibliography and official source links

1. Luo, C., Cheng, N., Ren, Z., Jiang, H., Chen, W., Wang, A., and Li, P. “3D Object Visibility Prediction in Autonomous Driving.” arXiv:2403.03681, 2024. <https://arxiv.org/abs/2403.03681>.
2. Hu, P., Ziglar, J., Held, D., and Ramanan, D. “What You See is What You Get: Exploiting Visibility for 3D Object Detection.” arXiv:1912.04986, 2019. <https://arxiv.org/abs/1912.04986>.
3. Zhang, S., Wang, C., Cheng, M., and Li, J. “Automated Visibility Field Evaluation of Traffic Sign Based on 3D LiDAR Point Clouds.” *ISPRS Archives* XLII-2/W13, 1185–1190, 2019. DOI `10.5194/isprs-archives-xlii-2-w13-1185-2019`. <https://doi.org/10.5194/isprs-archives-xlii-2-w13-1185-2019>.
4. Caesar, H. et al. “nuScenes: A Multimodal Dataset for Autonomous Driving.” CVPR 2020. DOI `10.1109/CVPR42600.2020.01149`. <https://doi.org/10.1109/CVPR42600.2020.01149>.
5. Geiger, A., Lenz, P., and Urtasun, R. “Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite.” CVPR 2012. DOI `10.1109/CVPR.2012.6248074`. <https://doi.org/10.1109/CVPR.2012.6248074>.
6. Jiang, Y., Javanmardi, E., Tsukada, M., and Esaki, H. “Towards Efficient Roadside LiDAR Deployment: A Fast Surrogate Metric Based on Entropy-Guided Visibility.” IEEE IV 2025. DOI `10.1109/IV64158.2025.11097554`. <https://doi.org/10.1109/IV64158.2025.11097554>.
7. ROS `sensor_msgs/LaserScan` official message definition. <https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html>.
8. Point Cloud Library, official `pcl::RangeImage` API. <https://pointclouds.org/documentation/classpcl_1_1_range_image.html>.
9. nuScenes official devkit and source. <https://github.com/nutonomy/nuscenes-devkit>.
10. KITTI official object devkit readme/source mirror. <https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt>.
