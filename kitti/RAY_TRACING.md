# KITTI geometric ray-tracing metric

`inspect_frame.py` can estimate per-object geometric occlusion from rays emitted at the Velodyne/LiDAR origin.

## Run one frame

```bash
/usr/bin/python3 kitti/inspect_frame.py \
  --data-root /home/<username>/OpenPCDet/data/kitti \
  --frame-id 000000 \
  --ray-trace
```

The optional angular resolution defaults to `0.1` degrees:

```bash
python3 kitti/inspect_frame.py --data-root "$OPENPCDET_ROOT/data/kitti" \
  --frame-id 000000 --ray-trace --ray-resolution 0.2
```

## Definition

For each labelled object, the script samples azimuth/elevation ray directions whose centres intersect that object's 3-D KITTI box. Each ray is tested against every other labelled box. The closest intersected box is the geometrically visible object.

```text
ray occlusion (%) = 100 * occluded target rays / target rays
```

A target ray is **occluded** when another labelled box is closer to the LiDAR origin. A ray is **visible** when the target box is the closest box. `DontCare` labels are excluded from blockers because they are image regions, not reliable 3-D objects.

The result is geometric occlusion, not a replay of the exact KITTI laser firing pattern. KITTI point-cloud files do not reliably expose the original laser ring for this analysis, so the metric uses an angular grid. Smaller angular resolutions give finer estimates but take longer.

## Coordinate handling

KITTI labels are in camera coordinates and specify the bottom centre of each object. The implementation:

1. reads the camera calibration;
2. constructs the eight camera-space box corners;
3. transforms corners and ray directions into the LiDAR frame;
4. performs an oriented-box slab intersection for every ray;
5. compares the first intersection distance across objects.

The existing point-in-box count uses the same KITTI bottom-centre convention.

## Interpretation and limitations

- This is a per-object visibility/occlusion measure, not a detector score.
- Occlusion means another annotated object is in front of the target. Missing returns, sparse sampling, and unlabelled scenery are not counted as occlusion.
- KITTI's `occlusion` field is an image-based human annotation, so it should be used as a validation reference rather than ground truth for this 3-D metric.
- Overlapping or nearly touching boxes are resolved by the nearest ray-box intersection.
- The default `0.1` degree grid is a practical baseline. Report the resolution with results and use the same resolution for comparisons.

## Suggested validation

For each object, retain the printed ray occlusion percentage, KITTI occlusion label, range, and point count. Compare the metric by KITTI occlusion group and plot detection rate against ray-occlusion bins when detector results are available.
