# Candidate Partial-Occlusion Metrics

These are exploratory geometric proxies available from the current Mixed Signals data and code. They are **not validated physical visibility or detection-recoverability metrics**.

## Notation

For one sensor–object observation:

- `H`: measured returns inside the target GT box (`hit_count`)
- `B`: measured returns before the box along rays whose extended directions intersect it (`before_count`)
- `P`: measured returns beyond the box along rays whose directions intersect it (`beyond_count`)
- `C = H + B + P`: candidate measured rays
- `r`: sensor-to-box-centre range
- `A_proj`: box area projected toward the sensor
- `d_z`: GT-box height
- `T`: set of target surface-grid cells
- `E`, `Q`: surface cells supported by the ego sensor and a partner sensor

A score based on very few points or candidate rays is weak evidence. Always report its numerator and denominator.

## Return-based metrics available now

| Name | Formula | Description | High implies | Low implies |
|---|---|---|---|---|
| **Measured-return hit fraction** | `H / C` | Fraction of candidate measured rays whose returns lie inside the GT box; this is the current `V proxy`. | Most associated measured directions return inside the box. | Returns commonly stop before or continue beyond the box; the cause is ambiguous. |
| **Before-return fraction** | `B / (H + B)` | Fraction of returns that stop before the box among returns that stop before or inside it. | Stronger measured evidence of foreground obstruction. | Returns generally reach the target box, or there is insufficient evidence. |
| **Pass-through fraction** | `P / C` | Fraction of candidate measured rays whose returns occur behind the GT box. | Possible empty box volume, sparse object structure, calibration error, or annotation mismatch. | Candidate returns usually stop before or inside the box. |
| **Raw return support** | `H` | Number of measured points inside the GT box. | Dense sensor support for the object. | Sparse or absent support, potentially due to range, resolution, or occlusion. |
| **Range/area-normalized support** | `H r² / A_proj` | Approximate point support corrected for distance and projected box size. | More returns than expected from range and apparent size. | Less support than expected, potentially indicating occlusion or poor incidence angle. |
| **Before-depth severity** | `median((entry_i - range_i) / entry_i)` for before-box returns | Relative distance between foreground returns and the target box. | Blocking returns occur substantially in front of the target. | Blocking returns lie near the target boundary and may reflect annotation or calibration noise. |

Undefined fractions should be reported as `unknown`, not zero.

## Spatial-support metrics available from hit points

| Name | Formula | Description | High implies | Low implies |
|---|---|---|---|---|
| **Height coverage** | `(z_95 - z_5) / d_z` in box-local coordinates | Robust fraction of the object's vertical extent covered by returns. | Returns span most of the object's height. | Returns occupy only a narrow vertical section, consistent with partial visibility or sparse sampling. |
| **Vertical-band occupancy** | `occupied height bands / total height bands` | Divides the box into a small fixed number of height bands and counts bands containing returns. | Support is distributed vertically. | Only part of the object's height is supported. |
| **Facing-surface occupancy** | `occupied facing-surface cells / total facing-surface cells` | Projects returns onto sensor-facing GT-box faces and measures occupied grid cells. | Broad spatial support across visible box surfaces. | Concentrated or missing support, consistent with partial visibility. |

## GT-box geometric metrics

| Name | Formula | Description | High implies | Low implies |
|---|---|---|---|---|
| **Labelled-box occlusion fraction** | `blocked target-face samples / target-face samples` | Casts rays to sampled target-box faces and counts samples for which another labelled GT box is closer. | Other labelled objects geometrically block much of the target. | Little target area is blocked by labelled objects. |
| **Labelled-box visible fraction** | `1 - labelled-box occlusion fraction` | Complement of labelled-box occlusion using sampled target-facing surfaces. | Most sampled target surface is unobstructed by labelled boxes. | Much of the target is blocked by labelled boxes. |

These metrics cannot see walls, vegetation, road furniture, or other unlabelled blockers.

## Collaborative-view metrics

Choose one spatial coverage metric, preferably facing-surface occupancy, before computing these.

| Name | Formula | Description | High implies | Low implies |
|---|---|---|---|---|
| **Partner complementary gain** | `(|E ∪ Q| - |E|) / |T|` | Additional target surface supported after adding one partner to the ego view. | The partner contributes previously unsupported object regions. | The partner adds little new geometric support. |
| **Viewpoint redundancy** | `|E ∩ Q| / |E ∪ Q|` | Jaccard overlap between ego and partner supported surface cells. | The two sensors observe mostly the same regions. | Their supported regions differ, or both have little support. |
| **Visibility opportunity** | `max(V_partner) - V_ego` | Difference between ego and best-partner score for a chosen visibility proxy `V`. | At least one partner has substantially stronger geometric evidence than ego. | The best partner offers little score improvement. |
| **Shared-blindness proxy** | `1 - max_s(V_s)` | Inverts the best score available from any sensor for the object. | Every sensor has a poor score for the object. | At least one sensor has a strong score. |
| **Multi-agent union coverage** | `|E ∪ Q₁ ∪ ... ∪ Qₙ| / |T|` | Fraction of target surface cells supported by all selected agents together. | Collaborative sensors collectively cover much of the object. | Significant target regions remain unsupported across all sensors. |
| **Marginal agent gain** | `(coverage(S ∪ {a}) - coverage(S))` | Additional surface coverage contributed by adding agent `a` to an existing sensor set `S`. | The added agent provides useful new geometry. | The added agent is mostly redundant, suggesting diminishing returns. |

`Visibility opportunity` indicates an opportunity for collaboration, **not proven detection recovery**. A best-partner score can also miss cases where several individually partial views become useful only when combined.

## Recommended first experiment

Start with:

1. Raw return support
2. Before-return fraction
3. Pass-through fraction
4. Four-band vertical occupancy
5. Partner complementary gain using those vertical bands

Keep range, projected box size, class, sensor identity, `H`, `B`, `P`, and `C` beside every score. Exclude sensor–self-box pairs where the sensor origin lies inside the target box.
