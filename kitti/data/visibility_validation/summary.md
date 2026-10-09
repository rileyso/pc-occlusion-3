# Definition-3 visibility vs KITTI occluded

Split: val (3769 frames); 32x32 samples per face.
Targets: 19617 non-DontCare objects; 1253 with occluded=3 (unknown) dropped.
Positive class: occluded > 0. AUC 0.5 = no separation.
CIs: 1000 frame-level bootstrap resamples (frames from one drive are correlated).

## all

| occluded | n | mean V | median V | median range (m) | median points |
|---|---:|---:|---:|---:|---:|
| 0 | 9709 | 0.975 | 1.000 | 22.8 | 168 |
| 1 | 5555 | 0.722 | 0.750 | 27.5 | 106 |
| 2 | 4353 | 0.392 | 0.331 | 25.9 | 65 |

| Stratum | n | AUC 1-V [95% CI] | AUC range | AUC -log1p(points) |
|---|---:|---:|---:|---:|
| all | 19617 | 0.867 [0.861, 0.872] | 0.568 | 0.616 |
| 0-30 m | 11724 | 0.882 [0.876, 0.890] | 0.679 | 0.728 |
| 30-60 m | 6608 | 0.857 [0.848, 0.866] | 0.447 | 0.516 |
| 60-1000 m | 1285 | 0.773 [0.752, 0.795] | 0.494 | 0.500 |

## Car

| occluded | n | mean V | median V | median range (m) | median points |
|---|---:|---:|---:|---:|---:|
| 0 | 6814 | 0.987 | 1.000 | 24.5 | 176 |
| 1 | 3956 | 0.724 | 0.734 | 28.0 | 106 |
| 2 | 3074 | 0.341 | 0.310 | 25.8 | 68 |

| Stratum | n | AUC 1-V [95% CI] | AUC range | AUC -log1p(points) |
|---|---:|---:|---:|---:|
| all | 13844 | 0.892 [0.885, 0.898] | 0.545 | 0.606 |
| 0-30 m | 8116 | 0.917 [0.911, 0.923] | 0.691 | 0.763 |
| 30-60 m | 5120 | 0.865 [0.856, 0.875] | 0.414 | 0.546 |
| 60-1000 m | 608 | 0.722 [0.676, 0.761] | 0.457 | 0.458 |

## Pedestrian

| occluded | n | mean V | median V | median range (m) | median points |
|---|---:|---:|---:|---:|---:|
| 0 | 1295 | 0.904 | 1.000 | 12.6 | 208 |
| 1 | 533 | 0.595 | 0.642 | 15.6 | 111 |
| 2 | 414 | 0.370 | 0.189 | 17.3 | 47 |

| Stratum | n | AUC 1-V [95% CI] | AUC range | AUC -log1p(points) |
|---|---:|---:|---:|---:|
| all | 2242 | 0.777 [0.754, 0.798] | 0.612 | 0.732 |
| 0-30 m | 2034 | 0.801 [0.779, 0.824] | 0.619 | 0.755 |
| 30-60 m | 191 | 0.645 [0.580, 0.713] | 0.617 | 0.689 |
| 60-1000 m | 17 | 0.500 [0.500, 0.500] | 0.500 | 0.154 |

## Cyclist

| occluded | n | mean V | median V | median range (m) | median points |
|---|---:|---:|---:|---:|---:|
| 0 | 520 | 0.977 | 1.000 | 27.2 | 52 |
| 1 | 136 | 0.875 | 1.000 | 21.6 | 36 |
| 2 | 49 | 0.267 | 0.189 | 18.8 | 29 |

| Stratum | n | AUC 1-V [95% CI] | AUC range | AUC -log1p(points) |
|---|---:|---:|---:|---:|
| all | 705 | 0.660 [0.622, 0.698] | 0.467 | 0.605 |
| 0-30 m | 414 | 0.647 [0.597, 0.696] | 0.644 | 0.837 |
| 30-60 m | 245 | 0.684 [0.614, 0.760] | 0.445 | 0.545 |
