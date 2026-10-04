# Collaborative 3D-detection models with DAIR-V2X support

Research date: 2026-10-01

Scope: candidates whose **official** code explicitly names DAIR-V2X/DAIR-V2X-C support and whose official release provides pretrained weights or a checkpoint download. Sources are first-party repositories, official project/dataset pages, or the authors' paper pages. Checkpoint availability does not imply that the checkpoint can run on a raw DAIR download without the matching converter, annotations, calibration, and loader.

## Shortlist

| Candidate | DAIR-V2X support | Pretrained checkpoint | Best use |
|---|---|---|---|
| [CoAlign](#coalign) | Explicit DAIR-V2X dataset support; LiDAR cooperative 3D detection | Official Google Drive folder | Direct DAIR-V2X-C collaborative 3D-detection baseline |
| [HEAL](#heal) | Explicit DAIR-V2X-C support; LiDAR, camera, and heterogeneous collaboration | Official Hugging Face model repository, including `HEAL_DAIR.zip` and DAIR baselines | Broader multi-modality/heterogeneous experiments |
| [UniV2X](#univ2x) | Official code targets V2X-Seq-SPD, the sequential DAIR-V2X family dataset; cooperative-perception stage includes detection-related driving outputs | Official Google Drive links for cooperative-perception stages | Sequential V2X / end-to-end extension, not the simplest detector-only baseline |

## CoAlign

**Method/model.** CoAlign, *Robust Collaborative 3D Object Detection in Presence of Pose Errors* (ICRA 2023). The official repository lists LiDAR support, DAIR-V2X support, and CoAlign among its collaborative 3D-detection methods. It supports feature-map fusion and pose-error simulation.

- **Code:** <https://github.com/yifanlu0227/CoAlign>
- **Paper:** <https://arxiv.org/abs/2211.07214>
- **Exact DAIR-V2X support:** the README's dataset-support checklist marks DAIR-V2X; its data layout includes `my_dair_v2x/{v2x_c,v2x_i,v2x_v}`. The repository specifically says to use its complemented DAIR-V2X annotations for the 360-degree detection setup.
- **Checkpoint:** [CoAlign DAIR-V2X Google Drive folder](https://drive.google.com/drive/folders/1zsgEMTGpB_Llz66SqeegXXKdaduIegXE?usp=sharing). Download the files into `opencood/logs`, as instructed by the README.
- **Required DAIR annotation download:** [complemented DAIR-V2X-C annotations](https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view?usp=sharing). This is data, not a model checkpoint, but it is part of the repository's expected DAIR setup.
- **Framework/version expectation:** OpenCOOD-style installation; CoAlign says its installation is the same as OpenCOOD apart from additional dependencies. The official requirements pin/expect packages including `easydict~=1.9`, `opencv-python-headless~=4.5.1.48`, `shapely==2.0.0`, `efficientnet_pytorch==0.7.0`, and `pyquaternion`, but do not pin a complete Python/PyTorch/CUDA matrix. Use the repository's OpenCOOD installation guide and expect to compile its CUDA extensions.
- **Coordinate/data caveats:** Do not point the loader at an arbitrary raw DAIR directory. Use the repository's expected `my_dair_v2x` layout and complemented cooperative annotations. DAIR has vehicle and infrastructure agents (`v2x_v`, `v2x_i`) and calibration/pose metadata; the fusion loader must apply those transforms. The complemented labels change the evaluation population from the original vehicle-camera-view-limited labels to 360-degree vehicle-side coverage. Checkpoint weights alone do not contain the other agent's point cloud, calibration, or coordinate conversion.

## HEAL

**Method/model.** HEAL (HEterogeneous ALliance), *An Extensible Framework for Open Heterogeneous Collaborative Perception* (ICLR 2024). It is the successor-style unified framework in the same author's code family and supports LiDAR-only, camera-only, and heterogeneous collaborative perception. For LiDAR 3D detection, the repository documents Pyramid Fusion and the usual OpenCOOD detector/fusion modules.

- **Code:** <https://github.com/yifanlu0227/HEAL>
- **Paper:** linked from the official repository README (ICLR 2024).
- **Exact DAIR-V2X support:** the README's dataset checklist marks `DAIR-V2X-C`; it recommends DAIR-V2X-C and says the repository uses the complemented annotations. The documented DAIR example has one vehicle and one roadside unit.
- **Checkpoint:** official [HEAL Hugging Face model repository](https://huggingface.co/yifanlu/HEAL/tree/main). The repository contains `HEAL_DAIR.zip` (direct download: <https://huggingface.co/yifanlu/HEAL/resolve/main/HEAL_DAIR.zip>) plus DAIR-specific heterogeneous baselines such as `HeterBaseline_DAIR_lidar_attfuse_2023_09_10_12_27_23.zip`, `HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26.zip`, and camera/heterogeneous variants. The README calls this location its benchmark-checkpoint store.
- **Framework/version expectation:** the official installation example is conda Python 3.8 with PyTorch 1.12.0, torchvision 0.13.0, torchaudio 0.12.0, and CUDA toolkit 11.6; then install `requirements.txt`, run `python setup.py develop`, install spconv 1.2.1 or 2.x, and compile the project's CUDA extensions. Important compatibility note: the released checkpoints were stored with spconv 1.2.1 and are not compatible with spconv 2.x; the README also reports a faulty input-channel issue in the benchmark SECOND checkpoints.
- **Coordinate/data caveats:** use the repository's DAIR-V2X-C conversion/layout and complemented annotations, not only the original DAIR labels. The heterogeneous example trains a collaboration base using vehicle and RSU data, then adapts models; this is not equivalent to concatenating two already vehicle-frame point clouds. Preserve DAIR calibration, agent identity, and vehicle/RSU pose transforms. For a fair comparison, match the checkpoint's modality, fusion config, spconv version, and annotation release.

## UniV2X

**Method/model.** UniV2X, *End-to-End Autonomous Driving through V2X Cooperation* (AAAI 2025). This is a broader sequential/end-to-end V2X system rather than a small detector-only package. Its official training/evaluation documentation provides a cooperative-perception stage and explicitly releases its stage checkpoints.

- **Code:** <https://github.com/AIR-THU/UniV2X>
- **Paper:** <https://arxiv.org/abs/2404.00717>
- **Exact DAIR-V2X support:** the official data-preparation document says it uses [V2X-Seq-SPD](https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e?usp=sharing), the sequential V2X perception dataset released by the DAIR-V2X authors. The parent [DAIR-V2X repository](https://github.com/AIR-THU/DAIR-V2X) lists V2X-Seq-SPD and DAIR-V2X downloads. Treat this as support for the DAIR/V2X-Seq-SPD family, not proof that the UniV2X config consumes the ordinary static DAIR-V2X-C layout unchanged.
- **Checkpoint:** the official [train/eval table](https://github.com/AIR-THU/UniV2X/blob/main/docs/TRAIN_EVAL.md) links cooperative-perception weights: [`univ2x_coop_e2e_stg1.pth`](https://drive.google.com/file/d/1Ugm4fHZW8Tz0M-Gfcf1q4GWOGaLacb1a/view?usp=sharing) and [`univ2x_coop_e2e_stg2.pth`](https://drive.google.com/file/d/1V2vLqpjJencg2dZoGtwPb9UQQwsK74hN/view?usp=sharing). The same table links the cooperative tracking config and checkpoint; use the configuration paired with the desired stage.
- **Framework/version expectation:** the official install guide creates conda Python 3.8, installs CUDA toolkit 11.1.1, and follows the repository's pinned/legacy perception stack. Its requirements include `numpy==1.21.5`, `einops==0.4.1`, `motmetrics==1.1.3`, `casadi==3.5.5`, and `pytorch-lightning==1.2.5`; follow the repository's exact PyTorch/torchvision installation commands rather than using this repository's OpenPCDet environment.
- **Coordinate/data caveats:** the official preparation flow expects the V2X-Seq-SPD directory structure and runs its provided preparation scripts. Do not substitute DAIR-V2X-C files without checking the config and temporal metadata assumptions. This system combines cooperative perception with tracking/planning/mapping stages, so its checkpoint and output semantics are broader than a standalone 3D bounding-box detector.

## Useful but not checkpoint-qualified for this request

These projects are relevant, but the official materials inspected here do not satisfy all three requirements (explicit DAIR-V2X support, downloadable pretrained checkpoint, and a clear detector path) as cleanly as the shortlist:

- **Where2comm:** official README explicitly marks DAIR-V2X support and documents `early`, `late`, `intermediate`, `no`, and `intermediate_with_comm` inference. It tells users to train from scratch or continue from a checkpoint, but does not link a DAIR-V2X pretrained checkpoint in the README. Code: <https://github.com/MediaBrain-SJTU/Where2comm>.
- **V2X-ViT:** official README provides a Google Drive checkpoint and an inference command, but the documented pretrained config targets `v2xset/test`, not DAIR-V2X. Code: <https://github.com/DerrickXuNu/v2x-vit>.
- **OpenCOOD:** official model zoo/checkpoints are clearly downloadable, but its own README identifies OPV2V as the official implementation and does not establish a DAIR-V2X checkpoint path. Code: <https://github.com/DerrickXuNu/OpenCOOD>.
- **MMDetection3D:** excellent single-agent 3D detection framework with downloadable checkpoints and custom-dataset support, but the official repository does not document DAIR-V2X or multi-agent collaborative fusion as a supported dataset/model path. It can be used as a detector baseline after converting DAIR-V2X data, but it is not itself a downloadable collaborative DAIR-V2X model. Code: <https://github.com/open-mmlab/mmdetection3d>; model zoo: <https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/model_zoo.md>.

## Recommendation for this repository

Start with **CoAlign DAIR-V2X** for the smallest direct collaborative 3D-detection experiment. Evaluate **HEAL** when heterogeneous vehicle/RSU or camera/LiDAR fusion is important. Use **UniV2X** only when the sequential V2X-Seq-SPD and broader end-to-end task stack are part of the question. In all cases, record the exact annotation release, data conversion, calibration convention, checkpoint archive, Python/PyTorch/CUDA/spconv versions, and fusion configuration; a learned checkpoint is not a substitute for DAIR's multi-agent geometry and loader.
