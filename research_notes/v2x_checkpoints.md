# Downloadable collaborative 3D-detection checkpoints for DAIR-V2X

Research date: 2026-10-04. Primary sources only: authors' repositories, READMEs/configs, official model storage, and papers. “Downloadable” means the official source exposes weights; it does **not** mean weights run against an unconverted DAIR download.

## Verdict

| Model | DAIR support | Official weights | Collaborative 3D detection | Practical verdict |
|---|---|---|---|---|
| **CoAlign** (ICRA 2023) | Explicit **DAIR-V2X** support; expects complemented annotations and its `dataset` layout | [DAIR-V2X Google Drive folder](https://drive.google.com/drive/folders/1zsgEMTGpB_Llz66SqeegXXKdaduIegXE?usp=sharing) | Yes, LiDAR feature-map fusion; pose-error robustness | **Best direct baseline** and simplest starting point |
| **HEAL** (ICLR 2024) | Explicit **DAIR-V2X-C** support; vehicle + RSU; complemented annotations | [Official HF model repo](https://huggingface.co/yifanlu/HEAL), especially [`HEAL_DAIR.zip`](https://huggingface.co/yifanlu/HEAL/resolve/main/HEAL_DAIR.zip) and [DAIR LiDAR baselines](https://huggingface.co/yifanlu/HEAL/tree/main) | Yes; LiDAR, camera, and heterogeneous collaborative perception | **Strongest successor option**; broadest modality support, but the most fragile environment |
| **UniV2X** (AAAI 2025) | Uses **V2X-Seq-SPD**, the sequential DAIR/V2X-Seq family—not ordinary static DAIR-V2X-C unchanged | [Stage-1 checkpoint](https://drive.google.com/file/d/1Ugm4fHZW8Tz0M-Gfcf1q4GWOGaLacb1a/view?usp=sharing), [stage-2 checkpoint](https://drive.google.com/file/d/1V2vLqpjJencg2dZoGtwPb9UQQwsK74hN/view?usp=sharing) | Cooperative perception is part of a tracking/mapping/planning stack | Use only if sequential/end-to-end V2X is wanted; **not** the drop-in detector choice |
| **Where2comm** (NeurIPS 2022) | README names DAIR-V2X and provides DAIR inference support | No official DAIR-V2X pretrained archive found in its README; [official repo](https://github.com/MediaBrain-SJTU/Where2comm) documents training/checkpoint use | Yes; selective communication over OpenCOOD fusion | Relevant method, but fails the requested “download weights now” criterion |
| **V2X-ViT** (ECCV 2022) | Official weights target **V2XSet**, not DAIR-V2X | [Google Drive checkpoint folder](https://drive.google.com/drive/folders/1h2UOPP2tNRkV_s6cbKcSfMvTgb8_ZFj9?usp=sharing) | Yes, early/late/intermediate fusion | Useful architecture baseline, **not a verified DAIR checkpoint** |

## 1. CoAlign: direct DAIR-V2X checkpoint

- **Official code:** [yifanlu0227/CoAlign](https://github.com/yifanlu0227/CoAlign)
- **Paper:** [Robust Collaborative 3D Object Detection in Presence of Pose Errors](https://arxiv.org/abs/2211.07214)
- The README's dataset support and preparation sections explicitly include DAIR-V2X, with a local `dataset` directory and the repository's expected DAIR layout. Its checkpoint section links the [CoAlign DAIR-V2X folder](https://drive.google.com/drive/folders/1zsgEMTGpB_Llz66SqeegXXKdaduIegXE?usp=sharing).
- It is collaborative **LiDAR 3D detection**, not a single-agent detector. The loader needs both agents' observations plus DAIR calibration/poses and coordinate transforms.
- The README also requires the authors' [complemented DAIR-V2X-C annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/); original labels alone are not a safe reproduction target.
- **Inference:** use the repository's OpenCOOD-derived `opencood/tools/inference.py`, the downloaded checkpoint directory/config, and the fusion mode configured by that checkpoint. Inspect the checkpoint config before running rather than assuming a raw DAIR path.
- **Caveats:** older OpenCOOD-style code and CUDA extensions; expect legacy PyTorch/spconv dependency work. The Drive folder is an explicit release link, but its file names/contents should be checked after download.

## 2. HEAL: best CoAlign successor for DAIR-V2X-C

- **Official code:** [yifanlu0227/HEAL](https://github.com/yifanlu0227/HEAL)
- **Paper:** [HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception](https://arxiv.org/abs/2401.13964)
- The CoAlign README itself points users to HEAL as the later unified framework. HEAL's data-preparation section explicitly lists **DAIR-V2X-C**, says it uses complemented annotations, and describes vehicle/RSU collaboration.
- **Weights:** official [Hugging Face model repository](https://huggingface.co/yifanlu/HEAL) exposes [`HEAL_DAIR.zip`](https://huggingface.co/yifanlu/HEAL/resolve/main/HEAL_DAIR.zip) plus DAIR-specific LiDAR, camera, and LiDAR-camera baseline archives, for example [`HeterBaseline_DAIR_lidar_attfuse...zip`](https://huggingface.co/yifanlu/HEAL/resolve/main/HeterBaseline_DAIR_lidar_attfuse_2023_09_10_12_27_23.zip).
- **Inference:** HEAL retains the OpenCOOD-style test flow; its README's Basic Train/Test section and `opencood/tools/inference.py` use a checkpoint `model_dir`, a config whose `validation_dir` points at the test set, and a selected fusion method. Use the archive's included config rather than mixing a different modality/config.
- **Caveats:** official setup is legacy (Python 3.8, PyTorch 1.12/CUDA 11.6 example, compiled extensions). The README warns that released checkpoints were made with **spconv 1.2.1** and are not compatible with spconv 2.x; it also warns of faulty SECOND input-channel metadata. This is a real reproducibility risk, not a reason to silently substitute a newer environment.

## 3. UniV2X: newer, but not a drop-in detector

- **Official code:** [AIR-THU/UniV2X](https://github.com/AIR-THU/UniV2X)
- **Paper:** [End-to-End Autonomous Driving through V2X Cooperation](https://arxiv.org/abs/2404.00717)
- The official [data preparation](https://github.com/AIR-THU/UniV2X/blob/main/docs/DATA_PREP.md) and [train/eval table](https://github.com/AIR-THU/UniV2X/blob/main/docs/TRAIN_EVAL.md) target V2X-Seq-SPD, the sequential DAIR/V2X-Seq dataset. The repository releases cooperative stage weights, including [`univ2x_coop_e2e_stg1.pth`](https://drive.google.com/file/d/1Ugm4fHZW8Tz0M-Gfcf1q4GWOGaLacb1a/view?usp=sharing) and [`univ2x_coop_e2e_stg2.pth`](https://drive.google.com/file/d/1V2vLqpjJencg2dZoGtwPb9UQQwsK74hN/view?usp=sharing).
- **Inference:** follow the paired config and scripts in `docs/TRAIN_EVAL.md`; the weights are stage-specific and cover cooperative perception/tracking/mapping/planning, not a standalone CoAlign-compatible 3D-box command.
- **Caveats:** do not point it at ordinary DAIR-V2X-C files without validating temporal metadata and the V2X-Seq-SPD directory structure. The official install is legacy Python 3.8/CUDA 11.1-era software with a larger dependency stack.

## What to run first

1. **Need downloadable weights and direct static DAIR collaborative 3D detection:** start with **CoAlign**.
2. **Need the strongest practical successor, heterogeneous modalities, or more DAIR-specific released archives:** try **HEAL**, pinning its documented old environment/spconv 1.2.1.
3. **Need sequential V2X and an end-to-end autonomy stack:** evaluate **UniV2X**, but treat it as a separate pipeline.

Every collaborative run additionally needs the matching DAIR/V2X data conversion, both agents' point clouds/images as applicable, calibration/poses, complemented annotations where required, and the repository's exact config. A checkpoint alone is not an inference dataset.

## Sources

- [DAIR-V2X official repository](https://github.com/AIR-THU/DAIR-V2X) and [dataset portal](https://thudair.baai.ac.cn/index)
- [CoAlign README](https://github.com/yifanlu0227/CoAlign/blob/main/README.md)
- [HEAL README](https://github.com/yifanlu0227/HEAL/blob/main/README.md) and [HEAL model files](https://huggingface.co/yifanlu/HEAL/tree/main)
- [UniV2X data preparation](https://github.com/AIR-THU/UniV2X/blob/main/docs/DATA_PREP.md), [installation](https://github.com/AIR-THU/UniV2X/blob/main/docs/INSTALL.md), and [train/eval](https://github.com/AIR-THU/UniV2X/blob/main/docs/TRAIN_EVAL.md)
- [Where2comm README](https://github.com/MediaBrain-SJTU/Where2comm/blob/main/README.md)
- [V2X-ViT README](https://github.com/DerrickXuNu/v2x-vit/blob/main/README.md)
