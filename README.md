****# Vision Transformers & Self-Supervised Learning

Transformer architectures for vision, from scratch ViT through modern self-supervised representation learning. A hands-on course in three modules:

1. **Foundations** — implement ViT from scratch, compare against Swin / CoAtNet / DeiT, build masked image modeling (MAE).
2. **Transformer-based perception** — DETR family for detection, segmentation, and tracking.
3. **Image & video SSL** — contrastive methods, masked image modeling, DINOv2, V-JEPA, evaluation harness, and applied projects.

Each module has runnable notebooks (small enough to train on a single GPU) and exercise notebooks with placeholders for your own implementation. Solutions are included alongside.

> Companion repo: **[intro-to-vlms](../vision_language_course)** picks up where this leaves off — from CLIP through modern VLMs, alignment, and agentic systems.

---

## Module 1 — Transformer Foundations

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| X | Week 1 | Re-implement basic ViT from scratch (no framework) | Code | ViT paper (Dosovitskiy), lucidrains' vit-pytorch [https://arxiv.org/abs/2010.11929] | [001_vit_from_scratch.ipynb](module1_transformer_foundations/001_vit_from_scratch.ipynb) |
| X | Week 2 | Compare ViT with Swin, CoAtNet, DeiT + deep dive on multi-head self-attention in spatial domain | Theory/Compare | Papers: Swin, DeiT, CoAtNet; timm repo; "Attention Is All You Need", Annotated Transformer | [002_compare_ViTs.ipynb](module1_transformer_foundations/002_compare_ViTs.ipynb), [002_CoAtNet.ipynb](module1_transformer_foundations/002_CoAtNet.ipynb) |
| X | Week 3 | Implement masked image modeling (MIM) pretraining | Code | MAE (He et al.), SimMIM | [003_MAE.ipynb](module1_transformer_foundations/003_MAE.ipynb) |
| X | Week 4 | Visualize attention maps, frozen feature extraction, linear probing. Compare DINOv2 vs CLIP attention. | Analysis/Code | DINOv2, DINO, CLIP papers | |

---

## Module 2 — Transformers for Detection, Segmentation & Tracking

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| X | Week 1 | DETR — build a Transformer detector from scratch | Code | DETR paper (Carion et al.), [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| 🔲 | Week 2 | Building on top of DETR — LW-DETR, RF-DETR | Code/Compare | LW-DETR, RF-DETR papers |
| 🔲 | Week 3 | Segmentation and pose — MaskDINO (segmentation), DETR-Pose (pose estimation), Extend `vanilla_detr.py` for segmentation| Code | MaskDINO, DETR-Pose papers |
| 🔲 | Week 4 | Multi-object tracking — MOTR, MOTRv2, TrackFormer, SAM | Code | MOTR, MOTRv2, TrackFormer, SAM papers |

---

## Module 3 — Self-Supervised Image & Video Representation Learning

* Goal: build up from the *theory* of what makes a good representation, work through every major family of image SSL, move into video SSL, and finish with evaluation + applied projects on real domains.
* Anchor question for the whole module: **what does a good representation look like, and how do we know we've learned one without labels?**

### Phase 1 — Theory & Foundations of Representation Learning

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| ◐ | Week 1 | Representation learning theory — explanatory factors, smoothness, invariance/equivariance, disentanglement, why SSL works at all | Theory | Bengio "Representation Learning: A Review and New Perspectives" (2013), LeCun's "Cake" / EBM-SSL talks | [intro_to_representation_learning.ipynb](module3_ssl_image_and_video/intro_to_representation_learning.ipynb) |
| 🔲 | Week 2 | Information-theoretic view — mutual information, InfoNCE bound, information bottleneck, why MI bounds are loose in practice | Theory | "On Mutual Information Maximization for Representation Learning" (Tschannen et al.), InfoNCE (Oord et al.), "On Variational Bounds of Mutual Information" (Poole et al.) | |
| 🔲 | Week 3 | Pretext-task era — rotation prediction, jigsaw puzzles, colorization, context prediction. Implement one from scratch as a baseline | Code/Theory | Gidaris (rotation), Noroozi (jigsaw), Zhang (colorization), Doersch (context) | |

### Phase 2 — Contrastive & Joint-Embedding Image SSL

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 4 | Implement SimCLR from scratch — augmentation pipeline, projection head, NT-Xent loss. Train on CIFAR-10 / TinyImageNet on a single GPU | Code | SimCLR paper, [google-research/simclr](https://github.com/google-research/simclr), [lightly-ai/lightly](https://github.com/lightly-ai/lightly) | |
| 🔲 | Week 5 | MoCo v1→v2→v3 — momentum encoder + queue, then ViT backbone. Compare against SimCLR on the same data | Code/Compare | MoCo papers, [facebookresearch/moco](https://github.com/facebookresearch/moco), [facebookresearch/moco-v3](https://github.com/facebookresearch/moco-v3) | |
| 🔲 | Week 6 | Non-contrastive methods — BYOL (predictor + stop-grad), SwAV (clustering), Barlow Twins (cross-corr), VICReg (var/inv/cov). Understand *why* no negatives still works | Theory/Code | BYOL, SwAV, Barlow Twins, VICReg papers; [vturrisi/solo-learn](https://github.com/vturrisi/solo-learn) | |
| 🔲 | Week 7 | Read "Understanding self-supervised learning dynamics without contrastive pairs" (Tian et al.) + the collapse-prevention literature | Theory | Tian et al. 2021, "Towards the Generalization of Contrastive SSL" (Wang & Isola), alignment & uniformity paper | |

### Phase 3 — Masked Image Modeling & Distillation-based SSL

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 8 | Re-visit MAE (from Module 1), compare against SimMIM, BEiT, iBOT, data2vec. Pixel target vs feature target vs discrete-token target | Theory/Compare | MAE, SimMIM, BEiT, iBOT, data2vec papers; [003_MAE.ipynb](module1_transformer_foundations/003_MAE.ipynb) | |
| 🔲 | Week 9 | DINO & DINOv2 — student/teacher with EMA, centering, sharpening, multi-crop. Implement DINO from scratch on a small dataset | Code | DINO, DINOv2 papers; [facebookresearch/dino](https://github.com/facebookresearch/dino), [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | |
| 🔲 | Week 10 | I-JEPA — predicting in *representation space* instead of pixel space. Read paper, run inference, understand why latent prediction beats pixel reconstruction for semantic features | Theory/Code | I-JEPA paper, [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | |
| 🔲 | Week 11 | DINOv3 + Franca (fully open-source DINOv2-class). Compare DINOv3 register tokens vs DINOv2; reproduce a small Franca run | Code/Compare | DINOv3, [valeoai/Franca](https://github.com/valeoai/Franca), "Vision Transformers Need Registers" (Darcet et al.) | |

### Phase 4 — Video Self-Supervised Learning

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 12 | VideoMAE / VideoMAEv2 — tube masking, why video needs *higher* masking ratios than images, dual masking. Run inference + small finetune on Kinetics subset | Code | VideoMAE, VideoMAEv2 papers; [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) | |
| X | Week 13 | V-JEPA & V-JEPA 2.1 — life-of-an-input walkthrough: 3D patch embed, multi-level deep supervision, mask-token predictor, EMA target | Code/Theory | V-JEPA, V-JEPA 2 papers; [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | [v-jepa2_1.ipynb](module3_ssl_image_and_video/v-jepa2_1.ipynb) |
| 🔲 | Week 14 | Temporal-coherence & motion-aware SSL — CVRL, TimeContrast, MaskedFeat, ST-MAE. Why slow-features and temporal-equivariance priors matter | Theory/Compare | CVRL, MaskedFeat, ST-MAE, "Slow Feature Analysis" (Wiskott & Sejnowski) | |
| 🔲 | Week 15 | Cross-modal SSL for video — audio-visual (AVID-CMA, MAViL), video-text (VideoCLIP, InternVideo). Discuss CLIP itself as an SSL method | Theory | AVID-CMA, MAViL, VideoCLIP, InternVideo papers | |

### Phase 5 — Evaluation Frameworks for Representation Learning

* How do we actually measure whether a representation is *good*? This phase is its own deliverable — a small `eval/` library you can reuse for the rest of the course.

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** | **Solutions** |
| --- | --- | --- | --- | --- | --- |
| 🔲 | Week 16 | Build a probe harness — linear probe, kNN probe, fine-tune. Apply to a frozen DINOv2 / V-JEPA backbone on CIFAR-100 / iNaturalist / UCF-101 | Code/Eval | DINOv2 eval recipe, [VISSL](https://github.com/facebookresearch/vissl), [lightly benchmarks](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html) | |
| 🔲 | Week 17 | Dense / structured evaluation — frozen-features for detection (COCO), segmentation (ADE20k), depth (NYUv2), tracking. The "DINOv2-style" eval suite | Eval | DINOv2 paper §5, "How well do SSL models transfer?" (Ericsson et al.) | |
| 🔲 | Week 18 | Robustness & OOD probing — ImageNet-A/C/R, ObjectNet, VTAB-1k. Disentanglement metrics (β-VAE, FactorVAE, DCI) on a synthetic dataset | Eval/Theory | VTAB paper, "Challenging Common Assumptions in Disentanglement" (Locatello et al.) | |

#### Deep dive on SSL evaluation benchmarks

| **Benchmark** | **What it measures** | **Modality** |
| --- | --- | --- |
| **Linear probe @ ImageNet-1k** | Canonical headline number — is the feature linearly separable for 1k classes? | Image |
| **kNN @ ImageNet-1k** | Quality of the *raw* embedding geometry without any training on top | Image |
| **VTAB-1k** | 19 diverse tasks in low-data regime — natural, specialized, structured | Image |
| **ADE20k / Cityscapes** (frozen features) | Dense semantic features — pixel-level not just image-level | Image (dense) |
| **COCO detection / instance seg** (frozen) | Localization quality of the frozen backbone | Image (dense) |
| **NYUv2 depth / NAVI** (frozen) | Geometric features, 3D awareness from 2D pretraining | Image (dense) |
| **Kinetics-400 / SSv2** (linear / finetune) | Action recognition — Kinetics rewards appearance, SSv2 rewards temporal reasoning | Video |
| **UCF-101 / HMDB-51** (linear / finetune) | Smaller video benchmarks, common for compute-constrained eval | Video |
| **EPIC-Kitchens** | Egocentric, long-tail action recognition + anticipation | Video |
| **ImageNet-A / C / R / Sketch / ObjectNet** | Robustness to natural adversarials, corruptions, renditions, OOD | Image |
| **DCI / MIG / FactorVAE score** | Disentanglement — does the representation align with true factors of variation? | Synthetic |

### Phase 6 — Applied Projects: SSL That Actually Shipped

| **Status** | **Week** | **Task / Goal** | **Category** | **Resources** |
| --- | --- | --- | --- | --- |
| 🔲 | Week 19 | DINOv2-as-backbone project — pick a niche domain (medical, satellite, microscopy, retail) and beat a supervised baseline with frozen DINOv2 features + small head | Project | DINOv2 repo, RetFound (retinal), SatMAE (satellite), [Prov-GigaPath](https://github.com/prov-gigapath/prov-gigapath) (pathology) |
| 🔲 | Week 20 | Video SSL for embodied AI — read VC-1 ("Where are we in the search for an artificial visual cortex?") and run V-JEPA-2 features on a robotics / video-QA task | Project | VC-1 paper, V-JEPA 2 release, [facebookresearch/eai-vc](https://github.com/facebookresearch/eai-vc) |
| 🔲 | Week 21 | Capstone — train a small SSL model from scratch on your own domain (≤ 100k unlabeled images or ≤ 1k unlabeled videos), evaluate with the Phase-5 harness, write up what worked | Capstone | All of the above |

### Suggested reading / watching

#### Theory & survey
- Bengio, Courville, Vincent — **"Representation Learning: A Review and New Perspectives"** (the canonical reference, still worth reading)
- Yann LeCun — **"A Path Towards Autonomous Machine Intelligence"** (the JEPA / world-model position paper)
- Ericsson, Gouk, Hospedales — **"How Well Do Self-Supervised Models Transfer?"** (CVPR 2021) — sober empirical comparison
- Balestriero et al. — **"A Cookbook of Self-Supervised Learning"** (Meta, 2023) — practical recipes and failure modes

#### Talks / videos
- **Yann LeCun — Energy-Based SSL / JEPA** talks (multiple venues, search "LeCun JEPA")
- **Lucas Beyer — "Vision Transformers"** and **"DINO / DINOv2"** talks
- **Mathilde Caron** — DINO presentation (original author)
- **Yannic Kilcher** paper walkthroughs — SimCLR, MoCo, BYOL, DINO, MAE, V-JEPA
- **Stanford CS231n** lecture on self-supervised learning
- **MIT 6.S898 Deep Learning** — SSL lectures

#### Libraries to know
- **[lightly](https://github.com/lightly-ai/lightly)** — clean PyTorch implementations of every major image SSL method, good for ablations
- **[solo-learn](https://github.com/vturrisi/solo-learn)** — research-grade SSL training library
- **[VISSL](https://github.com/facebookresearch/vissl)** — Meta's SSL benchmarking framework (older but still useful)
- **[mmselfsup](https://github.com/open-mmlab/mmselfsup)** — OpenMMLab's SSL toolbox

#### Domain success stories worth studying
- **RetFound** (Nature 2023) — MAE on 1.6M retinal images, transfers to ocular and systemic disease
- **SatMAE / Scale-MAE** — masked autoencoders for satellite imagery
- **Prov-GigaPath** — pathology foundation model
- **VC-1 / V-JEPA 2** — embodied / robotics
- **DINOv2 + SAM** — segmentation pipelines built on frozen SSL features
