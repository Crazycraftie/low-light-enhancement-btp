# Background Knowledge — Theory, Literature, Research Gap

This file holds the theory and literature knowledge built up while planning the project (outside this repo).
Use it for the Introduction, Related Work, Background and Method sections of the report and slides.
**Write everything in fresh words — never copy sentences from papers.** Verify every citation (see bottom).

---

## 1. The problem

- Low-light images (dim, backlit, uneven, extremely dark, coloured light) suffer from: hidden content,
  low contrast, **amplified noise**, colour distortion.
- Why it matters: night photography (phones), surveillance, autonomous driving, and **machine vision**
  (detectors perform poorly in the dark).
- Phone cameras struggle most: small sensors/lenses, need for fast processing.
- Brightening alone is not enough: it also amplifies noise and can shift colours or over-expose light sources.

## 2. Classical methods

- **Histogram equalization (HE):** spreads brightness values over the full range. Simple; over-enhances, boosts noise.
- **CLAHE (Zuiderveld, 1994):** HE on small tiles with a clip limit → better local contrast, less over-enhancement.
- **Gamma correction:** out = in^γ with γ < 1; brightens dark pixels more. Global, no noise handling.
- **Retinex theory (Land):** image = reflectance × illumination. Reflectance = true object colours;
  illumination = light falling on the scene. Enhance by estimating/brightening illumination.
  e.g. **LIME (Guo et al., TIP 2017)** estimates an illumination map.
- Weaknesses of classical Retinex methods: assume reflectance alone is the ideal output; **ignore noise**
  (it stays or is amplified); hand-crafted priors are hard to choose; optimization is slow.

## 3. Deep learning landscape (from the Li et al. TPAMI survey)

Methods grouped by **learning strategy**:

| Strategy | Meaning | Examples |
|---|---|---|
| Supervised | Trained on dark/bright pairs (~73% of methods) | LLNet, Retinex-Net, KinD, MBLLEN, MIRNet |
| Reinforcement | Learns exposure adjustments via rewards | DeepExposure |
| Unsupervised | No pairs, usually GAN-based | EnlightenGAN |
| Zero-shot | No pairs at all; non-reference losses | Zero-DCE, ExCNet, RRDNet, RUAS |
| Semi-supervised | Mix of paired + unpaired | DRBN |

- Supervised also splits into: plain end-to-end, **deep Retinex** (sub-networks for illumination/reflectance),
  and realistic-data-driven methods.
- Problems with supervised: real paired data is hard to collect; synthetic darkening (gamma + noise)
  looks unrealistic; poor generalization. Unsupervised: unstable training, colour shifts.
  Zero-shot: hard to design good non-reference losses.
- Common network design: **U-Net** (encoder–decoder with skip connections). Issues: tiny pixel values
  in dark images; skip connections can carry noise to the output; designs often borrowed from other tasks.
- ~1/3 of methods combine deep learning with Retinex theory.
- Data formats: RGB (common) vs raw sensor data (better, camera-specific; e.g. Learning to See in the Dark).
- **Common losses:** L1/L2 (pixel), SSIM, perceptual (VGG features), smoothness/TV, adversarial,
  exposure (non-reference, used by zero-shot methods).

### Survey's key findings
1. No method wins on every dataset/metric.
2. Real phone images make most methods fail (weak generalization).
3. Supervised methods score higher; zero-shot generalize better but score lower.
4. **Visual quality and metric scores often disagree.**
5. Enhancement helps face detection in the dark, but accuracy stays low.
6. RGB methods struggle with **low light + heavy noise**.

### Survey's open problems / future directions (relevant ones)
- **Removing unknown noise** — methods often amplify real noise; unsolved. → motivates my contribution.
- Uneven illumination (light sources get over-brightened).
- Generalization to real images.
- **Specialized architectures such as transformers** → later realized by Retinexformer (2023).
- **Task-specific evaluation** (does enhancement help detection?) → my detection experiment.

## 4. Datasets

| Dataset | Type | Details | Use in project |
|---|---|---|---|
| LOL-v1 | Paired, real | 500 pairs (485 train / 15 test); captured by changing exposure time & ISO | Training + PSNR/SSIM/LPIPS |
| LOL-v2 Real | Paired, real | 689 train / 100 test | Evaluation (generalization) |
| LOL-v2 Synthetic | Paired, synthetic | 900 train / 100 test | Evaluation |
| LIME / DICM / MEF | Unpaired, real | 10 / 64 / 17 images | NIQE |
| ExDark | Detection | 7,363 dark images, 12 classes | YOLOv8 mAP on a subset |

- There is **no single agreed benchmark** in the field → I evaluate on several datasets.
- _Note added in the repo: our DICM copy (Retinexformer README link) has **69** images, not 64 — use 69 (verified count)._

## 5. Metrics

| Metric | Needs GT? | Measures | Better |
|---|---|---|---|
| PSNR | Yes | Pixel-level fidelity (log of inverse MSE) | ↑ |
| SSIM | Yes | Structural similarity (luminance, contrast, structure) | ↑ (max 1) |
| LPIPS | Yes | Perceptual distance using deep features | ↓ |
| NIQE | No | Naturalness vs statistics of natural images | ↓ |
| mAP@0.5, mAP@0.5:0.95 | Box labels | Detection accuracy | ↑ |
| Dark-region PSNR | Yes | PSNR only in darkest 30% of pixels (my addition) | ↑ |

- PSNR/SSIM don't match human judgement well → add LPIPS, NIQE, and task-based mAP.
- `--GT_mean` evaluation trick (rescales output using ground-truth brightness) inflates PSNR —
  NOT used for my main numbers; explains why some papers report higher values.

## 6. Attention and transformers (background for Retinexformer)

- Image split into tokens. Each token → **Query** (what am I looking for), **Key** (what do I contain),
  **Value** (what I pass on).
- Attention = softmax(QKᵀ / √d) · V. √d keeps values stable. Multi-head = several attentions in parallel.
- Cost grows with the **square of the number of tokens** → pixel-level attention on a 400×600 image
  (240,000 pixels) is infeasible.
- Workarounds: patches (ViT), local windows (Swin), **attention across channels instead of pixels**
  (Restormer, Retinexformer) → cost linear in image size.

## 7. Retinexformer (Cai et al., ICCV 2023) — main model

- **Motivation:** earlier deep Retinex methods ignore corruptions hidden in dark regions (noise, artifacts)
  and those introduced by brightening; many use multi-stage training.
- **Modified Retinex model:** adds **perturbation terms** to reflectance and illumination to model corruption.
- **One-stage Retinex framework (ORF):**
  1. **Illumination estimator:** input = dark image + light-up prior (mean of RGB channels) →
     predicts a **light-up map**; lit-up image = input × light-up map (still noisy). Also outputs
     **light-up features**.
  2. **Corruption restorer:** removes noise, artifacts, colour distortion from the lit-up image.
- **Illumination-Guided Transformer (IGT):** the restorer; U-shaped (encoder, bottleneck, decoder) built from
  **Illumination-Guided Attention Blocks (IGAB)**.
- **IG-MSA (core novelty):** self-attention computed **across channels** (linear cost) and **guided by the
  light-up features** (multiplied into the Values) → regions under different lighting are treated differently;
  dark regions can borrow information from well-lit ones.
- Small model (~1.6M parameters — verify exact figure and FLOPs in the paper). Loss: L1.
  _Note added in the repo: our count of the LOL-v1 checkpoint is 1.61M parameters (results/speed.csv)._
- Gap left open: guides attention with **illumination**, but never explicitly estimates **noise level**.

## 8. SNR-Aware (Xu et al., CVPR 2022)

- **SNR** = signal-to-noise ratio. Very dark regions → low SNR (mostly noise); brighter → high SNR.
- **SNR map:** blur the (grayscale) image → treat |original − blurred| as noise → SNR = blurred / noise.
  No labels needed.
- Two branches: **short-range** (convolutions, good in high-SNR regions) and **long-range** (transformer,
  helps low-SNR regions borrow distant information). SNR map decides each branch's contribution per location
  and **masks very noisy patches** in attention.
- Relation to Retinexformer: Retinexformer guides by **illumination**, SNR-Aware by **noise level**.
  Related but not identical (dark ≠ always low SNR).

## 9. Other baselines (one line each)

- **Zero-DCE (CVPR 2020):** zero-shot; predicts pixel-wise brightness curves applied iteratively; 4
  non-reference losses (spatial consistency, exposure control, colour constancy, illumination smoothness);
  tiny and very fast.
- **SCI (CVPR 2022):** self-calibrated illumination learning; lightweight, unsupervised, fast inference.
- **LLFormer (AAAI 2023):** transformer for ultra-high-definition low-light enhancement (UHD-LOL benchmark).
- **GSAD (NeurIPS 2023) / Diff-Retinex (ICCV 2023):** diffusion-based; strong quality, slow (many steps).
- **Retinex-Net (BMVC 2018):** decomposition + enhancement nets; introduced LOL.
- **KinD (ACM MM 2019):** improved Retinex decomposition with restoration of reflectance.
- **EnlightenGAN (TIP 2021):** unpaired GAN with attention-guided U-Net.
- **Restormer (CVPR 2022):** efficient transformer for restoration; channel-wise ("transposed") attention.

## 10. Field evolution and research gap (for Motivation)

Classical (HE, CLAHE, Retinex) → supervised CNNs (LLNet, Retinex-Net, KinD) → unsupervised/zero-shot
(EnlightenGAN, Zero-DCE, SCI) → transformers (SNR-Aware, LLFormer, Retinexformer) → diffusion (GSAD, Diff-Retinex).

Gaps:
1. **Noise amplification in dark regions** remains unsolved (survey's open problem).
2. Retinexformer uses illumination guidance but no explicit noise-level awareness; SNR-Aware has noise
   awareness but no Retinex framework.
3. Most methods are judged by PSNR/SSIM only; whether they help **machine tasks** (detection) is rarely checked.
4. Diffusion models offer quality at a large speed cost.

**My project:** reproduce Retinexformer + strong baselines; evaluate on fidelity, perception, no-reference
quality, dark regions, efficiency and **downstream detection**; pilot a **noise-aware (SNR-weighted) loss**
in a **controlled experiment** (same fine-tuning, only the loss changes).

## 11. Reference list (verify each — see note)

Survey: Li et al., "Low-Light Image and Video Enhancement Using Deep Learning: A Survey", IEEE TPAMI.
Main: Cai et al., "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement", ICCV 2023.
Xu et al., "SNR-aware Low-light Image Enhancement", CVPR 2022.
Ma et al., "Toward Fast, Flexible, and Robust Low-Light Image Enhancement" (SCI), CVPR 2022.
Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" (Zero-DCE), CVPR 2020.
Wang et al., "Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method" (LLFormer), AAAI 2023.
Hou et al., "Global Structure-Aware Diffusion Process for Low-Light Image Enhancement" (GSAD), NeurIPS 2023.
Yi et al., "Diff-Retinex: Rethinking Low-light Image Enhancement with A Generative Diffusion Model", ICCV 2023.
Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement" (Retinex-Net, LOL), BMVC 2018.
Yang et al., "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement" (LOL-v2), IEEE TIP 2021.
Zhang et al., "Kindling the Darkness" (KinD), ACM MM 2019.
Jiang et al., "EnlightenGAN", IEEE TIP 2021.
Lore et al., "LLNet", Pattern Recognition 2017.
Chen et al., "Learning to See in the Dark", CVPR 2018.
Liu et al., "Retinex-inspired Unrolling with Cooperative Prior Architecture Search" (RUAS), CVPR 2021.
Zamir et al., "Restormer", CVPR 2022. Zamir et al., "MIRNet", ECCV 2020.
Guo et al., "LIME", IEEE TIP 2017. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994.
Land & McCann, "Lightness and Retinex Theory", JOSA 1971.
Vaswani et al., "Attention Is All You Need", NeurIPS 2017. Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), ICLR 2021.
Loh & Chan, "Getting to Know Low-Light Images with the Exclusively Dark Dataset" (ExDark), CVIU 2019.
Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (SSIM), IEEE TIP 2004.
Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (LPIPS), CVPR 2018.
Mittal et al., "Making a 'Completely Blind' Image Quality Analyzer" (NIQE), IEEE SPL 2013.
Ultralytics YOLOv8 (software citation from Ultralytics docs).

**Note:** get every BibTeX entry from DBLP, Google Scholar or the official paper page — do not type author
lists or years from memory. If a detail here conflicts with the actual paper, the paper wins.
