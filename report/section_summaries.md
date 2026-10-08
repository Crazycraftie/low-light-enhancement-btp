# Report — plain-language summary of every section (for the viva)

## Abstract
Dark photos are dim, noisy and colour-shifted; brightening also brightens the noise. We reproduced Retinexformer and
8 other methods, tested them all the same way (quality, darkest regions, speed, object detection), found three
"hidden" problems in how the field evaluates, and tried our own noise-aware loss. The loss gave only a tiny, consistent
improvement in dark regions. **Be ready to explain:** every number in the abstract and where it comes from.

## I. Introduction (+ Related Work, Contributions)
Why low light is hard (few photons → noise), why brightening is not enough, and how the field evolved: classical
(HE, CLAHE, LIME) → deep Retinex CNNs (Retinex-Net, KinD) → zero-shot (Zero-DCE, SCI) → transformers (SNR-Aware,
LLFormer, Retinexformer) → diffusion (GSAD). The gap: Retinexformer knows where it is *dark* but not where it is *noisy*;
methods are rarely tested on a machine task. **Be ready to explain:** the four bullets of "Contributions" in your own words;
the difference between illumination-aware and noise-aware.

## II. Background
Retinex: image = reflectance (object colours) × illumination (light). Self-attention compares every token with every
other token → cost grows with N² (N = number of pixels = 240,000 for a LOL image → 5.8×10¹⁰ entries, impossible).
Attention across channels makes a C×C matrix instead → cost grows linearly with image size.
**Be ready to explain:** Q, K, V in one sentence each; why √d; why channel attention is cheap.

## III. Methodology
Retinexformer = illumination estimator (makes a light-up map and light-up features; lit-up image = I·L + I) +
corruption restorer (U-shaped transformer; in every attention block the values V are multiplied by the light-up
features, then attention is computed across channels). Our loss: compute an SNR map from the dark input
(blur → noise = |gray − blur| → SNR = blur/noise), turn it into weights (noisy pixels up to 2× the weight of clean
ones, average weight exactly 1), and weight the L1 loss with it. Two design decisions (rank normalisation; mean = 1).
FFT loss = compare Fourier amplitudes. Detection: off-the-shelf YOLOv8m on raw vs enhanced ExDark images.
**Be ready to explain:** Eq. (3) lit-up image; Eq. (5)–(7) SNR weights; *why* min-max normalisation was rejected
(93% of pixels got weight ≈ 2 → it would just double the learning rate); why weights average to 1.

## IV. Experimental Setup
Datasets (LOL-v1 485/15, LOL-v2 real/syn, LIME/DICM/MEF, ExDark 1,200), metrics (PSNR, SSIM, LPIPS, NIQE,
dark-region PSNR, mAP), three fairness rules (no ground truth at test time; no checkpoint picked by test score; no
testing on training images), and all training settings. The 4 fine-tuning runs see exactly the same batches.
**Be ready to explain:** what each metric measures and whether higher/lower is better; the LOL-v1/LOL-v2-real overlap
(91/100); why the Mac GPU was not used for LLFormer/YOLO.

## V. Results
(A) Retinexformer has the best PSNR on all LOL sets; GSAD best LPIPS. (B) We reproduce Retinexformer, SNR-Aware and
LLFormer to 0.01 dB; GSAD's paper numbers use a ground-truth brightness trick worth 4–8.5 dB. Re-training
Retinexformer gives 23.10 dB vs 25.15 released (same gap as another user, 23.45). (C) NIQE: no single winner.
(D) Detection: raw images are best (0.662 vs ≤0.603); enhancement lowers recall. (E) Retinexformer 151 ms vs GSAD
2.4–4.9 s. (F) Ablation: SNR loss B vs control A: +0.01 dB overall, +0.02 dB in dark regions, wins on 13/15 dark
regions (p = 0.004) — consistent but tiny; FFT loss does nothing.
**Be ready to explain:** why a 0.02 dB gain is "not meaningful" even with p = 0.004 (statistically consistent ≠
practically important); why fine-tuning lowered LOL-v1 PSNR for every run including the control.

## VI. Discussion & Limitations
Why the SNR loss did little: weights only 0.67–1.33; the noise estimate also fires on edges/texture (Fig. 4e);
Retinexformer already knows where it is dark; short fine-tuning. Limitations: 15 test images, one seed, free-GPU
limits, one detector without fine-tuning, GSAD only on 200 detection images.
**Be ready to explain:** the edge/noise confusion (point at Fig. 4e: yellow = high weight on edges).

## VII. Plan  /  VIII. Conclusion
Next: put SNR guidance *inside* the attention (like SNR-Aware), separate noise from edges, sweep α with 3 seeds,
train from scratch, larger test sets, fine-tune the detector on enhanced images.
**Be ready to explain:** why moving the SNR idea from the loss into the network is a sensible next step.
