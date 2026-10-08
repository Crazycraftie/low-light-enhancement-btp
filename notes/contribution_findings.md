# My contribution: SNR-weighted loss — findings (honest version)

**Experiment.** Retinexformer fine-tuned from the released LOL-v1 weights for 10,000 iterations, four times, identical
except for the loss: A = L1 (control), B = SNR-weighted L1 (mine), C = L1 + FFT, D = SNR-L1 + FFT. Same data order
(pre-computed batches, seed 42), same LR schedule, final weights only. Full settings: `notes/experiment_log.md`;
numbers: `results/tables.md` table 3. LOL-v1 has 15 test images, so whole-image differences under ~0.1–0.2 dB are noise.

| | LOL-v1 PSNR | dark-30% PSNR | LOL-v2-syn | NIQE mean | ExDark mAP50 |
|---|---|---|---|---|---|
| start point (released) | 25.15 | 25.31 | 16.19 | 3.64 | 0.627 |
| A: L1 (control) | 24.31 | 24.89 | 16.61 | 3.64 | 0.627 |
| **B: SNR-weighted L1 (mine)** | 24.32 | 24.91 | 16.62 | 3.70 | 0.626 |
| C: L1 + FFT | 24.31 | 24.89 | 16.61 | 3.64 | 0.626 |
| D: SNR-L1 + FFT | 24.32 | 24.91 | 16.62 | 3.70 | 0.626 |

1. **Did B beat A?** Only by a hair. Whole-image PSNR +0.01 dB (B better on 11/15 images, sign test p = 0.12 —
   not significant). In the **darkest 30%** of each image, where the loss puts its weight, B is better on **13/15**
   images (p = 0.007) — a *consistent* effect in the intended place — but the size is tiny: +0.01 to +0.05 dB per
   image (median +0.01). Visually A and B are indistinguishable (`results/figures/failures_lolv1.png`).
   NIQE is slightly worse for B on DICM (3.64 vs 3.51). Detection: no change (0.626 vs 0.627 mAP50).
2. **FFT loss (C) and both (D):** no measurable effect; D ≈ B, C ≈ A.
3. **Fine-tuning itself lowered LOL-v1 PSNR** for every run, including the control (25.15 → 24.31), while training
   L1 improved 12% (0.040 → 0.035) and LOL-v2-syn rose (16.19 → 16.6). With 485 training images the extra training
   fits the training set and moves away from the released checkpoint, which already looked unusually good (our
   from-scratch run: 23.10 dB; GitHub issue #132: 23.45 dB). This affects A and B equally, so the A-vs-B comparison
   stays fair — but no run beats the released model.

**Why so small? (likely reasons)**
- **The re-weighting is gentle by design.** Weights span only 0.67–1.33 and average 1 (to avoid a disguised 2× learning
  rate — the plan's min-max version gave 96% of pixels weight ≈ 2). Dark pixels get only ~13% more weight than average.
- **Retinexformer already models illumination** (its Retinex-based illumination estimator guides attention), so the
  network already "knows" where the image is dark; a loss weight adds little on top.
- **Short fine-tuning at a low LR (2e-5 → 1e-6)** from a converged model leaves little room for any loss to change behaviour.
- **L1 is already robust**, so emphasising some pixels changes the optimum only slightly.

**What to try in the second half of the project**
- Put SNR guidance **inside the network** (e.g. SNR map as an extra input or attention mask, as SNR-Aware does)
  instead of in the loss.
- Sweep **alpha** (2, 4, 8) and a sharper mapping (e.g. top-k darkest pixels only), with **several seeds** per setting
  to measure run-to-run noise.
- Train **from scratch** with the SNR loss (not only fine-tune), and evaluate on a larger paired set than 15 images
  (e.g. LOL-v2-syn train/test or SID) so 0.1 dB effects become measurable.
- A **noise-specific metric** in dark regions (e.g. residual noise variance in flat dark patches) instead of PSNR alone.

**Bottom line for the report:** the noise-aware loss is implemented and tested under a properly controlled setup; it
produces a consistent but practically negligible gain in dark regions (+0.01–0.05 dB) and no gain overall, in
detection, or on unpaired sets. A negative result with a clear diagnosis is still a result — and the analysis above
explains why and what would be more promising.
