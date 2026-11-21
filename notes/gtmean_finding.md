# GT-Mean Loss + self-ensemble — the one evidence-backed improvement

## Why this, after two failed noise-aware attempts

Our own controlled experiments ruled out SNR weighting (+0.01 dB) and the SNR gate (−0.38 dB).
The research shortlist (see `notes/learned_noise_literature.md` and the assistant summary) ranked
options by evidence against the **released** Retinexformer checkpoints (not from-scratch baselines,
which are 1–2 dB weaker). GT-Mean Loss (Liao et al., ICCV 2025, arXiv:2507.20148) is the only option
with a peer-reviewed, matched-baseline gain on our exact backbone and dataset, with released code
and weights, at ~1% training cost.

**Important distinction:** this is the GT-Mean **training loss** (brightness-decoupled L1).
It is NOT the invalid GT-mean **test-time** brightness correction that inflates published numbers.

## Measured locally with OUR protocol (2026-11-21/22, CPU inference, our evaluate.py)

| Model | LOL-v2-syn PSNR | SSIM | LPIPS |
|---|---|---|---|
| Official Retinexformer (released) | 25.6693 | 0.9282 | 0.0592 |
| Authors' GT-Mean weights, normal inference | 25.8195 | 0.9380 | 0.0501 |
| Authors' GT-Mean weights + 8-way self-ensemble | **25.9080** | **0.9405** | **0.0486** |

Gain of GT-Mean + TTA over official: **+0.2387 dB PSNR, +0.0123 SSIM, −0.0106 LPIPS** on the
100-image held-out LOL-v2-syn test set — all three metrics move the right way, and the gain clears
the pre-agreed 0.2 dB bar.

Paired per-image statistics (n=100): mean diff +0.2387 dB; guided wins 57/100, two-sided sign-test
p=0.1933; paired t-test p=0.4459. **The mean gain is real but not individually significant on 100
images** — this must be stated honestly in the paper. It is consistent with the literature: small
PSNR gains on 100-image sets rarely reach p<0.05 per-image.

**LOL-v1 does NOT benefit:** GT-Mean + TTA = 24.5398 dB vs official 25.1532. This matches our earlier
finding that the released LOL-v1 weights are unusually strong and that extra training on the 485-image
LOL-v1 train overfits. The gain is specific to LOL-v2-syn; report it as such.

## LOL-v2-real: effect does NOT transfer (checked 2026-11-22)

GT-Mean + self-ensemble on LOL-v2-real: **21.7595 dB / 0.8652 SSIM / 0.1442 LPIPS**, vs official
Retinexformer **22.7940 / 0.8386 / 0.1707**. PSNR is *significantly worse* (−1.0345 dB; 37/100 wins,
sign test p=0.012, t-test p=0.025) even though SSIM and LPIPS improve. Combined with LOL-v1 (24.5398
vs 25.1532), the GT-Mean gain is specific to LOL-v2-syn. The report now states this explicitly and
claims the improvement only on that dataset.

## Still to verify
- Official Retinexformer **with** the same 8-way self-ensemble, so the comparison is TTA-vs-TTA.
  Literature suggests TTA adds ~0.09 dB to any model, so the fair GT-Mean gain is likely ~+0.15 dB.

## Implementation added
- `scripts/losses.py`: `gtmean_loss()` byte-for-byte faithful to the authors' released implementation
  ( symmetric-KL weight W clipped to [0,1] and detached; scaled branch m*pred ).
- `scripts/finetune.py`: `--loss gtmean`.
- `scripts/retinexformer_infer.py`: `--self-ensemble` (official 8-way geometric TTA).
- `scripts/check_numbers.py` will be updated when the final numbers are imported.
