# SNR-Aware (CVPR 2022) published SNR map tested in Retinexformer's attention

The user asked whether a *published* noise-aware design could work where our own SNR gate failed.
We reproduced the exact SNR map from the authors' code (models/base_model.py in
JIA-Lab-research/SNR-Aware-Low-Light-Enhance) and screened it through our parameter-free IG-MSA gate,
with the official LOL_v2_synthetic.pth weights, on the deterministic 90-image validation split.

Published map: dark = 0.299R+0.587G+0.114B; light = blur5(dark); noise = |dark-light|;
SNR = light/(noise+1e-4); mask = SNR / global_max(SNR) in [0,1]. Gate h = 1+alpha*(1-mask), mean-normalised.

| alpha | validation PSNR (dB) |
|---|---|
| 0 (no gate) | **28.1173** |
| 1 | 28.0547 |
| 2 | 28.0225 |
| 4 | 27.9984 |
| 8 | 27.9831 |

Result: **the ungated control is best**, and PSNR falls monotonically with gate strength. The published
SNR map does not help either — consistent with our own rank-based gate (control 25.21 vs guided 24.83
on the 100-image test) and the SNR-weighted loss (+0.01 dB overall). Three independent noise-aware
signals all fail to improve this architecture because Retinexformer already estimates illumination
internally; the extra map adds no usable information. Data: `results/snr_aware_screen.csv`.
Code: `scripts/snr_aware_map.py`, `scripts/snr_aware_screen.py`.

Note: the full SNR-Aware network (CVPR 2022) is a weaker baseline than Retinexformer on all our sets
(24.61/21.48/24.14 dB vs 25.15/22.79/25.67), so reimplementing it would not produce an improvement.
