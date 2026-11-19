# Recent literature supporting the learned noise-aware attention attempt

Consulted 2026-11-19 to design the second contribution (`scripts/learned_noise.py`).
These are *motivation/design references*, not results copied from them.

| Paper | Venue | Relevance to our design |
|---|---|---|
| **DarkIR: Robust Low-Light Image Restoration** (Feijoo et al., CVPR 2025) | CVPR 2025 | Shows RetinexFormer fails under noise/blur; argues for robustness to real degradations. Motivates noise-aware design. |
| **An illumination-guided dual attention vision transformer for LLIE (IGDFormer)** (Wen et al., Pattern Recognition 2025) | PR 2025 | Extends one-stage Retinex with *disturbance terms* for "noise hidden in darkness" and corruptions from light-up; uses illumination-guided dual attention. Closest prior art to our idea — we must cite and differentiate (we use a CNN-estimated noise map fused into IG-MSA values, not dual attention). |
| **DiffLight: Integrating Content and Detail for LLIE** (Feng et al., CVPRW 2024) | CVPRW 2024 | Dual branch: a *denoising enhancement* branch + a detail-preservation branch. Supports a separate learned noise branch rather than a fixed multiplier. |
| **PPformer: pixel-wise and patch-wise cross-attention** (Dang et al., CVIU 2024) | CVIU 2024 | Notes that lightweight attention still "produces artifacts and noise when enhancing extremely dark images"; suggests integrating Retinex theory into attention. |
| **CWNet: Causal Wavelet Network** (Zhang et al., ICCV 2025) | ICCV 2025 | Frequency-domain recovery; supports loss/frequency ablations if the learned gate helps. |
| **RetinEV: Event-Based Illumination Estimation** (Sun et al., ICCV 2025) | ICCV 2025 | Fine-grained illumination cues improve reflectance enhancement; again shows illumination guidance is the right place for extra signals. |

## Design decision

The parameter-free SNR gate (`snr_guidance.py`) failed because a fixed multiplier adds no capacity
to interpret the noise signal, and the blur-residual map confuses edges with noise. Following
IGDFormer / DiffLight, we instead **learn** the noise map with a small CNN and fuse it into
IG-MSA value features through learned 1x1 convolutions. New parameters are separate from the
released checkpoint, so pretrained weights load exactly and only the new modules are random at
init. `--noise-scale 0` is bit-identical to the official model (verified).

**Protocol (agreed): select the variant by validation PSNR only; evaluate the held-out test set once.**
If it wins on validation and then on test with a paired sign test, we report an improvement.
Otherwise this becomes a third controlled negative result in the paper.
