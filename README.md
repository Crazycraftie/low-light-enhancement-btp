# Noise-Aware Transformer-Based Low-Light Image Enhancement with Downstream Object Detection Evaluation

B.Tech Project — Dhirendra Pratap Singh (2301EE49), supervisor Dr. Rajib Kumar Jha, IIT Patna.

Reproduction of Retinexformer (ICCV 2023) and eight baselines under one evaluation protocol, a
noise-aware SNR-weighted loss with a controlled dark-region study, and a downstream object-detection
study on ExDark. Final deliverables: `report/report.pdf`, `presentation/final.pptx`, `presentation/final.pdf`.

## Main results

- Retinexformer (released) matches the paper: **25.15 dB** on LOL-v1 (paper 25.16), **25.67 dB** on LOL-v2-syn.
- Noise-aware SNR-weighted loss: consistent, statistically significant dark-region gain — **13/15** images,
  one-sided sign test **p=0.004**, +0.01–0.05 dB in the darkest 30%, with global fidelity preserved.
- Official 8-way self-ensemble: **25.6693 → 25.8269 dB** on LOL-v2-syn (+0.1576 dB, **64/100** images,
  sign test **p=0.0066**, paired t-test p=0.0015).
- Protocol findings: GSAD's published numbers rely on a ground-truth brightness correction; **91/100**
  LOL-v2-real test images occur in LOL-v1 training; raw ExDark images detect best (mAP50 **0.662**).

Full numbers, tables and figures are in `results/`.

## Repository layout

```
scripts/       enhancement, losses, evaluation, tables, slides, plotting
notebooks/     GPU experiment notebooks (Retinexformer training, SNR-loss fine-tuning, GSAD, speed/detection)
results/       results.csv, dark_region.csv, detection.csv, per-image scores, figures, tables
report/        IEEE LaTeX source, PDF, bibliography, Overleaf copy
presentation/  final deck (pptx + pdf), speaker notes (viva_prep.md)
baselines/     cloned baseline code (gitignored)
data/          datasets (gitignored)
notes/         experiment log, findings, background
```

`scripts/check_numbers.py` verifies that every decimal number in the report, slides and this file is
traceable to `results/*.csv`. All reported numbers are reproducible from the scripts and CSVs in the repo.
