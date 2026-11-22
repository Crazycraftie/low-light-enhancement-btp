# CLAUDE.md — BTP: Low-Light Image Enhancement

## About me and how to help
- B.Tech student; final review panel expects me to EXPLAIN everything, not just run it.
- Explain code and errors step by step in simple words. Comment code well.
- This repo is the final, submitted research project. Keep it clean and consistent.

## Project
**Noise-Aware Transformer-Based Low-Light Image Enhancement with Downstream Object Detection Evaluation.**
Reproduce Retinexformer (ICCV 2023) and strong baselines under one protocol, test a noise-aware
SNR-weighted loss, and evaluate enhancement for object detection on ExDark.

Main paper: https://github.com/caiyuanhao1998/Retinexformer

## Final results (all in results/*.csv)
- Reproduction: released Retinexformer 25.15 dB LOL-v1 (paper 25.16), 25.67 dB LOL-v2-syn.
- SNR-weighted loss: dark-30% PSNR 24.89 (control) -> 24.91 (SNR); 13/15 dark wins, one-sided sign p=0.004.
- Self-ensemble (official 8-way TTA): 25.6693 -> 25.8269 dB LOL-v2-syn; 64/100 wins, sign p=0.0066.
- Protocol: GSAD GT-brightness trick; 91/100 LOL-v2-real test images in LOL-v1 train; raw ExDark mAP50 0.662 > enhancers.
- `scripts/check_numbers.py` verifies all decimal numbers in report/slides/README are traceable to CSVs.

## Conventions
- Output folder: `results/<method>/<dataset>/`; classical in `results/classical/<he|clahe|gamma>/<dataset>/`.
- Dataset names: `LOLv1`, `LOLv2-real`, `LOLv2-syn`, `LIME`, `DICM`, `MEF`.
- Dataset folder `data/` is gitignored; weights (`*.pth`) and zips are gitignored.
- Working tree should stay clean; commit and push after every change.

## Deliverables
- `report/report.pdf` — IEEE two-column paper; source `report/main.tex`, `report/refs.bib`.
  Build: `cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.
- `presentation/final.pptx` + `presentation/final.pdf` — built by `scripts/make_slides.py`.
- `presentation/viva_prep.md` — slide-by-slide speaker notes and Q&A.
- `notes/` — experiment log, findings, background (not submitted).

## Environment
- Mac (Apple Silicon), conda env `llie`, Python 3.10. Always `conda activate llie`.
- Heavy jobs ran on Kaggle GPUs. Mac gotchas: NIQE on CPU; large images need `torch.mps.empty_cache()`.
