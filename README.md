# Noise-Aware Transformer-Based Low-Light Image Enhancement with Downstream Object Detection Evaluation

B.Tech Project. We reproduce **Retinexformer** (ICCV 2023) and strong baselines (classical, Zero-DCE, SCI, SNR-Aware,
LLFormer, the diffusion model GSAD), evaluate them on **image quality** (PSNR, SSIM, LPIPS, NIQE, dark-region PSNR),
**efficiency** (parameters, time per image on one GPU) and **usefulness for object detection in the dark** (YOLOv8 on
ExDark), and test a **noise-aware SNR-weighted loss** in a controlled fine-tuning experiment.

**Main findings**
1. Released Retinexformer weights reproduce the paper (LOL-v1 **25.15 dB** vs 25.16). Re-training from scratch with the
   official config gives **23.10 dB** — the same gap another user reported (GitHub issue #132, 23.45 dB).
2. GSAD's published numbers use a **ground-truth brightness correction** in its test script (+4 to +8.5 dB). Without it,
   GSAD is below Retinexformer in PSNR on every LOL set, but has the best LPIPS — and is 16–32× slower.
3. **No enhancer helps a COCO-pretrained YOLOv8 in the dark**: raw ExDark images give the best mAP50 (0.662 vs
   0.572–0.603 on 1,200 images; 0.707 vs 0.603–0.633 on a 200-image subset incl. GSAD). The loss is in recall.
4. **SNR-weighted loss:** consistent but negligible gain in the darkest regions (+0.01–0.05 dB, better on 13/15 LOL-v1
   images), no gain overall, on unpaired sets, or in detection. See `notes/contribution_findings.md`.
5. Data issue found: **91 of the 100 LOL-v2-real test images are LOL-v1 training images** — any model trained on
   LOL-v1 must not be evaluated on LOL-v2-real.

## Results (full tables: [`results/tables.md`](results/tables.md))

**LOL-v1 / LOL-v2-real (PSNR dB / SSIM / LPIPS)** — all scored with the same `scripts/evaluate.py`

| Method | LOL-v1 | LOL-v2-real |
|---|---|---|
| Input | 7.77 / 0.191 / 0.560 | 9.72 / 0.196 / 0.519 |
| Gamma (0.4) | 14.53 / 0.620 / 0.351 | 18.70 / 0.659 / 0.318 |
| Zero-DCE | 14.86 / 0.562 / 0.335 | 18.06 / 0.580 / 0.313 |
| SCI | 14.85 / 0.527 / 0.339 | 17.36 / 0.542 / 0.307 |
| SNR-Aware (released by authors) | 24.61 / 0.840 / 0.151 | 21.48 / **0.848** / 0.157 |
| LLFormer | 23.65 / 0.816 / 0.169 | – (LOL-v1 weights; leak) |
| GSAD (no GT correction) | 22.73 / **0.850** / **0.103** | 20.14 / 0.845 / **0.114** |
| **Retinexformer** (released) | **25.15** / 0.843 / 0.131 | **22.79** / 0.839 / 0.171 |
| Retinexformer (my training, 150k it.) | 23.10 / 0.828 / 0.143 | – (leak) |

**Efficiency — one Tesla T4, 600×400:** SCI 1.5 ms (0.0003 M params) · Zero-DCE 19 ms (0.079 M) · Retinexformer 151 ms
(1.61 M) · LLFormer 874 ms (24.55 M) · GSAD 2,360 ms with 10 steps / 4,874 ms with 20 steps (17.44 M).

**Detection — YOLOv8m mAP50 on ExDark (test split, 1,200 images):** raw **0.662** · Retinexformer 0.603 · SCI 0.595 ·
Zero-DCE 0.572. (200-image subset: raw **0.707** · GSAD 0.633 · Retinexformer 0.628 · SCI 0.626 · Zero-DCE 0.603.)

**Ablation (fine-tuning from released LOL-v1 weights, only the loss differs):** LOL-v1 PSNR A (L1) 24.31 · **B (SNR-L1,
mine) 24.32** · C (L1+FFT) 24.31 · D (SNR-L1+FFT) 24.32; dark-30% PSNR A 24.89 · B 24.91.

## Figures (`results/figures/`)
`grid_lolv1.png`, `grid_lime.png` (all methods, zoomed dark regions) · `failures_lolv1.png` · `detection_examples.png` ·
`bars_psnr_ssim.png` · `quality_vs_speed.png` · `training_loss.png`, `training_val_psnr.png` · `finetune_curves.png` ·
`label_check/` (ExDark labels drawn on images) · `detection/<method>/` (YOLO predictions).

## Setup
```bash
conda create -n llie python=3.10 && conda activate llie
pip install torch torchvision opencv-python pillow scikit-image lpips pyiqa einops addict lmdb natsort yacs yapf thop timm gdown ultralytics
# baseline code is cloned into baselines/ (gitignored):
git clone https://github.com/caiyuanhao1998/Retinexformer baselines/Retinexformer   # + Zero-DCE, SCI, LLFormer, GSAD
```
Datasets go in `data/` (gitignored): LOL-v1/v2 and LIME/DICM/MEF from the Retinexformer README links; ExDark from its
GitHub (non-commercial research terms). GPU work runs on **Kaggle** via the notebooks in `notebooks/`.

**Mac (Apple Silicon) warning:** MPS gives wrong results for LLFormer (−2 dB) and YOLOv8 (mAP 0.42 vs 0.65) — the scripts
force CPU there. NIQE needs float64 → CPU. Speeds were measured only on a Kaggle T4.

## How to run (in order)
| Step | Command / notebook |
|---|---|
| Classical baselines + evaluate everything | `bash scripts/eval_all.sh` (uses `classical.py`, `evaluate.py`) |
| Zero-DCE, SCI, LLFormer | `python scripts/run_light_baselines.py --method zerodce --input data/LOLv1/Test/input --dataset LOLv1` |
| Retinexformer (any checkpoint, any folder) | `python scripts/retinexformer_infer.py --repo baselines/Retinexformer --opt <yml> --weights <pth> --input <dir> --output <dir>` |
| Authors' released results (SNR-Aware) | `python scripts/import_released.py --src <dir> --gt <dir> --method snr_aware --dataset LOLv1` |
| Train Retinexformer from scratch | `notebooks/retinexformer_train.ipynb` (Kaggle, auto-resume) → `scripts/plot_training.py` |
| GSAD (diffusion) | `notebooks/gsad_test.ipynb` (saves outputs with and without the GT-brightness correction) |
| ExDark → YOLO | `scripts/exdark_to_yolo.py` → `scripts/check_labels.py` → `scripts/build_yolo_sets.py` → `scripts/yolo_eval.py` |
| Speed on one GPU + GSAD on ExDark-200 | `notebooks/day2_speed_gsad_exdark.ipynb` |
| SNR-loss experiment (runs A–D) | `scripts/losses.py`, `scripts/finetune.py`, `notebooks/day3_finetune.ipynb` |
| Dark-region metric | `python scripts/eval_dark_regions.py --pred <dir> --gt <dir> --low <dir> --method <name> --dataset LOLv1` |
| Tables / plots / grids | `scripts/make_tables.py`, `scripts/plot_results.py`, `scripts/make_grid.py`, `scripts/find_failures.py` |

Every script has a docstring with a usage example. All settings, problems and fixes are logged in
[`notes/experiment_log.md`](notes/experiment_log.md); detection analysis in [`notes/detection_findings.md`](notes/detection_findings.md).

## Repo layout
```
scripts/    all code (see table above)          notebooks/  Kaggle notebooks
results/    results.csv, tables.md, speed.csv, detection*.csv, dark_region.csv, per_image/, figures/, training_logs/
runs/       fine-tuning logs + configs (weights gitignored)     notes/  plans, findings, experiment log
data/, baselines/   datasets and cloned baseline code (gitignored)
```

## References
- Y. Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, ICCV 2023.
- X. Xu et al., *SNR-Aware Low-light Image Enhancement*, CVPR 2022.
- L. Ma et al., *Toward Fast, Flexible, and Robust Low-Light Image Enhancement* (SCI), CVPR 2022.
- C. Guo et al., *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement* (Zero-DCE), CVPR 2020.
- T. Wang et al., *Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method* (LLFormer), AAAI 2023.
- J. Hou et al., *Global Structure-Aware Diffusion Process for Low-Light Image Enhancement* (GSAD), NeurIPS 2023.
- C. Li et al., *Low-Light Image and Video Enhancement Using Deep Learning: A Survey*, IEEE TPAMI 2022.
- Y. P. Loh and C. S. Chan, *Getting to Know Low-light Images with the Exclusively Dark Dataset*, CVIU 2019.
- C. Wei et al., *Deep Retinex Decomposition for Low-Light Enhancement* (LOL), BMVC 2018; W. Yang et al., *Sparse Gradient Regularized Deep Retinex Network* (LOL-v2), TIP 2021.
- G. Jocher et al., *Ultralytics YOLOv8*, 2023.
