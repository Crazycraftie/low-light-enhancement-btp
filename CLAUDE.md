# CLAUDE.md — BTP: Low-Light Image Enhancement

## About me and how to help
- B.Tech student doing my BTP (B.Tech Project). Beginner in deep learning and image processing.
- Midsem review panel has high expectations: I must be able to EXPLAIN everything, not just run it.
- Explain code and errors step by step in simple words. When you write code, comment it well.
- Timeline: 3 days of coding (Day 1–3 below), then 2 days for report + PPT.

## Project
**Title:** Noise-Aware Transformer-Based Low-Light Image Enhancement with Downstream Object Detection Evaluation

**Idea:** Reproduce Retinexformer (ICCV 2023) and strong baselines, evaluate on image quality AND on
whether enhancement helps object detection in the dark, then pilot a noise-aware improvement
(SNR-weighted loss) motivated by noise amplification in dark regions.

**Main paper:** Retinexformer — https://github.com/caiyuanhao1998/Retinexformer
**Related:** SNR-Aware (CVPR 2022), SCI (CVPR 2022), Zero-DCE (CVPR 2020), LLFormer (AAAI 2023),
Diff-Retinex (ICCV 2023) / GSAD (NeurIPS 2023), Li et al. LLIE survey (IEEE TPAMI).

## Environment
- MacBook (Apple Silicon). Conda env `llie` (Python 3.10). PyTorch MPS available (`True`).
  Always `conda activate llie` and check the prompt shows `(llie)` before installing anything.
- Heavy jobs (training, deep model inference) run on **Kaggle** GPU notebooks (~30 GPU h/week,
  12 h session limit, outputs saved only via "Save Version"). One notebook per model.
- Repo: `~/Desktop/low-light-enhancement-btp` (GitHub, pushed daily).

## Repo structure and conventions
```
data/        # datasets — gitignored, never commit
baselines/   # cloned code from other papers
scripts/     # my scripts
notebooks/   # Kaggle notebooks
results/     # outputs: results/<method>/<dataset>/  + results/results.csv
notes/       # paper notes, literature review, day plans
```
- `.gitignore` excludes `data/`, `*.pth`, `*.pt`, `*.zip`.
- Output folder rule: `results/<method>/<dataset>/` (e.g. `results/zerodce/LOLv1/`).
  Exception: classical methods live in `results/classical/<he|clahe|gamma>/<dataset>/`.
- Dataset names used in tables: `LOLv1`, `LOLv2-real`, `LOLv2-syn`, `LIME`, `DICM`, `MEF`.

## Datasets (structure follows Retinexformer README)
- LOL-v1: `data/LOLv1/{Train,Test}/{input,target}` — 485 train / 15 test pairs
- LOL-v2: `data/LOLv2/{Real_captured,Synthetic}/{Train,Test}/{Low,Normal}` — real 689/100, syn 900/100
  (LOLv2-real files are named `low00690.png` / `normal00690.png`; `evaluate.py` strips the prefix to match.)
- **LEAKAGE:** 91/100 LOLv2-real TEST images are identical to LOLv1 TRAIN images (checked 2026-10-06).
  Never report a LOLv1-trained model on LOLv2-real as a real result (LLFormer row is marked INVALID).
- Unpaired (NIQE only): `data/unpaired/{LIME,DICM,MEF}` — links in Retinexformer README
- ExDark (detection, 12 classes, map to COCO ids, e.g. People→person, Motorbike→motorcycle, Table→dining table)

## Existing scripts
- `scripts/classical.py` — HE, CLAHE, gamma.
  `python scripts/classical.py --input data/LOLv1/Test/input --dataset LOLv1`
- `scripts/evaluate.py` — PSNR/SSIM/LPIPS (paired, `--gt`), NIQE (`--niqe`, uses pyiqa).
  Appends/updates one row per method+dataset in `results/results.csv`; per-image scores in `results/per_image/`.
  `python scripts/evaluate.py --pred results/classical/clahe/LOLv1 --gt data/LOLv1/Test/target --method CLAHE --dataset LOLv1`
- `scripts/eval_all.sh` — runs classical + evaluate for every method × dataset that has outputs
  (`--no-classical` to only evaluate).
- `scripts/run_light_baselines.py --method zerodce|sci|llformer` — official weights, writes `results/<method>/<dataset>/`
  and `results/timing.csv`. Repos cloned in `baselines/`. SCI uses `medium.pt`; LLFormer weights are LOL-v1-trained.
- `scripts/retinexformer_infer.py --opt <yml> --weights <pth> --input --output [--method]` — any Retinexformer
  checkpoint on any folder; architecture from yml `network_g`, pad = yml `val.window_size` (4, as official test);
  appends speed + params to `results/speed.csv`. LOL sets use matching weights; LIME/DICM/MEF use `LOL_v1.pth`.
- `scripts/exdark_to_yolo.py` → `data/exdark_yolo/raw/` (ExDark TEST split, 100/class, seed 0, ≤1024 px, COCO ids,
  EXIF orientation NOT applied). `scripts/check_labels.py` → `results/figures/label_check/`.
- `notebooks/gsad_test.ipynb` — GSAD on Kaggle; saves `noadjust/` (main) and `gtmean/` (authors' GT-brightness trick).
- `scripts/import_released.py` — imports authors' released result images → `results/<method>_released/<dataset>/`.
  Used for SNR-Aware (Retinexformer README "results of compared methods" Drive).
- `notebooks/retinexformer_train.ipynb` — Kaggle training; auto-resumes if previous version output is attached.

## Mac gotchas (8 GB Apple Silicon)
- pyiqa NIQE needs float64 → MPS unsupported → `evaluate.py` runs NIQE on CPU.
- LLFormer gives WRONG outputs on MPS (up to −2 dB PSNR) → runs on CPU (~17 s/image).
- Big images (LIME 2000×1500) swap the Mac → scripts call `torch.mps.empty_cache()` per image.
- Mac timings are unreliable (swap) → measure speed for all methods on one Kaggle GPU later.
- zsh does not word-split `$var` → loops using `set -- $e` must run under `bash`.
- `ultralytics` (YOLOv8) pins numpy to 1.26.4 here (macOS excludes numpy 2.0–2.3.4; 2.3.5+ needs Python 3.11).
  Verified 2026-10-07: all metrics identical after the downgrade. Don't "fix" the pip warning about opencv-headless.

## Methods to compare
Input (reference), HE, CLAHE, Gamma, Zero-DCE, SCI, SNR-Aware, LLFormer, Retinexformer (pretrained),
Retinexformer (my training), one diffusion model (GSAD or Diff-Retinex), my SNR-weighted variant + control.
Metrics: PSNR, SSIM, LPIPS, NIQE, inference time, params/FLOPs, detection mAP@0.5 and mAP@0.5:0.95.

## Retinexformer notes (Kaggle)
- Setup: clone → `pip install einops gdown addict future lmdb natsort yacs yapf lpips thop timm` →
  `python setup.py develop --no_cuda_ext`. Kaggle uses PyTorch 2 (paper used 1.11): small metric
  differences are expected — mention in report.
- Link data instead of editing configs: `ln -s /kaggle/input/<dataset>/LOLv1 data/LOLv1` (same for LOLv2).
- Test: `python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_LOL_v1.yml --weights pretrained_weights/LOL_v1.pth --dataset LOL_v1`
- Train: `timeout 37800 python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml`
  (stops before Kaggle's 12 h limit so checkpoints are saved; resume next session via `resume_state` in the yml).
- NEVER use `--GT_mean` for main numbers (uses ground-truth brightness; inflates PSNR).
- Fallback if SNR-Aware/LLFormer won't install (2 h / 1 h limit): use baseline results shared in the
  Retinexformer README Google Drive, evaluate with my script, label "results released by authors".

## 3-day coding plan
Full Day 1 plan: `notes/day1_plan.md`.
**Day 1 — baselines + start training:** launch Retinexformer training on Kaggle first; classical baselines +
evaluate on all LOL sets and unpaired sets; run Zero-DCE, SCI, SNR-Aware, LLFormer, pretrained Retinexformer;
evaluate everything; sanity-check against paper numbers; commit.
**Day 2 — diffusion + detection:** resume training; run GSAD or Diff-Retinex (record time/image);
convert ExDark to YOLO format (1,000+ images), run YOLOv8 on raw vs each enhanced version, record mAP.
**Day 3 — contribution + figures:** evaluate my trained Retinexformer; SNR-weighted loss (SNR map as in
SNR-Aware: blur input, noise = |input − blurred|, SNR = blurred / noise; upweight low-SNR regions);
fine-tune control (original loss) and mine for the SAME iterations from pretrained weights; optional FFT-loss
variant (ablation). Figures: comparison grids, failure-case crops, loss curves, PSNR/SSIM bars,
quality-vs-speed plot, detection examples. Clean README.

Fallback order if behind: drop diffusion → second loss variant → full-size ExDark. Everything else is core.

## Current status
- Done: env, folder structure, survey, all datasets in `data/`; classical, Zero-DCE, SCI, Retinexformer (pretrained)
  on all 6 sets; SNR-Aware (released) and LLFormer on LOL sets. Reproduced paper numbers on LOL-v1
  (Retinexformer 25.15 vs 25.16, SNR 24.61, LLFormer 23.65, Zero-DCE 14.86, SCI 14.85).
- Training session 1 launched on Kaggle 2026-10-06 18:51 UTC (notebook `retinexformer-train`, T4,
  0.265 s/iter, ~135–140k of 150k iters per 10.5 h session; checkpoints every 5k). Experiment folder is
  `experiments/RetinexFormer_LOL_v1` (BasicSR names it after the yml FILE, options.py:47).
- Next (Day 2): attach session-1 output as input → Save & Run All again (auto-resumes, ~1.5 h left);
  measure all methods' speed on one Kaggle GPU; diffusion baseline; ExDark detection.
