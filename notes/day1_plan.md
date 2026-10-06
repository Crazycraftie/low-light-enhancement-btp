# Day 1 Plan — Evaluation Pipeline, All Baselines, Start Training

**Goal:** every baseline evaluated with my own script, and Retinexformer training running on Kaggle before I sleep.

**Output folder rule:** `results/<method>/<dataset>/` (classical: `results/classical/<he|clahe|gamma>/<dataset>/`).
Dataset names: `LOLv1`, `LOLv2-real`, `LOLv2-syn`, `LIME`, `DICM`, `MEF`.

**Download links (Retinexformer README):**
| What | Google Drive |
|---|---|
| LOL-v1 | https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H |
| LOL-v2 | https://drive.google.com/file/d/1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U |
| LIME/NPE/MEF/DICM/VV | https://drive.google.com/drive/folders/1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp |
| Pretrained weights | https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV |
| Baseline results (fallback) | https://drive.google.com/drive/folders/1P75bv6jBp8UxcLDhqMACvIQXck2sS9da |
| Training logs | https://drive.google.com/drive/folders/1HU_wEn_95Hakxi_ze-pS6Htikmml5MTA |

`--GT_mean` uses ground-truth brightness and inflates PSNR. **Never use it for main numbers.**

## Hour 0 — Setup (laptop)
`conda activate llie`; `pyiqa` installed; datasets in `data/` (see table above).

## Hours 1–2 — Retinexformer on Kaggle
- Notebook A `notebooks/retinexformer_test.ipynb`: clone, install, link data, download weights, test 3 LOL sets,
  run on LIME/DICM/MEF with `scripts/retinexformer_unpaired.py`, zip outputs.
- Notebook B `notebooks/retinexformer_train.ipynb`: same setup, `save_checkpoint_freq` ≤ 5000,
  `timeout 37800 python3 basicsr/train.py --opt Options/RetinexFormer_LOL_v1.yml`, then **Save Version → Save & Run All (Commit)**.

## Hours 3–4 — Classical baselines (laptop)
`bash scripts/eval_all.sh` (runs classical.py on all 6 sets + evaluates Input/HE/CLAHE/Gamma).
Checkpoint: `results/results.csv` has rows for Input, HE, CLAHE, Gamma. Classical should beat Input on PSNR.

## Hours 5–6 — Zero-DCE and SCI (`notebooks/zerodce_sci.ipynb`)
Clone official repos, run pretrained weights on all 6 sets → `/kaggle/working/results/<method>/<dataset>`, zip, download.

## Hours 7–8 — SNR-Aware (time limit 2 h) → fallback: author-released results from the Drive above.
## Hour 9 — LLFormer (time limit 1 h) → fallback as above, or UHD-LOL weights as a generalisation test (say so).

## Hours 10–11 — Evaluate everything (laptop)
Unzip into `results/<method>/<dataset>/`, re-run `bash scripts/eval_all.sh`.
Sanity: my Retinexformer PSNR ≈ what `test_from_dataset.py` printed; each method ≈ its paper (≈10 dB off → wrong weights / names).

## Hour 12 — Save and check training
`git add . && git commit -m "Day 1: ..." && git push`. If repo gets heavy, ignore `results/**/*.png`.
Note iterations/second from the Kaggle log → how many 10.5 h sessions the full schedule needs.

## Day 1 done when
- [ ] `results/results.csv` has Input, HE, CLAHE, Gamma, Zero-DCE, SCI, SNR-Aware, LLFormer, Retinexformer on all 3 LOL sets
- [ ] Classical, Zero-DCE, SCI have NIQE on LIME, DICM, MEF
- [ ] Pretrained Retinexformer numbers close to the paper
- [ ] Training running on Kaggle
- [ ] Everything pushed to GitHub
