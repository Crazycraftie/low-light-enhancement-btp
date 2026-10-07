# Detection findings — does enhancement help YOLOv8 in the dark? (ExDark)

**Setup.** COCO-pretrained YOLOv8m, no fine-tuning; 1,200 ExDark *test-split* images (100 per class,
seed 0, ≤1024 px, 3,776 boxes); same images and labels for every method, only pixels change.
`imgsz=1024, conf=0.001, iou=0.6`, predictions restricted to the 12 ExDark classes. Run on CPU
(MPS gave wrong mAP: 0.42 vs 0.65 on the same 100 images). Scripts: `exdark_to_yolo.py`,
`build_yolo_sets.py`, `yolo_eval.py`. Raw numbers: `results/detection.csv`, `results/detection_per_class.csv`.

| Input to YOLOv8m | mAP50 | mAP50-95 | Precision | Recall | LOLv1 PSNR of the enhancer |
|---|---|---|---|---|---|
| Raw (dark) | **0.662** | **0.345** | 0.682 | **0.611** | 7.77 (input) |
| Retinexformer (LOL-v1 weights) | 0.603 | 0.310 | 0.675 | 0.540 | 25.15 |
| SCI | 0.595 | 0.302 | 0.678 | 0.529 | 14.85 |
| Zero-DCE | 0.572 | 0.291 | 0.671 | 0.502 | 14.86 |

**Findings**
1. **No enhancer helps; all three lower mAP50** (−5.9 to −9.1 points). The detector does better on the
   original dark images than on any "improved" version.
2. **The drop is almost entirely recall** (0.611 → 0.50–0.54) while precision barely moves (~0.67–0.68):
   enhancement makes YOLO *miss* objects rather than hallucinate them.
3. **Among enhancers, the ranking follows image quality:** Retinexformer (best PSNR) hurts least, Zero-DCE
   hurts most — but even the best restoration model does not beat raw. Image quality for humans ≠
   usefulness for a machine.
4. **Per class:** every class gets worse except **boat**, which improves with every enhancer (0.485 → 0.51–0.54).
   Large drops for bicycle, motorcycle, cat, chair.
5. **Likely reasons (hypotheses, not proven here):** brightening amplifies sensor noise in dark regions
   (the noise problem our SNR-weighted loss targets), shifts colours, and moves images away from what a
   COCO-trained detector expects, while YOLOv8 is already fairly robust to darkness. Literature on
   "enhance-then-detect" reports similar mixed/negative results for detectors that are not fine-tuned.

**Second comparison: 200-image subset incl. GSAD (diffusion).** GSAD needs ~13.7 s/image on a T4 at ExDark
size, so all methods were re-scored on a fixed 200-image subset (`results/exdark_subset200.txt`, seed 0; the
Kaggle-rebuilt images were verified pixel-identical by SHA-1). Compare rows **within** this table only — the
subset is slightly easier than the 1,200 (raw 0.707 vs 0.662).

| Input to YOLOv8m (200 imgs) | mAP50 | mAP50-95 | Precision | Recall | Time/img (T4) |
|---|---|---|---|---|---|
| Raw (dark) | **0.707** | **0.378** | 0.700 | **0.637** | — |
| GSAD (diffusion, LOL-v1 weights, 20 steps) | 0.633 | 0.342 | 0.668 | 0.560 | 13.7 s (ExDark size) |
| Retinexformer (LOL-v1 weights) | 0.628 | 0.340 | 0.677 | 0.553 | 0.15 s (600×400) |
| SCI | 0.626 | 0.330 | 0.630 | 0.577 | 0.002 s |
| Zero-DCE | 0.603 | 0.317 | 0.711 | 0.513 | 0.019 s |

6. **The diffusion model does not change the conclusion:** GSAD is the best enhancer for detection by a hair
   (0.633 vs 0.628 for Retinexformer — within noise on 200 images) and still 7.4 points below raw, while being
   ~90× slower than Retinexformer. Same pattern: recall drops, precision roughly unchanged.

**Limitations.** One detector, zero-shot (no fine-tuning on enhanced images), one random subset
(no confidence intervals), noisy ExDark labels (loose boxes, "person" on posters — see
`results/figures/label_check/`), enhancers trained on LOL (domain gap to ExDark's mixed lighting).
SNR-Aware not included (only released LOL outputs exist). GSAD only on the 200-image subset (speed).
