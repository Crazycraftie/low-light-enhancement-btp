# Report outline (IEEE two-column, LumiNet-style structure + extras good LLIE papers have)

Title: *Noise-Aware Transformer-Based Low-Light Image Enhancement with Downstream Object Detection Evaluation*
(final research report). Author / roll no. / supervisor / institute / date: as in the submitted paper.

| § | Section | Content | Figures / tables |
|---|---|---|---|
| – | Abstract (~200 w) + Index Terms | problem, what we did, key numbers, honest contribution status | – |
| I | Introduction | why low light matters; darkness+noise+colour; classical limits; deep learning; research gap | Fig. 1 teaser (dark input → enhanced, zoom shows amplified noise) |
| I-A | Related Work | classical → supervised CNN/deep Retinex → unsupervised/zero-shot → transformers → diffusion; enhancement for detection | Table I literature summary (method, idea, limitation) |
| I-B | Contributions and organisation | 5 bullets (benchmark under one protocol; protocol findings: GT-mean correction and LOL-v2 overlap; detection study; controlled SNR-weighted loss study; matched from-scratch SNR-guided attention study) | – |
| II | Background | Retinex model; self-attention cost O(N²) vs channel attention O(N·C²) | – |
| III | Methodology | A. overview pipeline; B. Retinexformer (ORF, IGT, IG-MSA) from code; C. proposed SNR-weighted loss (eqs + design decisions); D. FFT loss; E. detection protocol | Fig. 2 pipeline; Fig. 3 architecture; Fig. 4 SNR map/weight example |
| IV | Experimental Setup | datasets, metrics (with eqs), baselines, implementation (Kaggle T4, PyTorch 2.11, iters, lr, batch, crop, seed, α), evaluation protocol rules | Table II datasets; Table III metrics ↑/↓; Table IV fine-tuning settings |
| V | Results | A. quantitative (LOL); B. no-reference; C. qualitative; D. reproduction of Retinexformer; E. GSAD protocol; F. detection; G. efficiency; H. ablation | Tables V–IX; Figs 5–10 (grid, LIME grid, training curves, detection examples, quality–speed, fine-tuning curves) |
| VI | Discussion and limitations | what worked/didn't and why; failure cases; small test set; compute limits; incidents | Fig. 11 failures |
| VII | SNR-Guided Attention: Results | gate design; sweep over α; matched from-scratch runs; held-out test evaluation and negative result | Table X (SNR sweep and final) |
| VIII | Conclusion | – | – |
| – | References | 28 verified (Crossref / arXiv / official pages) | – |

All tables from `scripts/make_latex_tables.py` (reads `results/*.csv`); every number traceable to a CSV.
