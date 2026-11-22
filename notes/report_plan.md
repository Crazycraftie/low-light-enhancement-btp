# Report + Presentation Notes (Days 4–5)

**Deliverables**
1. A **research-paper-style final report**, modelled on the professor's paper (structure, style, formatting).
2. A **presentation**, built with a clean academic template (colours, layouts, look), with speaker notes.

**Inputs**
- `references/prof_paper.pdf` (or .tex) — professor's research paper: **style reference only**
- `references/prof_ppt.pptx` — professor's presentation: **template and style reference only**
- `notes/background_knowledge.md` — theory, literature, research gap, reference list
- `notes/*.md` — survey summary, paper notes, experiment log, detection & contribution findings
- `results/tables.md`, `results/*.csv`, `results/figures/` — all numbers and figures
- `CLAUDE.md`, `README.md`, `scripts/` — project description and methods

## Ground rules (apply to everything)

1. **Never invent or estimate a number.** Every value must come from `results/*.csv` or `results/tables.md`.
   If something is missing, leave `[MISSING: ...]` and tell me.
2. **Never copy text** from my professor's paper or any research paper. Use the professor's documents only
   for structure, section order, formatting, figure/table style, citation style and slide design.
3. **Don't reuse figures from papers.** Draw our own diagrams (architecture, pipeline). If a figure is
   adapted from a paper, the caption says "Adapted from [ref]".
4. **Verify every reference** (DBLP / Google Scholar / official page). No reference from memory.
5. **Final framing:** this is the completed work with all experiments finished. Limitations,
   shortened training, compute limits, small test sets, and results that did not improve are stated honestly.
6. As a beginner defending the work in a viva, a 5-line plain-language summary was provided after each
   section, flagging anything to be ready to explain.
7. Keep `references/` out of git (add it to `.gitignore`) — professor's files must not go to GitHub.

---

## Phase 0 — Study the references (30 min)

**Professor's paper** → `report/style_notes.md` was written:
- Format (LaTeX/IEEE two-column? Word? single-column?), page count, section names and order,
  abstract length, how figures/tables are captioned and numbered, citation style ([1] vs author-year),
  equation style, tone.
- It was LaTeX/IEEE, so the report was built in LaTeX to match its format.

**Professor's PPT** (inspected with `python-pptx` where available) → added to `style_notes.md`:
- Slide size, available layouts (names), colours, fonts, title style, logo/footer, slide count,
  section order, how much text per slide, how results are shown.

`style_notes.md` was shown before continuing.

---

## Phase 1 — Report outline (approved before writing)

Final structure (section names/order adapted from the professor's paper):

1. **Title, author, supervisor, institute**
2. **Abstract** (~150–200 words): problem, approach, key quantitative results, contribution status.
3. **Introduction:** importance of low-light enhancement; challenges (darkness, noise, colour);
   limits of classical methods; why deep learning; contributions of this work as a bullet list.
4. **Related Work:** classical → supervised CNN → unsupervised/zero-shot → transformer → diffusion
   (from `background_knowledge.md` §2, 3, 9, 10). End with the research gap.
5. **Background:** Retinex theory; self-attention and why it's costly for images; channel-wise attention.
6. **Method:**
   - Retinexformer: ORF (illumination estimator + corruption restorer), IGT, IG-MSA — with our own diagram.
   - **Proposed noise-aware loss:** SNR map equations, weighting `w = 1 + α(1 − SNR_norm)`,
     SNR-weighted L1, FFT loss variant — with equations and a small diagram.
   - Detection-based evaluation protocol (ExDark → COCO class mapping, YOLOv8).
7. **Experimental Setup:** datasets table; metrics table (↑/↓); baselines; implementation details
   (Kaggle GPU, PyTorch version, iterations, lr, batch, crop, seed, α); training schedule notes.
8. **Results:**
   - Quantitative comparison (LOLv1, LOLv2-real, LOLv2-syn) — table
   - No-reference results (LIME, DICM, MEF) — table
   - Qualitative comparison — grids with zoomed crops
   - Reproduction of Retinexformer — mine vs pretrained vs paper, with explanation of the gap
   - Detection results — table + example figure; discuss PSNR ranking vs detection ranking
   - Efficiency — params, ms/image, quality-vs-speed plot
   - **Ablation (control vs SNR vs FFT vs both)** incl. dark-region PSNR and win counts
9. **Discussion & Limitations:** what worked, what didn't, why; small test set; shortened training;
   failure cases figure.
10. **SNR-Guided Attention: Results:** gate design, sweep over α, matched from-scratch runs,
    and the held-out test evaluation.
11. **Conclusion**
12. **References** (BibTeX, verified)

The outline with figures/tables per section was reviewed before writing.

---

## Phase 2 — Figures for the report

- **Pipeline overview figure:** dataset → enhancers → evaluation (fidelity, perceptual, no-reference,
  dark-region, detection).
- **Retinexformer architecture diagram** (our own drawing: illumination estimator → light-up → IGT with IGABs/IG-MSA).
- **SNR-weighted loss diagram:** dark input → SNR map → weight map → weighted L1. A real
  example SNR map and weight map computed from one LOL image were included (`scripts/viz_snr.py`).
- Figures from `results/figures/` were reused (grids, failures, curves, bars, quality-vs-speed, detection).
- All figures: vector PDF or 300-dpi PNG, readable fonts, consistent method names and colours
  across every figure and table.

---

## Phase 3 — Write the report, one section at a time

- Order: Method → Experimental Setup → Results → Discussion → Related Work → Introduction → Abstract → Conclusion
  (the abstract was written last, once results were final).
- After each section: it was compiled and a 5-line plain summary + "things to be ready to explain" was provided.
- Each section was reviewed before moving on.
- Tables were generated from CSVs by a script (`scripts/make_latex_tables.py` or equivalent), not typed by hand.

**Build:** LaTeX project in `report/main.tex`, `report/refs.bib`, `report/figures/`; compiled locally and
exported to `report/report.pdf`, with an Overleaf-ready zip also produced.

---

## Phase 4 — Presentation (built from a clean academic template)

The deck was built with `python-pptx` from a clean academic template, keeping slide masters/layouts
consistent so colours, fonts and logos match; the professor's PPT was a style reference only.
Output: `presentation/final.pptx`. The slide outline below follows the professor's slide order and density.

Final deck: **21 slides, 12–15 minute talk.**

| # | Slide | Content |
|---|---|---|
| 1 | Title | Title, my name, supervisor, date |
| 2 | Motivation | One striking before/after image + why it matters (night photos, surveillance, driving, detection) |
| 3 | Problem | Darkness + noise + colour shift; why brightening amplifies noise (zoomed example) |
| 4 | Literature evolution | Timeline: classical → CNN → unsupervised → transformer → diffusion |
| 5 | Literature summary | Compact table: key methods, idea, limitation |
| 6 | Research gap | 3–4 gaps + where my project fits |
| 7 | Objectives | What this work covers |
| 8 | Background | Retinex theory + why attention is costly / channel attention (simple diagrams) |
| 9 | Retinexformer | Our architecture diagram, ORF + IG-MSA in 3 bullets |
| 10 | Proposed noise-aware loss | SNR map → weight map example images + the formula |
| 11 | Experimental setup | Datasets, metrics (↑/↓), baselines, hardware |
| 12 | Quantitative results | Main table, best values highlighted |
| 13 | Qualitative results | One grid with zoomed crops |
| 14 | Detection results | mAP table + raw vs enhanced detection example |
| 15 | Ablation | Control vs SNR vs FFT; dark-region PSNR; honest takeaway |
| 16 | Efficiency | Quality-vs-speed plot |
| 17 | Limitations & failures | Failure crops + honest limits |
| 18 | SNR-guided attention results | Gate design, sweep and held-out test outcome |
| 19 | Conclusion + references | Key takeaways; main references |

Slide rules: max ~5 short bullets per slide; figures large; every number from the CSVs;
same method names/colours as the report; slide numbers on.

**Speaker notes on every slide:** what was said in simple spoken English (~1 min per slide), plus
1–2 likely panel questions for that slide with short answers.

---

## Phase 5 — Consistency check and viva prep

1. `scripts/check_numbers.py` was written: every number in the report and slides matches `results/*.csv`;
   mismatches were reported and resolved.
2. Figure/table numbering, all figures referenced in text, every citation in the bib and vice versa, and
   method names were checked for consistency.
3. `presentation/viva_prep.md` was created: 20 likely questions with short answers in simple words
   (problem choice, Retinexformer vs earlier Retinex methods, attention cost, SNR map, why control run,
   PSNR/SSIM limits, gap to paper numbers, 15-image test set confidence, detection findings,
   and the SNR-loss results).
4. `CLAUDE.md` status was updated. Changes were committed and pushed (no `references/`, no weights).

---

## Day split
- **Day 4:** Phase 0 → 1 → 2 → 3 (report completed).
- **Day 5:** Phase 4 → 5 (slides, consistency check, viva prep); the talk was rehearsed at least twice.
