# Report + Presentation Instructions (Day 4–5)

**Deliverables**
1. A **research-paper-style midsem report**, modelled on my professor's paper (structure, style, formatting).
2. A **presentation**, built on my professor's PPT (template, layouts, look), with speaker notes.

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
5. **Midsem framing:** this is work done so far + plan for the second half. Be honest about limitations,
   shortened training, compute limits, small test sets, and results that didn't improve.
6. I'm a beginner and must defend this in a viva: after each section, give me a 5-line plain-language
   summary of what it says, and flag anything I should be able to explain.
7. Keep `references/` out of git (add it to `.gitignore`) — professor's files must not go to GitHub.

---

## Phase 0 — Study the references (30 min)

**Professor's paper** → write `report/style_notes.md`:
- Format (LaTeX/IEEE two-column? Word? single-column?), page count, section names and order,
  abstract length, how figures/tables are captioned and numbered, citation style ([1] vs author-year),
  equation style, tone.
- If it's LaTeX/IEEE: build the report in LaTeX. Otherwise match its format.

**Professor's PPT** (inspect with `python-pptx`) → add to `style_notes.md`:
- Slide size, available layouts (names), colours, fonts, title style, logo/footer, slide count,
  section order, how much text per slide, how results are shown.

Then **stop and show me `style_notes.md`** before continuing.

---

## Phase 1 — Report outline (approve before writing)

Proposed structure (adapt section names/order to the professor's paper):

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
10. **Plan for the Remaining Semester:** next steps (e.g. SNR guidance inside IG-MSA attention,
    tuning α, more data, full training) as a timeline table.
11. **Conclusion**
12. **References** (BibTeX, verified)

Show me the outline with planned figures/tables per section. **Wait for approval.**

---

## Phase 2 — Figures for the report

- **Pipeline overview figure:** dataset → enhancers → evaluation (fidelity, perceptual, no-reference,
  dark-region, detection).
- **Retinexformer architecture diagram** (our own drawing: illumination estimator → light-up → IGT with IGABs/IG-MSA).
- **SNR-weighted loss diagram:** dark input → SNR map → weight map → weighted L1. Include a real
  example SNR map and weight map computed from one LOL image (add a small script `scripts/viz_snr.py`).
- Reuse figures from `results/figures/` (grids, failures, curves, bars, quality-vs-speed, detection).
- All figures: vector PDF or 300-dpi PNG, readable fonts, consistent method names and colours
  across every figure and table.

---

## Phase 3 — Write the report, one section at a time

- Order: Method → Experimental Setup → Results → Discussion → Related Work → Introduction → Abstract → Conclusion
  (write the abstract last, once results are final).
- After each section: compile, then give me the 5-line plain summary + "things to be ready to explain".
- I review each section before you move on.
- Tables generated from CSVs by a script (`scripts/make_latex_tables.py` or equivalent), not typed by hand.

**Build:** if LaTeX — `report/main.tex`, `report/refs.bib`, `report/figures/`. Compile locally if a TeX
install exists; otherwise produce a zip I can upload to Overleaf. Export the final PDF to `report/report.pdf`.

---

## Phase 4 — Presentation (built on the professor's template)

**Use `references/prof_ppt.pptx` as the template** with `python-pptx`: open it, keep its slide masters/layouts,
remove its content slides, and add new slides using its own layouts — so colours, fonts and logos match.
Output: `presentation/midsem.pptx`. Adapt the slide plan below to the professor's slide order and density.

Target: **15–18 slides, 12–15 minute talk.**

| # | Slide | Content |
|---|---|---|
| 1 | Title | Title, my name, supervisor, date |
| 2 | Motivation | One striking before/after image + why it matters (night photos, surveillance, driving, detection) |
| 3 | Problem | Darkness + noise + colour shift; why brightening amplifies noise (zoomed example) |
| 4 | Literature evolution | Timeline: classical → CNN → unsupervised → transformer → diffusion |
| 5 | Literature summary | Compact table: key methods, idea, limitation |
| 6 | Research gap | 3–4 gaps + where my project fits |
| 7 | Objectives | What this midsem covers |
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
| 18 | Plan for remaining semester | Timeline |
| 19 | Conclusion + references | Key takeaways; main references |

Slide rules: max ~5 short bullets per slide; figures large; every number from the CSVs;
same method names/colours as the report; slide numbers on.

**Speaker notes on every slide:** what I say in simple spoken English (~1 min per slide), plus
1–2 likely panel questions for that slide with short answers.

---

## Phase 5 — Consistency check and viva prep

1. Write `scripts/check_numbers.py` (or do it manually): every number in the report and slides
   matches `results/*.csv`. Report mismatches.
2. Check: figure/table numbering, all figures referenced in text, every citation in the bib and vice versa,
   same method names everywhere.
3. Create `presentation/viva_prep.md`: 20 likely questions with short answers in simple words
   (problem choice, Retinexformer vs earlier Retinex methods, attention cost, SNR map, why control run,
   PSNR/SSIM limits, gap to paper numbers, 15-image test set confidence, detection findings, next steps).
4. Update `CLAUDE.md` status. Commit and push (no `references/`, no weights).

---

## Day split
- **Day 4:** Phase 0 → 1 → 2 → 3 (report complete).
- **Day 5:** Phase 4 → 5 (slides, consistency check, viva prep), then I rehearse at least twice.
