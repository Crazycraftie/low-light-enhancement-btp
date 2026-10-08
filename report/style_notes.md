# Style notes — professor's papers (style reference only, no text copied)

Three papers by **M. H. Kolekar (IIT Patna)** are in `references/` (git-ignored). **No PPT was available.**

| Paper | Venue | Format | Pages |
|---|---|---|---|
| **LumiNet** (backlit image enhancement, GAN) | IEEE Trans. Instrum. Meas. 72, 2023 | **IEEE two-column, LaTeX (pdfTeX)**, US Letter | 14 |
| Mod-R2AUNet (MRI tumour segmentation) | Elsevier BSPC 2025 | Elsevier two-column | 12 |
| CNN transfer learning (MRI classification) | Springer MTAP 2025 | Springer single-column | 24 |

**Primary model: LumiNet.** It is the closest to this project (image enhancement, LOL dataset, NIQE, PSNR,
efficiency table, ablation, downstream recognition task) and uses the IEEE format → **build the report in LaTeX with
`IEEEtran` (journal, two-column)**. No TeX on the Mac → deliver an **Overleaf-ready zip** + PDF compiled there.

## Structure (LumiNet; the other two follow the same pattern with numbered sections)
- Title, authors (+ affiliation footnote), **Abstract** (~215 words, one paragraph, states method + datasets +
  metrics + result), **Index Terms** (4–5 keywords).
- **I. INTRODUCTION** — motivation and applications, then subsections
  **A. Related Works** (no separate Related Work section) and **B. Contribution and Organization of Paper**
  (contributions as a list, then "The rest of the paper is organized as follows...").
- **II. METHODOLOGY** — A. Overview (pipeline figure), B. Proposed Architecture, C. Loss Function,
  D. downstream application (LumiNet: license-plate recognition → ours: detection on ExDark).
- **III. EXPERIMENT** — A. Dataset, B. Evaluation Metrics (each metric with its equation), C. Implementation
  Details, D. Results (1) Quantitative Analysis, 2) Qualitative Analysis), E. Ablation Study.
- **IV. CONCLUSION AND FUTURE WORK**, then References.
- Elsevier paper adds "Analysis of model complexity" and "Limitations and future research" subsections →
  we add **Computational Efficiency** and **Limitations** (needed anyway for the midsem framing).

## Figures and tables
- Figures: caption **below**, "Fig. 7. Qualitative analysis of ... (a) Input. (b) ...", sub-images labelled
  (a), (b), ... with a key in the caption; zoomed insets used in qualitative figures.
- Tables: IEEE style — "TABLE II" + caption in **small caps above** the table, roman numerals; best value
  in **bold**; arrows ↑/↓ next to metric names; efficiency table (params, time) as its own table.
- Wide qualitative grids span both columns (`figure*`).

## Citations, equations, tone
- **Numeric IEEE citations** [1], [5]–[13] in order of appearance; "Zhang et al. [41]" when naming authors.
  References in IEEE format (initials first, "IEEE Trans. ...", vol., no., pp., month year). LumiNet has 63 refs.
- Equations numbered (1), (2), … at the right margin; every symbol defined right after the equation.
- Tone: formal, third person / "we", past tense for experiments; compares to "state-of-the-art methods".

## Presentation (no template available)
- Proposal: build `presentation/midsem.pptx` with python-pptx from a **clean, consistent academic template**
  made by us: 16:9, white background, one accent colour (same blue as the report figures #2a78d6), sans-serif,
  slide numbers + footer "Midsem BTP — Low-Light Enhancement — IIT Patna", IIT Patna logo only if you provide it.
- If your department has a standard midsem template, put it in `references/` and I will use its layouts instead.

## Decisions for you
1. OK to follow **LumiNet's IEEE two-column structure** (Related Work inside the Introduction)?
2. Report via **Overleaf** (no LaTeX installed locally)?
3. Slides: our own clean template, or will you get a department template / logo?
4. Your name, roll number, supervisor's name/title, and the date for the title page.
