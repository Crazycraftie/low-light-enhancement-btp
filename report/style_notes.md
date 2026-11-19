# Style notes — professor's papers (style reference only, no text copied)

Three papers by **M. H. Kolekar (IIT Patna)** are in `references/` (git-ignored). **No PPT was available.**

| Paper | Venue | Format | Pages |
|---|---|---|---|
| **LumiNet** (backlit image enhancement, GAN) | IEEE Trans. Instrum. Meas. 72, 2023 | **IEEE two-column, LaTeX (pdfTeX)**, US Letter | 14 |
| Mod-R2AUNet (MRI tumour segmentation) | Elsevier BSPC 2025 | Elsevier two-column | 12 |
| CNN transfer learning (MRI classification) | Springer MTAP 2025 | Springer single-column | 24 |

**Primary model: LumiNet.** It is the closest to this project (image enhancement, LOL dataset, NIQE, PSNR,
efficiency table, ablation, downstream recognition task) and uses the IEEE format → the report was built in LaTeX with
`IEEEtran` (journal, two-column); an **Overleaf-ready zip** and the compiled PDF were produced.

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
- **IV. CONCLUSION**, then References.
- Elsevier paper adds "Analysis of model complexity" and "Limitations" subsections →
  we added **Computational Efficiency** and **Limitations** (both present in the final report).

## Figures and tables
- Figures: caption **below**, "Fig. 7. Qualitative analysis of ... (a) Input. (b) ...", sub-images labelled
  (a), (b), ... with a key in the caption; zoomed insets used in qualitative figures.
- Tables: IEEE style — "TABLE II" + caption in **small caps above** the table, roman numerals; best value
  in **bold**; arrows ↑/↓ beside metric names; efficiency table (params, time) as its own table.
- Wide qualitative grids span both columns (`figure*`).

## Citations, equations, tone
- **Numeric IEEE citations** [1], [5]–[13] in order of appearance; "Zhang et al. [41]" when naming authors.
  References in IEEE format (initials first, "IEEE Trans. ...", vol., no., pp., month year). LumiNet has 63 refs.
- Equations numbered (1), (2), … at the right margin; every symbol defined right after the equation.
- Tone: formal, third person / "we", past tense for experiments; compares to "state-of-the-art methods".

## Presentation (no template available)
- The deck `presentation/final.pptx` was built with python-pptx from a **clean, consistent academic template**
  made by us: 16:9, white background, one accent colour (same blue as the report figures #2a78d6), sans-serif,
  slide numbers + footer "BTP — Low-Light Enhancement — IIT Patna", with the IIT Patna logo where available.
- A standard department template, when available, was checked in `references/` and its layouts were used instead.

## Decisions (as made)
1. The report follows **LumiNet's IEEE two-column structure** (Related Work inside the Introduction).
2. The report was prepared for **Overleaf** and compiled locally as well.
3. The slides use the clean academic template above (a department template or logo, if supplied, was used instead).
4. Title page: Dhirendra Pratap Singh (2301EE49), supervisor Dr. Rajib Kumar Jha; no date, by choice.
