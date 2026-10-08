"""
Generate every LaTeX table of the report from the CSVs (no number is typed by hand).

Reads   results/results.csv, dark_region.csv, detection.csv, speed.csv, reported_in_papers.csv,
        results/per_image/*.csv, results/per_image_dark/*.csv
Writes  report/tables/tab_*.tex  (IEEE style: booktabs, caption + label live in main.tex)

  tab_main.tex        LOL-v1 / LOL-v2-real / LOL-v2-syn: PSNR, SSIM, LPIPS for all methods (best in bold)
  tab_reproduce.tex   our numbers vs the numbers reported in the original papers (same protocol)
  tab_noref.tex       NIQE on LIME / DICM / MEF
  tab_detection.tex   YOLOv8m on ExDark: 1,200 images and the 200-image subset (incl. GSAD)
  tab_efficiency.tex  parameters and time per image on one Tesla T4
  tab_ablation.tex    fine-tuning runs A-D (+ start point): fidelity, dark-region, wins vs control, NIQE, detection

Usage:
    python scripts/make_latex_tables.py
"""

import csv
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "report" / "tables"

# Display names, identical in every table/figure of the report.
NAME = {"Input": "Input (no enhancement)", "HE": "HE", "CLAHE": "CLAHE", "Gamma": "Gamma ($\\gamma{=}0.4$)",
        "Zero-DCE": "Zero-DCE~\\cite{guo2020zerodce}", "SCI": "SCI~\\cite{ma2022sci}",
        "SNR-Aware (released)": "SNR-Aware$^{\\dagger}$~\\cite{xu2022snr}", "LLFormer": "LLFormer~\\cite{wang2023llformer}",
        "GSAD": "GSAD~\\cite{hou2023gsad}", "Retinexformer": "Retinexformer~\\cite{cai2023retinexformer}",
        "Retinexformer (my training)": "Retinexformer (our re-training)"}
MAIN = ["Input", "HE", "CLAHE", "Gamma", "Zero-DCE", "SCI", "SNR-Aware (released)", "LLFormer", "GSAD",
        "Retinexformer", "Retinexformer (my training)"]
# results that are not a fair main-table entry (cross-dataset weights, or LOL-v1 train / LOL-v2-real test overlap)
EXCLUDE = {("Retinexformer (my training)", "LOLv2-real"), ("Retinexformer (my training)", "LOLv2-syn"),
           ("LLFormer", "LOLv2-real"), ("LLFormer", "LOLv2-syn")}


def load(name, keys):
    p = R / name
    return {tuple(r[k] for k in keys): r for r in csv.DictReader(open(p))} if p.exists() else {}


def f(v, nd):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def bold_best(rows, cols, higher):
    """rows: list of lists of strings; bold the best numeric value in each listed column."""
    for c, hi in zip(cols, higher):
        vals = [float(r[c]) for r in rows if r[c] not in ("--", "")]
        if not vals:
            continue
        best = max(vals) if hi else min(vals)
        for r in rows:
            if r[c] not in ("--", "") and abs(float(r[c]) - best) < 1e-12:
                r[c] = "\\textbf{" + r[c] + "}"
    return rows


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n")


def per_image(folder, method, dataset, col):
    p = R / folder / f"{method}_{dataset}.csv"
    return {r["image"]: float(r[col]) for r in csv.DictReader(open(p))} if p.exists() else {}


def main():
    res = load("results.csv", ["method", "dataset"])
    sets = ["LOLv1", "LOLv2-real", "LOLv2-syn"]

    # ---- 1. main comparison ----
    rows = []
    for m in MAIN:
        r = [NAME[m]]
        for d in sets:
            x = {} if (m, d) in EXCLUDE else res.get((m, d), {})
            r += [f(x.get("psnr"), 2), f(x.get("ssim"), 3), f(x.get("lpips"), 3)]
        rows.append(r)
    rows = bold_best(rows, list(range(1, 10)), [True, True, False] * 3)
    L = ["\\begin{tabular}{l ccc ccc ccc}", "\\toprule",
         " & \\multicolumn{3}{c}{LOL-v1} & \\multicolumn{3}{c}{LOL-v2-real} & \\multicolumn{3}{c}{LOL-v2-syn} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Method & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\",
         "\\midrule"]
    for i, r in enumerate(rows):
        L.append(" & ".join(r) + " \\\\")
        if MAIN[i] in ("Gamma", "SCI"):
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_main.tex", L)

    # ---- 2. reproduction vs reported ----
    rep = list(csv.DictReader(open(R / "reported_in_papers.csv")))
    L = ["\\begin{tabular}{ll cc cc r}", "\\toprule",
         " & & \\multicolumn{2}{c}{Reported} & \\multicolumn{2}{c}{Ours} & \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "Method & Set & PSNR & SSIM & PSNR & SSIM & $\\Delta$PSNR \\\\", "\\midrule"]
    for r in rep:
        ours_m = "GSAD (GT-mean, authors protocol)" if r["method"] == "GSAD" else r["method"]
        o = res.get((ours_m, r["dataset"]), {})
        nm = {"SNR-Aware (released)": "SNR-Aware$^{\\dagger}$", "GSAD": "GSAD$^{\\ddagger}$"}.get(r["method"], r["method"])
        d = float(o["psnr"]) - float(r["psnr"]) if o else None
        L.append(f"{nm} & {r['dataset'].replace('LOLv', 'LOL-v')} & {float(r['psnr']):.2f} & {float(r['ssim']):.3f} & "
                 f"{f(o.get('psnr'), 2)} & {f(o.get('ssim'), 3)} & {(('%+.2f' % d) if abs(d) >= 0.005 else '0.00') if d is not None else '--'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_reproduce.tex", L)

    # ---- 3. NIQE ----
    up = ["LIME", "DICM", "MEF"]
    rows = [[NAME[m]] + [f(res.get((m, d), {}).get("niqe"), 3) for d in up] for m in MAIN
            if any(res.get((m, d), {}).get("niqe") for d in up)]
    rows = bold_best(rows, [1, 2, 3], [False] * 3)
    L = ["\\begin{tabular}{l ccc}", "\\toprule", "Method & LIME & DICM & MEF \\\\", "\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    write("tab_noref.tex", L)

    # ---- 4. detection ----
    det = load("detection.csv", ["method"])
    dn = [("raw", "Raw (no enhancement)"), ("zerodce", "Zero-DCE"), ("sci", "SCI"), ("retinexformer", "Retinexformer"), ("gsad", "GSAD")]
    rows = []
    for k, lab in dn:
        a, b = det.get((k,), {}), det.get((k + "_200",), {})
        rows.append([lab, f(a.get("mAP50"), 3), f(a.get("mAP50-95"), 3), f(a.get("recall"), 3),
                     f(b.get("mAP50"), 3), f(b.get("mAP50-95"), 3), f(b.get("recall"), 3)])
    rows = bold_best(rows, [1, 2, 3, 4, 5, 6], [True] * 6)
    L = ["\\begin{tabular}{l ccc ccc}", "\\toprule",
         " & \\multicolumn{3}{c}{1{,}200 images} & \\multicolumn{3}{c}{200-image subset} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Input to YOLOv8m & mAP50 & mAP50-95 & Recall & mAP50 & mAP50-95 & Recall \\\\", "\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    write("tab_detection.tex", L)

    # ---- 5. efficiency ----
    sp = load("speed.csv", ["method"])
    order = [("SCI (medium)", "SCI"), ("Zero-DCE", "Zero-DCE"), ("Retinexformer", "Retinexformer"), ("LLFormer", "LLFormer"),
             ("GSAD (LOLv2-real weights, 10 sampling steps)", "GSAD (10 steps)"), ("GSAD (LOLv1 weights, 20 sampling steps)", "GSAD (20 steps)")]
    L = ["\\begin{tabular}{l rr}", "\\toprule", "Method & Params (M) & Time (ms/img) \\\\", "\\midrule"]
    for k, lab in order:
        r = sp[(k,)]
        pm = float(r["params_M"])
        L.append(f"{lab} & {pm:.4f} & {float(r['ms_per_image']):,.1f} \\\\" if pm < 0.1 else
                 f"{lab} & {pm:.2f} & {float(r['ms_per_image']):,.1f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_efficiency.tex", L)

    # ---- 6. ablation ----
    dark = load("dark_region.csv", ["method", "dataset"])
    runs = [("Retinexformer", "Released (start point)", "rf_pretrained_gpu_200gpu"),
            ("FT-A control (L1)", "A: $\\mathcal{L}_1$ (control)", "ft_A_l1_200gpu"),
            ("FT-B SNR-L1 (mine)", "B: $\\mathcal{L}_{\\mathrm{SNR}}$ (ours)", "ft_B_snr_200gpu"),
            ("FT-C L1+FFT", "C: $\\mathcal{L}_1{+}\\lambda\\mathcal{L}_{\\mathrm{FFT}}$", "ft_C_fft_200gpu"),
            ("FT-D SNR-L1+FFT", "D: $\\mathcal{L}_{\\mathrm{SNR}}{+}\\lambda\\mathcal{L}_{\\mathrm{FFT}}$", "ft_D_snr_fft_200gpu")]
    a_img = per_image("per_image", "FT-A control (L1)", "LOLv1", "psnr")
    a_dark = per_image("per_image_dark", "FT-A control (L1)", "LOLv1", "dark_psnr")
    rows, wins = [], {}
    for m, lab, dk in runs:
        x = res.get((m, "LOLv1"), {})
        syn_m = "Retinexformer (LOL-v1 weights)" if m == "Retinexformer" else m
        niqe = [res.get((m, d), {}).get("niqe") for d in up]
        niqe_mean = sum(float(v) for v in niqe) / 3 if all(niqe) else None
        w1 = w2 = "--"
        if m.startswith("FT-") and m != "FT-A control (L1)":
            b_img = per_image("per_image", m, "LOLv1", "psnr")
            b_dark = per_image("per_image_dark", m, "LOLv1", "dark_psnr")
            k1 = sum(b_img[i] > a_img[i] for i in a_img); k2 = sum(b_dark[i] > a_dark[i] for i in a_dark)
            w1, w2 = f"{k1}/{len(a_img)}", f"{k2}/{len(a_dark)}"
            # exact one-sided sign test: P(at least k wins out of n | p = 0.5)
            wins[m] = (k1, k2, sum(comb(15, i) for i in range(k1, 16)) / 2 ** 15, sum(comb(15, i) for i in range(k2, 16)) / 2 ** 15)
        rows.append([lab, f(x.get("psnr"), 2), f(x.get("ssim"), 3), f(x.get("lpips"), 3),
                     f(dark.get((m, "LOLv1"), {}).get("dark_psnr"), 2), w1, w2,
                     f(res.get((syn_m, "LOLv2-syn"), {}).get("psnr"), 2),
                     f(niqe_mean, 3), f(det.get((dk,), {}).get("mAP50"), 3)])
    rows = bold_best(rows, [1, 2, 3, 4, 7, 8, 9], [True, True, False, True, True, False, True])
    L = ["\\begin{tabular}{l ccc c cc c c c}", "\\toprule",
         " & \\multicolumn{3}{c}{LOL-v1} & Dark 30\\% & \\multicolumn{2}{c}{Wins vs.\\ A} & LOL-v2-syn & NIQE & ExDark-200 \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}",
         "Run & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ & PSNR$\\uparrow$ & image & dark & PSNR$\\uparrow$ & mean$\\downarrow$ & mAP50$\\uparrow$ \\\\",
         "\\midrule"]
    for i, r in enumerate(rows):
        L.append(" & ".join(r) + " \\\\")
        if i == 0:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_ablation.tex", L)
    for m, (k1, k2, p1, p2) in wins.items():
        print(f"{m}: wins vs A {k1}/15 (one-sided sign-test p={p1:.4f}), dark {k2}/15 (p={p2:.4f})")
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("tab_*.tex"))))


if __name__ == "__main__":
    main()
