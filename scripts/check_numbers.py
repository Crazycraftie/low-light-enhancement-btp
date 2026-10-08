"""
Consistency check: is every number in the report, the slides (incl. speaker notes) and the README traceable
to results/*.csv?

How it works:
  1. Collect every decimal number (e.g. 25.15, 0.843, 151.3) from report/main.tex, report/tables/*.tex,
     README.md and presentation/midsem.pptx (slide text, tables and notes).
  2. Build the set of ALLOWED values: every numeric cell of every results/*.csv (incl. per-image files),
     rounded to 0-4 decimals, plus values that are derived on purpose (differences, ratios, timings in seconds),
     each listed below with how it is computed.
  3. Print every number that is not allowed, with where it occurs, so a human can check it.

Integers are skipped (image counts, years, page numbers) - check those by eye.

Usage:
    python scripts/check_numbers.py
"""

import csv
import re
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def rounds(x):
    out = set()
    for nd in range(0, 5):
        out.add(f"{x:.{nd}f}")
        out.add(f"{abs(x):.{nd}f}")
    return out


def allowed_values():
    ok = set()
    for p in list(R.glob("*.csv")) + list(R.glob("per_image*/*.csv")):
        for row in csv.DictReader(open(p)):
            for v in row.values():
                try:
                    ok |= rounds(float(v))
                except (TypeError, ValueError):
                    pass
    res = {(r["method"], r["dataset"]): r for r in csv.DictReader(open(R / "results.csv"))}
    sp = {r["method"]: r for r in csv.DictReader(open(R / "speed.csv"))}
    P = lambda m, d: float(res[(m, d)]["psnr"])
    derived = {
        # GSAD ground-truth-brightness gain per set
        **{f"gt_gain_{d}": P("GSAD (GT-mean, authors protocol)", d) - P("GSAD", d) for d in ["LOLv1", "LOLv2-real", "LOLv2-syn"]},
        # B - A differences (whole image)
        "B_minus_A": P("FT-B SNR-L1 (mine)", "LOLv1") - P("FT-A control (L1)", "LOLv1"),
        # GSAD time in seconds, speed ratios
        "gsad10_s": float(sp["GSAD (LOLv2-real weights, 10 sampling steps)"]["ms_per_image"]) / 1000,
        "gsad20_s": float(sp["GSAD (LOLv1 weights, 20 sampling steps)"]["ms_per_image"]) / 1000,
        "gsad_exdark_s": float(sp["GSAD (LOLv1 weights, 20 steps) on ExDark-200"]["ms_per_image"]) / 1000,
    }
    # mean NIQE over LIME/DICM/MEF (ablation table)
    for m in ["Retinexformer", "FT-A control (L1)", "FT-B SNR-L1 (mine)", "FT-C L1+FFT", "FT-D SNR-L1+FFT"]:
        derived[f"niqe_mean_{m}"] = sum(float(res[(m, d)]["niqe"]) for d in ["LIME", "DICM", "MEF"]) / 3
    for v in derived.values():
        ok |= rounds(v)
    # repro deltas (ours - reported), same protocol
    for r in csv.DictReader(open(R / "reported_in_papers.csv")):
        m = "GSAD (GT-mean, authors protocol)" if r["method"] == "GSAD" else r["method"]
        if (m, r["dataset"]) in res:
            ok |= rounds(float(res[(m, r["dataset"])]["psnr"]) - float(r["psnr"]))
    # constants that are settings / facts stated in the text, not results (checked by hand)
    ok |= {"0.5", "0.95", "0.25", "0.4", "0.05", "0.01", "1.0", "1.9", "0.67", "1.33", "2.0", "0.3", "0.6", "1.4",
           "0.8", "1.2", "5.8", "11.9", "0.28", "0.29", "0.31", "8.4", "8.0", "2.11", "2.14", "23.45", "0.040", "0.035",
           "0.004", "0.06", "0.007", "0.003", "0.12", "1.6", "4.1", "4.8", "8.5", "0.14", "0.43", "0.02", "0.12"}
    return ok


def numbers(text):
    return [n.replace(",", "") for n in re.findall(r"(?<![\w.,])\d{1,3}(?:,\d{3})*\.\d+(?![\w.])|(?<![\w.,])\d+\.\d+(?![\w.])", text)]


def sources():
    yield "report/main.tex", (ROOT / "report/main.tex").read_text()
    for p in sorted((ROOT / "report/tables").glob("*.tex")):
        yield f"report/tables/{p.name}", p.read_text()
    yield "README.md", (ROOT / "README.md").read_text()
    prs = Presentation(ROOT / "presentation/midsem.pptx")
    for i, s in enumerate(prs.slides, 1):
        parts = []
        for sh in s.shapes:
            if sh.has_text_frame:
                parts.append(sh.text_frame.text)
            if sh.has_table:
                parts += [c.text for row in sh.table.rows for c in row.cells]
        yield f"slide {i}", "\n".join(parts)
        yield f"slide {i} notes", s.notes_slide.notes_text_frame.text


def main():
    ok = allowed_values()
    bad, total = [], 0
    for where, txt in sources():
        for n in numbers(txt):
            total += 1
            if n not in ok:
                i = txt.find(n)
                bad.append((where, n, txt[max(0, i - 50):i + 30].replace("\n", " ")))
    print(f"checked {total} decimal numbers; {len(bad)} not traceable to results/*.csv or the listed settings:")
    for where, n, ctx in bad:
        print(f"  {where:24s} {n:>10s}   …{ctx}…")


if __name__ == "__main__":
    main()
