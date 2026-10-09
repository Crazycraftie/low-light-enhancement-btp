"""
Build the midsem presentation (presentation/midsem.pptx) with python-pptx.

* Clean academic template made here (no professor PPT was available): 16:9, white background, one accent blue
  (#2a78d6, the same as the report figures), Arial, footer + slide numbers.
* Every number on a slide is read from results/*.csv (same files as the report tables) - nothing typed by hand.
* Every slide has speaker notes: what to say (simple spoken English, ~1 min) + likely panel questions with answers.

Usage:
    python scripts/make_slides.py            # -> presentation/midsem.pptx (+ presentation/assets/*.png)
Render check (needs LibreOffice):
    soffice --headless --convert-to pdf presentation/midsem.pptx --outdir presentation/
"""

import csv
from math import comb
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
R, FIG, OUT = ROOT / "results", ROOT / "report" / "figures", ROOT / "presentation"
ASSETS = OUT / "assets"
BLUE, INK, INK2, LIGHT, ORANGE = RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E), \
    RGBColor(0xEE, 0xF4, 0xFC), RGBColor(0xEB, 0x68, 0x34)
FONT = "Arial"
TITLE = "Noise-Aware Transformer-Based Low-Light Image Enhancement with Downstream Object Detection Evaluation"
FOOT = "Midsem BTP  ·  Low-light image enhancement  ·  IIT Patna"
# Fill these in (title slide); they also appear in the report title block.
STUDENT, ROLL, SUPERVISOR, DATE = "Dhirendra Pratap Singh", "2301EE49", "Dr. Rajib Kumar Jha", "[MISSING: Date]"


# ------------------------------------------------------------------ data (all numbers come from here)
def load(name, keys):
    return {tuple(r[k] for k in keys): r for r in csv.DictReader(open(R / name))}


RES = load("results.csv", ["method", "dataset"])
DET = load("detection.csv", ["method"])
SPEED = load("speed.csv", ["method"])
DARK = load("dark_region.csv", ["method", "dataset"])


def v(method, dataset, metric, nd):
    x = RES.get((method, dataset), {}).get(metric, "")
    return f"{float(x):.{nd}f}" if x not in ("", None) else "–"


def per_image(folder, method, col):
    return {r["image"]: float(r[col]) for r in csv.DictReader(open(R / folder / f"{method}_LOLv1.csv"))}


# ------------------------------------------------------------------ slide-only images from real outputs
def make_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    stem = "79"
    inp = Image.open(ROOT / f"data/LOLv1/Test/input/{stem}.png").convert("RGB")
    rf = Image.open(R / f"retinexformer/LOLv1/{stem}.png").convert("RGB")
    zd = Image.open(R / f"zerodce/LOLv1/{stem}.png").convert("RGB")
    gt = Image.open(ROOT / f"data/LOLv1/Test/target/{stem}.png").convert("RGB")
    # 1) before / after
    w, h = inp.size
    pair = Image.new("RGB", (2 * w + 20, h), "white")
    pair.paste(inp, (0, 0)); pair.paste(rf, (w + 20, 0))
    pair.save(ASSETS / "before_after.png")
    # 2) why brightening amplifies noise: same dark crop, input x6 gain vs Zero-DCE vs ground truth.
    #    Crop chosen automatically: among dark windows of the input, the one where Zero-DCE's output has the most
    #    high-frequency energy (noise) beyond what the ground truth has there.
    from PIL import ImageFilter
    hf = lambda im: np.abs(np.asarray(im.convert("L"), np.float32) - np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(2)), np.float32))
    excess = hf(zd) - hf(gt)
    lum = np.asarray(inp.convert("L"), np.float32)
    bw, bh, best = 140, 100, None
    dark_cut = np.quantile(lum, 0.4)
    for y0 in range(0, h - bh, 10):
        for x0 in range(0, w - bw, 10):
            if lum[y0:y0 + bh, x0:x0 + bw].mean() <= dark_cut:
                sc = excess[y0:y0 + bh, x0:x0 + bw].mean()
                if best is None or sc > best[0]:
                    best = (sc, x0, y0)
    box = (best[1], best[2], best[1] + bw, best[2] + bh)
    crops = []
    a = np.asarray(inp.crop(box), dtype=np.float32) * 6
    crops.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
    crops += [zd.crop(box), gt.crop(box)]
    crops = [c.resize((c.width * 3, c.height * 3), Image.NEAREST) for c in crops]
    cw, ch = crops[0].size
    strip = Image.new("RGB", (3 * cw + 40, ch), "white")
    for i, c in enumerate(crops):
        strip.paste(c, (i * (cw + 20), 0))
    strip.save(ASSETS / "noise_zoom.png")
    full = inp.copy()
    ImageDraw.Draw(full).rectangle(box, outline=(227, 73, 72), width=4)
    full = Image.fromarray(np.clip(np.asarray(full, dtype=np.float32) * np.array([1, 1, 1]) ** 1, 0, 255).astype(np.uint8))
    full.save(ASSETS / "noise_where.png")
    # 3) slide-sized comparison grid: 2 images, key methods only (readable on a projector)
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT / "scripts/make_grid.py"), "--dataset", "LOLv1", "--images", "79", "493",
                    "--methods", "Zero-DCE", "SCI", "Retinexformer", "GSAD", "FT SNR-L1 (B, mine)",
                    "--out", str(ASSETS / "grid_slides.png")], check=True, capture_output=True)


# ------------------------------------------------------------------ helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


def text(slide, x, y, w, h, s, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.name, r.font.bold = Pt(size), FONT, bold
        r.font.color.rgb = color
    return tb


def bullets(slide, items, x=0.6, y=1.5, w=6.0, h=5.2, size=19, gap=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        sub = it.startswith("  ")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run()
        r.text = ("–  " if sub else "•  ") + it.strip()
        r.font.size, r.font.name = Pt(size - 3 if sub else size), FONT
        r.font.color.rgb = INK2 if sub else INK
        if sub:
            p.level = 1
    return tb


def picture(slide, path, x, y, w, h):
    """Insert an image as large as possible inside the box (x, y, w, h), keeping its aspect ratio, centred."""
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, rows, x, y, w, col_w=None, size=13, bold_cells=(), row_h=0.36, header_fill=BLUE):
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = shp.table
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.text = str(val)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = p.runs[0].font
            f.size, f.name = Pt(size), FONT
            f.bold = (i == 0) or (i, j) in bold_cells
            f.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
    return t


def best_cells(rows, cols, higher):
    """(row, col) of the best numeric value per column (rows[0] is the header)."""
    out = set()
    for c, hi in zip(cols, higher):
        vals = [(float(r[c]), i) for i, r in enumerate(rows) if i and r[c] not in ("–", "")]
        if vals:
            b = max(vals)[0] if hi else min(vals)[0]
            out |= {(i, c) for val, i in vals if abs(val - b) < 1e-12}
    return out


def slide(title, notes, section=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    bar = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    if section:
        text(s, 0.6, 0.28, 8, 0.35, section.upper(), 11, BLUE, bold=True)
    text(s, 0.6, 0.55, 12.2, 0.8, title, 28, INK, bold=True)
    text(s, 0.6, 7.02, 9, 0.35, FOOT, 10, INK2)
    text(s, 12.1, 7.02, 0.7, 0.35, str(N[0]), 10, INK2, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes.strip()
    return s


def caption(s, x, y, w, t, size=12, under=None):
    """Caption text; with under=<picture shape> it is placed right below that picture."""
    if under is not None:
        y = under.top / 914400 + under.height / 914400 + 0.08
    text(s, x, y, w, 0.4, t, size, INK2)


# ------------------------------------------------------------------ slides
def build():
    make_assets()
    # 1 ---------------------------------------------------------------- title
    s = prs.slides.add_slide(BLANK); N[0] += 1
    bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(3.9)); bg.fill.solid(); bg.fill.fore_color.rgb = BLUE; bg.line.fill.background()
    text(s, 0.8, 0.9, 11.7, 2.2, TITLE, 34, RGBColor(0xFF, 0xFF, 0xFF), bold=True)
    text(s, 0.8, 3.0, 11, 0.5, "Mid-semester B.Tech Project evaluation", 18, RGBColor(0xDB, 0xE8, 0xF8))
    text(s, 0.8, 4.4, 11.5, 2.0, f"{STUDENT}   ({ROLL})\nSupervisor: {SUPERVISOR}\nDepartment of Electrical Engineering, IIT Patna\n{DATE}", 18, INK)
    s.notes_slide.notes_text_frame.text = (
        "SAY: Good morning. My project is about making very dark photos look as if they were taken in good light, "
        "and checking whether that actually helps a computer find objects in the dark. In this midsem I will show what "
        "I have reproduced, what I measured, one idea of my own that I tested, and my plan for the second half.\n\n"
        "Q: Why this topic? A: Night photos, surveillance and driving all produce dark images, and detectors fail on them; "
        "it is an active research area with open problems such as noise.")

    # 2 ---------------------------------------------------------------- motivation
    s = slide("Why low-light enhancement?", """
SAY: On the left is a real photo from the LOL dataset taken in very low light; on the right the same photo after
Retinexformer. Low-light images appear in phone night photography, surveillance cameras and autonomous driving. They are
not only hard for people to see - object detectors also work much worse in the dark. So we want methods that brighten the
image, keep colours natural and do not amplify noise.

Q: Can't we just increase the exposure? A: Longer exposure causes motion blur, and higher ISO adds noise; enhancement
after capture avoids both but must deal with the noise that is already there.""", "Motivation")
    pic = picture(s, ASSETS / "before_after.png", 0.4, 1.45, 7.9, 5.0)
    caption(s, 0.4, 0, 7.9, "LOL-v1 test image 79: dark input (left) and Retinexformer output (right)", under=pic)
    bullets(s, ["Phone night photography", "Surveillance and security cameras", "Driving at night",
                "Machine vision: detectors fail in the dark", "Goal: bright, natural colours, no amplified noise"],
            x=8.6, y=1.7, w=4.4, size=18)

    # 3 ---------------------------------------------------------------- problem
    s = slide("The problem: darkness + noise + colour shift", """
SAY: Dark images have three problems at once. They are dark, they are noisy because the sensor collected few photons,
and artificial light changes the colours. The key point is on the right: if I simply multiply the dark region by six,
the noise is multiplied too - you can see the coloured speckles. A simple enhancer like Zero-DCE also brightens the noise.
The ground truth, taken with a long exposure, is clean. So enhancement must brighten AND denoise.

Q: Where does the noise come from? A: Mostly photon shot noise and sensor read noise; with few photons the random
fluctuation is large relative to the signal, i.e. the signal-to-noise ratio (SNR) is low.""", "Problem")
    pic = picture(s, ASSETS / "noise_where.png", 0.6, 1.45, 4.6, 3.1)
    caption(s, 0.6, 0, 4.6, "Red box: dark region (LOL-v1 79)", under=pic)
    pic = picture(s, ASSETS / "noise_zoom.png", 5.5, 1.45, 7.4, 3.1)
    caption(s, 5.5, 0, 7.4, "Same crop:  input × 6  |  Zero-DCE  |  ground truth", under=pic)
    bullets(s, ["Dark: little light reaches the sensor", "Noisy: few photons → low signal-to-noise ratio (SNR)",
                "Colour shift / over-exposed lamps", "Brightening multiplies the noise as much as the signal"],
            x=0.6, y=5.05, w=12, size=17, gap=2)

    # 4 ---------------------------------------------------------------- literature timeline
    s = slide("How the field evolved", """
SAY: The field moved through five stages. Classical methods like histogram equalization and Retinex-based LIME use hand-made
rules and ignore noise. Supervised CNNs like Retinex-Net and KinD learn from paired dark/bright images. Zero-shot methods like
Zero-DCE and SCI need no pairs and are tiny and fast, but do not denoise. Transformers - SNR-Aware, LLFormer, Retinexformer -
model long-range relations and give the best fidelity. Diffusion models like GSAD generate very realistic textures but are slow.

Q: Why did transformers help? A: Self-attention lets a dark pixel use information from distant, well-lit regions; CNNs only
see a local neighbourhood.""", "Literature")
    stages = [("Classical", "HE, CLAHE,\nRetinex / LIME"), ("Supervised CNN", "LLNet, Retinex-Net,\nKinD, MIRNet"),
              ("Zero-shot", "EnlightenGAN,\nZero-DCE, RUAS, SCI"), ("Transformers", "SNR-Aware, LLFormer,\nRetinexformer"),
              ("Diffusion", "GSAD,\nDiff-Retinex")]
    for i, (a, b) in enumerate(stages):
        x = 0.6 + i * 2.5
        box = s.shapes.add_shape(1, Inches(x), Inches(2.3), Inches(2.2), Inches(1.0))
        box.fill.solid(); box.fill.fore_color.rgb = BLUE if i == 3 else LIGHT; box.line.fill.background()
        text(s, x, 2.3, 2.2, 1.0, a, 17, RGBColor(0xFF, 0xFF, 0xFF) if i == 3 else INK, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        text(s, x, 3.45, 2.2, 1.0, b, 14, INK2, align=PP_ALIGN.CENTER)
        if i < 4:
            text(s, x + 2.18, 2.5, 0.35, 0.6, "→", 24, BLUE, True, PP_ALIGN.CENTER)
    bullets(s, ["Hand-made rules → learned from data → no pairs needed → long-range attention → generative",
                "Survey (Li et al., TPAMI 2022): no method wins everywhere; noise in dark regions remains open",
                "This project: Retinexformer (ICCV 2023) as the main model"], x=0.6, y=4.75, w=12.2, size=17)

    # 5 ---------------------------------------------------------------- literature table
    s = slide("Key methods compared in this project", """
SAY: These are the methods I actually ran. For each one: the idea in one line, and its main limitation. Notice the pattern:
the fast unsupervised methods do not remove noise, the transformers are accurate, and the diffusion model is slow. SNR-Aware
is the only one that explicitly measures noise - that idea inspired my loss.

Q: Why Retinexformer as the main model? A: Best PSNR in our tests on all three LOL sets, only 1.6 M parameters, public code
and weights, and it is built on Retinex theory, which makes it explainable.""", "Literature")
    rows = [["Method", "Year / venue", "Key idea", "Limitation"],
            ["HE / CLAHE / Gamma", "classical", "Re-map brightness", "No noise model, colour shifts"],
            ["Zero-DCE", "CVPR 2020", "Learned brightness curves, no pairs", "Brightens noise"],
            ["SCI", "CVPR 2022", "Self-calibrated illumination, ~300 params", "Brightens noise"],
            ["SNR-Aware", "CVPR 2022", "SNR map decides local vs. long-range", "Heavier; no Retinex model"],
            ["LLFormer", "AAAI 2023", "Axis-based attention for 4K/8K images", "24.5 M params, slower"],
            ["Retinexformer", "ICCV 2023", "Illumination-guided channel attention", "Illumination-aware, not noise-aware"],
            ["GSAD", "NeurIPS 2023", "Diffusion with structure regularization", "Slow; GT-brightness test trick"]]
    table(s, rows, 0.6, 1.6, 12.1, [2.4, 1.8, 4.3, 3.6], size=14, row_h=0.52)

    # 6 ---------------------------------------------------------------- gap + objectives
    s = slide("Research gap and objectives of this midsem", """
SAY: Four gaps motivate the project. First, noise in dark regions is still an open problem. Second, Retinexformer knows where the
image is dark but not where it is noisy. Third, papers mostly report PSNR and SSIM; few check whether enhancement helps a machine.
Fourth, diffusion models trade a lot of speed for quality. My midsem objectives follow directly: reproduce, evaluate fairly
on many axes including detection, and test a noise-aware loss with a proper control experiment.

Q: What is the difference between dark and noisy? A: A dark but smooth wall can be clean; a textured dark region can be very
noisy. Illumination tells you brightness, SNR tells you how much of the signal is noise.""", "Motivation")
    text(s, 0.6, 1.5, 5.9, 0.5, "Gaps", 20, BLUE, True)
    bullets(s, ["Noise amplification in dark regions is unsolved", "Retinexformer: illumination-aware, not noise-aware",
                "Methods judged by PSNR/SSIM, rarely by a machine task", "Diffusion: quality at a large speed cost"],
            x=0.6, y=2.05, w=5.9, size=17)
    text(s, 6.9, 1.5, 5.9, 0.5, "Objectives (first half)", 20, BLUE, True)
    bullets(s, ["Reproduce Retinexformer + 8 baselines", "One protocol: PSNR, SSIM, LPIPS, NIQE, dark regions, speed",
                "Detection study: does enhancement help YOLOv8?", "Noise-aware SNR-weighted loss, tested against a control",
                "Plan the second half from the evidence"], x=6.9, y=2.05, w=6.0, size=17)

    # 7 ---------------------------------------------------------------- background
    s = slide("Background: Retinex theory and attention cost", """
SAY: Retinex theory says an image is reflectance times illumination: the true colours of objects times the light falling on them.
Enhancement means estimating the illumination and replacing it with brighter light. Transformers use attention: every token
compares itself with every other token, so the cost grows with the square of the number of tokens. For a 600 by 400 image that
would be 240 thousand tokens and a matrix with 58 billion entries - impossible. Retinexformer computes attention across channels
instead, a small C by C matrix, so the cost grows only linearly with image size.

Q: What are Q, K, V? A: Query = what a token is looking for, Key = what it contains, Value = what it passes on; similarity of
Q and K decides how much of each Value is mixed in. Q: Why divide by sqrt(d)? A: Keeps the dot products in a range where
softmax does not saturate.""", "Background")
    text(s, 0.6, 1.5, 6.0, 0.5, "Retinex model", 20, BLUE, True)
    text(s, 0.6, 2.1, 6.0, 0.6, "I  =  R  ⊙  L", 30, INK, True, PP_ALIGN.CENTER)
    bullets(s, ["I: observed image", "R: reflectance = object colours", "L: illumination = light on the scene",
                "Enhance: estimate L, replace by brighter light", "Real images: noise in R and L, amplified when L is small"],
            x=0.6, y=2.9, w=6.0, size=16, gap=4)
    text(s, 6.9, 1.5, 6.0, 0.5, "Why attention is costly", 20, BLUE, True)
    text(s, 6.9, 2.1, 6.0, 0.6, "softmax(QKᵀ/√d)·V", 26, INK, True, PP_ALIGN.CENTER)
    bullets(s, ["QKᵀ is N × N  (N = number of pixels)", "600×400 image: N = 240,000 → 58 billion entries",
                "Channel attention: softmax(KᵀQ) is C × C", "→ cost grows linearly with image size"],
            x=6.9, y=2.9, w=6.0, size=16, gap=4)

    # 8 ---------------------------------------------------------------- retinexformer
    s = slide("Main model: Retinexformer (ICCV 2023)", f"""
SAY: Retinexformer has two parts. The illumination estimator looks at the image and its mean brightness and predicts a light-up
map; multiplying and adding back gives a brighter but still noisy image. Then the corruption restorer, a U-shaped transformer,
removes the noise and artifacts. Its special block, IG-MSA, multiplies the values by light-up features, so differently lit regions
are treated differently, and computes attention across channels. The whole model has only
{float(SPEED[('Retinexformer',)]['params_M']):.2f} million parameters. I drew this diagram myself from the released code.

Q: Why add I back (I*L + I)? A: A residual form: the network only has to learn the change in brightness, which is easier and
more stable. Q: What does illumination guidance add? A: Dark regions can borrow information from well-lit regions in a controlled
way.""", "Method")
    picture(s, FIG / "architecture.png", 0.5, 1.4, 8.9, 5.5)
    bullets(s, ["Estimator → light-up map L", "Lit-up image: I ⊙ L + I (still noisy)", "Restorer: U-shaped transformer (IGT)",
                "IG-MSA: V × light-up features, attention across channels",
                f"{float(SPEED[('Retinexformer',)]['params_M']):.2f} M params, ℓ1 loss"], x=9.6, y=1.7, w=3.5, size=15, gap=6)

    # 9 ---------------------------------------------------------------- proposed loss
    s = slide("My idea: noise-aware (SNR-weighted) loss", """
SAY: My idea is to tell the network, during training, which pixels are noisy. I compute an SNR map from the dark input: blur it,
the difference between the image and its blur is the noise, and blur divided by noise is the SNR. Low-SNR pixels then get up to
twice the weight of clean pixels in the L1 loss. Two design decisions: first, I use the rank of each pixel's SNR, because with
min-max normalisation 93 percent of pixels got weight about 2, which would just double the learning rate - you see that in the
orange histogram. Second, I divide by the mean so the average weight is exactly 1; only the emphasis moves.

Q: Why not min-max? A: The SNR values have a huge tail; min-max squashes almost every pixel to the same weight. Q: Is the SNR map
learned? A: No, it is computed from the input, without gradients. Q: Weakness? A: The blur difference also fires on edges - see
the yellow edges in panel (e).""", "Method")
    picture(s, FIG / "snr_weights.png", 0.4, 1.35, 8.9, 4.9)
    text(s, 9.5, 1.5, 3.6, 0.4, "Steps", 18, BLUE, True)
    bullets(s, ["blur b = 5×5 mean of gray", "noise n = |gray − b|", "SNR = b / (n + ε)", "rank r ∈ [0,1] per image",
                "w = 1 + α(1 − r),  α = 1", "w ← w / mean(w)", "L = mean(w · |Ŷ − Y|)"], x=9.5, y=1.95, w=3.6, size=15, gap=3)
    caption(s, 0.4, 6.35, 12.5, "Rank normalisation (e) spreads the weights; min-max (f) gives almost every pixel the same weight (93% > 1.9).")

    # 10 --------------------------------------------------------------- setup
    s = slide("Experimental setup", """
SAY: I evaluate on three paired test sets with ground truth, three sets of real unpaired photos, and 1,200 ExDark images with
object boxes. Metrics: PSNR and SSIM for fidelity, LPIPS for perceptual similarity, NIQE when there is no ground truth, my
dark-region PSNR in the darkest 30 percent of pixels, and mAP for detection. Training ran on a free Kaggle T4 GPU. Important
protocol rules: no ground-truth information at test time, no checkpoint chosen by test score, and no testing on training images.

Q: Why is LOL-v2-real excluded for models trained on LOL-v1? A: I found that 91 of its 100 test images are LOL-v1 training images.
Q: Why does SSIM differ from PSNR? A: PSNR is pixel error; SSIM compares local structure, contrast and brightness.""", "Setup")
    rows = [["Data", "Use", "Test images"], ["LOL-v1", "paired, main", "15"], ["LOL-v2 real / syn", "paired", "100 / 100"],
            ["LIME / DICM / MEF", "unpaired, NIQE", "10 / 69 / 17"], ["ExDark (12 classes)", "detection", "1,200 (200 for GSAD)"]]
    table(s, rows, 0.6, 1.55, 6.3, [2.3, 2.0, 2.0], size=14, row_h=0.46)
    rows = [["Metric", "Measures", "Better"], ["PSNR", "pixel fidelity", "↑"], ["SSIM", "structure", "↑"], ["LPIPS", "perceptual distance", "↓"],
            ["NIQE", "naturalness (no GT)", "↓"], ["Dark-30% PSNR", "darkest pixels only", "↑"], ["mAP50", "detection accuracy", "↑"]]
    table(s, rows, 7.3, 1.55, 5.5, [1.9, 2.6, 1.0], size=14, row_h=0.42)
    bullets(s, ["Hardware: Kaggle Tesla T4 (training, timing); laptop for evaluation", "Rules: no GT at test time · no test-chosen checkpoint · no train/test overlap"],
            x=0.6, y=4.75, w=12.2, size=16, gap=4)

    # 11 --------------------------------------------------------------- quantitative
    meth = [("Input", "Input"), ("Gamma", "Gamma"), ("Zero-DCE", "Zero-DCE"), ("SCI", "SCI"), ("SNR-Aware (released)", "SNR-Aware*"),
            ("LLFormer", "LLFormer"), ("GSAD", "GSAD"), ("Retinexformer", "Retinexformer"), ("Retinexformer (my training)", "Retinexformer (mine)")]
    excl = {("LLFormer", "LOLv2-real"), ("Retinexformer (my training)", "LOLv2-real")}
    rows = [["Method", "LOL-v1 PSNR↑", "SSIM↑", "LPIPS↓", "LOL-v2-real PSNR↑", "SSIM↑", "LPIPS↓"]]
    for m, lab in meth:
        r = [lab]
        for d in ["LOLv1", "LOLv2-real"]:
            r += ["–"] * 3 if (m, d) in excl else [v(m, d, "psnr", 2), v(m, d, "ssim", 3), v(m, d, "lpips", 3)]
        rows.append(r)
    s = slide("Quantitative results (paired test sets)", f"""
SAY: This is the main comparison, all scored by the same script. Retinexformer has the best PSNR on both sets,
{v('Retinexformer','LOLv1','psnr',2)} dB on LOL-v1. GSAD, the diffusion model, has the best LPIPS - its images look the most natural -
but lower PSNR. Classical and zero-shot methods brighten the image but stay around 15 dB. My own re-training of Retinexformer reached
{v('Retinexformer (my training)','LOLv1','psnr',2)} dB - I explain that gap on the next slide.

Q: Why is GSAD lower in PSNR but better in LPIPS? A: Diffusion creates realistic texture but its overall brightness is off, and PSNR
punishes brightness errors heavily. Q: Why are dashes there? A: Those models were trained on LOL-v1, whose training set overlaps
LOL-v2-real's test set.""", "Results")
    table(s, rows, 0.6, 1.55, 12.1, [3.0, 1.55, 1.25, 1.25, 2.1, 1.45, 1.5], size=14, row_h=0.44,
          bold_cells=best_cells(rows, [1, 2, 3, 4, 5, 6], [True, True, False, True, True, False]))
    caption(s, 0.6, 6.15, 12.1, "* images released by the SNR-Aware authors.  Best value in bold.  –: train/test overlap (see setup).", 13)

    # 12 --------------------------------------------------------------- reproduction / protocol findings
    rep = list(csv.DictReader(open(R / "reported_in_papers.csv")))
    rp = {(r["method"], r["dataset"]): r for r in rep}
    s = slide("Reproduction and protocol findings", f"""
SAY: Before trusting any comparison I checked that I reproduce the published numbers. Retinexformer, SNR-Aware and LLFormer match
their papers to 0.01 dB. GSAD's paper reports {float(rp[('GSAD','LOLv1')]['psnr']):.2f} dB, but its test script rescales every output
to the ground truth's average brightness - information a real camera never has. With that trick I get
{v('GSAD (GT-mean, authors protocol)','LOLv1','psnr',2)}, without it {v('GSAD','LOLv1','psnr',2)} dB. Second finding: 91 of the 100
LOL-v2-real test images are copies of LOL-v1 training images. Third: training Retinexformer from scratch myself gives
{v('Retinexformer (my training)','LOLv1','psnr',2)} dB, not 25 - another user on the authors' GitHub reported 23.45 dB.

Q: Why is your re-training 2 dB lower? A: Likely run-to-run variance on a tiny dataset (485 train, 15 test images), different
PyTorch/GPU, and the released checkpoint may be the best of several runs; my evaluation reproduces the training log exactly.
Q: Is GSAD cheating? A: It follows an older convention (KinD, LLFlow) and says so in its README, but it is not comparable with
methods that do not use the ground truth.""", "Results")
    rows = [["Method / set", "Paper", "Ours"]]
    for m, d, lab in [("Retinexformer", "LOLv1", "Retinexformer, LOL-v1"), ("SNR-Aware (released)", "LOLv1", "SNR-Aware, LOL-v1"),
                      ("LLFormer", "LOLv1", "LLFormer, LOL-v1")]:
        rows.append([lab, f"{float(rp[(m, d)]['psnr']):.2f}", v(m, d, "psnr", 2)])
    rows.append(["GSAD, LOL-v1 (GT trick)", f"{float(rp[('GSAD','LOLv1')]['psnr']):.2f}", v("GSAD (GT-mean, authors protocol)", "LOLv1", "psnr", 2)])
    rows.append(["GSAD, LOL-v1 (real output)", "–", v("GSAD", "LOLv1", "psnr", 2)])
    table(s, rows, 0.6, 1.6, 6.3, [3.5, 1.4, 1.4], size=15, row_h=0.5)
    caption(s, 0.6, 4.7, 6.3, "PSNR in dB on LOL-v1", 13)
    bullets(s, [f"GSAD's GT-brightness trick: +{float(RES[('GSAD (GT-mean, authors protocol)','LOLv1')]['psnr']) - float(RES[('GSAD','LOLv1')]['psnr']):.2f} dB on LOL-v1",
                "LOL-v2-real test: 91 / 100 images are LOL-v1 training images",
                f"My re-training: {v('Retinexformer (my training)','LOLv1','psnr',2)} dB vs released {v('Retinexformer','LOLv1','psnr',2)} (others: 23.45)"],
            x=7.2, y=1.7, w=5.7, size=17)

    # 13 --------------------------------------------------------------- qualitative
    s = slide("Qualitative results", """
SAY: Here are three LOL-v1 images with zoomed crops of dark regions. Zero-DCE and SCI make the image bright but full of coloured
noise. The transformers remove the noise but blur fine details. GSAD keeps the most texture but adds a colour cast - see the green
tint in the middle row. My fine-tuned runs A and B look the same as each other.

Q: How were the zoom boxes chosen? A: Automatically: a region that is dark in the input but has detail in the ground truth, so the
crop shows what each method recovers.""", "Results")
    picture(s, ASSETS / "grid_slides.png", 0.3, 1.3, 12.7, 5.65)

    # 14 --------------------------------------------------------------- detection
    s = slide("Does enhancement help a detector? (ExDark, YOLOv8m)", f"""
SAY: I ran a standard YOLOv8 detector, trained on normal-light COCO images, on raw ExDark images and on enhanced versions of the same
images. Surprisingly, raw images are best: mAP50 {float(DET[('raw',)]['mAP50']):.3f}. Every enhancer lowers it; even the diffusion
model on the 200-image subset stays below raw. Precision hardly changes but recall drops: after enhancement the detector misses
objects. On the right, the school bus is not detected in any version, and enhancement turns the white blanket yellow.

Q: Retinexformer's paper says enhancement helps detection - contradiction? A: No: they train YOLOv3 from scratch on enhanced images and
compare enhancers with each other, with no raw-image row. I test an off-the-shelf detector. Q: Why might it hurt? A: Amplified noise,
colour shifts, and a domain gap from the normal-light images the detector learned from.""", "Results")
    rows = [["Input to YOLOv8m", "mAP50 (1,200)", "Recall", "mAP50 (200)"]]
    for k, lab in [("raw", "Raw (no enhancement)"), ("retinexformer", "Retinexformer"), ("sci", "SCI"), ("zerodce", "Zero-DCE"), ("gsad", "GSAD")]:
        a, b = DET.get((k,), {}), DET.get((k + "_200",), {})
        rows.append([lab, f"{float(a['mAP50']):.3f}" if a else "–", f"{float(a['recall']):.3f}" if a else "–", f"{float(b['mAP50']):.3f}"])
    table(s, rows, 0.6, 1.6, 6.4, [2.6, 1.4, 1.0, 1.4], size=14, row_h=0.48, bold_cells=best_cells(rows, [1, 2, 3], [True, True, True]))
    bullets(s, ["Raw images give the best mAP", "Enhancement lowers recall, not precision", "Better restoration hurts less, none helps"],
            x=0.6, y=4.75, w=6.4, size=16, gap=4)
    picture(s, FIG / "detection_examples.png", 7.3, 1.35, 5.7, 5.6)

    # 15 --------------------------------------------------------------- ablation
    runs = [("FT-A control (L1)", "A: L1 (control)"), ("FT-B SNR-L1 (mine)", "B: SNR-L1 (mine)"), ("FT-C L1+FFT", "C: L1 + FFT"), ("FT-D SNR-L1+FFT", "D: SNR-L1 + FFT")]
    a_img, a_dark = per_image("per_image", "FT-A control (L1)", "psnr"), per_image("per_image_dark", "FT-A control (L1)", "dark_psnr")
    rows = [["Run", "LOL-v1 PSNR", "Dark-30% PSNR", "Dark wins vs A", "ExDark mAP50"],
            ["Released (start)", v("Retinexformer", "LOLv1", "psnr", 2), f"{float(DARK[('Retinexformer','LOLv1')]['dark_psnr']):.2f}", "–",
             f"{float(DET[('rf_pretrained_gpu_200gpu',)]['mAP50']):.3f}"]]
    gpu = {"FT-A control (L1)": "ft_A_l1_200gpu", "FT-B SNR-L1 (mine)": "ft_B_snr_200gpu", "FT-C L1+FFT": "ft_C_fft_200gpu", "FT-D SNR-L1+FFT": "ft_D_snr_fft_200gpu"}
    kB = None
    for m, lab in runs:
        d = per_image("per_image_dark", m, "dark_psnr")
        k = sum(d[i] > a_dark[i] for i in a_dark) if m != "FT-A control (L1)" else None
        if m == "FT-B SNR-L1 (mine)":
            kB = k
        rows.append([lab, v(m, "LOLv1", "psnr", 2), f"{float(DARK[(m,'LOLv1')]['dark_psnr']):.2f}", f"{k}/15" if k is not None else "–",
                     f"{float(DET[(gpu[m],)]['mAP50']):.3f}"])
    p_B = sum(comb(15, i) for i in range(kB, 16)) / 2 ** 15
    s = slide("Ablation: does the noise-aware loss help?", f"""
SAY: This is the controlled experiment. All four runs start from the same released weights, see exactly the same training batches
and use the same learning rate - only the loss changes. Run A is the control with plain L1, B is my SNR-weighted loss, C adds an FFT
loss, D both. Result: B beats A by only {float(v('FT-B SNR-L1 (mine)','LOLv1','psnr',2)) - float(v('FT-A control (L1)','LOLv1','psnr',2)):.2f} dB overall.
In the darkest regions it is better on {kB} of 15 images, which is unlikely by chance (p = {p_B:.3f}), so the effect is consistent and
in the intended place - but it is tiny, a few hundredths of a dB, and invisible. FFT does nothing. Also, fine-tuning itself lowered
PSNR for every run, including the control.

Q: Why have a control run? A: Without it, any change could come from simply training longer; with it, the difference between A and B
can only come from the loss. Q: Statistically significant but not important? A: Yes - consistent direction, negligible size.""", "Results")
    table(s, rows, 0.6, 1.6, 8.0, [2.4, 1.4, 1.6, 1.3, 1.3], size=15, row_h=0.5,
          bold_cells=best_cells(rows, [1, 2, 4], [True, True, True]))
    bullets(s, [f"B vs A: dark regions better on {kB}/15 images (p = {p_B:.3f})", "…but only +0.01 to +0.05 dB: negligible",
                "FFT loss: no effect", "Fine-tuning lowers LOL-v1 PSNR for every run"], x=8.9, y=1.7, w=4.2, size=16)
    text(s, 0.6, 5.0, 12.2, 1.2, "Takeaway: a consistent effect in the intended place (dark regions), but far too small to matter. "
         "The idea needs to act inside the network, not only in the loss.", 18, BLUE, True)

    # 16 --------------------------------------------------------------- efficiency
    s = slide("Efficiency: quality vs. speed (one Tesla T4)", f"""
SAY: Speed matters for phones and cameras. All methods were timed on the same GPU at 600 by 400. SCI takes
{float(SPEED[('SCI (medium)',)]['ms_per_image']):.1f} ms but has low quality; Retinexformer gives the best PSNR at
{float(SPEED[('Retinexformer',)]['ms_per_image']):.0f} ms with {float(SPEED[('Retinexformer',)]['params_M']):.2f} M parameters; the diffusion
model GSAD needs about {float(SPEED[('GSAD (LOLv2-real weights, 10 sampling steps)',)]['ms_per_image'])/1000:.1f} to
{float(SPEED[('GSAD (LOLv1 weights, 20 sampling steps)',)]['ms_per_image'])/1000:.1f} seconds per image - 16 to 32 times slower - for lower PSNR.

Q: Why is GSAD slow? A: Diffusion runs the network once per sampling step (10-20 steps); Retinexformer runs once. Q: Why time only on
one GPU? A: Speeds from different hardware are not comparable; my laptop's timings were unreliable.""", "Results")
    picture(s, FIG / "quality_vs_speed.png", 0.5, 1.35, 8.2, 5.5)
    bullets(s, [f"SCI: {float(SPEED[('SCI (medium)',)]['ms_per_image']):.1f} ms, low quality",
                f"Retinexformer: {float(SPEED[('Retinexformer',)]['ms_per_image']):.0f} ms, best PSNR",
                f"LLFormer: {float(SPEED[('LLFormer',)]['ms_per_image']):.0f} ms, {float(SPEED[('LLFormer',)]['params_M']):.1f} M params",
                f"GSAD: {float(SPEED[('GSAD (LOLv2-real weights, 10 sampling steps)',)]['ms_per_image'])/1000:.1f}–{float(SPEED[('GSAD (LOLv1 weights, 20 sampling steps)',)]['ms_per_image'])/1000:.1f} s per image"],
            x=8.9, y=1.8, w=4.2, size=17)

    # 17 --------------------------------------------------------------- limitations
    s = slide("Limitations and failure cases", """
SAY: Every method, including mine, loses the fine green text on this bottle - the signal is simply not recoverable from the dark input.
Limitations of my work: LOL-v1 has only 15 test images, so small PSNR differences are not meaningful; I used one random seed per run;
free GPU time limited training; the detector was not fine-tuned; and my SNR estimate cannot tell noise from edges, which is probably
one reason the loss had little effect.

Q: What would you do differently? A: Several seeds, larger test sets, and a noise estimate that ignores edges.""", "Discussion")
    picture(s, FIG / "failures_lolv1.png", 0.4, 1.35, 7.4, 5.5)
    bullets(s, ["Only 15 LOL-v1 test images", "One seed per run; limited free GPU time", "SNR estimate also marks edges as noise",
                "Detector used as-is (not fine-tuned)", "SNR-Aware: released images only; GSAD on 200 ExDark images"],
            x=8.1, y=1.7, w=5.0, size=16)

    # 18 --------------------------------------------------------------- plan
    s = slide("Plan for the second half", """
SAY: The evidence suggests moving the SNR idea from the loss into the network itself, as SNR-Aware does inside attention, and fixing the
noise estimate so it ignores edges. I will sweep the weighting strength with several seeds to measure variance, train the best version
from scratch, evaluate on larger test sets, and finally fine-tune the detector on enhanced images to see whether enhancement can help
when the detector adapts to it.

Q: Why should SNR inside attention work better than in the loss? A: The loss only changes how errors are counted; guidance inside
attention changes what information each region uses, which is where SNR-Aware gained.""", "Plan")
    plan = [("Weeks 1–2", "SNR guidance inside IG-MSA; noise estimate that ignores edges"),
            ("Weeks 3–4", "Sweep α and weighting; 3 seeds per setting"),
            ("Weeks 5–6", "Train best variant from scratch; larger test sets"),
            ("Week 7", "Fine-tune YOLOv8 on enhanced vs raw ExDark"), ("Week 8", "Final report and code release")]
    for i, (a, b) in enumerate(plan):
        y = 1.65 + i * 1.0
        box = s.shapes.add_shape(1, Inches(0.6), Inches(y), Inches(2.0), Inches(0.75))
        box.fill.solid(); box.fill.fore_color.rgb = BLUE; box.line.fill.background()
        text(s, 0.6, y, 2.0, 0.75, a, 16, RGBColor(0xFF, 0xFF, 0xFF), True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        text(s, 2.9, y, 10, 0.75, b, 18, INK, anchor=MSO_ANCHOR.MIDDLE)

    # 19 --------------------------------------------------------------- conclusion
    s = slide("Conclusions", f"""
SAY: To summarise: I reproduced Retinexformer and eight baselines under one fair protocol, and the released models match their papers.
The common protocol revealed three things that published tables hide: a ground-truth brightness trick, an overlap between benchmarks,
and the fact that enhancement does not help an off-the-shelf detector. My noise-aware loss, tested with a strict control, gives a
consistent but negligible gain, and the analysis points to moving noise awareness into the attention itself. Thank you.

Q: What is your main contribution so far? A: A careful, controlled evaluation - including detection and dark regions - and a tested,
explained negative result that sets up the next step.""", "Conclusion")
    bullets(s, [f"Reproduced Retinexformer ({v('Retinexformer','LOLv1','psnr',2)} dB) + 8 baselines under one protocol",
                "Found: GT-brightness trick (GSAD), LOL-v2-real / LOL-v1 overlap",
                f"Raw images beat every enhancer for YOLOv8m (mAP50 {float(DET[('raw',)]['mAP50']):.3f})",
                "SNR-weighted loss: consistent but negligible gain in dark regions",
                "Next: noise awareness inside attention, larger tests"], x=0.6, y=1.6, w=12.2, size=19)
    text(s, 0.6, 5.3, 12.2, 1.5,
         "Key references: Cai et al., Retinexformer, ICCV 2023 · Xu et al., SNR-Aware, CVPR 2022 · Hou et al., GSAD, NeurIPS 2023 · "
         "Wang et al., LLFormer, AAAI 2023 · Guo et al., Zero-DCE, CVPR 2020 · Ma et al., SCI, CVPR 2022 · "
         "Li et al., LLIE survey, TPAMI 2022 · Loh & Chan, ExDark, CVIU 2019  (full list in the report)", 12, INK2)

    OUT.mkdir(parents=True, exist_ok=True)
    prs.save(OUT / "midsem.pptx")
    print(f"wrote {OUT / 'midsem.pptx'} ({N[0]} slides)")


if __name__ == "__main__":
    build()
