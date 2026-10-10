# Viva preparation — slide-by-slide guide

How to use this file while practising:
1. Open `presentation/midsem.pdf` next to this file and go slide by slide.
2. For each slide: read **"On the slide"**, then **"Explain"** (every bullet / table row / picture in simple words),
   then say the **"Say it"** part out loud (about 1 minute), then answer the **questions** without looking.
3. **My doubts (D1, D2, …)** are questions I asked myself while reviewing the slides — the panel may ask the same.
4. Part B at the end has 20 general questions; Part C has one honest "mistake" story.

All numbers come from `results/*.csv` (same as the report). If unsure about a number, say "about" and point to the table.

---------------------------------------------------------------------------------------------------------------------
## Slide 1 — Title

**On the slide:** title, "Mid-semester B.Tech Project evaluation", my name and roll number, supervisor, department.

**Say it:** "Good morning. My project is about making very dark photos look as if they were taken in good light, and
checking whether that actually helps a computer find objects in the dark. I will show what I reproduced, what I measured,
one idea of my own that I tested, and my plan for the second half."

**Q: Why this topic?** Night photos, surveillance and driving all produce dark images; detectors fail on them; noise in
dark regions is still an open problem (the TPAMI survey by Li et al. says so).

**D1. Is the title okay? Too long? Does it reflect the project?**
Length is normal for a research title (13 words). It has three parts and each matches real work:
- *Transformer-based enhancement* = Retinexformer reproduced and compared with 8 baselines → strong match.
- *Downstream object detection evaluation* = YOLOv8 on ExDark, raw vs every enhancer → strong match.
- *Noise-aware* = SNR-weighted loss, a pilot at midsem; noise awareness inside the network comes in the second half →
  it describes the full-year goal more than the midsem result.
Keep the title (it is likely the registered one).

**D2. How do I explain the title?** (the full explanation is on slide 2 below)
One sentence: "I make very dark photos bright and clear using a transformer network, pay special attention to the noise
that appears when brightening, and check whether the brightened photos actually help a computer detect objects."

---------------------------------------------------------------------------------------------------------------------
## Slide 2 — Project at a glance: what the title means

**On the slide:** one goal line; three blue cards (one per part of the title) with a status line each; a flow row at the
bottom: Dark photo → Enhancer → Image quality + Object detection.

**Explain:**
- **Goal line** — "make dark photos bright and clean, without amplifying noise, and check whether this helps a machine see."
  This is the whole project in one line.
- **Card 1 – Transformer-based enhancement.** Main model = Retinexformer (ICCV 2023). A transformer is a network built on
  *attention*: each part of the image decides which other parts are useful to it, so it can use information from the whole
  image (a CNN only sees a small neighbourhood). I reproduced it and compared it with 8 other methods using one common
  evaluation script ("one protocol"). *Status: 25.15 dB vs paper 25.16 → pipeline verified* (see D4).
- **Card 2 – Noise-aware.** When you brighten a dark image, the noise is brightened too, and the darkest parts are brightened
  most, so they become the noisiest. My SNR-weighted loss computes a signal-to-noise map from the dark input and gives
  low-SNR (dark/noisy) pixels more weight during training. Tested against a control run (same everything except the loss).
  *Status: pilot tested; next step is noise awareness inside the network.*
- **Card 3 – Downstream detection evaluation.** "Downstream" = a later task that uses the enhanced image. I run a ready-made
  detector (YOLOv8m) on 1,200 dark ExDark images: once on the raw images, once on each enhanced version. *Finding: better-looking ≠ better
  detection (raw best, mAP50 0.662)* (see D5).
- **Flow row** — the pipeline of the whole project: a dark photo goes into an enhancer; the output is judged in two ways:
  image quality (PSNR, SSIM, LPIPS, NIQE) and object detection (YOLOv8 mAP).

**Say it (30 s):** "My title has three parts. Low-light enhancement means turning a dark photo into one that looks well lit.
Transformer-based: my main model is Retinexformer, which I reproduced and compared with eight methods. Noise-aware: brightening
amplifies noise most in the darkest regions, so I test a loss that gives those regions more weight. Downstream detection: I
also ask whether enhancement actually helps a detector find objects in the dark. The rest of the talk follows these three parts."

**Questions:**
- *What do you mean by noise-aware?* The method uses an estimate of where the noise is (the SNR map) and treats those regions
  differently — here, more weight in the training loss.
- *Why transformer and not CNN?* Attention uses the whole image to decide how much to brighten each part; Retinexformer keeps it
  cheap by computing attention across channels instead of pixels.
- *Why detection?* Real uses (surveillance, driving) care whether machines can see objects, not only whether the photo looks nice.

**D2 (full). How do I explain each word of the title?**
- *Low-light image enhancement*: turn a dark photo into one that looks well lit. Problems: darkness, noise, colour shift.
- *Transformer-based*: attention lets each region use information from the whole image. Retinexformer = Retinex theory
  (image = reflectance × illumination) + transformer; cheap because attention is across channels, not pixels.
- *Noise-aware*: brightening also brightens noise, most in the darkest parts. An SNR map (signal vs noise per pixel) finds
  those regions and the loss gives them more weight.
- *Downstream detection evaluation*: check whether enhancement helps YOLOv8 find objects, not only whether PSNR goes up.

**D3. Are we doing anything new, or just copying papers?**
- *Taken from papers:* all models (official code + weights), datasets, metrics, and the SNR-map idea (SNR-Aware, CVPR 2022).
- *My own work:* the SNR-weighted loss design with rank normalisation (I showed min-max fails: 93% of pixels got weight ≈ 2);
  the controlled A-vs-B experiment with a sign test; one fair benchmark of 9+ methods with the same script, plus dark-region
  PSNR and speed on one GPU; the detection study (raw beats every enhancer); the protocol findings (GSAD's GT-brightness trick
  worth 4.1–8.5 dB; 91/100 LOL-v2-real test images are LOL-v1 training images); the re-training gap (23.10 vs 25.16).
- *How to say it:* "The first half built a reliable base: I reproduced the main paper and eight baselines under one fair
  protocol and found two protocol issues. My SNR-weighted loss gave a consistent but very small gain, so I learned that changing
  only the loss is not enough; in the second half the noise information goes inside the network."
- Do NOT call the loss a success; do NOT say "first to discover" — say "I found / verified".

**D4. "25.15 dB vs paper 25.16" — is that a bad result?**
No, it is good. It is a reproduction check, not a competition: I ran the authors' released weights through MY evaluation script
and got the paper's number (0.01 dB difference). That proves my data, PSNR code and inference are correct, so all my other
numbers can be trusted. If I had got 23 or 27, my pipeline would be broken. The real gap is a different row: my own training
from scratch = 23.10 dB (slide 13).

**D7. What are the "8 other methods"?** (slide 2)
1 HE (histogram equalization), 2 CLAHE, 3 Gamma correction — classical, no AI;
4 Zero-DCE (2020) — small AI that learns brightness curves without paired photos; 5 SCI (2022) — tiny, very fast AI;
6 SNR-Aware (2022) — transformer that also uses a noise map (gave me the SNR idea); 7 LLFormer (2023) — transformer for 4K/8K;
8 GSAD (2023) — diffusion model, realistic but very slow. My own re-training of Retinexformer is an extra row.
Say: "Eight methods covering every generation: three classical, two small fast AI models, two transformers, one diffusion model."

**D8. What is 25.15 dB and what does it signify?** (slide 2)
dB is the unit of PSNR (peak signal-to-noise ratio). LOL-v1 gives pairs: a dark photo and the correct bright photo of the same
scene (ground truth). Compare the model's output with the ground truth pixel by pixel: small difference → high PSNR.
Formula: PSNR = 10·log10(255² / MSE), MSE = average squared pixel error. Rough guide on LOL-v1: dark input 7.77 dB, simple
methods ≈ 15 dB, strong AI ≈ 24 dB, Retinexformer 25.15 dB (best). Every +3 dB ≈ halves the error. 25.15 = average over the
15 LOL-v1 test photos.
It signifies two things: (1) Retinexformer's outputs are the closest to the correct photos among all methods tested;
(2) the paper reports 25.16 and I got 25.15 with my own code → my evaluation setup is correct, so my other numbers can be trusted.
If asked "is PSNR enough?": no — it only counts pixel differences; a slight brightness offset costs many dB even if the photo looks
fine. That is why I also use SSIM, LPIPS, NIQE and detection accuracy.

**Note on wording:** visible slide labels say "proposed" (not "mine"); in speech you can still say "my loss".

**D5. "Raw images best" for detection — is that the desired result?**
Not what I hoped, but a correct and useful finding. Hypothesis: enhancement helps YOLOv8. Result: raw images mAP50 0.662, every
enhancer lower (0.572–0.603). Holds on 1,200 and on 200 images; the loss is in recall, precision stays similar. Reason: YOLOv8
(trained on COCO) is already fairly robust to darkness, while enhancement amplifies noise and shifts colours, so images look
unlike what the detector learned. It shapes the next step (fine-tune the detector on enhanced images).
Say: "Looks better does not mean detects better — that is exactly why I evaluate on detection and not just PSNR."

---------------------------------------------------------------------------------------------------------------------
## Slide 3 — Why low-light enhancement?

**On the slide:** a large before/after picture; one row "Where dark images occur".

**Explain the picture:** LOL-v1 test image 79 (a kitchen cupboard). **Left** = the real dark input from the dataset — you can
hardly see anything. **Right** = the output when I ran Retinexformer (released weights) on it — the cupboard, glass doors and
steel pots are clearly visible with natural colours. This is our own result, not a paper figure.
**Explain the bullets:** phone night photography; surveillance cameras; night driving, robots, drones (all must work at night);
detectors miss objects in the dark (machines suffer too, not only people).

**Say it:** "On the left is a real photo from the LOL test set taken in very low light; on the right is the same photo after I ran
Retinexformer on it. Dark images appear in phones, surveillance, driving and robots, and detectors work much worse on them. But
simply brightening is not enough, because the noise is brightened too — that is the next slide."

**Questions:**
- *Can't we just increase the exposure?* Longer exposure → motion blur; higher ISO → more noise. Enhancement after capture avoids
  both but must deal with the noise already there.

**D6. Is this image from our own results?** Yes. Left = LOL-v1 test input 79 (dataset). Right = output of Retinexformer (released
weights) that I ran myself (`results/retinexformer/LOLv1/79.png`). No figure in the slides is copied from a paper.

---------------------------------------------------------------------------------------------------------------------
## Slide 4 — The problem: darkness + noise + colour shift

**On the slide:** left: our dark image 79 with a red box; right: three labelled crops of that box; three bullets; one blue
key-message line.

**Explain the pictures:**
- **Left (red box):** our LOL-v1 test image 79. The box marks a dark region chosen automatically (dark in the input, and where a
  simple enhancer adds the most extra noise compared with the ground truth).
- **Right, three crops of the same box:**
  1. *Input × 6 (just brighter)* — I simply multiplied the dark pixels by 6. It gets brighter, but it is full of grainy
     speckles (the noise was multiplied by 6 too) and the white wall looks grey-green.
  2. *Zero-DCE* — a simple learned enhancer: also bright, also noisy.
  3. *Ground truth* — the same scene photographed with a long exposure: clean, white wall, clear steel pot.
**Explain the bullets:**
- *Dark: little light reaches the sensor, details are hidden.*
- *Noisy: few photons → low SNR → speckles.* Light arrives in random packets (photons); with few photons the random
  fluctuation is large compared with the signal → low signal-to-noise ratio → grain.
- *Colour shift: weak light distorts colours* — see the grey-green wall in crop 1 vs the white wall in the ground truth.
**Explain the blue line (key message):** *Brightening multiplies noise as much as signal → enhancement must brighten AND
denoise.* If pixel = signal + noise, then 6 × pixel = 6 × signal + 6 × noise: the SNR does not improve. That is why simple
brightening is not enough, and why this project cares about noise.

**Say it:** "Dark images have three problems at once: dark, noisy and colour-shifted. On the right, multiplying the dark region by
six makes it brighter, but the noise is multiplied too and the white wall turns grey-green. Zero-DCE also brightens the noise. The
ground truth is clean. So enhancement must brighten AND denoise."

**Questions:**
- *Where does the noise come from?* Photon shot noise (random arrival of light) and sensor read noise. With few photons the
  fluctuation is large relative to the signal.
- *Why is the ground truth clean?* Longer exposure / more light → many more photons → high SNR.
- *Why × 6?* Just a simple gain that brings the dark region to roughly normal brightness, to show what pure brightening does.

---------------------------------------------------------------------------------------------------------------------
## Slide 5 — How the field evolved

**On the slide:** five boxes in a timeline (Classical → Supervised CNN → Zero-shot → Transformers → Diffusion), method names
under each, a row of orange boxes "What each stage left open", two bullets.

**Explain each stage:**
- **Classical (HE, CLAHE, Retinex/LIME):** hand-made rules. HE spreads the brightness histogram; CLAHE does it locally with a
  limit; LIME estimates illumination with Retinex theory. No learning, no noise model.
- **Supervised CNN (LLNet, Retinex-Net, KinD, MIRNet):** learn from pairs of dark + bright images. Better, but need paired data.
- **Zero-shot (EnlightenGAN, Zero-DCE, RUAS, SCI):** no pairs needed; tiny and fast; but they do not remove noise.
- **Transformers (SNR-Aware, LLFormer, Retinexformer)** — highlighted: attention captures long-range relations → best fidelity.
- **Diffusion (GSAD, Diff-Retinex):** generative models that create realistic texture, but run the network many times → slow.
**Orange row — what each stage left open (= why the next stage came):** classical: no learning, no noise model → CNNs learn
from data; CNNs: need paired dark/bright data → zero-shot methods need no pairs; zero-shot: brighten the noise too →
transformers restore better; transformers: not noise-aware, and attention is costly (Retinexformer fixes the cost with channel
attention, not the noise); diffusion: very slow because it runs the network many times.
**Bullets:** the survey (Li et al., TPAMI 2022) says no method wins everywhere and noise in dark regions is still open; this
project takes Retinexformer as the main model and tries to make it noise-aware.

**Say it:** "The field moved from hand-made rules, to learning from pairs, to methods that need no pairs, to transformers with
long-range attention, to slow but realistic diffusion models. Transformers are the current best for fidelity, so my main model
is Retinexformer."

**Questions:**
- *Why did transformers help?* A dark pixel can use information from distant, well-lit regions; a CNN sees only a local window.
- *What is "zero-shot / unsupervised" here?* Trained without paired ground truth, using hand-designed losses (e.g. exposure,
  smoothness).

---------------------------------------------------------------------------------------------------------------------
## Slide 6 — Key methods compared in this project

**On the slide:** a table: method, year/venue, key idea, limitation. These are the methods I actually ran.

**Explain each row:**
- **HE / CLAHE / Gamma (classical):** re-map brightness with a fixed formula. Gamma: output = input^0.4 (brightens darks more).
  Limitation: no noise model, colours shift.
- **Zero-DCE (CVPR 2020):** a small CNN predicts brightness curves for each pixel, trained without pairs. Brightens noise.
- **SCI (CVPR 2022):** self-calibrated illumination; at test time only a ~300-parameter block runs → extremely fast. Brightens noise.
- **SNR-Aware (CVPR 2022):** computes an SNR map; low-SNR regions use long-range attention, high-SNR regions local convolution.
  The only one that explicitly models noise → inspired my loss. Heavier, no Retinex model.
- **LLFormer (AAAI 2023):** transformer with axis-based attention, designed for 4K/8K images. 24.5 M params, slower.
- **Retinexformer (ICCV 2023):** illumination-guided channel attention (slide 9). Limitation: knows where it is dark, not where
  it is noisy.
- **GSAD (NeurIPS 2023):** diffusion model with a structure regulariser. Slow; its test script uses a ground-truth brightness trick.

**Say it:** "Notice the pattern: fast unsupervised methods do not remove noise, transformers are accurate, diffusion is slow.
SNR-Aware is the only one that explicitly measures noise — that idea inspired my loss."

**Questions:**
- *Why Retinexformer as main model?* Best PSNR in my tests on all LOL sets, only 1.61 M parameters, public code and weights,
  based on Retinex theory so it is explainable.

---------------------------------------------------------------------------------------------------------------------
## Slide 7 — Research gap and objectives of this midsem

**On the slide:** a table "Gap in the literature | What I did in the first half | Shown in", and one blue line.

**Explain each row:**
1. *Noise amplification in dark regions is unsolved* → I designed an SNR-weighted loss and tested it against a control run
   → shown in "My idea" and "Ablation".
2. *Retinexformer: illumination-aware, not noise-aware* — it uses brightness to guide attention, but a dark smooth wall and a dark
   noisy region look the same to it → I keep Retinexformer as the base and add noise information through the SNR map → "Main model".
3. *Methods judged by PSNR/SSIM, rarely by a machine task* → detection study with YOLOv8 + my dark-region PSNR → "Detection".
4. *Diffusion: quality at a large speed cost* → I timed every method on the same GPU → "Efficiency".
5. *Papers use different test protocols* (e.g. GSAD's ground-truth trick) → I reproduced Retinexformer + 8 baselines and scored
   all of them with ONE script → "Results", "Reproduction".
**Blue line:** the second-half plan is based on this evidence.

**Say it:** "Each row pairs a gap with what I did about it and where I show it."

**Questions:**
- *Difference between dark and noisy?* A dark smooth wall can be clean; a dark textured region can be very noisy. Illumination
  = how bright; SNR = how much of the signal is noise.
- *Is "different protocols" really a gap?* Yes — my reproduction shows GSAD's published PSNR includes a 4.85 dB boost from
  ground-truth brightness, so tables in papers are not directly comparable.

---------------------------------------------------------------------------------------------------------------------
## Slide 8 — Background: Retinex theory and attention cost

**On the slide:** left: I = R ⊙ L with five bullets; right: softmax(QKᵀ/√d)·V with four bullets.

**Explain the left (Retinex):**
- I = observed image; R = reflectance = the true colours of objects; L = illumination = light falling on the scene;
  ⊙ = multiply pixel by pixel.
- Enhancement = estimate L and replace it by brighter light (keep R).
- In real dark images both R and L contain noise; dividing by a small L (making it brighter) amplifies that noise.
**Explain the right (attention cost):**
- Attention: every token (pixel) builds a Query, Key, Value. QKᵀ compares every token with every other token → an N × N matrix.
- 600×400 image → N = 240,000 pixels → N² ≈ 58 billion entries → far too much memory.
- Retinexformer's trick: attention across channels, KᵀQ is only C × C (C = 40, 80 or 160 in its three levels) → small;
  so the cost grows only linearly with image size.

**Say it:** "Retinex says image = reflectance × illumination; enhancement replaces the illumination. Normal attention compares every
pixel with every other pixel, which is impossible for a 600×400 image; Retinexformer compares channels instead, so it is cheap."

**Questions:**
- *What are Q, K, V?* Query = what a token looks for, Key = what it contains, Value = what it passes on. Q·K similarity (after
  softmax) decides how much of each Value is mixed in.
- *Why divide by √d?* Keeps the dot products small enough that softmax does not saturate (all weight on one token).
- *Does channel attention lose spatial information?* Each channel is a whole feature map, so attention mixes whole maps; spatial
  detail is handled by the convolutions and the U-shape.

---------------------------------------------------------------------------------------------------------------------
## Slide 9 — Main model: Retinexformer (ICCV 2023)

**On the slide:** the architecture diagram (I drew it myself from the released code) and five bullets.

**Explain the diagram, top row (left → right):**
1. *dark input I.*
2. *Illumination estimator* (blue box): concatenates I with its channel mean (mean of R, G, B = rough brightness) → 1×1 conv →
   5×5 depth-wise conv → gives **light-up features F_lu** → 1×1 conv → **light-up map L**.
3. *I_lu = I ⊙ L + I* — the lit-up image: brighter but still noisy. The "+ I" is a residual (curved arrow): the network only
   learns how much to add.
4. *corruption restorer = IGT* — removes noise, artifacts and colour errors.
5. *enhanced output.*
**Bottom-left — the IGT (Illumination-Guided Transformer):** a U-shape with C = 40 channels. Going down: IGAB blocks then 4×4
conv with stride 2 (halve size, double channels: C → 2C → 4C with 1, 2, 4 heads). Going up: deconv ×2. Dashed arrows = skip
connections (copy detail from the encoder to the decoder). Last step: 3×3 conv and add I_lu. F_lu is down-sampled with the
features and fed to every block.
**Bottom-right — inside one IGAB:** Q, K, V from the features; **V ← V ⊙ F_lu** (the illumination guide: values are scaled by
how lit each region is); attention A = softmax(s·K̂ᵀQ̂) is C × C per head; then projection + position encoding (PE) + residual;
then a feed-forward network with LayerNorm + residual.
**Bullets:** estimator → light-up map; lit-up image still noisy; restorer = U-shaped transformer; IG-MSA = values × light-up
features, attention across channels; 1.61 M parameters, trained with ℓ1 loss (mean absolute error).

**Say it:** "Retinexformer has two parts. The estimator predicts a light-up map; multiplying and adding back gives a brighter but
noisy image. The restorer, a U-shaped transformer, removes the noise. Its special block multiplies the values by light-up features,
so differently lit regions are treated differently, and attention is across channels. Only 1.61 million parameters."

**Questions:**
- *Why I·L + I?* Residual form: the network learns only the change, which is easier and more stable.
- *What does illumination guidance add?* Dark regions can borrow information from well-lit regions in a controlled way.
- *What is ℓ1 loss?* Average of |output − ground truth| over all pixels.

---------------------------------------------------------------------------------------------------------------------
## Slide 10 — Proposed: noise-aware (SNR-weighted) loss

**On the slide:** a 7-panel figure (a–g) made from image 79, and the formula steps.

**Explain the figure:**
- **(a) dark input** — shown brightened (γ = 0.4) only so you can see it.
- **(b) local mean = signal** — 5×5 blur of the gray image: the smooth "true" brightness.
- **(c) |gray − mean| = noise** — what is left after removing the smooth part: noise (and, unfortunately, edges).
- **(d) SNR (log scale)** — signal ÷ noise. Bright, smooth areas (white cupboard) have high SNR; dark noisy areas low SNR.
- **(e) weight: rank (used)** — the loss weight per pixel: yellow = high weight (low SNR: dark inside of the cupboard, and the
  edges), blue = low weight (bright clean areas). Range about 0.67–1.33, average exactly 1.
- **(f) weight: min-max (rejected)** — almost uniform colour: nearly every pixel gets the same weight → useless.
- **(g) histogram** — orange (min-max): 93% of pixels have weight > 1.9, i.e. all ≈ 2 → the loss just doubles (like doubling
  the learning rate). Blue (rank): weights spread evenly from 1 to 2.
**Explain the steps:**
1. b = 5×5 mean of the gray input (signal). 2. n = |gray − b| (noise). 3. SNR = b / (n + ε); ε = 0.0001 avoids division by 0.
4. r = rank of each pixel's SNR in the image, scaled to [0, 1]. 5. w = 1 + α(1 − r) with α = 1 → lowest SNR gets 2, highest 1.
6. w ← w / mean(w) → average weight 1, so the overall loss size does not change, only the emphasis.
7. L = mean(w · |Ŷ − Y|) — weighted ℓ1 between output Ŷ and ground truth Y.

**Say it:** "I tell the network during training which pixels are noisy. From the dark input I compute an SNR map: blur = signal,
difference = noise, ratio = SNR. Low-SNR pixels get up to twice the weight in the ℓ1 loss. Two design choices: rank instead of
min-max, because min-max gave 93% of pixels the same weight; and dividing by the mean so only the emphasis moves."

**Questions:**
- *Why not min-max?* SNR has a huge tail; a few pixels with enormous SNR squash everyone else to the same weight.
- *Is the SNR map learned?* No — computed from the input, no gradients, nothing learned.
- *Weakness?* The blur difference also fires on edges and texture (yellow edges in panel e), so it is not a pure noise estimate.
- *Where did the SNR formula come from?* SNR-Aware (CVPR 2022) uses the same map inside its network; I use it in the loss.

---------------------------------------------------------------------------------------------------------------------
## Slide 11 — Experimental setup

**On the slide:** a data table, a metric table, and two lines (hardware, rules).

**Explain the data table:**
- *LOL-v1* — 15 paired test images (dark + bright of the same scene); main benchmark.
- *LOL-v2 real / syn* — 100 + 100 paired test images (real captures and synthetic darkening).
- *LIME / DICM / MEF* — 10 / 69 / 17 real dark photos without ground truth → only NIQE.
- *ExDark* — real night photos with object boxes in 12 classes (bicycle, boat, bottle, bus, car, cat, chair, cup, dog,
  motorbike, people, table); 1,200 images for detection ("GSAD: 200" because GSAD is slow).
**Explain the metric table:**
- *PSNR* (↑) — pixel error in decibels; higher = closer to ground truth.
- *SSIM* (↑, 0–1) — compares local structure, contrast and brightness.
- *LPIPS* (↓) — distance between deep-network features; lower = looks more similar to humans.
- *NIQE* (↓) — "naturalness" without ground truth; lower = more natural.
- *Dark-30% PSNR* (↑) — PSNR only on the darkest 30% of pixels of the input (where noise is worst); my addition.
- *mAP50* (↑) — detection accuracy: a box counts as correct if it overlaps the true box by ≥ 50% (IoU ≥ 0.5); averaged
  over classes.
**Hardware:** free Kaggle Tesla T4 GPU for training and all timings; laptop for evaluation.
**Rules:** no ground truth used at test time; no checkpoint picked by test score; no testing on images seen in training.

**Questions:**
- *Why exclude LOL-v2-real for LOL-v1-trained models?* 91 of its 100 test images are LOL-v1 training images.
- *Why do PSNR and SSIM disagree sometimes?* PSNR = pixel error (punishes brightness offsets heavily); SSIM = structure.

---------------------------------------------------------------------------------------------------------------------
## Slide 12 — Quantitative results (paired test sets)

**On the slide:** one table: 9 rows × (LOL-v1 PSNR/SSIM/LPIPS, LOL-v2-real PSNR/SSIM/LPIPS). Best value in bold. A blue
summary line: "Retinexformer: best PSNR · GSAD: best LPIPS (most natural-looking) · classical / zero-shot ≈ 15 dB".

**Explain the rows:**
- *Input* (7.77 dB) — the dark image itself; the starting point.
- *Gamma, Zero-DCE, SCI* (~14.5–14.9 dB on LOL-v1) — brighter, but noise and colour errors keep them low.
- *SNR-Aware** (24.61) — strong; * = evaluated on images released by its authors (their code could not be run).
- *LLFormer* (23.65) — strong; dashes on LOL-v2-real because its weights were trained on LOL-v1 (overlap).
- *GSAD* (22.73) — diffusion: lower PSNR but **best LPIPS** (0.103) and best LOL-v1 SSIM — looks most natural.
- *Retinexformer* (25.15 / 22.79) — **best PSNR** on both sets.
- *Retinexformer (re-trained)* (23.10) — my own training from scratch; gap explained on slide 13.

**Say it:** "All rows are scored by the same script. Retinexformer has the best PSNR on both sets. GSAD looks the most natural
(best LPIPS) but has lower PSNR. Classical and zero-shot methods stay around 15 dB."

**Questions:**
- *Why is GSAD lower in PSNR but better in LPIPS?* Diffusion makes realistic texture, but its overall brightness is off and PSNR
  punishes brightness errors heavily.
- *Why only ~15 dB for Zero-DCE?* It brightens but does not denoise or correct colours, and its brightness level differs from
  the ground truth.

---------------------------------------------------------------------------------------------------------------------
## Slide 13 — Reproduction and protocol findings

**On the slide:** left: table "Paper vs Ours" (PSNR, LOL-v1) and a blue conclusion line; right: three findings and a picture
pair proving the dataset overlap.

**Explain the table:**
- Retinexformer 25.16 → 25.15; SNR-Aware 24.61 → 24.61; LLFormer 23.65 → 23.65: **reproduced**.
- GSAD (GT trick) 27.84 → 27.57: with the authors' test trick I almost reproduce their number.
- GSAD (real output) – → 22.73: without the trick (the paper does not report this).
**Blue line:** "Released models reproduce their papers → my evaluation pipeline is correct" (see D4).
**Explain the three findings:**
1. *GSAD's test rescales outputs to the ground-truth brightness: +4.85 dB (LOL-v1).* Its test script scales every output so its
   average brightness equals the ground truth's — information a real camera never has. Worth 4.1–8.5 dB depending on the dataset.
2. *LOL-v2-real: 91/100 test images are LOL-v1 training images.* I compared small thumbnails of every LOL-v2-real test image with
   every LOL-v1 training image: 91 match exactly (difference 0), even with the same file numbers. So any LOL-v1-trained model
   would be "tested" on images it has seen → those rows are excluded.
3. *My re-training: 23.10 dB vs released 25.15 (GitHub #132: 23.45).* Same official config, 150k iterations; another user on the
   authors' GitHub got 23.45.
**Explain the picture:** left = LOL-v2-real TEST image 00771, right = LOL-v1 TRAIN image 771 — the same photo (a roof structure)
in both datasets. This is the proof of finding 2, from our own data.

**Questions:**
- *Why is your re-training 2 dB lower?* Run-to-run variance on a tiny dataset (485 train / 15 test), PyTorch 2 vs 1.11, a
  different GPU, and the released weights may be the best of several runs. My evaluation matches the training log exactly, so it
  is not a metric bug.
- *Is GSAD cheating?* It follows an older convention (KinD, LLFlow) and states it in its README — but it is not comparable with
  methods that do not use the ground truth.
- *How did you find the overlap?* By comparing 32×32 grey thumbnails of all 100 test images against all 485 training images;
  91 had zero difference.

---------------------------------------------------------------------------------------------------------------------
## Slide 14 — Qualitative results

**On the slide:** a grid. Columns: Input | Zero-DCE | SCI | Retinexformer | GSAD | FT SNR-L1 (B, proposed) | Ground truth.
Rows: image 79 (cupboard), its zoomed crop, image 493 (toys and a colour chart), its zoomed crop. Red boxes = zoomed region.

**Explain what to look at:**
- **Crop of 79 (steel pot):** Zero-DCE and SCI are bright but full of coloured speckle noise; Retinexformer, GSAD and B are clean.
- **Crop of 493 (white tube):** Zero-DCE / SCI noisy and bluish; Retinexformer and B clean but smooth — the small **green text**
  on the tube (visible in the ground truth) is lost; GSAD is brightest with the most texture but the background turns
  **greenish** (colour drift).
- **B (proposed) vs Retinexformer:** almost identical — matches the tiny numbers on slide 16.
- Zoom boxes were chosen automatically: dark in the input but with detail in the ground truth.

**Say it:** "Zero-DCE and SCI make the image bright but full of colour noise. The transformers remove the noise but smooth fine
detail. GSAD keeps texture but drifts in colour. My run B looks like Retinexformer."

---------------------------------------------------------------------------------------------------------------------
## Slide 15 — Does enhancement help a detector? (ExDark, YOLOv8m)

**On the slide:** a table, three bullets, and a picture (3 ExDark images × Raw (dark) | Retinexformer | B (proposed)) with the
caption "Our YOLOv8m detections: bus missed in all versions; enhancement turns the white blanket yellow".

**Explain the table:**
- mAP50 on 1,200 images: Raw **0.662** > Retinexformer 0.603 > SCI 0.595 > Zero-DCE 0.572.
- Recall: Raw 0.611 → 0.502–0.540 after enhancement (more objects missed).
- mAP50 on the 200-image subset (needed because GSAD is slow): Raw **0.707** > GSAD 0.633 > Retinexformer 0.628 > SCI 0.625 >
  Zero-DCE 0.603.
**Explain the picture (boxes = YOLO detections with confidence):**
- **Row 1 (street lamp, bicycle):** bicycle found in all three (0.92 / 0.90 / 0.90) — enhancement changes little.
- **Row 2 (school bus):** NOT detected in any version, even though the enhanced image clearly shows a bus — and enhancement adds
  visible grain noise.
- **Row 3 (cats, bottle):** in raw, the cats are detected as "dog" (0.81, 0.25); after enhancement one big "dog" box (0.86/0.88)
  and the bottle confidence rises (0.32 → 0.57); but the **white blanket turns bright yellow** — a colour shift.
**Bullets:** raw gives the best mAP; enhancement lowers recall, not precision; better restoration hurts less, but none helps.

**Say it:** "A standard YOLOv8 trained on normal COCO images works best on the raw dark images. Every enhancer lowers mAP, mainly by
lowering recall. The images look better to us but are less familiar to the detector."

**Questions:**
- *Retinexformer's paper says enhancement helps detection — contradiction?* No: they train YOLOv3 from scratch on enhanced images
  and compare enhancers with each other (no raw-image row). I test an off-the-shelf detector.
- *Why might it hurt?* Amplified noise, colour shifts, and a domain gap from the images the detector learned from.
- *How did you get ExDark labels into YOLO format?* Converted the 12 ExDark classes to their COCO ids (People → person,
  Motorbike → motorcycle, Table → dining table) and checked them by drawing boxes on images.

---------------------------------------------------------------------------------------------------------------------
## Slide 16 — Ablation: does the noise-aware loss help?

**On the slide:** a table of 5 runs, four bullets, a takeaway line.

**Explain the table:**
- *Released (start)* — the released Retinexformer weights all runs start from (25.15 dB).
- *A: L1 (control)*, *B: SNR-L1 (proposed)*, *C: L1 + FFT*, *D: SNR-L1 + FFT* — fine-tuned for the SAME iterations, same batches,
  same learning rate; only the loss differs.
- *Dark-30% PSNR* — PSNR on the darkest 30% of pixels. A 24.89 vs B 24.91.
- *Dark wins vs A* — on how many of the 15 images the run beats A in dark regions: B 13/15, C 3/15, D 12/15.
- *ExDark mAP50* — all ≈ 0.626–0.627: no detection change.
**Explain the bullets:**
1. *B vs A: dark regions better on 13/15 images (p = 0.004).* Sign test: if there were no real effect, each image would be a coin
   flip; getting 13 or more wins out of 15 by chance has probability 0.004.
2. *…but only +0.01 to +0.05 dB: negligible.* Consistent direction, but far too small to see.
3. *FFT loss: no effect.* (FFT loss compares the frequency content of output and ground truth.)
4. *Fine-tuning lowers LOL-v1 PSNR for every run* (25.15 → ~24.3), including the control — so it is the extra training, not my loss.
**Takeaway:** consistent effect in the intended place, but too small to matter → noise awareness must act inside the network.

**Questions:**
- *Why have a control run?* Without it, any change could come from simply training longer; with it, A–B differences come only
  from the loss.
- *Significant but not important?* Yes — consistent direction, negligible size.
- *Why did all runs drop from 25.15?* With only 485 training images, extra training moves away from a released checkpoint that
  seems unusually good (my from-scratch training got 23.10). A and B drop equally, so the comparison is fair.

---------------------------------------------------------------------------------------------------------------------
## Slide 17 — Efficiency: quality vs. speed (one Tesla T4)

**On the slide:** a scatter plot and four bullets.

**Explain the plot:** x-axis = time per 600×400 image on one Tesla T4 GPU, **log scale** (each grid step = 10× slower);
y-axis = LOL-v1 PSNR. Each dot is a method, labelled with its parameter count. Top-left is ideal (fast and good).
- SCI and Zero-DCE: far left (fast) but low (~15 dB).
- Retinexformer: top, at 151 ms with 1.61 M params — best trade-off.
- LLFormer: 874 ms, 24.55 M params, lower PSNR.
- GSAD: far right (2.4 s with 10 steps, 4.9 s with 20 steps), 17.44 M params, lower PSNR.
**Bullets:** SCI 1.5 ms and Zero-DCE 19 ms, both low quality; Retinexformer 151 ms, best PSNR; LLFormer 874 ms, 24.5 M params;
GSAD 2.4–4.9 s per image.
**Blue line:** Retinexformer is the best quality–speed trade-off; GSAD is 16–32× slower (2,360 or 4,874 ms ÷ 151 ms) for lower PSNR.

**Questions:**
- *Why is GSAD slow?* Diffusion runs the network once per sampling step (10–20 steps); Retinexformer runs once.
- *Why time on one GPU?* Speeds on different hardware are not comparable; my laptop's timings were unreliable (memory swapping).

---------------------------------------------------------------------------------------------------------------------
## Slide 18 — Limitations and failure cases

**On the slide:** failure-case figure (left) and five bullets.

**Explain the figure:** columns Input | Retinexformer | GSAD | FT control (A) | FT SNR-L1 (B) | Ground truth.
- **Image 493, tube crop:** the ground truth has small **green text** on the tube; every method loses it — the signal is not
  recoverable from the dark input. GSAD also adds a greenish background.
- **Image 1, wooden cabinet crop:** the fine **wood grain** in the ground truth is smoothed away by all methods; GSAD's colour is
  more orange.
- A and B look the same — again matching the tiny numerical difference.
**Explain the bullets:**
- *Only 15 LOL-v1 test images* — small differences (tenths of a dB) are within noise.
- *One seed per run; limited free GPU time* — I cannot measure run-to-run variance yet.
- *SNR estimate also marks edges as noise* — panel (e) on slide 10; probably one reason the loss had little effect.
- *Detector used as-is (not fine-tuned)* — a fine-tuned detector might benefit from enhancement.
- *SNR-Aware: released images only; GSAD on 200 ExDark images* — limits of what I could run.

**Questions:**
- *What would you do differently?* Several seeds, larger test sets, and a noise estimate that ignores edges.

---------------------------------------------------------------------------------------------------------------------
## Slide 19 — Plan for the second half

**On the slide:** five time blocks.

**Explain each block:**
- *Weeks 1–2:* put SNR guidance inside IG-MSA (like SNR-Aware does in attention) + a noise estimate that ignores edges.
- *Weeks 3–4:* sweep α (weight strength) and weighting scheme, 3 seeds each → measure variance.
- *Weeks 5–6:* train the best variant from scratch; evaluate on larger test sets.
- *Week 7:* fine-tune YOLOv8 on enhanced vs raw ExDark → does enhancement help when the detector can adapt?
- *Week 8:* final report and code release.

**Questions:**
- *Why should SNR inside attention work better than in the loss?* The loss only changes how errors are counted; guidance inside
  attention changes which information each region uses — that is where SNR-Aware gained.

---------------------------------------------------------------------------------------------------------------------
## Slide 20 — Conclusions

**Explain each bullet:**
- Reproduced Retinexformer (25.15 dB) + 8 baselines under one protocol → a trustworthy base.
- Found: GT-brightness trick (GSAD) and LOL-v2-real / LOL-v1 overlap → how published numbers should be read.
- Raw images beat every enhancer for YOLOv8m (mAP50 0.662) → looks better ≠ detects better.
- SNR-weighted loss: consistent but negligible gain in dark regions → change only the loss is not enough.
- Next: noise awareness inside attention, larger tests.
- Then "Thank you — questions?" and the key references (full list in the report).

**Questions:**
- *Main contribution so far?* A careful, controlled evaluation — including detection and dark regions — and a tested, explained
  negative result that sets up the next step.

=====================================================================================================================
# Part B — 20 general questions (practise without slides)

**1. Why did you choose this problem?**
Dark images are everywhere (night photos, surveillance, driving) and they hurt both people and detectors. Brightening
alone amplifies noise, which the field itself (Li et al. survey) lists as unsolved. And whether enhancement helps machines
is rarely tested — so there is room for a careful study.

**2. What is Retinex theory?**
An image = reflectance × illumination. Reflectance is the true colour of objects, illumination is the light falling on them.
Enhancement = estimate the illumination and replace it with brighter light. In real dark images both parts are noisy, and
dividing by a small illumination amplifies that noise.

**3. How is Retinexformer different from earlier Retinex networks (Retinex-Net, KinD)?**
Earlier ones decompose the image in separate stages and ignore the noise that appears when brightening. Retinexformer is
one-stage: it predicts a light-up map, brightens (I·L + I), and then a transformer explicitly removes the corruptions.
Its attention is guided by illumination features, so dark and bright regions are treated differently.

**4. Why is self-attention expensive for images, and how does Retinexformer avoid it?**
Standard attention builds an N×N matrix, N = number of pixels. For 600×400 that is 240,000² ≈ 58 billion entries.
Retinexformer computes attention across channels: a C×C matrix (C = 40–160), so the cost grows only linearly with image size.

**5. What are Q, K, V?**
Query = what a token looks for, Key = what it contains, Value = what it passes on. Query–key similarity (after softmax)
decides how much of each value is mixed in. Dividing by √d keeps the numbers in a range where softmax works well.

**6. How do you compute the SNR map?**
Gray image → 5×5 blur = local signal; |gray − blur| = noise; SNR = blur / (noise + 0.0001). High SNR = clean and bright,
low SNR = dark or noisy. It is computed from the input, without gradients — nothing is learned.

**7. Why rank normalisation instead of min–max?**
SNR has a huge tail (a few pixels have enormous values). With min–max, 93% of pixels in our example got weight ≈ 2, so the
loss would just be 2× L1 — effectively a doubled learning rate, not a re-weighting. Rank spreads weights evenly from 0.67 to
1.33, and dividing by the mean keeps the average weight at exactly 1.

**8. Why did you need a control run?**
Fine-tuning for 10,000 more iterations can change results by itself. Run A (plain L1) gets exactly the same start weights,
data order, learning rate and iterations as my run B; only the loss differs. So any A–B difference is caused by the loss.

**9. So did your loss work?**
Honestly: barely. In dark regions B beats A on 13 of 15 images (sign test p ≈ 0.004), so the effect is consistent and in
the intended place — but it is only +0.01 to +0.05 dB, invisible. Overall +0.01 dB, no change on NIQE or detection.
"Statistically consistent" is not the same as "practically important".

**10. Why was the effect so small?**
(i) The weights only range 0.67–1.33. (ii) The noise estimate also fires on edges and texture (Fig. 4e), not only on noise.
(iii) Retinexformer already knows where it is dark through its light-up features. (iv) Short fine-tuning at a small
learning rate from an already converged model.

**11. Why did fine-tuning lower LOL-v1 PSNR for every run (25.15 → ~24.3)?**
Training loss improved, so learning worked; but with only 485 training images the extra training moves away from a released
checkpoint that seems unusually good. Our from-scratch training reached 23.10 and another user reported 23.45. Since A and B
drop equally, the comparison is still fair.

**12. Why is your re-trained Retinexformer 2 dB below the paper?**
Same official config, full 150k iterations, and my evaluation reproduces the training log exactly — so it is not a metric
bug. Likely: run-to-run variance on a tiny dataset (15 test images), PyTorch 2.11 vs 1.11 and a different GPU, and the
released weights may be the best of several runs. An independent report on the authors' GitHub (issue #132) got 23.45 dB.
The released weights do reproduce the paper (25.15 vs 25.16).

**13. What are the limits of PSNR and SSIM?**
PSNR only measures pixel error; a slightly wrong overall brightness costs many dB even if the image looks fine. SSIM looks at
local structure but still needs a reference. That is why I also report LPIPS (perceptual), NIQE (no reference), dark-region
PSNR and detection mAP. Example: GSAD has the best LPIPS but lower PSNR.

**14. How confident can you be with only 15 LOL-v1 test images?**
Not very, for small differences: a few tenths of a dB in average PSNR are within noise. That is why I also count per-image
wins, use the 100-image LOL-v2 sets, and say the B–A difference is negligible in size.

**15. What did you find about GSAD?**
Its test script rescales every output to the ground truth's average brightness — information a real camera does not have.
With the trick I reproduce its paper (27.57 vs 27.84 dB on LOL-v1); without it, the real output gives 22.73 dB, below
Retinexformer. The trick is worth 4.1–8.5 dB. GSAD still has the best LPIPS, and it is 16–32× slower.

**16. What is the LOL-v1 / LOL-v2-real overlap?**
91 of the 100 LOL-v2-real test images are identical to LOL-v1 training images (I compared thumbnails). So any model trained on
LOL-v1 would be "tested" on images it has seen. I never evaluate LOL-v1-trained models on LOL-v2-real.

**17. Why does enhancement not help YOLOv8?**
YOLOv8 was trained on normal-light COCO images and is already fairly robust to darkness. Enhancement amplifies noise and shifts
colours, so the image looks different from what the detector learned; recall drops (0.611 → 0.502–0.540) while precision stays
about the same. Better restoration hurts less, but none beats raw (0.662 mAP50).

**18. Doesn't Retinexformer's paper show that enhancement helps detection?**
Different protocol: they train YOLOv3 from scratch on enhanced ExDark images and compare enhancers with each other — their table
has no raw-image row. I test an off-the-shelf detector, so the two results answer different questions.

**19. Why did you run some things on the CPU instead of the Mac GPU?**
I compared outputs directly: on the Mac GPU (MPS), LLFormer was up to 2 dB wrong and YOLOv8 gave mAP50 0.42 instead of 0.65 on
the same images. Retinexformer, Zero-DCE and SCI matched the CPU to within one grey level. All timings come from one Kaggle T4.

**20. What will you do in the second half?**
Move noise awareness into the network: use the SNR map inside IG-MSA (as SNR-Aware does inside attention), with a noise estimate
that ignores edges; sweep α with 3 seeds; train the best variant from scratch; evaluate on larger test sets; and fine-tune
YOLOv8 on enhanced images to test whether enhancement helps when the detector can adapt.

---
# Part C — one honest mistake story

**Bonus — one honest incident you can mention if asked about mistakes:** the first fine-tuning run started from the wrong
weights (a Google Drive ID pointed to LOL-v2-real weights). I found it because the results were implausible (worse even on
training images), ruled out other causes, confirmed by file checksum, archived the invalid results and re-ran with a
checksum check in every notebook.
