# Viva preparation — 20 likely questions with short answers

Numbers are from `results/tables.md` / `report/report.pdf`. Practise saying each answer in 2–4 sentences.

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
**Bonus — one honest incident you can mention if asked about mistakes:** the first fine-tuning run started from the wrong
weights (a Google Drive ID pointed to LOL-v2-real weights). I found it because the results were implausible (worse even on
training images), ruled out other causes, confirmed by file checksum, archived the invalid results and re-ran with a
checksum check in every notebook.
