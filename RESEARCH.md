# clone-voice — Literature Review

Status: v1 · Conducted: 2026-10-05, to ground `DESIGN.md` in actual research before Batch 0 starts.

Scope note: this is a targeted review aimed at actionable findings for this specific app, not an exhaustive survey. Most citations are peer-reviewed/primary (arXiv, conference proceedings); a couple are secondhand or very recent/unreviewed and are flagged inline as lower-confidence — treat their specific numbers as directional, not settled fact.

## Finding 1 — Alignment-free training is the modern standard; re-scope the Whisper step

Every strong 2023–2025 system reviewed avoids feeding an *external* frame-level forced alignment into the generative model itself: VALL-E has no duration model (implicit via AR length); VITS/GPT-SoVITS's decoder and StyleTTS2 use an *internal*, differentiable alignment (monotonic alignment search / Gaussian duration upsampling) trained jointly; F5-TTS/E2TTS pad text to audio length and let attention find the correspondence; CosyVoice does pure next-token prediction. **Voicebox** (Meta) is the one modern system that *did* require classic forced-aligned phoneme durations — and its own successor, **E2TTS**, deleted the aligner and duration predictor a year later, reporting this as a simplification that didn't hurt (arguably helped) naturalness/robustness.

This doesn't make ASR useless — it answers a different problem. VALL-E's training corpus was pseudo-labeled by a forced aligner *once, upstream*, purely because the raw audio had no transcripts. GPT-SoVITS's own documented fine-tuning recipe runs ASR over each training clip plus manual proofreading before fine-tuning — i.e. transcript generation/verification, not duration conditioning fed into the model.

**Decision:** keep the Whisper step, but re-scope it from "forced alignment for the model" to **transcript verification + long-clip segmentation**. Never pass word/frame timestamps into GPT-SoVITS as duration or alignment conditioning — nothing in this model lineage consumes that signal. Add a transcript-mismatch confidence check: AR/implicit-alignment models are known to hallucinate, skip, or repeat words on mismatched prompts, so catching bad input before training matters more here than it would for older explicit-duration systems.

Secondary note (SV2TTS, Jia et al. 2018): a speaker encoder's ability to generalize to an unseen voice depends on the *diversity of the encoder's own pretraining speakers*, not on how much audio the user supplies. Worth remembering during troubleshooting — if cloning quality is poor on an objectively clean, well-recorded clip, the fix may be a better-pretrained encoder/model, not "ask the user for more audio."

## Finding 2 — Data-efficiency curve is front-loaded; revised duration tiers

Our original tiers (placeholders, no evidence) assumed similarity keeps climbing roughly linearly out to 10+ minutes. The literature says the curve saturates much earlier for **speaker similarity** specifically:

- AdaSpeech (Microsoft, ICLR 2021): quality "drops quickly" below ~10 sentences (~20–30s); ~20 sentences (~1 min) is their adaptation sweet spot.
- An XTTS-v2 reference-duration ablation (1/3/6/10/20/40s; unreviewed MA thesis, Univ. Groningen, 2025 — lower confidence) found the big jump between 1–10s with diminishing returns past ~20s.
- Neural Voice Cloning with a Few Samples (Arik et al., Baidu 2018): speaker classification accuracy ~60% (1 sample) → 75% (5) → 85% (10) — most gain captured by 5–10 *distinct utterances*, not more seconds of one utterance.

What *does* keep improving with more minutes of data is naturalness/prosody range and robustness across varied text, not raw similarity — these are different axes and our tier design should track them separately.

**Revised tiers (replacing the placeholders in `DESIGN.md` §8.2):**

| Tier | Strategy |
|---|---|
| **< 15s** | Zero-shot only, no weight updates. Below ~10–20s, fine-tuning reliably hurts more than it helps (AdaSpeech ablation). |
| **15s – 1min** | Minimal adaptation: tiny conditioning path only (speaker embedding + conditional-layer-norm scale/bias, ~5K params, à la AdaSpeech) or attention-only rank-limited LoRA. |
| **1 – 5min** | Light LoRA/adapter fine-tune (attention + decoder conditioning). |
| **5 – 20min** | Standard fine-tune: wider LoRA rank / more unfrozen blocks. Gains here are mostly naturalness/prosody range — similarity is largely already saturated. This is also roughly where a held-out validation split becomes statistically meaningful. |
| **20min+** | Extended fine-tune with validation split, as originally planned. |

Important distinction confirmed by the thesis and by zero-shot papers generally: **reference-clip length for zero-shot conditioning** (sweet spot ~6–10s, diminishing after ~20s) and **amount of data used to update weights** (table above) are different axes — don't conflate them in the tier logic.

## Finding 3 — Parameter-efficient fine-tuning (LoRA/adapters), not full fine-tune, at every tier below the deepest

Multiple 2024–2025 papers show LoRA/adapter/partial fine-tuning reaches 90–98%+ of full-fine-tune quality at 0.25–11% of trainable parameters, with materially lower overfitting/catastrophic-forgetting risk — exactly the risk profile of fine-tuning on a 30-second to few-minute clip:

- **VoiceTailor** (SNU, 2024): LoRA on 0.25% of params matches/beats baselines, including beating XTTS-v2's SMOS with far less data/model size.
- **NanoVoice** (SNU, 2024): batch-wise LoRA adaptation — comparable quality, 4x faster training, 45% fewer parameters at 40-speaker scale.
- **LoRP-TTS** (2025): LoRA lifts speaker similarity by up to 30 percentage points specifically on **noisy, non-studio, single-recording** prompts — directly relevant since we can't assume studio-quality user uploads.
- **ADAPTERMIX** (Interspeech 2023): adapters on 11% of params give a 5% speaker-preference improvement using **under 1 minute** of data — validates our sub-1-minute tier being non-trivial if architected right.
- **CSP-FT** (2025): tuning only ~8% of layers matches full-fine-tune fidelity, trains 2x faster, and **mitigates catastrophic forgetting** — the main risk our adaptive tiers need to guard against throughout.
- IEEE Access (2024/2025) LoRA-on-VITS study: comparable quality to full fine-tune with ~90% fewer tuned parameters, confirmed across LibriTTS/VCTK/Common Voice/Korean.

**Decision:** Batch 3's training orchestration should target parameter-efficient (LoRA/adapter/conditional-layer-norm) tuning as the default mechanism at every tier except possibly the deepest (20min+), not full-model fine-tuning. **Open risk:** whether GPT-SoVITS's own training scripts expose a LoRA/adapter mode out of the box, or only full fine-tune — to be confirmed once we can actually inspect/run the code on the Linux box. If only full fine-tune is exposed, the fallback mitigation is small step counts + early stopping at the low tiers rather than attempting to patch in PEFT ourselves.

## Finding 4 — Automatic evaluation: a WER hard gate, then SECS + UTMOS ranking

For auto-ranking multiple generated candidates per request, and for a post-training quality signal:

1. **Hard gate:** transcribe each candidate with Whisper, compute WER against the (normalized) input text; discard/heavily penalize candidates above a threshold (~20–30% WER). This catches outright failures — garbled audio, dropped/repeated words — that continuous similarity/quality metrics can miss entirely.
2. **Rank survivors** by a weighted score: **SECS** (speaker-embedding cosine similarity — via ECAPA-TDNN or WavLM-SV; Resemblyzer is a lighter-weight option) weighted higher, **UTMOS/UTMOSv2** (automatic MOS prediction, no reference needed) weighted lower. Compute SECS against the **centroid** of reference embeddings if multiple reference clips/segments exist — more stable than comparing to one clip.

**How much to trust these as human-judgment proxies:**
- SECS reaches up to **0.78 Pearson correlation** with averaged human MUSHRA similarity scores — close to the ~0.84 ceiling set by human inter-rater agreement itself (Deja et al., Amazon, Interspeech 2022, 730K+ ratings).
- UTMOS system-level Spearman correlation with human MOS is **~0.9–0.99**, but utterance-level correlation is visibly noisier (~0.897) — trust it for ranking/averaging, not as a single precise absolute score. A follow-up paper ("Attacking UTMOS") shows it's adversarially gameable, so never optimize training directly against it — evaluation only.

**Decision:** for the post-training quality signal shown to the user (Project Detail page), synthesize ~5–10 held-out test sentences, average SECS (vs. reference centroid) and UTMOS across them, and present as a **directional band** ("strong match" / "likely good" / etc.) rather than a precise score — averaging tames utterance-level noise that a single number would hide.

## Finding 5 — Vocoder: HiFi-GAN vs BigVGAN, and a real tension with our earlier GPT-SoVITS version pick

- **HiFi-GAN** (Kong et al., NeurIPS 2020) remains a solid, efficient baseline (ships in GPT-SoVITS v1/v2/v2Pro, and in XTTS v2) but has a documented failure mode: its Leaky-ReLU generator extrapolates poorly out-of-distribution — i.e. it degrades specifically on **unseen recording conditions**, which is exactly what we should expect from arbitrary user-recorded reference audio.
- **BigVGAN** (NVIDIA, ICLR 2023) targets precisely that gap with Snake periodic activations + anti-aliased multi-periodicity; its own benchmarks show the HiFi-GAN gap *widening* under noise/OOD material and narrowing only on clean studio speech. GPT-SoVITS **v3 adopted BigVGAN v2** (24kHz) specifically to improve timbre similarity, but introduced a known metallic-artifact issue from non-integer upsampling on small fine-tunes. **v4** replaced that with a custom 48kHz vocoder fixing the artifact — but upstream itself says v4 "needs more testing" vs v3.
- **Vocos** (ICLR 2024) matches BigVGAN-level MOS at a fraction of the compute — not adopted here since latency isn't our bottleneck in an offline, GPU-accelerated personal tool.

**Tension to flag explicitly:** our original design doc picked **v2Pro/v2ProPlus** on 2026 quality-per-compute comparisons — but v2Pro's vocoder is still HiFi-GAN-family, not BigVGAN. Given our top priority is similarity/naturalness on *unpredictable, self-recorded* audio (not compute efficiency), the OOD-robustness case for v3/v4's BigVGAN-family vocoder is directly relevant and cuts the other way. Upstream hasn't fully settled v3-vs-v4 either. **Decision:** don't pick a winner from secondhand research alone — add an explicit A/B step in Batch 3/4 comparing v2ProPlus against v4 (and possibly v3) on a deliberately imperfect (not pristine studio) test clip, since that's the realistic use case, and let real output decide.

## Finding 6 — Reference-audio quality thresholds

- **Duration/VAD:** ≥3s hard minimum, 6–10s recommended, ≥60% speech-to-total-duration ratio after VAD trimming — surfaced to the user as explicit warnings rather than silently accepting anything shorter/noisier.
- **Loudness normalization:** keep unconditional — EBU R128 / ITU-R BS.1770 to ≈ -23 LUFS with a true-peak ceiling is standard practice in TTS data pipelines (more specific than our earlier vague "normalize to a target LUFS").
- **Denoising — "optional, flagged, light" was the right call, confirmed by multiple angles:**
  - Generic speech-enhancement models are shown to distort speaker-discriminative features (speaker-verification literature).
  - A reported production case: denoising raised a generic quality metric (DNSMOS 3.46→3.73) while *dropping* speaker similarity (SIM 66.7→63.8) on noisy input.
  - **RVCBench** (2026 — very recent, lower confidence than the above): denoising audio already degraded by noise/perturbation left similarity and MOS "noticeably inferior" to generating from a clean reference — i.e. denoising doesn't recover what noise already cost, it trades one error for another.
  - A domain-adversarial-training approach (2020) that builds noise-invariance into the cloning model itself outperformed a denoise-then-clone pipeline — the deeper fix is model robustness, not a preprocessing patch.
  - **Decision:** do not escalate to aggressive or default-on denoising even if it looks good on a few test clips; keep it optional and flagged exactly as designed.
- **Dereverberation:** literature isolating this specifically (vs. plain noise) is thin — treat with the same light/flagged philosophy but lower confidence than the denoising findings above.

## Finding 7 — Watermarking: defer, with an explicit trigger

**AudioSeal** (Meta FAIR/Inria/Kyutai, 2024) is cheap and effective: ~7ms to embed, ~3ms to detect, near-transparent quality cost (PESQ 4.47, STOI 0.997), robust to common edits (avg. AUC 0.97 vs. WavMark's 0.84), MIT-licensed. Cost is not the reason to skip it.

The reason to skip it **for now** is threat-model fit: it's built for platform-scale proactive detection of unknown, mass-distributed deepfakes, not one person generating clips for themselves. Its robustness claim also depends on the detector weights staying private — meaningless in a personal local app where anyone with app access has the detector too. There's no verifier in the loop for purely personal output, so the practical benefit today is near zero.

**Decision:** formally deferred, tied to a concrete trigger — **revisit if the tool is ever shared/distributed, or if outputs leave the user's own hands.** Cheap to add at that point since it's open-source and the integration cost is low. In the meantime, the genuinely free win is tagging AI-generated output in the exported file's own metadata (e.g. a comment field) — near-zero cost, worth doing in Batch 4's export step regardless.

## Paper index

1. Arik, Chen, Peng, Ping, Zhou — *Neural Voice Cloning with a Few Samples* (Baidu, 2018). [arXiv:1802.06006](https://arxiv.org/abs/1802.06006)
2. Jia, Zhang, Weiss et al. — *Transfer Learning from Speaker Verification to Multispeaker TTS* / SV2TTS (Google, NeurIPS 2018). [arXiv:1806.04558](https://arxiv.org/abs/1806.04558)
3. Chen, Wu, Wang et al. — *VALL-E* (Microsoft, 2023). [arXiv:2301.02111](https://arxiv.org/abs/2301.02111) · *VALL-E X*: [arXiv:2303.03926](https://arxiv.org/abs/2303.03926)
4. Shen et al. — *NaturalSpeech 2* (Microsoft, 2023). [arXiv:2304.09116](https://arxiv.org/abs/2304.09116) · *NaturalSpeech 3*: [arXiv:2403.03100](https://arxiv.org/abs/2403.03100)
5. Le, Vyas et al. — *Voicebox* (Meta, NeurIPS 2023). [arXiv:2306.15687](https://arxiv.org/abs/2306.15687)
6. Eskimez et al. — *E2 TTS* (Microsoft, 2024). [arXiv:2406.18009](https://arxiv.org/abs/2406.18009)
7. Chen et al. — *F5-TTS* (2024, ACL 2025). [arXiv:2410.06885](https://arxiv.org/abs/2410.06885)
8. Li, Han, Raghavan, Mischler, Mesgarani — *StyleTTS 2* (Columbia, NeurIPS 2023). [arXiv:2306.07691](https://arxiv.org/abs/2306.07691)
9. Alibaba — *CosyVoice* (2024). [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) · *CosyVoice 2*: [arXiv:2412.10117](https://arxiv.org/abs/2412.10117)
10. RVC-Boss et al. — *GPT-SoVITS* — no arXiv paper; GitHub README/wiki only. [github.com/RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
11. Chen, Tan, Ren, Xu, Sun, Zhao, Qin — *AdaSpeech* (Microsoft, ICLR 2021). [arXiv:2103.00993](https://arxiv.org/abs/2103.00993)
12. Yan, Tan, Li, Qin et al. — *AdaSpeech 2* (Microsoft/Tsinghua, ICASSP 2021). [arXiv:2104.09715](https://arxiv.org/abs/2104.09715) · *AdaSpeech 3* (Interspeech 2021): [arXiv:2107.02530](https://arxiv.org/abs/2107.02530)
13. Kim et al. — *VoiceTailor* (SNU, 2024). [arXiv:2408.14739](https://arxiv.org/abs/2408.14739)
14. Yeom, Park et al. — *NanoVoice* (SNU, 2024). [arXiv:2409.15760](https://arxiv.org/abs/2409.15760)
15. *LoRP-TTS* (2025). [arXiv:2502.07562](https://arxiv.org/abs/2502.07562)
16. *Leveraging LoRA for Parameter-Efficient Fine-Tuning in Multi-Speaker Adaptive TTS* (IEEE Access, 2024/2025).
17. *ADAPTERMIX* (Interspeech 2023). [arXiv:2305.18028](https://arxiv.org/abs/2305.18028)
18. *CSP-FT* (2025). [arXiv:2501.14273](https://arxiv.org/abs/2501.14273)
19. Deja, Sánchez, Roth, Cotescu — *Automatic Evaluation of Speaker Similarity* (Amazon, Interspeech 2022). [arXiv:2207.00344](https://arxiv.org/abs/2207.00344)
20. Saeki, Xin, Nakata, Koriyama, Takamichi, Saruwatari — *UTMOS* (U. Tokyo, Interspeech 2022). [arXiv:2204.02152](https://arxiv.org/abs/2204.02152)
21. Zhu, Qiye — *Zero-Shot Voice Cloning with Minimal Data: Impact of Reference Duration* (MA thesis, Univ. Groningen/Campus Fryslân, 2025 — unreviewed, lower confidence). [link](https://campus-fryslan.studenttheses.ub.rug.nl/708/)
22. Kong, Kim, Bae — *HiFi-GAN* (NeurIPS 2020). [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)
23. Lee, Kim, Chun, Yoon — *BigVGAN* (NVIDIA/KAIST, ICLR 2023). [arXiv:2206.04658](https://arxiv.org/abs/2206.04658)
24. Siuzdak — *Vocos* (ICLR 2024). [arXiv:2306.00814](https://arxiv.org/abs/2306.00814)
25. San Roman, Fernandez, et al. — *AudioSeal* (Meta FAIR/Inria/Kyutai, 2024). [arXiv:2401.17264](https://arxiv.org/abs/2401.17264)
26. Chen et al. — *WavMark* (2023). [arXiv:2308.12770](https://arxiv.org/abs/2308.12770)
27. Lorenzo-Trueba, Fang, Wang, Echizen, Yamagishi, Kinnunen — *Can we steal your vocal identity from the Internet?* (Odyssey 2018). [arXiv:1803.00860](https://arxiv.org/abs/1803.00860)
28. *Data Efficient Voice Cloning from Noisy Samples with Domain Adversarial Training* (2020). [arXiv:2008.04265](https://arxiv.org/abs/2008.04265)
29. *RVCBench* (2026 — very recent, lower confidence). [arXiv:2602.00443](https://arxiv.org/abs/2602.00443)
