# clone-voice — Design Document

Status: draft v1 · Last updated: 2026-10-05

## 1. Overview

A cross-platform, GPU-accelerated desktop tool for personal voice cloning:

1. User creates a **project**, uploading a reference audio clip (any length) and its reference text.
2. The app preprocesses the audio and **fine-tunes a voice model** on it — training effort scales with how much usable audio is provided (adaptive hybrid: near-instant for a short clip, deeper fine-tune for more data).
3. User enters arbitrary **target text**; the app synthesizes it in the cloned voice.
4. A **dashboard** lists all projects so the user can revisit, retrain, or generate from any of them later.

Top priority, in order: **voice similarity and naturalness** first, everything else (speed, polish) second.

## 2. Goals / Non-goals

**Goals (v1)**
- Cross-platform: developed on Windows, run/tested on Linux (GPU), should also run on macOS.
- GPU-accelerated via CUDA when available, CPU fallback otherwise.
- Local web-style GUI, usable as a quasi-native app window.
- English only.
- Multiple saved projects, each independently trainable and re-usable.
- Personal/local use — no licensing pressure, but we still default to permissively-licensed components.

**Non-goals (v1 — explicitly deferred)**
- Multilingual / cross-lingual cloning.
- Real-time/streaming synthesis (batch generation is fine).
- Multi-user / cloud / hosted deployment.
- Mobile.
- Signed, installer-style packaging (run from source for v1; see Batch 7).

## 3. Key architecture decisions

Grounded in a literature review conducted 2026-10-05 — see [RESEARCH.md](RESEARCH.md) for the full evidence and citations behind anything marked "research-backed" below.

| Decision | Choice | Why |
|---|---|---|
| Training depth | **Adaptive hybrid**, with research-backed tiers (§8.2) | Short clip → fast few-shot adaptation; more audio → deeper fine-tune. Tier boundaries and what changes at each one are no longer placeholders — see RESEARCH.md Finding 2. |
| Core TTS/voice-clone model | **GPT-SoVITS**, checkpoint **TBD between v2ProPlus and v4** | MIT-licensed, actively maintained (RVC-Boss/GPT-SoVITS, 59k+ stars), purpose-built for exactly this workflow. v2ProPlus was the initial pick on quality-per-compute grounds, but research surfaced a real tension: v2Pro's vocoder is still HiFi-GAN-family, while v3/v4 moved to BigVGAN-family specifically for better robustness on noisy/out-of-distribution audio — which matters more for us than compute efficiency, since we can't assume studio-quality uploads. That question is now secondary to the VRAM finding below. |
| Integration shape | **Confirmed 2026-10-06** (direct inspection of the RVC-Boss/GPT-SoVITS repo, commit `48b1a01`): **subprocess for training**, **direct Python import for inference**. Training (`s1_train.py`, `s2_train.py`/`s2_train_v3_lora.py`) is standalone-script-only — GPT-SoVITS's own `webui.py` drives it via `subprocess.Popen(..., shell=True)`, so our adapter does the same rather than trying to import unstable internals. Inference has a genuinely clean, maintainer-used class API: `GPT_SoVITS.TTS_infer_pack.TTS.TTS` + `TTS_Config` (their own `api_v2.py` imports it directly) — our Batch 4 inference adapter should import this directly instead of shelling out. | Resolves the integration-shape open risk from §13 without guessing. |
| Fine-tuning method | **Full fine-tune is the only option for v2/v2Pro/v2ProPlus (both stages) and for the GPT/s1 stage on every version.** LoRA exists **only** for v3/v4's SoVITS/s2 stage (confirmed in `s2_train_v3_lora.py` via the `peft` library, rank 16/32/64/128, default 32). The "default to parameter-efficient tuning below the deepest tier" plan from RESEARCH.md Finding 3 was general TTS literature, not this specific codebase — GPT-SoVITS itself doesn't give us that lever uniformly. See the VRAM finding below for why this matters a lot more than it sounds. | RESEARCH.md Finding 3 was about the broader literature; this row is what the actual upstream code supports. |
| Forced alignment / ASR | **faster-whisper**, rescoped to **transcript verification + long-clip segmentation** | Originally framed as "alignment for the model." Research shows modern TTS architectures (including GPT-SoVITS's own lineage) don't consume external alignment/duration conditioning at all — Whisper's real job is confirming the user's reference text matches what's actually said, and chopping long uploads into clean segments (RESEARCH.md Finding 1). |
| Candidate ranking / quality scoring | **Whisper WER hard gate, then SECS + UTMOS weighted ranking** | Replaces a vague "Resemblyzer similarity" plan with a two-stage approach validated against human judgment in the literature (RESEARCH.md Finding 4): discard candidates with broken transcription first, then rank survivors by speaker-similarity (weighted higher) and automatic naturalness (weighted lower). |
| GPU scope | **CUDA + CPU fallback only** | Covers the realistic use case; AMD/ROCm and MPS explicitly out of scope to keep the matrix small. |
| GUI | **Gradio UI**, launched either inside a **pywebview** window or as a plain served page | Minimal UI code, cross-platform "for free". pywebview needs a display; GPU boxes are frequently headless, so the launcher supports a `--headless` mode that just serves Gradio on the network. |
| License posture | Personal use, but default to permissive components anyway (MIT/Apache) | No current pressure, but keeps the option to share later cheap. |
| Watermarking | **Deferred**, explicit trigger: revisit if the tool is ever shared/distributed | AudioSeal is cheap and effective but solves a platform-scale detection problem we don't have yet (RESEARCH.md Finding 7). Free interim step: tag AI-generated output in export file metadata (§8.3). |

## 4. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Language | **Python 3.10** | Confirmed, not a placeholder: current PyTorch (2.14+) happily supports 3.10-3.14, so it isn't the constraint — GPT-SoVITS is. Its own tested range is 3.9-3.11 and its official conda setup uses `python=3.10` directly; 3.9 is already past CPython's security-support end-of-life. `scripts/setup_env.*` warns (doesn't hard-block) if the detected interpreter is outside 3.10/3.11. |
| ML runtime | PyTorch + torchaudio | CUDA build primary; CPU-only build as fallback. Exact CUDA toolkit version depends on the Linux box's driver — discovered in Batch 0. |
| Voice model | GPT-SoVITS (vendored, subprocess/CLI-wrapped — see §9 risk notes) | Training + inference engine. |
| ASR/transcript verification | faster-whisper | Confirms user-supplied reference text matches what's actually said, and segments long uploads — not model-level alignment (see §3). Also the WER hard-gate at generation time. |
| Audio I/O/DSP | ffmpeg (system dep), soundfile, pydub or librosa | Format conversion, resampling. |
| VAD | silero-vad (or webrtcvad as a lighter fallback) | Silence trimming / segment boundaries; also used to compute the speech-to-total-duration ratio (§8.1 threshold). |
| Loudness normalization | pyloudnorm | EBU R128 / ITU-R BS.1770 target ≈ -23 LUFS with a true-peak ceiling. |
| Speaker similarity (SECS) | Resemblyzer, or speechbrain ECAPA-TDNN/WavLM-SV if Resemblyzer proves too coarse | Cosine similarity vs. the centroid of reference embeddings; weighted higher in candidate ranking. |
| Naturalness scoring (UTMOS) | A UTMOS/UTMOSv2 checkpoint | No-reference automatic MOS prediction; weighted lower in candidate ranking, never used as a training objective. |
| Parameter-efficient fine-tuning | `peft` (HF) if GPT-SoVITS doesn't expose LoRA/adapters natively | Only needed if Batch 3 finds GPT-SoVITS's training scripts support full fine-tune only — see §13. |
| GUI | Gradio + pywebview | See §8. |
| Packaging (v1) | Plain `requirements-base.txt` + `requirements-{cuda,cpu}.txt` + setup scripts + `pyproject.toml` (editable install) | Split so torch's CUDA/CPU-specific `--index-url` never shares a file with plain-PyPI packages, avoiding index-precedence ambiguity. Installer/PyInstaller packaging deferred to Batch 7. |
| Testing | pytest, pytest-mock, pytest-cov | Markers `slow` / `gpu` separate heavy/real-model tests from the fast default suite. |

## 5. System modules

```
clone-voice/
  core/
    audio/        # io, resample/normalize, VAD, chunking
    align/         # whisper-based forced alignment
    projects/      # project CRUD, state machine, on-disk persistence
    train/         # adaptive recipe selection, background training runner, GPT-SoVITS adapter
    infer/         # text normalize/chunk, synthesis, candidate ranking, stitching, export
    models/        # device detection, model/weights registry & bootstrap
  gui/
    app.py         # Gradio Blocks wiring
    dashboard.py / new_project.py / project_detail.py / generate.py
    launcher.py    # pywebview wrapper + headless fallback
  scripts/
    setup_env.sh / setup_env.ps1
    check_gpu.py
    download_models.py
    run.sh / run.ps1
  tests/           # mirrors core/ structure
  projects/        # user data (see §6), not checked into VCS
```

## 6. On-disk project data model

```
projects/<project-id>/
  project.json             # name, created_at, state, language, config, training history
  raw/
    reference_audio.<ext>
    reference_text.txt
  processed/
    segments/              # chunked + aligned audio segments
    segments.json          # segment -> text -> timing -> alignment confidence
  training/
    checkpoints/
    logs/
    training_config.json   # recipe tier used, steps, device, duration
  output/
    <generation-id>.wav
    <generation-id>.json   # source text, timestamp, candidate scores
```

## 7. Project state machine

```
created -> preprocessing -> preprocessed -> training -> trained
                 |                              |
                 v                              v
               error <--------------------------+
```

- `trained` projects can re-enter `training` (retrain with more/different audio) and can **generate** at any time while in `trained`.
- `error` stores a human-readable message and the state it failed from, surfaced in the GUI.

## 8. Core pipelines

### 8.1 Preprocessing
1. Transcode arbitrary input audio (wav/mp3/flac/m4a/...) to mono WAV at the model's expected sample rate via ffmpeg.
2. Loudness-normalize to ≈ -23 LUFS (EBU R128 / ITU-R BS.1770) with a true-peak ceiling.
3. VAD-based silence trimming (leading/trailing + long internal pauses); compute the speech-to-total-duration ratio.
4. Quality gate, surfaced to the user rather than silently accepted (research-backed thresholds, RESEARCH.md Finding 6): hard-block below ~3s usable speech, warn below ~6–10s or below ~60% speech ratio.
5. Optional light denoise — only if noise is detected above a threshold, flagged to the user, never default-on. Research confirms aggressive/default denoising trades noise for lost speaker-identity cues and should stay conservative (RESEARCH.md Finding 6).
6. Transcript verification (not forced alignment — see §3): run faster-whisper over the audio and compare its output to the user-supplied reference text; flag low-confidence or mismatched regions instead of silently training on them.
7. Segment into ~2–10s training chunks on sentence/silence boundaries (from Whisper's own segmentation, not frame-level forced alignment); persist chunk audio + text + confidence to `processed/`.

### 8.2 Training (adaptive hybrid)
1. Sum usable segment duration from `processed/`.
2. Pick a recipe tier from duration — research-backed, not placeholder (RESEARCH.md Finding 2; distinguishes *amount of data used to update weights* from *zero-shot reference-clip length*, which are different axes):
   - `< 15s` → zero-shot only, no weight updates.
   - `15s–1min` → minimal adaptation: tiny conditioning path (speaker embedding + conditional-layer-norm scale/bias) or attention-only rank-limited LoRA.
   - `1–5min` → light LoRA/adapter fine-tune (attention + decoder conditioning).
   - `5–20min` → standard fine-tune: wider LoRA rank / more unfrozen blocks. Gains here are mostly naturalness/prosody range, not similarity (largely saturated by this point).
   - `20min+` → extended fine-tune with a held-out validation slice — the only tier where full-model fine-tuning is considered (§3 Fine-tuning method; all tiers below this default to parameter-efficient tuning).
3. Device selection: CUDA if available, else CPU — CPU training is likely only practical at the smallest tiers; the UI warns accordingly.
4. Runs as a background process; streams step/epoch/loss/ETA back to the GUI; supports cancel (keep-partial vs. discard, user's choice).
5. On completion: persist checkpoint + training metadata, transition project to `trained`, then immediately run the quality-signal check (§8.3 step 8) so the user gets a similarity/naturalness read without needing to generate anything manually first.

### 8.3 Inference / generation
1. Load the project's trained checkpoint (cached in memory across requests).
2. Normalize target text (numbers, abbreviations, punctuation).
3. Split into sentence-level chunks respecting the model's max input length.
4. Synthesize a small number of candidates per chunk at varied sampling seed/temperature.
5. **Hard gate**: transcribe each candidate with Whisper, compute WER against the input text; discard/heavily penalize candidates above ~20–30% WER (catches garbled/dropped/repeated-word failures that similarity scores alone can miss — RESEARCH.md Finding 4).
6. **Rank** surviving candidates by a weighted score: SECS (speaker-similarity vs. the reference-embedding centroid, weighted higher) + UTMOS (no-reference naturalness, weighted lower); auto-pick the best per chunk.
7. Stitch chunks with crossfade/silence-matched joins.
8. Export WAV (+ optional MP3) to `output/` with generation metadata, including the SECS/UTMOS scores achieved. Tag the exported file's own metadata as AI-generated (near-zero cost, the one watermarking-adjacent step done now — see §3).

**Quality signal** (run once right after training, §8.2 step 5, and re-runnable on demand): synthesize ~5–10 held-out test sentences, average their SECS (vs. reference centroid) and UTMOS, and show the user a directional band ("strong match" / "likely good" / etc.) rather than a raw score — averaging tames the utterance-level noise either metric has on its own (RESEARCH.md Finding 4).

## 9. GUI design & launch modes

Pages: **Dashboard** (list/open/rename/delete projects) · **New Project** (upload audio + text, name it; surfaces the §8.1 quality-gate warnings inline) · **Project Detail** (preprocessing report, start/monitor/cancel training, and the post-training quality-signal band from §8.3) · **Generate** (text in, audio out, per-project generation history with each result's SECS/UTMOS scores).

Launch modes (`scripts/run.*`):
- **Windowed** (default): Gradio server + pywebview window wrapping it. Needs a display/windowing system.
- **Headless** (`--headless`): Gradio server bound to `0.0.0.0`, URL printed to console, no pywebview — for a headless GPU box, accessed from a browser on another machine. This is likely the **primary** mode on the Linux test machine unless it has a desktop environment.

## 10. Batch plan

Work proceeds **one batch at a time**. Each batch = implementation + heavy tests + a short "what to run / what to expect" note from Claude. You install/run on the Linux GPU machine and report results back before the next batch starts.

| # | Batch | Delivers | Test focus | Exit criteria (what you confirm) |
|---|---|---|---|---|
| 0 | **Scaffolding & environment** | Repo layout, `requirements-{cuda,cpu}.txt`, `check_gpu.py`, `setup_env.*`, config/logging, pytest scaffold | Config loading; device-detection logic (mocked torch); directory creation in temp dirs | `setup_env` succeeds, `check_gpu.py` correctly reports your GPU/driver, batch-0 tests pass |
| 1 | **Audio preprocessing & alignment** | `core/audio/*`, `core/align/*` | Resample/normalize/VAD on synthetic audio (sine/silence, no real speech needed); alignment tested against a **mocked** whisper output; one optional real-model smoke test | Tests pass; optional real-audio spot check reported |
| 2 | **Project manager & dashboard backend** | `core/projects/*` (no GUI yet) | Full CRUD + state-transition tests, persistence round-trips — pure Python, no ML deps | Tests pass (should work even before ML deps are installed — good early sanity check) |
| 3 | **Training orchestration** | `core/train/*`, GPT-SoVITS adapter | Recipe-tier selection (pure logic); runner tested against a **fake** trainer (progress/cancel/error paths); one real tiny-scale fine-tune smoke test on your GPU | Mocked tests pass; real smoke-test timing/errors reported — riskiest integration point, expect iteration |
| 4 | **Inference / generation pipeline** | `core/infer/*` | Text normalize/chunk (pure); stitching math on synthetic numpy audio; ranking on mocked embeddings; one real end-to-end generation using the batch-3 checkpoint | Tests pass; real output reported (duration, non-silence, subjective quality) |
| 5 | **GUI** | `gui/*`, pywebview launcher + headless fallback | Gradio callback functions tested directly as plain functions; manual click-through checklist (not automated) | You walk through create→train→generate in the real UI, report UX bugs |
| 6 | **Model/weights bootstrap** | `scripts/download_models.py` (real downloads + checksums), cross-platform launch scripts | Download manager tested against mocked HTTP; checksum-verification tested with known good/bad hashes | Real multi-GB download completes on your box, checksums match |
| 7 | **Polish (stretch/v2)** | Consent/disclaimer step, settings (sampling params), audio-quality warnings in UI, PyInstaller exploration | Scoped once we get here | — |

Batches are sized so each test-and-report cycle stays fast — if a batch turns out too big in practice, it gets split further rather than bundled.

**Progress:** Batch 0 confirmed complete 2026-10-05 (20/20 tests passing, `check_gpu.py` verified against real hardware). Batch 1 confirmed complete 2026-10-05 (58/58 tests passing, including Batch 0's). Batch 2 confirmed complete 2026-10-05 (113/113 tests passing) — also where a `.gitignore` anchoring bug (`projects/`/`models/` matching nested `core/projects/`/`core/models/`, not just the root-level data dirs) was found and fixed; see the dev-workflow memory for the sync-mechanism implication.

## 11. Testing strategy

- Default `pytest` run must stay fast and dependency-light: anything that needs the real GPT-SoVITS/whisper models or a GPU is marked `@pytest.mark.slow` / `@pytest.mark.gpu` and skipped by default.
- Everything ML-heavy is dependency-injected/mocked in the default suite so tests are meaningful even before heavy ML packages are installed.
- Synthetic audio fixtures (sine tones, silence, noise) cover most of `core/audio` without needing a real voice recording.
- Coverage expectation: high on `core/` (pure logic), lighter on `gui/` (thin wiring layer, verified mostly by manual checklist).

## 12. Collaboration / workflow protocol

This is the agreed process for the whole project, not just this doc:

- Claude writes code and setup/download **scripts** only — never installs packages, downloads models, or executes code/tests on the Windows dev machine.
- One batch at a time: implementation + heavy tests + a short "run this, expect that" note.
- You install/run on the Linux GPU machine (or report you're using the headless path) and send back results — console output, errors, pass/fail.
- Claude fixes/iterates on the reported results before the next batch opens.

## 13. Open risks / decisions to confirm once real installs happen

### Critical finding (2026-10-06): training likely doesn't fit on this GPU at all

Direct inspection of the GPT-SoVITS repo plus its own changelogs and community reports: fine-tuning VRAM needs are **v3 full FT 12-14GB, v3/v4 LoRA 8GB (official floor), v2/v2Pro/v2ProPlus full FT ~6GB+ practical floor with no LoRA option at all**. The target machine has **3.7GB total VRAM**. Community "unsupported GPU" lists for this project explicitly name the desktop "RTX 3050 4G" card (which has *more* VRAM than this laptop part) as incompatible for training. `config.py` itself hard-forces CPU-only training below ~3.6GB total VRAM — this card clears that specific numeric gate by a hair, which is not the same as the real 6-14GB requirement being met. There is no dedicated low-VRAM mode in the project; the community's actual answer for underpowered GPUs is cloud/rented GPUs, not a local setting.

Net: **inference-only (no local weight updates, ever) is the realistic path on this hardware** unless real GPU VRAM availability changes (bigger/different GPU, or a cloud-burst training step). This is a decision for you, not something to silently route around — see the question below before Batch 3's code is written.

**Update (2026-10-06, real run):** less pessimistic than the secondhand research suggested. A real GPT-stage (s1) fine-tune on the `minimal` tier (16 chunks, 50.9s, 4 epochs, fp16) actually ran on the 3.7GB card — model loaded, GPU engaged, epoch 1 completed with reasonable metrics. One transient CUDA OOM warning appeared mid-epoch, but PyTorch recovered from it on its own rather than crashing. So the smallest tier may be genuinely viable locally; deeper tiers (more data, more epochs, and the SoVITS/s2 stage, a different and likely heavier model) remain the open question — the community VRAM figures may still bind there. Treat this as "edge of feasibility for the smallest tier," not "confirmed infeasible everywhere."

- **GPT-SoVITS integration shape — confirmed, no longer open.** Subprocess for training (matches what `webui.py` itself does), direct Python import (`TTS_infer_pack.TTS`) for inference. See §3.
- **Does GPT-SoVITS expose LoRA/adapter/PEFT tuning natively — confirmed, no longer open.** Only for v3/v4's SoVITS stage, never for the GPT stage, never for v2-family. See §3. This resolves the question but doesn't change the VRAM finding above — LoRA's 8GB floor is still ~2x this GPU's budget.
- **v2ProPlus vs v3 vs v4 checkpoint choice** — now secondary to the VRAM finding. If the path forward is inference-only, this becomes a pure quality/robustness question again (noise-robustness still favors v3/v4's BigVGAN-family vocoder, per RESEARCH.md Finding 5) with no training-VRAM downside to weigh against it, since nothing trains locally either way.
- **Vendoring GPT-SoVITS itself**: `scripts/vendor_gpt_sovits.{sh,ps1}` clone the repo pinned to commit `48b1a0169a28582a8984402f82cf438d3bfa6aca` (the commit our adapter's script names/args were confirmed against) and run GPT-SoVITS's own installer (`install.sh --device CU126 --source HF`), which downloads ~5.2GB of pretrained checkpoints and installs GPT-SoVITS's own Python deps (pytorch_lightning, peft, funasr, etc.) into whatever environment is active. This is a slice of what was planned for Batch 6, pulled forward out of necessity - nothing can really train without it. Real pretrained-checkpoint paths per version (confirmed against `config.py`'s own lookup dicts) are encoded in `core/train/gpt_sovits_adapter.py`'s `default_paths()`. **Open risk**: running GPT-SoVITS's installer in the same env as our own torch 2.14.1+cu126 could touch/conflict with it - not yet observed for real, flagged to watch for when the vendoring script is actually run.
- **GPT-SoVITS's own data-prep pipeline** (`GPT_SoVITS/prepare_datasets/1-get-text.py` → `2-get-hubert-wav32k.py` → `2-get-sv.py` [v2Pro/v2ProPlus only] → `3-get-semantic.py`) must run before either training stage - confirmed and implemented in `core/train/data_prep.py`, env-var interface verified directly against `webui.py`'s caller code rather than guessed. None of the 4 scripts hard-require a GPU (all fall back to CPU automatically, just slower) - meaning data prep itself may succeed on this hardware even where the actual fine-tune stage runs out of VRAM, which is useful diagnostic signal either way.
- **Real-run fix (2026-10-06):** all 4 prep scripts need `PYTHONPATH` set to include both the vendored repo root and `GPT_SoVITS/` itself - `1-get-text.py` does no `sys.path` setup of its own at all, and `webui.py`'s own `Popen` call doesn't set this either (empirically verified with a disposable probe script against the real repo, not guessed). `data_prep.py`'s `_build_prep_env` handles this now.
- **Real-run fix (2026-10-06):** `build_s1_config`/`build_s2_config` now load GPT-SoVITS's own real template config (`s1longer.yaml`/`s1longer-v2.yaml`, `s2.json`/`s2v2Pro.json`/`s2v2ProPlus.json` - confirmed verbatim against the repo, v1/v2/v3/v4 all share plain `s2.json`) and overlay only the keys `webui.py` itself overlays, instead of building a config from scratch. The from-scratch version was missing real acoustic hyperparameters (`filter_length` etc.) and crashed `3-get-semantic.py` with `AttributeError: 'HParams' object has no attribute 'filter_length'` on the first real run. Also confirmed: `3-get-semantic.py` specifically gets the **unmodified template path**, not a mutated temp config - `data_prep.py`'s `run_get_semantic` now points there directly via `gpt_sovits_adapter.s2_template_path()`.
- **Confirmed hardware** (2026-10-05, via `check_gpu.py` on the Linux box): NVIDIA GeForce RTX 3050 Laptop GPU, CUDA 12.6, ~3.7GB total VRAM, torch 2.14.1+cu126. VRAM is tight enough that the LoRA/adapter default (§3, RESEARCH.md Finding 3) is likely necessary, not just preferable — worth re-checking against actual memory usage once Batch 3 runs a real fine-tune.
- **Python version**: resolved to 3.10, see Tech Stack table — still worth a glance at `check_gpu.py`/`setup_env.*` output in case your box's default `python3` triggers the version warning.
- **Recipe-tier step counts/timing** — the duration *boundaries* are now research-backed (§8.2), but how many steps/how long each tier actually takes on your specific GPU is still only confirmed once we see real training runs.
- **pywebview system deps on your Linux distro** (GTK/Qt backend) — if missing or the box is headless, `--headless` mode is the real primary path; worth knowing which applies before Batch 5.
- **Citation confidence**: most of RESEARCH.md is peer-reviewed/primary, but the XTTS-v2 duration-ablation thesis and the 2026 RVCBench paper are newer/unreviewed — their specific numbers are directional, not settled; don't hard-code thresholds derived only from those two without a sanity check against real results.

## References

Full literature review (29 papers, organized by finding) is in [RESEARCH.md](RESEARCH.md). Below are the non-academic/project-level sources used directly in this doc.

- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — core model, MIT license.
- [GPT-SoVITS version features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90features-(%E5%90%84%E7%89%88%E6%9C%AC%E7%89%B9%E6%80%A7)) — v1–v4 / Pro / ProPlus differences.
- [GPT-SoVITS v3/v4 features wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v3v4%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)) — v3 metallic-artifact issue and the v4 fix.
