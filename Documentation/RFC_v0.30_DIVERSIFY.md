# RFC v0.30 — Diversify + one animate theme

**Status:** decisions locked 2026-05-29 — phase 0 in flight.

**Predecessors:**
- [`RFC_v0.29_BATCH_PRODUCTIVITY.md`](RFC_v0.29_BATCH_PRODUCTIVITY.md) — animate in scenarios + animate_format + SDXL `plakat.animate`.
- [`RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md`](RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md) — multi-CN + AnimateLCM + Bund `plakat.animate`.
- [`RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md`](RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md) — feature completeness across SD 1.5 + SDXL.

## 1. TL;DR

After three consecutive AnimateDiff cycles (v0.27 scope, v0.28 single-
script polish, v0.29 batch polish), v0.30 diversifies. The animate
surface is mature enough to share oxygen with non-animate work, and
two carries have grown long enough to deserve attention:

1. **Embedding (Textual Inversion) runtime injection.** The v0.16
   phase 9 carry — eight cycles old. The parser + merger + inspector
   have shipped since v0.16, but **runtime injection** has been
   blocked by candle's private `clip::Config.vocab_size`. v0.30 closes
   it via a small vendored CLIP fork.

2. **LCM-LoRA in t2i.** v0.28 wired AnimateLCM for animate; Civitai
   LCM-LoRAs for single-image generation still don't trigger 4-step
   inference. Reuses the v0.28 LCM scheduler wiring on the t2i path.

3. **Per-frame video ControlNet.** The headline animate carry on
   every deferral list since v0.27. `--control-video PATH` reads a
   video, per-frame annotates, builds per-frame CN residuals,
   composes with sliding-window long-form. SD 1.5 + SDXL.

4. **`plakat doctor` enrichment.** Model probe (cached weights pass
   end-to-end?), GPU memory estimate, ffmpeg presence, API keys
   probe. Universal triage utility.

Six phases, ~10-11 sessions total. Embedding TI is the gating risk
(phase 0) — surface first, replace with a polish item if the vendored
CLIP direction doesn't hold.

## 2. Why this is the v0.30 cycle

1. **Three AnimateDiff cycles is enough for now.** v0.27/v0.28/v0.29
   shipped a complete animate surface (scope → script UX → batch UX).
   The remaining animate items are quality work (FreeNoise, per-layer
   splice) that can wait without users feeling pain. Meanwhile non-
   animate plakat has accumulated carries.

2. **Embedding TI is the longest-running open carry.** v0.16 phase 9
   (mid-2026-04) shipped the parser + `plakat embedding info`
   inspector but bailed at load time with a `--embedding` flag that
   refused to run. Eight cycles later, Civitai still hosts thousands
   of TI files that plakat can't apply. Closing this carry is the
   single biggest "make the existing surface deliver on its promise"
   item left.

3. **LCM-LoRA in t2i unlocks the v0.28 LCM win for the bigger user
   base.** v0.28 wired AnimateLCM for AnimateDiff. The same scheduler
   + 4-step inference works for static t2i with a different LoRA
   (LCM-LoRA, not AnimateLCM). Cleanly composable; ~2 sessions.

4. **Per-frame video CN is the headline animate carry.** It's the
   most-requested feature still on any animate deferral list. Picking
   it up in a diversify cycle (rather than waiting for a full animate
   quality cycle) lets it ship sooner.

5. **`plakat doctor` pays for itself in triage.** Every user report
   currently requires us to ask "is ffmpeg installed?", "which models
   are cached?", "do you have an HF token?". One subcommand answers
   all three.

## 3. Phase plan

### Phase 0 — Embedding TI runtime injection (gating risk)

**Goal:** close the v0.16 phase 9 carry. SD 1.5 + SD 2.1 first;
SDXL within the same phase if the pattern lands cleanly.

**Approach:** vendor a minimal CLIP text transformer in
`src/pipelines/vendored_clip.rs` mirroring candle's
`stable_diffusion::clip` module (~430 LOC). Make `vocab_size` public
on the vendored `Config` (the single blocker). Keep the forward pass
bit-identical to candle's so the no-embedding path is unaffected
numerically.

The parser, merger, and registration logic from v0.16 phase 9
(`src/pipelines/embedding.rs`) already produce:
- An extended `token_embedding.weight` matrix written to a tempfile.
- A `MergeReport` with `new_vocab_size` and per-embedding token
  registration (trigger string, base token ID, num tokens).

What's missing is the **runtime wiring**:
1. Build the vendored CLIP transformer with `vocab_size = new_vocab_size`.
2. Load the extended safetensors via VarBuilder.
3. Inject the new trigger tokens into the tokenizer via
   `Tokenizer::add_tokens`.
4. Drop the bail at `sd_core.rs:243-253` and route the embedding
   stack through this path.

**Vendored CLIP design constraints:**
- Exact tensor key naming match (`text_model.embeddings.token_embedding.weight`,
  `text_model.encoder.layers.{i}.self_attn.{k,v,q,out}_proj.{weight,bias}`, etc.).
- Public `Config` with all current fields visible (`vocab_size`,
  `embed_dim`, etc.). Keep the existing public constructors (`v1_5`,
  `v2_1`, `sdxl`, `sdxl2`) with the same numerics. Add
  `Config::with_vocab(base: Self, vocab_size: usize) -> Self` for the
  embedding override path.
- Same public API surface: `new(vs, c)`, `forward_with_mask`,
  `forward_until_encoder_layer`, `impl Module`.
- Surgically small: text encoder only. No image encoder, no
  projection head beyond what's already inline.

**Integration points (phase 0):**
- `src/pipelines/sd_core.rs`: change `text_encoder_l` field type to
  the vendored `ClipTextTransformer`. When `req.embeddings.is_empty()`,
  build with base vocab + base safetensors (no tempfile, no overhead).
  When non-empty, parse + merge into tempfile + build with extended
  vocab. Mutate `tokenizer_l` with the registered token strings via
  `Tokenizer::add_tokens`.
- `src/pipelines/sdxl_clip.rs::SdxlClipGTextTransformer`: change inner
  type to vendored `ClipTextTransformer`. Same wrapper logic, same
  EOT-pooling. Optional CLIP-G vocab override for SDXL TI.

**Out of phase 0 scope:**
- AnimateDiff text encoder: stays on candle's CLIP (animate doesn't
  expose `--embedding` through the pipeline path today).
- SD3, Flux, stylize: same — keep candle's CLIP. Migration can happen
  in later cycles if there's appetite.

**Acceptance:**
- SD 1.5 + `--embedding PATH` (synthetic A1111-format TI fixture)
  produces output without bailing.
- SDXL + `--embedding PATH` (CLIP-L-only TI, single-encoder format)
  produces output. Dual `clip_l` + `clip_g` SDXL TI parser support is
  a phase 0 stretch goal; deferred to phase 1 of v0.31 if it slips.
- No-embedding numerical regression test (vendored vs. candle CLIP
  output on a fixed prompt + seed).

**Risk:** if upstream candle changes its CLIP internals in a way our
fork would have to track (new optimizations, attention impl swaps),
maintenance grows. Mitigation: keep the fork surgically small and
documented; revisit if candle ever makes `vocab_size` public.

~3 sessions.

### Phase 1 — LCM-LoRA in t2i

**Goal:** detect LCM-LoRA in the LoRA stack, force-set scheduler to
LCM + 4 steps + CFG=1.0 for the t2i path. SD 1.5 + SDXL.

**Approach:** v0.28 added the LCM scheduler for AnimateLCM. Reuse it.
Detection heuristic: LoRA filename or metadata contains "lcm" (or
explicit `--lcm` override). When detected, override scheduler choice
and step count with a `tracing::info!` notification.

**Acceptance:**
- `plakat generate --lora <civitai_lcm_sd15_lora> --steps 4`
  produces output (and ideally output coherent at 4 steps).
- Same for SDXL.
- Tutorial entry showing the 10× speedup.

~2 sessions.

### Phase 2 — Per-frame video ControlNet (SD 1.5)

**Goal:** `--control-video PATH` reads a video, per-frame extracts to
PNG, runs the existing annotator per frame, builds per-frame CN
residuals. Composes with sliding-window long-form.

**Approach:**
- Extend `imaging::video` with input decode: ffmpeg subprocess to
  extract frames to a tempdir as PNGs at the requested frame rate.
- New CLI flag `--control-video PATH` on `plakat animate`.
- New `ControlSource::Video(PathBuf)` variant alongside the existing
  static-image source.
- Per-frame CN residual computation in the animate loop.

**Acceptance:**
- A 16-frame stock video + canny CN + a stylization prompt produces
  16 stylized frames that follow the video's structure.
- Works with sliding-window long-form (frame count > 16).

~3 sessions.

### Phase 3 — Per-frame video CN (SDXL adaptation)

**Goal:** mirror phase 2 onto `AnimateDiffSdxlPipeline`. Same flag,
same loop pattern.

**Approach:** small follow-on once phase 2's video-decode infra +
per-frame CN injection pattern is established.

**Acceptance:** SDXL animate + `--control-video` produces output.

~1 session.

### Phase 4 — `plakat doctor` enrichment

**Goal:** a single subcommand that surfaces "what's installed, what
works, what's missing" in one view.

**Approach:**
- Model probe: enumerate cached HF models, attempt a 1-step inference
  on each, report pass/fail.
- GPU memory estimate: query device, report total + free.
- ffmpeg presence + version (probe `ffmpeg -version`).
- API keys probe: HF token (HF_TOKEN env or saved login), Civitai
  token (CIVITAI_API_KEY env). Report present/absent (NEVER print
  values).
- Output mode: human-readable by default, `--json` for machine
  consumption.

**Acceptance:**
- Runs end-to-end on a working install, reports all-green.
- Runs on a broken install (no ffmpeg, no models), reports failures
  with actionable next steps.

~1 session.

### Phase 5 — Cycle close-out

Standard 7-step release: README rewrite, RELEASE_HISTORY archive,
attribution scan, tag, cargo publish, merge to main, start v0.31 dev
cycle.

~0.5 session.

## 4. Decisions locked

1. **Cycle shape:** diversify + one animate theme (B + D + A1, in
   v0.30 brainstorm terms). Embedding TI + LCM-LoRA t2i + per-frame
   video CN + `plakat doctor`. ~10-11 sessions.
2. **Embedding TI risk posture:** commit to vendored CLIP fork. Vendor
   minimal text encoder with public `vocab_size`. Maintenance cost
   accepted; revisit if candle exposes `vocab_size` publicly.

## 5. What's NOT in v0.30

Deferred to v0.31+:
- FreeNoise / FreeInit long-form (animate quality).
- Per-layer motion splice (RFC v0.27 §3.2 escalation).
- HotShot-XL integration.
- AnimateLCM-SDXL (upstream repo not publicly available).
- Mixed-kind scenarios share pipeline cache (v0.29 carry).
- Better wildcards (nested + weighted).
- Pony Diffusion preset.
- INT8 SDXL UNet quantization.
- `plakat civitai sync DIR`.
- Migrating AnimateDiff / SD3 / Flux / stylize off candle's CLIP onto
  the vendored CLIP. Only sd_core uses vendored in v0.30.

## 6. Related

- [`MEMORY.md`](../.. -relative) — cycle index.
- v0.16 phase 9 release notes in `RELEASE_HISTORY.md` — origin of the
  embedding runtime carry.
- v0.28 AnimateLCM scheduler wiring — reused by phase 1 (LCM-LoRA t2i).
- v0.28 `imaging::video` output-side ffmpeg wrapper — extended by
  phase 2 (input-side decode).
