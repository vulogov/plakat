# RFC v0.31 — Diversify-2 + INT8 headline

**Status:** decisions locked 2026-05-30 — phase 0 in flight.

**Predecessors:**
- [`RFC_v0.30_DIVERSIFY.md`](RFC_v0.30_DIVERSIFY.md) — embedding TI runtime via vendored CLIP, LCM-LoRA t2i, per-frame video CN, doctor enrichment.
- [`RFC_v0.29_BATCH_PRODUCTIVITY.md`](RFC_v0.29_BATCH_PRODUCTIVITY.md) — animate in scenarios, mixed-kind cache carry.

## 1. TL;DR

v0.30 just shipped a diversify cycle that closed the longest-
running open carry (TI runtime, v0.16) plus the biggest animate
carry (video CN). v0.31 continues diversifying with the
**single highest-impact non-animate item left**: INT8 SDXL UNet
quantization. Puts SDXL on 12 GB consumer GPUs — a dramatic
widening of plakat's effective user base.

Around the INT8 headline, four supporting items:

1. **SDXL dual-encoder TI parser** — closes the v0.30 phase 0
   stretch goal. CLIP-L-only SDXL TIs already work; the
   `clip_l` + `clip_g` dual-tensor format is the open gap and
   the majority of SDXL TIs on Civitai use it.
2. **INT8 SDXL UNet quantization** — headline.
3. **Better wildcards** (nested `{a|{b|c}}` + weighted
   `{0.7::common|0.3::rare}`) — compositional prompt power.
4. **Mixed-kind scenarios pipeline cache** — v0.29 carry.

Five phases, ~9 sessions. Phase 1 is the gating risk (INT8
codec direction).

## 2. Why this is the v0.31 cycle

1. **INT8 SDXL widens the user base more than any other open
   item.** SDXL at F16 needs ~16 GB UNet peak — outside reach of
   most consumer cards. INT8 quantization targets ~7-8 GB UNet
   peak which fits a 12 GB card with everything else in the
   pipeline. v0.14 NF4 Flux work gives a partial template for
   the codec patterns.

2. **Three animate cycles + a diversify cycle is enough animate
   for now.** v0.30 already shipped per-frame video CN (the most-
   requested carry). The remaining animate items (FreeNoise,
   per-layer splice, HotShot-XL) are quality work that can wait
   one more cycle without users feeling pain.

3. **The SDXL dual TI parser closes a clean v0.30 stretch goal.**
   The merger, vendored CLIP, tokenizer-mutation infrastructure
   are all in place from v0.30 phase 0. Closing dual-TI parser
   support means more SDXL Civitai TIs work end-to-end.

4. **Wildcards and mixed-kind cache pay down backlog.** Wildcards
   are a power-user feature that composes well with v0.25
   `--look`/`--genre` presets. Mixed-kind cache closes the v0.29
   carry that costs ~10 GB on hybrid scenarios.

## 3. Phase plan

### Phase 0 — SDXL dual-encoder TI parser

**Goal:** drop the v0.16 parser bail that rejects `clip_g`. Apply
both `clip_l` and `clip_g` tensors during SDXL load. Shared
trigger token registered against the same new vocab IDs in both
CLIP-L and CLIP-G tokenizers.

**Design constraints:**
- `parse_safetensors` returns either a single-encoder
  `ResolvedEmbedding` (existing) or a new dual variant. Simplest
  shape: extend `ResolvedEmbedding` with an optional `clip_g`
  vectors field. When present, the SDXL load path merges into
  both encoders; when absent, only CLIP-L gets the extension.
- The trigger registration list must agree between CLIP-L and
  CLIP-G — both encoders' vocabs are extended by the same N
  tokens, named `trigger`, `trigger_1`, ..., `trigger_{N-1}`.
- Embedding-dim mismatch detection: CLIP-L tensor must be 768d,
  CLIP-G tensor must be 1280d. Reject early with a clear pointer
  to a CLIP-L-only TI if the user accidentally loads an SD 1.5
  TI against SDXL.
- Non-SDXL pipelines (SD 1.5, SD 2.1) bail when a dual TI is
  passed — there's no CLIP-G to merge into.

**Integration:**
- `pipelines/embedding.rs` — extend parser + add
  `merge_dual_embeddings_into_te_weights` (or extend the
  existing merger to handle both encoders in one pass).
- `pipelines/sd_core.rs` — when `variant == Sdxl` and any
  `req.embeddings[i].has_clip_g()`, merge CLIP-G via the same
  tempfile + vendored CLIP + `Config::with_vocab` pattern v0.30
  phase 0 established for CLIP-L. `tokenizer_g.add_tokens` for
  the same trigger strings.
- The `EmbeddingRegistration` list already produced by the
  merger carries the trigger + base_token_id + num_tokens. For
  dual TI, the same registrations apply to both tokenizers
  (same vocab offsets) because we extend both by the same N
  tokens.

**Acceptance:**
- SDXL + `--embedding PATH` (synthetic dual TI fixture: both
  `clip_l` and `clip_g` keys) produces output without bailing.
- Tokenizer behaviour: typing the trigger in the prompt resolves
  to the new IDs in both encoders.
- No-embedding numerical regression: SDXL output unchanged when
  no `--embedding` flag is passed.
- SD 1.5 + dual TI bails with a pointer to a CLIP-L-only TI.

**Scope:**
- Diffusers convention (single file with `clip_l` + `clip_g`
  top-level keys) is the v0.31 target.
- The older "two separate top-level files" Auto1111 convention
  is out of scope; document.

~2 sessions.

### Phase 1 — INT8 SDXL UNet quantization (gating risk)

**Goal:** SDXL on 12 GB consumer GPUs via INT8 UNet weights.
Target: ~7-8 GB UNet peak (from ~16 GB F16 baseline).

**Approach:**
1. **0.5-session validation spike** at phase start. Confirm
   candle's quantization story supports INT8 on SDXL UNet shapes
   (Conv2d + Linear at the SDXL block sizes). Options:
   - candle's `quantized_*` modules (GGUF Q-types) — used by
     Flux GGUF, may not cover SDXL UNet shapes cleanly.
   - Vendored codec, modeled on `pipelines/nf4_codec.rs` +
     `pipelines/nf4_loader.rs` from v0.14 Flux NF4 work.
   - bitsandbytes-style INT8 quantization (per-channel scales +
     int8 weights) — most user-familiar.
   - Decide direction based on what's reachable in 0.5 sessions.
2. **Codec + loader** (~1.5 sessions): vendor or extend codec,
   write loader that maps GGUF or bnb-style quantized SDXL UNet
   safetensors onto plakat's `SdxlUNet2DConditionModel` field
   layout.
3. **Pipeline integration** (~1 session): expose via a new
   `--quant int8` flag on `plakat generate` (or a model alias
   like `sdxl-int8`). Hard-bail when paired with `--refiner`
   (the refiner UNet would need its own quantized weights;
   defer).
4. **Tests + tutorial** (~0.5 session): synthetic UNet + INT8
   round-trip test; tutorial entry showing the 12 GB workflow.

**Bail plan:**
- If the validation spike shows the codec direction is hostile
  (e.g. no reasonable INT8 path covers Conv2d at all SDXL
  shapes), swap phase 1 for one of the deferred polish items:
  vendored CLIP rollout (~2 sess) or Pony preset + civitai
  sync (~2-3 sess combined). The cycle stays diversified
  rather than sinking on a hostile codec.

**Acceptance:**
- `plakat generate --model sdxl-int8 "a portrait"` succeeds at
  1024² on a 12 GB GPU (or the validation environment).
- Numerical output is within reasonable distance of F16 baseline
  (LPIPS or simple visual side-by-side — quantization always
  drifts; the goal is "still recognisable as the same prompt").

~3-4 sessions on validation pass; 0 sessions on validation fail
(swap to a polish item).

### Phase 2 — Better wildcards (nested + weighted)

**Goal:** compositional prompt power for batch generation.

**Approach:**
- Existing parser (likely in `src/prompt/wildcards.rs` or
  similar — confirm at phase start) handles flat
  `{a|b|c}` alternation.
- Add nested: `{a|{b|c}}` — recursive descent.
- Add weighted: `{0.7::common|0.3::rare}` — alternation with
  explicit probability weights (default 1.0 each, normalized).
- Wildcard RNG seeded from `--seed` for reproducible expansion.

**Acceptance:**
- Round-trip tests on synthetic nested + weighted prompts.
- Tutorial entry showing a batch where weighted wildcards
  produce a 70/30 split over many generations.

~2 sessions.

### Phase 3 — Mixed-kind scenarios pipeline cache

**Goal:** stop carrying both t2i and animate pipelines when a
scenario mixes `type: generate` and `type: animatediff` tasks.

**Approach:**
- Today: scenarios with both kinds hold both pipelines in memory
  simultaneously (~10 GB peak on SD 1.5; worse on SDXL).
- Fix: cache key derived from current task's kind. When the
  loop switches kinds, drop the non-matching cached pipeline
  before loading the new one.
- Mirrors the v0.26 stylize cache slot pattern.

**Acceptance:**
- Memory diff: hybrid scenario runs at single-pipeline peak
  rather than both-pipelines peak.
- Same task outputs as the v0.30 baseline (cache eviction is
  invisible to the user beyond memory).

~1 session.

### Phase 4 — Cycle close-out

Standard 7-step release: README rewrite, RELEASE_HISTORY archive,
attribution scan, tag, cargo publish, merge to main, start v0.32
dev cycle.

~0.5 session.

## 4. Decisions locked

1. **Cycle shape:** diversify-2 + INT8 headline + better
   wildcards (phases 0-3 plus close-out). ~9 sessions when
   phase 1 validation holds.
2. **Polish item:** better wildcards (not Pony preset / civitai
   sync / vendored CLIP rollout). Compositional prompt power is
   higher-leverage of the available items.

## 5. What's NOT in v0.31

Deferred to v0.32+:
- FreeNoise / FreeInit long-form (animate quality).
- Per-layer motion splice (RFC v0.27 §3.2).
- HotShot-XL integration.
- AnimateLCM-SDXL (upstream still not publicly available).
- Pony Diffusion preset.
- `plakat civitai sync DIR` bulk download.
- Vendored CLIP rollout to AnimateDiff/SD3/Flux/stylize.
- SDXL VAE warmup / smart caching.
- Auto1111 two-separate-files SDXL TI convention (only the
  diffusers single-file dual format is in v0.31 scope).

## 6. Related

- v0.30 phase 0 release notes — vendored CLIP +
  embedding TI runtime infrastructure that phase 0 here extends.
- v0.14 NF4 Flux release notes — codec + loader pattern that
  phase 1 may template against.
- v0.26 stylize cache slot pattern — template for phase 3
  mixed-kind cache.
