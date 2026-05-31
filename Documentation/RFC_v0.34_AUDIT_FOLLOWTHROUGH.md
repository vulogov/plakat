# RFC: v0.34 — audit follow-through

**Status:** active, drafted 2026-05-31 at phase 0 start.
**Branch:** `0.34.0` (cut from `main` after v0.33.0 merge).

## Goal

Close the gaps v0.33 left behind while the audit table and
metadata-builder context are still fresh. Three of the four
feature phases turn v0.33's "half-shipped" or "surveyed but not
fixed" outputs into "actually useful." Fourth phase clears v0.32
debt.

## Constraints

- **Additive schema.** Every existing PNG tEXt sidecar from v0.33
  must still parse. New fields are `Option`/`Vec` with serde
  `default` + `skip_serializing_if`. Same guarantee v0.33 phase 0
  made about v0.32; v0.34 extends it.
- **Data plumbing only (phase 0).** Pipeline-side population must
  not add new I/O, change generation behaviour, or shift the
  byte-output of `--seed N`. Plumb info the resolution layer
  already knows; don't introduce new resolution work.
- **Determinism fixes are explicit (phase 1).** Where v0.33's
  audit said `?` or `⚠ Metal-u32`, the fix may change pixel
  output for previously-affected runs. That's not a regression —
  it's the fix the audit recommended. Document in migration notes.
- **No new error taxonomy.** v0.33 phase 1 established the
  decorator pattern on anyhow contexts. v0.34 extensions follow
  the same pattern.

## Phase plan

### Phase 0 — Pipeline-side structured stack population

**Scope (locked after survey 2026-05-31):**

Pre-survey assumption: all pipelines (SD t2i, SDXL, SD3, Flux,
AnimateDiff, Stylize, Portrait) have in-pipeline metadata-build
sites that just need wiring. Post-survey reality: **only `t2i.rs`
builds `GenerationMetadata` in-pipeline.** SD3, Flux, AnimateDiff,
img2img, and portrait either emit metadata at the CLI dispatcher
level with no resolved-state hook OR don't emit
`GenerationMetadata` at all.

Adding `GenerationMetadata` to SD3/Flux/AnimateDiff is a behaviour
change (PNG sidecars appearing where none exist today) and a
new feature surface — not the data-plumbing-only work this phase
promised. **Deferred to a separate phase / cycle.**

**In scope:**
- `t2i.rs` SD 1.5 + SDXL paths: populate `lora_stack` from
  `req.loras` and `control_stack` from `req.controls` at the
  metadata-build site. Both types are derivable from spec alone —
  no double-resolution, no I/O cost, no behaviour change.
- New helpers: `LoraSpec::to_entry() -> LoraEntry` in
  `pipelines/lora.rs`, `ControlSpec::to_entry() -> ControlEntry`
  in `pipelines/controlnet.rs`.
- Fallback logic: when `req.lora_stack` / `req.control_stack` is
  `None` AND `req.loras` / `req.controls` is non-empty, derive
  the entries from specs. When the Request already carries an
  explicit stack (e.g. style runtime, scripting), use that
  unchanged.

**Out of scope (deferred):**
- **Embedding stack population.** `EmbeddingEntry` needs
  `embed_dim` / `num_tokens` / `dual_encoder` — only available
  after loading the safetensors. Getting them at the build site
  means either double-loading the TI files (~5-50 KB each — small
  but non-zero behaviour change) or extending `Pipeline::load`
  to expose resolved embedding info (architectural — a different
  shape of change than "data plumbing only"). Deferred to a
  later phase in this cycle if time allows, otherwise v0.35.
- **Other pipelines** (SD3, Flux, AnimateDiff, Stylize, Portrait).
  Need `GenerationMetadata`-emitting paths added first — that's
  a separate effort.

**Acceptance:**
- `plakat generate "..." --lora civitai:12345:0.7 --controlnet canny ./edges.png`
  produces a PNG sidecar where `lora_stack` carries
  `{display: "civitai:12345", scale: 0.7, source: "civitai", ...}`
  and `control_stack` carries
  `{kind: "canny", image: "./edges.png", strength: 1.0, ...}`.
- All existing tests pass — byte-identity on `--seed N` preserved.
- New unit tests cover spec→entry helpers per source kind
  (Local / Hub / Civitai for LoRA; image / from / video for CN).
- `v032_sidecar_still_parses` and `v033_sidecar_still_parses`
  tests stay green.

### Phase 1 — Determinism fixes (act on v0.33 phase 3 audit)

Two concrete fixes:

- **VAE encode `set_seed()` placement** in img2img / stylize
  NEEDS-VERIFICATION rows. Read each path, decide whether
  `set_seed()` runs before or after VAE encode. Either fix to
  call `set_seed()` immediately before any RNG-touching step,
  or document why the current order is correct. Goal: flip both
  `?` rows in the audit to `✓`.
- **Metal full-width seed.** Currently `as u32` truncation in
  the Metal seed path — seeds above 2^32 collide. Hash
  `u64 → (u32, u32)`, mix both into the Metal RNG. Goal: flip
  the 8 `⚠ Metal-u32` audit rows to `✓`. Re-run
  `plakat doctor --reproducibility-check` for verification.

**Migration impact:**
- `--seed N` with `N < 2^32`: byte-identical output (fits in u32).
- `--seed N` with `N >= 2^32`: previously aliased to
  `N mod 2^32`, now distinct. This is the fix, not a regression.
- img2img / stylize `--seed N --strength 0.7`: may change if
  audit reveals `set_seed()` was in the wrong place. Documented
  as a determinism fix in migration notes.

### Phase 2 — Per-task failure text in `--json-summary`

Extend `TaskRunRecord` with `error: Option<String>`. Wrap task-
dispatch sites with catch-and-record so failed records carry
`e.to_string()`. Acceptance: scenarios with one bad task show
`"status": "failed", "error": "..."` in the JSON summary.

### Phase 3 — v0.32 carry closures (3 items, no stretch)

1. **Animate-side VAE cache passthrough.**
   `AnimateDiffPipeline::load` and `AnimateDiffSdxlPipeline::load`
   accept `Option<Arc<AutoEncoderKL>>`. Mixed-kind scenarios stop
   paying the ~330 MB rebuild cost on every t2i ↔ animate kind
   switch.
2. **Scripting `plakat.load` Bund word — VAE cache passthrough.**
   Accept cache param from scenario runner's cache map.
3. **Auto1111 two-separate-files SDXL TI convention.** Parser
   accepts the two-file split format and stitches them into the
   v0.31 phase 0 dual-encoder TI path.

### Phase 4 — Cycle close-out

Standard 7-step release: README, RELEASE_HISTORY, attribution
scan, tag, cargo publish, merge, open v0.35 dev cycle, memory.

## Risks

- **Phase 0 scope contraction surprise.** Survey revealed phase 0
  is smaller than estimated because only t2i.rs has the build
  site. Documented above — this RFC is the lock.
- **Phase 0 embedding gap.** Acceptance criterion adjusted from
  "LoRA + embedding + CN" to "LoRA + CN" because embeddings need
  resolution. Honest trade-off recorded.
- **Phase 1 VAE encode placement could surface a real bug.** If
  `set_seed()` runs after VAE encode in img2img, fixing it
  changes existing `--seed N --strength 0.7` output. Mitigation:
  document the change in migration notes; accept the numerical
  break (correctness > byte-compat for determinism).
- **Phase 1 Metal seed change is API-visible.** See migration
  impact above. Existing in-range seeds unchanged; previously-
  collided high seeds now distinct.
- **Phase 3 Auto1111 two-files TI format variations.** Civitai
  publishes TIs in inconsistent formats. Mitigation: target the
  most common A1111 export shape; reject + suggest fix for
  others (per v0.33 phase 1 error hint pattern).

## What's NOT in v0.34

Deferred to v0.35+:
- Per-layer motion splice (RFC v0.27 §3.2 escalation).
- HotShot-XL integration.
- AnimateLCM-SDXL (externally blocked).
- INT8 SDXL UNet (blocked on candle quantized Conv2d).
- Plakat server mode.
- PixArt Sigma / Stable Cascade.
- **Embedding stack population** (phase 0 deferral; needs
  resolution-layer extension).
- **`GenerationMetadata` for SD3 / Flux / AnimateDiff** (phase 0
  deferral; pipelines need metadata-emitting paths added first).
- v0.33 phase 0 dead-code warning on `ReproGuarantee::label`
  (decide: wire into human render or drop).
