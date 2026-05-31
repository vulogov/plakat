# RFC: v0.35 — PixArt Sigma (diversify-4)

**Status:** active, drafted 2026-05-31 at phase 0 start.
**Branch:** `0.35.0` (cut from `main` after v0.34.0 merge).

## Goal

Add **PixArt Sigma** as plakat's fourth model family. Breaks the
two-cycle polish chain (v0.33 + v0.34) by going back to diversify.
PixArt-Σ is the most practical externally-available new family —
distilled variants fit 12 GB VRAM at 1024², partial infrastructure
reuse with SD3 + Flux via the shared T5-XXL text encoder, well-
supported LoRA ecosystem on Civitai.

## Architecture

PixArt-Σ is a **Diffusion Transformer (DiT)**:

- **DiT-XL/2 backbone** (~600M params). PixArt-Σ adds KV-
  compression (sparse cross-attention to the T5 sequence) on top
  of PixArt-α. adaLN-zero modulation. **Phase 1.**
- **T5-XXL text encoder** (~4.7B params). Same
  `candle_transformers::models::t5::T5EncoderModel` SD3 and Flux
  use today. Sharded as 3 files in the canonical Sigma
  checkpoint. **Phase 0.**
- **SD-family KL-VAE** (~330 MB). 8× downsample, 4 latent
  channels — same shape as SDXL VAE. Reused via the v0.34 phase 3
  Arc-cache mechanism. **Phase 0.**
- **DPM++ sampler** (PixArt-Σ's published recommendation).
  Phase 2.
- **PixArt LoRA** (diffusers format, distinct from kohya-ss /
  SD-family). Phase 4.

## Constraints

- **First variant ships in phase 2: `PixArt-Σ-XL-2-1024-MS`**
  (canonical Sigma, fits 12 GB VRAM, matches SDXL training
  resolution). 512-MS + 2K-MS deferred to v0.36 unless trivially
  reachable via alias resolution.
- **No new `SdVariant` arm.** PixArt is not an SD-family
  architecture (DiT, not UNet). It gets its own `Variant::PixArt`
  on `t2i::Variant` + its own pipeline module. `SdCore` is not
  involved.
- **T5 is NOT extracted to a shared module in v0.35.** PixArt
  duplicates the sd3.rs / flux.rs T5 load pattern. The extraction
  is a separate cleanup; doing it inside this cycle would expand
  scope. Tracked as a v0.36+ deferral.
- **LoRA is a LOCKED phase 4**, not stretch.
- **Seed plumbing**: PixArt routes through
  `pipelines::seeds::prepare_seed` (the v0.34 phase 1 chokepoint)
  from day one — phase 2 must wire this when adding `set_seed`
  calls.
- **VAE cache**: PixArt's `LoadRequest.vae_cache` field is
  populated in phase 0; the scenario + scripting cache wiring
  comes when phase 2 lands inference and scenarios can actually
  invoke PixArt.

## Phase plan

### Phase 0 — T5 wiring + alias registration + dispatch stub

**Scope (locked at phase 0 start):**

- Add `pixart`, `pixart-sigma`, `pixart-1024` aliases to
  `hf::ALIAS_TABLE`. `all_known_aliases()` picks them up
  automatically (v0.33 phase 1 error-hint mechanism).
- Add `Variant::PixArt` to `t2i::Variant` + detection +
  `is_pixart()` helper.
- Add `BaseFamily::PixArt` to `preset::discovery` +
  `BaseModel::PixArt` to `style::catalog` so the exhaustive
  matches in the discovery + catalog modules stay sound.
- New module `src/pipelines/pixart.rs` exporting `Pipeline`,
  `LoadRequest`, `RunRequest`, `Pipeline::load`, and `run`.
  `Pipeline::load` actually downloads + loads T5 + VAE
  (no DiT — phase 1 work). `run` calls `Pipeline::load` then
  bails with a clear "phase 1 not yet implemented" message —
  proves the full dispatch path.
- `t2i::Pipeline::load` bails on PixArt with a pointer at
  `pipelines::pixart::Pipeline::load` (parallels the Flux + SD3
  bail pattern).
- `t2i::run` routes PixArt to `pixart::run`.

**Acceptance:**
- `crate::hf::resolve_alias("pixart")` returns
  `"PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"`.
- `Variant::detect("pixart-sigma")` returns `Variant::PixArt`.
- The pipeline module compiles + exports `Pipeline::load`.
- `cargo test --lib` passes (existing 1099 baseline + new alias
  + dispatch tests).
- Smoke: `plakat generate "..." --model pixart` fails with the
  intended phase 1 message AFTER T5 + VAE successfully load
  (proves the foundation works end-to-end).

**Risk:** PixArt repo file layout assumptions (3-shard T5,
`tokenizer/tokenizer.json`, `vae/diffusion_pytorch_model.safetensors`).
Mitigation: the layout is the diffusers default; the canonical
Sigma checkpoint follows it. Any mismatch surfaces clearly at
first `Pipeline::load` invocation as an HF 404.

### Phase 1 — DiT-XL/2 backbone in candle

Implement DiT-XL/2 with PixArt-Σ's KV-compression. Multi-head
attention + adaLN-zero modulation. ~600M params. Tensor naming
must match the upstream safetensors weights — load-bearing.

**Acceptance:**
- Forward-pass shape-test at 1024² produces `(1, 4, 128, 128)`
  latent output for the DiT.
- Numerical sanity check against a fixed-seed forward at lower
  resolution (e.g. 256²) if a reference is available.
- v0.27 phase 2 motion_module.rs tensor-naming history applies:
  walk the safetensors keys explicitly + write the mapping.

**Risk:** Tensor naming. Mitigation: explicit safetensors key
walk + per-layer shape tests before higher integration. F16
numerical drift on DiT + adaLN — borrow the fp32-layernorm
pattern SD3 + Flux already use.

### Phase 2 — Pipeline orchestration

Denoising loop + scheduler integration + VAE decode. Default
scheduler: DPM++ (Sigma's recommended). `device.set_seed` routes
through `pipelines::seeds::prepare_seed`.

**Acceptance:**
- `plakat generate "a misty forest at dawn" --model pixart`
  produces a valid 1024² PNG end-to-end.
- Output passes a sanity check (non-NaN, reasonable pixel range).
- v0.34 phase 1 determinism plumbing carries over: PixArt earns
  a ✓ row in `plakat doctor --reproducibility-check`.

### Phase 3 — CLI integration + presets + doctor

- `plakat doctor` lists PixArt in the model family section.
- `plakat doctor --reproducibility-check` includes a PixArt row.
- v0.25 look/genre preset routing extends to `BaseFamily::PixArt`
  — at least 3 of the existing look presets render cleanly on
  PixArt as an acceptance threshold.
- Per-task scenario VAE cache wires (scenario.rs +
  scripting/ctx.rs cache-lookup sites).

**Acceptance:** Scenario with mixed t2i + PixArt tasks against
the SDXL+PixArt model alias reuses the VAE across kinds (log
line confirms).

### Phase 4 — PixArt LoRA support (LOCKED, not stretch)

Diffusers-format LoRA parser. Civitai PixArt LoRA ingestion
(extends `BaseFamily::PixArt::civitai_matches`). Show resolved
LoRAs in the v0.34 phase 0 `lora_stack` sidecar field.

**Acceptance:** `plakat generate "..." --model pixart --lora
civitai:NNNNN:0.7` loads a PixArt LoRA from Civitai, applies at
the configured scale, and shows up in the PNG sidecar's
`lora_stack`.

### Phase 5 — Cycle close-out

Standard 7-step release: README, RELEASE_HISTORY, attribution
scan, tag, cargo publish, merge, open v0.36 dev cycle, memory.

## Risks

- **Phase 1 DiT tensor naming.** Highest-failure-probability
  step. Mitigation: explicit safetensors key walk + per-layer
  shape tests.
- **Phase 2 numerical drift on DiT + adaLN.** Borrow the fp32-
  layernorm pattern from SD3 + Flux.
- **Phase 4 PixArt LoRA format variation.** Sigma LoRAs from
  different trainers may use different keys. Mitigation: ship
  Civitai-canonical format first; surface clear errors for
  unknowns (per v0.33 phase 1 error hint pattern).
- **T5 weight duplication.** PixArt + SD3 + Flux each have their
  own T5 load code. Extraction deferred to v0.36+.

## What's NOT in v0.35

Deferred to v0.36+:
- PixArt-Σ-XL-2-512-MS + 2K-MS variants.
- PixArt LCM variants (2-step generation).
- PixArt ControlNet integration.
- PixArt portrait / face-preservation integration.
- AnimateDiff-style motion adapter for PixArt (none published
  upstream).
- **T5 loader extraction** to a shared module (SD3 + Flux + PixArt
  all duplicate today; cleanup is a separate cycle).
- v0.34 carries: `GenerationMetadata` for non-t2i pipelines,
  embedding-stack population, server mode.
- Per-layer motion splice, HotShot-XL.
- AnimateLCM-SDXL (externally blocked).
- INT8 SDXL UNet (blocked on candle quantized Conv2d).
- Stable Cascade.
