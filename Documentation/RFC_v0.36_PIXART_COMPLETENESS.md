# RFC: v0.36 — PixArt completeness

**Status:** active, drafted 2026-05-31 at phase 0 start.
**Branch:** `0.36.0` (cut from `main` after v0.35.0 merge).

## Goal

Close the deferrals v0.35 explicitly punted while the DiT / T5 /
LoRA context from v0.35 is fresh. Mirrors v0.34's audit follow-
through after v0.33's audit work: finish the integration so the
new model family is actually usable in plakat's batch +
programmatic flows.

## Constraints

- **Additive schema.** Every existing flag / host word / config
  key / scenario field / PNG sidecar from v0.35 keeps its shape.
- **No new model architectures.** Each phase extends the v0.35
  phase 1 DiT — KV-compression is an addition to the existing
  Block, not a new transformer.
- **Seed plumbing through `pipelines::seeds::prepare_seed`** for
  every new dispatch site (v0.34 phase 1 chokepoint).
- **VAE Arc-cache reuse.** PixArt scenarios sharing alias with
  SDXL t2i scenarios automatically reuse the VAE via v0.34
  phase 3.

## Phase plan

### Phase 0 — Scenario PixArt dispatch

**Scope:**

- New `pixart_pipeline: Option<pixart::Pipeline>` cache slot in
  `scenario.rs`, mirrors the Flux + SD3 pre-load pattern: load
  once at scenario start when `variant.is_pixart()`, otherwise
  `None`. Scenario-level LoRAs resolve + merge at load time
  (the v0.35 phase 4 tempfile-merge path).
- VAE cache priming from the freshly pre-loaded PixArt pipeline
  (so SDXL t2i tasks that run after PixArt in the same scenario
  reuse the VAE).
- `sd_pipeline_applicable` excludes `variant.is_pixart()` so the
  SD-family lazy-reload path doesn't fire on PixArt scenarios.
- `sd_per_task_lora_preflight` skips PixArt (parallels the
  Flux + SD3 skip).
- PixArt dispatch arm inside the generate task body, inserted
  before the SD3 arm (`if let Some(pp) = pixart_pipeline.as_mut()`).
  Per-task loop honours `eff_count` by stepping the seed per
  image, calls `pp.generate(...)`, builds `GenerationMetadata`
  with `lora_stack` populated, writes via
  `save_rgb_u8_with_metadata`.

**Out of scope (deferred to v0.36 phase 2/3 or v0.37):**

- Per-task PixArt LoRA overrides. PixArt has no runtime per-task
  LoRA swap (merge-at-load); per-task overrides require a per-
  task reload. Phase 0 ships scenario-level LoRAs only.
- Tiled, img2img, ControlNet on PixArt — none implemented for
  PixArt today; bail-loud paths land alongside the v0.37 PixArt
  ControlNet / portrait work.

**Acceptance:**
- Scenario file with mixed SDXL t2i + PixArt tasks compiles +
  loads cleanly.
- VAE cache HIT log appears on the kind switch.
- Existing scenario regression tests stay green.
- `preflight_pixart_model_skips` test locks the preflight
  extension.

### Phase 1 — Scripting `plakat.pixart` Bund word

Mirror `plakat.load` + `plakat.animate` structure.
`ScriptCtx.loaded_pixart` slot. VAE cache shared via
`ScriptCtx.vae_cache` (already supports PixArt aliases per v0.35
phase 0).

**Acceptance:** `plakat.pixart "..." 1024 1024` from a bund
script produces an image; cache HIT log on the second call with
the same alias.

### Phase 2 — PixArt-Σ-XL-2-512-MS variant

`pixart-512` alias → `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`.
Variant detection resolves to a `DitConfig` with `sample_size:
32` (vs 64 for 1024-MS). All other architecture identical.

**Acceptance:** `--model pixart-512` produces 512² output.

### Phase 3 — KV-compression + 2K-MS variant (LOCKED)

Extends `pipelines::pixart_dit` with optional KV-compression on
self-attention. Adds
`kv_compression: Option<KvCompressionConfig>` to the `Config`,
applied per-layer based on the upstream config. Conv1d on K and
V to downsample the image-token sequence (typically 2× factor).

`pixart-2k` alias → `PixArt-alpha/PixArt-Sigma-XL-2-2K-MS` (4K
latent at 2048² output).

**Risks:** Tensor naming (history) + F16 numerical drift on the
compression Conv1d. Mitigations per v0.35 phase 1 (explicit key
walk, shape tests; fp32 fallback for the compression step).

### Phase 4 — LCM variants

Survey upstream Σ-LCM availability; either native checkpoint
integration or compose with the v0.30 phase 1 LCM-LoRA t2i
mechanism through v0.35 phase 4's PEFT LoRA path.

### Phase 5 — Cycle close-out

Standard 7-step release.

## What's NOT in v0.36

Explicitly deferred to v0.37+:

- **PixArt ControlNet integration** — larger feature; own phase.
- **PixArt portrait / face-preservation** — bigger lift.
- **PixArt per-task runtime LoRA swap in scenarios** — phase 0
  ships scenario-level LoRAs; per-task overrides need a per-
  task reload (expensive) or runtime swap (PixArt has none).
  Either lands alongside ControlNet/portrait or its own polish
  phase.
- **PixArt tiled / img2img** — not implemented today.
- **T5 loader extraction** across SD3 + Flux + PixArt — cleanup
  cycle.
- **Metadata completion** for SD3 / Flux / AnimateDiff / stylize
  / portrait — v0.34 carry.
- **Plakat server mode.**
- All long-standing deferrals (per-layer motion splice,
  HotShot-XL, AnimateLCM-SDXL externally blocked, INT8 SDXL
  externally blocked, Stable Cascade).
