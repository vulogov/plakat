# RFC: v0.37 — Stable Cascade (diversify-5)

**Status:** active, drafted 2026-05-31 at phase 0 start.
**Branch:** `0.37.0` (cut from `main` after v0.36.0 merge).

## Goal

plakat's **fifth model family** — Stable Cascade. A 3-stage
architecture distinct from every existing family: not a single
UNet (SD), not DiT (PixArt), not MMDiT (SD3), not Flux DiT. Three
coupled models chain at inference: text → Stage C → Stage B →
Stage A → image.

After v0.35 + v0.36's PixArt sequence (diversify then polish),
v0.37 alternates back to diversify. Stable Cascade is the only
credibly-implementable new family that hasn't been touched.

## Constraints

- **Additive schema.** Every existing flag / host word / config
  key / scenario field / PNG sidecar from v0.36 keeps its shape.
- **CLIP-G reuse.** Stable Cascade's text encoder is CLIP-G —
  the same vendored CLIP plakat already ships for SDXL CLIP-G
  (v0.30 phase 0 vendoring, v0.32 phase 1 rollout). Zero new
  text-encoder code needed.
- **VAE Arc-cache compatibility.** Stage A is a custom small VAE
  (~3.6M params, not the SD-family `AutoEncoderKL`) — it doesn't
  fit the existing Arc cache mechanism. v0.37 introduces it as
  its own type; v0.38+ may extend cache compatibility.
- **Seed plumbing through `pipelines::seeds::prepare_seed`** for
  every new dispatch site (v0.34 phase 1 chokepoint).

## Phase plan

### Phase 0 — Foundation: aliases, variant, Pipeline stub, CLIP-G

**Scope (locked at phase 0 start):**

- Add aliases `stable-cascade` + `cascade` →
  `stabilityai/stable-cascade` in `hf::ALIAS_TABLE`. The Lite
  variant alias lands in phase 2/3 alongside the variant routing.
- Add `Variant::StableCascade` to `t2i::Variant` + detection +
  `is_cascade()` helper.
- Add `BaseFamily::StableCascade` to `preset::discovery` (+
  `from_model_arg`, `cache_slug`, `civitai_matches`,
  `hf_repo_matches_base` arms).
- Add `BaseModel::StableCascade` to `style::catalog` (+
  `from_variant`, `slug` arms).
- New module `src/pipelines/cascade.rs` exporting `Pipeline`,
  `LoadRequest`, `RunRequest`, `Pipeline::load`, and `run`.
  `Pipeline::load` actually downloads + loads CLIP-G + tokenizer.
  `run` calls `Pipeline::load` then bails with a clear "phase 1
  not yet implemented" message — proves the full dispatch path.
- `t2i::Pipeline::load` bails on Cascade with pointer at
  `pipelines::cascade::Pipeline::load` (parallels Flux + SD3 +
  PixArt bail pattern).
- `t2i::run` routes Stable Cascade to `cascade::run` before
  PixArt / SD3 / Flux / SD dispatch.

**Acceptance:**
- `crate::hf::resolve_alias("stable-cascade")` returns
  `"stabilityai/stable-cascade"`.
- `Variant::detect("cascade")` returns `Variant::StableCascade`.
- The pipeline module compiles + exports `Pipeline::load`.
- `cargo test --lib` passes (existing 1141 baseline + new tests).
- Smoke: `plakat generate "..." --model stable-cascade` fails
  with the intended phase 1 message AFTER CLIP-G successfully
  loads (proves the foundation works end-to-end).

**Risk:** Stable Cascade repo file layout assumptions
(`text_encoder/model.safetensors` for CLIP-G, `tokenizer/
tokenizer.json`). Mitigation: the layout is the diffusers
default; the canonical `stabilityai/stable-cascade` checkpoint
follows it.

### Phase 1 — Stage A VAE in candle

Small 3.6M-param VAE. Encode 1024² → 32×32 latent; decode 32×32
latent → 1024² image. Tensor names from upstream `vqgan/
diffusion_pytorch_model.safetensors` in the diffusers repo.

**Acceptance:** forward-pass shape test verifies encode/decode
round-trip.

~1 session.

### Phase 2 — Stage B latent prior

~1.5B-param UNet. Takes Stage C's 24×24 latent + text → Stage
A's 32×32 latent. Variant-aware: Full vs Lite Stage B may differ
in width/depth. Closer to SDXL UNet than to PixArt DiT.

**Acceptance:** forward-pass shape test at small dims.

~2 sessions.

### Phase 3 — Stage C high-res prior

~3.6B-param UNet. Text → 24×24×16 latent. CLIP-G cross-attention.
Variant-aware (Full vs Lite Stage C — Lite is the motivation for
the Lite variant). Largest model in the cycle.

**Acceptance:** forward-pass shape test at small dims. Numerical
verification at smoke time (real weights).

Risk: tensor naming + numerical drift mitigations parallel
v0.35 phase 1 / v0.36 phase 3.

~2-3 sessions.

### Phase 4 — 3-stage pipeline orchestration

`text → Stage C → Stage B → Stage A → image` chained denoising.
Two scheduler instances (C and B each get their own denoise loop;
A is one-shot decode). Seed plumbing through `pipelines::seeds::
prepare_seed` for both stages. Standard CFG on Stage C.

**Acceptance:** `plakat generate "..." --model stable-cascade`
produces a valid 1024² PNG end-to-end (smoke at user time —
~12 GB downloads + ~24 GB VRAM).

~1-2 sessions.

### Phase 5 — CLI integration + presets + doctor + scenarios

- `plakat doctor` lists Stable Cascade.
- `--reproducibility-check` row classified Guaranteed.
- v0.25 look preset routing extends to `BaseFamily::StableCascade`.
- Scenario integration parallel to PixArt's v0.36 phase 0
  (`cascade_pipeline` slot + dispatch arm + VAE cache passthrough
  if compatible).

~1 session.

### Phase 6 — Cycle close-out

Standard 7-step release.

~0.5 session.

## Risks

- **Phase 3 Stage C tensor naming** (history: v0.27 phase 2
  motion_module, v0.35 phase 1 DiT, v0.36 phase 3 KV-compress).
  Mitigation: explicit safetensors key walk + per-layer shape
  tests before higher integration.
- **3-stage chaining numerical drift.** Stages B and C each
  produce latents that feed downstream stages. F16 drift across
  three stages may compound; mitigation: fp32 fallback for
  layernorm scales (borrow SD3 / Flux pattern).
- **Lite variant config detection.** Need to clearly distinguish
  Full vs Lite stages from repo path. Mitigation: alias-based
  routing similar to PixArt's `for_pixart_repo`.
- **Scheduler compatibility.** Stable Cascade uses an EDM-style
  scheduler for each prior; plakat's existing schedulers may
  not all compose. Phase 4 surfaces this.

## What's NOT in v0.37

Deferred to v0.38+:

- **Stable Cascade LoRA support.** Less-established community
  LoRA ecosystem than PixArt or SD-family; ship core first.
- **Scripting `plakat.cascade` Bund word.** Mirrors `plakat.
  pixart` from v0.36 phase 1; defers to a follow-up cycle.
- **Stable Cascade img2img / ControlNet.**
- **PixArt v0.36 carries** (α-LCM checkpoint, ControlNet,
  portrait, img2img).
- **T5 loader extraction** across SD3 + Flux + PixArt (Cascade
  doesn't use T5 — uses CLIP-G — so this debt doesn't affect
  v0.37 but stays open).
- **`GenerationMetadata` for SD3 / Flux / AnimateDiff / stylize
  / portrait** (v0.34 carry). Cascade itself will emit metadata
  from phase 4.
- All long-standing deferrals (per-layer motion splice,
  HotShot-XL, AnimateLCM-SDXL externally blocked, INT8 SDXL
  externally blocked, plakat server mode).
