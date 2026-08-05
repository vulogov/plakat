# ROADMAP — plakat 6.3.0 `plakat texture` (RFC TEXTURE-1)

The implementation plan for the 6.3.0 flagship (RFC [`RFC_TEXTURE_1.md`](Documentation/RFC_TEXTURE_1.md)):
a prompt/photo → a **seamless, tileable PBR material set**. This maps the RFC's decisions to concrete
modules/paths and a CI-vs-GPU test split, and front-loads the weight-free half so value ships (and is
golden-tested) before any generation exists — the discipline that worked for `persona` and `bookart`.

**Theme:** *a material is data — resolved, generated seamlessly, derived, measured, exported.*

---

## G0 — feasibility gating (do FIRST, before B1)

Each item is a small runnable probe with an **exit criterion**. The whole point is to de-risk the
load-bearing assumptions (chiefly circular-conv) before committing the build.

- **G0.1 — circular-conv probe. LOAD-BEARING.**
  - *Probe:* implement a `circular_pad2d(&Tensor, p)` in candle (wrap edges via `narrow` + `cat`); prove
    it on Metal (candle-Metal ≤4-D rule). Then thread a `seamless` flag through the own-UNet conv helper
    (`sd_train/blocks.rs` `conv` + `sd_train/unet.rs` `conv_in`/`conv_out`) so every 3×3 conv wrap-pads
    instead of zero-pads, and run **one real sd15/sdxl generation** with it on.
  - *Exit:* the generated image's **tileability-x/-y seam** (L2 of the wrapped boundary) drops
    markedly vs the zero-pad baseline (target: ≥5× lower), on Metal, no NaN/garbage.
  - *Artifact:* `examples/texture_seamless_probe.rs`.
- **G0.2 — VAE seam.** With G0.1 on (UNet-only circular), measure the **residual** seam after VAE
  decode. *Exit:* a decision — (a) UNet-only ships (residual < threshold), (b) add an offset-½ seam-
  repair blend, or (c) wrap the candle `AutoEncoderKL` decoder convs. Record the measured residual.
- **G0.3 — delighting.** Run IC-Light on a baked-shadow texture crop; measure **albedo-flatness** (low-
  freq luminance variance) with/without. *Exit:* IC-Light meaningfully flattens (or → fall back to a
  flat-lighting prompt anchor + a gentle high-pass delight; record which).
- **G0.4 — normal correctness.** Height→normal on a known fixture (a hemisphere / a ramp) vs the
  analytic normal. *Exit:* per-texel angular error < a few degrees; OpenGL(+Y)/DirectX(-Y) both correct.
- **G0.5 — tiled-upscale tiling.** 1K→2K tiled upscale of a seamless albedo; measure post-upscale seam.
  *Exit:* tiling survives (seam stays below threshold) — else the tiled upscaler needs a circular-tile
  mode (note it for B5).

**Recommended de-risking slice after G0:** one circular-conv albedo → derived normal → preview →
tileability score, end-to-end, before building the rest.

---

## Milestones

### B0 — spec + resolver + `lint`/`show` (pure, golden-tested). DONE.
- `src/texture/{spec,compile,lint,mod}.rs` + `src/cli/texture.rs`; registered in `lib.rs` + `cli/mod.rs`
  (`Command::Texture`, not feature-gated). Mirrors bookart B0.
- `TextureSpec` (permissive serde, HJSON, every field optional; roughness/metallic accept scalar OR
  `"from-albedo"` OR a `"<prompt>"` via `serde_json::Value`). `compile::resolve` → `RenderPlan`
  (Serialize+PartialEq, byte-stable: seamless mode/axes, `ChannelSource`/`HeightSource`, export plan,
  the compiled albedo prompt with flat-light + tileable anchors + anti-shadow/anti-seam negative).
  `lint` (vocab nearest-match + ranges, non-zero exit). `new`/`lint`/`show` CLI.
- **Weight-free, 10 tests green** (bare/full-spec parse, scalar-or-string channels, deterministic
  resolve, map-filtering, lint ranges + typo suggestions). Verified live: `new → lint → show`. Full
  suite 1749 green.

### B1 — derivation core + scorecard + `verify`/`derive` (WEIGHT-FREE). DONE.
- `src/texture/{derive,scorecard}.rs` + `Material` type.
- `derive.rs`: `Material::derive(albedo, height?, …, roughness, metallic)` → the full set. **normal** via
  circular Sobel → tangent-space (G0.4 math, `normal_strength` gain ×6, OpenGL/DirectX Y), **AO** via
  circular cavity (below-neighbourhood occlusion), **height** from luminance (circular blur +
  autocontrast), **roughness/metallic** scalar | from-albedo heuristic. All pure, deterministic,
  **circular** (wrapped ops → derived maps tile). `write_channels`.
- `scorecard.rs`: tileability-x/-y (edge-wrap join/interior), normal-validity (unit +Z fraction),
  albedo-flatness (low-freq luma std), channel-consistency; hard gate = tiling + normal + consistency,
  flatness advisory.
- `texture derive <albedo> --out DIR [--height --normal-strength --normal-y]` + `texture verify <dir>`.
- **Weight-free, 5 new tests** (flat/ramp normal + Y-flip, deterministic derive, seam flagged, wrapping
  passes). Verified live: a tileable albedo → 6 correct maps (blue-dominant normal, cavity AO, dielectric
  metallic), scorecard PASS (tileability 1.14/0.84, normal-valid 1.000). Full suite 1754 green.

### B2 — preview renderer + export. DONE.
- `src/texture/{preview,export}.rs`.
- `preview.rs`: pure-Rust **Cook-Torrance-lite** PBR shade (GGX D + Schlick-GGX G + Fresnel; one
  directional + ambient·AO; sRGB→linear albedo; Reinhard+gamma) of a lit **sphere** or **plane** that
  wrapped-bilinear-samples the tiled material through its perturbed (TBN) normal. `Shape::{Sphere,Plane}`.
- `export.rs`: `orm_pack` (R=AO/G=rough/B=metal = the glTF metallicRoughness[GB]+occlusion[R] layout),
  `channel_filename` (plakat / unity `_BumpMap`… / unreal `T_*`), a minimal **glTF 2.0** material doc,
  and `material.json` manifest (recipe + written maps + scorecard + FNV **spec-hash**). `write_material`
  = channels(named) + orm + preview + gltf + manifest.
- `texture derive` now writes the full dir; `texture preview` / `texture export` re-render / re-pack.
- **Weight-free, +4 tests** (preview deterministic+lit shading range, ORM pack values, naming maps,
  full-dir write w/ valid glTF+manifest JSON). Verified live: a sphere+plane preview (proper diffuse
  falloff + normal-mapped relief), unreal re-pack (`T_BaseColor`…`T_ORM`) + glTF. Full suite 1758 green.
  *(The whole weight-free half — derive + score + preview + export — is done + demoable.)*

### B3 — the seamless engine primitives (headline enabler). DONE.
- **Module:** `src/texture/seamless.rs` — self-contained + CI-tested, so B4 applies them **without**
  risky surgery on the shared corr-1.0 generation stack. (G0.1 finding: the own UNet's ResNet convs are
  candle's, not plakat-owned; the disproportionate-risk full-vendor is escalated-to-only-if-measured, per
  G0.2's measure-first rule.)
- `circular_pad2d(t, px, py)` (per-axis, from the G0.1 probe) · `roll2d(t, dx, dy)` (circular shift — the
  **per-step latent roll** that spreads the zero-pad conv seam across the tile, tileable diffusion with
  no model change) · `SeamlessConv2d` (the vendored-ResNet escalation path) · `feather_seam` (weight-free
  hairline blend — the G0.2 VAE seam-repair) · `Axes{Both,X,Y}`.
- **4 tests** (exact wrap, per-axis pad, roll circular+invertible, feather shrinks a seam). Full suite green.

### B4 — seamless generation + delighting + `texture render` e2e. DONE.
- **Module:** `src/texture/render.rs`. **NO sampler surgery** — the measure-first bet paid off: albedo
  via `api::Generate` with the **flat/tileable prompt** (§5) + a post-hoc `feather_seam` (B3) was enough.
- **Delight = weight-free homomorphic flatten** (`derive::flatten_lighting` — divide out the low-freq
  illumination; circular so it stays tileable). Chosen over IC-Light, which is subject-oriented
  (expects an RGBA cut-out) and G0.3-dubious on a flat texture — the flat-lighting prompt + this flatten
  are the primary path. → derive (B1) → scorecard → export (B2). Rejection sampling `--attempts N`.
- `texture render <spec> --out DIR`. **GPU. Live-verified on Metal**: `mossy cobblestone` sdxl 1024²
  → **tileability x 0.08 / y 0.07** (≪ 1.5), normal-valid 1.000, albedo-flat 0.013, PASS; the 2×2 tile
  shows **no seam** at the join, the lit sphere is a proper material preview.
- *(The per-step latent-roll + vendored `SeamlessConv2d` remain the escalation path if a future
  high-frequency material's feather residual fails — but it clears the scorecard as-is.)*

### B4 — albedo generation + delighting + `texture render` e2e
- **Modules:** `src/texture/render.rs` (the router: resolve → circular-conv albedo → IC-Light delight →
  derive (B1) → score → export (B2)). `src/api` `Texture` builder stub.
- Reuse `api::Generate` with the seamless scope on; `ic_light` for delight (or the G0.3 fallback).
- `texture render <spec> --out DIR` end-to-end (text-to-material). Rejection sampling `--attempts N`
  (regen albedo seed until tileability + flatness pass).
- **GPU.** Live-verified on Metal (a seamless, delit, derived material).

### B5 — height (depth-CN) + tiled upscale
- `channels.height = auto` → a depth-ControlNet pass conditioned on albedo (reuse `controlnet`), else
  `from-albedo`. `page.upscale = 2k|4k` → tiled upscale in a **tileability-preserving** mode (circular
  tile seams; per G0.5 — add a circular mode to `tiled.rs` if needed). Derivation runs post-upscale.
- **GPU.** Live-verified: a 2K seamless material with a generated height.

### B6 — image-to-material + polish
- `texture from <image> --out DIR` — crop-to-tileable (+ optional `--material` re-gen) + delight →
  the full derive/export path. Rejection sampling shared with B4.
- **GPU.**

### B7 — integration parity (the bookart lesson — bookart A1–A5 as template)
- `plakat::api::Texture` (full builder). Scenario `type: texture`. Compile `type: texture`
  (`texture-material:`/`-size:`/`-upscale:`/`-seamless:`). Bund `plakat.texture.*`
  (render/derive/preview → image handles). `--import` a preview + recipe into a photos album.
  `doctor` section.
- Touchpoints mirror bookart's scenario_task (7) / compile (parser/resolver/emitter) / words / api.

### B8 — corpus + docs + the CUT
- `corpus/texture_*.{hjson,sh}` + `TEXTURE_CORPUS.md`. `Documentation/TEXTURE.md` +
  `Tutorials/TEXTURE_TUTORIAL.md` + README what's-new + docs-index. **Release 6.3.0**: bump
  Cargo.toml+lock, gate green, FF `git push 6.3.0:main`, tag v6.3.0 → CI 6-asset + `cargo publish
  --locked --allow-dirty --no-default-features`, `gh release edit` (owner vulogov GH token; NO Claude
  coauthor).

---

## Module layout (additive)

```
src/texture/{spec,compile,lint,derive,scorecard,preview,export,render,seamless,mod}.rs
src/cli/texture.rs
src/pipelines/sd_train/{unet,blocks}.rs      # + seamless flag (only touch to an existing pipeline)
src/api  (Texture builder)                   # + scenario/compile/scripting integration (B7)
assets/texture/  (naming maps, glTF template)
examples/texture_seamless_probe.rs           # G0.1
```

## Build discipline

Front-load **B0–B2** — pure, GPU-free, golden-tested — so the whole *derive + preview + export* half is
proven under `cargo test --no-default-features --lib` **before** any generation. B3 (seamless) is the
one net-new pipeline touch and is gated by G0.1. B4+ are additive on top. Keep candle tensor ops **≤4-D
on Metal** ([[reference_candle_metal_4d_matmul]]) — the circular pad is `narrow`+`cat` (rank-safe).

## Reuse (don't rebuild)

Own SD UNet (`sd_train/*` — circular conv) · `ic_light.rs` (delight) · `controlnet.rs` depth (height) ·
`tiled.rs`/`diffusion_upscale.rs`/`real_esrgan.rs` (2K/4K) · `map`'s `noise` (procedural height option) ·
bookart `compile.rs`/`scorecard.rs`/`finish/canvas.rs` (DPI-PNG) + the A5 recipe-sidecar pattern +
integration surfaces (bookart A1–A5) as the template.

## Release-flow reminders (auto-memory)

Bump `Cargo.toml` **+ `Cargo.lock`** in sync (`--locked` CI); gate = `cargo test --no-default-features
--lib`; new capability in `doctor`; **no Claude/Anthropic co-authoring**; FF `main` via `git push
6.3.0:main` + tag → 6-asset CI + `cargo publish --locked --allow-dirty --no-default-features`; `gh
release edit` (**GH_TOKEN = vulogov owner, do NOT `env -u`**); `assets/filipok.md` + `corpus/images`
untracked → `--allow-dirty`.
