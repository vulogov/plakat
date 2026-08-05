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

### B1 — derivation core + scorecard + `verify`/`derive` (WEIGHT-FREE — ships value)
- **Modules:** `src/texture/{derive,scorecard}.rs`.
- `derive.rs`: **normal** from height (circular Sobel → tangent-space, `normal_strength`, OpenGL/DirectX
  Y), **AO** from height (multi-dir circular cavity), **roughness/metallic** (scalar | from-albedo
  heuristic). All pure, deterministic, circular (so derived maps tile).
- `scorecard.rs`: tileability-x/-y, normal-validity, albedo-flatness, channel-consistency, value-range
  (RFC §12).
- `texture derive <albedo.png> --out DIR` (channels from a supplied albedo, **no GPU**) + `texture
  verify <mat-dir>`.
- **Weight-free.** Tests: normal from a known height fixture (feeds G0.4), tileability of a synthetic
  seamless vs seam, ORM round-trip. **CI-gated.**

### B2 — preview renderer + export
- **Modules:** `src/texture/{preview,export}.rs`.
- `preview.rs`: pure-Rust PBR shade (Cook-Torrance-lite, one directional + ambient, samples the tiled
  material w/ the derived normal) → `preview.png`. `--shape sphere|plane`.
- `export.rs`: ORM pack (R=AO/G=rough/B=metal), naming conventions (plakat/unity/unreal), optional glTF
  2.0 material, correct color spaces (albedo sRGB / data linear; 16-bit height+normal option), recipe
  sidecar + tEXt (reuse bookart `recipe_metadata` + `imaging::io`).
- `texture preview` / `texture export`.
- **Weight-free.** Tests: preview determinism + non-blank, ORM lossless pack/unpack, glTF parses, naming
  maps. **CI-gated.** *(After B2: a full material can be derived + previewed + exported from a supplied
  albedo — the whole weight-free half is done + demoable.)*

### B3 — the seamless engine (headline; post-G0.1)
- **Modules:** `src/texture/seamless.rs`; **+`seamless` flag** into `sd_train/{unet,blocks}.rs` (the
  only touch to an existing pipeline).
- `circular_pad2d` + the conv wiring from G0.1, productionised (per-axis `x|y|both`). A `SeamlessScope`
  (thread-local/opts-threaded) so generation opts carry it without changing every signature.
- **GPU.** Tests: circular-pad shape/values (CPU, CI); the seam-reduction is measured live (Metal),
  logged not asserted (per the Tier-2 CPU-canonical rule — don't assert Metal fp drift).

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
