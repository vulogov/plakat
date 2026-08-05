# RFC TEXTURE-1 — `plakat texture`: seamless PBR material synthesis

**Status:** Draft · **Target:** plakat 6.3.0 flagship · **Author:** Vladimir Ulogov
**Siblings in shape:** [`RFC_BOOKART_1.md`](RFC_BOOKART_1.md) · [`RFC_PERSONA_1.md`](RFC_PERSONA_1.md) · [`RFC_FRACTALS_1.md`](RFC_FRACTALS_1.md)

## 1. Summary

`plakat texture` turns a **prompt or a photo** into a **seamless, tileable PBR material set** — the
standard metal/rough channel maps (**albedo · normal · roughness · metallic · height · ambient
occlusion**) — flat-lit, exported engine-ready, with a built-in lit preview. Like `persona` and
`bookart`, a material is treated as **structured data**: a small HJSON `TextureSpec` resolved
deterministically, generated through a hybrid pipeline, and *measured* (a tileability + PBR-validity
scorecard). Fully additive — nothing here touches existing behaviour.

Where a bare prompt gives one lit, non-tiling RGB image, `plakat texture` gives a **material** you can
drop into Unity / Unreal / Blender / a glTF and tile across a surface.

The headline capability — and the reason this belongs *in* plakat rather than a diffusers wrapper — is
**native seamlessness by circular convolution**: because plakat owns its SD UNet
([`sd_train/unet.rs`](../src/pipelines/sd_train/unet.rs), the 2.6.0 own-UNet default), we replace the
convolutions' zero-padding with **circular (wrap) padding**, so the diffusion output tiles edge-to-edge
*by construction* — not by post-hoc offset-blending, which is what every local tool that wraps diffusers
must do.

## 2. Motivation

- **Local, private, owns-the-stack.** Pure-Rust inference on candle; no Substance/Materialize, no cloud
  texture services. The same values as the rest of plakat.
- **A material is data, not a picture.** A game texture is a *set* of coherent channels with hard
  correctness constraints (it must tile; the normal map must encode tangent-space slope; albedo must be
  delit). A prompt can't express that; a spec + a measured pipeline can.
- **plakat already owns the hard parts.** Own SD UNet (circular conv), IC-Light (delighting), tiled
  upscale, ControlNet (depth/tile), the `noise` heightfield engine from `map`, and the
  spec→resolve→render→measure spine from `persona`/`bookart`. This flagship *synthesises* them.

## 3. Owner decisions (locked 2026-08-05)

1. **Seamless = native circular convolution** (own-UNet conv padding → circular). The clean
   differentiator; offset-inpaint is at most a fallback (see G0.2 / VAE).
2. **Channel strategy = hybrid.** Generate **albedo + height**; **derive** normal (Sobel→tangent-space)
   and AO (from height); **heuristic + per-channel override** for roughness and metallic.
3. **In scope (all):** delighting via **IC-Light**, a **pure-Rust PBR preview** render, **engine-ready
   export** (ORM-packed + naming conventions + glTF), and **image-to-material** (photo → tileable PBR).
   Text-to-material is the baseline.
4. **Resolution = 1K native → tiled 2K/4K.** Generate at 1024² (SDXL-friendly); tiled-upscale to 2K/4K
   on request.

## 4. The output contract

`plakat texture render <spec> --out mat/` produces a **material directory**:

```
mat/
  albedo.png        # base color, flat-lit (sRGB)
  normal.png        # tangent-space normal (linear, +Y "OpenGL" by default; --normal-y flips to DirectX)
  roughness.png     # linear, 0 = mirror … 1 = matte
  metallic.png      # linear, 0 = dielectric … 1 = metal
  height.png        # linear displacement / bump
  ao.png            # linear ambient occlusion
  orm.png           # packed: R=AO, G=roughness, B=metallic  (Unreal/glTF-friendly)
  preview.png       # a lit sphere/plane shade of the full set
  material.json     # the resolved recipe + channel manifest + scorecard
  material.gltf     # (optional) a glTF 2.0 material referencing the maps
```

Every channel is **exactly tileable at the same resolution**, in the correct **color space** (albedo
sRGB; data maps linear), with the recipe embedded in a PNG `tEXt` chunk + the `.json` sidecar (the
`bookart` A5 pattern).

## 5. The `TextureSpec` (HJSON)

Permissive serde exactly like `PersonaSpec` / `BookArtSpec`: **every field optional** (a bare `{}`
resolves to a neutral 1K material), enums are strings (unknown → `lint` suggests, not a hard fail),
unknown keys ignored (forward-compatible).

```hjson
{
  schema: "texture/1"
  material: "mossy cobblestone"        # the prompt (text-to-material); or use `from_image`
  from_image: "cobbles.jpg"            # image-to-material (crop-to-tileable + delight); optional

  seamless: {
    mode: "circular"                   # circular (default) | offset | none
    axes: "both"                       # both | x | y   (trim sheets tile one axis)
  }

  channels: {
    height: "auto"                     # auto (depth-CN pass) | from-albedo | "<prompt>"
    roughness: 0.6                     # a scalar default, OR "from-albedo" OR a "<prompt>"
    metallic: 0.0                      # scalar | "from-albedo" | "<prompt>"
    normal_strength: 1.0               # slope gain when deriving normal from height
    ao_strength: 1.0
    normal_y: "opengl"                 # opengl (+Y) | directx (-Y)
  }

  delight: true                        # IC-Light flatten baked lighting on albedo (default true)

  page: {                              # (naming echoes bookart) — the raster target
    size: 1024                         # native gen size (square)
    upscale: "none"                    # none | 2k | 4k   (tiled, tileability-preserving)
    tiling_check: true                 # run the seam scorecard
  }

  export: {
    maps: ["albedo","normal","roughness","metallic","height","ao"]   # which to write
    orm: true                          # also write the packed ORM
    gltf: false                        # also write a glTF 2.0 material
    naming: "plakat"                   # plakat | unity | unreal   (channel filename convention)
    preview: true
  }

  model: "sdxl"                        # base for the diffusion passes
  seed: 0
  steps: 28
}
```

## 6. The pipeline (layer model)

Mirrors the `bookart` render router — **resolve → generate → derive → measure → export**:

```
TextureSpec
   │  (resolve: defaults, model family, color spaces, seamless mode)
   ▼
RenderPlan
   │  Layer 1 — ALBEDO   : circular-conv diffusion @ 1024² (§7) → delight (IC-Light, §9)
   │  Layer 2 — HEIGHT   : depth-ControlNet pass conditioned on albedo (or from-albedo luminance)
   │  Layer 3 — DERIVE   : normal (Sobel→tangent, §8) · AO (from height) · roughness/metallic
   │                        (scalar | from-albedo heuristic | own conditioned pass)
   │  Layer 4 — UPSCALE  : optional tiled 2K/4K, tileability-preserving (§10)
   │  Layer 5 — MEASURE  : the tileability + PBR-validity scorecard (§12)
   │  Layer 6 — EXPORT   : channel PNGs + ORM + preview + glTF + sidecar (§11)
   ▼
material directory
```

Deterministic and pure wherever it can be (derivation, packing, preview, scorecard are weight-free and
CI-testable); only the diffusion + IC-Light + upscale layers need weights.

## 7. The seamless engine (headline)

**Native circular convolution.** candle's `Conv2dConfig` only zero-pads, so we implement a **circular
pad** (`Tensor` `narrow` + `cat` the wrapped edges) applied before each `Conv2d` with `padding: 0`. We
thread a `seamless: bool` (and per-axis) flag through the own-UNet conv construction
([`sd_train/unet.rs`](../src/pipelines/sd_train/unet.rs), [`blocks.rs`](../src/pipelines/sd_train/blocks.rs))
— every 3×3 conv gets a wrap-pad instead of a zero-pad. Result: the latent (and therefore the image)
**wraps edge-to-edge by construction**. `axes: x|y` tiles a single axis (trim sheets).

**The VAE wrinkle (G0.2).** The SD VAE decoder is candle's `AutoEncoderKL` (not plakat-owned), and its
convs also touch the boundary. Three candidate strategies, decided by a G0 probe measuring the residual
seam of **UNet-only** circular conv:
- (a) **UNet-only** — if the residual VAE-boundary seam is below the scorecard threshold, ship it.
- (b) **VAE seam-repair** — a final offset-by-½ + hairline blend/inpaint at the exact tile boundary
  (cheap, robust, model-agnostic) to clean any residual.
- (c) **Wrap the VAE decoder convs** — a circular-pad wrapper around `AutoEncoderKL`'s decode (most
  correct, most surface). Preferred only if (a)+(b) prove insufficient.

The tileability scorecard (§12) makes this a *measured* choice, not a guess.

## 8. Channel derivation (weight-free, deterministic)

- **Height.** `auto` = a depth-ControlNet pass conditioned on the albedo (reuse `controlnet` depth);
  `from-albedo` = a fast luminance→height (blurred, contrast-normalised). Always tileable (derived from
  a tileable source, with circular gradients).
- **Normal (tangent-space).** From height `h`: `n = normalize(-∂h/∂x·s, -∂h/∂y·s, 1)` via **circular**
  Sobel (so the normal map tiles), scaled by `normal_strength` `s`, encoded to `[0,1]` RGB. `normal_y`
  selects OpenGL (+Y) or DirectX (-Y). Verified against a reference (G0.4).
- **AO.** From height: a cheap horizon/cavity approximation (multi-directional circular height
  comparison) → `[0,1]`, `ao_strength`.
- **Roughness / Metallic.** `scalar` (a flat map) | `from-albedo` (luminance/saturation heuristic:
  darker+desaturated → rougher; a metal-cue heuristic for metallic) | `"<prompt>"` (an albedo-
  conditioned diffusion pass for the fussy cases). Default: `roughness 0.6`, `metallic 0.0`.

## 9. Delighting

Albedo must be **flat-lit** (no baked highlights/shadows) or it double-lights in an engine. Default
`delight: true` runs the albedo through **IC-Light** ([`ic_light.rs`](../src/pipelines/ic_light.rs)) with
a uniform target light to normalise illumination, plus a flat-lighting prompt anchor at generation
time. G0.3 validates that IC-Light actually flattens a baked-shadow texture (it was trained for portrait
relighting — texture is a new regime to verify).

## 10. Resolution + tiled upscale

Generate at **1024²** (SDXL). `upscale: 2k|4k` runs the **tiled** upscaler
([`tiled.rs`](../src/pipelines/tiled.rs) / [`diffusion_upscale.rs`](../src/pipelines/diffusion_upscale.rs))
in a **tileability-preserving** mode (circular tile seams so the upscaled map still wraps). Derivation
(normal/AO) runs *after* upscale so the detail maps carry the full resolution. G0.5 confirms tiling
survives upscale.

## 11. Export

- **Channel PNGs** in the correct color space (albedo sRGB, data maps linear/16-bit where it matters —
  height/normal benefit from 16-bit).
- **ORM pack** — R=AO, G=roughness, B=metallic (glTF/Unreal convention).
- **Naming conventions** — `plakat` (as above) | `unity` (`_MainTex`/`_BumpMap`/`_MetallicGloss`) |
  `unreal` (`_BaseColor`/`_Normal`/`_ORM`).
- **glTF 2.0** — an optional `.gltf` with a `pbrMetallicRoughness` material referencing the maps (a
  drop-in for viewers / DCC import).
- **Preview** — a **pure-Rust** PBR shade: a sphere (or plane) lit by one directional + ambient light,
  sampling the tiled material with the derived normal — a Cook-Torrance-lite BRDF, no GPU, deterministic
  — so you *see* the material without opening a 3D tool.

## 12. The scorecard (measured quality)

Pure, weight-free, the `bookart` `verify` analog:

| Probe | Checks |
|---|---|
| **tileability-x / -y** | edge-wrap error — L2 of the left-vs-right (top-vs-bottom) boundary seam after wrap; the seamless guarantee, *falsifiable* |
| **normal-validity** | every texel is a unit vector with +Z (a valid tangent-space normal), B≈1 bias correct |
| **albedo-flatness** | low-frequency luminance variance (a proxy for "delit" — no baked gradient) |
| **channel-consistency** | all maps identical resolution + aspect; ORM packs losslessly |
| **value-range** | roughness/metallic/AO in `[0,1]`; height uses its full range |

Drives `--attempts N` rejection sampling (regenerate the albedo seed until tileability + flatness pass).

## 13. CLI surface

```
plakat texture new       <out.hjson> [--material "…" --size 1024 --model sdxl]   scaffold a spec
plakat texture lint      <spec>                                                  validate (no weights)
plakat texture show      <spec>                                                  the resolved plan
plakat texture render    <spec> --out DIR [--seed --steps --attempts --upscale 2k]   the full set
plakat texture from      <image> --out DIR [--material "…"]                       image-to-material
plakat texture derive    <albedo.png> --out DIR [--height H]                      channels from an existing albedo (no gen)
plakat texture preview    <mat-dir> --out preview.png [--shape sphere|plane]      re-render the lit preview
plakat texture export     <mat-dir> --naming unreal [--gltf --orm]               re-pack for an engine
plakat texture verify     <mat-dir>                                              the tileability/PBR scorecard
```

`derive` / `preview` / `export` / `verify` / `lint` / `show` are **weight-free** (run anywhere, no GPU).

## 14. Integration parity (the `bookart` lesson — do it from day one)

- **scenario** — a `type: texture` task.
- **compile** — a `type: texture` block (`texture-material:` / `-size:` / `-upscale:` / `-seamless:`).
- **Bund** — `plakat.texture.render` / `.derive` / `.preview` → image handles (the ORM/preview) into the
  `plakat.save` pipeline.
- **library API** — `plakat::api::Texture` (`from_prompt` / `from_image` · `size`/`seed`/`upscale`/…
  · `run() -> Material`).
- **photos** — `--import` a preview + recipe into an album.

## 15. Reuse map (build on, don't rebuild)

| Need | Reuse |
|---|---|
| circular-conv diffusion | **own SD UNet** `sd_train/unet.rs` + `blocks.rs` (the enabler) |
| delighting | `pipelines/ic_light.rs` |
| height conditioning | `pipelines/controlnet.rs` (depth) |
| tiled 2K/4K | `pipelines/tiled.rs` · `diffusion_upscale.rs` · `real_esrgan.rs` |
| heightfield / noise | `map`'s `noise` engine (procedural height option) |
| spec/resolve/scorecard/CLI spine | `bookart` (`compile.rs`, `scorecard.rs`, `finish/canvas.rs` DPI-PNG, the command shape) |
| recipe sidecar + tEXt | `bookart` A5 (`recipe_metadata` pattern) + `imaging::io` |
| integration surfaces | `bookart` scenario/compile/Bund/API (A1–A5) as the template |

## 16. G0 gating (do FIRST — feasibility before the build)

1. **G0.1 circular-conv probe.** Implement circular pad in candle; wire it into one own-UNet conv path;
   generate a tile and measure the tileability-x/y seam vs a zero-pad baseline. Confirm it works on
   **Metal** (the candle-Metal ≤4-D rule applies). *The load-bearing item.*
2. **G0.2 VAE seam.** Measure UNet-only residual seam → pick strategy (a) ship / (b) seam-repair /
   (c) wrap VAE (§7).
3. **G0.3 delighting.** Does IC-Light flatten a baked-shadow texture? A/B albedo-flatness with/without.
4. **G0.4 normal correctness.** Derived normal vs a reference (a known height→normal fixture); confirm
   tangent-space encoding + OpenGL/DirectX Y.
5. **G0.5 tiled-upscale tiling.** Does 1K→2K tiled upscale preserve the wrap? Measure post-upscale seam.

Recommended de-risking slice: **one albedo (circular-conv) + derived normal + preview + tileability
score**, end-to-end, before the rest.

## 17. Module layout (additive)

```
src/texture/{spec,compile,lint,render,derive,scorecard,export,preview,mod}.rs
src/texture/seamless.rs            # circular-pad + the UNet/VAE wiring
src/pipelines/sd_train/…           # +seamless flag (the only touch to an existing pipeline)
src/cli/texture.rs
src/api (Texture builder) · scenario/compile/scripting integration (mirrors bookart A1–A5)
assets/texture/  (naming conventions, glTF template)
```

## 18. Milestones

- **B0** — spec + lexicon/defaults + `lint`/`show` + `RenderPlan` (pure, golden-tested).
- **B1** — derivation core (`derive.rs`: normal/AO/roughness/metallic from a height/albedo) +
  scorecard + `texture verify`/`derive` — **weight-free, ships value without generation**.
- **B2** — the preview renderer (`preview.rs`, pure-Rust PBR) + export (ORM/naming/glTF).
- **B3** — the **seamless engine** (`seamless.rs`: circular conv in the own UNet) — the headline
  (post-G0.1).
- **B4** — albedo generation (circular-conv) + delighting (IC-Light) + `texture render` end-to-end.
- **B5** — height (depth-CN) + tiled upscale (tileability-preserving).
- **B6** — `from` (image-to-material) + rejection sampling.
- **B7** — integration parity (scenario/compile/Bund/API/import).
- **B8** — corpus + docs + the release cut.

Front-load B0–B2 (pure, CI-tested: a full material can be *derived + previewed + exported* from a
supplied albedo before any generation exists), exactly as `bookart` front-loaded its finisher/scorecard.

## 19. Honest scope / non-goals

- **Not a substance-graph editor.** No node graphs, no procedural-only synthesis (the `noise` height is
  a helper, not a Substance Designer). Generation + derivation, measured.
- **Metal/rough only** (the modern standard) — no spec/gloss workflow in v1.
- **Tangent-space normals only** — no object-space.
- **Tileability is measured, not proven at the shader level.** The scorecard bounds the seam; a pixel-
  perfect guarantee depends on the VAE strategy (§7 / G0.2).
- **The preview is an approximation** (Cook-Torrance-lite, one light) — a sanity view, not a renderer.

## 20. Companion documents (to follow)

- `ROADMAP_TEXTURE_1.md` — the B0–B8 build plan (modules/paths/CI-vs-GPU), after G0.
- `Documentation/TEXTURE.md` + `TEXTURE_TUTORIAL.md` — on ship.
