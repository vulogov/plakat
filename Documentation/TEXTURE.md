# `plakat texture` — prompt-or-photo → seamless, tileable PBR material set

`texture` turns a **prompt** or a **photo** into a *seamless, tileable* PBR material set — the
metal/rough workflow: albedo · normal · roughness · metallic · height · ambient-occlusion — exported
**engine-ready** with a lit preview. It is the plakat 6.3 flagship (RFC
[`RFC_TEXTURE_1.md`](RFC_TEXTURE_1.md)), and it is **fully additive** — no existing command or output
changes.

**6.4 deepened it.** Spatially-varying **`metallic: "auto"` / `roughness: "auto"`** turn a composite
material (rusted iron, a gilded frame, chipped paint) into a *structured* mask instead of per-pixel
speckle; **anisotropy** writes a grain flow map for brushed metals; hand-painted **`--metallic-ref` /
`--roughness-ref`** masks are the ultimate override; **`texture blend`** cross-fades two materials
(stone → moss) weight-free; **`render --variations N`** renders seed variants side-by-side; and the
seamless **feather is now adaptive** (sized to the measured seam). All additive — no existing output
changes.

`texture` treats a material as **structured data**: a small HJSON document (a `TextureSpec`) is
resolved deterministically, then **generate → derive → measure → export**. The reason is the same one
behind `bookart` and `persona` — a text prompt is a poor instrument for a material. "a seamless rusted
metal texture" comes out with a visible seam at the tile join, baked-in highlights that fight every
light you put on it, a flat normal, and channels that disagree — four categorically different failures
(tiling, delighting, relief, channel-consistency) that need four different remedies, not a better
prompt. So the material is authored, resolved, generated, **measured against a scorecard**, and packed
in an engine's naming convention.

## The output contract — a material directory

Every material is **generated → derived → measured → packed** into a directory. All maps are exactly
tileable at one resolution.

| File | What | Colour space |
|---|---|---|
| `albedo.png` | base colour | sRGB |
| `normal.png` | tangent-space normal (+Y OpenGL default; DirectX flips G) | linear data |
| `roughness.png` | roughness | linear data |
| `metallic.png` | metallic | linear data |
| `height.png` | height / displacement | linear data |
| `ao.png` | ambient occlusion | linear data |
| `orm.png` | packed **R=AO, G=roughness, B=metallic** (the glTF / Unreal layout) | linear data |
| `preview.png` | a lit sphere (a sanity view) | sRGB |
| `material.json` | recipe + channel manifest + scorecard + a stable **spec-hash** | — |
| `material.gltf` | *(opt-in)* a glTF 2.0 material | — |

**Naming conventions** decide the on-disk file names (see `export`):

- `plakat` — `albedo.png`, `normal.png`, `roughness.png`, …
- `unity` — `normal_BumpMap.png`, … (Unity's suffix idiom)
- `unreal` — `T_BaseColor.png`, `T_ORM.png`, … (the `T_` prefix, packed ORM)

## Where the GPU is needed

Most of the pipeline is **weight-free** — it runs anywhere, no GPU, no download. Only generation and
depth touch a model.

| Command | Weights / GPU? |
|---|---|
| `new` · `lint` · `show` | no — pure spec work |
| `derive` | no — height→normal/AO/roughness/metallic is all CPU math |
| `verify` | no — the scorecard is pure measurement |
| `preview` | no — a Cook-Torrance-lite raster |
| `export` | no — re-pack + rename |
| `blend` | no — per-channel lerp through a mask |
| `render` | **yes** — a diffusion model generates the albedo |
| `from` | **depth only** — no generation; `height: auto` needs the depth model |

## Commands

```
plakat texture new     <out.hjson> [--material "…" --size 1024 --model sdxl]      scaffold a spec
plakat texture lint    <spec>                                                     validate (no weights, non-zero exit on error)
plakat texture show    <spec>                                                     the resolved plan
plakat texture derive  <albedo.png> --out DIR [--height H.png --roughness auto|from-albedo|<0..1> --metallic auto|from-albedo|<0..1> --metallic-ref M.png --roughness-ref R.png --anisotropy 0..1 --anisotropy-angle DEG --normal-strength 1.0 --ao-strength 1.0 --normal-y opengl|directx]   full PBR set from an albedo (NO GPU)
plakat texture verify  <mat-dir>                                                  the tileability / PBR scorecard (NO weights)
plakat texture preview <mat-dir> [--out P.png --shape sphere|plane --size 512]    re-render the lit preview (NO GPU)
plakat texture export  <mat-dir> [--out DIR --naming plakat|unity|unreal --orm --gltf]   re-pack for an engine (NO weights)
plakat texture blend   <dirA> <dirB> --out DIR [--mask mix|radial|x|y|<mask.png> --naming plakat]   cross-fade two materials (NO weights)
plakat texture render  <spec> --out DIR [--attempts N --variations N --keep-best --upscale none|2k|4k --metallic-ref M.png --roughness-ref R.png]   generate the material (GPU)
plakat texture from    <image> --out DIR [--material "…" --size 1024 --upscale none|2k|4k --height auto|from-albedo --metallic-ref M.png --roughness-ref R.png]   image→material (GPU: depth only)
```

### `new` — scaffold a spec

Writes a valid partial `TextureSpec` HJSON you then edit. `--material` seeds the prompt, `--size` the
page size (default `1024`), `--model` the diffusion base (default `sdxl`). Every field is optional, so
the scaffold is a starting point, not a straitjacket.

### `lint` — validate without weights

Checks the schema (`texture/1`), the enum vocabularies (`seamless.mode`, `seamless.axes`, `normal_y`,
`upscale`, `naming`), numeric ranges (the `0..1` channel scalars, strengths), and the
`material`-vs-`from_image` intent. Exits **non-zero** on any error, so it can gate CI. No network, no
weights.

### `show` — the resolved plan

Prints what a spec resolves to: text-to-material or image-to-material, the seamless mode + axes, each
channel's resolution (scalar / `from-albedo` heuristic / a generation prompt), whether delighting is
on, the page size + upscale target, the export map list + naming + ORM/glTF/preview flags, and the
model/seed/steps. No rendering.

### `derive` — a full PBR set from an albedo *(no GPU)*

The heart of the weight-free path. Given an `albedo.png` it derives the whole channel set — height,
normal, roughness, metallic, AO — and writes them to `--out DIR` with a `material.json` and preview.
Height comes from `--height H.png` if you supply one, else from albedo luminance. `--normal-strength`
and `--ao-strength` scale the derived relief and cavity; `--normal-y opengl|directx` picks the green
channel convention. Because every derivation is **circular** (see *Concepts*), the derived maps tile as
long as the albedo does. This is also the stage `render` and `from` call internally.

**Metallic / roughness sources.** `--metallic` and `--roughness` each accept a scalar `0..1` (a flat
map), `from-albedo` (a per-pixel heuristic), or **`auto`** (the default — a spatially-coherent
*region* vote, see *Spatially-varying metallic/roughness*). For a **composite** material (rusted iron =
bare steel + rust in one tile) `auto` returns a *structured* mask where `from-albedo` left speckle; for
a **single-class** material it collapses to the correct flat map. `auto` is **opt-in-by-default** but
worth an override when you know the class: a grey **dielectric** (stone, concrete, paper) should pass
`--metallic 0`, a raw **metal** `--metallic 1` — because metal-vs-dielectric is separated by
*saturation* and a grey dielectric can read close to bare metal (see the nuance below).

**Hand-painted overrides.** `--metallic-ref <png>` / `--roughness-ref <png>` take a grayscale mask PNG
and use it **verbatim** (resized to fit) — the ultimate override, ahead of `--metallic` / `--roughness`.
White = metal (for metallic). Available on `derive`, `render`, and `from`.

**Anisotropy.** `--anisotropy <0..1>` (0 = isotropic, default) turns on a grain flow map for
brushed/grained metals: it writes an `anisotropy.png` (RG = grain direction, B = strength) and makes
the lit preview's highlight **stretch along the grain**. `--anisotropy-angle <deg>` sets the direction;
omit it to **auto-detect** from the height's structure tensor. It is consumed by engine anisotropy
workflows (glTF `KHR_materials_anisotropy`).

### `verify` — the tileability / PBR scorecard *(no weights)*

Measures a material directory against what a good PBR set should be and prints the scorecard (below).
Pure measurement — it loads no model. Use it to decide whether a material is shippable, and it is the
gate that drives `render --attempts N`.

### `preview` — re-render the lit preview *(no GPU)*

Re-renders `preview.png` from the maps in a material directory. `--shape sphere|plane` chooses the
preview geometry, `--size` its resolution, `--out` the file. It is a **Cook-Torrance-lite** raster
under one light — a sanity view, not a renderer (see *Honest scope*).

### `export` — re-pack for an engine *(no weights)*

Re-packs an existing material directory for a target engine without regenerating anything. `--naming
plakat|unity|unreal` renames the maps to the engine idiom; `--orm` (re)writes the packed
R=AO/G=roughness/B=metallic image; `--gltf` emits a glTF 2.0 material; `--out` picks the destination.
So one generated material can be re-packed for Unity and Unreal from the same source with no GPU.

### `render` — generate the material *(GPU)*

The full text-to-material path. Resolves the spec, generates the albedo with the diffusion model
(seamless-aware — see *Concepts*), optionally upscales (`--upscale 2k|4k`, tileability-preserving),
derives the whole channel set, measures the scorecard, and packs the directory. `--attempts N` turns on
**rejection sampling**: it generates up to N seeds and keeps the first that clears the hard gate (else
the fewest-issues one). This is the one stochastic step; it is seed-locked and reproducible on a given
device.

`--variations N` is the **other** multi-seed mode: it renders N distinct seed *variants*
**side-by-side** into `<out>/var-0/`, `var-1/`, … (a spread to choose from), where `--attempts`
rejection-samples down to **one** passing result. Add `--keep-best` and the top-scoring variant is also
copied to `<out>` itself. `--metallic-ref` / `--roughness-ref` override the spec's metallic/roughness
with a hand-painted mask (see `derive`).

### `from` — image-to-material *(GPU: depth only)*

Turns a **photo** into a material. It runs **no generation** — the photo *is* the albedo — so it is
much cheaper than `render`. It makes the photo tileable (offset-and-heal, see *Concepts*), then derives
the channel set. The only model it may touch is the **depth** model, and only when `--height auto`:
depth-from-albedo gives the macro relief. `--height from-albedo` keeps it fully weight-free (luminance
height). `--material` lets you annotate the intent, `--size` the working resolution, `--upscale` the
tileability-preserving upscale. `--metallic-ref` / `--roughness-ref` supply hand-painted masks (see
`derive`).

### `blend` — cross-fade two materials *(no weights)*

Blends two material directories through a mask into **one** PBR set — e.g. stone → moss, clean → worn.
`<dirA>` is the material at mask 0, `<dirB>` at mask 255; **every channel lerps by the same mask** and
the normal is renormalised. `--mask` picks the blend:

- **`mix`** (default) — a **tileable** integer-frequency sine interleave; the blended material *still
  tiles*.
- **`radial`** — also **tileable** (a centred radial falloff).
- **`x`** / **`y`** — an intentional **transition sheet** (a left→right or top→bottom wipe) that
  **breaks tiling in that axis** on purpose.
- a **path to a grayscale PNG** — your own mask, used verbatim.

`--naming` packs the export in the engine idiom (`plakat` default). No weights, no GPU.

## The `TextureSpec` schema

Permissive serde, like `PersonaSpec` / `BookArtSpec`: **every field is optional**, enums are carried as
strings (caught by `lint`, not a hard failure), and unknown keys are ignored (forward-compatible). The
schema tag is `"texture/1"`. A full spec:

```hjson
{
  schema: "texture/1"

  # EITHER text-to-material …
  material: "worn rusted iron plating, industrial"
  # … OR image-to-material (a photo):
  # from_image: "scan.jpg"

  seamless: {
    mode: "circular"     # circular (default) | offset | none
    axes: "both"         # both (default) | x | y   — x/y = a trim sheet tiling one axis
  }

  channels: {
    height:   "auto"          # auto (depth+high-pass) | from-albedo (luminance) | "<prompt>"
    roughness: "auto"        # "auto" (region-vote, default) | scalar 0..1 | "from-albedo" | "<prompt>"
    metallic:  "auto"         # "auto" (region-vote, default) | scalar 0..1 | "from-albedo" | "<prompt>"
    anisotropy: 0.0           # 0..1 grain strength (0 = isotropic); omit for none
    anisotropy_angle: 0       # grain direction in degrees; omit to auto-detect (structure tensor)
    normal_strength: 1.0
    ao_strength: 1.0
    normal_y: "opengl"        # opengl (+Y, default) | directx (flips G)
  }

  delight: true               # default — flatten baked lighting

  page: {
    size: 1024
    upscale: "none"           # none (default) | 2k | 4k
    tiling_check: true
  }

  export: {
    maps: ["albedo", "normal", "roughness", "metallic", "height", "ao"]
    orm: true
    gltf: false
    naming: "plakat"          # plakat (default) | unity | unreal
    preview: true
  }

  model: "sdxl"
  seed: 0
  steps: 28
}
```

**The scalar-or-string channels are the subtlety.** A channel like `roughness` or `metallic` can be:

- **`"auto"`** (the default) — a **spatially-coherent region vote**: a structured mask for a composite
  material, a flat map for a single-class one (see *Spatially-varying metallic/roughness*);
- a **scalar** — `roughness: 0.6` — a flat, constant map (fastest, most predictable);
- **`"from-albedo"`** — a **per-pixel** heuristic derived from the albedo (bright/smooth → low
  roughness, etc.);
- a **generation prompt** — `roughness: "worn patches"` — *currently falls back to `from-albedo` in
  `derive`* (a per-channel generation pass is a documented fast-follow).

`height` follows a similar pattern: `auto` (depth model + luminance high-pass), `from-albedo`
(luminance), or a prompt.

`anisotropy` (0..1) and `anisotropy_angle` (degrees) are optional — set the strength for a
brushed/grained metal and omit the angle to auto-detect the grain direction from the height's structure
tensor (CLI `--anisotropy` / `--anisotropy-angle`).

## How it works — the concepts

### Seamless

The RFC's headline was native **circular convolution**; the shipped approach is **measure-first**.

- **Generated albedo** — the flat/tileable prompt keeps the field low-frequency, and a boundary
  **feather** makes it wrap. Measured tileability was excellent (e.g. ~0.1 against a 1.5 seam
  threshold).
- **Photo** — **offset-and-heal** (`make_tileable`): roll the image by half so the edge seams move to
  the *interior*, then feather the central cross where they now sit.

The boundary feather is now **adaptive**: its band is sized to the material's **measured raw seam** — a
thin band when the field is already near-tileable (so it smears less detail), the full band only for a
genuine seam.

A per-step **latent-roll** plus a **vendored circular ResNet** remain the documented **escalation
path** if a material's residual ever fails the scorecard — the measure-first path clears it in practice.

### Delighting

`delight: true` (default) removes baked lighting so the albedo lights correctly in any engine. It is a
**weight-free homomorphic flatten**: divide out the low-frequency illumination (done **circularly**, so
it stays tileable), plus a flat-lighting prompt anchor on generation. IC-Light was considered and
rejected — it is **subject-oriented** (it expects a cut-out); a texture is a different regime (a full
tileable field, no subject).

### Height

- **`auto`** = **Depth-Anything-V2** on the albedo for the **macro** relief, **combined with a
  luminance high-pass** for crisp **micro** detail. Depth alone gives a flat, mushy normal — the
  high-pass is what makes surface grain read.
- **`from-albedo`** = plain luminance height (weight-free).

### Normal / AO

Both are **derived from the height map** with **circular** Sobel (normal) and cavity (AO) operators —
the circularity is what makes the derived maps tile. OpenGL **+Y** is the default; `normal_y: directx`
flips the green channel.

### Spatially-varying metallic/roughness

`metallic: "auto"` / `roughness: "auto"` are the 6.4 headline. Where `from-albedo` decides **per pixel**
— and so scatters **speckle** across a mixed surface — `auto` runs a **spatially-coherent region vote**:
it segments the tile into regions and assigns each region one value, yielding a **structured** mask.

- For a **composite** material (rusted iron = bare steel + rust in one tile; a gilded frame; chipped
  paint) `auto` produces a clean mask — **bare-metal regions white, non-metal black** — instead of the
  per-pixel noise `from-albedo` left behind.
- For a **single-class** material it correctly **collapses to a flat map** (flat black for a dielectric,
  flat white for a raw metal).

**The nuance — why `auto` is opt-in.** Metal-vs-dielectric is separated by **saturation**: bare metal is
near-grey (sat ≈ 0.01), but a grey *dielectric* like stone or concrete still carries some colour (sat ≈
0.1–0.2), so the two can sit close on the axis. `auto` is therefore the sensible default but **not a
substitute for knowing the class**. When you know it, say so: a grey dielectric (stone, concrete, paper)
→ `--metallic 0`; a raw metal → `--metallic 1`. Reach for `auto` when the tile genuinely **mixes** metal
and non-metal. The ultimate override is a hand-painted `--metallic-ref` / `--roughness-ref` mask.

### Anisotropy

For **brushed/grained metals**, `anisotropy` (strength 0..1) writes an `anisotropy.png` **flow map**
(RG = grain direction, B = strength) and makes the lit preview's specular highlight **stretch along the
grain** instead of staying a round dot. The direction comes from `anisotropy_angle` (degrees) or, if you
omit it, is **auto-detected from the height's structure tensor**. Downstream it feeds engine anisotropy
workflows (glTF `KHR_materials_anisotropy`).

### Upscale

`2k` / `4k` is a **tileability-preserving Lanczos**: circular-pad → resize → crop, applied *before*
derivation so the derived channels come out at the upscaled resolution. **Real-ESRGAN is deliberately
avoided** — it tiles internally and hallucinates, both of which break the wrap. Tiling was verified to
**survive** the upscale.

## Reading the channels

The single most common point of confusion is a **flat metallic map** read as a bug. It usually isn't.

**Metallic is near-binary per material.** A surface is *either* a raw **metal** (metallic **1.0**,
white) *or* a **dielectric** — stone, wood, leaves, plastic, wet stone, paper (metallic **0.0**, black).
There is almost nothing in between. So for a **single-class** material a **flat** metallic map is the
**correct, expected** answer:

- **flat black** for a dielectric (stone, leaves, a river, concrete, wood);
- **flat white** for a conductor (brushed steel, gold, copper).

A metallic map only carries spatial **structure** when the tile is a **composite** that mixes metal and
non-metal — rusted iron (bare steel + rust), a gilded frame, chipped paint. That is exactly what
`metallic: "auto"` produces.

`verify` says this out loud in the scorecard. For a mixed tile it reports *"metallic is structured …
composite material"*; for a single-class tile *"metallic is uniform (black = dielectric) — correct for a
single-class material, not a defect"*. Both are passes — a flat map is a fact about the material, not a
failure of the pipeline. Corpus examples: **stone / leaves / river → flat black** (dielectrics),
**brushed steel → flat white** (conductor), **rusted iron → structured**.

## The scorecard

`texture verify` measures a material directory — pure, no weights — so quality is falsifiable and
repairable, and drives `render --attempts N` rejection sampling.

| Probe | Checks | Gate |
|---|---|---|
| **tileability-x** | edge-wrap join vs interior across the X seam — ~1 = seamless, > 1.5 = a seam | hard |
| **tileability-y** | the same across the Y seam | hard |
| **normal-validity** | unit vectors, +Z-facing (a well-formed tangent-space normal) | hard |
| **channel-consistency** | the channels agree with one another | hard |
| **albedo-flatness** | low-frequency luma std — a proxy for a properly de-lit albedo | advisory |

The **hard gate** is **tiling + normal + consistency**; **flatness is advisory** (a legitimately
uneven material can be dark in one corner). A material that clears the hard gate is shippable.

The scorecard also **narrates the metallic channel** so a flat map isn't misread: *"metallic is
structured … composite material"* for a mixed tile, or *"metallic is uniform (black = dielectric) —
correct for a single-class material, not a defect"* for a single-class one (see *Reading the channels*).

## Integration surfaces

The same render core drives every automation surface in 6.3:

- **scenario** — a `type: texture` task (inline `spec:` or `spec_file:`, plus `seed` / `upscale` /
  `attempts`) generates materials inside a batch scenario.
- **compile** — a `type: texture` block (`texture-from` / `texture-size` / `texture-upscale` /
  `texture-seamless` / `texture-height` directives; the prose is the **material prompt**) compiles a
  prose file to a texture scenario.
- **Bund** — `plakat.texture.render` / `.from` / `.preview` push the **lit preview** as an image handle
  into the existing `plakat.save` / `.metadata.write` / `.upscale` pipeline.
- **library API** — `plakat::api::Texture` (`from_prompt` / `from_image` / `from_spec` ·
  `model` / `size` / `seed` / `steps` / `upscale` / `attempts` · `run(out) -> Scorecard`), mirroring
  `Generate` / `Portrait` / `BookArt`.

## Honest scope

- **Not a Substance-graph editor.** No node graphs. `texture` is a spec → material pipeline, not a
  material-authoring graph tool.
- **Metal/rough workflow only.** Albedo · normal · roughness · metallic · height · AO. **No
  spec/gloss** (specular/glossiness) workflow.
- **Tangent-space normals only.** No object-space or world-space normals.
- **Tileability is *measured*, not shader-proven.** The scorecard *measures* the seam (an image
  statistic); it does not prove tiling inside a shader. In practice the measure-first path clears the
  gate comfortably.
- **The preview is an approximation.** `preview.png` is **Cook-Torrance-lite** under **one light** — a
  sanity view to confirm the maps are sane, not a production renderer. Judge the material in your engine.
- **Photo offset-heal softens a band.** `from` moves the seam to the interior and feathers it, which
  softens a band at the moved seam. A generative inpaint would be cleaner — a documented fast-follow.
- **Prompt-driven channels fall back today.** A channel given a generation prompt (e.g.
  `roughness: "worn patches"`) currently falls back to `from-albedo` in `derive`; a per-channel
  generation pass is a fast-follow.

## Companion documents

- [`RFC_TEXTURE_1.md`](RFC_TEXTURE_1.md) — the full design.
- [`Tutorials/TEXTURE_TUTORIAL.md`](Tutorials/TEXTURE_TUTORIAL.md) — a hands-on walkthrough.
