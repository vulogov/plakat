# `plakat texture` — prompt-or-photo → seamless, tileable PBR material set

`texture` turns a **prompt** or a **photo** into a *seamless, tileable* PBR material set — the
metal/rough workflow: albedo · normal · roughness · metallic · height · ambient-occlusion — exported
**engine-ready** with a lit preview. It is the plakat 6.3 flagship (RFC
[`RFC_TEXTURE_1.md`](RFC_TEXTURE_1.md)), and it is **fully additive** — no existing command or output
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
| `render` | **yes** — a diffusion model generates the albedo |
| `from` | **depth only** — no generation; `height: auto` needs the depth model |

## Commands

```
plakat texture new     <out.hjson> [--material "…" --size 1024 --model sdxl]      scaffold a spec
plakat texture lint    <spec>                                                     validate (no weights, non-zero exit on error)
plakat texture show    <spec>                                                     the resolved plan
plakat texture derive  <albedo.png> --out DIR [--height H.png --normal-strength 1.0 --ao-strength 1.0 --normal-y opengl|directx]   full PBR set from an albedo (NO GPU)
plakat texture verify  <mat-dir>                                                  the tileability / PBR scorecard (NO weights)
plakat texture preview <mat-dir> [--out P.png --shape sphere|plane --size 512]    re-render the lit preview (NO GPU)
plakat texture export  <mat-dir> [--out DIR --naming plakat|unity|unreal --orm --gltf]   re-pack for an engine (NO weights)
plakat texture render  <spec> --out DIR [--attempts N --upscale none|2k|4k]       generate the material (GPU)
plakat texture from    <image> --out DIR [--material "…" --size 1024 --upscale none|2k|4k --height auto|from-albedo]   image→material (GPU: depth only)
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

### `from` — image-to-material *(GPU: depth only)*

Turns a **photo** into a material. It runs **no generation** — the photo *is* the albedo — so it is
much cheaper than `render`. It makes the photo tileable (offset-and-heal, see *Concepts*), then derives
the channel set. The only model it may touch is the **depth** model, and only when `--height auto`:
depth-from-albedo gives the macro relief. `--height from-albedo` keeps it fully weight-free (luminance
height). `--material` lets you annotate the intent, `--size` the working resolution, `--upscale` the
tileability-preserving upscale.

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
    roughness: 0.6            # scalar 0..1 (a flat map) | "from-albedo" | "<prompt>"
    metallic:  "from-albedo"  # scalar 0..1 | "from-albedo" | "<prompt>"
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

**The scalar-or-string channels are the subtlety.** A channel like `roughness` can be:

- a **scalar** — `roughness: 0.6` — a flat, constant map (fastest, most predictable);
- **`"from-albedo"`** — a heuristic derived from the albedo (bright/smooth → low roughness, etc.);
- a **generation prompt** — `roughness: "worn patches"` — *currently falls back to `from-albedo` in
  `derive`* (a per-channel generation pass is a documented fast-follow).

`height` follows the same pattern: `auto` (depth model + luminance high-pass), `from-albedo`
(luminance), or a prompt.

## How it works — the concepts

### Seamless

The RFC's headline was native **circular convolution**; the shipped approach is **measure-first**.

- **Generated albedo** — the flat/tileable prompt keeps the field low-frequency, and a boundary
  **feather** makes it wrap. Measured tileability was excellent (e.g. ~0.1 against a 1.5 seam
  threshold).
- **Photo** — **offset-and-heal** (`make_tileable`): roll the image by half so the edge seams move to
  the *interior*, then feather the central cross where they now sit.

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

### Upscale

`2k` / `4k` is a **tileability-preserving Lanczos**: circular-pad → resize → crop, applied *before*
derivation so the derived channels come out at the upscaled resolution. **Real-ESRGAN is deliberately
avoided** — it tiles internally and hallucinates, both of which break the wrap. Tiling was verified to
**survive** the upscale.

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
