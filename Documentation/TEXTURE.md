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

**6.5 adds trim sheets & decals.** **`texture trim`** composes several finished materials into ONE
banded **trim-sheet atlas** (the game way to texture pipes/panels/edges from one material) with a UV
sidecar; **`texture decal make`** / **`texture decal apply`** build an alpha-masked overlay and stamp it
onto a base material, blending the normal by **Reoriented Normal Mapping (RNM)** so detail rides the
base surface. Both are **weight-free** — see *Trim sheets & decals*. Still additive — no existing output
changes.

**6.6 adds engine export / interop.** One flag — `export <dir> --engine <target>` (also on `render` and
`derive`) — picks an engine's naming convention **and** channel packing **and** material document in a
single shot, so a material lands correctly in glTF, Unreal, Unity HDRP, Godot, or a MaterialX/USD
pipeline. The point it guards against: engines pack the **same** PBR data **differently** (ORM vs the
HDRP mask map), and getting it wrong fails silently in-engine — see *Engine export*. Weight-free and
additive, like the rest of the command.

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
| `trim` | no — bands are pre-rendered material dirs; just composed into an atlas |
| `decal make` · `decal apply` | no — mask build + alpha/RNM composite is all CPU |
| `render` | **yes** — a diffusion model generates the albedo |
| `from` | **depth only** — no generation; `height: auto` needs the depth model |

## Commands

```
plakat texture new     <out.hjson> [--material "…" --size 1024 --model sdxl]      scaffold a spec
plakat texture lint    <spec>                                                     validate (no weights, non-zero exit on error)
plakat texture show    <spec>                                                     the resolved plan
plakat texture derive  <albedo.png> --out DIR [--height H.png --roughness auto|from-albedo|<0..1> --metallic auto|from-albedo|<0..1> --metallic-ref M.png --roughness-ref R.png --anisotropy 0..1 --anisotropy-angle DEG --normal-strength 1.0 --ao-strength 1.0 --normal-y opengl|directx --engine gltf|unreal|unity-hdrp|godot|materialx|plakat]   full PBR set from an albedo (NO GPU)
plakat texture verify  <mat-dir>                                                  the tileability / PBR scorecard (NO weights)
plakat texture preview <mat-dir> [--out P.png --shape sphere|plane --size 512]    re-render the lit preview (NO GPU)
plakat texture export  <mat-dir> [--out DIR --engine gltf|unreal|unity-hdrp|godot|materialx|plakat | --naming plakat|unity|unreal --orm --gltf --materialx]   re-pack for an engine (NO weights)
plakat texture blend   <dirA> <dirB> --out DIR [--mask mix|radial|x|y|<mask.png> --naming plakat]   cross-fade two materials (NO weights)
plakat texture trim    <spec> --out DIR [--size N --naming plakat|unity|unreal]                    compose materials into a trim-sheet atlas + UV sidecar (NO weights)
plakat texture decal make  --out DIR [--image PNG --mask PNG --shape circle|ring|stripe|splatter|crack --threshold 0..1 --color r,g,b --size N]   build a decal (NO weights)
plakat texture decal apply <base> <decal> --out DIR [--at x,y --scale FRAC --rotate DEG --tile]    stamp a decal onto a base (RNM normal) (NO weights)
plakat texture render  <spec> --out DIR [--attempts N --variations N --keep-best --upscale none|2k|4k --metallic-ref M.png --roughness-ref R.png --engine gltf|unreal|unity-hdrp|godot|materialx|plakat]   generate the material (GPU)
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
R=AO/G=roughness/B=metallic image; `--gltf` emits a glTF 2.0 material; `--materialx` emits a MaterialX
`standard_surface` document; `--out` picks the destination. So one generated material can be re-packed
for Unity and Unreal from the same source with no GPU.

The 6.6 **`--engine <target>`** preset does all of that in one flag (naming + packing + material doc) —
see *Engine export* below. It is also on `render` and `derive` (`--engine`), so a freshly generated or
derived material can land engine-ready without a second `export` call.

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

## Engine export

**One flag, one target, everything correct.** `texture export <dir> --engine <target>` picks an engine's
**naming convention**, its **channel packing**, and its **material document** in a single shot. The same
preset is on `texture render` and `texture derive` (`--engine`), in the library as
`plakat::api::texture_export(dir, engine, out)` and `Texture::engine(...)`, and in Bund as
`plakat.texture.export`. Targets:

```
plakat texture export <dir> --engine gltf | unreal | unity-hdrp | godot | materialx | plakat [--out DIR]
```

`--engine` **overrides** `--naming` / `--orm` (it sets them for you). The manual flags still work when
you want hand control instead of the preset — `--naming plakat|unity|unreal --orm --gltf --materialx`.

### Why a preset — engines pack the same PBR data differently

The trap the preset exists for: two engines take **identical** roughness / metallic / AO data and pack
it into **different channels of a different image**. Feed one engine the other's packing and it *fails
silently* — the material just looks wrong, with no error. The two conventions:

| Convention | Targets | Image | R | G | B | A |
|---|---|---|---|---|---|---|
| **ORM** | `gltf` · `unreal` · `godot` | `orm.png` / `T_ORM.png` | ambient occlusion | roughness | metallic | — |
| **HDRP mask map** | `unity-hdrp` | `mask_map.png` (RGBA) | metallic | ambient occlusion | detail mask | smoothness (= 1 − roughness) |

The Unity HDRP mask map is **not** ORM: different channel order, a fourth **smoothness** channel that is
the *inverse* of roughness, and a detail mask in B. `--engine unity-hdrp` writes the mask map; every ORM
target writes ORM. Picking the wrong one is the classic silent failure this preset removes.

### What each target writes

| `--engine` | Naming | Packed map | Material document |
|---|---|---|---|
| `gltf` | plakat | `orm.png` (ORM) | `material.gltf` — a complete glTF 2.0 material |
| `unreal` | `T_*` (`T_BaseColor`, `T_Normal`, …) | `T_ORM.png` (ORM) | — |
| `unity-hdrp` | HDRP (`_MainTex`, `_BumpMap`, …) | `mask_map.png` (**mask map, not ORM**) | — |
| `godot` | plakat | `orm.png` (ORM) | — |
| `materialx` | plakat | `orm.png` (ORM) | `material.mtlx` — a MaterialX 1.38 `standard_surface` graph |
| `plakat` | plakat (raw `albedo.png`, `normal.png`, …) | `orm.png` (ORM) | — |

- **`gltf`** writes a complete **glTF 2.0** material (`material.gltf`): `baseColor` + `metallicRoughness`
  (reading roughness/metallic from ORM's **G/B**) + `occlusion` (ORM's **R**, with an occlusion
  *strength*) + `normal` (with a normal *scale*), plus the ORM image.
- **`materialx`** writes a **MaterialX 1.38** `standard_surface` node graph (`material.mtlx`) — the
  interchange format read by **USD / Arnold / Karma / Substance** — plus ORM.
- **`unreal`** / **`godot`** / **`plakat`** write correctly-named channels + ORM; **`unity-hdrp`** writes
  HDRP-named channels + the mask map.

### glTF anisotropy

When the material carries an **anisotropy** map (from the 6.4 anisotropy feature — a brushed/grained
metal), the `gltf` export emits the **`KHR_materials_anisotropy`** extension (declared in the glTF's
`extensionsUsed`), whose texture is the **anisotropy flow map**. So a brushed-metal material carries its
grain direction into glTF instead of losing it at the export boundary.

### Non-goals

plakat emits the **textures**, a **standard material document** (glTF 2.0 / MaterialX 1.38), and
**correctly-named, correctly-packed channels**. It does **not** write binary engine formats
(`.uasset` / `.tres` / `.usdz`) or **KTX2**-compressed textures — importing the exported material into
the engine is your step. The value is that what you import is already packed the way the engine expects.

## Trim sheets & decals

Two 6.5 additions that **compose finished materials** rather than generate new ones — so both are
**weight-free**. A **trim sheet** packs several sub-materials into one banded atlas; a **decal** is an
alpha-masked overlay you stamp onto a base. Neither touches a model.

### `trim` — a trim-sheet atlas *(no weights)*

A **trim sheet** is the standard way games texture pipes, panels, edges, and trims from *one*
material: several sub-materials are composed into a **single atlas** of stacked horizontal **bands**,
each band tiling along its run axis (**U**). A model maps different faces to different vertical slices
of the atlas, so a whole prop set shares one texture.

```
plakat texture trim <spec> --out DIR [--size N] [--naming plakat|unity|unreal]
```

`<spec>` is a `TrimSpec` HJSON document; `--out` is the atlas material directory; `--size` overrides
the atlas edge (px, square); `--naming` picks the channel-file idiom (`plakat` default | `unity` |
`unreal`), exactly as `export`.

**The `TrimSpec` schema.** Bands stack **top → bottom** and their heights **normalise to sum 1** (the
last band absorbs rounding, so heights fill the atlas exactly):

```hjson
{
  schema: "trim/1"
  size: 1024
  naming: "plakat"
  bands: [
    { material: "pipe_mat/",   height: 0.5,  tile: "x",    label: "pipe"   }
    { material: "rivets_mat/",  height: 0.25, tile: "x",    label: "rivets" }
    { material: "panel_mat/",   height: 0.25, tile: "none", label: "panel"  }
  ]
}
```

- **`material`** — a finished material directory (from `render` / `from` / `derive`).
- **`height`** — a fraction of the atlas; all bands **normalise to sum 1**.
- **`tile`** — `"x"` (default) repeats the sub-material **horizontally** along its run axis; `"none"`
  **stretches** it to fill the band; `"y"` is accepted for completeness.
- **`label`** — the band's name, carried into the UV sidecar.

**Output.** The full channel set (albedo · normal · roughness · metallic · height · AO) + packed ORM,
a **Plane** preview (the atlas tiles in **U**, not V — so a flat-plane preview reads it correctly where
the sphere would not), and a **`trim.json`** UV-region sidecar. `trim.json` lists each band as
`{ label, u0, v0, u1, v1, tile }` so an engine or DCC can map faces to the right band without guessing.
Because the bands are **pre-rendered material dirs**, `trim` is pure composition — no weights, no GPU.

### `decal make` / `decal apply` — alpha-masked overlays *(no weights)*

A **decal** is a material plus an **opacity mask** (white = opaque) that layers onto a base material —
a stencilled logo, a crack, a leak, a patch of grime. You **`make`** the decal once, then **`apply`**
it onto any base.

**`decal make` — build the overlay.**

```
plakat texture decal make --out DIR [--image PNG] [--mask PNG] [--shape KIND]
                          [--threshold 0..1] [--color r,g,b] [--size N]
```

- **Albedo** comes from `--image <png>`, or — with no `--image` — from a solid `--color r,g,b` (0–255,
  default `40,40,40`) for a **procedural** decal.
- **Opacity** is resolved by precedence: **`--mask`** PNG (white = opaque) **>** **`--shape`**
  (procedural: `circle` | `ring` | `stripe` | `splatter` | `crack`) **>** **`--threshold`** (remove
  pixels **brighter** than this luma `[0,1]` — a white-background cutout) **>** all-opaque.
- `--size` sets the decal edge (px, square, default `512`). Output is a decal directory: the channel
  set + an `opacity.png`.

**A procedural decal takes its relief from the *shape*, an image decal from its *albedo*.** This is the
one nuance worth internalising. A procedural decal has a **flat solid-colour** albedo, which carries no
normal detail — so its **relief is derived from the shape** (opacity → height). This is *correct*, not
a workaround: RNM of a flat normal returns the base unchanged, so a flat-albedo decal with no shape
relief would leave no mark on the base normal. An **image** decal instead derives its relief from the
**albedo** (the image *is* the detail). So: give a procedural decal a `--shape` if you want it to read
in the normal; an image decal already carries its own.

**`decal apply` — stamp it onto a base.**

```
plakat texture decal apply <base> <decal> --out DIR [--at x,y] [--scale FRAC] [--rotate DEG] [--tile]
```

`<base>` is the base material directory, `<decal>` the decal directory (it needs `opacity.png`).
`decal apply` **alpha-blends** albedo / roughness / metallic / height weighted by the decal's opacity,
blends the **normal** via **Reoriented Normal Mapping (RNM)**, then **re-derives AO** from the new
height. The base is **preserved wherever the decal is transparent**.

- **`--at x,y`** — the decal's **normalised** centre on the base (default `0.5,0.5`).
- **`--scale FRAC`** — decal size as a **fraction of the base edge** (default `0.5`).
- **`--rotate DEG`** — decal rotation in degrees (default `0`).
- **`--tile`** — repeat the decal across the **whole** base (a repeating detail) instead of a single
  stamp.

**Why RNM, not a lerp.** A naive normal lerp between base and decal **flattens both** — it washes out
the base's surface tilt *and* the decal's own detail. **Reoriented Normal Mapping** instead reorients
the decal's detail so it **rides the base surface**: a decal stamped on a curved or tilted material
sits on the surface correctly instead of punching a flat patch through it. (This was validated in the
project's G0 probe.) No weights, no GPU.

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
    mode: "circular"     # circular (default) | offset | auto | mirror | none
    axes: "both"         # both (default) | x | y   — x/y = a trim sheet tiling one axis
  }
  # circular/offset — frequency-aware feather (6.19): matches the low-frequency tone across the seam with
  #   a smoothstep ramp while preserving all high-frequency detail (no blur band). auto (6.19) — measure
  #   the raw seam and pick the feather band, falling back to mirror when a hard seam would only smear.
  #   mirror — reflect so opposite edges are identical by construction (perfectly seamless, mirror pattern).

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

**Frequency-aware feather (6.19).** The feather no longer cross-fades raw pixels (which blurred detail
and could leave a tonal ramp). It now estimates the *low-frequency tone* on each edge and adds a
**smoothstep-decaying, half-magnitude offset** so the two edges' tone **meets** at the seam, while every
high frequency — the actual texture detail — is left untouched. A `seam_score` (cross-boundary jump ÷
interior baseline) makes the residual measurable. New **`mode: "mirror"`** reflects the tile so opposite
edges are identical by construction — a perfectly seamless boundary (at the cost of a mirror-symmetric
pattern), good for organic/fabric.

**Pigment-aware normal-from-photo (6.19).** For the image-to-material path, height is now estimated with a
**chroma gate**: coloured pigment detail (a red speck) is suppressed so it doesn't become fake geometry,
while neutral micro-relief is kept — better normals from photos.

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
