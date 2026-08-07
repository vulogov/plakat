# `plakat texture` tutorial

A hands-on pass through the whole pipeline: scaffold a spec and see what it resolves to, **derive** a
full PBR set from an albedo you already have (no GPU), **verify / preview / export** it for an engine,
then **render** a material from a prompt, turn a **photo** into one, **upscale** to 2K, and finish with
the integration surfaces. `texture` turns a prompt or a photo into a *seamless, tileable* PBR material
set — the reference is [`../TEXTURE.md`](../TEXTURE.md); the full design is
[`../RFC_TEXTURE_1.md`](../RFC_TEXTURE_1.md).

**New in 6.4** (section 8 below): spatially-varying `metallic: "auto"` / `roughness: "auto"` for
composite materials, hand-painted `--metallic-ref` / `--roughness-ref` masks, `--anisotropy` grain maps,
`texture blend` to cross-fade two materials, and `render --variations N` for seed spreads.

**New in 6.5** (section 9 below): `texture trim` composes finished materials into a banded trim-sheet
atlas + a UV sidecar, and `texture decal make` / `decal apply` build an alpha-masked overlay and stamp
it onto a base material (RNM normal blend). Both are weight-free.

**New in 6.6** (section 10 below): `--engine <target>` — a one-flag engine-export preset (glTF / Unreal /
Unity HDRP / Godot / MaterialX / plakat) that sets naming + channel packing + material document together,
so a material lands correctly in each engine (ORM vs the HDRP mask map). Weight-free; also on `render`
and `derive`.

Build the release binary first — debug diffusion is ~50× slower:

```sh
cargo build --release
alias plakat=./target/release/plakat
export PLAKAT_OOM_GUARD_GB=0        # the macOS free-page guard mis-fires under render loops
```

**Most of this tutorial needs no GPU and no weights.** The whole authoring/derive/measure/pack path —
`new` · `lint` · `show` · `derive` · `verify` · `preview` · `export` — runs anywhere. Only `render`
(step 4) needs a diffusion model, and `from` (step 5) touches only a depth model, and only for
`height: auto`. Each step below is labelled.

## 1. Author a spec (no weights)

Scaffold, then edit:

```sh
plakat texture new iron.hjson --material "worn rusted iron plating, industrial" --size 1024 --model sdxl
```

That writes a valid partial `TextureSpec`. Open it — every field is optional, so change the `seamless`
mode, the `channels` (a flat `roughness: 0.6` vs a heuristic `roughness: "from-albedo"`), `delight`, or
the `export` block to taste. Validate whenever you like:

```sh
plakat texture lint iron.hjson       # schema · enum vocab · numeric ranges · material-vs-from_image
```

`lint` exits non-zero on errors, so it can gate CI.

```sh
plakat texture show iron.hjson       # the resolved plan — no rendering
```

`show` prints what the spec resolves to: text-to-material or image-to-material, the seamless mode +
axes, how each channel resolves (scalar / `from-albedo` / a prompt), whether delighting is on, the page
size + upscale target, and the export map list + naming + ORM/glTF/preview flags.

## 2. Derive a full PBR set from an albedo (no GPU)

You don't need to generate anything to get a material — if you already have a base-colour image (a
generated albedo, a flat scan, a hand-painted tile), `derive` builds the whole channel set from it on
the CPU:

```sh
plakat texture derive myalbedo.png --out iron_mat/ --normal-strength 1.0 --ao-strength 1.0 --normal-y opengl
```

Out comes a full material directory: `albedo.png`, `normal.png`, `roughness.png`, `metallic.png`,
`height.png`, `ao.png`, a packed `orm.png`, a `preview.png`, and a `material.json` (recipe + channel
manifest + scorecard + a stable spec-hash). Height here is derived from albedo luminance; supply your
own with `--height H.png` if you have a real height map. `--normal-strength` / `--ao-strength` scale the
relief and cavity; `--normal-y directx` flips the green channel for DirectX engines. Every derivation is
**circular**, so the derived maps tile as long as the albedo does.

## 3. Verify, preview, and export for an engine (no weights)

Measure the material against what a good PBR set should be:

```sh
plakat texture verify iron_mat/
```

It reports **tileability-x** and **tileability-y** (edge-wrap join vs interior — ~1 = seamless, > 1.5 =
a seam), **normal-validity** (unit vectors, +Z), **channel-consistency**, and **albedo-flatness** (a
de-lit proxy). The **hard gate** is tiling + normal + consistency; **flatness is advisory**. If it
clears the hard gate, the material is shippable.

Re-render the lit preview at any shape/size (a sanity view — Cook-Torrance-lite, one light — not a
renderer):

```sh
plakat texture preview iron_mat/ --shape sphere --size 512 --out iron_preview.png
plakat texture preview iron_mat/ --shape plane                       # see it as a flat tile instead
```

Now pack it for an engine — no regeneration, just rename + re-pack:

```sh
plakat texture export iron_mat/ --out iron_unreal/ --naming unreal --orm --gltf
```

`--naming unreal` writes `T_BaseColor.png`, `T_ORM.png`, …; `--orm` (re)writes the packed
R=AO/G=roughness/B=metallic image; `--gltf` emits a glTF 2.0 material. Swap `--naming unity` for Unity's
`normal_BumpMap.png` idiom, or the default `plakat` for plain `albedo.png…`. One material, re-packed for
any engine from the same source.

## 4. Render a material from a prompt (weights)

The full text-to-material path — the one step that needs a diffusion model. It generates the albedo,
upscales (optional), derives the whole channel set, measures the scorecard, and packs the directory:

```sh
plakat texture render iron.hjson --out iron_gen/ --attempts 4
```

The generation is **seamless-aware**: the flat/tileable prompt keeps the field low-frequency and a
boundary **feather** makes it wrap, so the albedo tiles without a Real-ESRGAN pass. `--attempts 4` turns
on **rejection sampling** — it generates up to four seeds and keeps the first that clears the scorecard's
hard gate (else the fewest-issues one). Generation is the one stochastic step; it is seed-locked
(`seed` in the spec) and reproducible on a given device. Verify and preview it exactly as in step 3:

```sh
plakat texture verify iron_gen/
plakat texture preview iron_gen/ --out iron_gen_preview.png
```

## 5. Turn a photo into a material (weights: depth only)

`from` makes a **material out of a photo**. It runs **no generation** — the photo *is* the albedo — so
it is far cheaper than `render`. It makes the photo tileable (offset-and-heal: roll by half so the edge
seams move to the interior, then feather the central cross), then derives the channel set:

```sh
plakat texture from cobble.jpg --out cobble_mat/ --material "cobblestone street" --height auto
```

With `--height auto` the only model it touches is **Depth-Anything-V2** — depth-from-albedo gives the
macro relief, combined with a luminance high-pass for crisp micro-detail. Want it fully weight-free?
Use luminance height instead:

```sh
plakat texture from cobble.jpg --out cobble_mat/ --height from-albedo       # no models at all
```

Note the offset-heal softens a band at the moved seam — that is expected (a generative inpaint would be
cleaner; it is a documented fast-follow). Verify to see the tiling score.

## 6. Upscale to 2K (weights or not, depending on step 4/5)

Both `render` and `from` can upscale **before** derivation, so the whole channel set comes out at the
higher resolution:

```sh
plakat texture render iron.hjson --out iron_2k/ --upscale 2k
plakat texture from   cobble.jpg --out cobble_4k/ --upscale 4k --height from-albedo
```

The upscale is a **tileability-preserving Lanczos** (circular-pad → resize → crop) — Real-ESRGAN is
deliberately avoided because it tiles internally and hallucinates, both of which break the wrap. Tiling
was verified to survive the upscale, so `verify` on the 2K/4K directory should still clear the tiling
gate.

## 7. The integration surfaces (6.3)

`texture` is not only a subcommand — the same render core drives every automation surface:

```yaml
# scenario — a type: texture task (inline spec or a spec_file), batched:
- type: texture
  spec_file: iron.hjson
  seed: 7
  upscale: 2k
  attempts: 4
```

```
# compile — a type: texture block; the prose is the material prompt, directives tune it:
texture-from   photo.jpg
texture-size   1024
texture-upscale 2k
texture-seamless both
texture-height auto
```

```
# Bund — push the lit preview as an image handle into the save/metadata/upscale pipeline:
plakat.texture.render   plakat.texture.from   plakat.texture.preview
```

```rust
// library API — plakat::api::Texture, mirroring Generate / Portrait / BookArt:
let card = plakat::api::Texture::from_prompt("worn rusted iron plating")
    .model("sdxl").size(1024).seed(7).steps(28)
    .upscale("2k").attempts(4)
    .run("iron_mat/")?;          // -> Scorecard
```

`from_prompt` / `from_image` / `from_spec` pick the source; `model` / `size` / `seed` / `steps` /
`upscale` / `attempts` tune it; `run(out)` writes the material directory and hands back the `Scorecard`
you'd otherwise read with `verify`.

## 8. Composite materials, the metallic channel, and the new 6.4 knobs

### "Why is my metallic map black?" — it's correct

The most common surprise is a **flat black metallic map**, read as a bug. It almost never is. **Metallic
is near-binary per material**: a surface is *either* a raw **metal** (metallic `1.0`, white) *or* a
**dielectric** — stone, wood, leaves, plastic, wet stone, paper (metallic `0.0`, black). Almost nothing
sits in between. So for a **single-class** material a flat map is the **right answer**:

- **flat black** for a dielectric — stone, leaves, a river, concrete, wood;
- **flat white** for a conductor — brushed steel, gold, copper.

`verify` says so out loud: for a single-class tile it prints *"metallic is uniform (black = dielectric) —
correct for a single-class material, not a defect"*. A metallic map only carries spatial **structure**
when the tile is a **composite** that mixes metal and non-metal in one image.

### The composite case — rusted iron with `metallic: "auto"`

Rusted iron is the textbook composite: **bare steel** (a conductor) and **rust** (a dielectric) share
one tile. Here `metallic: "auto"` (the default) earns its keep — it runs a **spatially-coherent region
vote** and returns a **structured** mask (bare-metal regions white, rust black) where the old per-pixel
`from-albedo` scattered speckle:

```sh
plakat texture derive rusted_iron.png --out rust_mat/ --metallic auto --roughness auto
plakat texture verify rust_mat/        # → "metallic is structured … composite material"
```

Open `rust_mat/metallic.png`: you'll see clean metal-vs-rust regions, not noise. For a **single-class**
tile `auto` correctly **collapses to a flat map**, so it's safe to leave on — but when you *know* the
class, say so, because metal-vs-dielectric is separated by **saturation** and a grey dielectric (stone,
concrete) can read close to bare metal:

```sh
plakat texture derive stone.png  --out stone_mat/  --metallic 0    # known dielectric → flat black
plakat texture derive gold.png   --out gold_mat/   --metallic 1    # known raw metal → flat white
```

The ultimate override is a **hand-painted mask**, used verbatim (white = metal), on `derive`, `render`,
or `from`:

```sh
plakat texture derive rusted_iron.png --out rust_mat/ --metallic-ref my_metal_mask.png
```

### Anisotropy — brushed metal grain

For a brushed or grained metal, `--anisotropy` writes an `anisotropy.png` flow map (RG = grain
direction, B = strength) and makes the lit preview's highlight **stretch along the grain**. Give the
angle or omit it to auto-detect from the height's structure tensor:

```sh
plakat texture derive brushed_steel.png --out steel_mat/ --metallic 1 --anisotropy 0.85 --anisotropy-angle 0
plakat texture preview steel_mat/                    # the highlight is now a streak, not a dot
```

### `blend` — cross-fade two materials (no weights)

Blend two finished material directories into one PBR set — stone → moss, clean → worn. Every channel
lerps by the same mask and the normal is renormalised:

```sh
plakat texture blend stone_mat/ moss_mat/ --out mossy_stone/                 # mix (default) — still tiles
plakat texture blend stone_mat/ moss_mat/ --out mossy_stone/ --mask radial   # radial — also tiles
plakat texture blend clean_mat/ worn_mat/ --out wipe/ --mask x               # a transition sheet (breaks X tiling on purpose)
plakat texture blend stone_mat/ moss_mat/ --out mossy_stone/ --mask patches.png   # your own grayscale mask
```

`mix` and `radial` keep the result **tileable**; `x` / `y` are intentional transition sheets that break
tiling in that axis. No weights, no GPU.

### `render --variations` — a spread to choose from

`--attempts N` rejection-samples down to **one** passing material; `--variations N` instead renders N
distinct seed **variants side-by-side** into `<out>/var-0/`, `var-1/`, … so you can pick. Add
`--keep-best` to also copy the top-scoring variant to `<out>` itself:

```sh
plakat texture render iron.hjson --out iron_spread/ --variations 4 --keep-best
```

## 9. Trim sheets & decals (6.5, no weights)

Both of these **compose materials you already have** — no generation, no GPU.

### Compose a trim sheet from two materials

A **trim sheet** stacks several finished materials into ONE banded atlas, each band tiling
horizontally (along **U**) — the way games texture pipes, panels, and edges from a single material.
Write a tiny `TrimSpec`:

```hjson
# pipes.hjson
{
  schema: "trim/1"
  size: 1024
  bands: [
    { material: "pipe_mat/",  height: 0.6, tile: "x",    label: "pipe"  }
    { material: "panel_mat/",  height: 0.4, tile: "none", label: "panel" }
  ]
}
```

Bands stack top → bottom and their heights **normalise to sum 1** (the last band absorbs rounding).
`tile: "x"` repeats the sub-material across the band; `tile: "none"` stretches it. Compose:

```sh
plakat texture trim pipes.hjson --out pipe_trim/ --size 1024
```

Out comes a full material directory (albedo/normal/roughness/metallic/height/AO + ORM), a **Plane**
preview (the atlas tiles in U, not V — so a plane reads it right where a sphere wouldn't), and a
**`trim.json`** UV sidecar listing each band's `{ label, u0, v0, u1, v1, tile }` so an engine maps
faces to the right slice. `--naming unity|unreal` packs the atlas in an engine idiom, same as `export`.

### Make a crack decal and apply it onto a base

A **decal** is a material + an **opacity mask** (white = opaque) you stamp onto a base. Build a
procedural `crack` decal — no image, just a solid colour and a shape:

```sh
plakat texture decal make --out crack_decal/ --shape crack --color 20,20,20 --size 512
```

Opacity precedence is `--mask` PNG > `--shape` > `--threshold` > all-opaque; here the `crack` shape
supplies it. **Why the shape matters for relief:** a procedural decal has a **flat solid-colour**
albedo, which carries no normal detail — so its relief is derived from the **shape** (opacity → height).
That is correct, not a workaround: RNM of a flat normal returns the base unchanged, so without a shape a
solid-colour decal would leave no mark on the base normal. (An **image** decal instead takes its relief
from its albedo — the image *is* the detail — so it needs no shape.)

Now stamp it onto a base material:

```sh
plakat texture decal apply stone_mat/ crack_decal/ --out cracked_stone/ --at 0.5,0.5 --scale 0.6 --rotate 15
```

`decal apply` alpha-blends albedo/roughness/metallic/height, blends the normal via **Reoriented Normal
Mapping (RNM)**, and re-derives AO from the new height — the base is preserved wherever the decal is
transparent. `--at` is the normalised centre, `--scale` a fraction of the base edge, `--rotate` the
angle; add `--tile` to repeat the decal across the whole base instead of a single stamp. RNM (not a
naive lerp) is what lets the crack **ride** a curved or tilted surface instead of punching a flat patch
through it.

## 10. Export one material to several engines (6.6, no weights)

Step 3 packed a material with the manual `--naming` / `--orm` / `--gltf` flags. 6.6 adds a **one-flag
engine preset**: `--engine <target>` sets the naming convention **and** the channel packing **and** the
material document together, so the material lands correctly in each engine without you remembering which
is which. Take the `iron_mat/` from step 2 and pack it two ways:

```sh
plakat texture export iron_mat/ --out iron_unreal/ --engine unreal        # T_* names + ORM (R=AO, G=roughness, B=metallic)
plakat texture export iron_mat/ --out iron_hdrp/   --engine unity-hdrp    # HDRP names + a mask_map.png
```

**The one thing to notice** — those two outputs pack the *same* roughness/metallic/AO data
**differently**, and mixing them up fails silently in-engine:

- `--engine unreal` (and `gltf` / `godot`) writes **ORM**: `R = ambient occlusion`, `G = roughness`,
  `B = metallic`.
- `--engine unity-hdrp` writes a **mask map** (`mask_map.png`), which is **not** ORM: `R = metallic`,
  `G = ambient occlusion`, `B = detail mask`, `A = smoothness` — and smoothness is `1 − roughness`, the
  *inverse* of the roughness you'd feed ORM. Hand Unity an ORM image and it reads the channels wrong
  with no error; the preset is what keeps you out of that trap.

The other targets round out the set:

```sh
plakat texture export iron_mat/ --out iron_gltf/  --engine gltf        # + material.gltf (glTF 2.0 material)
plakat texture export iron_mat/ --out iron_mtlx/  --engine materialx   # + material.mtlx (MaterialX standard_surface, for USD/Arnold/Substance)
plakat texture export iron_mat/ --out iron_godot/ --engine godot       # ORM, Godot-ready
```

`--engine` also rides on `render` and `derive`, so a freshly generated or derived material can come out
engine-ready in one call — no second `export`:

```sh
plakat texture derive myalbedo.png --out iron_mat/ --engine unreal
plakat texture render iron.hjson   --out iron_gen/ --engine gltf
```

If the material has an **anisotropy** map (section 8), the `gltf` export also emits the
`KHR_materials_anisotropy` extension so a brushed metal keeps its grain in glTF. And the manual flags are
still there (`--naming plakat|unity|unreal --orm --gltf --materialx`) when you want to pack by hand
instead of by preset. Note plakat emits the textures + a standard material doc + correctly-packed
channels — **not** binary engine formats (`.uasset` / `.tres` / `.usdz`) or KTX2 compression; importing
into the engine is your step.

## Where to go next

- [`../TEXTURE.md`](../TEXTURE.md) — the command + schema reference, the output contract, the concepts,
  the scorecard, the integration surfaces, and the honest scope.
- [`../RFC_TEXTURE_1.md`](../RFC_TEXTURE_1.md) — the full design: circular convolution, the measure-first
  seamless approach, delighting, and the escalation path.
