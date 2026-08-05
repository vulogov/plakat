# `plakat texture` tutorial

A hands-on pass through the whole pipeline: scaffold a spec and see what it resolves to, **derive** a
full PBR set from an albedo you already have (no GPU), **verify / preview / export** it for an engine,
then **render** a material from a prompt, turn a **photo** into one, **upscale** to 2K, and finish with
the integration surfaces. `texture` turns a prompt or a photo into a *seamless, tileable* PBR material
set — the reference is [`../TEXTURE.md`](../TEXTURE.md); the full design is
[`../RFC_TEXTURE_1.md`](../RFC_TEXTURE_1.md).

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

## Where to go next

- [`../TEXTURE.md`](../TEXTURE.md) — the command + schema reference, the output contract, the concepts,
  the scorecard, the integration surfaces, and the honest scope.
- [`../RFC_TEXTURE_1.md`](../RFC_TEXTURE_1.md) — the full design: circular convolution, the measure-first
  seamless approach, delighting, and the escalation path.
