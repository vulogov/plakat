# `plakat product` — a subject → a studio product-shot / packshot

`product` turns a **subject** — a cutout PNG, a photo, or a text prompt — into a clean, repeatable
**studio product-shot**: the subject on a controlled background (white / grey sweep / gradient / a
generated scene), grounded with a physically-plausible **contact shadow** and **floor reflection**, at a
chosen camera angle, optionally relit to a named **lighting rig**. It is the plakat 6.9 flagship (RFC
[`RFC_PRODUCT_1.md`](RFC_PRODUCT_1.md)), and it is **fully additive** — no existing command or output
changes.

Like every plakat studio, a packshot is **structured data**, not a prompt. A product-shot is about two
things a prompt can't deliver: **repeatability** (the *same* lighting and grounding across a whole catalog
of different products) and **physical plausibility** (a subject that sits on the ground with a real
contact shadow, on a background that is actually pure white). So the shot is authored in a small HJSON
document (a `ProductSpec`) and resolved deterministically: **subject → light → ground → composite**.

**The weight-free half needs no GPU.** The background sweep, the contact shadow, the floor reflection, and
the composite are pure CPU — derived from the subject's **alpha matte**. Only two steps need a model:
**relighting** (IC-Light) and **generating** a subject/scene. With a supplied cutout and relight off, a
whole catalog renders with no GPU.

## The pipeline

```
ProductSpec ─▶ subject (cutout | photo→matte | prompt→gen→matte)
            ─▶ [relight to the rig]              (IC-Light — optional, model)
            ─▶ ground (contact shadow + reflection from the alpha)   ← the novel weight-free algorithm
            ─▶ composite onto the sweep / scene ─▶ shot.png + shot.meta.json
```

## The `ProductSpec`

Permissive HJSON — every field optional, unknown keys ignored, enums as strings (lint warns). `product
new` scaffolds one.

```hjson
{
  schema: "product/1"

  subject: {
    image:  "sneaker.png"          // a cutout (transparent) — kept pixel-exact (no VAE roundtrip)
    // or  photo:  "sneaker.jpg"    // matte it (U2Net) first
    // or  prompt: "a red running sneaker"   // generate → matte
    scale:  0.7                     // fraction of the canvas height the product fills
    anchor: "bottom"                // bottom (grounded, default) | center
  }

  canvas:  { size: "square", px: 1024, bg: "grey-sweep" }   // white | grey-sweep | gradient:top,bottom | scene
  scene:   { prompt: "a marble counter, soft daylight" }    // only when bg: "scene"

  lighting: { rig: "three-point", key_dir: "top-left", warmth: 0.0 }   // relight rig (opt-in)
  camera:   { angle: "eye" }                                            // eye | hero | top | three-quarter
  ground:   { shadow: "soft", reflection: "gloss", softness: 0.5, falloff: 0.6 }

  variants: [ { image: "sneaker_side.png", label: "side" } ]   // a catalog (product sheet)

  model: "sdxl"  seed: 7  steps: 30
}
```

## The novel piece — grounding

Placing a subject so it *sits on the ground* instead of floating is what separates a packshot from a
cut-out on a white square, and it's the part of `product` that is genuinely new. From the subject's
**alpha** alone:

- **Contact shadow** — the alpha projected to the ground plane (offset away from the key light,
  foreshortened by height, higher parts fainter), **clamped to the floor** so the soft-blur penumbra can't
  bleed a halo above the contact line, then softened. `shadow: "soft"` is a symmetric pool; `shadow:
  "hard"` is a directional cast that rakes away from the light.
- **Floor reflection** — the subject flipped about the foot-line, foreshortened by the camera angle, fading
  with distance. `reflection: "gloss"` (dim) | `"mirror"` (bright) | `"none"`.

All weight-free, from the matte — no GPU.

## Commands

```
plakat product new    <out.hjson>                                   # scaffold a spec
plakat product lint   <spec.hjson>                                  # schema / vocab / a subject source
plakat product show   <spec.hjson>                                  # resolved canvas / camera / ground
plakat product render <spec.hjson> --out shot.png [--subject <img>] [--relight] [--no-relight] [--device auto]
plakat product sheet  <spec.hjson> --out sheet.png                  # catalog contact sheet (main + variants)
plakat product turntable <spec.hjson> --out sheet.png --frames 5    # sweep the key light (relit each)
```

With a supplied cutout and relight off, `render`/`sheet` are **weight-free**.

## Subject sources & relight

- **Cutout** (`subject.image` / `--subject`) — used as-is, composited pixel-exact (logos and labels never
  distort). Weight-free.
- **Photo** (`subject.photo`) — matted (U2Net) into a cutout. Needs a model.
- **Prompt** (`subject.prompt`) — generated (`api::Generate`) then matted. Needs a model.
- **Relight** — `--relight` or a `lighting:` block re-illuminates the subject to the rig via IC-Light
  (`--no-relight` forces off). **Note:** IC-Light relights *and* recolors — set `warmth: 0` / a neutral
  `lighting.prompt` (or skip relight) to keep the product's true hue.

## Catalog

- **`product sheet`** — the main subject + each `variants[]` angle, rendered with the **same rig/ground**,
  tiled into a labelled contact sheet. Weight-free with cutouts.
- **`product turntable --frames N`** — one subject, the key light swept across N directions (relit each),
  tiled. Needs the relight model.

## Integration

- **Library**: [`plakat::api::Product`](API.md) — `Product::load(spec).subject(img).run("shot.png")`
  (+ `.sheet()` / `.turntable()`).
- **Scenario**: a `type: "product"` task in an HJSON batch.
- **Compile**: `plakat compile` turns prose ("a red sneaker") into a `type: product` task whose subject is
  that prompt.
- **Bund**: `plakat.product.*` (render / sheet).
- **Doctor**: `plakat doctor` reports the product capability.

## Limits (honest)

- **No 3D novel-view.** plakat has no NeRF/multi-view — the turntable rotates the *light*, not the object.
  For a real multi-angle catalog, supply angle cutouts via `variants`.
- **Grounding is a plausible approximation**, not a ray-traced render (a soft alpha-projected shadow +
  a flipped reflection).
- **Relight recolors.** Use `warmth: 0` / a neutral prompt, or a cutout with relight off, to keep the
  product hue exact.
