# Outpainting — extend an image past its borders

`plakat outpaint INPUT.png` grows the canvas of an existing image
and fills the new region with content that continues the original
scene. The classic use case: you have a square photo and want a
landscape banner; outpaint adds new sky on top + new ground below.

Mechanically, outpaint is **inpaint with a generated canvas + a
new-region mask**:

1. Allocate a bigger canvas (`new_w`, `new_h`).
2. Copy the input into the canvas at the correct offset.
3. Replicate the input's edge pixels into the new border (smooth
   low-frequency continuation, easier for the inpaint UNet to
   refine than a flat gray slab).
4. Build a mask where the new region is white and the original
   image area is black.
5. Hand off to `plakat img2img --mask` with the dedicated inpaint
   model.

This tutorial covers the per-side flag grammar, dimension snapping,
model choice, and composition with other plakat features.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md) +
  [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md). Outpaint is a thin
  wrapper around the inpaint flow; understanding inpaint mechanics
  helps when results don't quite match expectations.
- An input image to extend.

## 1. The simplest outpaint

```bash
plakat outpaint ./input.png \
    --prompt "a wide panoramic mountain valley, dramatic clouds" \
    --left 512 --right 512
```

That pads the input by 512 pixels on each side (1024 pixels wider
total). The new region gets filled with content guided by the
prompt + the replicated edge pixels.

Output: `./out/plakat-inpaint-<seed>.png` at `(orig_w + 1024,
orig_h)`. The inpaint model never re-paints the original area —
the mask is black there.

## 2. Per-side flags

Four flags control padding independently:

| Flag | Padding direction |
|---|---|
| `--left N` | adds N pixels of new canvas on the left |
| `--right N` | adds N pixels on the right |
| `--top N` | adds N pixels above |
| `--bottom N` | adds N pixels below |

Or use `--expand N` as a shorthand for `--left N --right N --top N
--bottom N`:

```bash
# All four sides by 256 pixels (input becomes a centered insert
# in a larger canvas).
plakat outpaint ./photo.png --expand 256 \
    --prompt "expanded landscape, dramatic sky"
```

Combine for asymmetric expansion:

```bash
# Just add sky on top; preserve the bottom of the input.
plakat outpaint ./skyline.png --top 400 \
    --prompt "dramatic stormclouds rolling in from above"
```

## 3. Dimension snapping

Outpaint snaps padding to the model's VAE / patch constraint
automatically:

- SD 1.5 / SD 2.1 / SDXL → multiples of 8
- Flux Fill → multiples of 16

So `--left 510` on SD 1.5 becomes `--left 504` internally (510 →
504, the nearest multiple of 8 below). The final canvas dims are
guaranteed to satisfy the model's input constraints.

A diagnostic line at startup reports the actual padding applied:

```
  outpaint: input 1024x768 + (504, 0, 0, 0) → 1528x768
```

## 4. Model choice

The default outpaint model is `sdxl-inpaint`. Two other supported
inpaint backbones:

| Model | When to use |
|---|---|
| `sdxl-inpaint` (default) | SDXL-trained inpaint checkpoint. ~24 GB BF16, balanced quality. |
| `sd15-inpaint` | SD 1.5-trained. ~4 GB. Use on a 6 GB GPU or when matching an existing SD 1.5 workflow. |
| `flux-fill-dev` | BFL's Flux inpaint variant. ~24 GB BF16 (or quantized via `flux-fill-dev-gguf`). The new-region completions tend to be more coherent at large outpaint distances. |

```bash
plakat outpaint photo.png --left 256 \
    --prompt "..." --model flux-fill-dev
```

Note: outpaint doesn't yet route through Flux Kontext — Kontext is
an editing model, not a region-completion model. Use Flux Fill for
Flux-quality outpaint.

## 5. Edge-replicate padding (what plakat does for you)

Before handing off to the inpaint model, plakat fills the new
region with **replicated edge pixels** from the original input.
A 200-pixel `--right` pad gets the rightmost column of the input
copied 200 times into the new region.

Why: a flat gray (or zero) starting state forces the inpaint UNet
to invent everything from noise + prompt. With replicated edges,
the new region starts as a smooth low-frequency continuation that
the UNet refines rather than invents. Output quality at large
outpaint distances improves noticeably.

You don't configure this — it's the default behavior. The
`--strength 1.0` pin (also automatic) tells the inpaint UNet to
treat the new region as fully open for repainting; the replicate
content is just an initialization hint.

## 6. Composition

### With `--prompt`

The prompt describes the desired new content. **It applies to the
ENTIRE canvas**, not just the new region — the inpaint UNet
re-attends to the original-area pixels each step but never updates
them (mask is black there). So:

```bash
# Bad: prompt mentions only the new region
plakat outpaint forest.png --top 400 --prompt "stormclouds"

# Better: prompt describes the whole intended scene
plakat outpaint forest.png --top 400 \
    --prompt "a dense pine forest under dramatic stormclouds"
```

### With `--seed`

Outpaint outputs are seeded same as t2i — fix `--seed N` for
reproducibility:

```bash
plakat outpaint photo.png --expand 256 --seed 42 \
    --prompt "..."
```

### With `--count N`

Generate multiple variants of the outpaint:

```bash
plakat outpaint photo.png --left 256 --count 4 \
    --prompt "..."
# → plakat-inpaint-<seed>.png, plakat-inpaint-<seed+1>.png, ...
```

Composes with `--grid` (v0.18) to bundle into a single comparison
PNG.

### With `--lora`

Inpaint LoRAs work the same way as t2i LoRAs:

```bash
plakat outpaint photo.png --left 256 \
    --prompt "..." \
    --lora civitai:99999:0.7 \
    --model sdxl-inpaint
```

## 7. Worked examples

### Square → letterbox

```bash
plakat outpaint portrait_1024.png \
    --left 512 --right 512 \
    --prompt "a wide cinematic landscape, the subject at center"
# 1024x1024 → 2048x1024 letterbox
```

### Add sky to a horizon-cropped photo

```bash
plakat outpaint horizon.png --top 600 \
    --prompt "the same scene with a dramatic sunrise sky filling the upper half"
# 1024x576 → 1024x1176
```

### Full surround (centered insert)

```bash
plakat outpaint subject.png --expand 384 --model flux-fill-dev \
    --prompt "the subject is a small detail in a vast misty mountain valley"
# 512x512 → 1280x1280, original is centered
```

### Sequence: iterative outpaints

Outpainting in stages can produce more coherent large extensions:

```bash
# Stage 1: extend horizontally
plakat outpaint photo.png --left 256 --right 256 \
    --prompt "..." --out ./stage1 --seed 100

# Stage 2: extend vertically from stage 1's output
plakat outpaint ./stage1/plakat-inpaint-100.png \
    --top 256 --bottom 256 --prompt "..." --out ./stage2 --seed 100
```

Each stage paints into a smaller new region than a single-stage
4-side expand, often yielding better continuity at the corners.

## 8. Limitations

- **No Kontext routing yet** — outpaint always uses an inpaint
  model (`sdxl-inpaint` / `sd15-inpaint` / `flux-fill-dev`).
  Kontext-based outpaint would need different conditioning
  geometry; deferred.
- **No `--aspect` on outpaint** — outpaint controls dimensions via
  the per-side flags, so `--aspect` would be redundant. Use
  `--left` + `--right` to hit a target aspect ratio.
- **No `--tiled`** — outpaint runs full-canvas img2img. Tiled
  outpaint composes with the per-tile blend, but the existing
  outpaint wrapper doesn't thread `--tiled` through. Workaround:
  small outpaints first, then `plakat upscale` or `plakat img2img
  --tiled` for the final resolution.
- **Stage-2 outpaints inherit stage-1 artifacts**. If stage 1
  produces visible seams, iterating won't fix them. Re-run with a
  different `--seed` or switch models.

## Where to next

- [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md) — the underlying
  inpaint mechanics. `--mask` / `--strength` / `--mask-feather`
  apply to outpaint internally even though the outpaint CLI
  doesn't surface all three.
- [`FLUX_TUTORIAL.md`](FLUX_TUTORIAL.md) §7 — Flux Fill for
  outpaint via `--model flux-fill-dev`.
- [`GENERATE.md`](../GENERATE.md) — `plakat outpaint` flag
  reference.
