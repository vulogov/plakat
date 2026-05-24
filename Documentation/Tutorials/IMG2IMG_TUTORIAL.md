# `plakat img2img` tutorial

`plakat img2img` transforms an existing image — either the whole
image (img2img mode) or just a region you mark with a mask (inpaint
mode). Same subcommand, two modes.

This tutorial covers when to use which mode, the strength dial, and
how to think about masks. For the runnable companion, see
[`examples/tutorials/IMG2IMG/`](../../examples/tutorials/IMG2IMG/).

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You should
  be comfortable with `--prompt`, `--seed`, `--steps`, `--guidance`,
  and the idea that diffusion models start from noise and denoise
  toward an image.
- A `plakat` binary with GPU support
  ([`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md) lists tiers).
  CPU works but is uncomfortably slow for img2img.

## 1. What img2img is

`plakat generate` starts from pure noise and runs the full
denoising trajectory (1000 → 0 in scheduler timesteps, simulated
across `--steps` iterations). The result is shaped entirely by the
text prompt.

`plakat img2img` starts from an existing image. It VAE-encodes that
image into the model's latent space, adds **partial noise** (how
much is controlled by `--strength`), then continues the denoise
trajectory from that partially-noisy state forward. The result
shares structure with the input but follows the new prompt.

The relationship between `--strength` and the trajectory:

```
strength=0.0:   image → encode → no noise → decode → same image out
strength=0.5:   image → encode → mid-noise → denoise from mid → mixed
strength=1.0:   image → encode → max noise → full denoise → text-driven
```

At strength 1.0, img2img is equivalent to t2i — the input is
discarded.

## 2. Your first img2img

```bash
plakat img2img photo.jpg \
    --prompt "watercolor painting of the same scene" \
    --seed 42
```

What this does:

1. Reads `photo.jpg`; snaps dimensions to a multiple of 8 (VAE
   constraint).
2. Loads SD 1.5 (downloaded on first use).
3. VAE-encodes the image; adds noise corresponding to `--strength 0.6`
   (the img2img default).
4. Runs 28 denoising steps under the prompt, producing latents that
   blend the source structure with the watercolor aesthetic.
5. VAE-decodes; saves as `out/plakat-img2img-42.png`.

Try increasing `--strength` toward 0.9 to see the source matter
less, or decreasing toward 0.3 to keep the source dominant.

## 3. When to use img2img vs t2i

- **t2i (`plakat generate`)** when you have a description but no
  starting image — the model invents everything.
- **img2img** when you have a starting point you want to refine,
  restyle, or use as a composition lock. Common uses:
  - Take a rough sketch → polished painting (high strength).
  - Take a photo → matching painted style (medium strength).
  - Take a generated image → variation with tweaked details (low
    strength, same seed).

If your starting point matters at all, img2img is the right tool.
If it doesn't, save the inference time and use `plakat generate`.

## 4. Inpaint: edit a region, keep the rest

Add `--mask`:

```bash
plakat img2img photo.jpg \
    --mask sky_mask.png \
    --prompt "stormy sky, dark clouds, lightning"
```

Now plakat treats the mask as a per-pixel "edit here" map:

- **White pixels** in the mask → fully repainted by the denoise.
- **Black pixels** → preserved exactly (pixel-identical to source).
- **Gray pixels** → blended proportionally.

Default strength for inpaint mode is **1.0** (full repaint inside
the mask), because you usually want maximum freedom inside the
marked region. Drop it lower if you want to refine rather than
replace.

### Making a mask

You have three options:

1. **Paint one in an editor.** Photoshop / GIMP / Krita: create a
   new grayscale image the same size as your input, paint the
   regions you want to inpaint in white. Save as PNG.

2. **Export a selection as alpha.** Select the region, save as PNG
   with alpha. plakat reads the alpha channel as the mask.

3. **Generate one programmatically.** Anything the `image` crate
   can write. The runnable tutorial uses a procedural script:
   `examples/draw_img2img_sample.rs`.

White = inpaint is the **AUTOMATIC1111 / diffusers convention**.
If your tools use the opposite convention, add `--mask-invert`.

## 5. Mask feathering

Hard mask edges produce visible seams along the boundary. Plakat
softens the mask with a separable box blur of radius
`--mask-feather <PX>` (default 8 px) before applying.

Effects of feather radius:

| Feather | When it's right |
|---|---|
| 0 | You want a hard, deliberate seam (rare; usually a mistake). |
| 4–8 | Mask edges are sharp and you want a clean blend. |
| 12–16 | Mask edges are deliberately rough; absorbs both the roughness and the SD latent quantisation. |
| 24+ | The inpaint region bleeds outward into preserved territory. Used intentionally for very gradual changes. |

Default 8 is a sensible starting point. Increase if you see hard
seams; decrease if the inpaint is leaking into pixels you wanted
preserved.

## 6. The strength dial in detail

For img2img (no mask), the typical question is "how much of the
source survives?" Reach for these defaults:

| Strength | Use case |
|---|---|
| 0.20 | Texture/colour tweak only. The image looks almost identical. |
| 0.35 | Subtle style transfer; composition fully preserved. |
| 0.55 | **Recommended starting point.** Real stylistic change, source still recognizable. |
| 0.75 | Heavy reinterpretation. Composition drifts. |
| 0.90 | Source is a vague layout hint at best. |

For inpaint mode, the question is different — inside the mask, you
usually want full latitude. Default 1.0 is right for most cases.
Drop to 0.7–0.8 when you're refining detail you want to keep
recognizable (e.g. fixing a face without changing the person).

## 7. Working with seeds

Each run uses a seed. Re-running with `--seed N` gives the same
output (modulo Metal RNG quirks — see
[`GENERATE.md`](../GENERATE.md)). This matters more for img2img
than for t2i because you often want to:

- Lock the seed while sweeping `--strength` to isolate the dial.
- Lock the seed while editing the prompt to compare changes
  cleanly.
- Sweep seeds (`--count 4 --seed 100`) to get four variations of
  the same edit and pick the best.

Reproducibility tip: with a fixed seed + same prompt + same
strength, img2img is fully deterministic. Use this to do "what if
I change just X" experiments without confounding noise.

## 8. Common patterns

### "Repair" a generation

You generated a great image except for one detail — a wrong
number of fingers, a weird object in the corner, a face that's
slightly off. Paint a mask over just that region and inpaint with a
descriptive prompt. The rest of the image stays untouched.

### "Restyle" a photo

You have a real-world photo; you want a painted / sketched /
abstract version. img2img at strength 0.45–0.6 with a stylistic
prompt. Keep `--steps` modest (20 is fine).

### "Continue" an iteration

Generated something close to right; want to push it further. img2img
at low strength (0.25–0.35) with a refined prompt, same seed.
Re-iterate as needed.

### Multi-region edit

Inpaint changes one region at a time, but you can chain them: run
inpaint with mask A, then run inpaint on the output with mask B.
Each pass preserves the unmasked area pixel-perfectly, so chaining
is lossless.

## 9. Limits

- **Mask resolution is quantised** to the latent grid
  (`image_w/8 × image_h/8`). Very fine mask boundaries lose
  precision — use feathering to absorb the quantisation rather
  than fight it.
- **SD3 img2img doesn't compose with ControlNet** —
  `--control-spec` on `plakat img2img --model sd35-*` bails
  loud. Use `plakat generate` for SD3 CN-guided outputs.
- **SD-family img2img + tiled** isn't wired — `plakat img2img
  --tiled` is SD3 only (v0.16). For SD 1.5 / SDXL high-res
  workflows, prefer `plakat generate --tiled` or chain
  `plakat upscale` after a smaller t2i.

What HAS been wired since the earlier "v0.8 limitations" of this
tutorial:

- **Flux img2img + inpaint** (v0.13) — `plakat img2img --model
  flux-dev`, `flux-fill-dev` (inpaint), GGUF variants. See §
  "Flux img2img and inpaint" in [`IMG2IMG.md`](../IMG2IMG.md).
- **SD3 / SD3.5 img2img + inpaint** (v0.15) — RePaint-style
  per-step mask blend. Tiled composes as of v0.16.
- **Outpaint** (v0.13) — `plakat outpaint` subcommand expands
  the canvas + builds the new-region mask, hands off to inpaint.
- **Scenario integration** (v0.13) — per-task `init-image:` /
  `mask:` / `strength:` / `outpaint:` for both SD and Flux.
- **Tiled + Flux Fill** (v0.16) — `plakat img2img --model
  flux-fill-dev --tiled` for 4K+ inpaint.
- **Tiled + SD3 img2img / inpaint** (v0.16) — `plakat img2img
  --model sd35-* --tiled` for 2K+ outputs.

## Where to next

- **Runnable companion**: four self-contained scripts +
  a procedurally-generated sample input + mask
  ([`examples/tutorials/IMG2IMG/`](../../examples/tutorials/IMG2IMG/)).
- **Full reference**: every flag, every edge case
  ([`Documentation/IMG2IMG.md`](../IMG2IMG.md)).
- **Combining with style transfer**: `--style` and `--style-ref`
  work on img2img the same as on `plakat generate`
  ([`STYLES.md`](../STYLES.md)).
- **Combining with artefacts**: artefact compositing happens
  *before* any post-processing — you can composite, then img2img
  the result for further polish. See
  [`ARTEFACTS.md`](../ARTEFACTS.md).
