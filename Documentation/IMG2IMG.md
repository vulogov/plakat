# `plakat img2img` — image-to-image and inpaint

Reference for the `plakat img2img` subcommand. For a runnable
walkthrough, see [`examples/tutorials/IMG2IMG/`](../examples/tutorials/IMG2IMG/).

## Modes

`plakat img2img` is one subcommand with two modes, selected by
whether `--mask` is supplied:

| Mode | Trigger | Default `--strength` | Behaviour |
|---|---|---|---|
| **img2img** | no `--mask` | 0.6 | Whole image re-denoised at the requested strength. |
| **inpaint** | `--mask PATH` | 1.0 | Only the masked region is repainted; the rest is preserved exactly. |

Both modes share the same denoise machinery; inpaint is just
img2img with a non-uniform mask.

## Quick start

```bash
# img2img: re-imagine an image
plakat img2img photo.jpg --prompt "watercolor painting of the same scene"

# inpaint: replace just the sky
plakat img2img photo.jpg --mask sky.png --prompt "stormy sky, lightning"

# control how much creative latitude the model gets
plakat img2img photo.jpg --prompt "..." --strength 0.4
```

## Required arguments

| Argument | Description |
|---|---|
| `<INPUT>` (positional) | Path to the source image. Any format the `image` crate reads (PNG, JPEG, WebP). |
| `--prompt <PROMPT>` | Text describing the desired output. |

## Common options

| Flag | Default | Description |
|---|---|---|
| `--negative <NEG>` | `""` | Negative prompt — things to discourage. |
| `--strength <F>` | 0.6 / 1.0 | img2img strength in `[0, 1]`. Default: 0.6 without mask, 1.0 with mask. |
| `--seed <N>` | random | Base seed. With `--count N`, additional outputs use seed+1, seed+2, ... |
| `-n, --count <N>` | 1 | Number of variations from the same input. |
| `--steps <N>` | 28 | Denoising steps. Tutorial recommends 20 with `euler-a`. |
| `--guidance <F>` | 7.5 | Classifier-free guidance scale. |
| `--scheduler <K>` | `default` | Scheduler name. Same set as `plakat generate`. |
| `--model <MODEL>` | `sd15` | Model alias or HF repo id. SD 1.5 / 2.1 / SDXL / SDXL-Turbo for the SD path; `flux-dev` / `flux-schnell` for Flux img2img; `flux-fill-dev` for Flux inpainting; `flux-kontext-dev` (v0.18) for Flux image editing (input becomes the reference, prompt the edit). |
| `--size <WxH>` | input dims | Override working resolution. Input is resized to match. Must be a multiple of 8. |
| `--out <DIR>` | `./out` | Output directory. Created if absent. |

## Inpaint mask options

Only meaningful when `--mask` is supplied.

| Flag | Default | Description |
|---|---|---|
| `--mask <PATH>` | — | Path to mask image. Auto-resized to working dimensions. |
| `--mask-feather <PX>` | 8 | Box-blur radius applied to the mask before the denoise — softens the inpaint↔preserve transition. |
| `--mask-invert` | off | Flip mask polarity. Black becomes "inpaint here", white becomes "preserve". Useful when your mask source uses the opposite convention. |

### Mask format

A mask is interpreted as a per-pixel scalar in `[0, 1]`:

- `1.0` (white) → inpaint this pixel.
- `0.0` (black) → preserve this pixel.
- intermediate values → blend proportionally.

The image can be in any of three formats:

| Source | Interpretation |
|---|---|
| Grayscale (`L8`) | Brightness directly. |
| RGB | Luminance (`Y = 0.299R + 0.587G + 0.114B`). |
| RGBA | The alpha channel. Useful for masks created in Photoshop / GIMP using selections. |

After loading, the mask is resized to the working resolution
(`--size` or the input's snapped dims), optionally inverted, then
feathered.

## Strength dial — what to pick

`--strength` is the standard img2img dial. It controls where on the
noise schedule the denoise starts:

- `0.0` — degenerate; returns the input unchanged (after a clean
 VAE roundtrip).
- `0.15–0.25` — subtle. Useful for "preserve composition, change
 texture / palette."
- `0.30–0.50` — moderate. Most common range for img2img refinement.
- **0.6 (img2img default)** — model has real creative room without
 abandoning the source.
- `0.7–0.9` — heavy. The source becomes more of a layout hint.
- `1.0` — full re-noise. Equivalent to t2i from scratch with the
 same prompt. **Default for inpaint mode** — you usually want full
 freedom inside the masked region.

For inpaint, lower strengths can be useful when the source contains
detail you want to **refine** rather than **replace** (e.g. fixing
a hand by inpainting at strength 0.7 instead of 1.0).

## Resolution handling

Three ways to set output dimensions, in priority order:

1. **`--size WxH`** — explicit, wins over everything. Multiple of 8
   required (VAE downsample factor).
2. **`--aspect 16:9 --base 1024`** (v0.18) — derived. The shorter
   side becomes `--base`; the longer side becomes
   `base × ratio`. Both axes are then rounded **down** to the
   nearest multiple of 8. Mutually exclusive with `--size`.
3. **default** — the input image's actual dimensions, rounded down
   to multiples of 8. A 1080×720 input becomes 1080×720; a 513×800
   input becomes 512×800.

When `--size` or `--aspect` produces dimensions that don't match the
input's, the input is resized (triangle filter) to match before
denoising. Tip: SD 1.5 was trained at 512²; outputs at 1024² often
look worse than at 512² for SD 1.5. For higher resolution, use
`--model sdxl --size 1024x1024` (or `--model sdxl --aspect 16:9
--base 896` for a landscape variant).

## Output naming

| Mode | Filename pattern |
|---|---|
| img2img (SD-family) | `plakat-img2img-<seed>.png` |
| inpaint (SD-family) | `plakat-inpaint-<seed>.png` |
| Flux (img2img / Fill) | `plakat-flux-<seed>.png` |
| SD3 img2img | `plakat-sd3-img2img-<seed>.png` |
| SD3 inpaint | `plakat-sd3-inpaint-<seed>.png` |

With `--count N`, files are `<prefix>-<base>.png`,
`<prefix>-<base+1>.png`, ... using consecutive seeds.

### `--grid` (v0.18)

With `--count N > 1`, pass `--grid` to also write a single
`<prefix>-grid-<base-seed>.png` combining all N outputs in a
near-square layout alongside the per-image PNGs. `--grid-cols N`
forces a specific column count (default `ceil(sqrt(count))`);
`--grid-padding PX` inserts a white border between cells (default
0, flush). The grid prefix tracks the backbone, so a Flux inpaint
sweep with `--count 4 --grid` produces `plakat-flux-grid-…`.

## Style + LoRA support

`--style`, `--style-ref`, `--lora`, and `--lora-scale` work
identically to `plakat generate`. The denoise is the same modulo
the partial-strength starting point.

## Flux img2img and inpaint

`plakat img2img --model flux-dev` (or `flux-schnell`, `flux-dev-gguf`)
runs rectified-flow img2img: the init image is VAE-encoded and mixed
with fresh noise at `t = strength`, then the truncated schedule
denoises from there. Same `--strength` convention as SD.

`plakat img2img --model flux-fill-dev --mask MASK init.png` runs BFL's
dedicated Flux.1-Fill-dev checkpoint. It has a 384-channel `img_in`
(64 noise + 64 masked-image-latent + 256 mask) — the mask drives the
denoise directly rather than being a RePaint-style overlay, so
`--strength` does not apply (the mask itself controls what changes).
Default `--guidance 30` per BFL's recommendation.

Both Flux paths compose with LoRA (PEFT + AI-Toolkit formats), GGUF
quantization (`flux-dev-gguf`, `--flux-quant-level`, `--quantize-t5`),
ControlNet via the standard `--control-spec` grammar, and — as of
**** — `--tiled` on Flux.1-Fill-dev for 4K+ inpaint via
per-tile masked-latent + mask packing.

```bash
# Flux img2img: re-imagine an init image, 70% strength
plakat img2img init.png --model flux-dev \
 --prompt "the same scene in a stained glass window" \
 --strength 0.7

# Flux inpaint (Fill model): only the masked region changes
plakat img2img init.png --mask region.png --model flux-fill-dev \
 --prompt "ornate carved stone arch"

# Flux on a 16 GB GPU via GGUF
plakat img2img init.png --model flux-fill-dev-gguf \
 --mask region.png --flux-quant-level Q5_K_M --quantize-t5 \
 --prompt "..."

# Flux Kontext (v0.18) — input is the reference, prompt describes
# the edit. Routes through Kontext's sequence-concat conditioning
# (not the rectified-flow init lerp that flux-dev img2img uses).
plakat img2img photo.png --model flux-kontext-dev \
 --prompt "make the lighting golden hour, warm tones"

# Same recipe via GGUF for 16 GB GPUs
plakat img2img photo.png --model flux-kontext-dev-gguf \
 --prompt "add snow on the rooftops" --flux-quant-level Q5_K_M

# Opt-in aspect-bucket snap (one of 17 BFL-recommended resolutions)
plakat img2img tall_photo.png --model flux-kontext-dev \
 --prompt "..." --kontext-bucket
```

`--strength` is ignored on Kontext (no flow-match init lerp);
`--mask` bails loud (use `flux-fill-dev` instead).

## SD3 / SD3.5 img2img and inpaint

`plakat img2img --model sd35-medium` (or any of `sd35-large`,
`sd35-large-turbo`, `sd3-medium`) runs MMDiT-flavoured img2img: the
init image is VAE-encoded with SD3's `(z - 0.0609) * 1.5305`
normalisation, mixed with fresh noise at `t = strength` using the
rectified-flow lerp, and denoised on a truncated schedule starting
at `strength`. Same `--strength` convention as Flux/SD.

`--mask MASK` adds RePaint-style inpaint: after each denoise step,
unmasked pixels are snapped back to `lerp(init, eps, t_next)` —
keeping them on the init's flow trajectory while the masked region
freely denoises. White = inpaint, black = preserve (use
`--mask-invert` for sources with the opposite convention).
`--mask-feather PX` softens the edge before downsampling to latent.

Output naming reflects the mode: `plakat-sd3-img2img-<seed>.png`
when no mask, `plakat-sd3-inpaint-<seed>.png` when masked.

```bash
# SD3.5 img2img — re-imagine the input at 60% strength
plakat img2img photo.png --model sd35-medium \
 --prompt "the same scene rendered as a watercolor"

# SD3.5 Large inpaint — replace just the masked region
plakat img2img photo.png --model sd35-large \
 --mask sky.png \
 --prompt "dramatic stormy sky, lightning"

# sd35-large-turbo + img2img — 4-step distilled, no CFG
plakat img2img photo.png --model sd35-large-turbo \
 --prompt "..." --guidance 0
```

SD3 LoRA (`--lora`) composes with img2img and inpaint. Diffusers
PEFT format is the supported convention (keys under
`transformer.transformer_blocks.{i}.attn.*` / `ff.*` / `norm1*`).
Affected Linears are merged into the MMDiT weights at load time.

Not supported on SD3 img2img: ControlNet (`--control*`). Passing
those flags raises an explicit error.

**Tiled SD3 img2img + inpaint** composes as of **** —
`plakat img2img --tiled --tile-size 1024 --tile-stride 768` runs
the rectified-flow init lerp + RePaint mask blend on the
per-tile velocity prediction. The Hann blend doesn't know about
the mask, so sharp mask boundaries can produce tile seams — use
`--mask-feather PX` to smooth them.

```bash
# 2K SD3.5 img2img
plakat img2img photo.png --model sd35-medium --size 2048x2048 \
 --prompt "rendered as a watercolor" \
 --tiled --tile-size 1024 --tile-stride 768

# 2K SD3.5 inpaint with a feathered mask
plakat img2img photo.png --model sd35-medium --size 2048x2048 \
 --mask sky.png --mask-feather 16 \
 --prompt "dramatic stormy sky" \
 --tiled --tile-size 1024
```

## Outpaint

For canvas expansion (extend an image past its borders), use the
dedicated `plakat outpaint` subcommand. It generates the expanded
canvas + new-region mask and hands off to the same inpaint flow:

```bash
plakat outpaint photo.png --prompt "wide landscape, panorama" \
 --left 512 --right 512 --model sdxl-inpaint

# All four sides, Flux Fill model
plakat outpaint photo.png --prompt "..." --expand 256 \
 --model flux-fill-dev
```

`plakat outpaint` snaps padding to the model's VAE / patch constraint
(8 for SD, 16 for Flux), replicates the input's edge pixels into the
new region (better seam continuity than flat gray), and pins
`--strength 1.0` (the new region has no original content to preserve).
`--grid` / `--grid-cols` / `--grid-padding` (v0.18) forward through to
the underlying inpaint flow, producing a `plakat-inpaint-grid-…` PNG
when `--count > 1`.

## Limits

- **Mask resolution is downsampled.** The latent-space mask is
 `image/8 × image/8`, so very fine mask boundaries lose precision.
 Use `--mask-feather` to absorb the quantisation instead of fighting
 it.
- **Scenarios** support per-task `init-image:` / `mask:` / `strength:`
 / `outpaint:` for both SD and Flux models. SD inpaint
 tasks in scenarios reload the pipeline per task (img2img doesn't yet
 share the t2i `Pipeline::load`-once shape). Flux scenarios share a
 single pipeline across tasks ( SD3 scenarios share a
 single pipeline as of 

## See also

- [Runnable tutorial](../examples/tutorials/IMG2IMG/) — four scripts
 + a sample landscape + sky mask.
- [`GENERATE.md`](GENERATE.md) — most flags are shared with
 `plakat generate`.
- [`ARTEFACTS.md`](ARTEFACTS.md) — the v2 artefact-blend pipeline
 uses the same underlying denoise primitives.
- [`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md) — expected
 speeds + memory tiers.
