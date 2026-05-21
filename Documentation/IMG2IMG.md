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
| `--model <MODEL>` | `sd15` | Model alias or HF repo id. **Flux is not supported.** |
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

When `--size` is omitted, plakat reads the input's actual dimensions
and rounds each axis **down** to the nearest multiple of 8 (the VAE
downsample factor). A 1080×720 input becomes 1080×720; a 513×800
input becomes 512×800.

When `--size` is set, the input is resized (triangle filter) to
match. Same constraint: the size must be a multiple of 8 on both
axes.

Tip: SD 1.5 was trained at 512²; outputs at 1024² often look
worse than at 512² for SD 1.5. For higher resolution, use
`--model sdxl --size 1024x1024`.

## Output naming

| Mode | Filename pattern |
|---|---|
| img2img | `plakat-img2img-<seed>.png` |
| inpaint | `plakat-inpaint-<seed>.png` |

With `--count N`, files are `plakat-img2img-<base>.png`,
`plakat-img2img-<base+1>.png`, ... using consecutive seeds.

## Style + LoRA support

`--style`, `--style-ref`, `--loras`, and `--lora-scale` work
identically to `plakat generate`. The denoise is the same modulo
the partial-strength starting point.

## Limits

- **No Flux support.** Flux uses a different latent space + scheduler
  combo that the img2img wrapper doesn't (yet) handle. Use SD 1.5,
  SD 2.1, or SDXL. Flux img2img is on the roadmap.
- **No outpainting.** `plakat img2img` doesn't extend the canvas;
  it only re-paints within the existing pixel grid. Outpainting is
  planned as a separate subcommand.
- **No scenario integration in v0.8.** `plakat scenario` doesn't yet
  expose per-task `init-image:` / `mask:` fields. CLI-only for now.
- **Mask resolution is downsampled.** The latent-space mask is
  `image/8 × image/8`, so very fine mask boundaries lose precision.
  Use `--mask-feather` to absorb the quantisation instead of fighting
  it.

## See also

- [Runnable tutorial](../examples/tutorials/IMG2IMG/) — four scripts
  + a sample landscape + sky mask.
- [`GENERATE.md`](GENERATE.md) — most flags are shared with
  `plakat generate`.
- [`ARTEFACTS.md`](ARTEFACTS.md) — the v2 artefact-blend pipeline
  uses the same underlying denoise primitives.
- [`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md) — expected
  speeds + memory tiers.
