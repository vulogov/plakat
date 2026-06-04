# Proof-corpus coverage matrix

The capability surface this corpus aims to demonstrate, and the driver
that proves each. Status: ✅ driver written · ⬜ to add · ⚠️ constrained
(gated / not on Metal). "Rendered" is checked once the output image is
committed under `images/`.

## Models (text-to-image)

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| Stable Cascade | `cascade.hjson` | ✅ | ⬜ | ungated, ~16 GB Metal |
| SD 1.5 | `sd.hjson` | ⬜ | ⬜ | ungated, ~4 GB |
| SDXL | `sd.hjson` | ⬜ | ⬜ | ungated, ~7 GB |
| PixArt-Σ | `pixart.hjson` | ⬜ | ⬜ | ungated |
| Flux (BF16) | `flux.hjson` | ⬜ | ⬜ | ⚠️ gated (dev) / ~33 GB; GGUF broken on Metal |
| SD3 / 3.5 | `sd3.hjson` | ⬜ | ⬜ | ⚠️ gated |

## Conditioning & adapters

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| ControlNet (canny, Cascade) | `cascade.hjson` | ✅ | ⬜ | per-task `control:` |
| ControlNet (canny/depth, SD/SDXL) | `controlnet.hjson` | ⬜ | ⬜ | |
| LoRA / DoRA (Cascade) | `lora.sh` | ⬜ | ⬜ | per-task Cascade LoRA not yet in scenarios → CLI |
| Image variation (Cascade) | `variation.sh` | ⬜ | ⬜ | `--image-variation` |
| Styles / Looks / Genres | `presets.hjson` | ⬜ | ⬜ | `style:` / `--look` / `--genre` |
| Portrait / identity | `portrait.hjson` | ⬜ | ⬜ | needs a reference face photo |

## Transforms & post

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| img2img (Cascade `--faithful`) | `img2img.sh` | ⬜ | ⬜ | CLI subcommand |
| Inpaint (masked) | `img2img.sh` | ⬜ | ⬜ | `--mask` |
| Outpaint | `outpaint.sh` | ⬜ | ⬜ | `plakat outpaint` |
| Upscale (ML) | `upscale.sh` | ⬜ | ⬜ | `plakat upscale` |

## Batch & scripting

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| Scenarios (HJSON batch) | every `*.hjson` | ✅ | ⬜ | the corpus is itself the proof |
| Bund scripting | `script.bund` | ⬜ | ⬜ | `plakat run` |
| Gallery generator | `plakat gallery` | ✅ | n/a | builds this corpus's index |

---

The non-`generate` features (img2img / inpaint / outpaint / upscale /
image-variation / scripting) aren't scenario-drivable, so they land as
small shell / Bund drivers whose outputs write into `images/` alongside
the scenario renders. `plakat gallery images --recursive` indexes them
all uniformly.
