# Proof-corpus coverage matrix

The capability surface this corpus aims to demonstrate, and the driver
that proves each. Status: ✅ driver written · ⬜ to add · ⚠️ constrained
(gated / not on Metal). "Rendered" is checked once the output image is
committed under `images/`.

## Models (text-to-image)

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| Stable Cascade | `cascade.hjson` | ✅ | ✅ | ungated, ~16 GB Metal |
| SD 1.5 | `sd15.hjson` | ✅ | ✅ | ungated, ~4 GB, 512²; incl. canny CN |
| SD 2.1 | `sd21.hjson` | ✅ | ✅ | ungated, ~5 GB, 768² v-prediction; alias repointed off the gated stabilityai repo |
| SDXL | `sdxl.hjson` | ✅ | ✅ | ungated, ~7 GB; incl. canny CN |
| PixArt-Σ | `pixart.hjson` | ✅ | ✅ | ungated |
| Flux (BF16) | `flux.hjson` | ⬜ | ⬜ | ⚠️ gated (dev) / ~33 GB; GGUF broken on Metal |
| SD3.5 Medium | `sd35.hjson` | ✅ | ✅ | ⚠️ gated; BF16-native ~16 GB Metal; strong text |

## Conditioning & adapters

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| ControlNet (canny, Cascade) | `cascade.hjson` | ✅ | ✅ | per-task `control:`; output within `images/cascade/` |
| ControlNet (canny, SDXL) | `sdxl.hjson` | ✅ | ✅ | auto-from; output within `images/sdxl/` |
| LoRA / DoRA (Cascade) | `lora.sh` | ⬜ | ⬜ | per-task Cascade LoRA not yet in scenarios → CLI |
| Image variation (Cascade) | `variation.sh` | ⬜ | ⬜ | `--image-variation` |
| Styles / Looks / Genres | `presets.hjson` | ⬜ | ⬜ | `style:` / `--look` / `--genre` |
| **Style LoRA training (SD1.5 / SDXL / SD3.5)** | `style_train.sh` → `style_gen.sh` | ✅ | ✅ | **train your own style** from a folder of images → a LoRA loadable via `--lora`. 9 watercolour refs → LoRA → style transfers on all three bases: SD1.5 (kohya, 128/128), SDXL (kohya, 560/560), SD3.5 (diffusers-PEFT, 191/191). `plakat style train --base {sd15,sdxl,sd35}`; training + generation separated. |
| Portrait / identity | `portrait.sh` + `portrait.hjson` | ✅ | ✅ | text personas (script) + `example.png` lookalike (scenario, IP-Adapter-Plus-Face) |
| AnimateDiff (motion) | `animate.sh` | ✅ | ✅ | text→short video; frames in `images/animate/` |

## Transforms & post

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| img2img (style transfer, SDXL) | `img2img.sh` | ✅ | ✅ | prompt-steered medium restyle; no LoRA/download |
| Inpaint (masked) | `img2img.sh` | ⬜ | ⬜ | `--mask` |
| Outpaint | `outpaint.sh` | ⬜ | ⬜ | `plakat outpaint` |
| Upscale (ML) | `upscale.sh` | ✅ | ✅ | Real-ESRGAN ×2 (Metal-safe); ×4 buffers OOM Metal → `--device cpu` |

## Batch & scripting

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| Scenarios (HJSON batch) | every `*.hjson` | ✅ | ✅ | the corpus is itself the proof |
| Scene × weather axes | `weather-scene.hjson` | ✅ | ✅ | one area (prompt-header) re-lit + re-weathered across both axes |
| Bund scripting | `script.bund` (`script.sh`) | ✅ | ⬜ | `plakat run` — load → generate → upscale → save handle-reuse chain (SD 1.5) |
| Gallery generator | `plakat gallery` | ✅ | n/a | builds this corpus's index |

---

The non-`generate` features (img2img / inpaint / outpaint / upscale /
image-variation / scripting) aren't scenario-drivable, so they land as
small shell / Bund drivers whose outputs write into `images/` alongside
the scenario renders. `plakat gallery images --recursive` indexes them
all uniformly.
