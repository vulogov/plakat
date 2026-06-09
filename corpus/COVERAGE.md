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
| Textual Inversion (embedding) | `embedding.sh` | ✅ | ⬜ | `generate --embedding` — inject a TI trigger at runtime (EasyNegative, SD 1.5); baseline vs +embedding pair. `PATH_OR_REPO[:trigger][:scale]`. |
| Looks / Genres (`--look` / `--genre`) | `looks.sh` / `genres.sh` | ✅ | ✅ | looks = the 8 bundled art-mediums on one subject (SDXL); genres = the bundled `anime` subject-domain (independent axis, they compose). Looks use **`--smart-discovery`**: a local LLM judges the Civitai candidate pool → the best *style* LoRA, rejecting characters (7/8 judged style LoRAs + chalk-pastel prompt-only). Genres pin a Civitai SDXL LoRA by id. `--scheduler euler-a` (Metal). |
| Civitai LoRA (by id) | `civitai.sh` | ✅ | ✅ | pull a LoRA from Civitai by model id (`civitai:<id>:scale`) + render — Eldritch Watercolor on SDXL. Needs `CIVITAI_API_KEY` for the auth-gated download. |
| **Style LoRA training (SD1.5 / SDXL / SD3.5)** | `style_train.sh` → `style_gen.sh` | ✅ | ✅ | **train your own style** from a folder of images → a LoRA loadable via `--lora`. 9 watercolour refs → LoRA → style transfers on all three bases: SD1.5 (kohya, 128/128), SDXL (kohya, 560/560), SD3.5 (diffusers-PEFT, 191/191). `plakat style train --base {sd15,sdxl,sd35}`; training + generation separated. |
| Portrait / identity | `portrait.sh` + `portrait.hjson` | ✅ | ✅ | text personas (script) + `example.png` lookalike (scenario, IP-Adapter-Plus-Face) |
| AnimateDiff (motion) | `animate.sh` | ✅ | ✅ | text→short video; frames in `images/animate/` |
| Stylize (IP-Adapter ref-variation) | `stylize.sh` | ✅ | ⬜ | apply a reference's *look* to a subject via IP-Adapter — no prompt, no training. SD 1.5 or **SDXL** (`--model sdxl`, sharper, native 1024²; SD 1.5 kept as fallback). NOTE: the IP-Adapter transfers content/appearance/palette, NOT painterly texture → a ref-*variation* tool (stays photoreal); for true painterly style use the LoRA paths / `--look`. `--ref-blur` suppresses ref content. |

## Transforms & post

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| img2img (style transfer, SDXL) | `img2img.sh` | ✅ | ✅ | prompt-steered medium restyle; no LoRA/download |
| Inpaint (masked) | `inpaint.sh` | ✅ | ✅ | `img2img --mask` — repaint a region (the sky) and preserve the rest; committed input + procedural mask, self-contained |
| Outpaint | `outpaint.sh` | ✅ | ✅ | `plakat outpaint` — pad the canvas + paint the new strip in-context (sdxl-inpaint). Clean: the masked region is conditioned on mid-gray (no dark bands) with a binary mask (no feather seams) |
| Upscale (ML) | `upscale.sh` | ✅ | ✅ | Real-ESRGAN ×2 (Metal-safe); ×4 buffers OOM Metal → `--device cpu` |
| Transparent (background knock-out) | `transparent.sh` | ✅ | ⬜ | `plakat transparent` — make the upper-left corner colour transparent → RGBA cut-out; generates a subject on a flat backdrop, then knocks it out (`--tolerance` softens edges) |

## Batch & scripting

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| Scenarios (HJSON batch) | every `*.hjson` | ✅ | ✅ | the corpus is itself the proof |
| Scene × weather axes | `weather-scene.hjson` | ✅ | ✅ | one area (prompt-header) re-lit + re-weathered across both axes |
| Bund scripting | `script.bund` (`script.sh`) | ✅ | ✅ | `plakat run` — load → generate → upscale → save handle-reuse chain (SD 1.5) |
| Gallery generator | `plakat gallery` | ✅ | n/a | builds this corpus's index |

---

The non-`generate` features (img2img / inpaint / outpaint / upscale /
image-variation / scripting) aren't scenario-drivable, so they land as
small shell / Bund drivers whose outputs write into `images/` alongside
the scenario renders. `plakat gallery images --recursive` indexes them
all uniformly.
