# Proof-corpus coverage matrix

The capability surface this corpus aims to demonstrate, and the driver
that proves each. Status: ✅ driver written · ⬜ to add · ⚠️ blocked /
constrained (gated / not on Metal / no asset to demo). "Rendered" is checked once the output image is
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
| LoRA / DoRA (Cascade) | `lora.sh` | ⚠️ | ⬜ | **BLOCKED (ecosystem, not code).** Engine is done (v0.38/v0.42): merges into Stage B (`decoder.`) + Stage C (`prior.`), kohya + diffusers-PEFT prefixes, DoRA auto-detected; works via CLI (`--lora`) AND scenarios (per-task `lora:`). But there's no ungated Cascade LoRA to demo — Cascade never grew SDXL's LoRA ecosystem, and plakat can't train one (training is SD1.5/SDXL/SD3.5 only). Parked like Flux: point a driver at a real Cascade LoRA if one ever surfaces. |
| Image variation (Cascade) | `variation.sh` | ✅ | ⬜ | `--image-variation` — condition on a reference's CLIP ViT-L/14 embedding (unCLIP-style); keeps semantics, re-composes. Pure (empty prompt) + steered. v0.42 Stage C image encoder. |
| Textual Inversion (embedding) | `embedding.sh` | ✅ | ⬜ | `generate --embedding` — inject a TI trigger at runtime (EasyNegative, SD 1.5); baseline vs +embedding pair. `PATH_OR_REPO[:trigger][:scale]`. |
| Artefact compositing (multi) | `artefact.sh` + `artefact.hjson` | ✅ | ✅ | `--artefact NAME@ZONE --artefact-blend` — **integral** multi-artefact compositing: canvas-relative scale, contact-shadow grounding, scene-ambient colour harmony, and a canny-ControlNet-guided re-paint that integrates the cutouts into the scene (not paste). Library built by matting anime cutouts (`--matte`). Reads most naturally on **stylized / anime** scenes (the re-paint helps there); photoreal stays a grounded composite. CLI **and** scenario. SDXL only (blend = SD-core img2img). |
| Looks / Genres (`--look` / `--genre`) | `looks.sh` / `genres.sh` | ✅ | ✅ | looks = the 8 bundled art-mediums on one subject (SDXL); genres = the bundled `anime` subject-domain (independent axis, they compose). Looks use **`--smart-discovery`**: a local LLM judges the Civitai candidate pool → the best *style* LoRA, rejecting characters (7/8 judged style LoRAs + chalk-pastel prompt-only). Genres pin a Civitai SDXL LoRA by id. `--scheduler euler-a` (Metal). |
| Civitai LoRA (by id) | `civitai.sh` | ✅ | ✅ | pull a LoRA from Civitai by model id (`civitai:<id>:scale`) + render — Eldritch Watercolor on SDXL. Needs `CIVITAI_API_KEY` for the auth-gated download. |
| **Style LoRA training (SD1.5 / SDXL / SD3.5)** | `style_train.sh` → `style_gen.sh` | ✅ | ✅ | **train your own style** from a folder of images → a LoRA loadable via `--lora`. 9 watercolour refs → LoRA → style transfers on all three bases: SD1.5 (kohya, 128/128), SDXL (kohya, 560/560), SD3.5 (diffusers-PEFT, 191/191). `plakat style train --base {sd15,sdxl,sd35}`; training + generation separated. |
| Portrait / identity | `portrait.sh` + `portrait.hjson` | ✅ | ✅ | text personas (script) + `example.png` lookalike (scenario, IP-Adapter-Plus-Face) |
| AnimateDiff (motion) | `animate.sh` | ✅ | ✅ | text→short video; frames in `images/animate/` |
| Stylize: ref-variation + InstantStyle | `stylize.sh` | ✅ | ⬜ | apply a reference's *look* to a subject via IP-Adapter — no prompt, no training. **Two paths:** (1) DEFAULT concat = ref-*variation* (transfers content/appearance/palette, NOT painterly texture → stays photoreal; `--ref-blur` suppresses ref content); (2) **`--instantstyle`** = true painterly STYLE transfer (SD 1.5 + SDXL), injecting the ref only into the style block (SDXL `up_blocks.0.attentions.1` / SD 1.5 `up_blocks.1.attentions.1`) via a decoupled IP cross-attn, `--style-scale` dials it. Driver renders the A/B (concat vs InstantStyle, same subject+ref) + a 2nd watercolour + the SD 1.5 backbone, from `corpus/style/watercolour`. |

## Transforms & post

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| img2img (style transfer, SDXL) | `img2img.sh` | ✅ | ✅ | prompt-steered medium restyle; no LoRA/download |
| Inpaint (masked) | `inpaint.sh` | ✅ | ✅ | `img2img --mask` — repaint a region (the sky) and preserve the rest; committed input + procedural mask, self-contained |
| Outpaint | `outpaint.sh` | ✅ | ✅ | `plakat outpaint` — pad the canvas + paint the new strip in-context (sdxl-inpaint). Clean: the masked region is conditioned on mid-gray (no dark bands) with a binary mask (no feather seams) |
| Upscale (ML) | `upscale.sh` | ✅ | ✅ | Real-ESRGAN ×2 (Metal-safe); ×4 buffers OOM Metal → `--device cpu` |
| Transparent (smart cut-out) | `transparent.sh` | ✅ | ✅ | `plakat transparent --matte` — content-aware **U2Net** smart cut-out: lifts a photoreal subject off ANY background (no chroma) → clean RGBA. Apple-on-a-table → tight cut-out, no fringe. (Corner flood-fill `--tolerance` stays for flat studio backdrops.) Weights auto-download (`vulogov98/u2net-universal`). |

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
