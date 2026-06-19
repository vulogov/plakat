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
| Flux (BF16) | `flux.hjson` | ⬜ | ⬜ | ⚠️ **CPU/CUDA-only; untested on Metal** (1.0 decision). Gated (dev), BF16 ~33 GB, and candle's Metal quantized matmul kernel corrupts GGUF (upstream, not plakat-fixable). Scoped out of the Metal-verified surface; see `FEATURE_TO_MODEL.md`. |
| SD3.5 Medium | `sd35.hjson` | ✅ | ✅ | ⚠️ gated; BF16-native ~16 GB Metal; strong text |

## Conditioning & adapters

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| ControlNet (canny, Cascade) | `cascade.hjson` | ✅ | ✅ | per-task `control:`; output within `images/cascade/` |
| ControlNet (canny, SDXL) | `sdxl.hjson` | ✅ | ✅ | auto-from; output within `images/sdxl/` |
| LoRA / DoRA (Cascade) | `lora.sh` | ⚠️ | ⬜ | **BLOCKED (ecosystem, not code).** Engine is done (v0.38/v0.42): merges into Stage B (`decoder.`) + Stage C (`prior.`), kohya + diffusers-PEFT prefixes, DoRA auto-detected; works via CLI (`--lora`) AND scenarios (per-task `lora:`). But there's no ungated Cascade LoRA to demo — Cascade never grew SDXL's LoRA ecosystem, and plakat can't train one (training is SD1.5/SDXL/SD3.5 only). Parked like Flux: point a driver at a real Cascade LoRA if one ever surfaces. |
| Image variation (Cascade) | `variation.sh` | ✅ | ✅ | `--image-variation` — condition on a reference's CLIP ViT-L/14 embedding (unCLIP-style); keeps semantics, re-composes. Pure (empty prompt) + steered. v0.42 Stage C image encoder. |
| Textual Inversion (embedding) | `embedding.sh` | ✅ | ✅ | `generate --embedding` — inject a TI trigger at runtime (EasyNegative via the ungated `embed/` mirror, SD 1.5); baseline vs +embedding pair. `PATH_OR_REPO[:trigger][:scale]`. |
| Artefact compositing (multi) | `artefact.sh` + `artefact.hjson` | ✅ | ✅ | `--artefact NAME@ZONE --artefact-blend` — **integral** multi-artefact compositing: canvas-relative scale, contact-shadow grounding, scene-ambient colour harmony, and a canny-ControlNet-guided re-paint that integrates the cutouts into the scene (not paste). Library built by matting anime cutouts (`--matte`). Reads most naturally on **stylized / anime** scenes (the re-paint helps there); photoreal stays a grounded composite. CLI **and** scenario. SDXL only (blend = SD-core img2img). |
| Looks / Genres (`--look` / `--genre`) | `looks.sh` / `genres.sh` | ✅ | ✅ | looks = the 8 bundled art-mediums on one subject (SDXL); genres = the bundled `anime` subject-domain (independent axis, they compose). Looks use **`--smart-discovery`**: a local LLM judges the Civitai candidate pool → the best *style* LoRA, rejecting characters (7/8 judged style LoRAs + chalk-pastel prompt-only). Genres pin a Civitai SDXL LoRA by id. `--scheduler euler-a` (Metal). |
| Civitai LoRA (by id) | `civitai.sh` | ✅ | ✅ | pull a LoRA from Civitai by model id (`civitai:<id>:scale`) + render — Eldritch Watercolor on SDXL. Needs `CIVITAI_API_KEY` for the auth-gated download. |
| **Style LoRA training (SD1.5 / SDXL / SD3.5)** | `style_train.sh` → `style_gen.sh` | ✅ | ✅ | **train your own style** from a folder of images → a LoRA loadable via `--lora`. 9 watercolour refs → LoRA → style transfers on all three bases: SD1.5 (kohya, 128/128), SDXL (kohya, 560/560), SD3.5 (diffusers-PEFT, 191/191). `plakat style train --base {sd15,sdxl,sd35}`; training + generation separated. |
| **DreamBooth subject LoRA** | `dreambooth.sh` | ✅ | ✅ | **v1.0 — Part 2, VERIFIED.** Learn a SUBJECT + class prior-preservation. Self-contained: GENERATES a synthetic subject (orange fox plush, 4 views) + class set (6 plushies), trains `style train --class-dir/--class-prompt --prior-weight 1.0`, renders the subject (`sks`) in new scenes. **Verified:** the learned fox plush faithfully reappears on a snowy peak AND on a skateboard in a neon city — token binds the subject, prior preservation keeps it a coherent plush (no overfit). sd15, 256² train. ⚠️ slow (~1–2 h). |
| **DreamBooth subject LoRA (SD3.5)** | `dreambooth_sd35.sh` | ✅ | ❌ | **v1.1 — CODE-VERIFIED, render CANNOT-VERIFY on 24 GB.** Same DreamBooth on the SD3.5 **MMDiT** trainer — prior-preservation class loss ported to the rectified-flow objective (`v=ε−x₀`, independent class σ/noise, λ-weighted). **Verified live this cycle:** training ran end-to-end (120 steps @ ~105 s/step; the ~2× per-step cost confirms the class forward runs), the LoRA **merges into the MMDiT correctly** (`sd35-medium` → sd3 pipeline). **Render OOMs at the LoRA-merge step** — even at 512² and even `--device cpu` (the macOS guard keys on system-wide pressure; the full MMDiT+T5+merge can't fit alongside apps). Mechanism is identical to the proven sd15/sdxl DreamBooth. `--base sd35 --class-dir/--class-prompt --prior-weight`. |
| **Textual Inversion training** | `embedding_train.sh` | ✅ | ✅ | **v1.1 — VERIFIED (sd15 / sd21 / sdxl).** `plakat embedding train` learns a new token embedding (a "word") from a few images, model FROZEN (only the placeholder vector(s) train, spliced into the CLIP forward). sd15/sd21 = one CLIP-L vector; **sdxl = a CLIP-L 768d + CLIP-G 1280d pair** (dual-encoder TI). Self-contained: GENERATES a stained-glass style set → trains → loads via `--embedding PATH:trigger` → renders new subjects. **Verified:** `a sgwin cat` clearly takes the learned jewel-glass look on all three. SDXL gotchas: LR **5e-4** (5e-3 over-cooks to a blob), render at scale **0.6** (1.0 tiles the window-motif). TI is subject-dependent (the locomotive takes it less). |
| **Resume training (`--resume`)** | `resume_train.sh` | ✅ | ✅ | **v1.0 — VERIFIED** (GPU run: log shows *"resuming from …-step10 at step 10/30"*, then 11→30 with the step-20 checkpoint). `plakat style train --resume …-step<N>.safetensors --steps M` reloads the adapters + continues the counter (all bases; sd35's fused-qkv PEFT inverse is unit-tested). Strictly additive. (The driver's final render hit memory pressure — the OOM guard aborted cleanly, host survived; re-run the render with freed RAM.) |
| **Regional prompting (`--region`)** | `regional.sh` | ✅ | ✅ | **v1.0 — Part 1, VERIFIED** (sd15 512²: alpine left + tropical right blend into one coherent scene, no center seam — feathered masks). `plakat generate "<base>" --region "x0,y0,x1,y1:prompt" …` — different prompts in different canvas regions of ONE image (MultiDiffusion: per-region predictions blended by bbox masks over the base). Driver: left = alpine, right = tropical. **SD 1.5 / SDXL** (UNet noise blend) **+ SD3.5** (MMDiT velocity blend, `predict_velocity_regional`); native res, reuses the verified tiled blend. Also a per-task **`regions:` scenario key** (all the above models). 2 unit tests (parse + mask). |
| Portrait / identity | `portrait.sh` + `portrait.hjson` | ✅ | ✅ | text personas (script) + `example.png` lookalike (scenario, IP-Adapter-Plus-Face) |
| AnimateDiff (motion) | `animate.sh` | ✅ | ✅ | text→short video; frames in `images/animate/` |
| Stylize: ref-variation + InstantStyle | `stylize.sh` | ✅ | ✅ | apply a reference's *look* to a subject via IP-Adapter — no prompt, no training. **Two paths:** (1) DEFAULT concat = ref-*variation* (content/appearance/palette, NOT texture → stays photoreal); (2) **`--instantstyle`** = true painterly STYLE transfer via a decoupled IP cross-attn on the style block. **SDXL (`up_blocks.0.attentions.1`) is the VERIFIED backbone** — clean watercolour at `--style-scale ~4` + `--strength ~0.8` (warm `figures` + cool `snow` proofs committed). **SD 1.5 (full `up_blocks.1`, all 3 attns) works but is EXPERIMENTAL** — weak style perception (photoreal→faint→melt, no clean-strong window; InstantStyle repo flags this), so not demoed. Driver: concat baseline + 2 SDXL watercolours from `corpus/style/watercolour`. |

## Transforms & post

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| img2img (style transfer, SDXL) | `img2img.sh` | ✅ | ✅ | prompt-steered medium restyle; no LoRA/download |
| Inpaint (masked) | `inpaint.sh` | ✅ | ✅ | `img2img --mask` — repaint a region (the sky) and preserve the rest; committed input + procedural mask, self-contained |
| Outpaint | `outpaint.sh` | ✅ | ✅ | `plakat outpaint` — pad the canvas + paint the new strip in-context (sdxl-inpaint). Clean: the masked region is conditioned on mid-gray (no dark bands) with a binary mask (no feather seams) |
| Upscale (ML) | `upscale.sh` | ✅ | ✅ | Real-ESRGAN ×2 (Metal-safe); ×4 buffers OOM Metal → `--device cpu` |
| Transparent (smart cut-out) | `transparent.sh` | ✅ | ✅ | `plakat transparent --matte` — content-aware **U2Net** smart cut-out: lifts a photoreal subject off ANY background (no chroma) → clean RGBA. Apple-on-a-table → tight cut-out, no fringe. (Corner flood-fill `--tolerance` stays for flat studio backdrops.) Weights auto-download (`vulogov98/u2net-universal`). |
| Segment / select (SAM) | `segment.sh` | ✅ | ✅ | **v1.0 — the compose-&-edit enabler.** `plakat segment` — click an object with point prompts (`--point X,Y[:bg]`, normalized or pixel) + `--grow`/`--feather` → a binary mask via **MobileSAM** (candle-transformers TinyViT, ~40 MB ungated `vulogov98/mobile-sam`, ~0.4 s on Metal). Mask feeds the existing `--mask` consumers. Driver: select the astronaut → `--invert` → `img2img --mask` (sd15) repaints the spacecraft interior into the **lunar surface**, subject preserved. Subject choice matters: cleanly-separable figures swap cleanly; a figure embedded in its props (the rope-rigged dock captain) keeps them. **v1.1: `--depth-band LO,HI`** — a click-free extra mask source via Depth-Anything-V2 (normalized depth, 1.0 = nearest); `0.45,1.0` lifts the foreground astronaut with no points, combinable with `--point` (intersect). **Verified** (`depth-foreground.png`); light enough for CPU (no memory wall). |

## Batch & scripting

| Capability | Driver | Status | Rendered | Notes |
|---|---|---|---|---|
| Scenarios (HJSON batch) | every `*.hjson` | ✅ | ✅ | the corpus is itself the proof |
| Scene × weather axes | `weather-scene.hjson` | ✅ | ✅ | one area (prompt-header) re-lit + re-weathered across both axes |
| Bund scripting | `script.bund` (`script.sh`) | ✅ | ✅ | `plakat run` — load → generate → upscale → save handle-reuse chain (SD 1.5) |
| Tiled hi-res scripting | `tiled_script.bund` (`tiled_script.sh`) | ✅ | ✅ | **v1.0**. `plakat.tiled.enable` routes the SD-family `plakat.generate` through `generate_tiled` — SDXL at 1280² from 768 tiles (above native), the scripting counterpart of `--tiled`. **Base-anchored** (generate a coherent base at tile_size² → upscale latent → tiled img2img REFINE at 0.55) to avoid MultiDiffusion global incoherence; per-tile `synchronize` bounds Metal memory to one tile. Verified: one coherent valley (single river/range/horizon), no seams. |
| Layered scenes (compose) | `compose.sh` + `compose_scene.hjson` | ✅ | ✅ | **v1.0 — Part 1 scene composition.** `plakat compose <scene.hjson>` stacks image layers (z-order) onto a canvas: a background fill + RGBA cut-outs placed by 9-grid/`x,y`, scaled (fraction of canvas), alpha-composited with opacity. **No GPU** — composes committed assets (valley + cottage + pine + balloon). |
| Compile (prose → scenario) | `compile.sh` + `compile/basic.txt` | ✅ | ✅ | **v1.2 — Track C, COMPILE-1.** `plakat compile prompts.txt` → scenario HJSON: blank-line blocks (free text + `key: value` commands) → one task per block; global→scene inheritance with per-command merge (concatenate / accumulate / last-wins); model-family-aware prompt profiles (SD15/SDXL/Flux); auto-negatives from the enhanced positive; `--lint` / `--dry-run`; `scenario -` stdin pipe. **Verified deterministic**: `--no-enhance --no-negative` → **byte-stable** `basic.hjson` (committed), validated via `scenario --dry-run` + pipe. No GPU / network / key. LLM path reuses the `--enhance` provider stack (`prompt::complete`). |
| Compose inline layers (generate/matte) | `compose_generate.sh` + `compose_generate_scene.hjson` | ✅ | ✅ | **v1.1.** A compose layer's pixels now come from `load:` (existing image), `matte:` (U2Net cutout on the fly), or `generate:` (t2i render inline; optional `model`/`seed`/`steps`/`gen_size`). **Verified:** generated beach backdrop + matted astronaut composited with no pre-made assets (`beach-generate-matte.png`). Light (sd15 512² + U2Net) — runs on CPU. |
| Gallery generator | `plakat gallery` | ✅ | n/a | builds this corpus's index |

---

The non-`generate` features (img2img / inpaint / outpaint / upscale /
image-variation / scripting) aren't scenario-drivable, so they land as
small shell / Bund drivers whose outputs write into `images/` alongside
the scenario renders. `plakat gallery images --recursive` indexes them
all uniformly.
