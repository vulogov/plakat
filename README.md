# plakat

[![Crates.io](https://img.shields.io/crates/v/plakat?label=crates.io&color=orange)](https://crates.io/crates/plakat)
[![Latest release](https://img.shields.io/github/v/release/vulogov/plakat?label=release&color=blue)](https://github.com/vulogov/plakat/releases/latest)
[![Downloads](https://img.shields.io/crates/d/plakat?color=brightgreen)](https://crates.io/crates/plakat)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-lightgrey)](https://unlicense.org/)

> **v6.1.0 — `plakat bookart` everywhere, and finished**: the 6.x flagship (RFC BOOKART-1) now lives on **every automation surface** — scenario `type: bookart`, `compile`, Bund `plakat.bookart.*`, the `plakat::api::BookArt` builder, and `--import` into a `plakat photos` album — plus **six** trained origin traditions (Russian / English / Japanese / American / Chinese / European), raster→SVG **tracing**, glyph-driven **initials** (real letterforms, any script), **EPUB** manuscripts, an OpenType **dingbat font** export, and one-command **ink-weight re-finishing**. [Release notes →](https://github.com/vulogov/plakat/releases/tag/v6.1.0) · [Guide →](Documentation/BOOKART.md) · [Transparency →](Documentation/BOOKART_TRANSPARENCY.md)

![](examples/scenario/forest_snow/plakat-1004.png)

Local text-to-image **and animation** across the major open model families —
SD 1.5 / 2.1, SDXL, SD 3.5, Flux, PixArt-Σ, and Stable Cascade — with img2img,
inpaint / outpaint, multi-ControlNet, LoRA / DoRA stacking, **your own trained
style _and_ subject (DreamBooth) LoRAs**, **InstantStyle** painterly style
transfer, **regional prompting**, identity-preserving portraits, AnimateDiff
video, ML upscaling, **SAM object selection**, **smart (U2Net) background
removal**, **layered-scene compositing**, integral artefact compositing,
**controllable synthetic people** (`plakat persona`), and
batch scenarios. All built on
[candle](https://github.com/huggingface/candle). Pure Rust inference. No Python,
no PyTorch, no external T2I services. Models are pulled from HuggingFace and
cached locally.

📸 **[See the gallery →](gallery/)** — example images with their prompts and settings.
🔬 **[Proof corpus →](corpus/)** — a reproducible body of images, plus the tooling to regenerate and index it, proving every pipeline works end to end.

## What's new in 6.1.0 — `plakat bookart` everywhere, and finished

The deferred tail of the 6.0 flagship: wire `bookart` into the rest of plakat and round out the feature.

- **Ecosystem integration parity.** A bookart ornament is now a first-class citizen everywhere: a
  scenario `type: bookart` task, a `compile` `type: bookart` block (prose → a bookart scenario), Bund
  words `plakat.bookart.render` / `.illustrate` / `.origin` / `.technique` (transparent image handles
  that flow into `plakat.save` / `.upscale`), the library facade `plakat::api::BookArt`, and
  `bookart render|illustrate --import <album>` to land an ornament — with its **recipe sidecar + PNG
  `tEXt` chunk** (origin / technique / spec-hash) — straight into a `plakat photos` album.
- **Six origin traditions.** Three new trained sd15 origin LoRAs — **american** (Howard Pyle),
  **european** (Gustave Doré engraving), **chinese** (woodblock outline) — join russian / english /
  japanese, all hosted and auto-resolved. `bookart origins` lists them; an optional
  `assets/bookart/lexicon.hjson` adds your own traditions with no rebuild.
- **Raster→SVG tracing** (`bookart vectorize`, and `--svg` on the diffusion/composite tiers; feature
  `bookart-trace`) · **glyph-driven initials** (`ornament.glyph` + `--font` renders a real letterform —
  any script, incl. Cyrillic — inside an ornamental frame) · **EPUB manuscripts**
  (`bookart manuscript book.epub`; feature `epub`) · an OpenType **dingbat font** (`bookart font` — type
  `a`–`h`, get a fleuron) · **ink-weight / transparency re-finishing** without re-rendering
  (`render --cache-raw` → `bookart edit --ink-weight …`) · richer procedural bands (Greek-key,
  L-system foliate scroll, knotwork interlace) and band-shaped composite cartouches.

*(`bookart-trace` and `epub` are opt-in Cargo features — the prebuilt release binaries build the
default feature set, so those two need a source build with `--features bookart-trace` / `--features epub`.
Glyph initials, the dingbat font, and all six origin LoRAs work in the release binaries.)*

## What's new in 6.0.0 — `plakat bookart`

The 6.x flagship (RFC BOOKART-1): compose **reusable, print-ready, transparent black-and-white book
ornaments** from a small HJSON spec — the decorative-ornament sibling of `persona`. Where a prompt
gives an uneven grey picture with an opaque box behind it, `bookart` treats an ornament as structured
data: resolved deterministically, rendered by a **hybrid router**, made transparent by a B/W-native
model, and placed on an exact print canvas.

```bash
plakat bookart new alice.hjson --origin russian --technique woodcut --type headpiece
plakat bookart render alice.hjson --out headpiece.png          # transparent, page-sized PNG
plakat bookart illustrate "a firebird among oak branches" --origin russian --out plate.png
plakat bookart kit alice.hjson --out kit/                       # a coherent matched set + contact sheet
plakat bookart manuscript book.md --kit alice.hjson --out ornaments/   # a whole book's per-chapter set
```

Highlights: a **hybrid render router** — *procedural* (vector-native guilloché borders, rosettes,
corners — crisp at any DPI, **zero weights**), *diffusion* (pictorial ornament in a trained tradition
via sd15 + an origin LoRA), and *composite* (a procedural frame with a diffusion picture inlaid);
**B/W-native transparency** (ink darkness *is* opacity — no halo, no page haze); a **symmetry engine**
(a geometric guarantee diffusion can't hold); **exact print sizing** (named page sizes → px at DPI, DPI
embedded); a print/ink **scorecard**; opt-in **born-vector SVG**; and the flagship **kit** (a coherent
matched set) + **manuscript** (a book's per-chapter ornaments, one command). Three origin LoRAs
(russian / english / japanese) are hosted and auto-resolved; a **generic line-art path** covers every
origin×technique without a LoRA. Fully additive. Start at
[`Documentation/BOOKART.md`](Documentation/BOOKART.md); the transparency model is in
[`Documentation/BOOKART_TRANSPARENCY.md`](Documentation/BOOKART_TRANSPARENCY.md).

*(6.0 shipped the standalone CLI; **6.1 added** the scenario / compile / Bund / library-API integration,
raster→SVG tracing, glyph-driven initials, three more origin traditions, an EPUB manuscript input, and an
OpenType dingbat font — see the 6.1.0 notes above.)*

## What's new in 5.0.0 — `plakat persona`

The 5.x flagship (RFC PERSONA-1): compose a **specific, reusable synthetic person** from a small HJSON
spec and render that same person recognisably across scenes and model families. Text prompts are a
poor instrument for identity — a mole moves between renders, a scar lands anywhere. `persona` treats a
person as structured data: resolved deterministically, conditioned geometrically, small details
realised by *compositing* (not prompting), anchored to one identity via a cast reference set, and
**measured** by a scorecard.

```bash
plakat persona new alice.hjson --name alice        # scaffold, or --tui to author interactively
plakat persona cast   alice.hjson --model sd15      # render + score → a coherence-checked reference set
plakat persona render alice-persona --scene "in a sunlit garden"   # into any scene (universal swap bridge)
plakat persona verify alice.hjson --image out.png   # the scorecard: did the render match the spec?
plakat persona repair alice.hjson --image out.png --attr eyes.color   # fix one thing, keep the render
```

Fully additive — no existing command or output changes. Highlights: a WFLW-98 geometry engine (pure,
no weights), a localized-detail subsystem (moles/scars/birthmarks/freckles/jewelry/dentition composited
at anatomical anchors), per-family calibration, three honest identity tiers (IP-Adapter · universal
face-swap · baked LoRA), multiperson attribution, a class-aware edit/repair loop, and a headless
interview with a live wireframe TUI. The render path is hardened against the characteristic
text-to-portrait failure modes — extreme face-macros, stylised non-photos, gibberish signage, and
jewelry pasted over hair — via a framing guard, a bust-grounded geometry conditioning map, a no-face
retry on both identity tiers, and occlusion-aware compositing. Start at
[`Documentation/PERSONA.md`](Documentation/PERSONA.md); worked demo in
[`corpus/PERSONA_CORPUS.md`](corpus/PERSONA_CORPUS.md).

## What's new in 4.11.0 — finishing the edit verbs

The two follow-ups deferred from the 4.9/4.10 edit-verbs work:

- **`remove --what` now SAM-refines the mask** — the OWL-ViT box is tightened to the object's actual
  outline with SAM (a foreground point at the box center + background hints just outside the edges),
  so the inpaint follows the object, not a rectangle. `--box-only` keeps the raw rectangle.
- **`replace-bg --keep "<subject>"`** — choose the kept subject by text (OWL-ViT → SAM) instead of the
  automatic U2Net salient matte. Handy when the salient object isn't the one you want.

```bash
plakat remove photo.png --what "the dog"
plakat replace-bg street.png --keep "the red car" --prompt "a showroom"
```

Both reuse the 4.10 OWL-ViT detector + SAM; default output stays byte-identical, everything is additive.

**Earlier releases** (v0.13 – 4.5):
[`Documentation/RELEASE_HISTORY.md`](Documentation/RELEASE_HISTORY.md).

## Install

`plakat` runs on every platform candle supports. Pick a backend at install
time — the CPU-only default works everywhere but is slow at real sizes.

```bash
# macOS — Apple Silicon GPU via Metal
cargo install plakat --features metal

# Linux — NVIDIA GPU via CUDA
cargo install plakat --features cuda
cargo install plakat --features cudnn        # CUDA + cuDNN convolutions

# Anywhere — CPU only
cargo install plakat
```

Optional features (off by default): `templates` (Tera pre-pass for `compile`),
`shaped-labels` (TrueType map labels for non-Latin scripts), and `onnx`
(`plakat convert-onnx`, to rebuild the hosted face-model weights yourself — needs
`protoc` at build time, which is why it's opt-in; everyone else downloads the
pre-built weights).

Requires Rust 1.85+ (edition 2024). On Apple hardware, see
[`Documentation/APPLE_REQUIREMENTS.md`](Documentation/APPLE_REQUIREMENTS.md)
for the minimum / recommended chip + memory tiers and expected
per-image speeds.

## Quick start

Prefer an interactive workflow? `plakat ui` is a full terminal UI —
load a model once and *talk* to it: conversational generation +
refinement, inline images, history, people, LoRA search/apply, prose →
scenario compile, and inpaint-mask painting, all keyboard-driven. See
[`Documentation/Tutorials/UI_TUTORIAL.md`](Documentation/Tutorials/UI_TUTORIAL.md).

```bash
plakat ui            # the interactive terminal UI
```

Or drive it from the command line:

```bash
# Text-to-image with SD 1.5
plakat generate "a brutalist poster of a whale, watercolor" --seed 42

# A1111-style attention syntax — emphasize "neon", dial down "city"
plakat generate "a cyberpunk (neon:1.4) street market in a [city]" \
    --model sd15 --seed 42

# Photo-guided portrait (IP-Adapter-Plus-Face)
plakat portrait "cinematic close-up, soft Rembrandt lighting" \
    --photo face.jpg --face-strength 0.8

# Image-to-image: restyle an existing image
plakat img2img photo.jpg --prompt "watercolor painting of the same scene"

# Inpaint: replace just the masked region (white = inpaint here)
plakat img2img photo.jpg --mask sky.png \
    --prompt "dramatic stormy sky, lightning"

# Outpaint: extend a photo past its borders
plakat outpaint photo.jpg --prompt "wide mountain valley, panorama" \
    --left 512 --right 512 --model sdxl-inpaint

# FLUX.1-dev quantized — runs on 16 GB consumer GPUs
plakat generate "..." --model flux-dev-gguf --flux-quant-level Q5_K_M \
    --quantize-t5 --size 1024x1024

# Flux Inpainting via Flux.1-Fill-dev
plakat img2img init.png --mask region.png --model flux-fill-dev \
    --prompt "stained glass window in the wall"

# Tiled hi-res Flux (4K outputs without OOM)
plakat generate "ultra-detailed architectural diagram" \
    --model flux-dev --size 3072x2048 \
    --tiled --tile-size 1024 --tile-stride 768

# Stable Diffusion 3.5 — Stability's MMDiT family
plakat generate "..." --model sd35-medium  # 2.5B params
plakat generate "..." --model sd35-large   # 8B params, the flagship
plakat generate "..." --model sd35-large-turbo  # 4-step distillation

# NF4 Flux — bitsandbytes 4-bit quantization. ~6 GB transformer.
plakat generate "..." --model flux-dev-nf4

# Flux Redux — image-conditioned Flux via SigLIP. Stack up to 4 refs.
plakat generate "in this style" --model flux-dev \
    --redux-image style.png:weight=0.7 \
    --redux-image subject.png:weight=0.4

# Hyper-FLUX / FLUX-Turbo presets — 8-step distillations
plakat generate "..." --model flux-dev --fast hyper-8

# LCM-LoRA SDXL — 4-step SDXL inference at ~5× the speed
plakat generate "a fantasy castle on a misty mountaintop" \
    --model sdxl --fast lcm-sdxl

# Same recipe for SD 1.5 — 4-step inference on the smaller backbone
plakat generate "a fantasy castle on a misty mountaintop" \
    --model sd15 --fast lcm-sd15

# ControlNet: layout-guided generation. Five conditioners ship with
# auto-annotators (depth, canny, openpose, lineart, softedge); each
# accepts either `from=PATH` (auto-annotate any photo) or
# `image=PATH` (use a pre-rendered map). Works on SD 1.5 / 2.1 /
# SDXL, Flux (Union Pro v2), and SD3 / SD3.5 (InstantX family).
plakat generate "a fox in tall grass" \
    --control-spec 'depth:from=reference_photo.jpg'

# Stack multiple conditioners — residuals are summed per denoise step,
# diffusers-style. Useful for "preserve this layout AND this pose":
plakat generate "knight on a stone bridge, cinematic" --model sdxl \
    --control-spec 'depth:from=scene.jpg:strength=0.8' \
    --control-spec 'openpose:from=person.jpg:strength=0.6'

# Wildcards in the prompt: `{a|b|c}` inline alternation + file-backed
# `__name__` random picks (Auto1111 / NovelAI grammar).
plakat generate "a {red|blue|green} fox in __warm-colors__ light" \
    --wildcard-dir ./wildcards --seed 42

# ADetailer: post-t2i face refinement via SCRFD + per-face img2img.
plakat generate "a couple at a forest cabin" \
    --model sd15 --size 768x1024 --adetailer

# Hires fix: generate at trained resolution, upscale, refine.
plakat generate "a vintage travel poster of Tokyo at night" \
    --model sd15 --size 768x768 \
    --hires-fix --hires-upscaler real-esrgan-x2 --adetailer

# `--grid` bundles a `--count N` sweep into a single shareable PNG.
# Also works on `plakat img2img` / `plakat portrait` / `plakat outpaint`
# (v0.18); the grid filename tracks the backbone prefix.
plakat generate "a peaceful koi pond" \
    --model sd15 --count 9 --seed 1000 --grid

# Live preview during long denoise runs — writes plakat-<seed>-preview.png
# every N steps (cheap latent → RGB projection; microseconds per write).
plakat generate "a fantasy castle on a misty mountaintop" \
    --model sd15 --steps 28 --preview-every 4 --size 768x768

# Civitai: browse + download community assets straight from the CLI.
plakat civitai search "watercolor" --type lora
plakat civitai download 12345

# Or use the LoRA spec shorthand — downloads + caches on first use.
plakat generate "a watercolor fox in tall grass" \
    --model sd15 --lora civitai:12345:0.7

# v0.18: A1111-style inline <lora:> tags in the prompt itself
# (matches the format Civitai LoRA cards embed in their examples).
plakat generate \
    "a watercolor fox in tall grass <lora:civitai:12345:0.7>" \
    --model sd15

# v0.18: BREAK keyword to chunk past CLIP's 77-token cap.
# Each chunk gets its own 77-token CLIP context.
plakat generate \
    "first half of an elaborate prompt with subject + composition \
     BREAK \
     second half with style + lighting + medium notes" \
    --model sd15

# v0.18: local LLM prompt enhancer (no API key — runs in-process).
plakat generate "a knight" --enhance local --model sd15

# v0.18: enhance auto — DeepSeek → Gemini → local based on env vars.
plakat generate "a knight" --enhance auto --model sd15

# v0.18: Flux Kontext for image editing — input is the reference,
# prompt describes the edit. Reference is VAE-encoded and
# sequence-concat'd onto the noise tokens.
plakat img2img photo.png --model flux-kontext-dev \
    --prompt "make the lighting golden hour, warm tones"

# Same recipe via GGUF for 16 GB GPUs.
plakat generate "make it sunset" --model flux-kontext-dev-gguf \
    --concept-image photo.png --flux-quant-level Q5_K_M

# v0.18: read back the recipe (prompt, seed, LoRAs, sampler) from
# any plakat-written PNG. Pipe --json-only to jq for scripting.
plakat metadata ./out/plakat-42.png
plakat metadata ./out/plakat-42.png --json-only | jq .seed

# v0.19: clone a PNG's recipe into a re-runnable shell command
plakat clone ./out/plakat-42.png

# v0.19: bundled negative-prompt presets
plakat generate "a sunlit forest" --model sd15 --negative-preset photo
plakat generate "anime girl, masterpiece" --model sd15 \
    --negative-preset anime --negative "purple hair"

# v0.19: WebP output for smaller share-ready files
plakat generate "..." --model sd15 --format webp

# v0.19: local prompt enhancer — disk cache makes repeat runs instant
plakat generate "a knight" --enhance local --enhance-cache --model sd15

# v0.19: doctor --json for CI / scripting
plakat doctor --json | jq -e '.device.aligned == true'

# v0.19: scenario --only / --limit / --dry-run for partial reruns
plakat scenario big.hjson --dry-run                       # validate
plakat scenario big.hjson --limit 3                       # first 3 tasks
plakat scenario big.hjson --only forest_scene,desert_scene
plakat scenario big.hjson --resume                        # skip done tasks

# v0.19: plakat animate --resume for crash recovery on long animates
plakat animate --from "..." --to "..." --frames 24 \
    --out ./morph --resume

# v0.19: Kontext + ControlNet composition (preserve depth structure)
plakat generate "make the lighting golden hour" \
    --model flux-kontext-dev --concept-image input.png \
    --control-spec 'depth:from=input.png:strength=0.7'

# v0.19: Kontext + Redux composition (edit + style transfer)
plakat generate "the same scene at golden hour" \
    --model flux-kontext-dev --concept-image input.png \
    --redux-image style_ref.png:weight=0.5

# Prompt-morph animation — interpolates two prompts over N frames.
# v0.18 adds SDXL on top of SD 1.5 / SD 2.1.
plakat animate \
    --from "a photo of a fox in a meadow" \
    --to "a photo of a cat in a meadow" \
    --frames 24 --seed 42 --gif --out ./fox_to_cat

# Weighted multi-reference portrait: merge facial features
# from several photos (averaging, aging, blending)
plakat portrait "a portrait, soft window light" \
    --photo person_age_25.jpg:0.6 \
    --photo person_age_55.jpg:0.4 \
    --face-strength 0.85

# Composite named cutout artefacts (trees, sky elements, houses, ...) 
# into named zones of the generated image. Add --artefact-blend for a
# masked img2img pass that smooths the pasted edges; --smart-zones
# derives zones from the image's own depth + luminance.
plakat generate "a green meadow under a blue sky" \
    --artefact oak@middle_plan/left \
    --artefact sun@sky/right \
    --artefact-blend --smart-zones

# Apply a bundled art style by name
plakat generate "a fox in tall grass" --style watercolor

# Detect a style from a reference photo, then apply it
plakat generate "a fox in tall grass" --style-ref ./inspiration.jpg

# Batch generation from a scenario file
export DEEPSEEK_API_KEY=sk-...
plakat scenario examples/scenario.hjson

# Resume a crashed batch — skips tasks whose output PNGs already exist
plakat scenario examples/scenario.hjson --resume

# Real-ESRGAN upscale to 4×
plakat upscale --in small.png --out big.png --method real-esrgan-x4
```

Every output PNG (from `generate`, `img2img`, `portrait`, etc.) ships
with an A1111-compatible `parameters` tEXt chunk + a sibling
`<filename>.json` carrying the structured recipe. Drop a PNG onto
A1111 Web UI / Civitai / ComfyUI / sd-prompt-reader to see the
prompt, seed, model, LoRAs inline. Pass `--no-metadata` for anonymous
PNGs.

Run `plakat <CMD> --help` for the flags on each subcommand.

## Subcommands

| Command | What it does |
|---|---|
| `generate <PROMPT>` | Single-shot text-to-image. SD 1.5 / 2.1 / SDXL / SDXL-Turbo / Flux (BF16, GGUF, NF4, **Kontext-dev** v0.18 — composes with ControlNet + Redux v0.19, **+ `--tiled` v0.20**) / SD3 / SD3.5. Built-in wildcards, A1111 attention syntax, inline `<lora:>` tags, `BREAK` keyword (SD-family), CLIP-skip, ADetailer, Hires fix, ControlNet, LoRA stacking, tiled hi-res, Flux Redux + concept variants, `--grid` bundling, `--preview-every`, PNG metadata + JSON sidecar, `--negative-preset` (+ user catalog v0.20), `--format webp` (Flux + SD3 in v0.20), `--enhance local\|auto` + cache/temp/tokens/system + **`--enhance-keep-original`** (v0.20), **`--recipe FILE.json`** (v0.20), **`--import <album>`** (v3.0 — land the output in a `plakat photos` album with its full recipe; also on `upscale`/`portrait`/`multiperson`/`img2img`/`outpaint`/`stylize`/`relight`). |
| `img2img <INPUT>` | Image-to-image transform with `--prompt`; supply `--mask` for masked inpaint instead. SD 1.5 / 2.1 / SDXL, Flux (`--model flux-dev` for img2img, `--model flux-fill-dev` for inpaint, **`flux-kontext-dev`** for image editing — v0.18, with `--tiled` for 4K+ inpaint), and SD3 / SD3.5 (RePaint-style inpaint, `--tiled` for 2K+ outputs). v0.18: `--aspect 16:9` size derivation. |
| `outpaint <INPUT>` | Extend an image past its borders. Per-side `--left`/`--right`/`--top`/`--bottom` or `--expand N` for all four. Defaults to `sdxl-inpaint`; `flux-fill-dev` works too. |
| `portrait <PROMPT>` | Portrait generation, optionally guided by one or more reference photos with weighted merging. IP-Adapter-Plus-Face or FaceID on SD 1.5 / SDXL. |
| `persona <SPEC>` | **v5.0 flagship (RFC PERSONA-1).** Compose a *specific, reusable synthetic person* from a small HJSON `PersonaSpec` and render that same person recognisably across scenes and model families. A WFLW-98 geometry engine (pure, no weights), landmark-anchored detail compositing (moles / scars / birthmarks / freckles / jewelry / dentition), per-family calibration, three identity tiers (IP-Adapter · universal face-swap · baked LoRA), multiperson attribution, a class-aware edit/repair loop, and a headless interview with a live wireframe TUI. Subcommands: `new` · `lint` · `show` · `geometry` · `calibrate` · `cast` · `render` · `verify` · `composite` · `repair` · `diff` · `bake` · `interview`. Fully additive. See [`PERSONA.md`](Documentation/PERSONA.md); worked demo in [`corpus/PERSONA_CORPUS.md`](corpus/PERSONA_CORPUS.md). |
| `bookart <SPEC>` | **v6.0 flagship (RFC BOOKART-1).** Compose *reusable, print-ready, transparent black-and-white book ornaments* from a small HJSON spec, in a chosen illustration tradition × technique, at an exact page size. A hybrid render router (vector-native **procedural** guilloché/borders/rosettes with zero weights · **diffusion** pictorial via sd15 + origin LoRA · **composite** frame + inlay), B/W-native transparency, a symmetry engine, a print/ink scorecard, opt-in born-vector SVG, and the flagship coherent **kit** + **manuscript** (a book's per-chapter ornaments). Subcommands: `new` · `lint` · `show` · `render` · `illustrate` · `verify` · `kit` · `manuscript` · `proof` · `diff` · `edit` · `blend`. Fully additive. See [`BOOKART.md`](Documentation/BOOKART.md). |
| `photos [DIR]` | **v3.0 flagship.** TUI photo & image collection manager: folder tree + thumbnail grid (RAW + every common format, EXIF), full image view, non-destructive curation (1–5 ratings, flag/reject, colour labels, tags) persisted per-album in a plain `album.hjson`, a live filter grammar + culling loupe, and a filesystem watcher. On by default (needs a graphics-capable terminal). See [`PHOTOS_TUTORIAL.md`](Documentation/Tutorials/PHOTOS_TUTORIAL.md). |
| `scenario <FILE>` | Batch generation from an HJSON config: scenes × weather × tasks × personas × styles. `--resume` skips already-generated outputs; v0.19 adds `--only NAME[,NAME,…]` (named-task filter), `--limit N` (first N tasks), polished `--dry-run` summary. `-` reads stdin. |
| `compile <PROMPTS>` | **v1.2**. Compile a prose `prompts.txt` (blank-line scenes + `key: value` commands) into a `scenario` HJSON — one task per block, model-family-aware prompt rewriting + auto-negatives via the `--enhance` stack. `--no-enhance`/`--no-negative` (deterministic), `--lint`, `--dry-run`, `--diff`, `--decompile`, `--compile-cache`. See [`COMPILE.md`](Documentation/COMPILE.md). |
| `map <DESCRIPTION>` | **v1.4–1.8**. Turn a prose world description into a fantasy map: LLM parse → `MapSpec v2`, then a geometry engine (terrain → hydrology → coastline → biomes → landmarks → roads → composite) — or, for a city/town spec (`urban` block), an **urban street graph** (wall, gates, blocks, waterfront). **`--map-render PATH`** writes the finished labelled map (`--map-style`, **`--map-urban-layout radial\|grid\|organic`**, **`--map-erosion <0..>1>`**); **`--map-render-sd PATH`** paints it with SD (`--map-sd-model`/`--map-sd-lora`/`--map-sd-tile`); **`--map-export-svg`/`--map-export-geojson`** export vectors. Also `--map-spec`, `--map-dump-{spec,…,features,conditioning,streets}`, `--seed`, `--map-tiles`/`--map-scale`. Geometry is a pure fn of (spec, seed) — byte-stable. A first-class step in `scenario` (`type: map`), `compile` (`map:` block), and `run` scripting (`plakat.map.*`). See [`ROADMAP_1.8.0.md`](Documentation/ROADMAP_1.8.0.md). |
| `style {detect,list,show,init,probe,train}` | Inspect, detect, and bootstrap art-style catalogs; **`train`** (v0.45) learns a style LoRA from a folder of images (SD 3.5). |
| `artefact {list,show}` | Inspect the artefact library (PNG cutouts placeable into named zones of generated images). |
| `civitai {search,info,download}` | Browse + download Civitai community assets (LoRAs, checkpoints, embeddings, ControlNet variants). |
| `embedding {info,flux-ip-adapter-info}` | Inspect Textual Inversion `.safetensors` files + XLabs Flux IP-Adapter weights. |
| `animate --from A --to B --frames N` | Prompt-morph animation: lerp text-encoder embeddings between two prompts to produce a smooth N-frame sequence at a fixed seed. Optional GIF bundling. SD 1.5 / SD 2.1 / SDXL + **Flux Dev / Schnell (v0.20)** via CLIP-L pooled + T5 lerp + flow-match. v0.19 adds `--resume` for crash recovery. |
| `stylize` | IP-Adapter style transfer on SD 1.5 (IN + REF → OUT). |
| `upscale` | Resize, classical or Real-ESRGAN. |
| `transparent` | Make every pixel matching the corner colour transparent. |
| `models {search,recommend,size,pull,ls,rm,aliases}` | Browse HuggingFace and manage the local cache. v0.20 adds **`aliases`** — enumerate every `--model` short-name plakat understands, grouped by family. `--family flux`, `--repo` (bare ids for piping), `--gated`. |
| `init [DIR]` | **v0.20**. Bootstrap a runnable starter project — `scenario.hjson` + `wildcards/` + `.gitignore`. Targets `sd15` + `enhancer: local` so first-run users with no HF token / no API key can generate end-to-end. `--minimal` writes only the scenario; `--force` overwrites. |
| `doctor` | Health-check FaceID / SCRFD setup, plus (v0.18) build/runtime device match, libcuda driver shim, HF cache disk usage. v0.19 adds `--json` for structured CI / scripting output. |
| `verify` | **v2.0**. Model-correctness harness (pure Rust — no python/torch). `--tier 0` structural/determinism (no downloads), `--tier 1` per-module correctness vs frozen reference tensors fetched from HF, `--tier 2` end-to-end perceptual. `--model`, `--golden-dir`, `--json`. See [`VERIFY.md`](Documentation/VERIFY.md). |
| `inspect <FILE>` | List every tensor in a `.safetensors` file. |
| `metadata <FILE.png>` | Read the v0.17 Auto1111 `parameters` PNG tEXt chunk + sibling `.json` sidecar. Reverse of the metadata write path. `--json-only` / `--params-only` to filter. |
| `clone <FILE.png>` | v0.19. Translate a PNG's metadata into a re-runnable `plakat generate` shell command. JSON sidecar preferred; falls back to parsing the Auto1111 chunk (works on Civitai uploads + A1111 Web UI outputs). `--one-line` for piping. |
| `run <SCRIPT.bund> \| --repl` | **v0.21**, **expanded in v0.22, deferrals closed in v0.23**. Drive plakat from a stack-based Bund script. v0.23 ships 33 host words across 9 namespaces (`plakat.lora.*`, `plakat.controlnet.*`, `plakat.refiner.*`, `plakat.adetailer.*`, `plakat.hires.*`, `plakat.artefact.*`, `plakat.style.*`, `plakat.enhance`, `plakat.inpaint`, core image surface) plus a pipeline cache + SD/Flux/SD3 all three families + 60+ config keys + Flux/SD3 ControlNet + SDXL refiner + clip_skip. Interactive REPL with `--repl`. See [`SCRIPTING.md`](Documentation/SCRIPTING.md) for the full reference and [`SCRIPTING_TUTORIAL.md`](Documentation/Tutorials/SCRIPTING_TUTORIAL.md) for the walkthrough. |

## Documentation

- **[`API.md`](Documentation/API.md)** — use plakat as a **Rust library**
  (`plakat::api`): a small builder API covering every non-UI feature
  (generate, img2img, upscale, relight, portrait, multiperson, map,
  animate, training, verify). Full reference with examples.
- **[Tutorials](Documentation/Tutorials/)** — beginner-friendly,
  step-by-step walkthroughs. Start here if you're new to plakat or
  text-to-image generation. See
  [Tutorials/README.md](Documentation/Tutorials/README.md) for the
  recommended reading order. Highlights:
  - [`GENERATE_TUTORIAL.md`](Documentation/Tutorials/GENERATE_TUTORIAL.md) —
    the foundation. Wildcards, A1111 attention syntax, CLIP-skip,
    ADetailer, Hires fix, Civitai, live preview, PNG metadata,
    grid output, Textual Inversion all sectioned within.
  - [`FLUX_TUTORIAL.md`](Documentation/Tutorials/FLUX_TUTORIAL.md) +
    [`SD3_TUTORIAL.md`](Documentation/Tutorials/SD3_TUTORIAL.md) —
    the modern model families.
  - [`CIVITAI_TUTORIAL.md`](Documentation/Tutorials/CIVITAI_TUTORIAL.md) —
    browsing, downloading, and using Civitai community assets.
  - [`ANIMATE_TUTORIAL.md`](Documentation/Tutorials/ANIMATE_TUTORIAL.md) —
    prompt-morph animation via `plakat animate`.
  - [`ADVANCED_PROMPTING_TUTORIAL.md`](Documentation/Tutorials/ADVANCED_PROMPTING_TUTORIAL.md) —
    A1111 attention syntax, the `BREAK` keyword for chunking past
    CLIP's 77-token cap, and inline `<lora:>` tags. Per-backbone
    composition matrix.
  - [`PROMPT_ENHANCER_TUTORIAL.md`](Documentation/Tutorials/PROMPT_ENHANCER_TUTORIAL.md) —
    `--enhance deepseek | gemini | local | auto`. The local arm
    runs Qwen2.5-1.5B in-process with no API key.
  - [`METADATA_TUTORIAL.md`](Documentation/Tutorials/METADATA_TUTORIAL.md) —
    `plakat metadata FILE.png` recovers the recipe (prompt, seed,
    LoRAs, sampler) from any plakat / A1111 / Civitai PNG. v0.19's
    companion `plakat clone PNG` emits a re-runnable shell command
    from that recipe.
  - [`SCENARIOS_TUTORIAL.md`](Documentation/Tutorials/SCENARIOS_TUTORIAL.md) —
    batch generation via HJSON. Cross-product expansion, per-task
    overrides, partial-rerun filters (v0.19 `--only` / `--limit`),
    real-world series-production examples.
  - [`OUTPAINT_TUTORIAL.md`](Documentation/Tutorials/OUTPAINT_TUTORIAL.md) —
    `plakat outpaint INPUT.png` grows an image's canvas. Per-side
    flag grammar, VAE-snapped dimensions, model choice, iterative-
    stage workflow.
  - [`SCRIPTING_TUTORIAL.md`](Documentation/Tutorials/SCRIPTING_TUTORIAL.md) —
    **v0.21**, **expanded in v0.22, deferrals closed in v0.23.**
    Drive plakat from a Bund script (`plakat run SCRIPT.bund`
    or `plakat run --repl`). Stack-based syntax, 33 `plakat.*`
    host words across 9 namespaces (incl. `plakat.style.*` and
    `plakat.inpaint`), pipeline cache, all three model families,
    SDXL refiner + clip_skip + Flux/SD3 ControlNet, composition
    patterns. See also [`SCRIPTING.md`](Documentation/SCRIPTING.md)
    for the reference.
  - Specialized portrait recipes:
    [aging interpolation](Documentation/Tutorials/PORTRAIT_HOW_TO_AGE.md)
    and
    [blending parents into a child portrait](Documentation/Tutorials/PORTRAIT_CHILD_PHOTO.md).
- **[Reference manuals](Documentation/)** — exhaustive per-feature
  documentation:
  - [`GENERATE.md`](Documentation/GENERATE.md) — text-to-image,
    schedulers, LoRAs, scenarios, upscaling, refiner, the `plakat
    civitai` / `plakat embedding` / `plakat animate` subcommands.
  - [`PERSONA.md`](Documentation/PERSONA.md) — portraits, identity
    preservation, ArcFace / SCRFD setup, multi-persona compositing.
  - [`STYLES.md`](Documentation/STYLES.md) — style catalogs, the
    `plakat style` subcommands, building your own catalog.
  - [`ARTEFACTS.md`](Documentation/ARTEFACTS.md) — placing named PNG
    cutouts into named zones of generated images.
  - [`IMG2IMG.md`](Documentation/IMG2IMG.md) — image-to-image and
    inpaint via `plakat img2img`.
  - [`CONTROLNET.md`](Documentation/CONTROLNET.md) — ControlNet
    conditioning (depth, canny, openpose, lineart, softedge) for
    SD 1.5 / 2.1, SDXL, Flux (Union Pro v2), and SD3 / SD3.5
    (InstantX adapter family).

## Reproducibility

A given `--seed` makes a render repeatable **on the same machine + backend**.
Across machines/backends — and on **Metal specifically** — renders are *not*
bit-reproducible: Apple Silicon's GPU kernels are non-deterministic, so identical
inputs can differ slightly between runs. Every output still embeds its full
recipe (prompt, seed, settings) as a PNG `parameters` chunk + JSON sidecar, and
`generate --reproducibility-check` re-runs a recipe to measure the drift.

## Releases

Pre-built binaries for the 0.7+ tags are attached to each
[GitHub release](https://github.com/vulogov/plakat/releases). The
release workflow ([`.github/workflows/release.yml`](.github/workflows/release.yml))
builds five archives on every `v*` tag push:

| Archive | Target | Backend | Notes |
|---|---|---|---|
| `plakat-vX.Y.Z-aarch64-apple-darwin.tar.gz` | aarch64-apple-darwin | Metal (Apple Silicon GPU) | |
| `plakat-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` | x86_64-unknown-linux-gnu | CPU only | Works on any Linux x86_64. |
| `plakat-vX.Y.Z-x86_64-unknown-linux-gnu-cuda.tar.gz` | x86_64-unknown-linux-gnu | **CUDA + CPU fallback** | Requires the NVIDIA CUDA 12 runtime libraries on the host (`libcudart.so.12`, etc.). |
| `plakat-vX.Y.Z-aarch64-unknown-linux-gnu.tar.gz` | aarch64-unknown-linux-gnu | CPU only | |
| `plakat-vX.Y.Z-x86_64-pc-windows-msvc.zip` | x86_64-pc-windows-msvc | CPU only | |

Each archive contains the `plakat` binary, `LICENSE`, `README.md`, and
the bundled `assets/` (artefact library + style catalog). A
`SHA256SUMS` file is attached to the same release for verification:
`shasum -a 256 -c SHA256SUMS`.

**Picking the right Linux binary**: if you have an NVIDIA GPU AND the
CUDA 12 runtime installed (`apt install nvidia-cuda-toolkit` on Debian/
Ubuntu, or via the NVIDIA installer), grab the `-cuda` variant —
it'll auto-detect your GPU and run inference there. Otherwise grab
the plain `x86_64-unknown-linux-gnu` archive (no CUDA runtime
dependency).

Intel Macs (`x86_64-apple-darwin`) are not pre-built — Apple Silicon
is the supported macOS target (Metal is the only GPU backend candle
offers on macOS). Install from source on Intel with
`cargo install plakat`.

## License

Free and unencumbered software released into the public domain
([Unlicense](https://unlicense.org/)).
