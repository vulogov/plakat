# plakat

![](examples/scenario/forest_snow/plakat-1004.png)

Local text-to-image generation, style transfer, LoRA stacking, ML upscaling,
identity-preserving portraits, and batch scenarios — all built on
[candle](https://github.com/huggingface/candle). Pure Rust inference. No
Python, no PyTorch, no external T2I services. Models are pulled from
HuggingFace and cached locally.

## What's new in v0.20 — recipe replay, project bootstrap, Flux animate, Kontext + tiled

Nine features in three groups. v0.20 picks up where v0.19 left
off on workflow polish (recipe-driven replay, project
bootstrap, user-defined negative-preset catalogs, Civitai
trigger-word display), then lands two unlock-grade composition
wins: Flux Kontext + tiled denoise for hi-res reference edits,
and Flux animate (T5 + CLIP-L lerp, flow-match per frame).

### Top picks (3 features)

- **`plakat generate --recipe FILE.json`**. Replay any prior
  generation from its JSON sidecar. Recipe fields fill in only
  where the CLI didn't set the flag explicitly — `--model`,
  `--seed`, `--negative`, etc. pass through unchanged when you
  override them. Useful for "re-render at higher steps" /
  "swap one LoRA, keep everything else" iterations. The
  prompt is never overridden — the recipe is structural, the
  prompt is creative.

  ```bash
  # Re-render a v0.17 generation at higher quality
  plakat generate "$(cat in.prompt)" --recipe in.json --steps 50
  ```

- **Flux + SD3 WebP output**. v0.19 shipped WebP on SD-family
  only with a warn+fallback for the modern backbones. v0.20
  threads `--format png|webp` through the Flux and SD3
  pipelines too. WebP is ~30% smaller at perceptually-
  equivalent quality; the JSON sidecar still works on every
  backbone (so `plakat metadata` / `plakat clone` round-trip
  unchanged).

- **Civitai LoRA trigger-word display**. When a `--lora
  civitai:NNNNNN` resolves (cache hit or fresh download),
  plakat now prints the LoRA's trained trigger words inline:

  ```text
    ✦ Civitai LoRA 2595428 (v2614696) trigger words: watercolor_(medium), some_trigger
      → consider adding these to your prompt for the LoRA to activate
  ```

  Silent LoRAs (no apparent effect because triggers were
  missing from the prompt) is one of the most common
  Civitai-LoRA friction points; this surfaces the fix at the
  exact moment users need it.

### Round-out (4 features)

- **`plakat models aliases [--family F] [--repo] [--gated]`**.
  Enumerates every `--model` short-name plakat recognises,
  grouped by family. `--family flux` filters; `--repo` prints
  bare HF repo ids (pipes into `xargs plakat models pull`);
  `--gated` lists HF_TOKEN-only repos. Refactor: the
  hand-written alias `match` became a static `ALIAS_TABLE`
  so adding an entry updates both resolution and the listing.

- **`plakat init [DIR]`**. Bootstraps a runnable starter
  project — `scenario.hjson` (sd15, `enhancer: local`, two
  tasks), `wildcards/` (subject / style / lighting with three
  options each), and a focused `.gitignore`. Targets the
  ungated SD 1.5 + on-device LLM enhancer so first-run users
  with no HF token + no API key can generate end-to-end.
  Companion fix: `scenario`'s enhancer validator gained the
  `local` / `local:<alias>` / `auto` providers (previously
  cloud-only — the gap is why a fresh init scenario couldn't
  dry-run).

- **User-defined negative-preset catalogs**. Drop a `.txt` file
  into `<plakat-config-dir>/negative-presets/` and the
  filename becomes a `--negative-preset` name. User files
  override built-ins; safety-checked names; empty files fall
  through to the built-in. Error output marks entries as
  `<name> (user)` or `<name> (user override)`.

- **`--enhance-keep-original`**. New flag on `plakat generate`
  and `plakat portrait`: joins the enhancer's rewrite with
  the user's original prompt via the SD-family `BREAK`
  keyword (each chunk gets its own 77-token CLIP slot, so
  original terms aren't diluted by the enhancer's added
  detail). SD-family only by design; Flux / SD3 warn once
  (their T5 ignores BREAK and has the budget to carry both
  phrasings).

### Big swings — Kontext + tiled, Flux animate (2 features)

- **Flux Kontext + tiled denoise**. Lifts the v0.18 bail at
  the Kontext + `--tiled` junction. Each tile slices the
  matching region of the reference latent, packs it,
  seq-concats onto the tile's noise tokens, pads CN
  residuals for the reference half, runs forward, strips the
  reference tail. Per-tile RoPE budget check fires up front
  (Kontext + tiled doubles the per-tile sequence; the bail
  interpolates the largest safe `--tile-size` into the error
  message, typically ≤608 px for Kontext-dev).

  ```bash
  plakat generate "fold the dress into a flowing cape" \
      --model flux-kontext-dev --concept-image portrait.jpg \
      --size 2048x2048 --tiled --tile-size 512
  ```

- **Flux animate**. `plakat animate --model flux-dev` (and
  `--model flux-schnell`) now work. Pre-encodes both endpoint
  prompts through CLIP-L + T5-XXL **once**, then per frame:
  lerp the `(clip_pooled, t5_emb)` pair → run Flux's
  flow-match denoise → save. T5 encode dominates the cost,
  so amortising it across frames is the whole point of
  animate. New `pub fn animate_frame` on `flux::Pipeline`;
  Kontext / Fill / Canny / Depth refused (no place for a
  reference per call). Flux is guidance-distilled, so
  `--negative` is a no-op (warns) and `--guidance` is the
  scalar that goes straight to the model — drop to 3.5
  (Dev) / 0.0 (Schnell).

  ```bash
  plakat animate \
      --from "an oil painting of a fox in a meadow" \
      --to   "an oil painting of a cat in a meadow" \
      --frames 24 --seed 42 --guidance 3.5 \
      --model flux-dev --size 1024x1024 --out ./morph --gif
  ```

### Deferred to v0.21

- **SD3 / SD3.5 animate** — the three-encoder (CLIP-L +
  CLIP-G + T5) lerp + MMDiT rectified-flow integrator wiring
  is its own refactor. `plakat animate --model sd35-*` bails
  with a clear "deferred" message; Flux animate in v0.20 is
  the proving ground for the per-frame-encoding approach
  SD3 will follow.
- **AnimateDiff** — motion-adapter weights + temporal-attention
  injection into the SD UNet. Genuinely new architecture
  (not covered by candle 0.10.2); slated for v0.21+ as its
  own multi-cycle effort rather than rushed into v0.20.

569 lib tests green; +60 new tests across the cycle.

**Earlier releases** (v0.13 – v0.19):
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

Requires Rust 1.85+ (edition 2024). On Apple hardware, see
[`Documentation/APPLE_REQUIREMENTS.md`](Documentation/APPLE_REQUIREMENTS.md)
for the minimum / recommended chip + memory tiers and expected
per-image speeds.

## Quick start

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
plakat generate "..." --model flux-dev-gguf --quant-level Q5_K_M \
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
    --concept-image photo.png --quant-level Q5_K_M

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
| `generate <PROMPT>` | Single-shot text-to-image. SD 1.5 / 2.1 / SDXL / SDXL-Turbo / Flux (BF16, GGUF, NF4, **Kontext-dev** v0.18 — composes with ControlNet + Redux v0.19, **+ `--tiled` v0.20**) / SD3 / SD3.5. Built-in wildcards, A1111 attention syntax, inline `<lora:>` tags, `BREAK` keyword (SD-family), CLIP-skip, ADetailer, Hires fix, ControlNet, LoRA stacking, tiled hi-res, Flux Redux + concept variants, `--grid` bundling, `--preview-every`, PNG metadata + JSON sidecar, `--negative-preset` (+ user catalog v0.20), `--format webp` (Flux + SD3 in v0.20), `--enhance local\|auto` + cache/temp/tokens/system + **`--enhance-keep-original`** (v0.20), **`--recipe FILE.json`** (v0.20). |
| `img2img <INPUT>` | Image-to-image transform with `--prompt`; supply `--mask` for masked inpaint instead. SD 1.5 / 2.1 / SDXL, Flux (`--model flux-dev` for img2img, `--model flux-fill-dev` for inpaint, **`flux-kontext-dev`** for image editing — v0.18, with `--tiled` for 4K+ inpaint), and SD3 / SD3.5 (RePaint-style inpaint, `--tiled` for 2K+ outputs). v0.18: `--aspect 16:9` size derivation. |
| `outpaint <INPUT>` | Extend an image past its borders. Per-side `--left`/`--right`/`--top`/`--bottom` or `--expand N` for all four. Defaults to `sdxl-inpaint`; `flux-fill-dev` works too. |
| `portrait <PROMPT>` | Portrait generation, optionally guided by one or more reference photos with weighted merging. IP-Adapter-Plus-Face or FaceID on SD 1.5 / SDXL. |
| `scenario <FILE>` | Batch generation from an HJSON config: scenes × weather × tasks × personas × styles. `--resume` skips already-generated outputs; v0.19 adds `--only NAME[,NAME,…]` (named-task filter), `--limit N` (first N tasks), polished `--dry-run` summary. |
| `style {detect,list,show,init,probe}` | Inspect, detect, and bootstrap art-style catalogs. |
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
| `inspect <FILE>` | List every tensor in a `.safetensors` file. |
| `metadata <FILE.png>` | Read the v0.17 Auto1111 `parameters` PNG tEXt chunk + sibling `.json` sidecar. Reverse of the metadata write path. `--json-only` / `--params-only` to filter. |
| `clone <FILE.png>` | v0.19. Translate a PNG's metadata into a re-runnable `plakat generate` shell command. JSON sidecar preferred; falls back to parsing the Auto1111 chunk (works on Civitai uploads + A1111 Web UI outputs). `--one-line` for piping. |

## Documentation

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
