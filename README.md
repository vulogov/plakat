# plakat

![](examples/scenario/forest_snow/plakat-1004.png)

Local text-to-image generation, style transfer, LoRA stacking, ML upscaling,
identity-preserving portraits, and batch scenarios — all built on
[candle](https://github.com/huggingface/candle). Pure Rust inference. No
Python, no PyTorch, no external T2I services. Models are pulled from
HuggingFace and cached locally.

## What's new in v0.17 — the prompt + reproducibility release

Ten phases focused on **prompt expressiveness**, **reproducibility**,
and **animation**. The cycle also upgrades the underlying candle ML
framework two minor versions and adds the long-asked `--lora civitai:`
shorthand:

- **A1111 prompt syntax**. `(red:1.4)` emphasis / `[blue]`
  de-emphasis / `((nested))` compounding / `\(escape\)` — the
  grammar every Civitai LoRA card uses in its example prompts.
  Applied to CLIP penultimate hidden states via per-token broadcast.
  SD 1.5 / SD 2.1 / SDXL.
- **PNG metadata + JSON sidecar**. Outputs ship with the
  Auto1111-compatible `parameters` PNG tEXt chunk + a sibling
  `<filename>.json` carrying the full recipe. A1111 / Civitai /
  ComfyUI / sd-prompt-reader all surface the prompt + seed + LoRAs
  + scheduler inline. `--no-metadata` opts out.
- **`--grid` output**. `--count N > 1` + `--grid` writes a single
  `plakat-grid-<seed>.png` combining all N outputs in a near-square
  layout. `--grid-cols` / `--grid-padding` for fine control.
- **`plakat animate`**. New subcommand for prompt-morph animation:
  lerp CLIP embeddings between two prompts at a fixed seed,
  producing a smooth N-frame sequence. `--gif` bundles into an
  animated GIF. SD 1.5 / SD 2.1.
- **Live preview during denoise**. `--preview-every N` writes a
  cheap latent-projection PNG every N steps so long runs aren't a
  black box. Microseconds per write — no meaningful runtime cost.
- **scenario `--resume` / `--force`**. Crashed scenario picks up
  where it left off by probing for already-existing output PNGs.
  No more restart-from-task-0.
- **`--lora civitai:NNNNNN`**. Skip the explicit
  `plakat civitai download` step — the LoRA spec parser now
  downloads + caches Civitai assets on first use via the
  shorthand. `civitai-version:NNNNNN` pins a specific version.
- **LCM-LoRA SDXL `--fast lcm-sdxl`**. Latent-Consistency
  distillation for SDXL bundled with the right scheduler and
  4-step / CFG-1.5 defaults. ~5× speedup over stock SDXL.
- **candle 0.8 → 0.10.2 upgrade**. Single 8-line trait-impl fix
  for `SimpleBackend::get_unchecked`. GGUF / NF4 / MMDiT /
  vendored Flux all intact, 304 tests still green at upgrade
  time.
- **SDXL refiner cleanup**. The "Known limitation" about missing
  `add_embedding` on the refiner was outdated since v0.11 phase
  8e. Stale docs replaced; regression tests pin the 5-time-id
  config so future refactors can't silently break the refiner's
  `text_time` micro-conditioning.

381 lib tests green; +77 new tests across the cycle.

## What's new in v0.16 — the productivity release

A dozen quality-of-life landings that connect community workflows
(Civitai browsing, ADetailer face fix, Hires fix, wildcards) to the
existing plakat backbone, plus deeper SD3 integration:

- **SD3 ControlNet (InstantX)**. `--control-spec` works on SD3 /
  SD3.5 via the InstantX adapter family. Multi-CN composition,
  step-gating, auto-annotation from a reference photo — same
  ergonomics SDXL + Flux ControlNet ship.
- **Tiled Flux Fill**. `--tiled` composes with Flux.1-Fill-dev for
  4K+ inpaint. Per-tile masked-latent + mask packing.
- **Tiled SD3 img2img + inpaint**. The rectified-flow init lerp +
  RePaint mask blend compose with the per-tile Hann blend.
- **Wildcards**. `{red|blue|green}` inline alternation +
  `__name__` file wildcards (Auto1111 / NovelAI grammar). Seeded
  from `--seed` for reproducibility.
- **CLIP-skip**. `--clip-skip N` for SD 1.5 / SD 2.1 — N=2 is the
  community default for anime checkpoints.
- **ADetailer-style face refinement**. `--adetailer` runs SCRFD
  on each output, crops + img2img-refines each face, feather-
  composites back. Reuses the t2i SdCore — no extra model load.
- **Hires fix**. `--hires-fix` escapes the trained-resolution
  ceiling: upscale (Lanczos / Real-ESRGAN) + img2img-refine.
  Composes with `--adetailer` for a 4K → fixed faces pipeline.
- **Civitai browser + downloader**. `plakat civitai search`,
  `info`, `download` — drop the resulting path into `--lora` /
  `--model`. Atomic streaming downloads with cache-hit
  short-circuit.
- **Auto-annotation for Flux concept variants**. `--concept-from
  PATH` auto-annotates a photo through Canny / Depth before feeding
  Flux.1-Canny-dev / Flux.1-Depth-dev.
- **SD3 pipeline caching + per-task LoRA**. Scenarios with
  `--model sd35-*` now share one SD3 pipeline across tasks; per-
  task `loras:` swap at runtime via the LoraLinear stack.
- **Textual Inversion** *(partial)*. Parser + `plakat embedding
  info` inspector. Runtime injection blocked by candle 0.8's
  private `clip::Config.vocab_size` — wiring lands when the
  candle API surface opens or alongside a vendored CLIP path.
- **SD UNet per-task LoRA preflight** *(partial)*. Detects the
  blocker upfront and emits actionable YAML-fold hints; bails
  loud with three concrete workarounds. Full UNet vendoring
  deferred — same candle private-internals blocker.
- **XLabs Flux IP-Adapter parser** *(partial)*. Inspector that
  reports per-block attention count + SigLIP/Flux dims. Per-block
  injection blocked by Flux's private `double_block_forward`;
  use `--redux-image` for working image conditioning today.

## What's new in v0.15 — runtime LoRA + SD3 maturation

- **Per-task LoRA in scenarios**. `tasks: [{ loras: [...] }]` applies
  and clears LoRAs between tasks at runtime — no model reload.
  Composes with the scenario-level LoRA set. Flux (BF16 / GGUF / NF4).
- **NF4 + ControlNet**. NF4 Flux composes with `--control-spec` via
  the residual-aware forward — same residual interleave the BF16 and
  GGUF backbones use, so a single CN checkpoint works on all three.
- **SD3 / SD3.5 img2img + inpaint**. RePaint-style inpaint with
  per-step mask blend, rectified-flow truncated schedule. Works
  across the lineup (Medium / Large / Turbo).
- **SD3 / SD3.5 LoRA**. Diffusers PEFT format, MMDiT-targeted keys.
- **Flux Canny-dev / Depth-dev variants**. BFL "concept" Flux
  checkpoints with conditioning baked into the 128-channel `img_in`.
  Pass `--concept-image PATH` with `--model flux-canny-dev`.
- **Tiled SD3**. MultiDiffusion-style tiled denoise for MMDiT —
  1024-px tiles work on every SD3 variant within the variant's
  `pos_embed_max_size` cap.
- **Scenario ↔ generate sync**. Per-task `fast`, `concept-image`,
  `enhance`, `tiled` overrides.
- **Two new tutorials**:
  [`FLUX_TUTORIAL.md`](Documentation/Tutorials/FLUX_TUTORIAL.md)
  walks through the Flux feature set end-to-end;
  [`SD3_TUTORIAL.md`](Documentation/Tutorials/SD3_TUTORIAL.md) does
  the same for the SD3 / SD3.5 family.

## What's new in v0.14 — the SD3.5 + NF4 + Redux release

- **Stable Diffusion 3 / 3.5 (MMDiT)**. New family — `sd35-medium`,
  `sd35-large`, `sd35-large-turbo`, `sd3-medium`. Triple text encoder
  (CLIP-L + CLIP-G + T5-XXL), 16-channel VAE, rectified-flow sampler
  with SD3 time-shift. CFG via `[neg, pos]` double-batch.
- **NF4 quantized Flux**. `--model flux-dev-nf4` loads lllyasviel's
  bitsandbytes NF4 pack — ~6 GB transformer at inference (4× weight
  savings vs BF16), pure-CPU dequant codec means it runs on any
  candle device. Phase 8b adds **NF4 + LoRA composition** via the
  same selective-dequant trick GGUF uses.
- **Flux Redux**. `--redux-image PATH` adds image conditioning via
  SigLIP-so400m + BFL's Redux adapter (729 tokens → seq-concat onto
  T5). Repeatable for multi-image stacks (`--redux-image
  style.png:weight=0.8 --redux-image subject.png:weight=0.5`). Cap
  of 4 with attention-cost guardrails. Composes with GGUF, NF4,
  LoRA, ControlNet, img2img, tiled.
- **Tiled SD 1.5 / 2.1**. `--tiled` now supported on the smaller
  SD backbones too (was SDXL-only in v0.12).
- **Flux Fill + ControlNet**. `plakat img2img --model flux-fill-dev
  --mask ... --control-spec depth:from=...` composes with the
  auto-annotator and multi-CN.
- **Hyper-FLUX / FLUX-Turbo presets**. `--fast hyper-8 | hyper-16 |
  turbo-alpha` bundles the matching distillation LoRA + recommended
  step count + guidance in one flag.
- **Shared SdCore**. Scenarios with mixed t2i + img2img tasks now
  load the SD backbone **once** (was: per-task). The t2i Pipeline's
  `Arc<SdCore>` is reused by img2img via the existing `from_core`
  path.

## What's new in v0.13 — the Flux modernization release

- **Quantized Flux (GGUF)**. Run FLUX.1-dev on 16 GB GPUs.
  `--model flux-dev-gguf` loads the 4-bit transformer (~7 GB vs ~24 GB BF16).
  `--quantize-t5` drops T5-XXL to ~3 GB. `--quant-level Q5_K_M` picks a
  different precision (Q2_K..F16 supported); same for `--t5-quant-level`.
- **Flux LoRA on quantized**. Diffusers PEFT and AI-Toolkit / kohya
  formats both compose with the GGUF backbone — affected Linears are
  dequantized once at load, rest of the model stays 4-bit.
- **Flux Inpainting**. `--model flux-fill-dev` + `--mask` runs BFL's
  dedicated 384-channel inpaint checkpoint via `plakat img2img`.
- **Flux Img2Img**. Rectified-flow init: `plakat img2img init.png
  --model flux-dev --strength 0.7 --prompt "..."`.
- **Tiled Flux denoise**. MultiDiffusion-style 2K–4K outputs on any
  Flux variant: `--tiled --tile-size 1024 --tile-stride 768`. Composes
  with ControlNet (per-tile residuals) and the tiled VAE decode.
- **Flux ControlNet polish**. Auto-annotators wire through to Flux
  (`--control-spec depth:from=photo.jpg` is now a one-liner). Step gating
  via `start=…:end=…`. Multi-Flux-CN with summed residuals.
- **Outpainting**. New `plakat outpaint` subcommand expands a canvas
  and hands off to the inpaint pipeline (SDXL-Inpaint, SD15-Inpaint, or
  Flux.1-Fill-dev).
- **Scenarios**. Every v0.13 feature above is now expressible in
  scenario HJSON: `quant-level:`, `t5-quant-level:`, `tiled:`,
  per-task `init-image:` / `mask:` / `strength:` / `outpaint:`, plus
  multi-CN via `controls: [...]`.

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

# Prompt-morph animation — interpolates two prompts over N frames.
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
| `generate <PROMPT>` | Single-shot text-to-image. SD 1.5 / 2.1 / SDXL / SDXL-Turbo / Flux (BF16, GGUF, NF4) / SD3 / SD3.5. Built-in wildcards, A1111 attention syntax, CLIP-skip, ADetailer face refinement, Hires fix, ControlNet, LoRA stacking, tiled hi-res, Flux Redux + concept variants, `--grid` bundling, `--preview-every` live previews, PNG metadata + JSON sidecar. |
| `img2img <INPUT>` | Image-to-image transform with `--prompt`; supply `--mask` for masked inpaint instead. SD 1.5 / 2.1 / SDXL, Flux (`--model flux-dev` for img2img, `--model flux-fill-dev` for inpaint, with `--tiled` for 4K+ inpaint), and SD3 / SD3.5 (RePaint-style inpaint, `--tiled` for 2K+ outputs). |
| `outpaint <INPUT>` | Extend an image past its borders. Per-side `--left`/`--right`/`--top`/`--bottom` or `--expand N` for all four. Defaults to `sdxl-inpaint`; `flux-fill-dev` works too. |
| `portrait <PROMPT>` | Portrait generation, optionally guided by one or more reference photos with weighted merging. IP-Adapter-Plus-Face or FaceID on SD 1.5 / SDXL. |
| `scenario <FILE>` | Batch generation from an HJSON config: scenes × weather × tasks × personas × styles. `--resume` skips already-generated outputs. |
| `style {detect,list,show,init,probe}` | Inspect, detect, and bootstrap art-style catalogs. |
| `artefact {list,show}` | Inspect the artefact library (PNG cutouts placeable into named zones of generated images). |
| `civitai {search,info,download}` | Browse + download Civitai community assets (LoRAs, checkpoints, embeddings, ControlNet variants). |
| `embedding {info,flux-ip-adapter-info}` | Inspect Textual Inversion `.safetensors` files + XLabs Flux IP-Adapter weights. |
| `animate --from A --to B --frames N` | Prompt-morph animation: lerp CLIP embeddings between two prompts to produce a smooth N-frame sequence at a fixed seed. Optional GIF bundling. SD 1.5 / SD 2.1. |
| `stylize` | IP-Adapter style transfer on SD 1.5 (IN + REF → OUT). |
| `upscale` | Resize, classical or Real-ESRGAN. |
| `transparent` | Make every pixel matching the corner colour transparent. |
| `models {search,recommend,size,pull,ls,rm}` | Browse HuggingFace and manage the local cache. |
| `doctor` | Health-check FaceID / SCRFD setup. |
| `inspect <FILE>` | List every tensor in a `.safetensors` file. |

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
