# plakat

![](examples/scenario/forest_snow/plakat-1004.png)

Local text-to-image generation, style transfer, LoRA stacking, ML upscaling,
and batch scenarios — all built on [candle](https://github.com/huggingface/candle).
Pure Rust inference. No Python, no PyTorch, no external T2I services.
Models are pulled from HuggingFace and cached locally.

## Features

- **Text-to-image** — Stable Diffusion 1.5 / 2.1 / SDXL / SDXL-Turbo and Flux
  (schnell / dev, BF16 on accelerators for numerical stability).
- **Scenario** — batch generation from an HJSON file: crosses scenes, weather,
  and per-task prompts; optionally restyles each output via IP-Adapter and
  upscales the result. Pipelines load once and are shared across tasks.
- **Style transfer** — IN + REF → OUT using IP-Adapter image projection
  (SD 1.5 base).
- **Portrait** — generate a portrait, optionally guided by a reference
  photo. IP-Adapter-Plus-Face on SD 1.5 (`--identity plus-face`) or SDXL
  (`--identity plus-face-sdxl`). Portrait-tuned defaults: 3:4 aspect,
  face/anatomy negatives baked in. Scenarios can define named **personas**
  and impose them per task — single-persona whole-image or multi-persona
  region-masked compositing via per-persona bboxes.
- **LoRA** — kohya, PEFT/diffusers, DoRA, LyCORIS LoHa (plain + Tucker), LoKr.
  Local files or HF repos (auto-discovered). UNet + both text encoders.
- **Nine schedulers** — DDIM, Euler (deterministic), Euler-Ancestral, Heun,
  UniPC, DPM++ 2M Karras, UniPC-exponential, LCM (for LCM-LoRA at 4–8 steps),
  DDPM.
- **Polish pass** — extra denoise pass (`--refine N`) on the final latents
  for sharper details (same model).
- **SDXL refiner** — `--refiner` enables the official
  `stable-diffusion-xl-refiner-1.0` UNet for the last 20% of the schedule.
- **Upscale** — classical resampling (Lanczos / bicubic / bilinear / nearest)
  and Real-ESRGAN (RRDBNet ported to candle: x2plus / x4plus / x4plus-anime-6B).
- **Transparent** — chroma-key the upper-left pixel to alpha.
- **Model browser** — search HF, recommend trending T2I models, inspect repo
  sizes, manage the local cache.
- **Prompt enhancement** — optional DeepSeek / Gemini rewrites before
  generation.

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

Requires Rust 1.85+ (edition 2024).

## Quick start

```bash
# 1. SD 1.5 baseline
plakat generate "a brutalist poster of a whale, watercolor" \
    --size 512x512 --steps 28 --seed 42

# 2. SDXL with Euler-A and a polish pass
plakat generate "a tranquil koi pond, soft light, cherry blossoms" \
    --model sdxl --size 1024x1024 \
    --steps 25 --guidance 6.0 \
    --scheduler euler-a --refine 6

# 3. LCM-LoRA at 4 steps with the matching scheduler
plakat generate "watercolor town at dusk" --model sd15 \
    --steps 4 --guidance 1.5 --scheduler lcm \
    --lora latent-consistency/lcm-lora-sdv1-5

# 4. SDXL LoRA pulled straight from HF (file auto-discovered)
plakat generate "ink sketch portrait of an astronaut" \
    --model sdxl --size 1024x1024 \
    --lora ostris/watercolor_style_lora_sdxl:0.8

# 5. Flux schnell — 4-step rectified-flow transformer
#    First run downloads ~31 GB of weights.
plakat generate "a cyberpunk samurai at dusk, neon reflections" \
    --model flux-schnell --size 1024x1024

# 6. Scenario — many prompts, models load once
export DEEPSEEK_API_KEY=sk-...
plakat scenario examples/scenario.hjson

# 7. Style transfer — recolour a photo in a painting's style
plakat stylize --in photo.jpg --ref painting.jpg --out styled.png \
    --strength 0.6

# 7b. Portrait from a reference photo (IP-Adapter-Plus-Face)
plakat portrait "cinematic close-up, soft Rembrandt lighting" \
    --photo face.jpg --face-strength 0.8 --size 768x1024

# 8. Real-ESRGAN upscale to 4×
plakat upscale --in small.png --out big.png \
    --method real-esrgan-x4 --device metal

# 9. Chroma-key the upper-left pixel to alpha
plakat transparent --in logo.png --out logo-alpha.png --tolerance 10

# 10. Browse and manage models
plakat models recommend --sort downloads --limit 10
plakat models size sdxl
plakat models ls
plakat models rm sd15 --yes
```

## Subcommands

| Command | What it does |
|---|---|
| `generate <PROMPT>` | Single-shot text-to-image. All quality knobs (scheduler, refine, LoRA, enhancer) attach here. |
| `portrait <PROMPT>` | Portrait generation, optionally guided by a reference photo (IP-Adapter-Plus-Face on SD 1.5). |
| `scenario <FILE>` | Batch-generate from an HJSON config. See [Scenario configuration](#scenario-configuration). |
| `stylize` | IP-Adapter style transfer on SD 1.5 (IN + REF → OUT). |
| `upscale` | Resize an image, classical or Real-ESRGAN. |
| `transparent` | Make every pixel matching the corner colour transparent. |
| `models {search,recommend,size,pull,ls,rm}` | Browse HF and manage the local cache. |

`plakat <CMD> --help` for full options on each. For a parameter-by-parameter
reference with what each does, see [GENERATE.md](GENERATE.md).

## Scenario configuration

`plakat scenario <FILE>` runs many tasks from a single
[HJSON](https://hjson.github.io/) file. HJSON is JSON minus the strict
syntax — unquoted keys, no required commas, `#` and `//` comments, multi-line
triple-quoted strings.

A working example lives at [`examples/scenario.hjson`](examples/scenario.hjson).
The schema:

```hjson
{
    # ===== generation parameters (override CLI defaults) =====
    model:           sdxl                     # alias or HF repo id
    device:          metal                    # auto | cuda[:N] | metal | cpu
    size:            1024x1024
    # aspect:        16:9                     # alternative to size
    # base:          1024                     # base resolution for aspect
    count:           2                        # images per task
    steps:           28
    guidance:        7.0
    seed:            42                       # base seed; +count per task
    out:             ./out/scenario

    scheduler:       euler-a                  # see Schedulers below
    refine:          6                        # extra same-model polish steps
    refine-strength: 0.3                      # 0..1
    refiner:         false                    # official SDXL refiner UNet
    refiner-frac:    0.8                      # 1−frac of schedule runs on refiner

    # ===== optional post-generate upscale =====
    upscale:
    {
        upscale: false                        # enable flag
        scale: 2.0                            # ignored for ML methods
        method: lanczos                       # see Upscale below
    }

    # ===== LoRA stacking =====
    loras:
    [
        ostris/watercolor_style_lora_sdxl:0.7
        ./local/character.safetensors
    ]
    lora-scale: 1.0

    # ===== prompt assembly =====
    # Final per-image prompt is built as:
    #   lora-header + ENHANCED(prompt-header + scene + weather + task + prompt-footer) + lora-footer
    # Stray leading/trailing commas in each fragment are normalised.
    lora-header:   "trigger tokens that the enhancer shouldn't rewrite,"
    lora-footer:   ", masterpiece, high detail"
    prompt-header: "fantasy art,"
    prompt-footer: ", soft natural lighting"
    enhancer:      deepseek                   # required: deepseek | gemini
    negative:      "blurry, deformed, watermark, text"

    # ===== catalogs =====
    scene:
    [
        {
            name: town
            prompt:
                '''
                a medieval European town with cobblestone streets,
                tiled rooftops, hanging wooden signs above shop fronts
                '''
        }
        # … more scenes
    ]
    weather:
    [
        {
            name: rain
            prompt: "heavy summer rain, glistening puddles, dark grey clouds"
        }
        # … more weathers
    ]

    # ===== personas (optional) =====
    # Named identities tasks can impose onto their output. Routes the task
    # through the SD 1.5 portrait pipeline (IP-Adapter-Plus-Face).
    personas:
    [
        {
            name: alice
            photo: ./refs/alice.jpg
            face-strength: 0.85               # 0..2, default 0.8
            # identity: plus-face             # plus-face | plus-face-sdxl | faceid | faceid-sdxl
            # negative: "smiling, mustache"   # persona-specific, prepended
        }
    ]

    # ===== tasks =====
    tasks:
    [
        {
            name: town_rainy
            scene: town
            weather: rain
            prompt: "merchants under awnings, children splashing"

            # Optional per-task fields:
            # style: ./refs/watercolor.jpg     # IP-Adapter REF for this task only
            # style-strength: 0.65
            # personas: [alice]                # single persona, whole image
            # personas:                        # multi-persona, region-masked compositing
            # [
            #     { name: alice, bbox: [0.05, 0.10, 0.48, 0.95] }
            #     { name: bob,   bbox: [0.52, 0.10, 0.95, 0.95] }
            # ]
            # Per-task overrides for: size, aspect, count, steps, guidance,
            # seed, negative, scheduler, refine, refine-strength, refiner-frac
        }
        # … more tasks
    ]
}
```

### Per-image pipeline

For every image a task generates, the steps applied (in order) are:

1. **Generate** with the loaded SD, Flux, OR portrait pipeline:
   - No `personas` → SD / Flux per the scenario's `model` →
     `plakat-<seed>.png` (or `plakat-flux-<seed>.png` for Flux).
   - Single bare-name persona → SD 1.5 portrait pipeline,
     one denoise pass → `plakat-portrait-<seed>.png`.
   - `{name, bbox}` personas (one or more) → SD 1.5 portrait pipeline,
     text-only base + one inpaint pass per persona, composited →
     `plakat-portrait-<seed>.png`.
2. **Stylize** (if the task has `style:`) → `<base>-styled.png`. Uses
   IP-Adapter on SD 1.5 regardless of the base model used in step 1.
3. **Upscale** (if `upscale.upscale: true`) → `<base>-styled-upscaled.png`
   if step 2 ran successfully, else `<base>-upscaled.png`.

Each step writes a new file — nothing is overwritten. A task with `count: 2`,
`style: ref.jpg`, and `upscale: true` produces **six files**: 2 originals + 2
styled + 2 upscaled-of-styled. Failure in step 2 or 3 doesn't abort the
scenario — the failure is logged and earlier outputs persist.

### Personas

`personas` is a top-level list of named identities. Each persona supplies
a reference photo and identity-strategy settings; tasks pull personas in
either **bare-name** (single persona, whole image) or **`{name, bbox}`**
(multi-persona, region-masked compositing) form. Either form routes
through the portrait pipeline:

```hjson
personas: [
  { name: alice, photo: ./refs/alice.jpg, face-strength: 0.85 }
  { name: bob,   photo: ./refs/bob.jpg }
]

tasks: [
  # Single persona, whole image.
  { name: alice_cafe, scene: cafe, weather: morning, prompt: "espresso",
    personas: [alice] }

  # Multi-persona compositing via region masks.
  { name: pair_at_table, scene: bistro, weather: golden_hour,
    prompt: "two friends sharing dessert",
    personas: [
      { name: alice, bbox: [0.05, 0.10, 0.48, 0.95] }
      { name: bob,   bbox: [0.52, 0.10, 0.95, 0.95] }
    ] }

  # No persona — regular t2i / Flux dispatch, unchanged behaviour.
  { name: empty_streets, scene: town, weather: rain, prompt: "no one" }
]
```

The portrait pipeline loads once at scenario start (only if at least one
task uses personas) and is shared across all persona tasks. Multi-persona
compositing runs one **base** denoise pass (text-only) plus one **inpaint**
pass per persona; each pass reuses the same loaded UNet + VAE + text
encoder.

**Four identity strategies ship:**

| `identity` | Base | Notes |
|---|---|---|
| `plus-face` (default) | SD 1.5 | IP-Adapter-Plus-Face; ~6.5 GB first-run download. |
| `plus-face-sdxl` | SDXL | SDXL portrait pipeline; ~9.5 GB first-run (CLIP-H shared with `plus-face`). |
| `faceid` | SD 1.5 | InsightFace ArcFace + UNet LoRA. Best identity preservation. Needs user-supplied ArcFace weights. |
| `faceid-sdxl` | SDXL | SDXL variant of `faceid`. |

Pick one per scenario — every persona in a scenario must share the same
strategy's base-model family. Standalone CLI pairs `--model sdxl` with
`plus-face-sdxl` / `faceid-sdxl`.

`faceid` / `faceid-sdxl` need either `PLAKAT_ARCFACE_WEIGHTS` (local) or
`PLAKAT_ARCFACE_HF=repo#file` (HuggingFace). Optional SCRFD auto-detection
via `PLAKAT_SCRFD_WEIGHTS` / `PLAKAT_SCRFD_HF` fills landmarks for proper
ArcFace alignment without manual `--face-bbox` flags. Run `plakat doctor`
to inspect setup; `plakat doctor --verify` to probe HF downloads.

**Limits:**
- SD 1.5 + SDXL only. Flux is not supported for portraits.
- `plus-face*` strategies have no automatic face detection — curate the
  reference photo, or use `--face-bbox` to mark the face manually.
- No persona-level LoRAs. Stack a likeness LoRA at scenario level.
- `plus-face*` identity quality is ~50–70% of the diffusers reference
  (no decoupled cross-attention in candle 0.8). `faceid*` strategies
  bypass this ceiling.
- Multi-persona wall time scales linearly: a 2-persona task runs 3
  denoise loops, a 3-persona task runs 4, etc.

Full reference (every field, every interaction, bbox-placement tips,
form-mixing rules, alignment priority, setup): [PERSONA.md](PERSONA.md).

### Output layout

Files land in `<out>/<task_name>/plakat-<seed>.png` (plus `-styled` and
`-upscaled` siblings when those steps run). Seeds advance contiguously per
the **global** `count`, so re-running a scenario produces identical
compositions even if per-task overrides shift individual values.

### Dry-run

```bash
plakat scenario file.hjson --dry-run
```

Validates the schema, lists every task's assembled prompt + planned seeds,
and shows what stylize/upscale steps would fire — without calling the
enhancer, downloading anything, or generating images.

### Performance

Every pipeline family loads once at scenario start and is shared across all
tasks:

| Pipeline | Weights kept resident | Saved per task |
|---|---|---|
| SD (`src/pipelines/t2i.rs`) | UNet, VAE, CLIP text encoder(s), merged LoRA(s) | ~10 s on Metal |
| Flux (`src/pipelines/flux.rs`) | Transformer (24 GB), AE, T5-XXL (9 GB), CLIP-L | ~30 s |
| Stylize (`src/pipelines/stylize.rs`, when any task has `style:`) | SD 1.5 base + IP-Adapter projection + CLIP-H (2.5 GB) | ~15 s |
| Real-ESRGAN (`src/imaging/upscale.rs`, when method is ML) | RRDBNet (~17 MB) | ~4 s |

### HJSON gotchas

- *Don't put `#` comments after unquoted string values.* HJSON unquoted strings
  extend to end-of-line, so `enhancer: deepseek # foo` becomes the literal
  string `"deepseek # foo"`. Put comments above the line.
- *Use multi-line array/object form for collections.* Inline
  `[ {a:1}, {b:2} ]` trips the parser; put each `{` on its own line.

## LoRA

The `--lora` flag (repeatable) accepts:

| Form | Meaning |
|---|---|
| `./local/file.safetensors` | Local path. |
| `org/repo` | HF repo — file auto-discovered (canonical names first, then largest `.safetensors`). |
| `org/repo#sub/path.safetensors` | HF repo, explicit file. |

Append `:0.7` for per-LoRA scale: `--lora foo.safetensors:0.7`. `--lora-scale
0.9` multiplies every per-file scale globally.

```bash
plakat generate "..." --model sd15 \
    --lora ./local-style.safetensors:0.8 \
    --lora someorg/character-lora:0.6 \
    --lora-scale 0.9
```

**Formats recognised** — plakat detects and applies all of these (UNet AND
both text encoders, where the LoRA targets them):

| Format | Detection key(s) |
|---|---|
| Standard LoRA / LoCon / DyLoRA | `lora_down` + `lora_up` (kohya), `lora_A` + `lora_B` (PEFT) |
| DoRA | standard LoRA keys + per-row `dora_scale` |
| LyCORIS LoHa | `hada_w1_a/b` + `hada_w2_a/b` |
| LyCORIS LoHa (Tucker) | LoHa keys + `hada_t1` + `hada_t2` |
| LyCORIS LoKr | `lokr_w1` (or `_a`+`_b`) + `lokr_w2` (or `_a`+`_b`) |

If a LoRA's tensor shapes don't match the base UNet, plakat skips those
targets and prints a diagnostic that infers which model the LoRA was actually
trained for (SD 1.5 / 2.1 / SDXL — by reading the cross-attention dim from
the delta).

## Schedulers

`--scheduler <KIND>` — default is the variant's built-in (DDIM for
SD 1.5/2.1/SDXL, Euler-Ancestral for SDXL-Turbo).

| Scheduler | Notes |
|---|---|
| `ddim` | Stable deterministic baseline. |
| `euler` | Deterministic Euler. Reproducible across runs. Pure F32 — works on Metal. |
| `euler-a` | Euler-Ancestral. Often higher quality than DDIM at the same step count; mildly stochastic. Metal-friendly. |
| `heun` | Heun second-order predictor-corrector. **2× the model evaluations** per `--steps`. Higher quality for the cost. Metal-friendly. |
| `unipc` | UniPC corrector with Karras sigmas (DPM-Solver++ family). **CUDA / CPU only** — candle's UniPC uses F64 ops Metal doesn't implement. |
| `dpmpp-2m` | UniPC without corrector. Crisper than `unipc`; A1111-style "safe default". CUDA/CPU only. |
| `unipc-exp` | UniPC with exponential sigma schedule. CUDA/CPU only. |
| `lcm` | LCM consistency scheduler — pair with an LCM-LoRA at 4–8 steps and `--guidance 1.0–2.0`. Metal-friendly. |
| `ddpm` | The original DDPM. Slow; mainly a reference. Metal-friendly. |

If unsure, on Metal: `euler-a` (stochastic) or `euler` (deterministic). On
CUDA/CPU: `dpmpp-2m`. With LCM-LoRA: `lcm`.

## Portrait

```
plakat portrait <PROMPT> [--photo PATH] [--face-strength F] [...]
```

Portrait-tuned text-to-image. With `--photo`, uses **IP-Adapter-Plus-Face**
on SD 1.5: the reference photo's CLIP-H penultimate hidden state is run
through a Perceiver resampler into 16 image tokens, which are concatenated
onto the text tokens and consumed by the UNet's cross-attention.

```bash
# Photo-guided (recommended use)
plakat portrait "cinematic close-up, golden hour, shallow depth of field" \
    --photo face.jpg --face-strength 0.8

# Text-only with portrait-tuned defaults (no photo, no extra download)
plakat portrait "studio portrait of an astronaut, dramatic lighting" \
    --size 768x1024 --steps 30 --scheduler euler-a
```

`--face-strength` scales the image-token contribution: `0.0` falls back to
text-only, `0.8` (default) is a strong likeness, `>1.0` over-amplifies the
face at the cost of prompt adherence. LoRAs and `--refine` stack normally.

Quality caveat for `plus-face*` strategies: candle 0.8's UNet exposes
no cross-attention hooks, so the *decoupled* IP-Adapter path (separate
`to_k_ip`/`to_v_ip` per block) is not wired up — identity tokens travel
via the same cross-attention as text. The result is recognisable but
not pixel-perfect (~50–70% of the diffusers reference). For higher
identity fidelity, use `--identity faceid` (SD 1.5) or `faceid-sdxl` —
those bypass this ceiling using ArcFace embeddings plus h94's UNet LoRA.

`plus-face` limits: no automatic face crop (pass a head-and-shoulders
photo, or use `--face-bbox` to mark the region); first run adds ~50 MB
for the Plus-Face safetensors plus ~2.5 GB for CLIP-H (shared with
`stylize`).

## Polish & refiner

`--refine N --refine-strength F` runs an additional `N`-step img2img pass on
the final latents using the **same** base model. Cheap way to sharpen
details. Recommended strength: `0.2–0.4`.

`--refiner --refiner-frac F` (SDXL only) switches to the **official SDXL
refiner UNet** for the last `1−F` of the schedule. Adds a ~6 GB download on
first run. Different mechanism from `--refine`; they can stack.

## Upscale

```
plakat upscale --in IN --out OUT --method <METHOD> [--scale N] [--device DEV]
```

**Classical** filters (no model, instant): `nearest`, `bilinear`, `bicubic`,
`lanczos` (default). `--scale` controls the factor.

**Real-ESRGAN** (ML, RRDBNet ported to candle):

| Method | Scale | Weights | Use case |
|---|---|---|---|
| `real-esrgan-x2` | ×2 | ~17 MB | Subtle resolution boost, photographic |
| `real-esrgan-x4` | ×4 | ~17 MB | Standard 4× upscale, recovers fine detail |
| `real-esrgan-anime-x4` | ×4 | ~9 MB | Line art / anime / illustration |

ML methods ignore `--scale` (the model's architecture fixes the factor) and
run on the device chosen by `--device`.

## Model cache & HF auth

Models live in `~/.cache/huggingface/hub` by default. Override with:

```bash
plakat --cache-dir /mnt/big/hf-cache generate "..."
# or
PLAKAT_CACHE_DIR=/mnt/big/hf-cache plakat generate "..."
```

For **gated repos** (e.g. `FLUX.1-dev`), set `HF_TOKEN` to a token from
<https://huggingface.co/settings/tokens> and accept the licence on the HF web
page first.

`plakat models size <repo>` shows what plakat would actually download for a
repo (with fp16 preference), separately from the repo's total Hub size.

## Prompt enhancement

```bash
export DEEPSEEK_API_KEY=sk-...
plakat generate "knight" --enhance deepseek

export GEMINI_API_KEY=...
plakat generate "knight" --enhance gemini
```

The enhancer rewrites the prompt with concrete visual detail (subject,
composition, lighting, medium, mood). It's **required** for scenarios (every
task is enhanced once; all `count` images share the enhanced prompt).

## License

Free and unencumbered software released into the public domain
([Unlicense](https://unlicense.org/)).
