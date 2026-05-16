# plakat

Local text-to-image, style transfer, LoRA, scenario, upscale and color-key
CLI built on [candle](https://github.com/huggingface/candle). Pure Rust
inference — no Python, no PyTorch, no external T2I services. Models are
pulled from HuggingFace and cached locally.

## Features

- **Text-to-image** — Stable Diffusion 1.5 / 2.1 / SDXL / SDXL-Turbo and
  Flux (schnell / dev, BF16 on accelerators for numerical stability)
- **Scenario** — batch-generate from an HJSON file: crosses scenes, weather,
  per-task prompts; optionally restyles each output with a reference image
  (IP-Adapter) and upscales the result. Loads SD models once across tasks.
- **Style transfer** — IN + REF → OUT using IP-Adapter image projection
- **LoRA** — kohya + diffusers/PEFT formats, local files or HF repos
  (auto-discovered), per-LoRA `:SCALE` and a global multiplier
- **Schedulers** — DDIM, Euler-Ancestral, UniPC / DPM-Solver++
- **Polish pass** — extra denoising pass (`--refine N`) at low strength
  for sharper details (same model)
- **SDXL refiner** — `--refiner` enables the official
  `stable-diffusion-xl-refiner-1.0` UNet for the last 20% of the schedule
- **Upscale** — classical resampling (Lanczos / bicubic / bilinear / nearest)
  + Real-ESRGAN (RRDBNet ported to candle: x2plus / x4plus / x4plus_anime_6B)
- **Transparent** — chroma-key the upper-left pixel to alpha
- **Model browser** — recommend trending T2I models, inspect repo sizes,
  manage the local cache
- **Prompt enhancement** — optional DeepSeek / Gemini rewrites for sharper
  prompts before generation

## Install

`plakat` runs on any platform that candle supports. Pick a backend at install
time — the default is CPU and is slow at real generation sizes.

```bash
# macOS (Apple Silicon GPU via Metal)
cargo install plakat --features metal

# Linux + NVIDIA GPU
cargo install plakat --features cuda
# or with cuDNN
cargo install plakat --features cudnn

# Anywhere, CPU only (slow)
cargo install plakat
```

Requires Rust 1.85+ (edition 2024).

## Quick start

```bash
# SD 1.5 baseline
plakat generate "a brutalist poster of a whale, watercolor" \
    --size 512x512 --steps 28 --seed 42

# SDXL with Euler-A scheduler and a polish pass
plakat generate "a tranquil koi pond, cherry blossoms, soft light" \
    --model sdxl --size 1024x1024 \
    --steps 25 --guidance 6.0 \
    --scheduler euler-a --refine 6

# LoRA pulled straight from HuggingFace (file auto-discovered)
plakat generate "watercolor town, period clothing, varied faces" \
    --model sdxl --size 1024x1024 \
    --lora ostris/watercolor_style_lora_sdxl

# Flux schnell — 4-step rectified-flow transformer
# (~31 GB download on first run)
plakat generate "a cyberpunk samurai at dusk" \
    --model flux-schnell --size 1024x1024

# Batch generation from an HJSON scenario file
export DEEPSEEK_API_KEY=sk-...
plakat scenario examples/scenario.hjson

# Style transfer (IN + REF → OUT)
plakat stylize --in photo.jpg --ref painting.jpg --out styled.png --strength 0.6

# Upscale
plakat upscale --in image.png --out image-2x.png --scale 2

# Chroma-key the upper-left pixel to alpha
plakat transparent --in logo.png --out logo-alpha.png --tolerance 10

# Browse and manage models
plakat models recommend --sort downloads
plakat models size sdxl
plakat models ls
plakat models rm sd15 --yes
```

## Subcommands

| Command | What it does |
|---|---|
| `generate <PROMPT>` | Single-shot text-to-image. All quality knobs (scheduler, refine, LoRA, enhancer) attach here. |
| `scenario <FILE.hjson>` | Batch-generate from a config file; see below. |
| `stylize` | IP-Adapter shared-cross-attention style transfer on SD 1.5. |
| `transparent` | Make the upper-left pixel's colour fully transparent (with tolerance). |
| `upscale` | Classical resize using Lanczos/bicubic/bilinear/nearest. |
| `models {search,recommend,size,pull,ls,rm}` | Manage the local model cache and browse HF. |

`plakat <CMD> --help` for full options on each. For a parameter-by-parameter
reference with what each does, see [GENERATE.md](GENERATE.md).

## Scenario configuration

`plakat scenario <FILE>` runs many tasks from a single
[HJSON](https://hjson.github.io/) file. HJSON is JSON minus the strict
syntax — unquoted keys, no required commas, `#` and `//` comments, multi-line
strings.

A working example lives at `examples/scenario.hjson`. The schema:

```hjson
{
    # ===== generation parameters (override CLI defaults) =====
    model:           sdxl                     # alias or HF repo id
    device:          metal                    # auto | cuda[:N] | metal | cpu
    size:            1024x1024
    # aspect:        16:9                     # alternative to `size`
    # base:          1024                     # base resolution used with `aspect`
    count:           2                        # images per task
    steps:           28
    guidance:        7.0
    seed:            42                       # base seed; +count per task
    out:             ./out/scenario

    scheduler:       euler-a                  # default | ddim | euler-a | unipc
    refine:          6                        # extra same-model polish steps
    refine-strength: 0.3                      # 0..1
    refiner:         false                    # real SDXL refiner UNet (SDXL only, +6 GB)
    refiner-frac:    0.8                      # last 20% of schedule runs on refiner

    # ===== optional post-generate upscale =====
    # See "Per-image pipeline" below for the styled-vs-original rule.
    # method options: nearest | bilinear | bicubic | lanczos
    upscale:
    {
        upscale: false
        scale: 2.0
        method: lanczos
    }

    # ===== LoRA stacking =====
    loras:
    [
        ostris/watercolor_style_lora_sdxl:0.7
        ./local/character.safetensors
    ]
    lora-scale: 1.0

    # ===== prompt assembly =====
    # Final prompt for each image:
    #     lora-header + ENHANCED(prompt-header + scene + weather + task + prompt-footer) + lora-footer
    # Stray leading/trailing commas in each fragment are normalized.

    lora-header:   "trigger tokens, that the enhancer shouldn't rewrite,"
    lora-footer:   ", masterpiece, high detail"
    prompt-header: "fantasy art,"
    prompt-footer: ", soft natural lighting"

    enhancer: deepseek          # required: deepseek | gemini
    negative: "blurry, deformed, watermark, text, jpeg artifacts"

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

    # ===== tasks: which (scene × weather) combinations to render =====
    #
    # Optional per-task fields:
    #   style:          path to a reference image. If set, every image this
    #                   task generates is also run through IP-Adapter
    #                   stylize using this image as REF. Original + styled
    #                   both persist.
    #   style-strength: 0..1 (default 0.6) — higher = closer to REF.
    tasks:
    [
        {
            name: town_rainy
            scene: town
            weather: rain
            prompt: "merchants under awnings, children splashing"
            # style: ./references/watercolor.jpg
            # style-strength: 0.65
        }
        # … more tasks
    ]
}
```

### Per-image pipeline

For every image a task generates, the steps applied (in order) are:

1. **Generate** with the loaded SD/Flux pipeline → `plakat-<seed>.png` (or
   `plakat-flux-<seed>.png` for Flux).
2. **Stylize** (if the task has `style`) → `plakat-<seed>-styled.png`. Uses
   IP-Adapter on SD 1.5 regardless of the base model used in step 1.
3. **Upscale** (if `upscale.upscale: true`) → either
   `plakat-<seed>-styled-upscaled.png` (when `style` was set and the styled
   file exists) or `plakat-<seed>-upscaled.png` (otherwise). Classical
   resampling via the existing `upscale` subcommand's filters.

Each step writes a new file — nothing is overwritten. A single task with
`count: 2`, `style: ref.jpg`, and `upscale: true` produces six files per
task (2 originals + 2 styled + 2 upscaled-of-styled).

Failure in step 2 or 3 doesn't abort the scenario; the failure is logged
and earlier outputs persist.

### Output layout

Outputs land in `<out>/<task_name>/plakat-<seed>.png` (plus `-styled` and
`-upscaled` siblings when those steps run). Seeds across tasks are
contiguous: task 1 uses `seed..seed+count-1`, task 2 uses
`seed+count..`, etc., so a re-run with the same file produces identical
compositions.

### Dry-run

`plakat scenario file.hjson --dry-run` validates the schema, lists every
planned prompt, and shows what stylize/upscale steps would fire — without
calling the enhancer or generating anything.

### Performance

For SD-family scenarios, the UNet/VAE/CLIP/LoRA are loaded **once** and
shared across all tasks (see `Pipeline` in `src/pipelines/t2i.rs`). For a
5-task scenario this saves roughly 5× the model-load time on Metal.

Flux scenarios reload per task (Flux has its own pipeline module; a shared
`FluxPipeline` is a future addition).

**HJSON gotchas**

- *Don't put `#` comments at the end of unquoted string values.* HJSON
  unquoted strings extend to end-of-line, so `enhancer: deepseek # foo`
  becomes the literal string `"deepseek # foo"`. Comments above the line
  are fine.
- *Use multi-line array/object form for collections.* Inline `[ {a:1}, {b:2} ]`
  trips up the parser; put each `{` on its own line.

## LoRA

The `--lora` flag accepts:

| Form | Meaning |
|---|---|
| `./local/file.safetensors` | Local path |
| `org/repo` | HF repo — file auto-discovered (canonical names first, then largest `.safetensors`) |
| `org/repo#sub/path.safetensors` | HF repo, explicit file |

Append `:0.7` for per-LoRA scale. Multiple `--lora` flags stack. A global
`--lora-scale 0.9` multiplies every per-file scale.

```bash
plakat generate "..." --model sd15 \
    --lora ./local-style.safetensors:0.8 \
    --lora someorg/character-lora:0.6 \
    --lora-scale 0.9
```

Formats recognized:

- **kohya** — `lora_unet_*.lora_down/.lora_up/.alpha`
- **diffusers / PEFT** — `*.lora_A/.lora_B[.default].weight`, with optional
  `base_model.model.` / `diffusion_model.` prefix stripping
- **Conv targets** — 2D Linear, 1×1 conv, and 3×3 conv (LCM-LoRA style)

If a LoRA's tensor shapes don't match the base UNet, plakat skips those
targets and prints a diagnostic that guesses which model the LoRA was
actually trained for (SD 1.5 / 2.1 / SDXL — by reading the cross-attention
dim from the delta).

## Schedulers

`--scheduler default | ddim | euler-a | unipc`

| Scheduler | When to use it |
|---|---|
| `default` | The variant's built-in (DDIM for SD 1.5/2.1/SDXL, Euler-Ancestral for SDXL-Turbo). |
| `ddim` | Stable baseline, deterministic given seed. |
| `euler-a` | Often the quality sweet spot at the same step count. Recommended for SD/SDXL. |
| `unipc` | DPM-Solver++ variant. **CUDA / CPU only** — candle's UniPC uses F64 ops Metal doesn't implement. |

## Polish pass (`--refine`)

`--refine N --refine-strength F` runs an additional `N`-step img2img pass on
the final latents using the **same** base model. Cheap way to sharpen details
and reduce artifacts. Recommended strength: `0.2–0.4`.

This is **not** the official SDXL refiner — that needs a separate UNet config
and ~6 GB of refiner weights. Plakat's pass uses the same model you just
generated with.

## Model cache

By default models live in `~/.cache/huggingface/hub` (standard HF location).
Override with:

```bash
plakat --cache-dir /mnt/big/hf-cache generate "..."
# or via env
PLAKAT_CACHE_DIR=/mnt/big/hf-cache plakat generate "..."
```

For **gated repos** (e.g. `FLUX.1-dev`) set `HF_TOKEN` to a token from
<https://huggingface.co/settings/tokens> and accept the repo's license on
the HF web page first.

`plakat models size <repo>` shows what plakat would actually download for a
repo (with fp16 preference), separately from the repo's total Hub size.

## Prompt enhancement

Pass a prompt through DeepSeek or Gemini to add concrete visual detail
(subject, composition, lighting, medium, mood) before generation:

```bash
export DEEPSEEK_API_KEY=sk-...
plakat generate "knight" --enhance deepseek

export GEMINI_API_KEY=...
plakat generate "knight" --enhance gemini
```

The enhancer is **required** for scenarios (every task is enhanced once,
then all `count` images share the enhanced prompt).

## What's not in here yet

- SwinIR / LDSR upscalers (Real-ESRGAN is in; other ML upscalers aren't)
- Text-encoder LoRA merging (UNet LoRA only)
- LyCORIS / DoRA decompositions
- Shared `FluxPipeline` across scenario tasks (SD is shared; Flux still
  reloads per task)
- Shared `StylizePipeline` (the IP-Adapter image encoder reloads per image
  inside the stylize step)
- Per-task overrides for things currently global (`scheduler`, `refine`,
  `negative`, …)
- Schedulers beyond DDIM / Euler-A / UniPC

## License

This is free and unencumbered software released into the public domain (the
[Unlicense](https://unlicense.org/)).
