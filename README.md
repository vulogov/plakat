# plakat

Local text-to-image, style transfer, LoRA, upscale and color-key CLI built on
[candle](https://github.com/huggingface/candle). Pure Rust inference — no
Python, no PyTorch, no external T2I services. Models are pulled from
HuggingFace and cached locally.

## Features

- **Text-to-image**: Stable Diffusion 1.5 / 2.1 / SDXL / SDXL-Turbo + Flux
  (schnell / dev)
- **Style transfer**: IN + REF → OUT using IP-Adapter image projection
- **LoRA**: kohya + diffusers/PEFT formats, local files or HF repos, with
  per-file `:SCALE` and a global multiplier
- **Schedulers**: DDIM, Euler-Ancestral, UniPC / DPM-Solver++
- **Polish pass**: second denoising pass (`--refine N`) for sharper details
- **Upscale**: classical resampling (Lanczos / bicubic / bilinear / nearest)
- **Transparent**: chroma-key the upper-left pixel to alpha
- **Model browser**: search recommended T2I models, inspect repo sizes, manage
  the local cache
- **Prompt enhancement**: optional DeepSeek / Gemini rewrites

## Install

`plakat` runs on any platform that candle supports. Pick a backend at install
time — the default is CPU and is slow on real-size generations.

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

# SDXL with a LoRA from HuggingFace (auto-discovers the .safetensors)
plakat generate "watercolor town, period clothing, varied faces" \
    --model sdxl --size 1024x1024 \
    --lora ostris/watercolor_style_lora_sdxl

# Flux schnell — 4-step rectified-flow transformer
# (~31 GB download on first run)
plakat generate "a cyberpunk samurai at dusk" \
    --model flux-schnell --size 1024x1024

# Style transfer
plakat stylize --in photo.jpg --ref painting.jpg --out styled.png --strength 0.6

# Upscale
plakat upscale --in image.png --out image-2x.png --scale 2

# Make a solid-color background transparent
plakat transparent --in logo.png --out logo-alpha.png --tolerance 10

# Browse and manage models
plakat models recommend --sort downloads
plakat models size sdxl
plakat models ls
plakat models rm sd15 --yes
```

## Model cache

By default models live in `~/.cache/huggingface/hub` (standard HF location).
Override with:

```bash
plakat --cache-dir /mnt/big/hf-cache generate "..."
# or via env
PLAKAT_CACHE_DIR=/mnt/big/hf-cache plakat generate "..."
```

For gated repos (e.g. FLUX.1-dev) set `HF_TOKEN` to a token from
<https://huggingface.co/settings/tokens>.

## Prompt enhancement

Pass a prompt through DeepSeek or Gemini to add concrete visual detail before
generation:

```bash
export DEEPSEEK_API_KEY=sk-...
plakat generate "knight" --enhance deepseek
```

Same for `--enhance gemini` with `GEMINI_API_KEY`.

## What's not in here yet

- Real SDXL refiner (separate UNet + weights — `--refine` is an honest
  single-model polish pass)
- ML upscaling (Real-ESRGAN / SwinIR) — `upscale` is classical only
- Text-encoder LoRA merging (UNet LoRA only)
- LyCORIS / DoRA decompositions
- Schedulers beyond DDIM / Euler-A / UniPC

## License

This is free and unencumbered software released into the public domain (the
[Unlicense](https://unlicense.org/)).
