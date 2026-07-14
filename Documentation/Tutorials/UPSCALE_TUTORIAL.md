# Upscale — enlarge an image

`plakat upscale` makes an image bigger, two ways:

- **Classical resampling** (default) — fast, deterministic interpolation
  (Lanczos and friends). No model, no download. Good for a quick size bump.
- **ML super-resolution** (`--method real-esrgan-*`) — **Real-ESRGAN** restores
  detail as it enlarges (sharper edges, less blur) instead of just interpolating.

```bash
# Classical Lanczos ×2 (instant, no download)
plakat upscale --in small.png --out big.png --scale 2

# Real-ESRGAN ×4 — ML detail restoration
plakat upscale --in photo.png --out photo-4x.png --method real-esrgan-x4
```

## Classical vs ML

| | Classical (`--scale N`) | ML (`--method real-esrgan-x2` / `-x4`) |
|---|---|---|
| Speed | instant | seconds (loads a small model) |
| Quality | soft (interpolated) | sharp (detail restored) |
| Download | none | ~64 MB Real-ESRGAN weights (first run) |
| Use for | a quick resize | rescuing a low-res render / photo |

## Flags

| Flag | Meaning |
|---|---|
| `--in` / `--out` | source image / output path |
| `--scale <N>` | classical scale factor (e.g. `2`, `4`) |
| `--method <M>` | resampler (`lanczos`, …) or ML model (`real-esrgan-x2`, `real-esrgan-x4`) |
| `--device <D>` | `metal` / `cpu` for the ML path |

## Metal note (×4 OOM)

Real-ESRGAN **×4** on a large input can exceed Apple Silicon's single-buffer
limit and OOM. plakat now catches that and tells you to **retry with
`--device cpu`** (slower, but no buffer cap) — or use **×2** instead. ×2 fits
comfortably on Metal.

```bash
plakat upscale --in big-render.png --out huge.png --method real-esrgan-x4 --device cpu
```

## Diffusion upscale (ControlNet-Tile)

A third mode, beyond classical and Real-ESRGAN: `--diffusion` runs a
diffusion **super-resolution** pass (ControlNet-Tile, SUPIR-lite).
Instead of interpolating (classical) or restoring learned detail from a
fixed SR net (Real-ESRGAN), it first pre-upscales with Lanczos, then
does a **tiled img2img refine** where each tile is guided by a
ControlNet-Tile conditioner. The ControlNet keeps each tile faithful to
the source structure while the diffusion pass **hallucinates coherent
detail** (skin pores, foliage, fabric weave). Tiles overlap and are
blended with a feathered seam, so the output has no visible grid.

```bash
# Coherent 512→2K with hallucinated detail
plakat upscale --in small.png --out big.png --scale 2 --diffusion \
  --model sd15 --tile 512 --overlap 96 --tile-strength 0.4 --steps 20 --guidance 6
```

SD 1.5 is the default backbone (tile 512). For more detail per tile use
SDXL: `--model sdxl --tile 1024`.

| Flag | Meaning |
|---|---|
| `--diffusion` | Enable the diffusion (ControlNet-Tile) upscale path |
| `--scale <N>` | Target scale factor (e.g. `2`, `4`) |
| `--model <M>` | Backbone: `sd15` (tile 512) or `sdxl` (tile 1024) |
| `--tile <PX>` | Tile size — `512` for sd15, `1024` for sdxl |
| `--overlap <PX>` | Overlap between tiles (feathered blend, no seams) |
| `--tile-strength <F>` | img2img strength per tile, `0.3`–`0.5`. Higher invents more detail but risks tile-to-tile drift |
| `--cn-strength <F>` | ControlNet residual scale (default `1.0`) |
| `--steps <N>` | Denoise steps per tile |
| `--guidance <F>` | CFG scale for the refine |
| `--prompt <STR>` | Detail prompt (default generic: "highly detailed, sharp focus, intricate texture") |
| `--seed <N>` | Fix the randomness |

The first run downloads the Tile ControlNet (~1.4 GB) plus the chosen SD
model. This mode is **heavier** than classical or Real-ESRGAN — it runs
a full diffusion pass per tile — so budget more time for large images.

A common pipeline: generate at a model's native size, then upscale — see
[`SCRIPTING_TUTORIAL`](SCRIPTING_TUTORIAL.md) for a `generate → upscale → save`
chain in one Bund script.
