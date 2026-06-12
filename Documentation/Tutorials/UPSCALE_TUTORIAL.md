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

A common pipeline: generate at a model's native size, then upscale — see
[`SCRIPTING_TUTORIAL`](SCRIPTING_TUTORIAL.md) for a `generate → upscale → save`
chain in one Bund script.
