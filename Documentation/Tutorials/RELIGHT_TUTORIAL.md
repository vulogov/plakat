# Relight — re-illuminate a subject (IC-Light)

`plakat relight` re-lights a foreground **subject** under a lighting you describe
in text — **no training, no reference photo of the light**. It preserves the
subject's identity (face, garb, shape) while changing the illumination and the
surrounding scene.

```bash
plakat relight portrait.png \
  --prompt "warm sunset light from the left, golden hour, cinematic" \
  --out relit.png
```

## What it is

It's IC-Light ("Imposing Consistent Light", lllyasviel). Under the hood:

- It's **SD 1.5-based**. plakat widens the UNet's input conv from 4→8 channels
  and merges the `lllyasviel/ic-light` *offset* over a base SD 1.5 UNet.
- The subject is matted off its background (U2Net), composited onto a neutral
  grey field, and VAE-encoded. That subject latent is **concatenated to the
  noisy latent at every denoise step** (the extra 4 channels), so the model
  always "sees" the subject while it invents lighting from your prompt.

The result: the captain's face and beard stay put; the light, mood, and backdrop
become whatever you asked for.

## Lighting presets *(6.23)*

The quickest way to relight is a **named preset** — a curated prompt **plus a
directional cue** (the subject is composited over a gradient that's brighter on the
light side, so the direction actually lands, not just the words):

```bash
plakat relight portrait.png --light key-left  --out out.png   # dramatic side key
plakat relight portrait.png --light rim        --out out.png   # backlit glowing edge
plakat relight portrait.png --light golden-hour --out out.png
plakat relight --list-lights                                   # key-left/right · top · rim · softbox ·
                                                               # golden-hour · sunset · moonlight ·
                                                               # candlelight · neon · overcast
```

Add your own `--prompt "…"` to *extend* a preset, or `--light-angle 45` to steer the
direction (0 = left, 90 = top, 180 = right, 270 = bottom). From the library:
`plakat::api::Relight::new("portrait.png").light("key-left").run().await?`.

## Examples (freeform prompt)

```bash
# Sunset rim light
plakat relight portrait.png \
  --prompt "dramatic sunset light from the left, warm golden rim light, magic hour" \
  --out sunset.png

# Cool blue twilight / moonlight
plakat relight portrait.png \
  --prompt "cool blue twilight, soft moonlight from the right, misty, cinematic" \
  --negative "flat lighting, washed out" \
  --out blue_hour.png

# Warm firelight from below
plakat relight portrait.png \
  --prompt "warm orange firelight from below, cozy tavern glow, flickering hearth" \
  --out hearth.png
```

## Flags

| Flag | Meaning |
|---|---|
| `<subject>` | subject image (positional). Its background is matted away automatically. |
| `--prompt` | the lighting / scene description (required). |
| `--negative` | negative prompt (default empty). |
| `--size <N>` / `<WxH>` | output size; `512` square by default. |
| `--steps <N>` | denoise steps (default `25`). |
| `--guidance <G>` | classifier-free guidance (default `2.0`). |
| `--seed <N>` | seed (omit for a random one). |
| `--out <PATH>` | output file, or a directory (a name is generated). Default `./`. |

## Tip: keep guidance LOW

IC-Light wants **low CFG — 1.5 to 3, default 2.0**. Higher guidance washes the
subject out (the light overpowers the identity). If your relit subject looks
blown-out or plasticky, drop `--guidance` rather than raising it.

## Hardware

SD 1.5 fits **24 GB comfortably**. Build with a GPU backend for usable speed:

```bash
cargo build --release --features metal   # Apple Silicon
cargo build --release --features cuda    # NVIDIA
```

## A showcase

`corpus/relight.sh` relights the cached sage-captain portrait under three moods
(sunset / blue_hour / hearth) and drops them in `corpus/images/relight/` — a
ready gallery row showing the same subject, three lights:

```bash
./corpus/relight.sh
plakat gallery corpus/images --recursive --out corpus/GALLERY.md
```
