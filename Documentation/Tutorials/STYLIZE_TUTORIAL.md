# Stylize — apply a reference image's look (IP-Adapter)

`plakat stylize` reads the *look* of a reference image and applies it to an
input subject via IP-Adapter — **no prompt, no training**. It's the quick way
to nudge an image toward a reference's appearance.

```bash
plakat stylize \
  --in  subject.png \
  --ref reference.png \
  --model sdxl \
  --out  stylized.png
```

## What it is (and isn't)

The IP-Adapter encodes the reference with CLIP and injects it into the
denoiser. That transfers the reference's **content, appearance, and palette** —
but **not** painterly *texture*. So stylize is a **ref-guided variation** tool:
output tends to stay photoreal, even on SDXL (the base paints fine from
prompts/LoRAs — the limit is the IP-Adapter, not the model).

For true painterly **style transfer** (watercolour washes, oil impasto, ink),
use one of:

- **`--look <medium>`** — a curated art-medium preset, optionally backed by an
  LLM-judged Civitai style LoRA (`--smart-discovery`). See
  [LOOKS_TUTORIAL](LOOKS_TUTORIAL.md).
- **`plakat style train`** — learn a style LoRA from your own images. See
  [TRAIN_STYLE_LORA_TUTORIAL](TRAIN_STYLE_LORA_TUTORIAL.md).

## Models

| `--model` | Notes |
|---|---|
| `sdxl` | Sharper, native 1024² (small inputs are scaled up — SDXL glitches below ~1024). Recommended. |
| `sd15` | Lighter (~4 GB), 512²–768². The original path, kept as a fallback. |

## Knobs

| Flag | Effect |
|---|---|
| `--strength` | How much of the input is re-denoised. Higher = heavier restyle, less of the original subject. ~0.5–0.7 is a good range. |
| `--ref-blur <sigma>` | Gaussian-blur the reference before encoding, to suppress its **content** so its broad look dominates (the "style not content" knob). Blur also softens texture, so it suits palette-driven refs. `0` = off. |
| `--ref-weight <w>` | Scale the reference's influence (`1.0` = full). Lower lets the input subject dominate. |

## When the reference has a subject

If the reference is a *photo of a person or object*, the IP-Adapter will try to
carry that subject into the output (a "content hijack"). `--ref-blur` reduces
this by blurring away the fine detail. For pure look transfer, prefer a
reference that is mostly texture/palette rather than a bold subject — or reach
for a style LoRA, which separates style from content by construction.
