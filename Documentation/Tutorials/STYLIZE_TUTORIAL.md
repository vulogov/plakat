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

- **`--instantstyle`** (SDXL) — the *same* `stylize` command with decoupled
  style-block injection, so the reference's *texture* transfers, not its content.
  The most direct true-style path — see [InstantStyle](#instantstyle--true-painterly-style-transfer-sdxl) below.
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

## InstantStyle — true painterly style transfer (SDXL)

The default path transfers *content/palette*, not texture. **`--instantstyle`**
fixes that: it injects the reference into the **style block only** (SDXL
`up_blocks.0.attentions.1`) via a decoupled IP cross-attention, so the
reference's *brushwork* transfers while its content stays out. Same `stylize`
command, no prompt, no training — the real style machine.

```bash
plakat stylize \
  --in  portrait.png \
  --ref watercolour.jpg \
  --instantstyle --style-scale 4 \
  --strength 0.8 --model sdxl \
  --out watercolour-portrait.png
```

Because it runs as img2img, the t2i-canonical `--style-scale 1.0` is too timid —
push **`--style-scale ~3–5`** and **`--strength ~0.8`** so the denoise has room
to repaint. The first run loads a second (vendored) UNet → extra memory.

| Flag | Effect |
|---|---|
| `--instantstyle` | Enable decoupled style-block injection (vs the default concat). SDXL recommended. |
| `--style-scale <s>` | Injection strength. Default `3.0`; raise (~4–5) for heavier paint, lower to keep more of the photo. |

**SD 1.5** (`--model sd15 --instantstyle`) works but is **experimental** — the
style block is correct (the full `up_blocks.1`), but SD 1.5's style perception is
weak: it styles softly, and pushing the scale melts structure rather than adding
clean texture. Prefer SDXL.
