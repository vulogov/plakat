# Flux tutorial

Flux (Black Forest Labs' 12B-parameter rectified-flow transformer)
is plakat's flagship model for high-quality text-to-image. This
tutorial walks through the full Flux feature set: quantization
(GGUF + NF4), LoRA, img2img + Fill inpaint, ControlNet, tiled
hi-res, Redux image conditioning, the "concept" variants
(Canny-dev / Depth-dev), and the `--fast` distillation presets.

For one-line reference of every flag, see
[`GENERATE.md`](../GENERATE.md) and [`IMG2IMG.md`](../IMG2IMG.md).
This tutorial focuses on the *why* and the trade-offs.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md).
- A GPU with at least 16 GB VRAM (see "memory tiers" below for
 how plakat lets you run Flux on less).
- A HuggingFace account + token for the gated `FLUX.1-dev` and
 `FLUX.1-Fill-dev` repos. `FLUX.1-schnell` is ungated.
- `HF_TOKEN` environment variable set, or `~/.cache/huggingface/token`
 from `huggingface-cli login`.

## 1. The Flux variants

| Alias | What it is | Steps | Guidance | Gated? |
|---|---|---|---|---|
| `flux-schnell` | 4-step distillation. Quality below Dev but no HF gate. | 4 | 1.0 | no |
| `flux-dev` | Full FLUX.1-dev. 12B params, ~24 GB BF16. | 28 | 3.5 | yes |
| `flux-fill-dev` | BFL's dedicated inpaint model. 128-channel `img_in` (64 noise + 64 masked-latent + 256 mask). | 28 | 30.0 | yes |
| `flux-canny-dev` | "Concept" model: canny conditioning baked into a 128-channel `img_in`. | 28 | 30.0 | yes |
| `flux-depth-dev` | Same shape as Canny-dev, depth conditioning. | 28 | 30.0 | yes |

The "concept" variants (Canny-dev / Depth-dev) are a separate
category from ControlNet — the conditioning is baked into the
transformer weights rather than added by a separate adapter
network. They use higher guidance (30 vs 3.5) and require a
pre-rendered conditioning map.

## 2. Your first Flux image

```bash
plakat generate "a Victorian mansion at twilight, cinematic" \
 --model flux-dev --seed 42 --size 1024x1024
```

First run downloads ~24 GB. Subsequent runs use the cached weights.
You'll see ~30s of model load + ~30s/step on a 24 GB GPU, so the
first generation completes in roughly 15 minutes. Don't be alarmed
by the load time — Flux is genuinely big.

If you see "out of memory" errors, jump to the quantization
section. Flux scales down well.

### Attention syntax (v0.18)

Flux now respects A1111 / NovelAI-style emphasis: `(token:1.4)`,
`[token]`, nested `((token))`. The per-token weight broadcasts onto
T5-XXL's hidden states (CLIP-L on Flux is pooled-only, so the weight
broadcast is a no-op there). This is the syntax every Civitai Flux
LoRA card uses on its example prompts.

```bash
plakat generate "a (cyberpunk:1.4) night market, [muted colors]" \
 --model flux-dev --seed 42
```

The T5 sentencepiece tokenizer may produce a slightly different
subtoken split for a segment in isolation vs the same text inside a
longer string. The weight-per-resulting-subtoken contract is
preserved either way; the visual effect of `(token:1.5)` matches
A1111's behavior even when the subtoken count drifts by one.

## 3. Quantization: GGUF vs NF4

Flux's BF16 weights are ~24 GB. Most consumer GPUs can't hold that
plus the activations needed for inference. plakat offers two
quantization paths to fit Flux into 12-16 GB GPUs:

### GGUF (community standard)

```bash
plakat generate "..." --model flux-dev-gguf --quant-level Q4_K_S
```

Uses [city96/FLUX.1-dev-gguf](https://huggingface.co/city96/FLUX.1-dev-gguf)
quantized weights. Default `Q4_K_S` is ~7 GB transformer; available
levels run Q2_K (smallest, lossiest) through F16 (no quantization,
same size as BF16). The trade-off:

| Level | Transformer size | Quality |
|---|---|---|
| Q2_K | ~3.5 GB | noticeable degradation |
| Q4_K_S | ~7 GB | imperceptible for most prompts |
| Q5_K_M | ~9 GB | slightly better than Q4 |
| Q8_0 | ~13 GB | indistinguishable from BF16 |
| F16 | ~24 GB | full precision |

Pair with `--quantize-t5` and `--t5-quant-level Q4_K_M` to also
quantize the T5-XXL text encoder (drops it from ~9 GB to ~3 GB).
Most users land on the combo `--model flux-dev-gguf --quant-level
Q5_K_M --quantize-t5` as the sweet spot.

### NF4 (bitsandbytes 4-bit)

```bash
plakat generate "..." --model flux-dev-nf4
```

Uses [lllyasviel/flux1-dev-bnb-nf4-v2](https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4-v2)
(~6 GB transformer). NF4 quantization is the bitsandbytes format
made popular by QLoRA — slightly different math than GGUF, with
per-block normalisation and a 16-value normal-distribution-aware
codebook. plakat runs NF4 via per-call dequant (no fused
dequant+matmul kernel like GGUF), so step time is slower than
GGUF, but the model loads faster.

**When to pick which**
- GGUF for speed on capable GPUs (16+ GB).
- NF4 for the smallest possible memory footprint (works on 12 GB).
- NF4 for first-time users — single-file download, no quant level to pick.

Both compose with LoRA and ControlNet.

## 4. LoRA stacking

Flux LoRAs use diffusers PEFT format and target the
DoubleStream / SingleStream transformer blocks. plakat reads them
from HuggingFace or local paths.

```bash
# Single LoRA from HF
plakat generate "..." --model flux-dev \
 --lora "alvdansen/frosting_lane_flux"

# Local file with custom strength
plakat generate "..." --model flux-dev \
 --lora "./my-style.safetensors:0.75"

# Stack multiple
plakat generate "..." --model flux-dev \
 --lora "user/style-lora:0.8" \
 --lora "user/character-lora:0.6"
```

LoRA composes with quantization. The merge happens at load time —
plakat dequantizes only the LoRA-targeted Linears (Q/K/V/proj/MLP
of each transformer block), applies the delta, leaves the rest
quantized. Memory savings preserved.

```bash
# LoRA on quantized Flux — works fine
plakat generate "..." --model flux-dev-gguf --quant-level Q5_K_M \
 --lora "user/style-lora:0.8"

plakat generate "..." --model flux-dev-nf4 \
 --lora "user/style-lora:0.8"
```

## 5. The `--fast` distillation presets

Hyper-FLUX and FLUX-Turbo are distillation LoRAs that let Flux
sample in 8-16 steps instead of 28. `--fast` bundles the right
preset:

```bash
# 8-step Hyper-FLUX
plakat generate "..." --model flux-dev --fast hyper-8

# 16-step Hyper-FLUX (higher quality, 2x slower)
plakat generate "..." --model flux-dev --fast hyper-16

# 8-step turbo-alpha
plakat generate "..." --model flux-dev --fast turbo-alpha
```

Each preset prepends the matching LoRA + sets step / guidance
defaults appropriate to the distillation. You can still stack
your own LoRAs alongside:

```bash
plakat generate "..." --model flux-dev --fast hyper-8 \
 --lora "user/style-lora:0.8"
```

`--fast` doesn't compose with `flux-fill-dev` (Fill's inpaint math
is incompatible with the distillation training schedule). NF4 +
`--fast` works — the preset LoRA merges via the runtime LoRA path.

## 6. ControlNet on Flux

plakat ships with [Shakker-Labs Union Pro v2](https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0)
as the default Flux ControlNet — one model covering canny, soft
edge, OpenPose, depth, and lineart via a mode index.

```bash
# Canny-guided generation. Provide a pre-rendered map:
plakat generate "..." --model flux-dev \
 --control-spec 'canny:image=./edges.png:strength=0.7'

# Or let plakat auto-annotate from a photo:
plakat generate "..." --model flux-dev \
 --control-spec 'canny:from=./photo.jpg:strength=0.7'

# Stack multiple ControlNets (residuals sum):
plakat generate "..." --model flux-dev \
 --control-spec 'depth:from=./scene.jpg:strength=0.8' \
 --control-spec 'openpose:from=./pose.jpg:strength=0.6'

# Step gating — apply only during the first half of the denoise:
plakat generate "..." --model flux-dev \
 --control-spec 'canny:from=./photo.jpg:strength=0.7:start=0.0:end=0.5'
```

Flux ControlNet works on Dev, Fill, GGUF, and NF4. It does **not**
compose with the "concept" variants (Canny-dev / Depth-dev) — those
have conditioning baked into the transformer, doubling up with a CN
would over-condition.

## 7. Flux img2img

Same `plakat img2img` subcommand, just pass `--model flux-dev`:

```bash
plakat img2img photo.jpg --model flux-dev \
 --prompt "the same scene rendered as a stained glass window" \
 --strength 0.7
```

The math: VAE-encode the input, lerp with fresh noise at
`t = strength`, denoise the truncated schedule. Same `--strength`
convention as SD-family img2img (0.0 = no change, 1.0 = ignore
input).

For inpaint — replacing just a masked region — use Flux.1-Fill-dev:

```bash
plakat img2img init.png --mask region.png \
 --model flux-fill-dev \
 --prompt "ornate carved stone window"
```

Fill's `img_in` is 384 channels (64 noise + 64 masked-latent + 256
image-space mask). The mask drives the denoise directly — no
RePaint-style strength blending. Default `--guidance 30` per BFL's
model card. Fill composes with ControlNet, and — —
with `--tiled` (per-tile masked-latent + mask slicing inside the
denoise loop; same Hann-blend the standard tiled path uses).

## 8. Tiled hi-res

Flux at >1024² runs out of memory on most GPUs. The tiled path
runs MultiDiffusion-style overlapping tiles:

```bash
plakat generate "ultra-detailed architectural diagram" \
 --model flux-dev --size 3072x2048 \
 --tiled --tile-size 1024 --tile-stride 768
```

Each step splits the canvas into 1024-px tiles (stride 768 means
256 px overlap), runs Flux per tile, blends predictions with a
2D Hann window. Total VRAM stays at the 1024² working set.

Composes with: GGUF / NF4, LoRA, ControlNet (per-tile residuals
sliced correctly), img2img, **Fill** (per-tile masked-latent + mask
packing, +). Does **not** compose with: the concept variants
(Canny-dev / Depth-dev), Redux.

```bash
# 4K tiled inpaint — Fill at canvas sizes that wouldn't fit
# whole-canvas in 24GB VRAM.
plakat generate "ornate carved stone window" \
 --model flux-fill-dev --size 2048x2048 \
 --image room.png --mask wall.png \
 --tiled --tile-size 1024 --tile-stride 768
```

## 9. Flux Redux (image conditioning)

Redux is BFL's "image prompt" adapter — encodes an image through
SigLIP-so400m and prepends 729 tokens onto the T5 hidden state.
Different from ControlNet (which guides structure) — Redux carries
*style + content* signal.

```bash
# One reference image
plakat generate "in this style" --model flux-dev \
 --redux-image style.png

# Multi-image with weights
plakat generate "..." --model flux-dev \
 --redux-image style.png:weight=0.8 \
 --redux-image subject.png:weight=0.4
```

Up to 4 reference images. Higher weights pull harder toward the
reference; weight=0 effectively disables an entry. Composes with
all base Flux variants, LoRA, ControlNet, tiled. Does **not**
compose with `flux-fill-dev` (Fill's 384ch `img_in` is incompatible
with Redux's token concat).

## 10. The "concept" variants — Canny-dev / Depth-dev

These are full Flux.1-dev models retrained with canny edge or depth
map conditioning baked into the `img_in` Linear (which becomes 128
channels). Unlike ControlNet (separate adapter), the conditioning is
part of the base model.

Two ways to supply the conditioning map: `--concept-image PATH` for
a pre-rendered map you already have, or `--concept-from PATH` to
auto-annotate a photo with the matching annotator (canny for
Canny-dev, depth for Depth-dev).

```bash
# Auto-annotate from a reference photo. plakat runs the right
# annotator based on which concept variant is loaded.
plakat generate "a Victorian mansion, gothic, twilight" \
 --model flux-canny-dev \
 --concept-from photo.jpg \
 --guidance 30

plakat generate "a polished marble statue of an angel" \
 --model flux-depth-dev \
 --concept-from reference.jpg \
 --guidance 30

# Or use a pre-rendered map you generated externally
plakat generate "..." --model flux-canny-dev \
 --concept-image edges.png \
 --guidance 30
```

BFL recommends guidance ~30 for the concept variants — much higher
than standard Flux's 3.5. Lower values underrespect the
conditioning; the math here is that the model needs strong CFG to
actually steer toward the conditioning map.

**Caveats**: concept variants don't compose with `--tiled`,
`--init-image`/`--mask`, `--redux-image`, or `--control-spec`
(adding ControlNet on top of baked conditioning would double-
condition).

## 10b. FLUX.1-Kontext-dev (image editing, v0.18)

Kontext is BFL's image-editing Flux variant. Different conditioning
mechanism from Canny / Depth: the reference image is VAE-encoded and
**sequence-concatenated** onto the noise tokens (with
`img_ids[..., 0] = 1` as a RoPE positional marker), instead of
being channel-concatenated into a widened `img_in`. `img_in` stays
at 64 channels — Kontext is architecturally identical to Flux.1-dev
at the Linear level; the difference lives entirely in the pipeline's
per-step concat + post-step strip.

Two ways to invoke:

```bash
# `plakat generate` — explicit reference flag
plakat generate "make the lighting golden hour" \
 --model flux-kontext-dev \
 --concept-image input.png

# `plakat img2img` — input positional IS the reference (natural
# for users coming from "edit this image" workflows)
plakat img2img input.png \
 --model flux-kontext-dev \
 --prompt "add snow on the ground"
```

GGUF (4-bit/6-bit/8-bit) is supported via the unsloth mirror — same
~7 GB transformer footprint as Flux.1-dev GGUF:

```bash
plakat generate "..." --model flux-kontext-dev-gguf \
 --concept-image input.png --quant-level Q6_K

# GGUF + LoRA composes (Kontext shares Dev's transformer layer
# names, so any flux-dev LoRA applies unchanged)
plakat generate "..." --model flux-kontext-dev-gguf \
 --concept-image input.png \
 --lora civitai:12345:0.7
```

NF4 is **not** supported (no upstream NF4 pack ships at time of
writing; use BF16 or GGUF instead).

### Aspect-bucket snap

BFL publishes 17 preferred resolutions for Kontext (1024×1024,
672×1568, 1568×672, and 14 intermediate aspect ratios, all multiples
of 16). Opt-in via `--kontext-bucket`:

```bash
plakat generate "..." --model flux-kontext-dev \
 --concept-image input.png \
 --size 1600x1200 --kontext-bucket
# → snaps 1600x1200 (4:3) to the closest published bucket
```

Off by default so non-Kontext workflows aren't surprised by silent
size changes. Per-task `kontext-bucket: true` in scenarios mirrors
the CLI flag.

### Caveats

- **Requires `--concept-image`** (the reference) — Kontext is an
  editing model, not a pure generator. Without a reference it bails.
- **No `--mask` / `--init-image`** — Kontext doesn't have a mask
  path or a flow-match init lerp. For mask-restricted edits use
  `flux-fill-dev` instead.
- **No `--tiled`** in this release — per-tile reference slicing
  isn't wired (the reference would have to be sliced along the
  same tile grid as the noise canvas).
- **`--redux-image` composes** (v0.19). Redux extends the txt
  sequence (+729 SigLIP tokens per image); Kontext extends the
  img sequence (+~1k VAE reference tokens). They're orthogonal in
  implementation but compound on Flux's RoPE budget. plakat
  computes the total effective seq length and:
  - warns at 3500 positions (RoPE may degrade);
  - bails at 4096 positions (the safe Flux RoPE cap) with
    actionable cleanup hints (`--size`, ref dims, `--redux-image`
    count).

  Use case: "edit this image in the style of these reference
  images" — Kontext provides the layout, Redux provides the
  style/aesthetic.

  ```bash
  plakat generate "the same scene at golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --redux-image style_ref.png:weight=0.5
  ```

  Rule of thumb for 1024×1024 outputs: ~2.5k positions baseline,
  +729 per Redux image. 2 Redux images stays under the warn
  threshold; 3+ pushes into the warn zone; 4 may approach the cap.
  Smaller outputs (768×768) buy ~700 more positions of headroom.
- **`--control-spec` composes** (v0.19). The ControlNet runs on the
  noise tokens only; its per-block residuals get zero-padded for
  the reference half (the reference is already conditioning via
  cross-attention, so no additional CN contribution there). Use
  case: "edit this image, preserve the depth structure" or
  "edit this image, keep the canny edges from a different source".

  ```bash
  plakat generate "make it golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --control-spec 'depth:from=input.png:strength=0.7'
  ```

  The CN's `start` / `end` step gating works normally. Multi-CN
  composes too (residuals sum per-block then pad uniformly).

## 11. Per-task LoRA in scenarios

Scenarios can declare per-task LoRA stacks that compose with the
scenario-level LoRAs:

```hjson
{
 model: flux-dev
 loras: [ "user/global-style-lora:0.5" ]
 enhancer: deepseek

 scene: [{ name: forest, prompt: "deep forest, sunbeams" }]
 weather: [{ name: golden, prompt: "golden hour" }]

 tasks: [
 {
 name: portrait, scene: forest, weather: golden,
 prompt: "a hooded figure",
 loras: [ "user/character-lora:0.7" ],
 },
 {
 name: landscape, scene: forest, weather: golden,
 prompt: "wide panorama, no figures",
 loras: [ "user/landscape-lora:0.8" ],
 },
 ]
}
```

The scenario-level `loras:` apply to every task (load-time merge).
The per-task `loras:` add additional LoRAs at runtime, swapped
between tasks. Supported on Flux variants (BF16 / GGUF / NF4). For
SD-family and SD3, use scenario-level `loras:` instead — plakat
raises an explicit error if you set per-task `loras:` on those.

## 12. Memory tiers — picking the right backbone

| GPU VRAM | Recommended config |
|---|---|
| 8 GB | `--model flux-dev-nf4`. Just barely fits. Skip `--tiled`. |
| 12 GB | `--model flux-dev-nf4` or `--model flux-dev-gguf --quant-level Q4_K_S --quantize-t5`. |
| 16 GB | `--model flux-dev-gguf --quant-level Q5_K_M --quantize-t5`. Most flexibility for stacking features. |
| 24 GB | `--model flux-dev` (BF16). Full quality. `--tiled` works for 2K-4K outputs. |
| 32 GB+ | `--model flux-dev` + `--tiled` for 4K+. Stack LoRAs freely. |

If a config OOMs, drop one quality tier (Q5_K_M → Q4_K_S, or
GGUF → NF4) or shrink `--size`. Flux is sensitive to size — going
from 1024² to 512² roughly halves VRAM.

## 13. Common gotchas

- **First-run downloads are slow.** Flux.1-dev is ~24 GB BF16, ~9 GB
 T5, ~250 MB CLIP. Plan for 30-45 GB of disk + matching download
 time on first use of each variant.
- **HF gate.** `flux-dev` / `flux-fill-dev` / `flux-canny-dev` /
 `flux-depth-dev` require accepting BFL's license terms on
 HuggingFace and exporting `HF_TOKEN`.
- **Guidance scales differ.** Standard Flux: 3.5. Fill / Canny /
 Depth: 30. Schnell: 1.0. plakat defaults to the variant's
 recommended value if you don't pass `--guidance`.
- **`--fast` ignores user `--steps` only if you didn't set it.**
 If you explicitly pass `--steps 28` with `--fast hyper-8`, plakat
 honors your choice. The preset only fills in defaults.
- **Concept variants need a conditioning map.** Either supply a
 pre-rendered canny / depth map via `--concept-image PATH`, or let
 plakat auto-annotate a photo via `--concept-from PATH` — it picks
 the right annotator (canny / depth) based on the loaded variant.

## What's next

- [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md) covers the broader
 img2img / inpaint flow (SD-family and Flux side-by-side).
- [`CONTROLNET_TUTORIAL.md`](CONTROLNET_TUTORIAL.md) goes deep on
 ControlNet structural guidance on SD; everything there transfers
 to Flux via `--model flux-dev` + the same flags.
- [`SD3_TUTORIAL.md`](SD3_TUTORIAL.md) — for the Stability AI MMDiT
 family. Different architecture from Flux but covers similar
 ground (variants, LoRA, img2img, tiled, ControlNet).
- [`GENERATE.md`](../GENERATE.md) — full flag reference.
