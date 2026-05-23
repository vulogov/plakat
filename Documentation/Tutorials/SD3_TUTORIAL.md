# SD3 / SD3.5 tutorial

Stable Diffusion 3 / 3.5 is Stability AI's MMDiT (Multimodal Diffusion
Transformer) architecture — a successor to the SD 1.5 / SDXL UNet
family. plakat supports the full lineup: SD3 Medium, SD3.5 Medium,
SD3.5 Large, and SD3.5 Large Turbo.

This tutorial walks through the variants, prompt encoding,
rectified-flow sampler, LoRA, img2img, tiled hi-res, and
ControlNet. SD3 ControlNet (InstantX-pattern) lands in v0.16 —
the resolver maps `--control-spec kind=...` to the matching
InstantX checkpoint per variant; see §11 below.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md).
- A HuggingFace account + `HF_TOKEN` — every Stability SD3 repo is
  gated.
- A GPU with at least 12 GB VRAM (SD3.5-Medium fits comfortably;
  SD3.5-Large wants more). See "memory" below.

## 1. The SD3 lineup

| Alias | Hidden | Depth | Recommended use |
|---|---|---|---|
| `sd3-medium` | 1536 | 24 | Original SD3 (June 2024). Known anatomy issues; SD3.5 is the recommended baseline. |
| `sd35-medium` | 1536 | 24 | SD3.5 Medium. The everyday workhorse. ~2.5B params. |
| `sd35-large` | 2432 | 38 | SD3.5 Large. The flagship. 8B params. |
| `sd35-large-turbo` | 2432 | 38 | 4-step distillation of Large. CFG-free (`--guidance 0`). |

All share the same architecture (MMDiT joint blocks) and tokenizer
setup (triple text encoder: CLIP-L + CLIP-G + T5-XXL). The
differences:

- **Hidden size** scales with depth × head_size. Large is wider
  → higher quality + more VRAM.
- **`pos_embed_max_size`** differs: 384 for SD3.5 Medium (can
  handle up to 1536² natively), 192 for SD3 Medium / SD3.5 Large
  (up to 1536² with tiled, but the implicit canvas is smaller).
- **Turbo's distillation** trains for 4 steps + no CFG. Pass
  `--steps 4 --guidance 0` (plakat picks these defaults
  automatically when you select the Turbo variant).

## 2. Your first SD3 image

```bash
plakat generate "a Victorian mansion at twilight" \
    --model sd35-medium --seed 42 --size 1024x1024
```

First run downloads the MMDiT weights (~5 GB for Medium, ~17 GB for
Large) plus the three text encoders (~9 GB T5, ~1.5 GB CLIP-L+G).
Plan for ~10 GB disk on Medium, ~25 GB on Large.

SD3 uses a **rectified-flow sampler** (same family as Flux) — not
the DDIM / Euler the SD UNet uses. Sampling is fast: 28 steps at
1024² takes ~30-60s on a 24 GB GPU. The Turbo variant cuts to ~4s
at 4 steps.

## 3. How prompts get encoded

SD3's MMDiT consumes two text-derived signals:

- **`y` (pooled, 2048d)** — `[CLIP-G_pooled (1280) || CLIP-L_pooled (768)]`.
  Carries the high-level semantic signal.
- **`context` (1, 77+t5_seq, 4096)** — sequence-concat of CLIP
  (zero-padded from 2048 → 4096) + T5's hidden state. Carries the
  fine-grained text signal.

T5-XXL is the workhorse for prompt understanding — it's the
encoder that lets SD3 follow long, detailed prompts. The 256-token
T5 budget is the canonical SD3 paper value; longer prompts get
truncated.

Practical effect: **SD3 follows detailed prompts much better than
SD 1.5 / SDXL**. You can write paragraphs of description and SD3
will respect them. Don't fight this — write longer prompts.

## 4. SD3 vs SD3.5 vs Large vs Turbo

```bash
# Original SD3 Medium — historical curiosity, anatomy issues
plakat generate "..." --model sd3-medium

# SD3.5 Medium — recommended baseline, 1024-1536² sweet spot
plakat generate "..." --model sd35-medium --size 1024x1024

# SD3.5 Large — flagship, more VRAM, more detail
plakat generate "..." --model sd35-large --size 1024x1024

# SD3.5 Large Turbo — 4 steps, no CFG, fastest path to good output
plakat generate "..." --model sd35-large-turbo
```

The Turbo variant **ignores** `--guidance` higher than 0. Its
distillation training has no conditional/unconditional pairing —
the model is trained to produce final outputs in 4 steps from
prompt alone. Pass `--guidance 0` (or let plakat default).

## 5. LoRA support

SD3 LoRAs use diffusers PEFT format and target the MMDiT joint
blocks. Stability publishes a LoRA cookbook with the canonical
target keys. plakat reads them and merges into the MMDiT weights at
load time via a tempfile (same pattern as Flux BF16 LoRA).

```bash
# Single LoRA from HF
plakat generate "..." --model sd35-large \
    --lora "user/style-lora"

# Stack multiple, custom strengths
plakat generate "..." --model sd35-large \
    --lora "user/style-lora:0.8" \
    --lora "user/character-lora:0.5"

# Apply at custom scale (multiplies every LoRA's own :weight)
plakat generate "..." --model sd35-large \
    --lora "user/style-lora" --lora-scale 0.7
```

LoRA composes with img2img and tiled. Scenarios support both
scenario-level `loras:` (merged into MMDiT at load time) and
per-task `loras:` (applied at runtime between tasks via the MMDiT
LoraLinear stack).

## 6. SD3 img2img

Same `plakat img2img` subcommand, just pass `--model sd35-*`:

```bash
# img2img — re-imagine the input at 60% strength
plakat img2img photo.png --model sd35-medium \
    --prompt "the same scene rendered as a watercolor"

# Inpaint — replace just the masked region (RePaint-style)
plakat img2img photo.png --model sd35-large \
    --mask sky.png \
    --prompt "dramatic stormy sky, lightning"

# Turbo + img2img: 4-step distilled, no CFG
plakat img2img photo.png --model sd35-large-turbo \
    --prompt "..." --guidance 0
```

The math: VAE-encode the init image, lerp with fresh noise at
`t = strength` using the rectified-flow interpolation, denoise on
a truncated schedule. Inpaint adds a per-step mask blend that
keeps unmasked pixels on the init's flow trajectory.

Output naming reflects the mode: `plakat-sd3-{img2img,inpaint}-<seed>.png`.

**Caveats**:
- SD3 ControlNet works on t2i only — the img2img CLI doesn't
  forward `--control-spec`. Use `plakat generate` for CN-guided
  outputs.
- Tiled + img2img doesn't compose — drop one or the other.

## 7. Tiled hi-res

```bash
plakat generate "ultra-detailed architectural diagram" \
    --model sd35-medium --size 2048x2048 \
    --tiled --tile-size 1024 --tile-stride 768
```

Same MultiDiffusion-style tiled denoise as SDXL / Flux. plakat
splits the latent into overlapping `tile-size`-px windows, runs
MMDiT per tile, blends predictions with a 2D Hann window.

MMDiT-specific constraint: `tile-size` divided by 16 (VAE_factor
8 × patch_size 2) must be ≤ the variant's `pos_embed_max_size`
(384 for SD3.5-Medium; 192 for SD3 / SD3.5-Large). At the default
1024-px tile, that's 64 patches per axis — well within either
cap. Pushing past ~3000-px tiles on SD3.5-Large would hit the
limit.

```bash
# 4K SD3.5 Medium
plakat generate "..." --model sd35-medium --size 4096x2160 \
    --tiled --tile-size 1024 --tile-stride 768

# Tiled SD3.5 Large — slower per tile but higher fidelity
plakat generate "..." --model sd35-large --size 2048x2048 \
    --tiled
```

Composes with LoRA. Does **not** compose with img2img / inpaint on
SD3 — drop one or the other.

## 8. Sampler & guidance

SD3 uses rectified-flow with a **time-shift** transform:

```
f(t) = shift · t / (1 + (shift - 1) · t)
```

over a linear `[1.0, 0.0]` step schedule. Different `shift` values
emphasize different parts of the schedule:

- `shift = 1.0` → linear (Turbo uses this).
- `shift = 3.0` → bias toward high-noise (most steps spent
  resolving early, low-info structure). Default for SD3 Medium /
  SD3.5 Medium / SD3.5 Large.

plakat picks the right shift per variant — you don't set it
manually.

**CFG (`--guidance`)**:
- SD3.5 Medium / Large / SD3 Medium: default 4.5 (Stability's
  recommendation across the lineup).
- Sd35-large-turbo: 0 (CFG-free distillation).

CFG works by double-batching `[neg, pos]` per step and blending:
`pred = neg + guidance · (pos - neg)`. Higher guidance = stronger
prompt adherence + more saturated outputs. SD3's 4.5 is gentler than
SDXL's typical 7.5 because MMDiT follows prompts better natively.

## 9. Memory tiers

| GPU VRAM | Recommended config |
|---|---|
| 12 GB | `--model sd35-medium`. Skip `--tiled` (working VRAM tight). |
| 16 GB | `--model sd35-medium --tiled` for >1024² outputs. |
| 24 GB | `--model sd35-large --size 1024x1024` straight, or Medium tiled at 4K. |
| 32 GB+ | `--model sd35-large --tiled` for 4K+ Large. |

If a config OOMs, drop the variant tier (Large → Medium) or shrink
`--size`. plakat doesn't currently ship quantized SD3 variants —
the only memory dial is "smaller variant + smaller size + tiled".
For lower-VRAM workflows, Flux NF4 / GGUF cover the 12–16 GB tier.

## 10. SD3 in scenarios

```hjson
{
    model: sd35-medium
    enhancer: deepseek

    scene: [{ name: forest, prompt: "deep forest, sunbeams" }]
    weather: [{ name: golden, prompt: "golden hour, warm light" }]

    tasks: [
        {
            name: a, scene: forest, weather: golden,
            prompt: "a hooded traveler walking the path",
        },
        {
            name: b, scene: forest, weather: golden,
            prompt: "a wide empty landscape, no figures",
            tiled: { size: 1024, stride: 768 },
        },
    ]
}
```

Per-task `tiled:` overrides work. Per-task `loras:` is supported
— each task's LoRA stack is applied to the MMDiT at runtime on top
of the scenario-merged baseline, then cleared at end-of-task so the
next task isn't contaminated. Per-task `init-image:` + img2img works
the same way it does for Flux.

## 11. SD3 ControlNet

plakat's SD3 ControlNet uses the [InstantX](https://huggingface.co/InstantX)
checkpoints — a small 12-layer transformer that consumes a
VAE-encoded conditioning latent and produces per-block residuals
added to the base MMDiT's joint-block hidden states.

```bash
# Single ControlNet — auto-annotate from a reference photo.
# Canny works on every SD3 variant.
plakat generate "a fantasy castle" \
    --model sd35-medium --size 1024x1024 \
    --control-spec 'canny:from=ref.jpg'

# Strength + step-gating window (active for the first 60% only):
plakat generate "..." --model sd35-large \
    --control-spec 'depth:from=room.jpg:strength=0.8:start=0.0:end=0.6'

# Multi-CN stack: two specs compose at residual level.
plakat generate "..." --model sd35-medium \
    --control-spec 'canny:from=edges.jpg:strength=0.7' \
    --control-spec 'openpose:from=pose.jpg:strength=0.5'
```

**Resolver matrix** (which InstantX repo each `kind=` resolves to,
per variant):

| Variant | Canny | Lineart | SoftEdge | OpenPose | Depth |
|---|---|---|---|---|---|
| `sd35-large` / `-turbo` | Canny | → Canny | Blur | (not released) | Depth |
| `sd35-medium` / `sd3-medium` | Canny | → Canny | → Canny | Pose | (not released) |

Combos marked `(not released)` bail loud with a clear error
suggesting the alternate variant. Lineart / SoftEdge fall back to
Canny on Medium (same pattern Flux Union Pro v2 uses — close-enough
edge channel).

**Composition**:
- Composes with `--lora` (CN residuals are added after LoRA-merged
  forward).
- Composes with multi-CN: each spec's residuals are summed
  block-wise before being fed to the MMDiT.
- Does **not** compose with `--tiled` — the per-tile conditioning
  slice isn't wired yet. plakat bails loud rather than ship
  silent-garbage.
- The img2img CLI doesn't carry `--control-spec` — CN-guided img2img
  is a future phase.

**Tip**: SD3 follows long prompts well, but ControlNet pulls the
geometry hard. Start at `strength=0.7` — full `1.0` often
over-constrains the output and the model can't deliver the
prompt's content. The step-gating window (`start=0.0:end=0.4`) is
the gentler dial: structure pull early, free composition late.

## 12. Common gotchas

- **Gated everywhere.** Every Stability SD3 repo requires accepting
  the license terms. If a download fails with 401, check that
  `HF_TOKEN` is set and you've clicked "agree" on the repo's HF
  page.
- **T5 download is big.** The first SD3 run downloads ~9 GB T5-XXL.
  No GGUF T5 path for SD3 yet (Flux has one via
  `--quantize-t5`; SD3 doesn't).
- **Don't pass `--guidance` to Turbo.** It's a CFG-free
  distillation; high guidance produces over-saturated, often-broken
  outputs. plakat defaults to 0 — only override if you know what
  you're doing.
- **Tiled + tile-stride.** Smaller `--tile-stride` = more overlap
  = smoother seams + more compute. Default 768 (with default
  `--tile-size 1024`, giving 256-px overlap) is the SDXL/Flux
  precedent and works well for SD3 too.
- **SD3 ControlNet.** Supported as of v0.16 via the InstantX
  family. See §11 for the resolver matrix and composition rules.
  Tiled + CN isn't wired — drop `--tiled` if you also want CN.

## What's next

- [`FLUX_TUTORIAL.md`](FLUX_TUTORIAL.md) — for Black Forest Labs'
  Flux family. Different architecture (rectified-flow + dual stream
  blocks) but covers similar ground.
- [`IMG2IMG_TUTORIAL.md`](IMG2IMG_TUTORIAL.md) — broader img2img
  flow across SD-family, Flux, and SD3.
- [`GENERATE.md`](../GENERATE.md) — full flag reference, including
  every SD3 alias.
