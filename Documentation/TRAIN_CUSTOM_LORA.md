# Train your own style LoRA

Turn a folder of images you like into a **paintable style** — a LoRA
`.safetensors` that loads into the model via `--lora` and makes plakat
generate in that style. This is *creation*, not detection: plakat learns
the look from your images and bakes it into a small adapter.

> This is different from `plakat style detect` / the style **catalog**,
> which only *recognises* a style from CLIP fingerprints and can't paint
> it. See `HOW_TO_CREATE_MY_OWN_STYLE.md` for the catalog/detection side.
> If you want to *render* in a style, you need a LoRA — train one here.

---

## Supported models

| Base | Flag | Status | LoRA format | Notes |
|------|------|--------|-------------|-------|
| **SD 1.5** (UNet) | `--base sd15` | ✅ **supported** | kohya | ungated, fastest to train + validate |
| **SDXL** (UNet) | `--base sdxl` | ✅ **supported** | kohya | ungated; dual-CLIP + add-time conditioning |
| **SD 3.5 Medium** (MMDiT) | `--base sd35` | ✅ **supported** | diffusers-PEFT | gated on HuggingFace; ~2.5 B-param transformer |

A LoRA is **bound to the base architecture** — an SD 1.5 LoRA only loads
on SD 1.5, not on SDXL / SD 3.5 / Flux. Train once per base you want to use.

The UNet bases (SD 1.5 / SDXL) write a **kohya** `.safetensors`
(`lora_unet_…lora_down/up.weight` + `.alpha`); SD 3.5 writes a
**diffusers-PEFT** `.safetensors` (`lora_A`/`lora_B`
/`alpha` keys), so it also loads in other diffusers-PEFT-aware tools.

---

## Hardware requirements

Training does full back-propagation through the base model, which is far
heavier than inference.

- **Apple Silicon + Metal**, **24 GB** unified memory (the reference
  target). Training runs in **mixed precision**: the frozen base is BF16
  (Metal-fast, half the memory) and the trainable LoRA is F32 (stable
  optimizer). This is what keeps SD 3.5 training inside 24 GB.
- **Resolution drives memory.** `--size 256` fits 24 GB. `--size 512`
  back-prop activations exceed 24 GB and get OOM-killed. Stay at 256
  unless you have more memory.
- **Speed.** ~**1.7 min/step** for SD 3.5 on an M-series Metal GPU
  (full-model backward is just heavy). 90 steps ≈ a couple of hours.
  Plan for it — and see the checkpointing note below so you can stop
  early.
- **Disk / download.** The base model downloads once (SD 3.5 Medium is
  gated — accept the licence on HuggingFace and authenticate first; the
  same model the corpus's `sd35.hjson` uses).
- CPU training technically works but is impractically slow — use Metal.

---

## Training corpus requirements

Put the images for **one** style in a single folder.

- **3 minimum, 5–15 ideal.** Fewer than 3 won't generalise; many more
  has diminishing returns for a style.
- **Same style, varied subjects.** Teach the *look*, not the *content* —
  e.g. for "watercolour", include harbours, forests, villages, all in
  the watercolour style. If every image is the same scene, the LoRA
  memorises that scene instead of the style.
- **Format:** JPEG or PNG. (HEIC/WebP/AVIF aren't decoded.)
- **Resolution:** 512 px+ on the short side is plenty — training
  downsamples to `--size` (256) anyway, but start from something crisp.
- **One style per folder.** To train several styles, run the trainer
  once per folder.

Example layout (the shipped watercolour corpus):

```
corpus/style/watercolour/
├── coast.jpeg
├── orchard.jpeg
├── quay.jpeg
├── snow-village.jpeg
└── … (9 watercolour illustrations)
```

---

## Quick start

**1. Train** (slow — run once):

```bash
plakat style train \
  --from-dir ./my_style_images \
  --base    sd35 \
  --trigger "wcstyle watercolour painting illustration" \
  --out     ./my_style.safetensors \
  --steps 90 --rank 16 --size 256
```

**2. Generate** with the trained LoRA (fast — reuse it as often as you
like; no retraining):

```bash
plakat generate "a fishing harbour with wooden boats, wcstyle watercolour painting illustration" \
  --model sd35-medium \
  --lora  ./my_style.safetensors \
  --steps 26 --size 768x768 --device metal --out ./out
```

Keep training and generation **separate** — training takes hours,
generation takes a minute, so you don't want to retrain every time you
render. (The corpus ships `style_train.sh` and `style_gen.sh` as exactly
this split.)

You should see `SD3 LoRA … → N/N targets merged` in the generation log
(191/191 for SD 3.5 Medium) — that confirms the LoRA actually applied.

---

## Training parameters

| Flag | Default | What it does |
|------|---------|--------------|
| `--from-dir` | — (required) | folder of style images (jpg/png) |
| `--base` | `sd35` | base model (Phase 1: `sd35` only) |
| `--trigger` | `"in this style"` | phrase woven into training; **put it in your prompts at inference** to invoke the style |
| `--out` | — (required) | output `.safetensors` path |
| `--steps` | `90` | training steps. More = stronger/cleaner style but linearly slower. 30–60 already shows the look; 90–150 refines it. |
| `--rank` | `16` | LoRA rank (capacity). 8 = lighter/faster, 32 = more capacity. 16 is a good default for a style. |
| `--size` | `256` | training resolution. **256 fits 24 GB**; raise only with more memory. |
| `--lr` | `1.5e-4` | learning rate. Lower = slower/safer, higher = faster but can overshoot. |

**The trigger phrase matters.** Pick something distinctive (a made-up
token like `wcstyle` plus a description). It's trained into the LoRA, and
including it in your generation prompt is how you "switch on" the style.

**Checkpointing.** The trainer saves `--out` every 30 steps, so a long
run is usable early — you can watch the first checkpoints render and stop
when the style is strong enough, without losing work.

---

## Tuning the result

- **Style too weak?** Train more steps, or raise the LoRA influence at
  inference with `--lora ./my_style.safetensors:1.3` (scale suffix), or
  raise `--rank`.
- **Style too strong / mangled subjects?** Fewer steps, or lower the
  inference scale (`…:0.7`).
- **OOM during training?** Lower `--size` (it's almost always the
  resolution), then `--rank`.
- **Too slow?** Fewer `--steps`; rely on the 30-step checkpoints to find
  the sweet spot. (Per-step time is fixed by the model size; it doesn't
  shrink with steps.)
- **Style doesn't show at inference?** Make sure the **trigger phrase is
  in your prompt**, and check the log says `N/N targets merged` (not
  `0/N`).

---

## How it works (brief)

- **Objective:** rectified-flow denoising (`x_σ = (1-σ)·x₀ + σ·ε`; the
  model predicts the velocity `ε - x₀`), the same objective SD 3.5 was
  trained with.
- **What's trained:** low-rank adapters on the MMDiT's **attention**
  projections (q/k/v/out on both the image and text streams) — the base
  weights stay frozen. ~191 targets for SD 3.5 Medium.
- **Memory plan:** encode the images→latents and the trigger→conditioning
  with the BF16 pipeline, drop it, then load the MMDiT for the training
  loop — so only one big model is resident at a time.
- **Output:** the adapters are written as diffusers-PEFT keys (fused
  `qkv` split into `to_q`/`to_k`/`to_v`), which plakat's `--lora` path
  resolves and merges into the MMDiT at load.

---

## Roadmap

- **0.46.0 (done):** SDXL and SD 1.5 (UNet) bases — a vendored, LoRA-wired
  SD UNet (separate from the inference path) trains the attention with a
  DDPM-epsilon objective; SDXL adds the dual-CLIP + add-time conditioning.
- **Next:** higher-resolution training as memory allows; gradient
  checkpointing; more base models (PixArt / Cascade).

## See also

- `STYLES_TUTORIAL.md` — using the bundled style catalog
- `HOW_TO_CREATE_MY_OWN_STYLE.md` — the detection/catalog side
- `corpus/style_train.sh` / `corpus/style_gen.sh` — the worked watercolour
  example (9 references → LoRA → fresh watercolour renders)
