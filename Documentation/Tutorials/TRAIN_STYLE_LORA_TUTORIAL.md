# Train your own style LoRA (tutorial)

**v0.45.** You have a folder of art you love and you want plakat to
*paint* in that style — not just recognise it. This tutorial walks you
from a folder of images to a trained LoRA you can drop onto any
generation with `--lora`.

> **Creation vs detection.** The earlier
> [`HOW_TO_CREATE_MY_OWN_STYLE.md`](HOW_TO_CREATE_MY_OWN_STYLE.md) builds
> a *catalog* that **detects** a style from CLIP fingerprints — useful for
> routing, but it can't render the style. This tutorial is the other half:
> it **trains a LoRA** that actually changes how the model paints. If your
> goal is "make my pictures look like these," you're in the right place.
>
> For the exhaustive flag-by-flag reference, see
> [`Documentation/TRAIN_CUSTOM_LORA.md`](../TRAIN_CUSTOM_LORA.md).

## What you'll do

1. Collect a small corpus of style images.
2. Run `plakat style train` to learn the style into a LoRA.
3. Generate fresh images in that style with `--lora`.
4. Tune the strength and decide when it's "done."

We'll use the shipped watercolour example throughout — nine watercolour
illustrations in `corpus/style/watercolour/`.

## Before you start

- Finish [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md) so you've rendered
  at least one image.
- **Phase 1 trains on SD 3.5 Medium**, which is **gated** — accept the
  licence on HuggingFace and authenticate (`huggingface-cli login`)
  first. (SDXL / SD 1.5 land in a later release.)
- **Hardware:** Apple Silicon + Metal, **24 GB** unified memory. Training
  is heavy — budget a couple of hours (see step 2). SDXL/SD1.5 + bigger
  GPUs come later.

---

## Step 1 — Collect your corpus

Put the images for **one** style in a folder. The golden rule: teach the
*look*, not the *content*.

- **3 minimum, 5–15 ideal.**
- **Same style, different subjects.** The watercolour set has harbours,
  forests, villages, snow — all watercolour, all different scenes. That's
  what lets the LoRA learn "watercolour" instead of "this one painting."
- JPEG or PNG, 512 px+ on the short side.

```
corpus/style/watercolour/
├── coast.jpeg      ├── passage.jpeg     ├── snow-village.jpeg
├── figures.jpeg    ├── quay.jpeg        ├── snow-wolves.jpeg
├── orchard.jpeg    ├── rain-dock.jpeg   └── traveller.jpeg
```

If every image were the same scene, the LoRA would memorise that scene.
Variety in subject + consistency in style is the whole game.

---

## Step 2 — Train

```bash
plakat style train \
  --from-dir corpus/style/watercolour \
  --base    sd35 \
  --trigger "wcstyle watercolour painting illustration" \
  --out     corpus/style/watercolour.safetensors \
  --steps 90 --rank 16 --size 256
```

What the flags mean (the ones you'll actually touch):

- `--trigger` — a phrase trained into the LoRA. Pick something
  distinctive — a made-up token (`wcstyle`) plus a description. You'll put
  this phrase in your prompts later to switch the style on.
- `--steps 90` — more steps = stronger, cleaner style, linearly slower.
  30–60 already shows the look; 90–150 refines it.
- `--rank 16` — the adapter's capacity. 16 is a good default for a style.
- `--size 256` — training resolution. **256 fits 24 GB; 512 will OOM.**

You'll see it work through three phases:

```
style-train: encoding 9 image(s) + caption "wcstyle watercolour painting illustration"
style-train: loading MMDiT (F32) for training
style-train: 95 trainable attention adapters (rank 16)
style-train: step 1/90 loss 0.68750
style-train: step 11/90 loss 0.64062
style-train: step 21/90 loss 0.45312
style-train: checkpoint @ step 30 → corpus/style/watercolour.safetensors
…
```

**This is slow** — roughly 1.7 min/step, because every step
back-propagates through the whole 2.5-billion-parameter model. 90 steps
is a couple of hours. Two things make that bearable:

- **The loss falls fast** (0.69 → 0.45 by step 21 above) — you can see it
  learning.
- **It checkpoints every 30 steps**, overwriting `--out`. So from step 30
  on you have a usable LoRA *even if you stop early* — render with it,
  and if the style's strong enough, `Ctrl-C` and move on.
- **Resume an interrupted run** with `--resume` (all bases — sd15/sdxl/sd35):
  point it at a numbered `…-step<N>.safetensors` checkpoint and it continues from
  that step up to `--steps`, so raise `--steps` to train further:

  ```bash
  plakat style train --base sd35 --in ./my-style \
    --resume my-style-step60.safetensors --steps 120   # 60 more steps
  ```

  Numbered checkpoints are written by default (unset `PLAKAT_TRAIN_SINGLE_FILE`).

> **Why it's separate from generation:** training takes hours, rendering
> takes a minute. You train **once**, then reuse the LoRA forever. The
> corpus ships this as two scripts — `style_train.sh` (this step) and
> `style_gen.sh` (the next) — so you never retrain just to render.

---

## Step 3 — Generate in your style

Now the fun part — and it's fast. Load the LoRA with `--lora` and put your
trigger phrase in the prompt:

```bash
plakat generate "a fishing harbour with wooden boats, wcstyle watercolour painting illustration" \
  --model sd35-medium \
  --lora  corpus/style/watercolour.safetensors \
  --steps 26 --size 768x768 --seed 42 --device metal --out ./out
```

Check the log line:

```
SD3 LoRA corpus/style/watercolour.safetensors → 191/191 targets merged (scale 1.00)
```

`191/191` means every adapter applied. If you see `0/191`, the LoRA
didn't take — re-check you're on `sd35-medium`.

Try it on subjects that were **never** in your corpus — a snow village, an
autumn orchard, a city street. A good style LoRA generalises: the
watercolour look carries onto all of them. (That's the corpus proof in
`corpus/images/style/` — three fresh subjects, all unmistakably
watercolour.)

Compare against the same prompt+seed **without** `--lora`: the base model
renders a photograph; the LoRA turns it into a watercolour. Same scene,
different medium — that's the LoRA doing its job.

---

## Train a SUBJECT instead of a style (DreamBooth)

The same command learns a **subject** ("my dog", a specific person/object) rather
than a style — point `--trigger` at an *instance prompt* with a rare token and add
a **class** set for prior preservation (a few generic examples of the category, so
the new token doesn't overfit or drag the whole class toward your photos):

```bash
plakat style train --base sd15 \
  --from-dir ./my-dog \
  --trigger "a photo of sks dog" \
  --class-dir ./generic-dogs --class-prompt "a photo of a dog" \
  --prior-weight 1.0 --steps 800 --rank 16
# then generate:  plakat generate "a photo of sks dog astronaut on the moon" \
#                   --model sd15 --lora my-dog.safetensors
```

`--class-dir` makes it a subject LoRA (DreamBooth): each step trains your subject
**and** a class image, so the model keeps its general "dog" while binding `sks` to
your dog. Omit `--class-dir` to skip prior preservation. sd15 / sdxl (sd35's
separate trainer doesn't support prior preservation yet). `--resume` works here too.

---

## Step 4 — Tune it

- **Style too weak?** Dial the influence up at render time with a scale
  suffix: `--lora corpus/style/watercolour.safetensors:1.3`. Or train more
  steps.
- **Style too strong / subjects mangled?** Dial it down:
  `…watercolour.safetensors:0.7`. Or train fewer steps.
- **Style not showing at all?** Make sure the **trigger phrase is in your
  prompt**, and that the log says `N/N targets merged`.
- **OOM during training?** It's almost always `--size` — keep it at 256.

The scale suffix is the quickest lever — you can train once and find the
right strength entirely at generation time.

---

## Where to next

- Flag-by-flag reference + the rectified-flow details →
  [`Documentation/TRAIN_CUSTOM_LORA.md`](../TRAIN_CUSTOM_LORA.md)
- The detection/catalog side (recognise a style) →
  [`HOW_TO_CREATE_MY_OWN_STYLE.md`](HOW_TO_CREATE_MY_OWN_STYLE.md)
- Using LoRAs in batch scenarios → [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md)
- The worked watercolour example → `corpus/style_train.sh` +
  `corpus/style_gen.sh`
