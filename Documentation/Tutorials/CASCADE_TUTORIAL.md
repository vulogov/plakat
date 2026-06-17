# Stable Cascade tutorial

Stable Cascade (Stability AI's Würstchen v3) is plakat's most
**memory-efficient** high-quality model. Where SDXL and Flux denoise
in a large latent space, Cascade does the heavy semantic work in a
tiny **24×24×16** latent — so the expensive stage is cheap, and you get
crisp 1024² images on modest hardware.

This tutorial walks the full Cascade feature set: the three-stage
pipeline and its two step budgets, prior vs decoder guidance, LoRA /
DoRA, **image variation** and **faithful img2img** (v0.42), ControlNet,
and driving Cascade from Bund scripts.

For one-line flag reference see [`GENERATE.md`](../GENERATE.md) and
[`IMG2IMG.md`](../IMG2IMG.md). This tutorial focuses on the *why*.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md).
- ~20 GB free disk for the weights (text encoder + 3 stages), and a
  GPU / Apple-Silicon machine with ≥16 GB unified memory.
- No HuggingFace token needed — the `stabilityai/stable-cascade` and
  `stabilityai/stable-cascade-prior` repos are ungated.

## 1. How Stable Cascade works

Cascade is **three** models chained, not one:

```
text ──CLIP-G──┐
               ▼
        Stage C (prior)        24×24×16 latent   ← the heavy, semantic stage
               │  (its output conditions Stage B)
               ▼
        Stage B (decoder)      16×128×128 latent  ← refines into VAE space
               │
               ▼
        Stage A (Paella VQGAN) ─decode→ 1024×1024 RGB
```

The key idea: **Stage C** — where prompt-following lives — operates on a
24×24×16 latent that's ~42× smaller per axis than the image. That's why
Cascade is fast and light despite Stage C being a 3.6 B-param prior.

Because the prior latent is fixed at 24×24×16, **Cascade output is
square** (default 1024×1024). Non-square `--size` is rejected.

## 2. Your first Cascade image

```bash
plakat generate "a serene mountain lake at golden sunrise, photorealistic" \
    --model stable-cascade --size 1024x1024 --seed 42
```

First run downloads the weights; later runs use the cache.

### The two step budgets

Cascade has **two** denoise loops, so it exposes two step counts:

```bash
plakat generate "a misty forest at dawn" --model stable-cascade \
    --stage-c-steps 20 --stage-b-steps 10 --guidance 4.0 --seed 42
```

- `--stage-c-steps` (default 20) — the semantic prior. Most of the
  quality lives here; raising it helps complex, multi-subject prompts.
- `--stage-b-steps` (default 10) — the decoder refine. Diminishing
  returns past ~12.
- If you pass only the unified `--steps N`, plakat splits it 2/3 to
  Stage C and 1/3 to Stage B (the upstream recommendation).

`--guidance` (~4.0 is the sweet spot) is the **prior** CFG scale.

## 3. Prior vs decoder guidance (v0.42)

Cascade actually has *two* classifier-free-guidance scales — one per
stage. `--guidance` drives Stage C (the prior). The new
**`--decoder-guidance`** (v0.42, default `1.1`) drives Stage B:

```bash
plakat generate "a baroque cathedral interior, volumetric light" \
    --model stable-cascade --guidance 4.0 --decoder-guidance 1.1 --seed 42
```

The decoder default of 1.1 is intentionally mild — Stage B's job is to
*refine*, not to re-impose the prompt. Push it slightly higher (1.3–1.5)
for a touch more contrast/detail; values much above that tend to
over-sharpen. Leave it at 1.1 unless you have a reason.

## 4. LoRA and DoRA (v0.42)

Stable Cascade LoRAs merge into Stage C (and optionally Stage B) at load
time. plakat handles both naming conventions and both decompositions:

```bash
plakat generate "a girl with long hair in a flower field, anime style" \
    --model stable-cascade \
    --lora ~/loras/cascade_anime.safetensors:1.0 --seed 42
```

- `--lora PATH:SCALE` — repeatable to stack multiple.
- **kohya / sd-scripts format** (`lora_prior_unet_…`) — the format most
  community Cascade LoRAs ship in — and the **diffusers / PEFT** dotted
  format both work.
- **DoRA** (Weight-Decomposed LoRA) is supported: plakat auto-detects
  the magnitude axis, so kohya-trained and PEFT-trained DoRAs both fuse
  correctly. (Getting this right was the v0.42 phase-1 campaign — see
  [`RELEASE_HISTORY.md`](../RELEASE_HISTORY.md).)
- `--lora-scale` globally multiplies every LoRA's per-spec scale.

Currently only attention targets are merged (community Cascade LoRAs
rarely touch feed-forward / conv layers); LyCORIS/LoHa/LoKr variants are
not yet supported.

## 5. Image variation (v0.42)

**Image variation** conditions Stage C on a *reference image's* CLIP
ViT-L/14 embedding (the unCLIP idea) — the output shares the reference's
semantics (subject, palette, mood) while re-composing it:

```bash
plakat generate "" \
    --model stable-cascade --image-variation ref.png --seed 7
```

The text prompt is optional and composes with the image:

```bash
# Vary on the reference, but steer toward winter
plakat generate "in deep winter snow" \
    --model stable-cascade --image-variation ref.png --seed 7
```

With an empty or short prompt the image embedding dominates — leave the
prompt empty for the purest "variations of this picture." The image
encoder (`image_encoder/` from the prior repo, ~1.2 GB) loads only when
`--image-variation` is used, so plain t2i never pays for it.

## 6. Faithful img2img (v0.42)

Plain Cascade img2img seeds Stage B from your init image's VAE latent —
it preserves *low-level structure* but Stage C still runs purely on
text, so content can drift. **`--faithful`** additionally conditions
Stage C on the init image's CLIP embedding, pulling the *content* toward
the init:

```bash
# Structure only
plakat img2img cottage.png --prompt "a cottage in deep winter snow" \
    --model stable-cascade --strength 0.6 --seed 42

# Structure + semantics (subject held closer to the init)
plakat img2img cottage.png --prompt "a cottage in deep winter snow" \
    --model stable-cascade --strength 0.6 --faithful --seed 42
```

Reach for `--faithful` when plain img2img wanders off-subject at higher
strengths.

## 7. ControlNet

Cascade ships a **canny** ControlNet that conditions Stage C on an edge
map. The weights auto-resolve from the model repo — you don't pass a
weights path. Supply the conditioning two ways:

> **Why Stage C only?** This is Stable Cascade's design, not a plakat
> limitation. The decoupled cascade applies ControlNet (and LoRA) to the
> Stage C prior *alone* — "the stages B and A models do not need to be
> updated" (Stability AI). Stage B is a fixed semantic-compressor /
> super-resolver that preserves Stage C's structure through the decode, so
> the edge control already survives to the final image. There is no
> Stage-B ControlNet to add.

```bash
# Auto-annotate any photo to canny edges
plakat generate "a cozy cottage in an autumn forest, photorealistic" \
    --model stable-cascade --control canny --control-from house.png \
    --control-strength 1.0 --seed 42

# Or hand it a pre-rendered edge map
plakat generate "a cozy cottage in an autumn forest" \
    --model stable-cascade --control canny --control-image edges.png
```

- `--control-strength` (default 1.0) scales the residual.
- `--control-start` / `--control-end` gate the active timestep window
  (fractions of the schedule) — useful to let structure set early then
  free the late steps for detail.

It composes with img2img (`plakat img2img … --control canny
--control-from …`).

## 8. Cascade in Bund scripts (v0.42)

Scripting drives Cascade through `plakat.cascade`, and it now honours the
shared `plakat.controlnet.*` words:

```bund
// verify_phase4_cascade_cn.bund
"canny" "house.png" plakat.controlnet.annotate   // push a canny CN

"stable-cascade" plakat.load

"20"  "stage_c_steps" plakat.config.set
"10"  "stage_b_steps" plakat.config.set
"4.0" "guidance"      plakat.config.set
"42"  "seed"          plakat.config.set

"a cozy cottage in an autumn forest, golden leaves, photorealistic" plakat.cascade
"/tmp/cottage.png" plakat.save
```

Run it with `plakat run script.bund`. Push the ControlNet spec *before*
`plakat.load` so the pipeline warms with its CN in a single pass. LoRA
stacks (`plakat.lora.add`) and the two step budgets
(`stage_c_steps` / `stage_b_steps`, or a unified `steps`) all work the
same as the CLI. See [`SCRIPTING_TUTORIAL.md`](SCRIPTING_TUTORIAL.md) for
the Bund basics.

## 9. Memory and speed notes

- Cascade's working set on 1024² is dominated by the Stage C prior; the
  full pipeline runs comfortably in ~16 GB unified memory.
- The default `20 + 10` step split renders in a fraction of the time a
  same-resolution Flux generation takes — Cascade's tiny prior latent is
  the whole point.
- All output is square. Want a different aspect? Generate square, then
  [`OUTPAINT_TUTORIAL.md`](OUTPAINT_TUTORIAL.md) to extend the canvas.

## Where to go next

- [`CONTROLNET_TUTORIAL.md`](CONTROLNET_TUTORIAL.md) — the general
  ControlNet workflow (depth + canny on SD / SDXL).
- [`SCRIPTING_TUTORIAL.md`](SCRIPTING_TUTORIAL.md) — Bund scripting from
  scratch.
- [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md) — batch Cascade
  generation via HJSON.
