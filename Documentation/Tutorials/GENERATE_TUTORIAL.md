# Generate your first image — a friendly tutorial

This walks you from "I have plakat installed" to "I'm running batch
generations with custom prompts, scenes, and weather variations." No
prior text-to-image experience assumed.

## What you'll learn

- How to turn a sentence into an image with `plakat generate`
- What the most common flags mean (and which ones to ignore at first)
- How to make the same image twice (seeds)
- How to switch from one-off commands to a **scenario file** that
  runs many variants in one go
- Where the output files land and how plakat names them

## Before you start

- You have `plakat` installed and runnable. If `plakat --help` prints
  the help screen, you're set. If not, see the main `README.md`.
- You have ~6 GB of disk space free. The first time plakat runs it
  downloads a model from HuggingFace (~4 GB for SD 1.5) and caches
  it. Future runs reuse the cache.
- You have a few minutes of patience for the first model download.

The tutorial uses **SD 1.5** (Stable Diffusion 1.5) throughout because
it's small, fast, and free. Plakat supports larger models (SDXL,
Flux), but those add download size and runtime cost without changing
the workflow you're learning here.

---

## 1. Your first image

The simplest possible invocation:

```bash
plakat generate "a fox sitting in tall grass"
```

That's the whole command. Plakat will:

1. Download the SD 1.5 base model from HuggingFace (only on first run).
2. Read your prompt.
3. Generate one image.
4. Save it to `./out/plakat-<seed>.png`.

When it's done, you'll see something like:

```
→ ./out/plakat-1234567890.png
```

Open that file. You should see a fox in tall grass. The exact
appearance depends on randomness, so yours will look different from
anyone else's running the same command.

### What just happened?

Text-to-image models like SD 1.5 don't "find" pictures of foxes; they
*generate* one pixel pattern at a time, starting from random noise
and progressively refining it toward something that matches your
prompt. The model has learned, during training, how words like "fox"
and "grass" correspond to visual patterns. The first time you run
this, plakat downloads that trained model. Subsequent runs reuse it.

---

## 2. Understanding the flags

Run `plakat generate --help` to see the full list. Most flags you can
ignore until you have a specific reason to use them. The five that
matter most when you're learning:

| Flag | What it does | Default |
|---|---|---|
| `--steps N` | How many refinement passes the model makes | 28 |
| `--guidance F` | How strictly to follow the prompt (lower = more creative, higher = more literal) | 7.5 |
| `--seed N` | Fix the randomness so you can re-run the same image | random |
| `--size WxH` | Output dimensions | `512x512` for SD 1.5 |
| `--out DIR` | Where to save the result | `./out` |

### Try this:

```bash
# Make a small image fast — good for testing
plakat generate "a fox in tall grass" --size 256x256 --steps 12

# Make a higher-quality image (slower)
plakat generate "a fox in tall grass" --steps 40 --guidance 8.0
```

Higher `--steps` produce more refined images but take longer. There
are diminishing returns past ~30 steps for SD 1.5; 12-20 is fine for
prototyping, 30-40 for "finished" work.

Higher `--guidance` makes the model adhere more strictly to your
prompt but can produce stiff, over-saturated output. The default 7.5
is a good middle ground.

---

## 3. Reproducibility — the seed

Each run picks a random number (the "seed") that drives the initial
noise the model refines. Two runs with different seeds produce
different images, even with the same prompt. Two runs with the **same
seed** produce the same image (bit-for-bit, assuming everything else
matches).

The seed appears in the filename: `plakat-1234567890.png` means
seed=1234567890.

If you make an image you like and want to iterate on it:

```bash
# Found a good fox at seed 42
plakat generate "a fox in tall grass" --seed 42

# Now try the same composition with different lighting
plakat generate "a fox in tall grass at sunset" --seed 42
```

Same seed + slightly different prompt is how you explore variations
without rerolling the whole composition.

---

## 4. Negative prompts: what NOT to draw

You can tell the model what to avoid:

```bash
plakat generate "a fox in tall grass" \
    --negative "blurry, deformed, low quality, watermark"
```

This won't always remove every blurry/deformed result, but it nudges
the model away from common failure modes. Plakat doesn't add a
negative prompt by default; you decide what to discourage.

---

## 5. Generating many images at once

Use `-n N` or `--count N`:

```bash
plakat generate "a fox in tall grass" --count 4
```

Plakat produces 4 images with seeds `seed`, `seed+1`, `seed+2`,
`seed+3`. Filenames are `plakat-<seed>.png` for each.

---

## 6. Picking a different model

The default model is SD 1.5. Plakat supports a few short aliases plus
arbitrary HuggingFace repo ids:

| `--model` value | What it is |
|---|---|
| `sd15` (default) | Stable Diffusion 1.5 — small, fast |
| `sd21` | Stable Diffusion 2.1 |
| `sdxl` | Stable Diffusion XL — larger, slower, generally higher quality |
| `sdxl-turbo` | SDXL trained for very few steps (~4) |
| `flux-schnell` | Flux Schnell — different architecture, fast |
| `flux-dev` | Flux Dev — higher-quality Flux variant |
| `<org>/<repo>` | Any HuggingFace text-to-image repo by id |

```bash
plakat generate "a fox" --model sdxl --size 1024x1024
plakat generate "a fox" --model sdxl-turbo --steps 4 --guidance 0
```

Each new model downloads ~4-12 GB on first use. SDXL-Turbo wants
`--guidance 0` and very few steps (it's a different kind of model).

For this tutorial, stick with `sd15` unless you have a reason to
switch.

---

## 7. Adding LoRAs — small style/character modifiers

A **LoRA** is a small "add-on" that modifies the base model's
behavior — typically to teach it a specific artistic style, character,
or visual concept. LoRAs are tiny (~10-200 MB) compared to a full
model.

You can apply one or more to a generation:

```bash
plakat generate "a fox in tall grass" \
    --lora "Arczisan/ink-watercolor:0.8"
```

The format is `<repo>:<scale>`. `0.8` means "apply this LoRA at 80%
strength." Higher = more of its influence; lower = subtler.

You can repeat `--lora` to stack multiple. They compose multiplicatively
through `--lora-scale` (a global multiplier, default 1.0).

If you don't know which LoRA to use, skip them. The base model alone
produces solid output.

(See `STYLES_TUTORIAL.md` for a higher-level way of applying styles
without picking individual LoRAs.)

---

## 8. From one-off commands to scenarios

When you find yourself running 5+ similar commands — say, the same
prompt across different scenes, weather conditions, or seeds —
**scenarios** are the right tool.

A scenario is an HJSON file (a JSON variant with comments and relaxed
syntax) that describes a batch of related generations. Plakat reads
it, builds the cartesian product of options, and runs the whole
batch.

### Your first scenario

Create a file `my_first_scenario.hjson`:

```hjson
{
    # Global settings for the whole batch.
    model: sd15
    base: 512
    steps: 28
    count: 1            # how many images per task
    out: ./out

    # The "enhancer" is a language model that polishes your prompt.
    # plakat requires this field; if you don't want enhancement, leave
    # it as `deepseek` and run with --dry-run for now.
    enhancer: deepseek

    # Scenes are reusable prompt fragments naming a place.
    scene:
    [
        {
            name: forest
            prompt: "an ancient mossy forest with shafts of light through the canopy"
        }
        {
            name: meadow
            prompt: "a wide open grass meadow with wildflowers"
        }
    ]

    # Weather is another reusable fragment.
    weather:
    [
        {
            name: dawn
            prompt: "soft early dawn light, golden tones, low mist"
        }
        {
            name: rainy
            prompt: "heavy summer rain, dark grey clouds, wet ground"
        }
    ]

    # Tasks combine scene + weather + a task-specific prompt.
    tasks:
    [
        {
            name: fox_forest_dawn
            scene: forest
            weather: dawn
            prompt: "a fox at the edge of the trees"
        }
        {
            name: fox_meadow_rain
            scene: meadow
            weather: rainy
            prompt: "a fox sheltering under a fallen log"
        }
    ]
}
```

Run it in dry-run mode first to see what prompts it would build:

```bash
# Dry-run lets you preview without calling the enhancer API or
# generating images. Useful for catching typos and previewing prompts.
DEEPSEEK_API_KEY=placeholder plakat scenario my_first_scenario.hjson --dry-run
```

You'll see plakat assemble:

```
▶ [1/2] fox_forest_dawn (scene=forest, weather=dawn)
  pre-enhance: an ancient mossy forest with shafts of light through the canopy,
               soft early dawn light, golden tones, low mist,
               a fox at the edge of the trees
  ...
```

Each task's pre-enhance prompt is the concatenation of:

```
prompt-header + scene + weather + task-prompt + prompt-footer
```

For an actual run (with the enhancer + real generation), you need a
DeepSeek API key (or Gemini — see `Documentation/GENERATE.md` for the
full enhancer reference):

```bash
export DEEPSEEK_API_KEY="sk-..."
plakat scenario my_first_scenario.hjson
```

This generates 2 images (one per task).

### What's the enhancer for?

The enhancer is a small language model (DeepSeek or Gemini) that
takes your pre-enhance prompt and rewrites it in language more
suitable for text-to-image models — adding lighting cues, style
descriptors, mood tokens, etc. Think of it as a polish step.

If you don't want enhancement, you can bypass it by writing one-off
`plakat generate` commands instead of using scenarios. Scenarios
*require* an enhancer field today.

---

## 9. Scaling up — the cartesian product

The real power of scenarios is when you want to cover *all combinations*:

```hjson
{
    # ... same global settings ...

    scene:
    [
        { name: forest, prompt: "..." }
        { name: meadow, prompt: "..." }
        { name: harbor, prompt: "..." }
    ]
    weather:
    [
        { name: dawn,   prompt: "..." }
        { name: rainy,  prompt: "..." }
        { name: snowy,  prompt: "..." }
    ]

    # 3 scenes × 3 weather × 2 count = 18 images in one run.
    count: 2
    tasks:
    [
        { name: forest_dawn,   scene: forest, weather: dawn,  prompt: "a fox" }
        { name: forest_rainy,  scene: forest, weather: rainy, prompt: "a fox" }
        { name: forest_snowy,  scene: forest, weather: snowy, prompt: "a fox" }
        { name: meadow_dawn,   scene: meadow, weather: dawn,  prompt: "a fox" }
        { name: meadow_rainy,  scene: meadow, weather: rainy, prompt: "a fox" }
        { name: meadow_snowy,  scene: meadow, weather: snowy, prompt: "a fox" }
        # ... and so on for harbor ...
    ]
}
```

You explicitly list each task you want — plakat doesn't auto-cross
scenes × weather. This is deliberate: you might want only some
combinations.

---

## 10. Per-task overrides

A task can override almost any global field for itself:

```hjson
tasks:
[
    {
        name: high_quality_one
        scene: forest
        weather: dawn
        prompt: "a fox"
        steps: 50              # this task uses 50 steps instead of 28
        guidance: 8.5          # and stronger guidance
        size: 768x768          # and a larger output
        seed: 42               # and a fixed seed
    }
]
```

Fields that *cannot* be per-task: `model`, `device`, `loras`,
`lora-scale`, `enhancer`, `upscale`. These force the shared pipeline
to reload, so they're locked at scenario start.

---

## 11. Output naming

Plakat writes outputs into `<out>/<task-name>/plakat-<seed>.png` (or
just `<out>/plakat-<seed>.png` if there's only one task).

Each task gets its own subdirectory so a batch run produces a tidy
tree:

```
out/
├── fox_forest_dawn/
│   ├── plakat-1000.png
│   └── plakat-1001.png   # second image (because count: 2)
├── fox_forest_rainy/
│   ├── plakat-1002.png
│   └── plakat-1003.png
└── ...
```

The base seed comes from the scenario's `seed:` field (default 0).
Task `idx` uses seeds `seed + idx*count` through `seed + idx*count +
count - 1`. With seed=1000 and count=2, task 0 gets 1000-1001, task 1
gets 1002-1003, and so on.

---

## 12. Wildcards (v0.16+)

Two ways to add randomness to a prompt without re-typing it:

**Inline alternation** — pick one of N options at random:

```bash
plakat generate "a {red|blue|green} {fox|cat|owl}" \
    --model sd15 --count 4 --seed 42
```

Each run with `--seed 42` reproduces the same picks. With no
`--seed`, OS entropy seeds the wildcard RNG (different output each
run).

**File wildcards** — pick a random line from a text file:

```bash
mkdir -p wildcards
echo -e "red\nblue\nemerald\ngolden" > wildcards/colors.txt
plakat generate "a __colors__ fox" \
    --wildcard-dir ./wildcards --model sd15
```

Files live under `<dir>/<name>.txt`. Comments (`#`) and blank lines
are skipped. Names accept letters, digits, `-`, and `_` (so
`__warm-colors__` and `__warm_colors__` both work).

Wildcards compose with the prompt enhancer (`--enhance`) — expansion
runs first, then the enhancer sees the concrete prompt.

## 13. CLIP-skip (v0.16+, SD 1.5 / SD 2.1)

The Auto1111 / NovelAI community default for SD 1.5 anime checkpoints
(Anything-v3, AnyLoRA, ...) is to read the **penultimate** CLIP
hidden state rather than the last. plakat exposes this as
`--clip-skip N`:

```bash
# Default — last layer (diffusers default, byte-identical to pre-v0.16):
plakat generate "..." --model sd15

# Penultimate — community-standard SD 1.5 anime path:
plakat generate "..." --model sd15 --clip-skip 2
```

SDXL ignores `--clip-skip` (its dual-encoder path already uses
penultimate by training default — plakat logs a warning if you pass
`--clip-skip > 1` on SDXL). Flux / SD3 don't use this flag at all
(T5 + CLIP-pooled architecture).

## 14. ADetailer — face refinement (v0.16+, SD-family)

SD/SDXL often produce lo-fi faces at non-face working resolutions
(small face in a big canvas — the model only had ~64² of latent for
the actual face). ADetailer is a post-pass that fixes this:

1. Detect each face with SCRFD.
2. Crop an expanded bounding box around the face (default +25% on
   each side).
3. Run img2img on the crop at a higher working resolution.
4. Feather-composite the refined crop back onto the original.

Enable it with `--adetailer`:

```bash
# SD 1.5 portrait — default 0.4 strength, 25% bbox padding, 512²
# working resolution per face.
plakat generate "a woman walking through a forest, full body shot" \
    --model sd15 --size 768x1024 --adetailer

# SDXL — 1024² working resolution per face matches SDXL native.
plakat generate "..." --model sdxl --size 1280x1920 \
    --adetailer --adetailer-size 1024
```

**Required setup**: ADetailer needs SCRFD weights. Same env vars
the FaceID portrait flow uses — either a local path:

```bash
export PLAKAT_SCRFD_WEIGHTS=/path/to/scrfd_10g_bnkps.safetensors
```

…or an HF spec:

```bash
export PLAKAT_SCRFD_HF="immich-app/SCRFD#scrfd_10g_bnkps.safetensors"
```

Without one of these, `--adetailer` bails loud.

**Knobs**:

| Flag | Default | Effect |
|---|---|---|
| `--adetailer-strength F` | `0.4` | img2img strength on each face crop. Lower = preserve identity, higher = re-imagine. `0.6+` will change the face. |
| `--adetailer-padding F` | `0.25` | Bbox expansion per side. More = better blending, less res per face. |
| `--adetailer-feather F` | `0.25` | Outer fraction of bbox that fades to 0. Softer seam vs sharper detail near edge. |
| `--adetailer-confidence F` | `0.5` | SCRFD score threshold. Faces below skipped. |
| `--adetailer-size N` | `512` | Working res for the face img2img (square, snapped /8). |
| `--adetailer-prompt STR` | (generic) | Override the face pass prompt. Default: "detailed face, sharp focus, high quality". |

**Restrictions**:
- SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo only. Flux / SD3 bail loud
  (their portrait paths aren't shipped yet).
- Runs once per output image — `--count 4` triggers four ADetailer
  passes total. Each face within an image runs separately.
- Composes with `--lora`, `--scheduler`, `--seed`. The face pass
  reuses the t2i SdCore so there's no extra model load.

## 15. Common issues

**Image takes forever / runs out of memory.**
SD 1.5 needs ~5 GB resident at 512² (its training resolution). SDXL
at 1024² needs ~10 GB. On Apple Silicon, plakat auto-detects Metal.
On a memory-tight machine, drop to a smaller model (`--model sd15`),
SD 1.5's native size (`--size 512x512`), and fewer steps
(`--steps 20`). For the full per-chip + per-model breakdown, see
[`APPLE_REQUIREMENTS.md`](../APPLE_REQUIREMENTS.md).

**Output is blurry / over-smooth.**
Try increasing `--guidance` (e.g., 9.0) and `--steps` (e.g., 40).
Also try adding `--negative "blurry, low quality"`.

**Output is way too literal / boring.**
Drop `--guidance` to 5.5 or 6.0. The model gets more creative
freedom.

**SDXL output looks washed out.**
SDXL likes higher resolutions. Try `--size 1024x1024` rather than
512x512.

**The enhancer fails with `DEEPSEEK_API_KEY not set`.**
Either export the key, switch to Gemini (`enhancer: gemini` +
`GEMINI_API_KEY`), or run with `--dry-run` to test without it.

**I want the same image every time for sharing/CI.**
Always pass `--seed`. Same seed + same prompt + same model + same
flags → identical bytes.

---

## Where to next

- **Make portraits with identity preservation** → `PORTRAIT_TUTORIAL.md`
- **Apply art styles to your generations** → `STYLES_TUTORIAL.md`
- **Build your own style catalog** → `HOW_TO_CREATE_MY_OWN_STYLE.md`
- **Full reference of every CLI flag** → `Documentation/GENERATE.md`
- **More on scenarios, upscaling, refiner pipelines** →
  `Documentation/GENERATE.md`
