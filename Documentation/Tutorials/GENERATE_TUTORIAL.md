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
| `pixart` | PixArt-Σ — DiT + T5, 1024² |
| `sana` | Sana 1.6B — linear-attention DiT + Gemma-2 text encoder, 1024², strong at long prose |
| `sana-1.5` | Sana-1.5 1.6B — the `qk_norm` refinement of the Sana DiT (else identical) |
| `<org>/<repo>` | Any HuggingFace text-to-image repo by id |

```bash
plakat generate "a fox" --model sdxl --size 1024x1024
plakat generate "a fox" --model sdxl-turbo --steps 4 --guidance 0
plakat generate "a fox in a misty forest, watercolor" --model sana --steps 20 --guidance 4.5
```

Each new model downloads ~4-12 GB on first use. SDXL-Turbo wants
`--guidance 0` and very few steps (it's a different kind of model).

**Sana** is worth a special mention: its Gemma-2-2B text encoder (plus a built-in
"complex human instruction" that enriches your prompt) makes it strong at **long,
detailed scene descriptions**. It generates at 1024² with a 32× deep-compression
autoencoder, so it's memory-light for its quality. Defaults: 20 steps, guidance 4.5,
the **DPM++ 2M flow** sampler (`--scheduler euler` for the lighter FlowMatchEuler);
output size must be a multiple of 32. On GPU the DiT + encoder run in BF16 (the encoder
is freed after encoding, so peak residency stays modest); on CPU everything is F32.

- **Variants:** `sana-600m` (smaller/faster 0.6B DiT), `sana-512` (512² — use `--size 512x512`),
  `sana-2k` (2048² — memory-heavy), and `sana-1.5` (the `qk_norm` checkpoint). Same DC-AE + Gemma-2;
  the DiT config is read per-model.
- **img2img:** `plakat img2img <image> --prompt "…" --model sana --strength 0.6` — the init is
  DC-AE-encoded and the flow loop starts from a strength-noised latent.
- **Inpaint:** add `--mask mask.png` (white = repaint, black = preserve; `--mask-feather`,
  `--mask-invert` as elsewhere). RePaint-style: the mask is pooled to the DC-AE **32× latent grid** —
  so the boundary is coarse (16×16 for a 512² image) — and after each step the preserve region snaps
  back onto the init while the masked region re-paints. Inpaint strength defaults to 1.0.

  ```bash
  plakat img2img town.png --prompt "a full moon in a green sky, watercolor" \
      --model sana --mask sky.png --mask-feather 8
  ```

For this tutorial, stick with `sd15` unless you have a reason to
switch.

### Few-step speed — `--fast` presets

SDXL at full quality is ~30 steps. **Distilled few-step presets** cut that
to 4–8 steps (roughly 4–7× faster) by bundling a published LoRA with the
right sampler settings. Just add `--fast`:

```bash
# SDXL-Lightning — 8 steps, Euler-trailing, CFG-free (near-full quality)
plakat generate "a lighthouse at sunset" --model sdxl --fast lightning-sdxl-8

# Or 4 steps for maximum speed
plakat generate "a lighthouse at sunset" --model sdxl --fast lightning-sdxl-4

# Hyper-SD is the other SDXL family; LCM-LoRA is the older option
plakat generate "a fox" --model sdxl --fast hyper-sdxl-8
```

The preset prepends its LoRA, sets the step count / guidance, and pins the
scheduler — you can still override any of those explicitly. SDXL presets
need a base `--model sdxl` and don't compose with `--refiner`. Flux has its
own presets (`hyper-8`, `hyper-16`, `turbo-alpha`). Run
`plakat doctor --capability` to see every preset grouped by family.

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

### Pulling LoRAs straight from Civitai

Instead of `plakat civitai download <ID>` + copy the printed
path, `--lora` accepts a `civitai:` shorthand that downloads on
first use and caches for every subsequent run:

```bash
# Model ID → latest version, primary file
plakat generate "..." --model sd15 --lora civitai:12345

# Model ID + scale
plakat generate "..." --model sd15 --lora civitai:12345:0.7

# Pin to a specific older version
plakat generate "..." --model sd15 --lora civitai-version:67890:0.6

# Pick a non-primary file inside the version
plakat generate "..." --model sd15 \
    --lora "civitai:12345#alternate-weights.safetensors:0.5"
```

The first run downloads to
`<plakat-cache>/civitai/model-<id>/version-<id>/`; subsequent runs
short-circuit on the cache. Gated assets require
`CIVITAI_API_KEY` from
<https://civitai.com/user/account> — the
plain `--lora civitai:...` works for public LoRAs without one.

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
 count: 1 # how many images per task
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

### Resuming a crashed / interrupted scenario

When a scenario with many tasks crashes partway (or you Ctrl-C
it), restart with `--resume` to pick up where it stopped:

```bash
plakat scenario my_big_scenario.hjson --resume
#   ↺ task1: all 4 output(s) already on disk — skipping
#   ↺ task2: all 4 output(s) already on disk — skipping
#   ▶ task3: generating ...
```

A task counts as "already done" when **every** expected output
PNG exists at the expected seed under the task's output
directory. Per-task seed numbering is reproducible across runs
(it derives from the scenario `seed:` + task index, not from
random state), so re-running the scenario lands the surviving
tasks on the same filenames the first run would have produced.

If you instead want to **regenerate everything from scratch**
(say, you re-trained a LoRA and want fresh outputs at the same
seeds), pass `--force`:

```bash
plakat scenario my_big_scenario.hjson --force
```

`--resume` and `--force` are mutually exclusive — clap rejects
both at once. Neither flag preserves the default behaviour:
existing files get silently overwritten.

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
 { name: dawn, prompt: "..." }
 { name: rainy, prompt: "..." }
 { name: snowy, prompt: "..." }
 ]

 # 3 scenes × 3 weather × 2 count = 18 images in one run.
 count: 2
 tasks:
 [
 { name: forest_dawn, scene: forest, weather: dawn, prompt: "a fox" }
 { name: forest_rainy, scene: forest, weather: rainy, prompt: "a fox" }
 { name: forest_snowy, scene: forest, weather: snowy, prompt: "a fox" }
 { name: meadow_dawn, scene: meadow, weather: dawn, prompt: "a fox" }
 { name: meadow_rainy, scene: meadow, weather: rainy, prompt: "a fox" }
 { name: meadow_snowy, scene: meadow, weather: snowy, prompt: "a fox" }
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
 steps: 50 # this task uses 50 steps instead of 28
 guidance: 8.5 # and stronger guidance
 size: 768x768 # and a larger output
 seed: 42 # and a fixed seed
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
│ ├── plakat-1000.png
│ └── plakat-1001.png # second image (because count: 2)
├── fox_forest_rainy/
│ ├── plakat-1002.png
│ └── plakat-1003.png
└── ...
```

The base seed comes from the scenario's `seed:` field (default 0).
Task `idx` uses seeds `seed + idx*count` through `seed + idx*count +
count - 1`. With seed=1000 and count=2, task 0 gets 1000-1001, task 1
gets 1002-1003, and so on.

---

## 12. Wildcards

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

## 13. Attention emphasis: `(red:1.2)` / `[blue]` (SD-family)

plakat accepts the Auto1111 / NovelAI prompt grammar — the same
`(emphasis)` / `[de-emphasis]` syntax that every Civitai LoRA card
uses in its example prompts. The model gets a strong directional
nudge at the tokens you bracket, without your needing to retrain
or pick different schedulers.

| Syntax | Effect |
|---|---|
| `(token)` | Boost weight by `×1.1`. Stack with nesting. |
| `((token))` | `×1.21` (1.1 × 1.1). |
| `(token:1.5)` | Boost weight to `×1.5` exactly. |
| `[token]` | Reduce weight by `×1/1.1 ≈ 0.909`. |
| `[[token]]` | `×0.826`. |
| `[token:0.6]` | Set weight to `×0.6` exactly. |
| `\(`, `\)`, `\[`, `\]` | Literal punctuation — escape when you actually mean the character. |

```bash
# Push the model toward "red" while gently down-weighting "small"
plakat generate "a (red:1.4) fox in a [small] meadow" \
    --model sd15

# Realistic Civitai-style prompt with multiple emphases
plakat generate \
    "masterpiece, best quality, (1girl:1.2), (red hair:1.3), [low quality]" \
    --model sd15
```

Implementation: the parser splits your prompt into weighted
segments, each segment tokenizes independently, and after the CLIP
forward pass plakat scales each token's hidden-state row by its
segment weight. The pooled CLIP-G output that SDXL feeds into
`add_embedding` is left unweighted (pooling collapses to one row,
so per-token weights have no meaningful target there).

**Compatibility**:
- SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo only. Flux + SD3 use
  T5 + CLIP-pooled; the weighting hook isn't wired for them in
  this release.
- Unbalanced parens (`a ( red fox` with no closing `)`) pass
  through as literal characters — no error, just no emphasis.
- Composes with `--clip-skip`, `--lora`, wildcards, ADetailer,
  Hires fix.

## 14. CLIP-skip (SD 1.5 / SD 2.1)

The Auto1111 / NovelAI community default for SD 1.5 anime checkpoints
(Anything-v3, AnyLoRA, ...) is to read the **penultimate** CLIP
hidden state rather than the last. plakat exposes this as
`--clip-skip N`:

```bash
# Default — last layer (diffusers default, byte-identical to previous releases):
plakat generate "..." --model sd15

# Penultimate — community-standard SD 1.5 anime path:
plakat generate "..." --model sd15 --clip-skip 2
```

SDXL ignores `--clip-skip` (its dual-encoder path already uses
penultimate by training default — plakat logs a warning if you pass
`--clip-skip > 1` on SDXL). Flux / SD3 don't use this flag at all
(T5 + CLIP-pooled architecture).

## 15. ADetailer — face refinement (SD-family)

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
the FaceID portrait flow uses. Convert InsightFace's `det_500m.onnx`
(inside `buffalo_sc.zip`) with plakat's own command — no Python:

```bash
# convert-onnx is opt-in (needs protoc at build time): cargo install plakat --features onnx
plakat convert-onnx det_500m.onnx scrfd_500m.safetensors --arch scrfd-500mf
export PLAKAT_SCRFD_WEIGHTS=$(pwd)/scrfd_500m.safetensors
```

(Most users never need this — the converted weights are already hosted and
auto-downloaded; rebuild them yourself only if you want to.)

…or point `PLAKAT_SCRFD_HF` at a converted file hosted on HuggingFace
(`<user>/<repo>#scrfd_500m.safetensors`). A raw `onnx2torch` dump will
**not** load — the keys must match plakat's loader, which `convert-onnx`
guarantees. Without one of these, `--adetailer` bails loud.

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

## 16. Browsing Civitai

[Civitai](https://civitai.com) is the major community hub for SD
checkpoints, LoRAs, embeddings, and ControlNet variants. plakat
ships a built-in browser so you don't have to copy URLs around.

**Search**:

```bash
# Top 10 LoRAs matching "watercolor"
plakat civitai search "watercolor" --asset-type lora

# Checkpoints, page 2 of 20
plakat civitai search "anime" --asset-type checkpoint --limit 20 --page 2
```

Each result shows the model ID, name, type, base model, trigger
words (for LoRAs), and the available files marked with `★` for
the primary (recommended) file.

**Info** — drill into one model:

```bash
plakat civitai info 12345
plakat civitai info https://civitai.com/models/12345
plakat civitai info "https://civitai.com/models/12345?modelVersionId=789"
```

**Download**:

```bash
# Download the latest version's primary file
plakat civitai download 12345

# Pin a specific version via URL
plakat civitai download "https://civitai.com/models/12345?modelVersionId=789"

# Pick a non-primary file by name
plakat civitai download 12345 --file "config.yaml"
```

The file lands at `<plakat-cache>/civitai/model-<id>/version-<id>/<filename>`
with a `metadata.json` alongside. plakat prints the absolute path —
drop it into `--lora` or `--model`:

```bash
plakat civitai download 12345
# → ~/.cache/plakat/civitai/model-12345/version-789/lora.safetensors

plakat generate "..." --model sd15 \
 --lora ~/.cache/plakat/civitai/model-12345/version-789/lora.safetensors
```

**Gated assets**: some Civitai models require an account. Set
`CIVITAI_API_KEY` from
<https://civitai.com/user/account> (API Keys section). Public
models work without one.

## 17. Hires fix — escape the trained-resolution ceiling

SD 1.5 was trained at 512², SDXL at 1024². Sampling much past
those introduces the "multi-head problem": the model loses track of
global composition across more tokens than it saw at train time and
produces repeated faces / doubled limbs / malformed crowds.

The standard mitigation: generate at the trained resolution, then
upscale + img2img-refine. plakat exposes this as `--hires-fix`:

```bash
# SD 1.5 → 1.5k canvas via 2x hires-fix (Lanczos upscale, 0.5
# refine strength).
plakat generate "an astronaut on a beach, full body shot" \
 --model sd15 --size 768x768 \
 --hires-fix

# SDXL → 4K via 2x ESRGAN upscale + img2img refine.
plakat generate "an architectural drawing of a cathedral" \
 --model sdxl --size 1024x1024 \
 --hires-fix --hires-upscaler real-esrgan-x2

# Aggressive refine — let the model rework small detail more.
plakat generate "..." --model sd15 --size 768x768 \
 --hires-fix --hires-strength 0.7 --hires-scale 2.0
```

**Knobs**:

| Flag | Default | Effect |
|---|---|---|
| `--hires-scale F` | `2.0` | Multiplier for classical upscalers. ML upscalers ignore (use native scale). |
| `--hires-strength F` | `0.5` | img2img strength on the upscaled image. Lower = preserve composition; higher = re-imagine. |
| `--hires-upscaler MODE` | `lanczos` | `lanczos / bicubic / bilinear / nearest / real-esrgan-x2 / real-esrgan-x4 / real-esrgan-anime-x4`. |
| `--hires-steps N` | (main `--steps`) | Step count for the refine pass. |

**Composition**:
- Composes with `--lora` (refine uses the same stack).
- Composes with `--adetailer` (face refinement runs on the upscaled
 image — gives better results than refining at the small res).
- Does **not** compose with `--artefact*` (the upscale changes
 image dims; the artefact compositor would misplace stamps —
 plakat bails loud).
- Does **not** compose with `--tiled` (tiled has its own 4K
 workflow; combining is redundant).

**SD-family only**: Flux / SD3 already have native tiled paths
(`--tiled`) for high-res output. `--hires-fix` on Flux / SD3 bails
loud — use `--tiled` instead.

**Recipe — 4K poster**:
```bash
plakat generate "a vintage travel poster of Tokyo at night" \
 --model sd15 --size 768x768 --steps 30 \
 --hires-fix --hires-upscaler real-esrgan-x2 --hires-strength 0.45 \
 --adetailer
```

## 18. Textual Inversion (partial)

Textual Inversion (TI, sometimes called "embeddings") learns new
"words" by training one or more embedding vectors against a small
image set. The output is a tiny `.safetensors` file (typically
5–50 KB) — much smaller than a LoRA.

**Inspect a TI file** with `plakat embedding info`:

```bash
plakat embedding info ./my-style.safetensors
# trigger: my-style
# shape: 1 vector(s) × 768 dim [SD 1.5 (CLIP-L 768)]
# usage: `my-style` once per prompt; ...

# Civitai TIs are also resolvable via HF:
plakat embedding info sd-concepts-library/cat-toy
```

The inspector reports the trigger word, vector count (1 for most
TIs, 2–8 for multi-vector concepts), embedding dim (768 for SD 1.5,
1024 for SD 2.1, 1280 for SDXL CLIP-G), and matches it against the
SD variant.

**Runtime injection** — passing `--embedding PATH:trigger:scale`
to `plakat generate` — is **not wired**. The parser + merger ship
(lib tests pin the contract), but candle 0.8 keeps
`clip::Config.vocab_size` private, blocking the in-place vocab
extension needed to register the new tokens. The wiring lands
when the candle API surface opens, alongside a vendored CLIP path.

In the meantime:
- Use `plakat embedding info` to verify TI files you've downloaded.
- For Civitai TIs that ship a LoRA equivalent: prefer the LoRA
 variant — `plakat generate --lora` works today.
- For TI-only concepts: convert via the [kohya-ss
 conversion script](https://github.com/kohya-ss/sd-scripts) and
 use as a LoRA.

## 19. Grid output — bundle a sweep into one image

With `--count N > 1`, `--grid` writes an additional
`plakat-grid-<base-seed>.png` next to the per-image files,
combining the N outputs in a near-square layout. Great for
prompt-iteration sweeps where you want to see all variations at
a glance.

```bash
plakat generate "a fox in {tall|short} grass at {dawn|noon|dusk}" \
    --wildcard-dir wildcards/ --count 6 --seed 1000 --grid
# → out/plakat-1000.png ... out/plakat-1005.png
# → out/plakat-grid-1000.png   (3×2 layout)
```

Knobs:

| Flag | Default | Effect |
|---|---|---|
| `--grid` | off | Enable grid composition. No-op when `--count == 1`. |
| `--grid-cols N` | `ceil(sqrt(count))` | Force column count. 4 → 2×2, 6 → 3×2, 9 → 3×3, 16 → 4×4. |
| `--grid-padding PX` | `0` | White-padding strip between cells. Higher = clearer cell separation; `0` = flush. |

The grid runs **last** in the post-processing pipeline so any
artefacts, ADetailer face refinement, or Hires-fix upscaling
land in the per-image files first and are reflected in the grid
cells.

## 20. Reproducibility — PNG metadata + JSON sidecar

Every `plakat generate` output ships with an A1111-compatible
`parameters` PNG tEXt chunk plus a sibling `<filename>.json`
sidecar. The chunk carries the recipe in the format any image
viewer in the SD ecosystem recognises — Auto1111 Web UI, Civitai
image uploader, ComfyUI drag-to-load, sd-prompt-reader, and the
various browser extensions all surface it inline.

```text
a fox in tall grass
Negative prompt: blurry
Steps: 28, Sampler: euler-a, CFG scale: 7.5, Seed: 42, Size: 512x512, Model: sd15, Generator: plakat 0.17.0
```

The JSON sidecar carries the same info in structured form — the
full LoRA list, ControlNet stack, refiner config, etc. Use it
when scripting around the recipe (e.g. "regenerate every PNG in
this directory at higher steps"):

```bash
plakat generate "a fox in tall grass" --seed 42
# → ./out/plakat-42.png
# → ./out/plakat-42.json    (sibling JSON sidecar)

cat ./out/plakat-42.json
# {
#   "prompt": "a fox in tall grass",
#   "model": "sd15",
#   "seed": 42,
#   "steps": 28,
#   ...
# }
```

To opt out entirely (e.g. you're shipping outputs externally and
don't want the recipe embedded), pass `--no-metadata`:

```bash
plakat generate "a fox" --no-metadata
# → ./out/plakat-<seed>.png   (no tEXt chunk, no sidecar)
```

## 21. Live preview during denoise

Long denoise runs are a black box — you click `plakat generate`
and wait. `--preview-every N` writes a low-cost latent
projection to `plakat-<seed>-preview.png` every N steps so you
can watch progress in any auto-refreshing image viewer.

```bash
plakat generate "a fantasy castle on a misty mountaintop" \
    --model sd15 --steps 28 --preview-every 4
# → ./out/plakat-<seed>-preview.png   (updated at steps 4, 8, 12, ...)
# → ./out/plakat-<seed>.png            (final, after step 28)
```

Knobs:

| Flag | Default | Effect |
|---|---|---|
| `--preview-every N` | `0` (off) | Write a preview every N denoise steps. `1` is per-step (lots of file churn); `4`-`8` is the typical setting. |
| `--preview-size PX` | `384` | Longer-side dimension of the preview PNG. Smaller = faster writes; larger = more detail visible. |

**How it works**: the preview is **not** a full VAE decode (that
would add hundreds of milliseconds per write). Instead, plakat
projects the partial latent through a community-derived
4-channel → RGB matrix (the same one A1111 and ComfyUI use for
their "approx" previews). Microseconds per write — adds no
meaningful runtime cost to the generation.

**Trade-off**: colours are recognisable but slightly off, edges
are blurry vs the final VAE-decoded output. Good enough for "is
the generation going the right direction?" feedback. The final
saved PNG always uses the full VAE decode.

**Compatibility**: SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo only.
Flux and SD3 use 16-channel latents with a different projection
matrix that isn't wired in this release — they ignore
`--preview-every`. The final output PNG is unaffected.

**Tip for live monitoring**:

```bash
# Linux / WSL — feh auto-reloads when the file changes
feh --reload 1 ./out/plakat-42-preview.png &
plakat generate "..." --seed 42 --preview-every 2 --steps 50

# macOS — Quick Look updates if you keep it open
qlmanage -p ./out/plakat-42-preview.png &
plakat generate "..." --seed 42 --preview-every 2 --steps 50
```

## 22. Quality knobs (free-quality guidance)

Four opt-in, default-off knobs that improve sampling quality at little
or no extra cost. All are **verify-safe**: leave them off and the output
is byte-identical to previous releases. Turn them on when you want a
specific improvement.

| Flag | What it does | When to use |
|---|---|---|
| `--pag-scale <x>` | Perturbed-Attention Guidance. `0` = off; try `2`–`3`. Runs an extra conditional forward with self-attention perturbed to identity, giving sharper structure/detail — especially at low `--guidance`. | Soft or mushy structure, or you're running low CFG. SD 1.5 / SDXL (perturbs the mid block) + PixArt; SD 3.5 experimental. |
| `--guidance-rescale <phi>` | CFG-rescale (~`0.7`). Rescales the guided prediction toward the conditional's statistics, curing high-`--guidance` over-exposure / colour wash-out. | High-CFG images that look blown-out or over-saturated. SD 1.5 / SDXL / PixArt / SD3. |
| `--freeu` | FreeU. Reweights the UNet up-blocks (backbone boost + Fourier skip low-pass) for richer detail/texture at no runtime cost. Tune with `--freeu-params b1,b2,s1,s2`. | Free detail/texture bump. SD 1.5 / SDXL. Defaults `1.2,1.4,0.9,0.2`; SDXL wants `1.3,1.4,0.9,0.2`. |
| `--dynamic-threshold <pctl>` | Imagen dynamic thresholding (~`99.5`). Clamps the predicted x0 to its per-sample percentile — another lever on high-CFG saturation. | High-CFG saturation, as an alternative/complement to `--guidance-rescale`. Epsilon SD (1.5 / SDXL). |

They compose freely. A good combined recipe for sharp, well-exposed
output at moderate CFG:

```bash
# Sharper, well-exposed SDXL at moderate CFG
plakat generate "a detailed owl in a forest" --model sdxl \
  --guidance 5 --pag-scale 2.5 --guidance-rescale 0.7 --freeu
```

Note `--pag-scale` costs an extra forward pass per step (roughly the
same overhead as CFG itself); `--freeu`, `--guidance-rescale`, and
`--dynamic-threshold` are essentially free.

## 23. Common issues

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
