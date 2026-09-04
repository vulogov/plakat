# Scenarios — batch generation from HJSON

`plakat scenario FILE.hjson` lets you describe a batch of related
generations in a single file and run them as one job: image-per-task
counts, per-task overrides, style + persona compositing, output
directory layout, all driven by an HJSON config rather than a
30-flag shell command per image.

Use cases:

- **Series production**: generate the same character across N scenes
  / weathers / poses for a graphic novel or storyboard.
- **A/B testing**: compare three art styles or four LoRA scales on
  the same prompt without re-typing flags.
- **Asset libraries**: build a folder of consistent backgrounds /
  character variants in one overnight run.
- **Reproducibility**: scenario files commit to git; rerunning is
  one command + the same seed yields the same outputs.

This tutorial covers the HJSON schema, the cross-product expansion
(`scene × weather × persona`), per-task overrides, partial-rerun
filters, and the v0.19 ergonomic additions (`--only`, `--limit`,
`--dry-run` polish).

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md).
- Optional: an account at deepseek.com or ai.google.dev for prompt
  enhancement (scenarios accept `enhancer: local` from v0.18 to
  skip the API key entirely).

## 1. The smallest possible scenario

```hjson
{
  model: sd15
  out: ./out/quickstart

  scenes: { meadow: "a fox in a sunlit meadow" }
  weathers: { clear: "clear skies" }

  tasks: [
    { name: t1, scene: meadow, weather: clear, prompt: "" }
  ]
}
```

Save as `quickstart.hjson` and run:

```bash
plakat scenario quickstart.hjson
```

What plakat does:

1. Loads the SD 1.5 backbone once.
2. For the single task `t1`: concatenates the scene + weather +
   task prompt → `"a fox in a sunlit meadow, clear skies"`.
3. Generates one image at `./out/quickstart/t1/plakat-<seed>.png`.

HJSON is JSON with relaxed punctuation — quotes optional on keys,
trailing commas allowed, comments via `#`. The file extension can
be `.hjson` or `.json` (plakat parses both).

## 2. Cross-product expansion

Scenarios shine when you generate the same task across multiple
scenes / weathers / personas:

```hjson
{
  model: sdxl
  count: 4               # 4 images per task
  out: ./out/series

  scenes: {
    forest:  "a wooden lodge nestled among tall pines"
    desert:  "a wooden lodge in arid red-rock badlands"
    coast:   "a wooden lodge above a windswept cliff overlooking the sea"
  }
  weathers: {
    dawn:    "soft pre-dawn light, gold-rimmed clouds"
    storm:   "moody thunderclouds, occasional lightning"
  }

  tasks: [
    { name: lodge_forest_dawn,  scene: forest, weather: dawn,  prompt: "" }
    { name: lodge_forest_storm, scene: forest, weather: storm, prompt: "" }
    { name: lodge_desert_dawn,  scene: desert, weather: dawn,  prompt: "" }
    { name: lodge_desert_storm, scene: desert, weather: storm, prompt: "" }
    { name: lodge_coast_dawn,   scene: coast,  weather: dawn,  prompt: "" }
    { name: lodge_coast_storm,  scene: coast,  weather: storm, prompt: "" }
  ]
}
```

That's 6 tasks × 4 images = 24 outputs. Each lands at
`./out/series/<task_name>/plakat-<seed>.png` (and `+1`, `+2`, `+3`
for the additional count).

The cross product is **explicit** — you list every combination as
its own task. This is verbose but lets you skip combinations that
don't make sense (a "lodge_coast_storm" might be great; a
"lodge_forest_storm" maybe not).

## 3. Per-task overrides

Most scenario-level fields can be overridden per task:

```hjson
{
  model: sd15            # scenario-level default
  count: 1
  steps: 28
  out: ./out

  scenes: { hero: "a battle-worn knight" }
  weathers: { studio: "studio lighting, neutral background" }

  tasks: [
    # Inherits scenario defaults
    { name: knight_default, scene: hero, weather: studio, prompt: "" }

    # Override count + steps for a specific task
    { name: knight_x4, scene: hero, weather: studio, prompt: "",
      count: 4, steps: 50 }

    # Override the model entirely (uses SDXL for the headshot)
    { name: knight_sdxl, scene: hero, weather: studio, prompt: "",
      model: sdxl, size: 1024x1024 }
  ]
}
```

Per-task overrides apply: `model`, `count`, `steps`, `guidance`,
`seed`, `size`, `aspect`, `base`, `scheduler`, `loras`,
`lora-scale`, `clip-skip`, `concept-image`, `fast`, `tiled`,
`refiner-frac`, `enhance`, `kontext-bucket`. The full schema is
in `Documentation/GENERATE.md`.

## 4. Per-task LoRA stacks (v0.15)

Per-task `loras:` arrays apply on top of the scenario-level stack:

```hjson
{
  model: sd15
  loras: [civitai:99999:0.5]    # scenario-level "base" LoRA

  tasks: [
    { name: variant_a, scene: ...,
      loras: [civitai:11111:0.7] }    # adds variant_a-specific LoRA
    { name: variant_b, scene: ...,
      loras: [civitai:22222:0.6] }    # adds variant_b-specific LoRA
  ]
}
```

Each task loads + applies + clears its specific LoRAs around the
generation — no model reload, just the LoRA-merge dance. Works on
SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo and Flux (BF16 / GGUF / NF4).

## 5. Personas (identity preservation)

Render the same person across many scenes by declaring a persona
once + referencing it per task:

```hjson
{
  model: sdxl
  out: ./out/character

  personas: [
    { name: alice,
      photo: ./refs/alice.jpg,
      face-strength: 0.85,
      identity: faceid-sdxl }
  ]

  scenes: {
    cafe:  "a small Parisian cafe interior"
    park:  "a sunlit autumn park bench"
    rain:  "a foggy rainy alley at night"
  }
  weathers: { neutral: "" }

  tasks: [
    { name: alice_at_cafe, scene: cafe, weather: neutral, prompt: "",
      personas: [alice] }
    { name: alice_at_park, scene: park, weather: neutral, prompt: "",
      personas: [alice] }
    { name: alice_in_rain, scene: rain, weather: neutral, prompt: "",
      personas: [alice] }
  ]
}
```

See [`PORTRAIT_TUTORIAL.md`](PORTRAIT_TUTORIAL.md) for the persona
mechanics. The scenario file just references personas by name —
the heavy lifting (ArcFace setup, alignment, identity strength)
is the same as the `plakat portrait` CLI.

## 6. Styles in scenarios

```hjson
{
  model: sd15
  style: watercolor           # bundled named style applied to every task

  tasks: [
    { name: t1, scene: ..., ... }
    { name: t2, scene: ..., ...,
      style: oil-painting }   # per-task override
  ]
}
```

Or detect a style from a reference photo at the scenario level:

```hjson
{
  model: sd15
  style-ref: ./inspiration.jpg

  tasks: [...]
}
```

See [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md) for the catalog
mechanics.

## 7. Prompt enhancement

```hjson
{
  enhancer: deepseek         # API-keyed (set DEEPSEEK_API_KEY)
  # or
  enhancer: gemini           # API-keyed (set GEMINI_API_KEY)
  # or
  enhancer: local            # v0.18 — local LLM, no API key
  # or
  enhancer: auto             # v0.18 — picks based on env vars

  tasks: [
    { name: t1, ..., enhance: false }  # opt out per-task
  ]
}
```

The enhancer runs once per task (after wildcard expansion, before
encoding). Local LLM weights are cached in-process, so a 100-task
scenario loads the LLM once. See
[`PROMPT_ENHANCER_TUTORIAL.md`](PROMPT_ENHANCER_TUTORIAL.md).

## 7b. Naturalize post-pass (`naturalize:`)

A scenario-global **`naturalize:`** field runs the weight-free de-slop pass over **every** image the
scenario produces — the same engine as `plakat naturalize`. The value is a **spec**: a preset,
optional content focuses, and (6.27) the art knobs `paper=` / `medium=` / `brush=`.

```hjson
{
  model: "sdxl"
  naturalize: "photo vegetation=1 sky=0.5"          // de-slop toward realism
  tasks: [ … ]
}
```

For painting styles, name the medium to get **real hand-media brush strokes** (not just grain):

```hjson
naturalize: "painting medium=watercolor paper=0.7"   // watercolor: strokes + edge pooling + paper
naturalize: "photo medium=oil brush=0.8"             // oil: Kuwahara + directional strokes + impasto
```

`medium=` fires the brush pass (`brush=` sets its strength, default 0.6; `brush=0` disables it while
keeping paper/grade). Media: `watercolor` · `oil`/`acrylic` · `gouache` · `ink` · `pastel`. Photos
(no `medium=`) are unaffected. See [`../NATURALIZE_TUTORIAL.md`](../NATURALIZE_TUTORIAL.md). *(This is
exactly what `plakat compile` emits from a `naturalize:` directive, so prose → scenario carries it too.)*

For the **model-backed** painterly repaint (6.27), use `repaint=<strength>`, and steer the brushwork with
a custom **`repaint-style="…"`** anchor. Its value has spaces + commas, so it must be **quoted** — pick an
inner quote that doesn't collide with the outer HJSON string (or write a quoteless HJSON line):

```hjson
// single-quote the outer string → double-quote the style (cleanest):
naturalize: 'repaint=0.28 medium=watercolor repaint-style="fine detailed watercolor, small strokes"'
// or double-quote outer → single-quote the style:
naturalize: "repaint=0.28 medium=watercolor repaint-style='fine detailed watercolor, small strokes'"
```

## 7c. Keep every pass — `unique-files` *(6.27)*

By default a re-run **overwrites** the previous outputs in the out dir. Set **`unique-files: true`** (or pass
**`--unique-files`** on the CLI) to write each run into a fresh timestamped subfolder
(`<out>/run-YYYY-MM-DD_HH-MM-SS-mmm/…`), so repeated passes of the same scenario are kept side by side:

```hjson
{
  model: "sd35"
  out: "./out/street"
  unique-files: true          // pass 1 → ./out/street/run-…-101/… ; pass 2 → run-…-880/…  (both kept)
  tasks: [ … ]
}
```

```bash
plakat scenario street.hjson --unique-files      # CLI forces it on regardless of the file
```

Because every pass lands in a new folder, `--resume` has nothing to skip (each run is fresh). This pairs
naturally with a **selection-corpus** workflow: bump `count:` (varies the seed per image) to render several
variations, then `plakat rank <run-dir>` — or `plakat generate … --keep-best K` — surfaces the best draws.
*(`plakat compile` passes `unique-files` straight through, so a prose `unique-files: true` reaches the scenario.)*

### Keep the raw render — `keep-prenaturalize` *(6.27)*

A naturalize pass edits each image **in place** by default. Set **`keep-prenaturalize: true`** to instead
write the naturalized result to a `<stem>.natural.png` sibling, leaving the raw pre-naturalize render next
to it — handy for A/B-ing the de-slop/repaint against the original, or ranking both:

```hjson
{
  naturalize: "repaint=0.2 medium=watercolor paper=0.3"
  keep-prenaturalize: true      // plakat-…-42.png (raw) + plakat-…-42.natural.png (naturalized)
  tasks: [ … ]
}
```
*(Also passes through from prose. Applies to both the model-backed `repaint=` and the weight-free specs.)*

## 7d. Auto-ranking — `ranking` *(6.28)*

Generating `count: N` and then naturalizing all N wastes time on frames you'd never keep. **`ranking:`**
scores each task's fresh renders (weight-free **AI-tell**, no model download), moves the subpar ones into a
**`culls/`** subfolder, and lets the artefact / style / upscale / naturalize / restore-faces passes touch
**only the keepers**.

```hjson
{
  count: 12
  ranking: on                              // defaults: AI-tell, threshold 0.5, min 1
  naturalize: "repaint=0.2 medium=watercolor"
  tasks: [ … ]
}
```
Full spec: `ranking: by=ai-tell threshold=0.5 min=2 max-tries=6`
- **`threshold=`** pass cutoff (AI-tell ≤ x keeps).
- **`min=`** the floor — never keep fewer than this many (the best sub-threshold ones are promoted so a
  task can't drop below `min`). Default `1`.
- **`by=`** `ai-tell` (default) — `aesthetic` and the **generate-until-`min` regeneration** (`max-tries`)
  land in a follow-up; today aesthetic falls back to AI-tell.

Culled frames sit in `culls/` (recoverable). Composes with `unique-files` (each run's ranking is scoped to
its own `run-…/` folder) and `keep-prenaturalize`. *(Passes through from prose too.)*

## 8. Partial-run filters (v0.19)

Three flags for working with subsets of a long scenario:

### `--dry-run`

Validates the scenario, prints the planned task list with expected
output paths, **skips generation**. Catches schema typos +
wildcard typos + missing reference paths before launching a 100-
task overnight batch.

```bash
plakat scenario big.hjson --dry-run

# Output:
# ▶ [1/12] alice_at_cafe (scene=cafe, weather=neutral)
#   (dry-run) would generate 4 image(s) with seeds 100..103 → ./out/alice_at_cafe
# ▶ [2/12] alice_at_park ...
# (dry-run) would have generated 48 image(s) across 12 task(s) → ./out
#   (no files written — drop --dry-run to actually generate)
```

### `--only TASK[,TASK,...]`

Run only the named tasks. Useful for iterating on one task without
re-running the batch:

```bash
plakat scenario big.hjson --only alice_at_cafe
plakat scenario big.hjson --only alice_at_cafe,alice_at_park
```

A typo'd name bails up front with the list of available tasks —
no wasted model load.

`seed_offset` advances on skipped tasks the same way it would on
the full run, so `--only` produces seeds identical to the full
batch's. Iterate without seed drift.

### `--limit N`

Run only the first N tasks (in scenario file order). Sanity-check
the first few before launching the full run:

```bash
plakat scenario big.hjson --limit 3            # first 3 tasks
plakat scenario big.hjson --only a,b,c --limit 2  # `a` and `b`
```

### `--resume`

Skip tasks whose expected output PNGs already exist on disk.
Pairs with `--only` (named-but-already-rendered tasks skip).
Crucial recovery flag when a long batch crashes:

```bash
plakat scenario big.hjson --resume
```

### `--force`

Regenerate every task even when outputs exist. Default behavior
silently overwrites; `--force` makes the intent explicit.
Mutually exclusive with `--resume`.

## 9. Animate scenarios (v0.29)

Until v0.29, scenarios only drove `plakat generate` (single-image
output). v0.29 adds AnimateDiff dispatch: every task can render a
motion-coherent N-frame sequence using the same per-task
override + filter machinery.

The trigger is `type: animatediff` (alias `animate`) at scenario
or task level. When set, every other animate flag the CLI
exposes — `frames`, `window-size`, `window-overlap`, `lcm`,
`motion-lora`, `motion-lora-scale`, `format`, `gif-delay-ms` —
becomes meaningful.

```hjson
{
    model: sd15
    type: animatediff       # scenario default — every task is animate
    frames: 16
    lcm: true               # 4-step AnimateLCM (~5× speedup)
    format: gif
    out: ./out/animations

    # Required scaffolding — even animate tasks need scene + weather
    # for the cross-product expansion machinery. Often these are
    # short additive phrases.
    scene: [
        { name: dawn,  prompt: "at dawn" }
        { name: night, prompt: "under starlight" }
    ]
    weather: [
        { name: mist,  prompt: "wreathed in mist" }
        { name: clear, prompt: "under clear skies" }
    ]

    tasks: [
        {
            name: cottage
            scene: dawn
            weather: mist
            prompt: "a watercolor cottage by a river"
        }
        {
            name: knight
            scene: night
            weather: clear
            prompt: "a knight in a forest, oil painting"
            frames: 32              # per-task override
            format: mp4
        }
    ]
}
```

Run with the standard scenario subcommands — every filter you've
already learned composes with animate tasks:

```bash
plakat scenario animate.hjson --dry-run       # preview the plan
plakat scenario animate.hjson                 # render every task
plakat scenario animate.hjson --resume        # skip rendered tasks
plakat scenario animate.hjson --only cottage  # one task
plakat scenario animate.hjson --limit 3       # cap run length
```

Each animate task lands in `<out>/<task_name>/frame-NNNN.png` +
JSON sidecars. When `format` is `gif` / `mp4` / `webm` / `all`,
an `animation.<ext>` lands in the same directory.

### Composing with the existing scenario machinery

Animate tasks honour the same overrides as generate tasks where
the semantics align:

| Field | Notes |
|---|---|
| `prompt` | Combined with `scene` + `weather` like generate. |
| `negative` | Per-task or scenario default. |
| `seed` | Per-task absolute seed (otherwise `seed + offset`). |
| `size` / `aspect` / `base` | Resolves W×H the standard way. |
| `steps` / `guidance` | LCM defaults override (4 / 1.5) unless explicit. |
| `scheduler` | Forced to `lcm` when `lcm: true`. |
| `control` / `controls` | Per-task ControlNet stack (multi-CN sums). |

### What animate tasks ignore

These fields stay scenario-level only or have no effect on
animate dispatch (they target generate-only pipeline features):
`count`, `refine`, `refine-strength`, `refiner-frac`, `personas`,
`artefacts`, `style` (per-task IP-Adapter), `style-ref`,
`init-image`, `mask`, `outpaint`, `redux-images`, `look`, `genre`,
`enhance`.

The scenario-level `enhancer` field is **optional** for all-
animate scenarios (the prompt enhancement step doesn't apply to
animate tasks). Mixed-kind scenarios (some `generate`, some
`animatediff` tasks) still need it for the generate side.

### Per-task LCM override

```hjson
{
    model: sd15
    type: animatediff
    lcm: true               # AnimateLCM at scenario level
    tasks: [
        # Default — uses AnimateLCM at 4 steps
        { name: quick,  scene: ..., weather: ..., prompt: "..." }
        # Override — disable LCM for one task (uses V3 + DDIM at 20)
        { name: hero,
          scene: ..., weather: ..., prompt: "...",
          lcm: false,
          steps: 28 }
    ]
}
```

Toggling LCM causes a pipeline reload (V3 ↔ AnimateLCM are
different motion adapters). Authors who want both modes in one
batch can sort tasks by LCM mode to minimize reloads.

### Multi-CN per animate task

```hjson
{
    model: sdxl
    type: animatediff
    tasks: [
        {
            name: stacked
            scene: ...
            weather: ...
            prompt: "..."
            controls: [
                { kind: depth, image: ./depth.png, strength: 0.8 }
                { kind: canny, auto-from: ./ref.jpg, strength: 0.4 }
            ]
        }
    ]
}
```

Multi-CN through animate works the same way it does through
generate — residuals from every conditioner sum per denoise step.
SD 1.5 + SDXL both supported.

See [`Documentation/ANIMATEDIFF.md`](../ANIMATEDIFF.md) for the
animate reference (motion-adapter table, memory budget, sliding-
window long-form math).

## 10. The DEEPSEEK_API_KEY environment variable

Scenarios use `${HOME}/.cache/huggingface` for model downloads but
ALSO honor:

| Env var | Purpose |
|---|---|
| `DEEPSEEK_API_KEY` | for `enhancer: deepseek` (API-keyed prompt rewriter) |
| `GEMINI_API_KEY` | for `enhancer: gemini` |
| `PLAKAT_ARCFACE_HF` | for ArcFace setup (persona alignment) |
| `PLAKAT_SCRFD_HF` | for SCRFD face detection (persona alignment) |
| `HF_TOKEN` | for gated HF repos (Flux variants, SD3 / SD3.5) |
| `PLAKAT_CACHE_DIR` | override the HF cache location |

`enhancer: local` (v0.18) doesn't need any API key — it runs an
in-process Qwen2.5-1.5B by default.

## 11. Worked example: character series

```hjson
{
  model: sdxl
  size: 1024x1024
  count: 2
  out: ./out/alice_series

  enhancer: local              # no API key needed

  personas: [
    { name: alice,
      photo: ./refs/alice.jpg,
      face-strength: 0.85,
      identity: faceid-sdxl }
  ]

  loras: [civitai:54321:0.4]   # scenario-level cinematic LoRA

  scenes: {
    morning_kitchen: "alice in a sunlit kitchen, holding a coffee mug"
    afternoon_park:  "alice walking through an autumn park, leaves swirling"
    evening_studio:  "alice editing photos in a dimly-lit home studio"
    night_balcony:   "alice on a high-rise balcony, city lights below"
  }

  weathers: {
    soft:      "soft natural lighting, shallow depth of field"
    cinematic: "cinematic chiaroscuro, dramatic shadows"
  }

  tasks: [
    { name: morning_soft,    scene: morning_kitchen, weather: soft,
      personas: [alice], prompt: "" }
    { name: morning_cine,    scene: morning_kitchen, weather: cinematic,
      personas: [alice], prompt: "" }
    { name: afternoon_soft,  scene: afternoon_park,  weather: soft,
      personas: [alice], prompt: "" }
    { name: evening_cine,    scene: evening_studio,  weather: cinematic,
      personas: [alice], prompt: "" }
    { name: night_cine,      scene: night_balcony,   weather: cinematic,
      personas: [alice], prompt: "" }
  ]
}
```

Run:

```bash
# Validate first
plakat scenario alice_series.hjson --dry-run

# Run the full batch
plakat scenario alice_series.hjson

# Re-run just morning shots after tweaking the scene prompt
plakat scenario alice_series.hjson --only morning_soft,morning_cine
```

5 tasks × 2 images = 10 images, all of "alice" in different
scenes / lightings, consistent identity, consistent style.

## 12. Quality knobs

The free-quality guidance bundle (see the
[`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md) "Quality knobs" section)
is available **scenario-global** — set it once and it applies to every
task:

```hjson
{
  model: sdxl
  guidance: 5
  pag-scale: 2.5              # Perturbed-Attention Guidance (0 = off)
  guidance-rescale: 0.7       # CFG-rescale — cures high-CFG wash-out
  freeu: true                 # FreeU up-block reweighting
  freeu-params: "1.3,1.4,0.9,0.2"   # SDXL defaults (optional)
  dynamic-threshold: 99.5     # Imagen dynamic thresholding (optional)
  tasks: [ ... ]
}
```

| Field | Type | Notes |
|---|---|---|
| `pag-scale` | number | `0` = off; `2`–`3` for sharper structure. |
| `guidance-rescale` | number | `~0.7`. |
| `freeu` | boolean | `true` / `false`. |
| `freeu-params` | string | `"b1,b2,s1,s2"`. |
| `dynamic-threshold` | number | percentile, `~99.5`. |

All default off (output unchanged when omitted).

## 13. Limitations

- **Cross-product is explicit**, not implicit. You list every
  `(scene, weather, persona)` combo as its own task. Some users
  prefer this (skip unwanted combos); others find it verbose. No
  implicit-product mode in this release.
- **One model per scenario** unless you override per task (cost:
  model reload per override). Mixing SD 1.5 + Flux per task is
  possible but pays the load cost.
- **Style detection runs once** at scenario load (`--style-ref`),
  not per task. Use per-task `style:` to vary.

## Where to next

- [`GENERATE.md`](../GENERATE.md) — exhaustive scenario HJSON
  schema reference.
- [`PORTRAIT_TUTORIAL.md`](PORTRAIT_TUTORIAL.md) — persona setup
  before referencing personas in scenarios.
- [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md) — style catalog
  setup for `style:` / `style-ref:`.
- [`PROMPT_ENHANCER_TUTORIAL.md`](PROMPT_ENHANCER_TUTORIAL.md) —
  the four `enhancer:` providers.
