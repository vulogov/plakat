# `plakat compile` — prose `prompts.txt` → scenario HJSON

`plakat compile` turns a natural-language `prompts.txt` into a ready-to-run
[`scenario`](../README.md) HJSON. Write scenes as paragraphs with optional
`key: value` commands; compile rewrites each through the LLM provider stack
(family-aware) with an auto-generated negative, and emits one task per block.

```bash
plakat compile prompts.txt                 # → prompts.hjson
plakat compile prompts.txt --out -         # → stdout
plakat compile prompts.txt --out - | plakat scenario -   # pipe straight to render
```

## The `prompts.txt` format

Blank-line-separated **blocks**. Each block is free-text lines (the description)
plus `key: value` **command** lines. `#` lines are comments. The **first block is
the global block** iff it has no free text — its commands become the scenario
defaults; every other block becomes one task.

```
# Global defaults (no free text here → this is the global block).
model: sdxl
style: cinematic photography, dramatic lighting
negative: blurry, low quality, watermark

# Scene 1
header: wide establishing shot,
A vast frozen tundra, a lone rider against an aurora-lit sky.
footer: 8k, award-winning landscape photography

# Scene 2
An elderly cartographer tracing coastlines by candlelight.
seed: 42
count: 2
```

### `@include` *(6.22)*

Split a large prose set across files: a line `@include <path>` is inlined with that file's contents
(relative to the including file, recursive, cycle-guarded) **before** parsing — so shared globals or a
library of scenes stay DRY without the heavier Tera template pre-pass.

```
model: sdxl

@include scenes/winter.txt
@include scenes/harbour.txt
@include scenes/*.txt              # glob (sorted) — a single `*` in the last path component
@include body.txt who=alice        # params: substitutes ${who} in the included text
```

## Commands

**Prompt commands** (shape or feed the LLM; never appear verbatim unless noted):

| Command | Merge | Effect |
|---|---|---|
| `header:` | concatenate | prepended to the prompt (joined with `, `; empty value resets the inherited global) |
| `footer:` | concatenate | appended to the prompt |
| `negative:` | concatenate | seed terms **guaranteed** in the generated negative — the auto-negative must contain them verbatim, or it's rejected and your seed terms are used as-is (so a weak model can't replace your explicit negative with a bad one) |
| `style:` | concatenate | injected into the LLM **system** prompt (shapes *how* it writes) |
| `translate:` | last-wins | pre-translate the body from this language to English (LLM) |
| `persona:` | concatenate | inject `~/.config/plakat/personas/<name>` into the system prompt |
| `component.<name>:` | — | *(global)* define a reusable prompt piece; referenced by `composition:` |
| `composition:` | concatenate | *(scene)* comma-list of `component.<name>` refs, assembled into the prompt **before** the prose (compose, then prose). Unknown ref → compile error. Extends header/footer/persona; doesn't replace them |

**Scenario commands** (straight to HJSON, no LLM):

| Command | Merge | HJSON |
|---|---|---|
| `model:` | last-wins | global `model` + per-scene **family profile** (scenarios share one model) |
| `lora:` / `loras:` | accumulate | global `loras` — value is a LoRA **spec** `SOURCE[:scale]` (scale default 1.0): a local `path.safetensors`, an HF repo `org/name` (add `#file`), or `civitai:ID` / `civitai-version:ID`. Repeatable (`lora:` per line) **and** comma-separable (`loras: a:1.2, b:1.0, c:0.8` on one line) — both accumulate into the same list. Put it in the global block — per-scene lora isn't emitted |
| `seed:` `count:` `size:` `steps:` `guidance:` `scheduler:` `refine:` | last-wins | scenario default (global block) or per-task override (scene block) |
| `name:` | last-wins | task name — auto-slugged from the first 6 words; **with enhancement** from the *enhanced English* prompt (so `translate:` blocks get a meaningful name, not `scene_N`); else sequential `scene_N`. Auto names are de-duplicated |
| `skip:` | last-wins | `true` omits the block |

**Scenario-parity pass-through** *(6.26.x)*: common scalar scenario fields pass straight to the
HJSON (global or per-scene), type-inferred — `aspect`, `base`, `device`, `offline`, `fast`, `lcm`,
`naturalize`, `upscale`, `refiner`, `refine-strength`, `refiner-frac`, `lora-scale`, `pag-scale`,
`guidance-rescale`, `freeu`, `dynamic-threshold`, `look`, `genre`, `style-ref`, `style-strength`,
`concept-image`, `init-image`, `strength`, `mask`, `mask-feather`, `mask-invert`, `outpaint`,
`format`, `frames`, `window-size`, `window-overlap`, `motion-lora`, `motion-lora-scale`,
`gif-delay-ms`, `kontext-bucket`, `quantize-t5`, `flux-quant-level`, `t5-quant-level`,
`smart-zones`. Plus the array specials: `region:` *(→ `regions`)*, **`redux:`** *(→ `redux-images`,
6.27)*, **`control: kind:image[:strength]`** *(→ the `controls` object array, 6.27)* — all
repeatable — and the generic **`set.<key>: value`** for anything else.

The **`naturalize:`** value is a full spec string, carried verbatim — so the whole post-pass is reachable
from prose/scenario, including the 6.27 media tokens: `naturalize: medium=oil brush=0.7 scale=0.6`
(weight-free brush strokes) or `naturalize: repaint=0.4 medium=watercolor` (model-backed painterly repaint,
the parity of `plakat naturalize --repaint`; companion tokens `repaint-lora=`, `repaint-model=`). A
successful `repaint=` is terminal unless the spec also names a weight-free knob (`paper=`, `grain=`, a focus).

**Scene/weather axes** *(6.27)*: define `scene.<name>: prompt` / `weather.<name>: prompt` in the
global block to author the scenario's scene×weather variation matrix; each scene block selects one
with `scene: <name>` / `weather: <name>`. With none defined, compile emits the single default axis
(`plain`/`any`). So a compiled TXT now reaches **full** parity with a hand-written scenario. (`tag:`
is still accepted but not emitted.)

### Naturalize post-pass — brush strokes & painterly `repaint` from prose *(6.27)*

`naturalize:` is a **global or per-scene** field whose value is a naturalize spec string, carried through to
the scenario verbatim and applied to every image the run produces. Two media tokens make the painterly
passes reachable straight from prose — no CLI flags:

| Spec | What runs | Model? |
|---|---|---|
| `naturalize: medium=oil brush=0.7 scale=0.6` | weight-free **brush strokes** (`--medium … --brush-strength …`) | no |
| `naturalize: repaint=0.4 medium=watercolor` | model-backed **painterly repaint** (`--repaint`) | yes |
| `naturalize: repaint=0.4 medium=oil repaint-lora=org/oil-lora:0.8` | repaint + a painterly LoRA | yes |
| `naturalize: repaint=0.4 medium=watercolor paper=0.3` | repaint **plus** an explicit weight-free knob (stacks) | yes |

A successful `repaint=` is **terminal** — the analog/paper pass is skipped unless the spec also names a
weight-free knob (`paper=`, `grain=`, a focus). Companion tokens mirror the CLI: `repaint-strength=` is the
`repaint=` value itself, plus `medium=`, `repaint-lora=`, `repaint-model=`.

> **Repaint model.** The repaint runs through the **img2img (UNet) pipeline — SD1.5 / 2.1 / SDXL only.**
> When the scenario's generation model is a transformer family (SD3/3.5, Flux, PixArt, Cascade, Sana) it
> can't img2img, so plakat automatically repaints on **SDXL** (with a note). To choose the repaint model
> yourself, add **`repaint-model=`** to the spec — e.g. a scenario on `model: sd35` that repaints on SDXL:
> `naturalize: "repaint=0.4 medium=watercolor repaint-model=sdxl"`.

Worked example — prose in, scenario out:

```text
# prompts.txt
model: sd35
naturalize: repaint=0.4 medium=watercolor

A quiet harbour at dawn, fishing boats, wet cobblestones.
```

```hjson
// pics.hjson (compiled)
{
  model: "sd35"
  naturalize: "repaint=0.4 medium=watercolor"
  tasks: [
    {
      name: a_quiet_harbour_at_dawn
      prompt: "A quiet harbour at dawn … (enhanced) …"
    }
  ]
}
```

Running that scenario renders the harbour, then repaints it as a watercolor. (Hand-written scenarios use the
identical `naturalize:` field — see NATURALIZE_TUTORIAL's "In scenarios / compiled prose" section.)

**Inheritance:** concatenate = global + scene merged; accumulate = global + scene
combined; last-wins = scene beats global.

**Model family** (`SD15` / `SDXL` / `SD3` / `Flux`) is detected from the scene model, else
the global model, else `--model`. It selects the prompt-writing profile and token budget: SD15 →
comma-keyword & <77 tokens; SDXL → mixed prose/keywords ~150; **SD3/3.5 → prose, ~256 tokens (T5-XXL, no
77-token CLIP cap)**; Flux → prose ~300, short or empty negative. Over-budget prompts are **condensed to
fit** the family (weights preserved) with a note, not just flagged.

## Task-type blocks (`type: …`)

A block can declare a **non-t2i task type** — it compiles to that task in the emitted scenario instead of
a text-to-image render. A `type:`-typed block (or one carrying that type's directives) may **omit the
prose description** — a spec-driven task needs no prompt. Supported types:

- **`type: map`** — a `plakat map` task (`map-spec:` / `map-style:` / `map-paint:` / `map-scale:` /
  `map-tiles:` / `map-sd-model:` / `map-sd-lora:` / `map-provider:`).
- **`type: bookart`** — a `plakat bookart` ornament task (`bookart-origin:` / `bookart-technique:` /
  `bookart-type:` / `bookart-page:` / `bookart-svg:`); free text = the ornament prompt (optional).
- **`type: texture`** — a `plakat texture` material task; free text = the material prompt, or
  `texture-from:` (image-to-material) / `texture-seamless:` / `texture-height:` / `texture-size:` /
  `texture-upscale:`.
- **`type: comic`** — `comic-spec-file:` (a full `ComicSpec`), else the prose is a single-panel page.
- **`type: product`** — `product-spec-file:` (a full `ProductSpec`), else the prose is the subject prompt.
- **`type: faceswap`** *(6.22)* — swap a source face into a scene image: `faceswap-scene:` /
  `faceswap-source:` (+ optional `faceswap-face:`). No prompt (renders from the images).
- **`type: fractal`** *(6.22)* — `fractal-spec:` (the spec string) / `fractal-kind:` / `fractal-palette:`.
  No prompt.
- **`type: animatediff`** (aka `animate`) — an AnimateDiff clip; uses the normal generate fields + the
  block's free text as the prompt.

```
# global: every scene is a bookart ornament in the russian tradition
type: bookart
bookart-origin: russian
bookart-technique: line

a firebird among oak branches
bookart-type: vignette
bookart-svg: true

bookart-type: border            # no prose → a procedural border
```

`plakat compile prompts.txt --out ornaments.hjson` then `plakat scenario ornaments.hjson` renders them.
See [`BOOKART.md`](BOOKART.md#integration-surfaces).

## Flags

| Flag | Default | Description |
|---|---|---|
| `INPUT` | — | `prompts.txt` (`-` = stdin) |
| `--out <PATH>` | `<stem>.hjson` | output (`-`/stdin → stdout) |
| `--compile-provider <P>` | `auto` | `deepseek`/`gemini`/`local`/`local:<alias>`/`auto` |
| `--model <NAME>` | `sdxl` | family fallback when no block names a model |
| `--compile-system <PATH>` | built-in | override the positive system prompt |
| `--no-enhance` | off | skip the positive LLM call (verbatim assembly) |
| `--no-negative` | off | skip the negative LLM call (seed terms verbatim) |
| `--compile-cache` | off | two-namespace SHA-256 disk cache |
| `--compile-cache-clear [all\|positive\|negative]` | — | clear the cache and exit |
| `--compile-parallel <N>` | `1` | max concurrent scenes; `0` = auto (deepseek 3, gemini 5, local/auto 1). Output order is preserved regardless. |
| `--lint` | off | validate (unknown commands, misplaced `skip:`, **duplicate task names**, **repeated commands**, **unknown model/scheduler**); no LLM |
| `--check` | off | *(6.22)* validate that the INPUT is a loadable **scenario** HJSON (deserialises + known task types) and exit — a no-model CI check for any scenario |
| `--explain` | off | *(6.22)* print the resolved model family + the exact LLM **system prompt** per scene and exit; no LLM |
| `--watch` | off | *(6.22)* re-compile whenever the input file changes (dev loop; pair with `--no-enhance`; file input only) |
| `--dry-run` | off | per-block summary + LLM-call count; no LLM |
| `--diff <PATH>` | — | per-task add/change/remove vs an existing scenario |
| `--decompile` | off | inverse: read a scenario HJSON → emit a `prompts.txt` (round-trips spec-tasks) |

The emitted scenario is **validated before writing** *(6.22)* — it must deserialise and every task `type`
must be known — so a compiled scenario is guaranteed loadable, not just well-formed text.

`--no-enhance --no-negative` is **fully deterministic** (no LLM) — the path the
proof corpus exercises (`corpus/compile.sh`).

## Caching

Two SHA-256 namespaces under `~/.cache/plakat/compile/`: `positive/` keys on
(provider, system, input); `negative/` keys on (provider, system, **enhanced
positive**, seeds) — so editing a positive prompt correctly invalidates its
negative. Opt-in (`--compile-cache`); clear with `--compile-cache-clear`.

## Pipe / round-trip

```bash
plakat compile prompts.txt --compile-cache --out - | plakat scenario - --out ./renders/
plakat compile scene.hjson --decompile         # scenario → prompts.txt (re-editable)
plakat compile prompts.txt --diff scene.hjson   # what changed since last compile
```

See [`Tutorials/COMPILE_TUTORIAL.md`](Tutorials/COMPILE_TUTORIAL.md) for a
walkthrough, and [`COMPILE_TEMPLATES.md`](COMPILE_TEMPLATES.md) for the optional
**Tera template pre-pass** (`.tera` inputs, `--features templates`) — generate a
scene series from a data file, with `--var`/`--vars` context and custom filters.
