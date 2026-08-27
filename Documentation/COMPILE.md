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
| `negative:` | concatenate | seed terms guaranteed in the generated negative |
| `style:` | concatenate | injected into the LLM **system** prompt (shapes *how* it writes) |
| `translate:` | last-wins | pre-translate the body from this language to English (LLM) |
| `persona:` | concatenate | inject `~/.config/plakat/personas/<name>` into the system prompt |

**Scenario commands** (straight to HJSON, no LLM):

| Command | Merge | HJSON |
|---|---|---|
| `model:` | last-wins | global `model` + per-scene **family profile** (scenarios share one model) |
| `lora:` | accumulate | global `loras` |
| `seed:` `count:` `size:` `steps:` `guidance:` `scheduler:` `refine:` | last-wins | per-task override |
| `name:` | last-wins | task name (auto from the first 6 words if absent) |
| `skip:` | last-wins | `true` omits the block |

**Inheritance:** concatenate = global + scene merged; accumulate = global + scene
combined; last-wins = scene beats global.

**Model family** (`SD15` / `SDXL` / `Flux`) is detected from the scene model, else
the global model, else `--model`. It selects the prompt-writing profile: SD15 →
comma-keyword & <75 tokens; SDXL → mixed prose/keywords 60–150; Flux → prose, short
or empty negative.

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
