# Compile prose into a scenario (`plakat compile`)

You have a batch of images in mind and you'd rather write them as paragraphs than
hand-author scenario HJSON. `plakat compile` is the bridge: write a `prompts.txt`,
compile it, render it.

## 1. Write `prompts.txt`

Blank lines separate scenes. The first block (if it has no description) sets
global defaults. `key: value` lines are commands; everything else is the scene's
description.

```
# Global — applies to every scene.
model: sdxl
style: cinematic photography, dramatic lighting, photorealistic
negative: blurry, low quality, watermark, text, deformed

# Scene 1
header: wide establishing shot,
A vast frozen tundra stretching to the horizon, a lone rider on horseback
silhouetted against an aurora-lit sky, sparse wind-bent pines.
footer: 8k, award-winning landscape photography

# Scene 2
An elderly cartographer hunched over a candlelit table, tracing coastlines
on parchment with a quill, ink-stained fingers.
seed: 42
count: 2
```

## 2. Compile

```bash
plakat compile prompts.txt            # → prompts.hjson
```

Each block becomes a scenario **task**: header + description + footer assembled
into the prompt, the LLM rewrites it for the model family (here SDXL), and a
negative is generated from the merged `negative:` seeds. Scene 2 carries its own
`seed: 42` and `count: 2`.

**Preview without spending LLM calls:**

```bash
plakat compile prompts.txt --dry-run   # block summary + call count
plakat compile prompts.txt --lint      # catch typos like `styl:` or `negaitve:`
```

**Deterministic (no LLM)** — assemble the prompt verbatim and pass the seed terms
through as the negative:

```bash
plakat compile prompts.txt --no-enhance --no-negative --out prompts.hjson
```

## 3. Render

```bash
plakat scenario prompts.hjson          # batch-generate every task
# or pipe straight through:
plakat compile prompts.txt --out - | plakat scenario -
```

That's the committed proof (`corpus/compile.sh`): the two-scene `basic.txt`
compiles to `basic.hjson` and renders the tundra rider and the cartographer
(`corpus/images/compile/`) — count and seed honoured per scene.

## Handy moves

- **Providers** — `--compile-provider local` runs fully offline; `deepseek` /
  `gemini` need an API key. `auto` tries in order.
- **Per-scene style** — add a `style:` line inside a block to steer just that
  scene (`style: oil painting, Rembrandt lighting`).
- **Different language** — `translate: French` (a block-level command) translates
  the description to English before enhancement.
- **Cache** — `--compile-cache` skips re-enhancing unchanged prompts; clear with
  `--compile-cache-clear`.
- **Iterate** — `--diff existing.hjson` shows which tasks changed since last time;
  `--decompile scene.hjson` turns a scenario back into an editable `prompts.txt`
  (it now round-trips spec-tasks — `faceswap` / `texture` / `product` / `comic` /
  `fractal` — not just plain generate scenes).
- **Watch** *(6.22)* — `plakat compile prompts.txt --watch --no-enhance` re-compiles
  the moment you save the file — an instant authoring loop.
- **Lint before you spend** — `--lint` catches unknown commands, **duplicate task
  names**, and **repeated commands** with no LLM cost; the compiled scenario is also
  validated before writing, so it's guaranteed to load.

## Non-t2i tasks in prose

A block can compile to a **non-generate task** — it renders from directives, not a
prompt. For example, swap a face into a photo *(6.22)*:

```
model: sdxl

name: swap-alice
type: faceswap
faceswap-scene: family.jpg
faceswap-source: alice.png
```

Or a fractal, a seamless material, a map, a book ornament — see the full list in
[`../COMPILE.md`](../COMPILE.md#task-type-blocks-type-). Combine with `@include`
to keep a big prose set tidy:

```
model: sdxl

@include scenes/portraits.txt
@include scenes/materials.txt
```

Full reference: [`../COMPILE.md`](../COMPILE.md).
