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

## Adding a LoRA

Add a `lora:` line to the **global block** — it's repeatable, so LoRAs stack, and it
becomes the scenario's top-level `loras: [...]` (applied to every task; a scenario runs one
model + one LoRA stack). Each value is a LoRA **spec**: a source with an optional `:scale`
(default `1.0`).

```
# Global block
model: sdxl
lora: /Volumes/AI/loras/my-style.safetensors:0.8     # a local file at 0.8 strength
lora: some-org/some-lora:0.7                          # a Hugging Face repo (org/name)
lora: some-org/some-lora#pytorch_lora_weights.safetensors:0.6   # …a specific file in that repo
lora: civitai:123456:0.75                             # a Civitai model id (latest version)
lora: civitai-version:789012:0.75                     # …or a pinned version id

# Scene 1
A misty pine forest at dawn, volumetric light through the canopy.
```

The spec grammar (same as the CLI `--lora` flag):

| Form | Example | Meaning |
|---|---|---|
| Local path | `path/to/lora.safetensors:0.8` | a file on disk (`:scale` optional) |
| HF repo | `org/name:0.7` | Hugging Face repo (one `/`); add `#file.safetensors` to pick a file |
| Civitai model | `civitai:123456:0.75` | Civitai model id (latest version) |
| Civitai version | `civitai-version:789012:0.75` | a pinned Civitai version id |

Rules of thumb: single-LoRA scales `0.5–0.9`; keep the **total** across a stack ≤ ~`1.2` so
it doesn't over-cook. A LoRA's **trigger words** (if any) go in your prose description, not the
`lora:` line. Put `lora:` in the global block — per-scene `lora:` isn't emitted (a scenario's
LoRA stack is shared across its tasks).

## Reusable components (`composition:`) *(6.26.x)*

Scenes repeat pieces — the same street, sky, or person across many shots. Define each piece once
as a **component** in the global block, then **compose** scenes from them:

```
# Global block — define reusable pieces
model: sdxl
component.street: cobblestone medieval street, timber-framed houses
component.sky:    bright clear sky, soft clouds
component.market: market stall with fruit and vegetables

# Scene 1 — compose from components, THEN add this scene's own prose
composition: component.street, component.market
A baker carrying a basket of bread.

# Scene 2 — composition only (no prose needed)
composition: component.street, component.sky
```

- **Define:** `component.<name>: <text>` in the global block. The name is yours (`street`, `sky`, …).
- **Compose:** `composition: component.a, component.b` (bare `a, b` also works) — resolved in list
  order. A block is valid with a **composition, prose, or both**.
- **Order:** `header` → **composition** → your prose → `footer` — so components come first, then the
  scene's unique detail (*compose, then prose*).
- The composed text is just the prompt, so it still gets **translated, enriched, and token-budget
  checked** like any other. An unknown `component.<name>` reference is a clear compile error.

This **extends** the existing keys — `header:`/`footer:`/`persona:`/`style:` and free-text all work
exactly as before; composition just slots in. *(A component's text is literal — one level, not itself
a composition.)*

**In a hand-authored scenario too.** The same feature works directly in scenario HJSON: a global
`components` map + a per-task `composition` list. At render the components fold into the task's
`prompt` (compose, then prose) and the enhancer enriches the whole — so a scenario is as DRY as a
compiled prose file:

```hjson
{
  model: "sdxl"
  components: { street: "cobblestone medieval street", sky: "bright clear sky" }

  tasks:
  [
    {
      name: market
      composition: ["street", "sky"]   // bare names or "component.street" both work
      prompt: "a baker carrying bread"  // ← appended after the components
    }
  ]
}
```

`plakat scenario` resolves it at load; `--check` validates the refs; an unknown component is a clear
error listing the defined names.

## Every command key

`key: value` lines are commands. Two kinds:

**Prompt keys** — shape or feed the LLM. Repeats concatenate (except `translate:`, last-wins):

| Key | Example | What it does | Needs LLM? |
|---|---|---|---|
| `header:` | `header: wide establishing shot,` | prepended to the prompt | no (assembled verbatim) |
| `footer:` | `footer: 8k, golden-hour light` | appended to the prompt | no (assembled verbatim) |
| `negative:` | `negative: blurry, extra fingers` | seed terms **guaranteed** in the auto-negative (a rogue model can't drop them; `--no-negative` uses them verbatim) | no |
| `style:` | `style: oil painting, Rembrandt lighting` | steers *how* the LLM writes (goes into its system prompt) | **yes** |
| `translate:` | `translate: Russian` | translate the description to English **before** enhancing | **yes** |
| `persona:` | `persona: gandalf` | inject `~/.config/plakat/personas/gandalf` into the system prompt (falls back to the bare name as a cue) | **yes** |
| `component.<name>:` | `component.sky: bright clear sky` | *(global)* define a reusable prompt piece — see [Reusable components](#reusable-components-composition-6260) | no |
| `composition:` | `composition: component.sky, component.street` | assemble named components into the prompt (before the prose) | no (assembled) |

> `style:`, `translate:` and `persona:` only take effect **with** enhancement — under `--no-enhance`
> the description is used verbatim and these are skipped (`header:`/`footer:`/`negative:` still apply).

**Scenario keys** — pass straight to the HJSON, no LLM. Set them in the **global block** as
defaults for every task, and/or inside a **scene block** to override that one task:

| Key | Example | Scope | What it does |
|---|---|---|---|
| `model:` | `model: sdxl` | global | the model to run + the LLM family profile (one model per scenario) |
| `lora:` | `lora: style.safetensors:0.8` | global | add a LoRA (repeatable → `loras: [...]`) — see above |
| `seed:` | `seed: 42` | global / scene | fixed seed for reproducibility |
| `count:` | `count: 4` | global / scene | images per task |
| `size:` | `size: 1024x768` | global / scene | render size `WxH` |
| `steps:` | `steps: 30` | global / scene | denoise steps |
| `guidance:` | `guidance: 9` | global / scene | CFG (higher = follow the prompt harder) |
| `scheduler:` | `scheduler: dpm++` | global / scene | sampler (`euler-a`, `dpm++`, `unipc`, …) |
| `refine:` | `refine: 10` | global / scene | SDXL refiner-pass steps |
| `name:` | `name: harbor-dawn` | scene | task name — see auto-naming below |
| `skip:` | `skip: true` | scene | omit this block from the output |

**Auto-naming.** If a block has no `name:`, compile derives one: the first six words of the
description, slugified (`a_misty_harbour_at_dawn_fishing`). With **enhancement**, the name is taken
from the *enhanced English* prompt — so a `translate:`-d block whose source is non-Latin gets a
meaningful English name instead of `scene_N` *(6.26.2)*. If there's still no usable slug (e.g.
`--no-enhance` on non-Latin prose), it falls back to a sequential `scene_1`, `scene_2`, … Auto
names are also de-duplicated with a numeric suffix so two similar scenes can't clobber each other.

**Scenario-parity keys** *(6.26.x)* — a compiled TXT can now drive (almost) everything a
hand-written scenario can. These scalar fields pass straight through to the HJSON (global block →
scenario top level; scene block → that task), type-inferred (`fast: true`, `pag-scale: 3.0`,
`aspect: 16:9`):

| Group | Keys |
|---|---|
| Sizing / device | `aspect`, `base`, `device`, `offline`, `fast`, `lcm` |
| Post-process | `naturalize`, `upscale` |
| Refiner / LoRA | `refiner`, `refine-strength`, `refiner-frac`, `lora-scale` |
| Quality knobs | `pag-scale`, `guidance-rescale`, `freeu`, `dynamic-threshold` |
| Style / image refs | `look`, `genre`, `style-ref`, `style-strength`, `concept-image` |
| img2img / inpaint | `init-image`, `strength`, `mask`, `mask-feather`, `mask-invert`, `outpaint` |
| Animate (video) | `format`, `frames`, `window-size`, `window-overlap`, `motion-lora`, `motion-lora-scale`, `gif-delay-ms` |
| Flux / quant | `kontext-bucket`, `quantize-t5`, `flux-quant-level`, `t5-quant-level`, `smart-zones` |

Plus two specials:
- **`region: X0,Y0,X1,Y1[,w=][,feather=]:prompt`** *(repeatable)* — regional prompting → the task's
  `regions` array (same syntax as `plakat generate --region`).
- **`set.<key>: value`** — the generic escape hatch: passes *any* scenario key straight through
  (for a field not in the list above, or a new one). E.g. `set.pag-scale: 3` ≡ `pag-scale: 3`.

```
# Global
model: sdxl
aspect: 16:9
naturalize: photo
pag-scale: 3.0

# Scene — regional prompting + a per-task override
region: 0,0.55,1,1:men, women and children walking
strength: 0.6
A clean medieval street at dawn.
```

**Task-type keys** — `type:` turns a block into a *non-generate* task (`faceswap`, `map`,
`texture`, `product`, `comic`, `bookart`, `fractal`), each with its own `<type>-…` directives.
See [the task-type section](#non-t2i-tasks-in-prose) below and [`../COMPILE.md`](../COMPILE.md).

**Not exposed in the text format:** compile emits one default **scene**/**weather** axis
(`scene: plain`, `weather: any`) — the scenario's scene×weather *variation matrix* isn't authored
from prose. `scene:`/`weather:` directives are accepted (so they don't trip `--lint`) but **not
emitted**; to use the matrix, hand-edit the compiled `.hjson`. For variations from prose, write one
block per variation instead.

Unknown keys are a `--lint` error, so `styl:` / `negaitve:` are caught before you spend an LLM call.

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
