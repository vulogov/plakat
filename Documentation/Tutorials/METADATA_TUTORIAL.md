# Reading PNG metadata — `plakat metadata FILE`

Every PNG plakat writes carries the full generation recipe — the
prompt, seed, model, sampler, LoRAs, ControlNet stack, refiner
config — embedded as a tEXt chunk in the PNG itself, plus a
sibling `.json` sidecar with the same data in structured form.
The chunk follows the AUTOMATIC1111 convention, so Civitai,
ComfyUI, sd-prompt-reader, and the A1111 Web UI all surface it
when you drag a plakat PNG into their interfaces.

`plakat metadata FILE.png` is the reverse direction: read the
chunk back out from the terminal. Useful when:

- You have an output PNG from a previous plakat run and want to
  recover the exact prompt without consulting shell history.
- You downloaded a PNG from Civitai and want to see what prompt /
  seed / model produced it.
- You're sharing an output with someone and want to verify the
  recipe is actually embedded before sending it on.
- You're debugging a scenario batch and want to confirm what
  parameters one task used.

## Prerequisites

- A working plakat binary.
- At least one PNG written by plakat v0.17 or later (older
  versions didn't embed the chunk yet). Civitai downloads + A1111
  outputs also work — the chunk format is shared.

## 1. The simplest invocation

```bash
plakat metadata ./out/plakat-42.png
```

Output:

```
# parameters (A1111 PNG tEXt)

a brutalist whale poster, watercolor on rough paper
Negative prompt: blurry, low quality
Steps: 28, Sampler: euler-a, CFG scale: 7.5, Seed: 42, Size: 512x768,
Model: sd15, LoRAs: civitai:12345:0.7, Generator: plakat 0.18.0

# sidecar (structured JSON)

{
  "prompt": "a brutalist whale poster, watercolor on rough paper",
  "negative": "blurry, low quality",
  "model": "sd15",
  "seed": 42,
  "steps": 28,
  "guidance": 7.5,
  "scheduler": "euler-a",
  "width": 512,
  "height": 768,
  "loras": ["civitai:12345:0.7"],
  "generator": "plakat 0.18.0"
}
```

Two sections by default — the A1111-format tEXt chunk and the JSON
sidecar. Both have the same data; the A1111 chunk is what other
tools surface, the JSON is what's easiest to script against.

## 2. Filtered output

When you only want one or the other:

```bash
# Only the A1111 chunk — for piping to A1111-compatible tools
plakat metadata foo.png --params-only

# Only the JSON sidecar — for piping to jq
plakat metadata foo.png --json-only | jq .seed
```

`--json-only` and `--params-only` are mutually exclusive.

## 3. Common workflows

### Recovering a forgotten seed

You generated something beautiful three days ago and want to
re-create it with a small tweak. Recovery:

```bash
plakat metadata ./out/plakat-grid-1000.png --json-only | jq .seed
# → 1000

plakat metadata ./out/plakat-1003.png --params-only
# → ...
#   Seed: 1003, ...
```

Re-run with the recovered seed + your tweak:

```bash
plakat generate "the same prompt but with one word changed" \
    --seed 1003 --model sd15
```

### Inspecting a Civitai download

Civitai PNG uploads embed the same A1111 chunk format. Drop one
into `plakat metadata` to see what recipe produced it:

```bash
plakat metadata ~/Downloads/civitai-1234567.png --params-only
```

You'll see the original poster's prompt + LoRA stack + seed. Use
that to either reproduce the image or adapt the recipe for your
own variant.

### Auditing a batch

After a long scenario run, spot-check a few outputs to confirm the
parameters landed correctly:

```bash
for f in ./out/scenario/*/plakat-*.png; do
    echo "=== $f ==="
    plakat metadata "$f" --params-only | head -3
done
```

The first three lines are: prompt, negative prompt, key=value
summary line. Enough to verify per-task overrides took effect.

### Piping to jq for analytics

The JSON sidecar is structured exactly. Easy to aggregate across a
batch:

```bash
# Distribution of seeds across a scenario output dir
find ./out/scenario -name "*.json" -exec \
    cat {} \; | jq -s 'map(.seed) | sort | unique | length'

# Find every output that used a specific LoRA
find ./out -name "*.json" -exec \
    sh -c 'jq -e ".loras[] | select(. | contains(\"civitai:12345\"))" "$1" \
           > /dev/null && echo "$1"' _ {} \;
```

## 4. What's in the chunk

The A1111 parameters string is a single text block with three
parts:

```
<prompt>
Negative prompt: <negative>
Steps: 28, Sampler: euler-a, CFG scale: 7.5, Seed: 42, Size: 512x768,
Model: sd15, ...
```

The third line is comma-separated `key: value` pairs. The values
plakat writes:

| Field | Example | Always present? |
|---|---|---|
| Steps | `28` | yes |
| Sampler | `euler-a` | yes |
| CFG scale | `7.5` | yes |
| Seed | `42` | yes |
| Size | `512x768` | yes |
| Model | `sd15` | yes |
| LoRAs | `civitai:12345:0.7, civitai-version:999:0.5` | when used |
| CLIP-skip | `2` | when `--clip-skip > 1` |
| Refiner | `0.75` | SDXL with refiner |
| Mode | `inpaint` / `img2img` / `animate` | non-t2i modes |
| Strength | `0.7` | img2img |
| Generator | `plakat 0.18.0` | yes |

The JSON sidecar carries the same fields plus a handful of
plakat-specific extras (e.g. `OriginalPrompt` when `--enhance`
rewrote the prompt, `Lerp t` on animate frames, `Animate from` /
`Animate to` on animate frames).

## 5. PNGs without metadata

When the PNG was written with `--no-metadata` (or wasn't written by
plakat / A1111 / Civitai), there's nothing to read:

```bash
plakat metadata generic.png
# stderr:
#   note: generic.png has no `parameters` tEXt chunk
#   (not a plakat / A1111 / Civitai output, or written with
#   --no-metadata).
```

The command exits successfully — "no metadata" isn't an error.
The note goes to stderr so it doesn't pollute scripts that pipe
through to other tools.

## 6. Limitations

- **A1111 format is free-form.** Some fields contain commas inside
  their values (e.g. a LoRA spec with embedded weights). The A1111
  parser community has tools that handle this; plakat's JSON
  sidecar is the structured alternative when you need exact field
  boundaries.
- **No editing.** `plakat metadata` is read-only. To re-write a
  PNG's metadata, the standard approach is regenerating from the
  recipe (the seed is in the chunk — that's the whole point).
- **No batch mode.** One file per invocation. Loop in shell when
  you need to process many.

## 7. Companion: `plakat clone` (v0.19)

Where `metadata` is read-only, `plakat clone` translates the same
recipe into a re-runnable `plakat generate` shell command:

```bash
$ plakat clone ./out/plakat-42.png
plakat generate 'a brutalist whale poster' \
    --negative 'blurry, low quality' \
    --model 'sd15' \
    --seed 42 \
    --size 512x768 \
    --scheduler euler-a \
    --lora 'civitai:12345:0.7'
```

Use it to:

- Reproduce a Civitai download as a local generation (swap in your
  own LoRA, tweak the prompt, rerun).
- Re-render an old plakat output at a different resolution: copy
  the command, change `--size`, run.
- Pipe a recipe into another shell: `plakat clone foo.png
  --one-line | ssh remote -- bash -s`.

JSON sidecar (when present) is preferred — it carries every field
losslessly. Falls back to parsing the Auto1111 chunk for non-plakat
PNGs (Civitai uploads, A1111 Web UI outputs). The fallback covers
the common fields (prompt, negative, Steps, Sampler, CFG, Seed,
Size, Model, Clip skip); plakat-specific extras (ControlNet stack,
refiner config, ADetailer flags) are sidecar-only.

img2img / inpaint / animate clones lose the input-image / mask /
animate-endpoint context (they're not in the recipe). `plakat clone`
always emits `plakat generate` regardless of original mode and
notes the mode mismatch in stderr.

## 8. The other companion: `plakat generate --recipe` (v0.20)

While `plakat clone` produces a shell command for re-running, the
v0.20 `--recipe` flag loads the JSON directly + runs the
generation. Same recipe, no shell-glue intermediate:

```bash
# Same setup as ./out/plakat-42.png but with a different prompt
plakat generate "a different scene description" \
    --recipe ./out/plakat-42.json

# Same recipe but tweak the seed
plakat generate "the original prompt" \
    --recipe ./out/plakat-42.json --seed 999

# Any CLI flag that differs from its default overrides the recipe
plakat generate "..." --recipe ./out/plakat-42.json \
    --steps 50 --guidance 6.5 --size 1024x1024
```

When to use which:

| Goal | Tool |
|---|---|
| See the recipe in the terminal | `plakat metadata FILE.png` |
| Get a shell command to copy + tweak | `plakat clone FILE.png` |
| Run a generation directly from the recipe | `plakat generate --recipe FILE.json` |

The positional `prompt` arg is ALWAYS taken from the CLI — the
recipe never overrides it. If you want a byte-equivalent rerun
(prompt + every other field), use `plakat clone` and pipe to bash.

## Where to next

- **`GENERATE_TUTORIAL.md`** §22 — the write side of metadata
  (what `--no-metadata` opts out of).
- **`GENERATE.md`** in the reference docs — the
  `parameters` chunk format reference and every field.
- **`CIVITAI_TUTORIAL.md`** — pairs naturally with this when
  inheriting Civitai community PNGs.
