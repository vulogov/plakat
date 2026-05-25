# Advanced prompting — attention, BREAK, inline LoRAs

The base `--prompt` flag covers the simple case: pass a sentence,
get an image. Three additional features let you write the kind of
elaborate prompts you see on Civitai LoRA cards and in
power-user-Reddit screenshots:

1. **Attention syntax** — `(red:1.5)` to emphasize specific words.
2. **BREAK keyword** — split a long prompt past CLIP's 77-token cap.
3. **Inline LoRA tags** — `<lora:civitai:12345:0.7>` to load LoRAs
   directly from the prompt instead of separate `--lora` flags.

All three originated in AUTOMATIC1111's WebUI; Civitai cards
assume you have them. Each works independently but they compose —
the combined power is the goal of this tutorial.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You
  should be comfortable with `--prompt`, `--seed`, `--steps`,
  `--lora`, and the relationship between seed and reproducibility.
- A working `plakat generate` against any SD-family or Flux model
  (the three features have different per-backbone scope; this
  tutorial calls out where each applies).

## 1. Attention syntax

CLIP encodes every prompt into a fixed-shape hidden state where
each token row carries the semantic signal for one word. Attention
syntax scales those rows: `(red:1.5)` multiplies "red"'s row by
1.5, telling the cross-attention "spend 50% more of your bandwidth
on this".

### Grammar

| Syntax | Weight |
|---|---|
| `(token)` | × 1.1 (default emphasis) |
| `((token))` | × 1.21 (nested) |
| `(token:1.5)` | × 1.5 (explicit) |
| `[token]` | × 1/1.1 ≈ 0.909 (default de-emphasis) |
| `[token:0.6]` | × 0.6 (explicit) |
| `\(`, `\)`, `\[`, `\]` | escaped — literal punctuation |

### Why it matters

```bash
# Without attention — "blue" might dominate "red" or vice versa
plakat generate "a fox in a red field of blue flowers" \
    --model sd15 --seed 42

# With attention — red explicitly wins
plakat generate "a fox in a (red:1.6) field of [blue] flowers" \
    --model sd15 --seed 42
```

Compare the two outputs at the same seed. The second has notably
more red dominance even though both contain the same words.

### Per-backbone support

| Backbone | Where the weights apply |
|---|---|
| SD 1.5 / SD 2.1 | CLIP-L penultimate hidden states |
| SDXL / SDXL-Turbo | CLIP-L penult + CLIP-G penult (both branches) |
| Flux | T5-XXL hidden states (CLIP-L on Flux is pooled-only) |
| SD3 / SD3.5 | CLIP-L penult + CLIP-G penult + T5 hidden |

The pooled CLIP outputs on SDXL / SD3 stay unweighted — pooling
collapses to a single row, so per-token weights have no target
there.

### Civitai LoRA cards lean on this heavily

Pull a random Civitai LoRA card and you'll see the example prompts
include `(masterpiece:1.2), (best quality:1.3), (1girl:1.1)` or
similar. These cards assume your CLI accepts the syntax — plakat
does, on every backbone.

```bash
plakat generate \
    "masterpiece, best quality, (1girl:1.2), (red hair:1.3), \
     [low quality, blurry, watermark]" \
    --model sd15
```

### Negative prompts

`--negative` accepts the same syntax. Boost unwanted-feature
suppression on overcrowded negatives:

```bash
plakat generate "portrait, soft window light" \
    --negative "(blurry:1.6), (low quality:1.4), [oversharp]" \
    --model sdxl --seed 42
```

### Sentencepiece caveat (Flux + SD3)

T5's sentencepiece tokenizer may split a segment into a slightly
different subtoken count when it appears in isolation vs inside a
longer string. The weight-per-resulting-subtoken contract is
preserved either way — the visual effect of `(token:1.5)` matches
AUTOMATIC1111's behaviour even when the subtoken count drifts by
one. Mostly invisible in practice; flagged here so you don't worry
about it.

## 2. BREAK — past CLIP's 77-token cap

CLIP truncates at 77 tokens (≈ 60-70 English words). Long prompts
silently lose their tail. The `BREAK` keyword splits the prompt
into chunks; each chunk gets its own 77-token CLIP context; the
per-chunk hidden states are concatenated along the sequence
dimension before the UNet's cross-attention consumes them. Cross-
attention has no max sequence length so the chunks just stack.

### Basic two-chunk

```bash
plakat generate \
    "a brutalist concrete whale poster, watercolor on rough \
     handmade paper, dramatic chiaroscuro lighting, cinematic \
     composition with off-center subject placement \
     BREAK \
     soft muted pastels in the negative space, hand-painted \
     feel with visible brushwork, no digital artifacts, 1970s \
     editorial illustration aesthetic" \
    --model sd15 --seed 42
```

That's two chunks of conditioning — 154 tokens total instead of
77. Without BREAK, everything after about "off-center subject
placement" would silently truncate.

### Three or more chunks

```bash
plakat generate \
    "subject and pose details here BREAK \
     environment and lighting details here BREAK \
     style and medium details here" \
    --model sd15
```

### Per-backbone support

| Backbone | BREAK behaviour |
|---|---|
| SD 1.5 / SD 2.1 | ✓ chunks the single CLIP-L encoder |
| SDXL / SDXL-Turbo | ✓ chunks both CLIP-L and CLIP-G; pooled `add_text_embeds` comes from chunk 0 |
| Flux | strips + warns (T5 has 256/512-token budget) |
| SD3 / SD3.5 | strips + warns (T5 budget + pooled `y` assumes single-chunk) |

On Flux and SD3 the BREAK keyword is **stripped** before encoding
(so you don't get a literal "BREAK" token in your prompt) and a
warning is logged. T5 already handles long prompts natively; the
workaround isn't needed.

### Word-boundary rules

BREAK is **case-sensitive** and **word-bounded**:

| Input | Split? |
|---|---|
| `prompt BREAK other` | ✓ |
| `prompt\nBREAK\nother` | ✓ |
| `prompt,BREAK,other` | ✓ |
| `BREAKING news` | ✗ (BREAK isn't a whole word) |
| `BREAKDOWN` | ✗ |
| `BREAKERS_v1` | ✗ (could be a LoRA name) |
| `break` / `Break` / `breakfast` | ✗ (case-sensitive) |

Adjacent BREAKs (`BREAK BREAK`) drop empty chunks. Leading or
trailing BREAK is fine — empty chunks before / after are dropped.

### Negative-prompt chunking

`--negative` is BREAK-aware too. CFG requires cond and uncond to
have the same total sequence length; whichever side has fewer
chunks is padded with empty chunks until they match.

```bash
plakat generate \
    "extensive positive prompt here BREAK with three chunks \
     BREAK because detail matters" \
    --negative "blurry, low quality" \
    --model sd15
```

Cond has 3 chunks → uncond gets padded to 3 chunks (the original
short negative + 2 empty ones).

## 3. Inline `<lora:>` tags

The third feature lets you load LoRAs **directly from the prompt**
instead of using separate `--lora` flags. Civitai LoRA cards embed
these inline; this matches the AUTOMATIC1111 convention.

### Grammar

| Form | Meaning |
|---|---|
| `<lora:NAME>` | weight = 1.0 |
| `<lora:NAME:0.7>` | explicit weight |
| `<lora:myfile.safetensors[:weight]>` | local file |
| `<lora:author/repo[#file.safetensors][:weight]>` | HuggingFace |
| `<lora:civitai:NNNNNN[#file][:weight]>` | Civitai by model id |
| `<lora:civitai-version:NNNNNN[:weight]>` | Civitai by version id |

The inner name follows the same grammar as the `--lora` flag.

### Single LoRA inline

```bash
# A Civitai watercolor LoRA at 70% strength, declared inline
plakat generate \
    "a fox in tall grass <lora:civitai:12345:0.7>" \
    --model sd15 --seed 42
```

After extraction the prompt becomes `"a fox in tall grass "` (with
a trailing space — A1111 leaves spacing alone) and the LoRA is
prepended to the model's adapter stack.

### Stacking multiple LoRAs

```bash
plakat generate \
    "<lora:style1:0.5> a fox in grass <lora:style2:0.3>" \
    --model sd15
```

Order in the prompt is preserved — `style1` loads first, `style2`
second. Later entries win on key collision during merge.

### Mixing inline + explicit

```bash
# Two LoRAs from the same workflow: one declared at the CLI,
# one inline. Both apply.
plakat generate "a fox <lora:style:0.7>" \
    --model sdxl \
    --lora civitai:99999:0.5
```

Inline tags land **first** in the stack; explicit `--lora` flags
land after (and so win on key collision — this matches the v0.16
per-task LoRA scenario behaviour).

### Negative-prompt tags

`<lora:>` tags in `--negative` are stripped silently. A1111
removes them without applying anything; plakat matches. LoRAs
modify the model itself, not the uncond branch's CLIP forward —
putting a tag in the negative is a no-op either way.

### Unbalanced tags are literal

A `<lora:` with no closing `>` is treated as a literal — no error,
just no extraction. Same robustness contract as wildcards and
attention syntax. Catches copy-paste mistakes without erroring out
a 30-minute scenario batch.

## 4. Putting all three together

```bash
plakat generate \
    "masterpiece, best quality, (1girl:1.2), watercolor portrait \
     <lora:civitai:12345:0.7> \
     BREAK \
     soft window light, golden hour, (cinematic depth:1.3), \
     hand-painted feel <lora:civitai:67890:0.5>" \
    --negative "(blurry:1.5), low quality, [oversharp], watermark" \
    --model sd15 --seed 42
```

What happens, in order:

1. **Wildcard expansion** — none here, but if you'd used
   `{red|blue}` it'd expand first.
2. **Enhance** — none here; if `--enhance` were set, the LLM would
   see the concrete (post-wildcard) prompt.
3. **`<lora:>` extraction** — two civitai LoRAs detected. Removed
   from the prompt; prepended to the LoRA stack.
4. **BREAK split** — one BREAK; two chunks. Each will get its own
   77-token CLIP context.
5. **Attention parsing** — `(1girl:1.2)`, `(cinematic depth:1.3)`,
   `(blurry:1.5)`, `[oversharp]` weights extracted per chunk.
6. **Per-chunk encode** — each chunk + weights through CLIP-L
   (and CLIP-G + T5 if on SDXL / SD3); hidden states concatenated.
7. **Negative** — same pipeline on the negative prompt; padded to
   match the cond chunk count.
8. **UNet denoise** — runs against the (now elaborated) cross-
   attention context.

## 5. Composition rules summary

| Feature | SD 1.5/2.1 | SDXL | Flux | SD3 |
|---|---|---|---|---|
| Attention `(token:1.5)` | ✓ | ✓ | ✓ (T5) | ✓ (all 3 encoders) |
| BREAK | ✓ | ✓ | strip+warn | strip+warn |
| Inline `<lora:>` | ✓ | ✓ | ✓ (BF16+GGUF) | partial (no LoRA on candle's SD3 path yet) |
| Wildcards `{a|b}` | ✓ | ✓ | ✓ | ✓ |

## 6. Where to next

- **`GENERATE.md`** in the reference docs — exhaustive per-flag
  documentation including edge cases not covered here.
- **`PROMPT_ENHANCER_TUTORIAL.md`** — `--enhance local | auto` for
  letting a small LLM rewrite your prompt before this whole
  pipeline runs.
- **`CIVITAI_TUTORIAL.md`** — the inline `<lora:civitai:...>`
  syntax pairs naturally with the Civitai browser + downloader
  workflow.
