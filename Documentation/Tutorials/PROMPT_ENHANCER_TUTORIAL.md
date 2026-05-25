# Prompt enhancement — let an LLM rewrite your prompts

Diffusion models follow detailed prompts better than terse ones.
"a knight" produces a generic knight; "a battle-worn knight in
weathered plate armor, low-angle shot, cinematic lighting from a
single window, oil-painted texture" produces what most users
actually want. Writing the detailed version every time is tedious.

`--enhance` automates it. plakat hands your prompt to an
instruction-tuned LLM, which rewrites it with concrete visual
detail (composition, lighting, medium, mood, style), then feeds
the rewritten prompt to the diffusion model.

Three providers ship in v0.18:

| Provider | Where the LLM runs | Cost | First-use setup |
|---|---|---|---|
| `deepseek` | DeepSeek API | metered API | `DEEPSEEK_API_KEY` env var |
| `gemini` | Google Gemini API | metered API | `GEMINI_API_KEY` env var |
| `local` | quantized LLM in-process | one-time ~1 GB download | none |
| `auto` | picks best available | varies | — |

This tutorial covers all four flavours, with the `local` path
(new in v0.18) as the focus — it's the one that needs no API key.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md).
- ~1 GB free for the Qwen2.5-1.5B-Instruct GGUF if you'll use
  `--enhance local`. ~230 MB if you only use the SmolLM2 fallback.
- For the API providers, an account at deepseek.com or
  ai.google.dev with a working key.

## 1. The simplest case

```bash
# Without enhancement — your prompt goes through verbatim
plakat generate "a knight" --model sd15 --seed 42

# With local enhancement (no API key needed)
plakat generate "a knight" --model sd15 --seed 42 --enhance local
```

Compare the two outputs. The second was generated against a
rewritten prompt that the LLM elaborated automatically. Same seed,
same model, vastly more detail in the result.

What plakat shows in the terminal:

```
  enhanced: a stoic knight in weathered medieval plate armor,
            standing in a moody twilight courtyard, low-angle
            cinematic shot, dramatic chiaroscuro lighting, oil-
            painted texture, intricate engraved details
```

That string is what got tokenized + encoded — your original
"a knight" never reached the diffusion model.

## 2. `--enhance auto`

When you don't care which provider runs, `auto` picks based on
what's available:

```bash
plakat generate "a knight" --enhance auto
```

Priority:
1. **DeepSeek** if `DEEPSEEK_API_KEY` is set
2. **Gemini** if `GEMINI_API_KEY` is set
3. **local** (always works — no API key required)

This makes `--enhance auto` the right default for shell aliases
and CI: on a developer machine with an API key, the cloud provider
fires (faster, better quality). On a sandbox machine without keys,
the local model fires. No reconfiguration needed across
environments.

## 3. Local enhancer — model choice

The default local model is **Qwen2.5-1.5B-Instruct** at Q4_K_M
quantization (~1 GB). Best quality of the shipped options.

```bash
# Same as `--enhance local`
plakat generate "..." --enhance local:qwen2.5-1.5b
```

Smaller fallback for memory-constrained machines:

```bash
# ~230 MB. Quality is "good enough" — adds adjectives and
# lighting cues. Less elaborate than Qwen2.5.
plakat generate "..." --enhance local:smollm2-360m
```

The fallback is useful when:
- Disk space is tight (the Q4_K_M GGUF lives in the HF cache)
- You're running on CPU only and the per-prompt latency matters
- You're doing high-throughput scenarios (the cache amortises load
  across tasks, but each enhance call is still ~3-5s on CPU)

### Picking a model: rule of thumb

| Your setup | Recommended |
|---|---|
| Metal / CUDA GPU available | `qwen2.5-1.5b` (default) |
| CPU only, ample RAM (≥ 8 GB) | `qwen2.5-1.5b` |
| CPU only, tight RAM | `smollm2-360m` |
| You already pay for DeepSeek / Gemini | Skip local — use the API |

## 4. First-run behaviour

The first time you invoke `--enhance local`, plakat downloads the
GGUF + tokenizer from HuggingFace and caches them. Subsequent runs
hit the cache directly.

```bash
# First run — visible download (~1 GB for Qwen, ~230 MB for SmolLM)
$ plakat generate "..." --enhance local
  → Downloading qwen2.5-1.5b-instruct-q4_k_m.gguf ... 1.02 GiB
  → Downloading tokenizer.json ... 11.4 MiB
  → enhanced: ...

# Second run — instant
$ plakat generate "..." --enhance local
  → enhanced: ...
```

The cache lives at `~/.cache/huggingface/hub/` (or wherever
`HF_HOME` / `PLAKAT_CACHE_DIR` points). Inspect via
`plakat doctor`'s "HF cache disk usage" section.

## 5. Reproducibility

The local enhancer uses **greedy decoding** by default. Same input
prompt + same model = same enhanced output, every time.

```bash
# Run this twice — the enhanced prompt is byte-identical
plakat generate "a knight" --enhance local --seed 42
plakat generate "a knight" --enhance local --seed 42
```

That matters for reproducible scenarios — your batch will produce
the same images across machines that have the same model cached.

API providers (DeepSeek, Gemini) are NOT reproducible — they
sample with temperature and may return different rewrites for the
same input. If reproducibility is critical, use `local`.

## 6. Scenarios — load once, enhance many

Scenarios with `enhancer: local` pay the GGUF load cost once and
reuse the model across every task:

```hjson
{
  model: sd15
  out: ./out
  enhancer: local

  scene: [
    { name: forest,  prompt: "a fox in a forest" }
    { name: desert,  prompt: "a fox in a desert" }
    { name: city,    prompt: "a fox in a city alley" }
  ]
}
```

Each task's prompt gets rewritten before generation. The model
loads on the first task and stays in memory for the rest — the
process-wide cache is keyed by `(alias, device)` so a single
scenario run pays at most one load cost.

Per-task override:

```hjson
{
  enhancer: local
  ...
  scene: [
    { name: forest, prompt: "a fox in a forest" }
    { name: raw,    prompt: "exact prompt I wrote", enhance: false }
  ]
}
```

`enhance: false` on a task skips the rewrite for that one task.
Useful when you've already hand-tuned a particular prompt.

## 7. API providers

If you want the speed + quality of a hosted model:

```bash
export DEEPSEEK_API_KEY=sk-...
plakat generate "a knight" --enhance deepseek
```

```bash
export GEMINI_API_KEY=...
plakat generate "a knight" --enhance gemini
```

Same system prompt as the local enhancer. Faster than local-on-CPU
(network round-trip is usually < 1s). Doesn't reproduce
deterministically across runs.

## 8. Bad output? Fall back gracefully

The local LLM occasionally returns:
- A refusal ("I cannot help with that request")
- An empty response
- A "Here's the rewritten prompt:" preamble before the actual rewrite
- Surrounding quotes around the prompt

plakat sanitizes all of these:

- Preambles like `Here's the rewritten prompt:` / `Sure!` /
  `Rewritten:` are stripped.
- Surrounding quotes (straight or curly) are stripped.
- Role tags (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`) are
  stripped.
- Refusal prefixes (`I cannot`, `I'm sorry`, `As an AI`, ...) cause
  the enhancer to **fall back** to the original un-enhanced prompt.
  A warning logs:

  ```
  WARN local enhancer (qwen2.5-1.5b) returned a refusal or empty
       output — falling back to the un-enhanced prompt
  ```

Falling back is the right call — feeding a literal "I cannot help
with that" into the diffusion encoder poisons the output. The
user's original prompt is the safe baseline.

## 9. When NOT to enhance

The enhancer rewrites your prompt. If you've **already**
hand-tuned a detailed prompt (e.g. for an existing scenario
batch), enhancement will alter it. That can be unwanted.

Skip enhancement when:
- You're iterating on a specific prompt and need exact control
- You're using A1111 attention syntax — the enhancer may drop or
  rearrange `(token:1.5)` markers
- You're using BREAK or inline `<lora:>` tags — the enhancer may
  mangle them

For these cases, **leave `--enhance` off entirely**, or write
prompts that pass cleanly through the LLM (it preserves most
content; it adds detail rather than restructuring).

## 10. Limitations

- **No streaming output**. The decode loop runs to completion
  before plakat continues. ~96 new tokens cap; on CPU that's
  ~3-5s for Qwen2.5-1.5B, ~1-2s for SmolLM2-360M.
- **Single system prompt**. The system prompt is built into the
  binary (`"You rewrite text-to-image prompts..."`); no flag yet
  to override at the CLI. Lands in a follow-up if there's demand.
- **No temperature flag yet**. Greedy decoding only on the local
  path. Bumping temperature for variety is a follow-up.
- **No safety filter**. The model itself may refuse some prompts;
  plakat doesn't add a separate filter.

## Where to next

- **`ADVANCED_PROMPTING_TUTORIAL.md`** — attention syntax, BREAK,
  inline LoRA tags. These layer on top of the enhancer (or work
  independently when `--enhance` is off).
- **`GENERATE.md`** in the reference docs — the
  `--enhance <PROVIDER>` flag reference.
- **`GENERATE_TUTORIAL.md`** §15 — how the scenario `enhancer:`
  key works.
