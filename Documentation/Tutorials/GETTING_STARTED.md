# Getting started with plakat — `plakat init` to your first image

The fastest path from a fresh checkout to a rendered image. **v0.20** added
`plakat init`, which writes a runnable starter project: a `scenario.hjson`
configured for the default open-weights model, a `wildcards/` directory
with three small example files, and a focused `.gitignore`. Targets SD 1.5
+ on-device LLM enhancer so first-run users with no HF token + no API key
can generate end-to-end.

## Prerequisites

- plakat installed; `plakat --help` works.
- ~3 GB free disk for the SD 1.5 weights on first run (one-time).
- ~1 GB free for the local LLM enhancer on first use (one-time).
- A device backend that works (auto-detected: CUDA / Metal / CPU).

No HF token. No API keys. The starter scenario uses only ungated
defaults.

## 1. Bootstrap a project

```bash
plakat init ./my-project
```

You get:

```text
my-project/
├── .gitignore             # keeps out/ from being committed
├── scenario.hjson         # two-task starter scenario
└── wildcards/
    ├── lighting.txt       # three lighting options
    ├── style.txt          # three painting styles
    └── subject.txt        # three subjects
```

The scenario file itself is short and editable — it uses the scenario
engine's scene/weather catalog (not the wildcard files; those are for
`plakat generate --wildcard-dir`).

## 2. Dry-run to validate

Before generating anything (which involves downloading weights), run
the scenario with `--dry-run`. This parses the HJSON, resolves prompts,
verifies the enhancer config, and prints what *would* happen — without
loading the model or calling the LLM:

```bash
plakat scenario ./my-project/scenario.hjson --dry-run
```

You should see two tasks queued (`forest_golden` + `harbor_rain`) with
pre-resolved prompts. If something's wrong (typo in HJSON, missing
field), it surfaces here in <1 second rather than 30 seconds into a
model load.

## 3. Generate

Drop the `--dry-run`:

```bash
plakat scenario ./my-project/scenario.hjson
```

What happens on first run:

1. SD 1.5 weights download to `~/.cache/plakat/` (~3 GB, one-time).
2. The local Qwen2.5-1.5B GGUF enhancer downloads on first call
   (~1 GB, one-time).
3. Each task's prompt is enhanced once, then rendered.
4. PNGs land in `./my-project/out/<task-name>/`.

Subsequent runs reuse both caches — no downloads, no setup time.

## 4. Iterate

The starter scenario is meant to be modified. Common edits:

- **Add a task.** Copy one of the two existing task blocks and edit
  the `name`, `scene`, `weather`, and `prompt`. The scenario engine
  cross-references `scene` and `weather` against the top-level
  catalogs by name.
- **Add a scene or weather entry.** Both are simple `{ name, prompt }`
  pairs in the top-level arrays. New entries become valid choices for
  every task's `scene:` / `weather:` field.
- **Bump quality.** Edit `steps: 28` → `steps: 50` for higher quality
  at ~2× the runtime. Edit `guidance: 7.0` to taste (5-8 is
  reasonable for SD 1.5).
- **Switch models.** `model: sd15` → `model: sdxl` for higher
  resolution (+~7 GB download on first use, larger working size).
  `model: flux-dev` works too but requires `HF_TOKEN` env var
  (gated) + ~24 GB of free memory.

After each edit, the same workflow applies: `--dry-run` to validate,
then drop the flag to actually generate.

## 5. The `wildcards/` directory

The wildcard files aren't used by the starter scenario — they're for
single-shot `plakat generate` calls. Try this from inside the project
directory:

```bash
plakat generate "__subject__ in a forest, __style__, __lighting__" \
    --model sd15 --wildcard-dir ./wildcards \
    --out ./out/wildcards
```

Each `__name__` token expands to a random non-blank line from
`wildcards/name.txt`. Wildcards are an alternative to the scenario
engine's catalog mechanism — useful when you want a one-shot
randomized generation rather than a structured cross-product.

## 6. Customizing the init template

`plakat init` writes a fixed set of files but exposes two flags:

```bash
# Just the scenario, nothing else (e.g. adding to an existing repo)
plakat init . --minimal

# Overwrite an existing scenario.hjson you regret editing
plakat init . --force
```

By default `plakat init` errors if any target file already exists,
so it won't silently clobber work in progress.

## Where to next

- **[`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md)** — single-shot
  `plakat generate` with full flag coverage.
- **[`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md)** — going deeper
  into the scenario engine: personas, per-task overrides, partial
  reruns, character-series workflows.
- **[`PROMPT_ENHANCER_TUTORIAL.md`](PROMPT_ENHANCER_TUTORIAL.md)** —
  what the `local` enhancer the starter uses actually does, plus how
  to swap to DeepSeek / Gemini for higher-quality rewrites.
