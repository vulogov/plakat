# Utilities — doctor, models, inspect, gallery, clone, init, motion-adapter

The small commands that surround the generators: environment checks, cache
management, file inspection, project scaffolding, and recipe round-tripping.
None download a model unless noted.

## `doctor` — health-check the environment

```bash
plakat doctor                # build/runtime device match, HF cache size, ffmpeg + API tokens (presence only)
plakat doctor --benchmark    # synthetic conv2d/matmul/resize latency on your device (~2s, no download)
plakat doctor --verify       # actively probe configured HF specs resolve (hits cache/network)
plakat doctor --json         # structured report
```

Run it first when something's off — it tells you what your hardware can do
before you download 30 GB.

## `models` — manage the local HuggingFace cache

```bash
plakat models aliases        # every --model short-name plakat understands, grouped by family
plakat models search QUERY   # browse HuggingFace
plakat models size <repo>    # disk footprint before downloading
plakat models ls | rm | pull # list / delete / pre-fetch cached models
```

## `inspect` — list the tensors in a `.safetensors`

```bash
plakat inspect weights.safetensors   # every tensor name, dtype, shape
```

The first thing to reach for when a weight load fails — see what's *actually* in
the file vs what the model expected.

## `gallery` — build a Markdown gallery index

```bash
plakat gallery ./images --recursive --out GALLERY.md
```

Reads each PNG's embedded recipe (JSON sidecar, else the A1111 `parameters`
chunk) and emits a thumbnail grid + per-image prompt/settings. It's the tool
that builds plakat's own proof corpus index.

## `clone` — turn a generated PNG back into a command

```bash
plakat clone an-image.png    # prints a re-runnable `plakat generate …` line
```

Reads the embedded recipe and reconstructs the command — pairs with
[`metadata`](METADATA_TUTORIAL.md) (inspect → translate). Works on plakat
outputs, Civitai uploads, and A1111 Web UI PNGs.

## `init` — scaffold a starter project

```bash
plakat init my-project       # writes scenario.hjson + wildcards/ + a .gitignore
```

A quick way to start a [scenario](SCENARIOS_TUTORIAL.md) project.

## `motion-adapter` — inspect AnimateDiff adapters

```bash
plakat motion-adapter list           # the AnimateDiff repos plakat supports
plakat motion-adapter info <repo>    # an adapter's config + tensor breakdown
```

Companion to [`animate`](ANIMATE_TUTORIAL.md).
