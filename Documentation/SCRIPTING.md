# `plakat run` — Bund scripting (v0.21)

Reference for `plakat run SCRIPT.bund` (file mode) and
`plakat run --repl` (REPL mode). For a tutorial-style walkthrough
with composition patterns, see
[`Tutorials/SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md).
For the design rationale + the seven architectural decisions
locked in, see [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md).

## Modes

```
plakat run SCRIPT.bund          # File mode — eval the file, exit
plakat run --repl               # Interactive REPL on the same surface
plakat run --repl --out PATH    # REPL with a custom output dir
```

Both modes share the same `ScriptCtx` singleton + the same seven
`plakat.*` host words. One process invocation = one script eval
(no concurrent scripts in one process — bundcore's VM has no
per-eval isolation).

## CLI flags

| Flag | Default | Effect |
|---|---|---|
| `SCRIPT` (positional) | required without `--repl` | Path to a `.bund` file. Read + eval'd as a single string. |
| `--out PATH` | `./out` | Output directory for relative paths passed to `plakat.save`. Created if missing. |
| `--repl` | off | Start interactive REPL instead of evaling a file. The positional is ignored when set. |

Plus every global plakat flag (`--device`, `--verbose`,
`--cache-dir`, …) still applies.

## Language

Plakat scripts are written in **Bund** — a Forth-style,
stack-based language by `vulogov/Bund`. Tokens either push a
value or pop some values + run an operation. Top-of-stack is the
most recently pushed; words pop from the top.

```bund
// Comments use // (rest of line). `#` is NOT a comment in Bund.
1 2 +                       // → 3 on top of stack
"hello"                     // → "hello" on top
"sd15" plakat.load          // pop "sd15", load that alias
```

The language ships lambdas, list literals (`[ a b c ]`), named
symbols (`:foo`), control flow, and arithmetic — but plakat
**only** registers the bundcore VM primitives (no filesystem,
no network, no shell, no sudo). The full Bund stdlib is
deliberately excluded per RFC decision #2. The seven plakat host
words listed below are the only domain-specific surface scripts
can reach.

## Host words

Every host word is namespaced `plakat.*`. Stack-effect notation
follows Forth: `( in1 in2 -- out1 )` means "pops in1 and in2;
pushes out1." Top-of-stack is the rightmost input (popped first).

### `plakat.echo ( s -- s' )`

Phase 1 smoke word. Pulls a string, pushes
`"[out=<out_dir>] <s>"` back. Useful for verifying the
integration but not part of any real script.

### `plakat.load ( model-alias -- )`

Records the model alias subsequent words use. Idempotent: calling
twice with the same alias is a no-op; calling with a different
alias overwrites.

Supported v0.21 aliases:

| Alias | Model |
|---|---|
| `sd15` | Stable Diffusion 1.5 (community mirror) |
| `sd21` | SD 2.1 (`plakat.portrait` bails; everything else works) |
| `sdxl` | Stable Diffusion XL base 1.0 |
| `sdxl-turbo` | SDXL-Turbo |

**Bails with a "Phase 2b" pointer:**

- Every Flux variant (`flux-dev`, `flux-schnell`, `flux-kontext-dev`,
  `flux-fill-dev`, `flux-canny-dev`, `flux-depth-dev`, the GGUF +
  NF4 variants).
- SD3 / SD3.5 (`sd3-medium`, `sd35-medium`, `sd35-large`,
  `sd35-large-turbo`).

Full HuggingFace repo IDs (e.g.
`stable-diffusion-v1-5/stable-diffusion-v1-5`) are accepted and
classify the same way the CLI's `--model` does.

### `plakat.generate ( prompt -- handle )`

Text-to-image. Renders one image with the loaded model and the
current `GenerationConfig`. Pushes an integer handle (≥ 1)
addressing the rendered image in `ScriptCtx.images`. Handles are
permanent for the script's lifetime — saving doesn't free them.

Bails if no model has been loaded.

### `plakat.img2img ( prompt input -- handle )`

Re-imagine an existing image at `strength`. The `input` arg
accepts two shapes:

| `input` type | Effect |
|---|---|
| string | Filesystem path; read directly. |
| integer | Image handle; the registry image is materialised to a tempfile bound to the host-fn stack frame. |

Handle reuse is the load-bearing affordance of the cycle:
`generate → img2img` composes without disk round-trip. The
source is **not consumed** — the same handle can be re-used in
multiple subsequent words.

### `plakat.portrait ( prompt photo -- handle )`

Identity-preserving portrait via IP-Adapter-Plus-Face. Same
two `input` shapes as `img2img`. Identity strategy is auto-picked
from the loaded model:

| Loaded model | IP-Adapter strategy |
|---|---|
| SD 1.5 | `PlusFace` |
| SDXL / SDXL-Turbo | `PlusFaceSdxl` |
| SD 2.1 | bails (no shipped Plus-Face checkpoint) |

v0.21 ships single-photo portraits only. FaceID variants + multi-
photo identity blends + manual `face_bbox` / `face_landmarks`
arguments are deferred to v0.22.

The IP-Adapter weight (image-token contribution) is controlled
via `plakat.config.set "face_strength"` — see config below.

### `plakat.upscale ( handle scale -- handle )`

Lanczos-3 resize. `scale` must be the integer `2` or `4`. ML
upscaling (Real-ESRGAN) is deferred to v0.22.

No async bridge needed — the resize is pure CPU + image-crate
work. Width/height overflow are guarded via `checked_mul` — at
scale 4 a >1 GP input would silently wrap otherwise; bails loud.

### `plakat.save ( handle path -- )`

Writes the handle's image to `path`. Relative paths resolve
against the script's `--out` directory; absolute paths pass
through unchanged. The handle is **not** consumed — the same
image can be saved to multiple paths.

If the parent directory doesn't exist, it's created. Format is
inferred from the extension (image crate's defaults: `.png` /
`.jpg` / `.webp` / `.bmp` / `.tiff`).

### `plakat.config.set ( value key -- )`

Mutates one knob on the script's accumulated `GenerationConfig`.
The mutation persists across all subsequent `plakat.generate` /
`img2img` / `portrait` calls within one script.

Stack order: bottom = value, top = key string.

| Key | Type | Default | Validation |
|---|---|---|---|
| `steps` | int | 28 | > 0 |
| `guidance` | float | 7.5 | finite |
| `seed` | int | (random) | ≥ 0 |
| `width` | int | per-family | > 0, ÷8, ≤ 4096 |
| `height` | int | per-family | same as width |
| `negative` | string | `""` | passthrough |
| `scheduler` | string | `default` | parsed via `SchedulerKind::FromStr` (see below) |
| `strength` | float | 0.75 | finite, `[0, 1]` |
| `face_strength` | float | 0.8 | finite, `[0, 1]` |

Per-family size defaults: SD 1.5 / 2.1 → 512²; SDXL / SDXL-Turbo
→ 1024². Setting `width` or `height` flips an internal
`size_explicit` flag — once you set either dimension, the
script's pinned size applies even if the loaded model would
prefer a different default.

Scheduler names: `default | ddim | euler-a | euler | heun |
unipc | dpmpp-2m | unipc-exp | lcm | ddpm` (case-insensitive,
with the usual aliases). An unknown name bails with the full
supported list.

Setting a key the validator doesn't know about bails with the
list of supported keys. Setting an int key with a non-integer
float (`7.5 "steps" plakat.config.set`) bails — explicit type
mismatch instead of silent truncation. NaN / Inf / out-of-range
all bail with the offending value in the message.

## REPL meta-commands

`plakat run --repl` launches an interactive line editor against
a **persistent** Bund instance. State (stack contents, named
lambdas, config knobs) survives across lines. Three meta-commands
start with `.` (Forth convention):

| Command | Effect |
|---|---|
| `.q` / `.quit` | Exit the REPL. Saves history. |
| `.s` / `.stack` | Non-destructive workbench listing, bottom-up. `[0]` is the bottom of the stack; higher indices are closer to the top. |
| `.help` | List every `plakat.*` word + the meta-commands + a few examples. |

After every successful eval that left a value on top, the REPL
echoes `=> <value>` (Forth REPL convention). `Value: Clone` lets
us peek non-destructively.

Line editing: rustyline 18. Ctrl-D exits clean; Ctrl-C clears the
partial line and keeps the REPL alive. History persists at:

| Platform | Path |
|---|---|
| Linux | `~/.config/plakat/repl_history` |
| macOS | `~/Library/Application Support/ai.plakat.plakat/repl_history` |
| Windows | `%APPDATA%\plakat\plakat\config\repl_history` |

Eval errors don't bail the REPL — they print to stderr and the
prompt comes back. Fix the line and try again.

## Architecture notes

- **Singleton context.** `ScriptCtx` is a process-wide
  `OnceLock<RwLock<…>>` because bundcore host functions are bare
  `fn` pointers, not closures, and can't capture state. One
  script per process by construction. RFC decision #3.
- **Built our own VM.** `bundcore::STDLIB` is empty when the
  `bund` binary crate isn't a dep (which plakat isn't). `Bund::new()`
  + `init_lib()` ends up registering only the multistackvm
  primitives plus the seven `plakat.*` words — no filesystem /
  network / shell access reachable from user scripts. RFC
  decision #2.
- **Async bridge.** Bundcore is fully synchronous; plakat
  pipelines are `async`. Each pipeline-touching host word does
  `tokio::task::block_in_place(|| Handle::current().block_on(...))`.
  Requires a multi-threaded tokio runtime; `cli::run::run`
  provides one. RFC §3.3.
- **No pipeline cache.** Every `plakat.generate` / `img2img` /
  `portrait` call loads the model. Acceptable trade-off for the
  v0.21 MVP (the alternative was a non-trivial pipeline-cache
  abstraction); deferred to v0.22.

## v0.21 limitations

| Limitation | Tracking |
|---|---|
| SD-family only (no Flux / SD3) | v0.22 phase 2b |
| No SD 2.1 portrait | (no upstream Plus-Face SD 2.1 checkpoint) |
| Single-photo portrait; no FaceID | v0.22 |
| Lanczos upscale x2 / x4 only | v0.22 ML upscale + arbitrary scale |
| No LoRA / ControlNet / refiner words | v0.22 |
| Every `generate` reloads the model | v0.22 pipeline cache |
| Scenarios not exposed to scripts | by design — HJSON scenarios already exist |
| AnimateDiff | carried over from v0.20 deferred list |

## See also

- [`Tutorials/SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md)
  — narrative walkthrough with the composition patterns.
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) —
  design doc, locked decisions, phase plan.
- `vulogov/Bund` on GitHub — the language reference. Most of the
  Bund stdlib isn't exposed to plakat scripts, but the syntax +
  concepts are the same.
