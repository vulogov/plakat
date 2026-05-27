# `plakat run` — Bund scripting (v0.24)

Reference for `plakat run SCRIPT.bund` (file mode) and
`plakat run --repl` (REPL mode). For a tutorial-style walkthrough
with composition patterns, see
[`Tutorials/SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md).
Design RFCs:
[`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) (foundations),
[`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md) (the 7-namespace expansion),
[`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) (the v0.22 deferrals closed in v0.23),
[`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md) (persona depth + scripting completion).

## Modes

```
plakat run SCRIPT.bund          # File mode — eval the file, exit
plakat run --repl               # Interactive REPL on the same surface
plakat run --repl --out PATH    # REPL with a custom output dir
```

Both modes share the same `ScriptCtx` singleton + the same 42
`plakat.*` host words. One process invocation = one script eval
(no concurrent scripts in one process — bundcore's VM has no
per-eval isolation).

## CLI flags

| Flag | Default | Effect |
|---|---|---|
| `SCRIPT` (positional) | required without `--repl` | Path to a `.bund` file. Read + eval'd as a single string. |
| `--out PATH` | `./out` | Output directory for relative paths passed to `plakat.save`. Created if missing. |
| `--repl` | off | Start interactive REPL instead of evaling a file. The positional is ignored when set. |
| `--device DEV` | `auto` | Override device: `auto | cuda[:N] | metal | cpu`. |
| `--cache-dir PATH` | env-driven | HuggingFace cache override. |

Plus every global plakat flag (`--verbose`, etc.) still applies.

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
deliberately excluded per v0.21 RFC decision #2.

## Host words (42 total)

Stack-effect notation follows Forth: `( in1 in2 -- out1 )` means
"pops in1 and in2; pushes out1." Top-of-stack is the rightmost
input (popped first).

### Core image surface (v0.21 + cache-aware v0.22 + v0.23 inpaint + v0.24 outpaint/stylize)

| Word | Stack effect | Notes |
|---|---|---|
| `plakat.echo` | `( s -- s' )` | Phase 1 smoke word. |
| `plakat.load` | `( alias -- )` | Resolve + cache pipeline. v0.22: cache-aware — same alias reuses the loaded model. v0.23: SD-family `plakat.load` warms the t2i slot by default. |
| `plakat.generate` | `( prompt -- handle )` | Text-to-image. v0.23: uses the SdT2i slot (refiner + clip_skip ready). |
| `plakat.img2img` | `( prompt input -- handle )` | `input` = path string OR image handle. |
| `plakat.inpaint` | `( prompt input mask -- handle )` | **v0.23 phase 5.** Mask-guided img2img. `input` = path or handle; `mask` = path. Honours `mask_feather` + `mask_invert` config keys. SD-family + SD3 wired; **v0.24 phase 9** adds Flux (requires `flux-fill-dev` variant). |
| `plakat.outpaint` | `( prompt input expand-spec -- handle )` | **v0.24 phase 4.** Extend past borders. `expand-spec`: `"expand=N"` (all 4 sides) OR `"left=L,right=R,top=T,bottom=B"`. Replicates edges + builds mask + dispatches to inpaint. SD-family + SD3. |
| `plakat.portrait` | `( prompt -- handle )` | **v0.24 phase 1**: photos come from `plakat.portrait.photo.add` stack (was `( prompt photo -- handle )` in v0.23). IP-Adapter-Plus-Face. SD-family only. |
| `plakat.stylize` | `( subject style -- handle )` | **v0.24 phase 6.** IP-Adapter style transfer. No prompt (CLI parity). Strength from `config.strength`. SD 1.5 only. |
| `plakat.upscale` | `( handle scale -- handle )` | Lanczos-3, integer `2` or `4`. |
| `plakat.save` | `( handle path -- )` | Relative paths resolve under `--out`. |
| `plakat.config.set` | `( value key -- )` | Mutate one knob. Stack order: value below, key on top. |

#### Supported aliases (v0.22)

All three families are first-class. The cache holds one pipeline
at a time and reloads when the alias family changes.

| Family | Aliases |
|---|---|
| SD 1.5 / 2.1 / SDXL | `sd15`, `sd21`, `sdxl`, `sdxl-turbo` |
| Flux | `flux-dev`, `flux-schnell`, `flux-kontext-dev`, `flux-fill-dev`, `flux-canny-dev`, `flux-depth-dev` (+ GGUF / NF4 variants) |
| SD3 / SD3.5 | `sd3-medium`, `sd35-medium`, `sd35-large`, `sd35-large-turbo` |

Full HF repo IDs are accepted and classify the same way as `--model`.

### `plakat.lora.*` — LoRA stack (phase 4)

| Word | Stack effect |
|---|---|
| `plakat.lora.add` | `( spec scale -- )` |
| `plakat.lora.clear` | `( -- )` |
| `plakat.lora.list` | `( -- s_1 … s_n n )` |

`spec` accepts the same grammar as `--lora` (local path,
`civitai:N`, `civitai-version:N`, HF `repo#file`). Mutations
invalidate the pipeline cache: the next image-producing word
rebuilds with the current LoRA set merged in.

### `plakat.controlnet.*` — ControlNet stack (phase 5 + v0.23 phases 6–7)

| Word | Stack effect | Behaviour |
|---|---|---|
| `plakat.controlnet.add` | `( kind image-path -- )` | Pre-rendered conditioning map. |
| `plakat.controlnet.annotate` | `( kind from-path -- )` | Auto-annotate from a regular image. SD-family only. |
| `plakat.controlnet.spec` | `( spec-string -- )` | Full grammar: `kind[:strength][:start][:end][@image=PATH][@from=PATH]`. |
| `plakat.controlnet.clear` | `( -- )` | |
| `plakat.controlnet.list` | `( -- s_1 … s_n n )` | |

**SD-family** flows through `Request.controls` at generate time
(per-call); stack mutations don't invalidate the cache.

**Flux + SD3** (v0.23 phases 6–7, v0.24 phase 8): the CN stack
threads into `LoadRequest.controlnets` at pipeline-load time.
Stack mutations call `mark_controlnets_changed` which drops the
Flux/SD3 slot (SD-family slots stay intact).

**Both `image=` and `from=` specs work** as of v0.24. `image=`
bakes the path in at load. `from=` lazy-annotates at first
generate using that call's width/height; the annotated PNG
caches on a per-pipeline tempdir (cleared when the pipeline
slot is invalidated). Dim mismatches re-annotate.

Single-CN: Union Pro v2 (Flux) / InstantX SD3 family by kind.
Multi-CN: residuals sum.

### `plakat.refiner.*` — SDXL refiner toggle (phase 6 + v0.23 phase 2)

| Word | Stack effect |
|---|---|
| `plakat.refiner.enable` | `( -- )` |
| `plakat.refiner.disable` | `( -- )` |

**Fully wired in v0.23.** Toggle drives `t2i::Pipeline.load`'s
`use_refiner` flag at the SdT2i cache slot — `plakat.generate` on
SDXL loads the official SDXL refiner UNet (~6 GB download on
first run) and splits the schedule at `refiner_frac` (default
0.8 = last 20% of steps). Non-SDXL aliases silently downgrade
with a warn (matches CLI `--refiner` behaviour). Toggling
mid-script invalidates the SdT2i slot so the next generate
reloads with the new state. Same-model polish via
`refine_steps` / `refine_strength` also wired.

### `plakat.adetailer.*` — face refinement (phase 7)

| Word | Stack effect |
|---|---|
| `plakat.adetailer.enable` | `( -- )` |
| `plakat.adetailer.disable` | `( -- )` |

Post-process: SCRFD detects faces; an img2img pass refines each
face crop; feather-composited back. Reuses the cached
`portrait::Pipeline::core()` so no second SD load. SD-family
only — Flux + SD3 bail.

### `plakat.hires.*` — hires fix (phase 8)

| Word | Stack effect |
|---|---|
| `plakat.hires.enable` | `( -- )` |
| `plakat.hires.disable` | `( -- )` |

Post-process: upscale (classical or Real-ESRGAN) + img2img refine
at moderate strength. Reuses the cached SD backbone. SD-family
only. When combined with `plakat.artefact.*` (non-empty stack),
`plakat.generate` bails — mirrors the CLI's
`--hires-fix` + `--artefact` mutual-exclusion gate.

### `plakat.artefact.*` — compose + blend (phase 9)

| Word | Stack effect |
|---|---|
| `plakat.artefact.add` | `( spec -- )` — `NAME[@ZONE[:SCALE]]` grammar |
| `plakat.artefact.clear` | `( -- )` |
| `plakat.artefact.list` | `( -- s_1 … s_n n )` |
| `plakat.artefact.blend.enable` | `( -- )` |
| `plakat.artefact.blend.disable` | `( -- )` |

Post-process pipeline: alpha-composite each artefact (in add
order), then optionally run a masked-img2img blend pass over the
zones. Same grammar as the CLI's `--artefact` flag (full-object
overrides remain HJSON-only). SD-family only.

### `plakat.style.*` — style catalog (v0.23 phase 4)

| Word | Stack effect |
|---|---|
| `plakat.style.apply` | `( id -- )` — pick a catalog style by id |
| `plakat.style.detect` | `( photo -- )` — pick by CLIP-H matching against a reference photo |
| `plakat.style.clear` | `( -- )` |
| `plakat.style.list` | `( -- s_1 … s_n n )` — push every catalog id + count |

State (`style_id` / `style_ref`) lives on `ScriptCtx`. Resolution
runs lazily at `plakat.generate` request-build time: catalog
LoRAs **override** the user LoRA stack for the load (CLI parity
with `--style ID` / `--style-ref PATH`); trigger phrase prepends
to the prompt; `negative_extras` appends to the negative via
`combine_negative`. Subsequent generates with the same style
cache-hit the style-laden pipeline. Stack mutations invalidate
the SD cache slots via `mark_loras_changed`. SD-family only —
Flux + SD3 bail. Tune with the `style_catalog` (path) +
`style_strength` ([0,1]) config keys.

### `plakat.enhance` — prompt rewriter (phase 10)

| Word | Stack effect |
|---|---|
| `plakat.enhance` | `( prompt -- enhanced )` |

Pure prompt transformer. Dispatches to the configured provider
(see `enhance_provider`). Greedy by default — reproducible. The
local LLM cache is global, so back-to-back enhance calls pay the
GGUF load cost once. `enhance_keep_original` joins the rewrite
with the original via `BREAK` on SD-family models; Flux + SD3
no-op (T5 ignores BREAK).

### `plakat.portrait.photo.*` — multi-photo identity (v0.24 phase 1)

| Word | Stack effect |
|---|---|
| `plakat.portrait.photo.add` | `( path-or-handle weight -- )` |
| `plakat.portrait.photo.clear` | `( -- )` |
| `plakat.portrait.photo.list` | `( -- s_1 … s_n n )` |

Multi-photo identity-blending stack consumed by
`plakat.portrait ( prompt -- handle )`. `path-or-handle` is a
string path OR an integer image handle (handle path is
materialised to a tempfile that survives the script's lifetime).
`weight` is `-1.0` for auto-fill (CLI's default) or `>= 0.0` for
an explicit weight. Weights normalise sum-to-1 at request-build
time. Mutations don't invalidate the cache (photos are
per-call). **`plakat.portrait` requires a non-empty stack** —
bails loudly otherwise.

### `plakat.embedding.*` — Textual Inversion (v0.24 phase 5)

| Word | Stack effect |
|---|---|
| `plakat.embedding.add` | `( spec -- )` |
| `plakat.embedding.clear` | `( -- )` |
| `plakat.embedding.list` | `( -- s_1 … s_n n )` |

Spec grammar matches the CLI's `--embedding`:
`"path[:trigger][:scale]"` (path or HF repo as source). Threaded
into `t2i::LoadRequest.embeddings` at SdT2i load time.
Mutations call `mark_loras_changed` (TI lives load-time
alongside LoRAs). **Effective only on `plakat.generate`'s SdT2i
path** — `plakat.img2img` + `plakat.portrait` use
`portrait::Pipeline`, which doesn't take embeddings (CLI parity:
`cli::img2img` + `cli::portrait` don't expose `--embedding`
either).

### `plakat.metadata.read` — sidecar reader (v0.24 phase 7)

| Word | Stack effect |
|---|---|
| `plakat.metadata.read` | `( path -- k_1 v_1 … k_n v_n n )` |

Reads the JSON sidecar plakat writes alongside every generated
PNG (the structured form of the A1111 `parameters` tEXt chunk).
Pushes every populated field as a `(key, value)` pair of strings
plus a count.

Required fields always present: `prompt`, `model`, `seed`,
`steps`, `guidance`, `scheduler`, `width`, `height`, `generator`.
Optional fields push only when set/non-empty: `negative`,
`loras`, `lora_scale`, `clip_skip`, `controls`, `refiner_frac`,
`mode`, `strength`, plus per-key `extras`.

Bails on missing sidecar / empty path / bad JSON. Write deferred
to v0.25.

## Post-process composition order

When `plakat.generate` runs with multiple post-process toggles
enabled, the SD-family path applies them in this order on the
rendered image:

1. **Artefacts** (compose + optional blend) — if `artefacts`
   non-empty.
2. **Hires fix** — if `hires_enabled`. *Mutually exclusive with
   artefacts; the generate call bails when both are set.*
3. **ADetailer** — if `adetailer_enabled`.

The CLI's same gate (`--hires-fix` rejects `--artefact*`) is
mirrored at the script layer.

## Config keys (v0.22)

All set via `plakat.config.set` (value below, key on top of
stack). Setting an unknown key bails with the full supported
list. Type mismatches bail.

### Core (v0.21)

| Key | Type | Default | Range |
|---|---|---|---|
| `steps` | int | 28 | > 0 |
| `guidance` | float | 7.5 | finite |
| `seed` | int | random | ≥ 0 |
| `width`, `height` | int | per-family | > 0, ÷8, ≤ 4096 |
| `negative` | string | `""` | passthrough |
| `scheduler` | string | `default` | parsed via `SchedulerKind` |
| `strength` | float | 0.75 | finite, `[0, 1]` |
| `face_strength` | float | 0.8 | finite, `[0, 1]` |

### Flux D-keys (phase 2)

| Key | Type | Default | Notes |
|---|---|---|---|
| `quantize_t5` | bool | false | T5 INT8 quantisation |
| `quant_level` | string | unset | Flux backbone quant (Q2_K…Q8_0) |
| `t5_quant_level` | string | unset | T5 quant level |
| `fast` | string | unset | bundled preset: `lcm-4`, `lcm-8`, `hyper-8`, `lightning-8`, `turbo-1` |
| `kontext_bucket` | bool | false | Honour reference image's aspect bucket for FLUX.1-Kontext |

### Tiled / phase 3

| Key | Type | Default | Notes |
|---|---|---|---|
| `tiled` | bool | false | SD3-tiled denoise |
| `tile_size` | int | 1024 | multiple of 16 |
| `tile_stride` | int | 768 | tile overlap |

### LoRA (phase 4)

| Key | Type | Default | Range |
|---|---|---|---|
| `lora_scale` | float | 1.0 | `[0, 2]` |

### Refiner / style (phase 6)

| Key | Type | Default | Range |
|---|---|---|---|
| `refine_steps` | int (Option) | unset | `(0, 500]` (same-model polish) |
| `refine_strength` | float | 0.3 | `[0, 1]` |
| `refiner_frac` | float | 0.8 | `[0, 1]` — wired in v0.23 phase 2 (SDXL refiner UNet split) |
| `style_strength` | float | 1.0 | `[0, 1]` — wired in v0.23 phase 4 (catalog LoRA multiplier) |
| `style_catalog` | string | empty | v0.23 phase 4. Catalog directory; empty → `assets/style_catalog` |

### ADetailer (phase 7)

| Key | Type | Default | Notes |
|---|---|---|---|
| `adetailer_strength` | float | 0.4 | `[0, 1]` |
| `adetailer_padding` | float | 0.25 | bbox expansion per side |
| `adetailer_feather` | float | 0.25 | mask feather fraction |
| `adetailer_confidence` | float | 0.5 | SCRFD threshold |
| `adetailer_size` | int | 512 | working size, /8 |
| `adetailer_prompt` | string | `"detailed face…"` | per-face prompt |

### Hires (phase 8)

| Key | Type | Default | Notes |
|---|---|---|---|
| `hires_scale` | float | 2.0 | `(1, 4]` |
| `hires_strength` | float | 0.5 | `[0, 1]` |
| `hires_upscaler` | string | `lanczos` | classical or Real-ESRGAN method |
| `hires_steps` | int (Option) | unset | falls back to `steps` |

### Artefact (phase 9)

| Key | Type | Default | Notes |
|---|---|---|---|
| `artefact_library` | string | empty | path override; empty → `assets/artefact_library` |
| `artefact_blend_strength` | float | 0.3 | `[0, 1]` |
| `artefact_smart_zones` | bool | false | Depth-Anything-V2-Small placement |

### Enhance (phase 10)

| Key | Type | Default | Notes |
|---|---|---|---|
| `enhance_provider` | string | `auto` | `auto`/`deepseek`/`gemini`/`local`/`local:<alias>` |
| `enhance_temp` | float (Option) | unset | `[0, 2]` (local only) |
| `enhance_max_tokens` | int (Option) | unset | `(0, 1024]` (local only) |
| `enhance_cache` | bool | false | SHA-256 disk cache for local |
| `enhance_system` | string | empty | path to custom system prompt |
| `enhance_keep_original` | bool | false | BREAK-join rewrite + original (SD-family) |

### Misc (phase 11)

| Key | Type | Default | Notes |
|---|---|---|---|
| `aspect` | string | empty | `W:H` (e.g. `16:9`); derives size when `width`/`height` unset |
| `base` | int | 768 | shorter-side resolution for `aspect`, /8 |
| `mask_feather` | int | 8 | img2img mask edge feather (px) |
| `mask_invert` | bool | false | img2img mask polarity |
| `clip_skip` | int | 1 | `[1, 12]` — wired in v0.23 phase 3 via t2i::Pipeline.encode_prompt. SD 1.5 / SD 2.1 only |
| `wildcard_dir` | string | empty | `__name__` wildcard directory |
| `negative_preset` | string | empty | `photo`/`painting`/`anime`/`cinematic` (combined with `negative`) |

### Persona (v0.24 phases 2–3)

| Key | Type | Default | Notes |
|---|---|---|---|
| `face_bbox` | string | empty | CSV `"x0,y0,x1,y1"` (4 floats in [0,1], x0<x1, y0<y1). Mirrors `--face-bbox`. Empty clears. |
| `face_landmarks` | string | empty | CSV 10 floats: `"LX,LY,RX,RY,NX,NY,MLX,MLY,MRX,MRY"`. Mirrors `--face-landmarks`. Takes precedence over `face_bbox`. |
| `identity_kind` | string | empty | One of `plus-face`, `plus-face-sdxl`, `face-id`, `face-id-sdxl` (plus aliases). Empty → auto-pick by alias. Overrides the v0.22 auto-pick rule for SD-family `plakat.portrait`. |

## REPL meta-commands

`plakat run --repl` launches an interactive line editor against
a **persistent** Bund instance. State (stack, named lambdas,
config knobs) survives across lines.

| Command | Effect |
|---|---|
| `.q` / `.quit` | Exit. Saves history. |
| `.s` / `.stack` | Non-destructive workbench listing, bottom-up. |
| `.help` | List every `plakat.*` word + the meta-commands. |

History persists at `~/.config/plakat/repl_history` (Linux),
`~/Library/Application Support/ai.plakat.plakat/repl_history` (macOS),
or `%APPDATA%\plakat\plakat\config\repl_history` (Windows).

## Architecture notes

- **Singleton context.** `ScriptCtx` is a process-wide
  `OnceLock<RwLock<…>>`. v0.21 RFC decision #3.
- **Pipeline cache (v0.22 phase 1).** `ScriptCtx::loaded` holds
  one `LoadedPipeline` enum variant
  (`SdFamily(portrait::Pipeline)` / `Flux(flux::Pipeline)` /
  `Sd3(sd3::Pipeline)`). Same-alias reuse skips the model load;
  family changes or LoRA mutations drop the cached pipeline.
- **Restricted stdlib.** `bundcore::STDLIB` is empty when the
  `bund` binary crate isn't a dep. Plakat ships only the 28
  `plakat.*` words on top of the bundcore VM primitives — no
  filesystem / network / shell access from user scripts. v0.21
  RFC decision #2.
- **Async bridge.** Bundcore is fully synchronous; plakat
  pipelines are `async`. Each pipeline-touching host word does
  `tokio::task::block_in_place(|| Handle::current().block_on(...))`.

## v0.24 limitations / deferred to v0.25+

All v0.23 carries closed in v0.24. The remaining items are
longer-term:

| Limitation | Tracking |
|---|---|
| AnimateDiff | new architecture (motion-adapter weights + temporal-attention); long-running carry from v0.20 — natural v0.25 big swing |
| SD3 / SD3.5 animate | 3-encoder lerp + MMDiT integrator (v0.20+ carry) |
| Real-ESRGAN ML upscaling in `plakat.upscale` | `plakat.hires` already exposes ML upscalers via `hires_upscaler`; standalone upscale word is Lanczos-only |
| `plakat.metadata.write` | gated on `plakat.save` attaching JSON sidecars; defer to v0.25 |
| `plakat.stylize` caching | one-shot load per call today (~5 GB); cache slot is a v0.25+ optimisation |

## v0.23 → v0.24 migration

One backwards-incompatible change in v0.24 phase 1:

```bund
// v0.23:
"alice.jpg" "a portrait" plakat.portrait

// v0.24:
"alice.jpg" 1.0 plakat.portrait.photo.add
"a portrait" plakat.portrait
```

`plakat.portrait` no longer takes a photo arg; photos come from
the `plakat.portrait.photo.add` collection stack. Per the v0.22
relaxed-compat decision.

## See also

- [`Tutorials/SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md)
  — narrative walkthrough with composition patterns.
- [`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md)
  — v0.24 design doc + locked decisions.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md)
  — v0.23 design doc.
- [`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md)
  — v0.22 design doc.
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) —
  v0.21 foundations.
- `vulogov/Bund` on GitHub — the language reference.
