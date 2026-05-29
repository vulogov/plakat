# `plakat run` — Bund scripting (v0.29)

Reference for `plakat run SCRIPT.bund` (file mode) and
`plakat run --repl` (REPL mode). For a tutorial-style walkthrough
with composition patterns, see
[`Tutorials/SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md).
Design RFCs:
[`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) (foundations),
[`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md) (the 7-namespace expansion),
[`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) (the v0.22 deferrals closed in v0.23),
[`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md) (persona depth + scripting completion),
[`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md) (art-medium presets + auto-LoRA discovery),
[`RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md) (AnimateDiff + every v0.25 carry),
[`RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md`](RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md) (multi-CN, AnimateLCM, `plakat.animate` bridge, motion-adapter inspection),
[`RFC_v0.29_BATCH_PRODUCTIVITY.md`](RFC_v0.29_BATCH_PRODUCTIVITY.md) (animate in scenarios + SDXL `plakat.animate` + `animate_format` config key).

## Modes

```
plakat run SCRIPT.bund          # File mode — eval the file, exit
plakat run --repl               # Interactive REPL on the same surface
plakat run --repl --out PATH    # REPL with a custom output dir
```

Both modes share the same `ScriptCtx` singleton + the same 50
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

## Host words (49 total)

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
| `plakat.upscale` | `( handle scale -- handle )` | **v0.26 phase 9**: scale = integer 2/4 (Lanczos) OR string `"real-esrgan-x2"` / `"real-esrgan-x4"` / `"real-esrgan-anime-x4"` (ML). |
| `plakat.animate` | `( prompt out_dir -- )` | **v0.28 phase 2 + v0.29 phases 0/1.** AnimateDiff single-prompt N-frame generation. Writes `frame-NNNN.png` + JSON sidecars under `out_dir`. Reads `animate_frames` / `animate_window_size` / `animate_window_overlap` / `animate_lcm` / `animate_format` from config. **SD 1.5 + SDXL** (variant detected from `plakat.load` alias; AnimateLCM is SD 1.5 only). |
| `plakat.save` | `( handle path -- )` | **v0.26 phase 8**: writes A1111 `parameters` tEXt + JSON sidecar when the handle has metadata (rendering paths populate it). Relative paths resolve under `--out`. |
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

### `plakat.look.*` — art-medium presets (v0.25 phase 8)

| Word | Stack effect |
|---|---|
| `plakat.look.apply` | `( name -- )` — pick a medium by name |
| `plakat.look.clear` | `( -- )` |
| `plakat.look.list` | `( -- l_1 … l_n n )` — push every catalog name + count |

State (`look_name`) on `ScriptCtx`. Apply runs lazily at
`plakat.generate` SD-family request-build time:
- Compositional fields (`prompt_prefix` / `prompt_suffix` /
  `negative_extras`) always apply.
- Override-only fields (`steps` / `guidance` / `scheduler_hint`)
  fill `ctx.config` slots that were left at defaults — explicit
  `plakat.config.set` values always win.
- Auto-LoRA discovery: when `ctx.loras` is empty AND the look has
  a `lora_query`, plakat searches Civitai → HF Hub → local cache
  for a compatible LoRA (filtered by the loaded base model) and
  prepends trigger words to the prompt.

Bundled looks: `ink-wash`, `watercolor`, `oil-painting`,
`charcoal`, `pencil`, `chalk-pastel`, `linocut`, `gouache`.
User-extensible via `$CONFIG_DIR/looks/*.json` (one PresetSpec
per file; filename stem is the catalog key). Stack mutations
invalidate the SD cache slots via `mark_loras_changed`.

Bund-side apply currently fires on the SD-family `plakat.generate`
path only; Flux + SD3 set the state correctly but apply happens
at the CLI level. See [`LOOKS.md`](LOOKS.md) for the full reference.

### `plakat.genre.*` — subject-domain presets (v0.25 phase 8)

| Word | Stack effect |
|---|---|
| `plakat.genre.apply` | `( name -- )` — pick a subject domain by name |
| `plakat.genre.clear` | `( -- )` |
| `plakat.genre.list` | `( -- g_1 … g_n n )` — push every catalog name + count |

Same shape as `plakat.look.*`. Independent axis — composes
additively with looks. Bundled: `anime`. User-extensible via
`$CONFIG_DIR/genres/*.json`. See [`GENRES.md`](GENRES.md).

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

### `plakat.metadata.*` — A1111 sidecar reader + writer (v0.24 phase 7, v0.26 phase 8)

| Word | Stack effect |
|---|---|
| `plakat.metadata.read` | `( path -- k_1 v_1 … k_n v_n n )` |
| `plakat.metadata.write` | `( handle path -- )` — **v0.26 phase 8** |

**Read** loads the JSON sidecar plakat writes alongside every
generated PNG (the structured form of the A1111 `parameters`
tEXt chunk). Pushes every populated field as a `(key, value)`
pair of strings plus a count. Required fields: `prompt`, `model`,
`seed`, `steps`, `guidance`, `scheduler`, `width`, `height`,
`generator`. Optional fields push only when set/non-empty:
`negative`, `loras`, `lora_scale`, `clip_skip`, `controls`,
`refiner_frac`, `mode`, `strength`, plus per-key `extras`. Bails
on missing sidecar / empty path / bad JSON.

**Write** (v0.26 phase 8) re-attaches the metadata from an
in-memory handle to an existing file: writes the JSON sidecar +
re-encodes the PNG with the A1111 `parameters` tEXt chunk. Bails
when the handle has no metadata attached (the rendering path
didn't populate it) or when the target file doesn't exist.

The full A1111-compatible writes flow:
- `plakat.generate` populates `GenerationMetadata` on the
  `ScriptCtx.images_metadata` slot at push time
  (`push_image_with_metadata`)
- `plakat.save` reads the metadata and writes both the PNG tEXt
  chunk + the `<name>.json` sidecar via
  `imaging::io::save_rgb_u8_with_metadata`
- `plakat.metadata.write` is for re-attaching metadata after
  edits (e.g. upscale → re-save with original generation
  parameters)

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

### Looks / Genres discovery (v0.25)

| Key | Type | Default | Notes |
|---|---|---|---|
| `offline_discovery` | bool | false | Skip remote LoRA discovery for `plakat.look.*` / `plakat.genre.*`. Mirrors the CLI `--offline` flag. When true, only the on-disk discovery cache + the local-cache scan run. |
| `animate_frames` | int | 16 | **v0.28.** Total output frames for `plakat.animate`. Values > `animate_window_size` engage the long-form sliding-window stitcher. |
| `animate_window_size` | int | 16 | **v0.28.** Per-window frame count for `plakat.animate` long-form. Must be ≤ 32 (motion-adapter `motion_max_seq_length`). |
| `animate_window_overlap` | int | 4 | **v0.28.** Cross-fade region in frames. Must be < `animate_window_size`. |
| `animate_lcm` | bool | false | **v0.28.** Switch `plakat.animate` to the AnimateLCM motion adapter + LCM scheduler. Defaults `steps=4`, `guidance=1.5` unless user already overrode either. ~5× speedup. SD 1.5 only. |
| `animate_format` | string | `"frames"` | **v0.29 phase 0.** Output format for `plakat.animate`: `"frames"` (PNGs only, default) / `"gif"` / `"mp4"` / `"webm"` / `"all"`. MP4 / WebM require ffmpeg on `$PATH`. |

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

## v0.29 limitations / deferred to v0.30+

v0.29 closed the v0.28 scripting + batch deferrals: `plakat.animate`
SDXL now works (via the v0.26 stylize cache slot pattern), animate
landed in HJSON scenarios (the largest plakat batch-driver gap),
and `animate_format` reaches every CLI output format. Remaining items:

| Limitation | Tracking |
|---|---|
| AnimateLCM SDXL | `wangfuyun/AnimateLCM-SDXL` still not publicly available. `--lcm --model sdxl` bails. v0.30 if upstream changes. |
| Per-frame video ControlNet | v0.28/v0.29 ship same-hint-every-frame conditioning. Per-frame video-to-video (a depth / canny video as guide) is v0.30+ territory. |
| Mixed-kind scenarios pay both pipeline costs | Scenarios with some generate + some animate tasks hold both pipelines resident. All-animate and all-generate pay only one. v0.30+ optimization. |
| Per-layer motion splice | v0.27/v0.28/v0.29 splice at block-output boundaries; the faithful diffusers `UNetMotionModel` splices INSIDE each block. RFC v0.27 §3.2 escalation if quality requires it. |
| Long-form > ~256 frames | Sliding-window long-form caps at ~256 frames before motion drift dominates. FreeNoise / FreeInit shared-noise schemes are v0.30+ candidates for cleaner long-form. |

## v0.28 → v0.29 migration

v0.29 is **fully additive**. No existing host word, config key,
scenario field, or stack effect changes shape. New surface:

```bund
// v0.29 phase 0: animate_format Bund key — GIF / MP4 / WebM from scripts
"sd15" plakat.load
"mp4"  "animate_format" plakat.config.set
"a watercolor cottage" "./out" plakat.animate

// v0.29 phase 1: plakat.animate now works on SDXL too
"sdxl" plakat.load
16    "animate_frames" plakat.config.set
1024  "width"          plakat.config.set
1024  "height"         plakat.config.set
"a knight in a forest, oil painting" "./out" plakat.animate
```

```hjson
# v0.29 phases 2 + 3: animate tasks in scenarios
{
    model: sd15
    type: animatediff       # scenario default
    frames: 16
    lcm: true
    format: gif
    tasks: [ { name: ..., scene: ..., weather: ..., prompt: "..." } ]
}
```

50 host words (same as v0.28); v0.29 added one config key
(`animate_format`) and the SDXL cache slot. See
[`ANIMATEDIFF.md`](ANIMATEDIFF.md) for the full animate reference
and [`Tutorials/SCENARIOS_TUTORIAL.md`](Tutorials/SCENARIOS_TUTORIAL.md)
§9 for the animate-in-scenarios narrative.

## v0.27 → v0.28 migration

v0.28 is **fully additive**. No existing host word, config key,
or stack effect changes shape. New surface:

```bund
// v0.28 phase 2: plakat.animate
"sd15" plakat.load
"true" "animate_lcm"     plakat.config.set      // 4-step AnimateLCM
32     "animate_frames"  plakat.config.set      // long-form (>16 → sliding window)
"a watercolor cottage at dawn" "./out" plakat.animate
// → ./out/frame-0000.png ... ./out/frame-0031.png + sidecars

// Composes with every existing ControlNet / look / genre / lora word.
"depth" "./depth.png" plakat.controlnet.add     // hint applied to every frame
```

Host word count: 49 → 50. See [`ANIMATEDIFF.md`](ANIMATEDIFF.md)
for the full animate reference.

## v0.25 → v0.26 migration

v0.26 is **fully additive**. No existing host word, config key,
scenario field, or stack effect changes shape.

What's new for Bund scripts:

```bund
// v0.26 phase 8: plakat.save automatically writes A1111 tEXt +
// JSON sidecar when the handle has metadata (every plakat.generate
// pushes metadata now). Existing scripts get sidecars for free.

// v0.26 phase 8: plakat.metadata.write re-attaches metadata
// after edits.
"a cottage" plakat.generate              // handle 1 with metadata
2 plakat.upscale                          // handle 2, no metadata
"cottage-2x.png" plakat.save              // saves plain PNG
1 "cottage-2x.png" plakat.metadata.write  // attach metadata after the fact

// v0.26 phase 9: plakat.upscale accepts Real-ESRGAN method strings.
"real-esrgan-x4" plakat.upscale          // ML upscale, x4
"real-esrgan-anime-x4" plakat.upscale    // anime-tuned variant

// v0.26 phase 10-11: plakat.look.* / plakat.genre.* now apply on
// Flux + SD3 generate paths too (v0.25 was SD-family only).
"flux-dev" plakat.load
"watercolor" plakat.look.apply
"a cottage" plakat.generate    // Flux auto-discovers a Flux-compatible LoRA
```

## v0.24 → v0.25 migration

v0.25 is **additive**. No existing host word, config key, or
stack effect changes shape. The new pieces are opt-in:

```bund
// Adopt the new axes:
"watercolor" plakat.look.apply
"anime"      plakat.genre.apply
"a knight" plakat.generate
```

If you don't apply a look / genre, behavior is byte-identical to
v0.24.

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
- [`LOOKS.md`](LOOKS.md) — art-medium presets reference.
- [`GENRES.md`](GENRES.md) — subject-domain axis reference.
- [`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md)
  — v0.25 design doc + locked decisions.
- [`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md)
  — v0.24 design doc.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md)
  — v0.23 design doc.
- [`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md)
  — v0.22 design doc.
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) —
  v0.21 foundations.
- `vulogov/Bund` on GitHub — the language reference.
