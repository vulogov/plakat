# plakat — release history

"What's new" sections for v0.13 through v0.32. The current
release's notes live in the [main README](../README.md). Older
cycles are archived here so the README stays focused on what's
new this turn.

For commit-level history see `git log`; for migration notes the
per-cycle commits carry the rationale + before/after.

## What's new in v0.32 — animate-lite + diversify-3

After two consecutive diversify cycles (v0.30 + v0.31), v0.32
pays down the animate quality backlog with the single most-
visible item — **FreeNoise long-form** — while continuing the
diversify momentum with two architectural / perf wins: a
**vendored CLIP rollout** to every SD-family pipeline (unlocks
`--embedding` everywhere in future cycles), and **SDXL VAE
caching** across scenario kind switches.

Three phases shipped: one animate quality win + two carry
closures. Cleanest cycle in five turns.

### FreeNoise long-form for AnimateDiff (closes v0.27 deferral)

```bash
# Existing long-form syntax — random noise per window (v0.27).
plakat animate --animatediff --model sd15 \
    --from "a misty forest at dawn" \
    --frames 64 --window-size 16 --window-overlap 4 \
    --format mp4

# Add --free-noise to share noise across overlapping windows —
# eliminates the cross-fade seam artefact (v0.32).
plakat animate --animatediff --model sd15 \
    --from "a misty forest at dawn" \
    --frames 64 --window-size 16 --window-overlap 4 \
    --free-noise --format mp4
```

Cao et al., "FreeNoise: Tuning-Free Longer Video Diffusion." The
flag pre-generates a `(total_frames, 4, H/8, W/8)` noise tensor
at the user's seed, then slices it per sliding-window — adjacent
windows automatically share noise in the overlap region because
they slice the SAME underlying tensor. The v0.27 phase 5
linear-ramp latent blend still runs, but the two sides being
blended now come from the same noise sequence, so the seam
disappears.

SD 1.5 + SDXL both ship. Opt-in flag preserves byte-identical
output on `--seed N --frames 64` when the flag is OFF (existing
runs unchanged). Composes with multi-CN, AnimateLCM, motion LoRAs.
Single-window runs (frames ≤ window-size) are no-ops.

### Vendored CLIP rollout (closes v0.30 architectural deferral)

v0.30 phase 0 vendored CLIP for `SdCore` only (to enable TI
runtime injection). v0.32 phase 1 finishes the rollout —
every SD-family pipeline now holds plakat's vendored CLIP-L
text encoder instead of candle's:

- `AnimateDiffPipeline::text_encoder` (SD 1.5)
- `AnimateDiffSdxlPipeline::text_encoder_l`
- `sd3::Pipeline::clip_l + clip_l_cfg`
- `flux::Pipeline::clip_text + clip_cfg`
- `stylize::Pipeline::text_encoder`

Numerically identical to candle's path (per the v0.30 phase 0
forward-pass tests). User-visible impact: zero in v0.32 — but
this is the architectural foundation. Future cycles can wire
`--embedding` (TI runtime injection) through the same
`Config::with_vocab(n)` pattern v0.30 phase 0 established for
SdCore, now that every pipeline holds the vendored type. A new
compile-time type-lock test in `vendored_clip::tests` prevents
silent regressions.

### SDXL VAE caching across mixed-kind scenarios

Scenarios that mix `type: generate` and `type: animatediff`
tasks used to rebuild the ~330 MB SDXL VAE on every kind switch
(after v0.31 phase 3's pipeline eviction kicked in). v0.32 phase
2 wraps `SdCore.vae` in `Arc<AutoEncoderKL>` and adds a
scenario-level cache keyed by base alias — subsequent t2i loads
against the same SDXL base reuse the cached Arc instead of
rebuilding from disk.

Auto-deref keeps every `.vae.encode(...)` / `.vae.decode(...)`
call site at the pipeline boundary unchanged.

```
INFO plakat: SdCore: reusing cached VAE (skipping ...vae.safetensors build)
```

shows up in logs when the cache hits. AnimateDiff load functions
don't yet take a vae param — animate-side sharing waits for
v0.33+. The cache helper itself (`vae_cache_lookup`) is a pure
generic function unit-tested with 6 decision cases.

### Documentation

- [`ANIMATEDIFF.md`](ANIMATEDIFF.md) — bumped to v0.32 with the
  FreeNoise long-form quick-start. Capability matrix adds the
  "shared-noise long-form" row.
- [`RFC_v0.32_ANIMATE_LITE_DIVERSIFY_3.md`](RFC_v0.32_ANIMATE_LITE_DIVERSIFY_3.md)
  — design doc, locked decisions, 4-phase plan.

### By the numbers

- **1030 lib + 47 integration tests = 1077 active tests** (+10
  lib across the cycle).
- 3 phase commits + RFC + close-out.
- v0.27 FreeNoise animate quality deferral **closed**.
- v0.30 vendored CLIP architectural deferral **closed**
  (rollout to all SD-family pipelines).
- v0.32 phase 2 partially closes the mixed-kind VAE rebuild
  cost — t2i side complete; animate-side sharing defers to v0.33.

### v0.31 → v0.32 migration

v0.32 is **fully additive**. Every existing flag, host word,
config key, and scenario field keeps its v0.31 shape. New surface:

- ✅ `--free-noise` flag on `plakat animate` (opt-in; off by
  default preserves byte-identical numerics).
- ✅ `SdLoadRequest.vae_cache` + `t2i::LoadRequest.vae_cache`
  fields (`Option<Arc<AutoEncoderKL>>`). External callers pass
  `None` for the v0.31 behaviour.
- ✅ Every SD-family pipeline's CLIP-L field type is now the
  vendored CLIP — same forward-pass numerics, source-level
  type-lock prevents future regressions.

## What's new in v0.31 — diversify-2 (INT8 bail, four wins)

The cycle started as **diversify-2 + INT8 SDXL headline** — close
the v0.30 SDXL dual-encoder TI stretch goal, ship INT8 SDXL UNet
quantization for 12 GB GPUs, add weighted wildcards, close the
v0.29 mixed-kind pipeline cache carry. Phase 1's INT8 validation
spike rejected the codec direction (candle 0.10.2 has no
quantized Conv2d, and SDXL UNet is conv-heavy). The RFC's bail
plan triggered: swap phase 1 for **Pony preset + `plakat civitai
sync`** — two visible polish wins from the same deferral list.

Four phases shipped: two carry closures (SDXL dual TI from v0.30,
mixed-kind cache from v0.29) + two new feature surfaces (Pony +
civitai sync) + one prompt-power addition (weighted wildcards).

### SDXL dual-encoder TI parser

```bash
plakat generate "a portrait in <my-sdxl-style> art" --model sdxl \
    --embedding ./my-sdxl-style.safetensors
```

v0.30's vendored CLIP + tokenizer-mutation infrastructure was
CLIP-L-only. v0.31 phase 0 drops the parser bail for the
`clip_l` + `clip_g` dual format and mirrors the v0.30 extension
pattern through SDXL CLIP-G. A stack can mix single-encoder and
dual-encoder TIs.

### Pony Diffusion preset

`--model pony` (also `pony-v6`, `pony-diffusion-v6`) resolves to
`AstraliteHeart/pony-diffusion-v6-xl`. `--look pony` prepends
the Pony quality tags (`score_9, score_8_up, ...`) and applies a
Pony-tuned negative.

### `plakat civitai sync USERNAME --out DIR`

Bulk-download a Civitai creator's library. Walks API cursor
pagination, picks each model's primary version + primary file,
copies into `--out DIR`. Idempotent on rerun; honours
`CIVITAI_API_KEY`.

### Weighted wildcards

`{WEIGHT::CHOICE|...}` syntax adds explicit weights to inline
alternation (relative; normalized; omitted defaults to 1.0).
Composes with the v0.16 nested syntax.

### Mixed-kind scenarios pipeline cache

Closes the v0.29 carry. Scenarios that mix `type: generate` and
`type: animatediff` tasks now drop the opposite-kind cached
pipeline at kind boundaries — peak memory drops by ~5-10 GB.

### Release workflow finally green

After four iterations across v0.29/v0.30/v0.31, the arm64
cross-build's apt-source dance landed clean: wholesale wipe of
`/etc/apt/sources.list`, `/etc/apt/sources.list.d/*`, and
`/etc/apt/apt-mirrors.txt`, then writes only two explicit
sources.

### By the numbers

- 1020 lib + 47 integration tests = **1067 active tests** (+23
  lib across the cycle).
- 4 phase commits + RFC + close-out + CI workflow fix.

## What's new in v0.30 — diversify + one animate theme

After three consecutive AnimateDiff cycles (v0.27 scope, v0.28
single-script polish, v0.29 batch polish), v0.30 picked fresh
ground. It closed the **longest-running open carry** (Textual
Inversion runtime injection, deferred since v0.16 — eight cycles
ago), extended v0.28's LCM scheduler wiring to single-image t2i,
shipped the **most-requested animate carry** (per-frame video
ControlNet), and enriched `plakat doctor` to cover the new
surface. Five phases, ~2150 LOC delta.

### Embedding (Textual Inversion) runtime injection

```bash
plakat generate "a portrait in <my-style> art" \
    --embedding ./my-style.safetensors
```

Civitai TIs now apply at inference time on SD 1.5 / SD 2.1 / SDXL
CLIP-L. The parser, merger, and `plakat embedding info`
inspector had shipped since v0.16; what was missing was a text
encoder that would accept a vocab larger than candle's stock
`clip::Config.vocab_size` (private). v0.30 vendors a minimal
~430-LOC CLIP text encoder (`src/pipelines/vendored_clip.rs`)
with a public `vocab_size` + `Config::with_vocab()` builder. The
existing merger appends TI vectors to `token_embedding.weight`
via a tempfile (mirroring LoRA's merge pattern); the tokenizer
gets the new trigger tokens registered via
`Tokenizer::add_tokens`. Multi-vector TIs render as `N`
consecutive tokens (`trigger`, `trigger_1`, ..., `trigger_{N-1}`).
SDXL dual-encoder TIs (files with both `clip_l` and `clip_g`)
were still rejected in v0.30; closed in v0.31 phase 0.

### LCM-LoRA in t2i

```bash
# Auto-detects from the LoRA source — no flag needed for canonical names
plakat generate "a misty forest, dawn light" \
    --lora latent-consistency/lcm-lora-sdv1-5
# → scheduler=lcm, steps=4, guidance=1.5 (~10× speedup)
```

Extends v0.28's AnimateLCM scheduler wiring to single-image
generation. Substring heuristic on `--lora` sources catches the
canonical `latent-consistency/lcm-lora-*` repos automatically;
explicit `--lcm` flag covers non-canonical names. SD 1.5 + SDXL.

### Per-frame video ControlNet (video-to-video)

```bash
plakat animate --animatediff --model sd15 \
    --from "a glowing neon dragon, cyberpunk alley, rain" \
    --control-spec 'openpose:video=./reference.mp4:strength=0.9' \
    --frames 16 --format mp4
```

The headline animate carry on every deferral list since v0.27.
`video=PATH` triggers ffmpeg input decode, even sub-sampling to
the animate frame budget, per-frame annotation, and per-frame CN
residuals injected through SD 1.5 + SDXL AnimateDiff sampling.
Composes with sliding-window long-form. HJSON scenarios pick up
the new `video:` field on `controls[]` entries.

### `plakat doctor` enrichment

Two new sections (human + `--json`): ffmpeg version probe (warn
when missing) and HF / Civitai API token presence (boolean only —
never the value).

### By the numbers

- 997 lib + 47 integration tests = **1044 active tests** (+35
  lib + +6 integration across the cycle).
- New module: `pipelines::vendored_clip` (~430 LOC).
- v0.16 phase 9 carry **closed**.
- v0.27 video CN deferral **closed**.

## What's new in v0.29 — batch productivity completion

v0.28 made the **single-script** AnimateDiff surface pleasant.
v0.29 makes it pleasant **at production scale** — closing the
loudest v0.28 deferrals: animate in HJSON scenarios (the biggest
plakat batch-driver gap), SDXL in `plakat.animate`, and the
final `animate_format` Bund config key. Six phases, ~1100 LOC
delta, zero new architectural risk.

### Animate in HJSON scenarios

```hjson
{
    model: sd15
    type: animatediff       # scenario default — every task is animate
    frames: 16
    lcm: true               # 4-step AnimateLCM (~5× speedup)
    format: gif
    out: ./out/animations
    scene:   [ { name: dawn,  prompt: "at dawn" } ]
    weather: [ { name: mist,  prompt: "misty" } ]
    tasks: [
        { name: cottage, scene: dawn, weather: mist,
          prompt: "a watercolor cottage" }
        { name: knight,  scene: dawn, weather: mist,
          prompt: "a knight in a forest, oil painting",
          frames: 32, format: mp4 }     # per-row overrides
    ]
}
```

```bash
plakat scenario animate.hjson --dry-run    # preview the plan
plakat scenario animate.hjson              # render every task
plakat scenario animate.hjson --resume     # skip rendered tasks
plakat scenario animate.hjson --only knight
```

The same scenario filters (`--resume` / `--only` / `--limit` /
`--dry-run`) work unchanged. Per-task overrides for `frames`,
`window-size`, `window-overlap`, `lcm`, `motion-lora`,
`motion-lora-scale`, `format`, `gif-delay-ms` compose with
scenario-level defaults. ControlNet through the existing
`control:` / `controls:` fields (multi-CN sum). All-animate
scenarios don't need the `enhancer:` field set (prompt
enhancement is t2i-only). Mixed-kind scenarios still need it for
the generate tasks.

Closes the **largest plakat batch-driver gap** identified in the
v0.28 cycle audit: `cli/scenario.rs::TaskDef` had zero animate-
related fields before this release.

### SDXL `plakat.animate`

```bund
"sdxl" plakat.load
16   "animate_frames" plakat.config.set
1024 "width"          plakat.config.set
1024 "height"         plakat.config.set
"a knight in a forest, oil painting" "./out" plakat.animate
```

Removes the v0.28 SD-1.5-only restriction in scripting. New
`ScriptCtx::loaded_animatediff_sdxl` cache slot mirrors the
v0.26 stylize slot pattern so multi-call scripts amortise the
~7 GB SDXL backbone load. SD 1.5 keeps its own cache slot
(`loaded_animatediff`) with a key encoding the LCM mode —
toggling `animate_lcm` between calls swaps the pipeline.
AnimateLCM-SDXL still bails loud (upstream repo not publicly
available).

### `animate_format` Bund config key

```bund
"sd15" plakat.load
"mp4" "animate_format" plakat.config.set
"a watercolor cottage" "./out" plakat.animate
```

The final v0.28 Bund surface gap. `animate_format` accepts the
same five strings as the CLI's `--format`: `frames | gif | mp4 |
webm | all`. MP4 / WebM need ffmpeg on `$PATH`; the availability
check fires before inference so install pointers come fast.

### CI workflow fix

The arm64 cross-build matrix step in `.github/workflows/release.yml`
gained a rewrite over `/etc/apt/apt-mirrors.txt` to use only the
single Azure mirror. The runner image's mirrorlist
(`URIs: mirror+file:///etc/apt/apt-mirrors.txt`) had grown
multiple amd64-only entries (security.ubuntu.com,
archive.ubuntu.com) that the v0.26 phase fix to `.list` / `.sources`
files didn't reach. v0.29 patches that hole.

### By the numbers

- 962 lib tests + 41 integration tests = **1003 active tests**
  (+9 lib + +4 integration across the cycle).
- 6 phase commits + RFC + CI workflow fix.
- 50 host words (unchanged); 1 new config key (`animate_format`).

### v0.28 → v0.29 migration

v0.29 was **fully additive**. Every existing flag, host word,
config key, and scenario field kept its v0.28 shape.

## What's new in v0.28 — AnimateDiff productivity polish

v0.27 made AnimateDiff feature-complete. v0.28 made it pleasant
to use in practice — closing the loudest v0.27 deferrals with
four targeted productivity wins.

### 4-step animate via AnimateLCM

`--lcm` switches the motion adapter to `wangfuyun/AnimateLCM`
(17 modules: 16 + a V1/V2-style mid-block), the scheduler to LCM,
and applies the diffusers-recommended defaults `steps=4
guidance=1.5`. SD 1.5 only (SDXL AnimateLCM not publicly
available).

### Multi-ControlNet stacking through animate

`--control-spec` repeatable, same grammar as `plakat generate`'s.
Mutually exclusive at parse time with the legacy single-CN flags.
Both SD 1.5 + SDXL supported.

### `plakat.animate` Bund host word

The last major CLI verb missing from Bund scripting. Stack effect:
`( prompt out_dir -- )`. Four new config keys: `animate_frames`
(16), `animate_window_size` (16), `animate_window_overlap` (4),
`animate_lcm` (false). SD 1.5 only in v0.28 (SDXL animate
scripting landed in v0.29).

### `plakat motion-adapter` inspection

`plakat motion-adapter list` + `plakat motion-adapter info REPO`,
parallel to `plakat civitai info`. Dumps motion adapter config +
per-block tensor breakdown.

### By the numbers

- 953 lib + 37 integration tests green (+5 lib + +6 integration
  across the cycle).
- 6 phase commits + RFC.
- 49 → 50 host words.
- AnimateLCM loader added; shares `load_from_repo` +
  `load_with_motion_loras` with V3 + SDXL beta.

## What's new in v0.27 — AnimateDiff feature complete

Eight phases close every AnimateDiff carry from the v0.26 cycle.
End-to-end inference now works on both SD 1.5 + SDXL, each with
optional ControlNet conditioning and a sliding-window long-form
mode that lifts V3's 32-frame cap.

### AnimateDiff inference (SD 1.5 + SDXL)

V3 SD 1.5 (`guoyww/animatediff-motion-adapter-v1-5-3`, 16 motion
modules) and SDXL beta (`guoyww/animatediff-motion-adapter-sdxl-beta`,
12 motion modules) share the same `MotionAdapterConfig` schema; the
only difference is `block_out_channels` matching each base UNet's
block layout. Both paths use the block-boundary motion splice
(sequential motion modules at each down/up block output).

### AnimateDiff + ControlNet

Single conditioning image, same hint applied to every frame. Five
kinds supported (`depth` / `canny` / `openpose` / `lineart` /
`softedge`). The CN runs at the full per-step batch and feeds
residuals into the motion UNet's down + mid hooks.

### Long-form sliding window

V3's 32-frame `motion_max_seq_length` was a hard cap on a single
window. Long-form mode chains overlapping windows with linear-ramp
latent-space blend, lifting the practical cap to ~256 frames.

### Motion-module tensor naming fix

A latent bug from the v0.26 phase 2 motion-module code: the
referenced tensor key paths (`motion_modules.{j}.temporal_transformer.norm`,
`attention_blocks.{0,1}`, `norms.{0,1,2}`, `pos_encoder.pe`) don't
exist in the real upstream safetensors. v0.27 phase 2 fixed all the
paths and collapsed `Vec<TemporalTransformerBlock>` → a single
inner block per motion module. Verified upstream against V3 SD 1.5
(540 keys) and SDXL beta (405 keys); both share the same convention.

### By the numbers

- 948 lib tests + 32 integration tests green (+14 lib across the cycle).
- 8 phase commits + RFC.
- 49 host words (unchanged from v0.26).
- SD 1.5 + SDXL motion adapter loaders share `load_from_repo` +
  `load_with_motion_loras` helpers.
- Shared free helpers `validate_long_form_window` +
  `stitch_long_form<F>` agree the SD 1.5 and SDXL `generate_long`
  paths on the blend math by construction.

## What's new in v0.26 — AnimateDiff infrastructure + every v0.25 carry closed

Thirteen phases close **every v0.25 carry** in one cycle: SD3 /
SD3.5 animate, Bund look/genre apply on Flux + SD3, scenario
auto-LoRA discovery, Real-ESRGAN in `plakat.upscale`,
`plakat.metadata.write`, `plakat.stylize` cache slot, plus the
full **AnimateDiff infrastructure** (motion adapter, temporal
modules, vendored UNet with motion splice, motion LoRAs, output
formats). Host word count 48 → 49.

End-to-end AnimateDiff inference was deferred to v0.26.1, then
folded into v0.27 phase 0 (where it closed alongside SDXL +
ControlNet + long-form sliding window).

### SD3 / SD3.5 animate

Closes the v0.20-era `plakat animate` SD3 bail:

```bash
# Three-encoder lerp (CLIP-L + CLIP-G + T5) + flow-match per frame.
plakat animate --model sd35-medium \
    --from "a quiet temple at dawn" \
    --to   "a quiet temple at sunset" \
    --frames 16 --gif-delay-ms 125
```

### `plakat.look.*` / `plakat.genre.*` on Flux + SD3

The v0.25 look/genre apply was SD-family only. v0.26 wires it on
the Flux + SD3 generate paths in `script_entry.rs`. LoRA
discovery filters by `BaseFamily::Flux` / `BaseFamily::Sd3` so
the cache key `(name, base_family)` finds the right LoRAs.

### Scenario auto-LoRA discovery (smart-cached)

100 tasks with `look: watercolor` fire **one** network call —
the discovered LoRA is cached per `(look_name, base_family)`
across the scenario.

### Real-ESRGAN in `plakat.upscale`

```bund
1 "real-esrgan-x4" plakat.upscale         // ML x4 upscale
1 "real-esrgan-anime-x4" plakat.upscale   // anime-tuned variant
```

`plakat.upscale` dispatches on arg type: integer (2 / 4) =
Lanczos, string = Real-ESRGAN ML method.

### `plakat.save` metadata + `plakat.metadata.write`

`plakat.save` writes the A1111 `parameters` PNG tEXt chunk +
JSON sidecar automatically when the handle has metadata
attached. New host word `plakat.metadata.write` re-attaches
metadata to an existing file.

### `plakat.stylize` cache slot

Closes the v0.24 'caching is a v0.25+ optimisation' deferral.
Multi-call scripts amortise the ~5 GB SD1.5 + IP-Adapter
weights load.

### AnimateDiff infrastructure

- AnimateDiff V3 motion adapter loader
  (`guoyww/animatediff-motion-adapter-v1-5-3`)
- 16 per-block temporal-attention modules built from real V3 weights
- Vendored SD 1.5 UNet (`Sd15MotionUNet`) with motion-module splice
  at block-output boundaries
- Motion LoRA composition (`--motion-lora SPEC`) — merges into the
  motion-adapter weights via the existing LoRA-merge pipeline
- `--format {frames, gif, mp4, webm, all}` with ffmpeg integration
- `AnimateDiffPipeline` assembly type

### By the numbers

- 934 lib tests + 20 integration tests green (+32 lib across the cycle).
- 13 phase commits + RFC.
- 48 → 49 host words.
- 7 v0.25 carries closed.
- New `MergeTarget::MOTION_ADAPTER` LoRA-merge variant.
- New `loaded_stylize` cache slot on `ScriptCtx`.

## What's new in v0.25 — art-medium presets + auto-LoRA discovery

Twelve phases ship two new preset axes — `--look` (art medium)
and `--genre` (subject domain) — with **automatic LoRA discovery**
from Civitai / HuggingFace / your local cache. Pick "watercolor"
or "ink-wash" with one flag and plakat composes the prompt, picks
a sampler, and finds a compatible LoRA matched to your loaded
base model. Host word count 42 → 48.

### `--look` — eight bundled art mediums

```bash
plakat generate --model sd15 --look watercolor "a cottage in the woods"
plakat generate --model sdxl --look oil-painting "a still life"
```

Eight mediums ship: `ink-wash`, `watercolor`, `oil-painting`,
`charcoal`, `pencil`, `chalk-pastel`, `linocut`, `gouache`. Each
bundles a prompt prefix/suffix + recommended sampler/steps/guidance
+ a `lora_query` that drives auto-discovery.

### `--genre` — independent subject-domain axis

```bash
plakat generate --model sdxl --look watercolor --genre anime "a knight"
```

`anime` ships built-in; user-extensible via `$CONFIG_DIR/genres/*.json`.

### Auto-LoRA discovery chain (Civitai → HF → local)

`--lora` empty → plakat searches for a compatible LoRA, filtering
by base-model compatibility. Trigger words from the discovered LoRA
auto-prepend to the prompt. `--offline` short-circuits to cache +
local-scan only.

### Surfaces

`--look` / `--genre` / `--offline` work on every prompt-driven
subcommand (`generate`, `portrait`, `img2img`, `inpaint`,
`outpaint`), in scenarios at both global + per-task level, and via
six new Bund host words (`plakat.look.{apply,clear,list}` +
`plakat.genre.{apply,clear,list}`).

### By the numbers

- 902 lib tests + 20 integration tests green (+85 lib across the cycle).
- 12 phase commits + RFC.
- 42 → 48 host words.
- 8 bundled looks + 1 bundled genre + user-extension directories wired.
- 3-source discovery chain (Civitai + HF Hub + local-cache scan)
  with on-disk caching keyed by `(name, base_model)`.

### Documentation

- [`LOOKS.md`](LOOKS.md) — flag reference + user-extension format.
- [`GENRES.md`](GENRES.md) — subject-domain axis.
- [`LOOKS_TUTORIAL.md`](Tutorials/LOOKS_TUTORIAL.md) — walkthrough.
- [`GENRES_TUTORIAL.md`](Tutorials/GENRES_TUTORIAL.md) — companion.
- [`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md) — design doc.

## What's new in v0.24 — persona depth + scripting completion

Eleven phases finish the Bund scripting surface. The
v0.21/22/23 arc closes: after v0.24 there's no "use the CLI
for X" gap for users staying in scripts. Word count 33 → 42.

### Persona depth (phases 1–3)

```bund
// Multi-photo portrait (BREAKING — see migration below).
"./alice.jpg" 0.7 plakat.portrait.photo.add
"./bob.jpg"   0.3 plakat.portrait.photo.add
"a couple at the beach" plakat.portrait

// Face alignment overrides (CLI parity with --face-bbox /
// --face-landmarks).
"0.2,0.1,0.8,0.7" "face_bbox" plakat.config.set
"0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68"
    "face_landmarks" plakat.config.set

// Identity-encoder variant override (the four FaceID kinds).
"face-id-sdxl" "identity_kind" plakat.config.set
```

CLI flags `--photo` / `--face-bbox` / `--face-landmarks` / all
four `--identity` variants already existed; v0.24 wires the
scripting surface to them.

### Scripting completion (phases 4–9)

Six v0.20+ carries close:

- **`plakat.outpaint`** — `( prompt input expand-spec -- handle )`.
- **`plakat.embedding.*`** — Textual Inversion stack (add / clear / list).
- **`plakat.stylize`** — IP-Adapter style transfer.
- **`plakat.metadata.read`** — JSON sidecar reader (read-only).
- **Flux + SD3 ControlNet `from=`** — lazy first-generate
  annotation; cached per-pipeline; dim changes invalidate.
- **Flux inpaint** via `plakat.inpaint` — wires the
  flux-fill-dev variant + channel-concat path.

### v0.23 → v0.24 migration

One backwards-incompatible change in phase 1:

```bund
// v0.23:
"alice.jpg" "a portrait" plakat.portrait

// v0.24:
"alice.jpg" 1.0 plakat.portrait.photo.add
"a portrait" plakat.portrait
```

`plakat.portrait` no longer takes a photo arg; photos come from
the `plakat.portrait.photo.add` collection stack.

### By the numbers

- 817 lib tests green (+45 across the cycle).
- 11 phase commits + RFC.
- 33 → 42 host words.
- Three new config keys (face_bbox / face_landmarks /
  identity_kind).

### Documentation

- [`SCRIPTING.md`](SCRIPTING.md) — full reference, updated for v0.24.
- [`SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md) §11.
- [`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md) — design doc.

## What's new in v0.23 — Bund deferrals closed

Nine phases close every "deferred to v0.23" stub v0.22 explicitly
took on, plus add two new things: the `plakat.style.*` catalog
namespace and the `plakat.inpaint` host word. Word count
28 → 33; smaller cycle than v0.22 (~7 phases of real work).

### Cache architecture: the SdT2i slot

`plakat.load "sdxl"` now warms a `t2i::Pipeline` slot in addition
to (or sharing `Arc<SdCore>` with) the v0.22 `portrait::Pipeline`
slot. `plakat.generate`'s SD-family path routes through SdT2i,
which carries the SDXL refiner UNet hook + the CLIP-skip encode
path that the v0.22 portrait cache didn't expose.

### `plakat.style.*` namespace (phase 4)

```bund
"poster-bold" plakat.style.apply       // by id
"./ref.jpg"   plakat.style.detect      // CLIP-H detect from photo
plakat.style.list                      // ( -- ...ids count )
plakat.style.clear
0.7 "style_strength" plakat.config.set
"a town square" plakat.generate
```

Resolution runs lazily at `plakat.generate` request-build time:
catalog LoRAs override the user LoRA stack for the load (CLI
parity with `--style ID`); trigger prepends to prompt;
`negative_extras` appends to negative.

### `plakat.inpaint` host word (phase 5)

```bund
"stained glass window in the wall"
   "./photo.png" "./mask.png"
   plakat.inpaint
   "result.png" plakat.save
```

Stack: `( prompt input mask -- handle )`. SD-family + SD3 wired;
Flux bail closed in v0.24.

### Flux + SD3 ControlNet (phases 6–7)

CN stack wires into `LoadRequest.controlnets` at pipeline-load
time. Stack mutations call `mark_controlnets_changed` which
drops the Flux/SD3 slot. v0.23 cap: `image=` specs only;
auto-annotate (`from=`) deferred to v0.24 phase 8.

### By the numbers

- 772 lib tests green (+14 across the cycle).
- 28 → 33 host words. `style_catalog` config key added; six v0.22
  deferred items closed.

### Documentation

- [`SCRIPTING.md`](SCRIPTING.md) — full reference, updated for v0.23.
- [`SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md) §10.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) — design doc.

## What's new in v0.22 — Bund words expansion

The v0.21 scripting MVP graduates to feature parity with the
CLI. Twelve phases ship 21 new host words across 7 namespaces,
a pipeline cache so multi-call scripts don't reload the model,
all three model families on the scripting surface, and 50+
new Category-B config keys.

### The 28 `plakat.*` words

```
// v0.21 core (cache-aware in v0.22):
plakat.load / .generate / .img2img / .portrait / .upscale / .save / .config.set / .echo

// LoRA stack (phase 4):
plakat.lora.add / .clear / .list

// ControlNet stack (phase 5):
plakat.controlnet.add / .annotate / .spec / .clear / .list

// Post-process toggles:
plakat.refiner.{enable,disable}      // phase 6: SDXL refiner switch
plakat.adetailer.{enable,disable}    // phase 7: SCRFD face refinement
plakat.hires.{enable,disable}        // phase 8: upscale + img2img refine
plakat.artefact.add / .clear / .list / .blend.{enable,disable}   // phase 9

// Prompt enhancer (phase 10):
plakat.enhance ( prompt -- enhanced )
```

### Pipeline cache + all three families

`ScriptCtx::loaded` holds one `LoadedPipeline` enum
(`SdFamily(portrait::Pipeline)` / `Flux(flux::Pipeline)` /
`Sd3(sd3::Pipeline)`). Same-alias reuse skips the model load
entirely. Family-specific config keys (`quantize_t5`,
`kontext_bucket` for Flux; `tiled`, `tile_size`,
`tile_stride` for SD3) flow through the same
`plakat.config.set` surface.

### By the numbers

- 758 lib tests green (+124 new across the cycle).
- 12 phase commits + 2 RFC commits + 1 release-notes commit.
- 28 host words (was 8 in v0.21). ~60 `GenerationConfig` keys.
- Four composition tests exercise multi-namespace state
  interaction end-to-end.

### Deferred to v0.23 (all closed in v0.23)

- Flux + SD3 ControlNet load-time wiring.
- SDXL refiner UNet load.
- `plakat.inpaint` (mask path argument).
- `clip_skip` wiring.
- `plakat.style.*` namespace.

## What's new in v0.21 — Bund scripting

One big swing this cycle: `plakat run SCRIPT.bund` ships a
stack-based DSL for driving plakat's pipelines from a script.
Composition wins that were awkward at the CLI (`generate →
upscale`, `generate → img2img → save`, multi-variation runs at a
pinned seed) become one-liners; an interactive REPL on the same
surface lands for exploration.

### The seven `plakat.*` words

```
plakat.load        ( model-alias -- )
plakat.generate    ( prompt -- handle )
plakat.img2img     ( prompt input -- handle )      // input: path OR handle
plakat.portrait    ( prompt photo -- handle )      // photo: path OR handle
plakat.upscale     ( handle scale -- handle )      // Lanczos x2/x4
plakat.save        ( handle path -- )
plakat.config.set  ( value key -- )                // steps/guidance/seed/...
```

```bund
"sdxl" plakat.load
40   "steps"     plakat.config.set
3.5  "guidance"  plakat.config.set
"a fox in a meadow" plakat.generate    // handle 1
  2 plakat.upscale                     // handle 2 (2048x2048)
  "fox-2k.png" plakat.save
"a fox in a meadow, painterly oil"
  1  plakat.img2img                    // refine handle 1 → handle 3
  "fox-refined.png" plakat.save
```

Handles address rendered images in an in-memory registry —
chains compose without disk round-trips. Sources aren't consumed
by downstream words, so the same generation can fan out into
upscale + img2img + portrait variants from one root.

### Interactive REPL

```text
$ plakat run --repl
plakat REPL (v0.21). Type .help for commands, .q to exit.
plakat> "sd15" plakat.load
plakat> 50 "steps" plakat.config.set
plakat> "a fox" plakat.generate
=> 1
plakat> "fox.png" plakat.save
plakat> .s
  [0] 1
plakat> .q
```

Persistent state across lines, history at `<plakat-config-dir>/repl_history`,
Forth-style meta-commands (`.q` / `.s` / `.help`), the `=>` echo
shows the top of the workbench after each successful eval.

### v0.21 limitations

- **SD-family only** — `sd15`, `sd21`, `sdxl`, `sdxl-turbo`.
  Flux + SD3 / SD3.5 bail at `plakat.load` with a clear "phase
  2b" pointer; both land in v0.22.
- **Single-photo portrait** — no FaceID / multi-photo / manual
  landmarks. v0.22.
- **Lanczos x2/x4 upscale only** — Real-ESRGAN ML upscaling in
  v0.22.
- **No LoRA / ControlNet / refiner words** — use the CLI directly
  if you need them; the scripting surface stays minimal in v0.21.
- **No pipeline cache** — every `plakat.generate` reloads the
  model. Acceptable for the MVP; cache work is v0.22.

### Documentation

- [`SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md)
  — narrative walkthrough: syntax in 60 seconds, every word with
  examples, composition patterns, the REPL, limitations.
- [`SCRIPTING.md`](SCRIPTING.md) — reference manual.
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md)
  — design RFC + the seven locked architectural decisions + phase
  plan. Read if you're contributing a new `plakat.*` word.

### By the numbers

- 634 lib tests green (+65 new across the cycle).
- 8 phase commits + 1 RFC commit + 1 release-notes commit.
- 8 host words (7 MVP + `plakat.echo` smoke). 9 `GenerationConfig`
  knobs.

## What's new in v0.20 — recipe replay, project bootstrap, Flux animate, Kontext + tiled

Nine features in three groups. v0.20 picks up where v0.19 left
off on workflow polish (recipe-driven replay, project
bootstrap, user-defined negative-preset catalogs, Civitai
trigger-word display), then lands two unlock-grade composition
wins: Flux Kontext + tiled denoise for hi-res reference edits,
and Flux animate (T5 + CLIP-L lerp, flow-match per frame).

### Top picks (3 features)

- **`plakat generate --recipe FILE.json`**. Replay any prior
  generation from its JSON sidecar. Recipe fields fill in only
  where the CLI didn't set the flag explicitly — `--model`,
  `--seed`, `--negative`, etc. pass through unchanged when you
  override them. Useful for "re-render at higher steps" /
  "swap one LoRA, keep everything else" iterations. The
  prompt is never overridden — the recipe is structural, the
  prompt is creative.

  ```bash
  # Re-render a v0.17 generation at higher quality
  plakat generate "$(cat in.prompt)" --recipe in.json --steps 50
  ```

- **Flux + SD3 WebP output**. v0.19 shipped WebP on SD-family
  only with a warn+fallback for the modern backbones. v0.20
  threads `--format png|webp` through the Flux and SD3
  pipelines too. WebP is ~30% smaller at perceptually-
  equivalent quality; the JSON sidecar still works on every
  backbone (so `plakat metadata` / `plakat clone` round-trip
  unchanged).

- **Civitai LoRA trigger-word display**. When a `--lora
  civitai:NNNNNN` resolves (cache hit or fresh download),
  plakat now prints the LoRA's trained trigger words inline:

  ```text
    ✦ Civitai LoRA 2595428 (v2614696) trigger words: watercolor_(medium), some_trigger
      → consider adding these to your prompt for the LoRA to activate
  ```

  Silent LoRAs (no apparent effect because triggers were
  missing from the prompt) is one of the most common
  Civitai-LoRA friction points; this surfaces the fix at the
  exact moment users need it.

### Round-out (4 features)

- **`plakat models aliases [--family F] [--repo] [--gated]`**.
  Enumerates every `--model` short-name plakat recognises,
  grouped by family. `--family flux` filters; `--repo` prints
  bare HF repo ids (pipes into `xargs plakat models pull`);
  `--gated` lists HF_TOKEN-only repos. Refactor: the
  hand-written alias `match` became a static `ALIAS_TABLE`
  so adding an entry updates both resolution and the listing.

- **`plakat init [DIR]`**. Bootstraps a runnable starter
  project — `scenario.hjson` (sd15, `enhancer: local`, two
  tasks), `wildcards/` (subject / style / lighting with three
  options each), and a focused `.gitignore`. Targets the
  ungated SD 1.5 + on-device LLM enhancer so first-run users
  with no HF token + no API key can generate end-to-end.
  Companion fix: `scenario`'s enhancer validator gained the
  `local` / `local:<alias>` / `auto` providers (previously
  cloud-only — the gap is why a fresh init scenario couldn't
  dry-run).

- **User-defined negative-preset catalogs**. Drop a `.txt` file
  into `<plakat-config-dir>/negative-presets/` and the
  filename becomes a `--negative-preset` name. User files
  override built-ins; safety-checked names; empty files fall
  through to the built-in. Error output marks entries as
  `<name> (user)` or `<name> (user override)`.

- **`--enhance-keep-original`**. New flag on `plakat generate`
  and `plakat portrait`: joins the enhancer's rewrite with
  the user's original prompt via the SD-family `BREAK`
  keyword (each chunk gets its own 77-token CLIP slot, so
  original terms aren't diluted by the enhancer's added
  detail). SD-family only by design; Flux / SD3 warn once
  (their T5 ignores BREAK and has the budget to carry both
  phrasings).

### Big swings — Kontext + tiled, Flux animate (2 features)

- **Flux Kontext + tiled denoise**. Lifts the v0.18 bail at
  the Kontext + `--tiled` junction. Each tile slices the
  matching region of the reference latent, packs it,
  seq-concats onto the tile's noise tokens, pads CN
  residuals for the reference half, runs forward, strips the
  reference tail. Per-tile RoPE budget check fires up front
  (Kontext + tiled doubles the per-tile sequence; the bail
  interpolates the largest safe `--tile-size` into the error
  message, typically ≤608 px for Kontext-dev).

  ```bash
  plakat generate "fold the dress into a flowing cape" \
      --model flux-kontext-dev --concept-image portrait.jpg \
      --size 2048x2048 --tiled --tile-size 512
  ```

- **Flux animate**. `plakat animate --model flux-dev` (and
  `--model flux-schnell`) now work. Pre-encodes both endpoint
  prompts through CLIP-L + T5-XXL **once**, then per frame:
  lerp the `(clip_pooled, t5_emb)` pair → run Flux's
  flow-match denoise → save. T5 encode dominates the cost,
  so amortising it across frames is the whole point of
  animate. New `pub fn animate_frame` on `flux::Pipeline`;
  Kontext / Fill / Canny / Depth refused (no place for a
  reference per call). Flux is guidance-distilled, so
  `--negative` is a no-op (warns) and `--guidance` is the
  scalar that goes straight to the model — drop to 3.5
  (Dev) / 0.0 (Schnell).

  ```bash
  plakat animate \
      --from "an oil painting of a fox in a meadow" \
      --to   "an oil painting of a cat in a meadow" \
      --frames 24 --seed 42 --guidance 3.5 \
      --model flux-dev --size 1024x1024 --out ./morph --gif
  ```

### Deferred to v0.21

- **SD3 / SD3.5 animate** — the three-encoder (CLIP-L +
  CLIP-G + T5) lerp + MMDiT rectified-flow integrator wiring
  is its own refactor. `plakat animate --model sd35-*` bails
  with a clear "deferred" message; Flux animate in v0.20 is
  the proving ground for the per-frame-encoding approach
  SD3 will follow.
- **AnimateDiff** — motion-adapter weights + temporal-attention
  injection into the SD UNet. Genuinely new architecture
  (not covered by candle 0.10.2); slated for v0.21+ as its
  own multi-cycle effort rather than rushed into v0.20.

569 lib tests green; +60 new tests across the cycle.

## What's new in v0.19 — local enhancer polish, partial-rerun, WebP, Kontext compositions

Nine features in three groups. Pairs with v0.18's larger surface
(Flux Kontext, A1111 attention on Flux+SD3, BREAK, local prompt
enhancer) — v0.19 sands down rough edges and unblocks the two
Kontext compositions deferred from v0.18.

### Top picks (3 features)

- **Enhancer CLI flag surface + disk cache**. The v0.18 local
  enhancer's internals get CLI flags: `--enhance-temp F` (default
  greedy / `0.0`), `--enhance-max-tokens N` (default 96),
  `--enhance-system PATH` (custom system prompt), and
  `--enhance-cache` (opt-in SHA-256 disk cache at
  `~/.cache/plakat/enhance/`). Cache hits skip the LLM forward
  entirely — scenarios re-enhancing the same prompts across runs
  go from ~3-5s per prompt to instant.
- **`plakat animate --resume`**. Long animates that crash on frame
  23 of 24 no longer require re-rendering all 24. The flag scans
  `<out>/frame-NNNN.png`, skips frames already on disk, re-runs
  only what's missing. Mirrors the scenario `--resume` pattern
  added in v0.17.
- **scenario `--only TASK[,TASK,…]` + `--limit N`**. Partial-rerun
  affordances for long batches. `--only` runs just the named
  tasks (typo'd names bail up front with the supported list);
  `--limit` runs the first N. Both compose with `--resume` and
  `--dry-run`. `seed_offset` advances on skipped tasks so a
  partial run produces seeds identical to the full batch — no
  drift when iterating.

### Round-out (4 features)

- **`plakat doctor --json`**. Structured CI / scripting output
  alongside the v0.18 health-check sections. Covers build /
  runtime device match, libcuda driver shim probe, HF cache disk
  usage. `jq` consumers can assert `.device.aligned == true` or
  `.cache.severity == "ok"`.
- **`--negative-preset photo | painting | anime | cinematic`**.
  Four bundled negative-prompt presets. Combine with `--negative`
  for preset-plus-user-extras. Saves the daily-driver
  `"blurry, low quality, watermark, ..."` copy-paste.
- **`plakat clone PNG`**. Reverse of `plakat metadata` (v0.18):
  reads a generated PNG's recipe + emits the `plakat generate`
  shell command that would re-create it. JSON sidecar preferred
  for lossless reproduction; falls back to parsing the Auto1111
  `parameters` chunk for Civitai uploads / A1111 outputs.
  `--one-line` for pipes.
- **WebP output format**. `--format png | webp` on
  `plakat generate`. WebP ships ~30% smaller files at perceptually-
  equivalent quality. Trade-off: WebP can't carry the Auto1111
  tEXt chunk (no drag-and-drop into A1111 / Civitai / ComfyUI);
  the JSON sidecar still works, so `plakat metadata` / `plakat
  clone` round-trip on WebP outputs. SD-family pipeline only in
  this release; Flux / SD3 warn and fall back.

### FLUX.1 Kontext composition unlocks (2 features)

- **Kontext + ControlNet**. Lifts the v0.18 phase 2 bail.
  ControlNet residuals (computed per-block from the CN forward on
  noise tokens) get zero-padded along the seq dim for Kontext's
  reference half before being added to the per-block flux
  intermediate state. The reference tokens get no CN contribution
  — they're already conditioning via cross-attention. Unlocks
  "edit this image, preserve the depth/canny structure" workflows.

  ```bash
  plakat generate "make it golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --control-spec 'depth:from=input.png:strength=0.7'
  ```

- **Kontext + Redux**. Lifts the v0.18 phase 2 bail with a RoPE
  budget gate. Total effective attention seq (txt + img + ref + N
  Redux tokens) is computed at dispatch; soft warn at 3500
  positions, hard bail at 4096 with actionable cleanup hints.
  Unlocks "edit this image in the style of these references" —
  Kontext provides the layout, Redux provides the aesthetic.

  ```bash
  plakat generate "the same scene at golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --redux-image style_ref.png:weight=0.5
  ```

### Two new tutorials

- [`SCENARIOS_TUTORIAL.md`](Documentation/Tutorials/SCENARIOS_TUTORIAL.md)
  — batch generation via HJSON. Cross-product expansion, per-task
  overrides, partial-rerun filters, real-world series-production
  examples.
- [`OUTPAINT_TUTORIAL.md`](Documentation/Tutorials/OUTPAINT_TUTORIAL.md)
  — `plakat outpaint INPUT.png`. Per-side flag grammar,
  VAE-snapped dimensions, model choice, iterative-stage workflow.

509 lib tests green; +40 new tests across the cycle.

## What's new in v0.18 — Flux Kontext, SDXL animate, BREAK, local enhancer, polish

The largest single-version cycle yet. Three workstreams plus a
follow-on wave of QoL features and three new tutorials.

### Top picks + round-out (7 phases)

- **A1111 attention syntax on Flux + SD3**. The v0.17 per-token
  weight broadcast (CLIP) now applies to T5-XXL hidden states on
  Flux and to all three penultimate streams on SD3 / SD3.5
  (CLIP-L + CLIP-G + T5). Every Civitai Flux LoRA card already
  embeds `(token:1.4)`-style emphasis in its example prompts;
  these now work as written. Sentencepiece alignment caveat
  documented.
- **SDXL `plakat animate`**. The prompt-morph animator (v0.17)
  extended from SD 1.5 / SD 2.1 to SDXL. Dual CLIP-L + CLIP-G
  hidden lerp, pooled `add_text_embeds` lerp, `add_time_ids`
  micro-conditioning threaded through.
- **Animate frame metadata**. Each `frame-NNNN.png` carries the
  Auto1111 `parameters` PNG tEXt chunk + a JSON sidecar with the
  lerp `t` parameter + `Animate from` / `Animate to` extras.
- **LCM-LoRA SD 1.5 `--fast` preset**. Same recipe as v0.17's
  `lcm-sdxl` against the smaller backbone — `--fast lcm-sd15` for
  4-step inference on SD 1.5 hardware.
- **`--grid` on `img2img` / `portrait` / `outpaint`**. The v0.17
  grid bundling now works on every `--count`-bearing subcommand.
  Per-backbone filename prefix preserved.
- **`--negative` attention verification**. Tests confirming the
  per-token weight broadcast works on the uncond branch across
  SD 1.5 / 2.1, SDXL, SD3.
- **`plakat doctor` enhancements**. Build / runtime device match,
  `libcuda.so.1` driver shim probe (Linux + `--features cuda`),
  HF cache disk usage report. Catches the CI-style "binary built
  with CUDA, no driver on host" silent fallback.

### FLUX.1 Kontext (BFL image editing)

Four phases bringing BFL's image-editing Flux variant online:

- **`--model flux-kontext-dev`** on `plakat generate` and
  `plakat img2img`. Reference image is VAE-encoded and
  sequence-concatenated onto the noise tokens (with
  `img_ids[..., 0] = 1` as the RoPE marker) — distinct mechanism
  from Canny/Depth which widen `img_in` to 128 channels.
- **`--concept-image PATH`** reused as the reference flag (same
  grammar as Canny/Depth, semantically the "image to edit"). On
  `plakat img2img`, the input positional becomes the reference
  natively.
- **GGUF support** via `unsloth/FLUX.1-Kontext-dev-GGUF`
  (`--model flux-kontext-dev-gguf`). Composes with LoRA (Kontext
  shares Dev's transformer layer names) and `--quantize-t5`.
- **`--kontext-bucket`** opt-in flag — snaps `--size` to the
  closest of 17 BFL-recommended Kontext resolutions before VAE
  encoding (off by default, surprise-free for non-Kontext flows).

### Follow-on wave (6 features)

- **`plakat metadata FILE.png`**. New subcommand reads the v0.17
  `parameters` PNG tEXt chunk + JSON sidecar back into the
  terminal. `--json-only` pipes cleanly to `jq`.
- **`--aspect`** on `plakat img2img`. Resolution priority:
  `--size > --aspect + --base > input image dims`. Composes with
  `--kontext-bucket`.
- **`plakat scenario --dry-run` polish**. The summary line now
  reads `(dry-run) would have generated …` instead of `✓ done`,
  and per-task previews show the output directory path so you can
  see file layout before launching a long batch.
- **A1111 inline `<lora:NAME[:weight]>` syntax**. Civitai LoRA
  cards embed these directly; plakat extracts them at the CLI
  boundary, parses via the v0.17 `LoraSpec` grammar (paths /
  HF repos / `civitai:NNN` shorthand), prepends to the LoRA
  stack, removes from the prompt before encoding.
- **`BREAK` keyword in prompts**. A1111 convention for chunking
  past CLIP's 77-token cap. Each chunk gets its own 77-token
  CLIP context; hidden states sequence-concatenate before
  cross-attention. SD 1.5 / 2.1 / SDXL; Flux + SD3 strip + warn
  (their T5 already has a 256/512-token budget).
- **Local prompt enhancer**. `--enhance local` runs a small
  instruction-tuned LLM in-process via candle's quantized
  backends. Qwen2.5-1.5B-Instruct (Q4_K_M, ~1 GB) as default,
  SmolLM2-360M (~230 MB) as CPU-budget fallback. Greedy decoding
  for reproducibility; `--enhance auto` picks DeepSeek → Gemini →
  local based on what env vars are set. No API key required for
  the local arm.

### Three new tutorials

- [`ADVANCED_PROMPTING_TUTORIAL.md`](Documentation/Tutorials/ADVANCED_PROMPTING_TUTORIAL.md)
  — attention syntax, BREAK, inline `<lora:>` as a coherent set.
- [`PROMPT_ENHANCER_TUTORIAL.md`](Documentation/Tutorials/PROMPT_ENHANCER_TUTORIAL.md)
  — `--enhance deepseek / gemini / local / auto`.
- [`METADATA_TUTORIAL.md`](Documentation/Tutorials/METADATA_TUTORIAL.md)
  — recovering recipes from PNG metadata.

465 lib tests green; +84 new tests across the cycle.

## What's new in v0.17 — the prompt + reproducibility release

Ten phases focused on **prompt expressiveness**, **reproducibility**,
and **animation**. The cycle also upgrades the underlying candle ML
framework two minor versions and adds the long-asked `--lora civitai:`
shorthand:

- **A1111 prompt syntax**. `(red:1.4)` emphasis / `[blue]`
  de-emphasis / `((nested))` compounding / `\(escape\)` — the
  grammar every Civitai LoRA card uses in its example prompts.
  Applied to CLIP penultimate hidden states via per-token broadcast.
  SD 1.5 / SD 2.1 / SDXL.
- **PNG metadata + JSON sidecar**. Outputs ship with the
  Auto1111-compatible `parameters` PNG tEXt chunk + a sibling
  `<filename>.json` carrying the full recipe. A1111 / Civitai /
  ComfyUI / sd-prompt-reader all surface the prompt + seed + LoRAs
  + scheduler inline. `--no-metadata` opts out.
- **`--grid` output**. `--count N > 1` + `--grid` writes a single
  `plakat-grid-<seed>.png` combining all N outputs in a near-square
  layout. `--grid-cols` / `--grid-padding` for fine control.
- **`plakat animate`**. New subcommand for prompt-morph animation:
  lerp CLIP embeddings between two prompts at a fixed seed,
  producing a smooth N-frame sequence. `--gif` bundles into an
  animated GIF. SD 1.5 / SD 2.1.
- **Live preview during denoise**. `--preview-every N` writes a
  cheap latent-projection PNG every N steps so long runs aren't a
  black box. Microseconds per write — no meaningful runtime cost.
- **scenario `--resume` / `--force`**. Crashed scenario picks up
  where it left off by probing for already-existing output PNGs.
  No more restart-from-task-0.
- **`--lora civitai:NNNNNN`**. Skip the explicit
  `plakat civitai download` step — the LoRA spec parser now
  downloads + caches Civitai assets on first use via the
  shorthand. `civitai-version:NNNNNN` pins a specific version.
- **LCM-LoRA SDXL `--fast lcm-sdxl`**. Latent-Consistency
  distillation for SDXL bundled with the right scheduler and
  4-step / CFG-1.5 defaults. ~5× speedup over stock SDXL.
- **candle 0.8 → 0.10.2 upgrade**. Single 8-line trait-impl fix
  for `SimpleBackend::get_unchecked`. GGUF / NF4 / MMDiT /
  vendored Flux all intact, 304 tests still green at upgrade
  time.
- **SDXL refiner cleanup**. The "Known limitation" about missing
  `add_embedding` on the refiner was outdated since v0.11 phase
  8e. Stale docs replaced; regression tests pin the 5-time-id
  config so future refactors can't silently break the refiner's
  `text_time` micro-conditioning.

381 lib tests green; +77 new tests across the cycle.

## What's new in v0.16 — the productivity release

A dozen quality-of-life landings that connect community workflows
(Civitai browsing, ADetailer face fix, Hires fix, wildcards) to the
existing plakat backbone, plus deeper SD3 integration:

- **SD3 ControlNet (InstantX)**. `--control-spec` works on SD3 /
  SD3.5 via the InstantX adapter family. Multi-CN composition,
  step-gating, auto-annotation from a reference photo — same
  ergonomics SDXL + Flux ControlNet ship.
- **Tiled Flux Fill**. `--tiled` composes with Flux.1-Fill-dev for
  4K+ inpaint. Per-tile masked-latent + mask packing.
- **Tiled SD3 img2img + inpaint**. The rectified-flow init lerp +
  RePaint mask blend compose with the per-tile Hann blend.
- **Wildcards**. `{red|blue|green}` inline alternation +
  `__name__` file wildcards (Auto1111 / NovelAI grammar). Seeded
  from `--seed` for reproducibility.
- **CLIP-skip**. `--clip-skip N` for SD 1.5 / SD 2.1 — N=2 is the
  community default for anime checkpoints.
- **ADetailer-style face refinement**. `--adetailer` runs SCRFD
  on each output, crops + img2img-refines each face, feather-
  composites back. Reuses the t2i SdCore — no extra model load.
- **Hires fix**. `--hires-fix` escapes the trained-resolution
  ceiling: upscale (Lanczos / Real-ESRGAN) + img2img-refine.
  Composes with `--adetailer` for a 4K → fixed faces pipeline.
- **Civitai browser + downloader**. `plakat civitai search`,
  `info`, `download` — drop the resulting path into `--lora` /
  `--model`. Atomic streaming downloads with cache-hit
  short-circuit.
- **Auto-annotation for Flux concept variants**. `--concept-from
  PATH` auto-annotates a photo through Canny / Depth before feeding
  Flux.1-Canny-dev / Flux.1-Depth-dev.
- **SD3 pipeline caching + per-task LoRA**. Scenarios with
  `--model sd35-*` now share one SD3 pipeline across tasks; per-
  task `loras:` swap at runtime via the LoraLinear stack.
- **Textual Inversion** *(partial)*. Parser + `plakat embedding
  info` inspector. Runtime injection blocked by candle 0.8's
  private `clip::Config.vocab_size` — wiring lands when the
  candle API surface opens or alongside a vendored CLIP path.
- **SD UNet per-task LoRA preflight** *(partial)*. Detects the
  blocker upfront and emits actionable YAML-fold hints; bails
  loud with three concrete workarounds. Full UNet vendoring
  deferred — same candle private-internals blocker.
- **XLabs Flux IP-Adapter parser** *(partial)*. Inspector that
  reports per-block attention count + SigLIP/Flux dims. Per-block
  injection blocked by Flux's private `double_block_forward`;
  use `--redux-image` for working image conditioning today.

## What's new in v0.15 — runtime LoRA + SD3 maturation

- **Per-task LoRA in scenarios**. `tasks: [{ loras: [...] }]` applies
  and clears LoRAs between tasks at runtime — no model reload.
  Composes with the scenario-level LoRA set. Flux (BF16 / GGUF / NF4).
- **NF4 + ControlNet**. NF4 Flux composes with `--control-spec` via
  the residual-aware forward — same residual interleave the BF16 and
  GGUF backbones use, so a single CN checkpoint works on all three.
- **SD3 / SD3.5 img2img + inpaint**. RePaint-style inpaint with
  per-step mask blend, rectified-flow truncated schedule. Works
  across the lineup (Medium / Large / Turbo).
- **SD3 / SD3.5 LoRA**. Diffusers PEFT format, MMDiT-targeted keys.
- **Flux Canny-dev / Depth-dev variants**. BFL "concept" Flux
  checkpoints with conditioning baked into the 128-channel `img_in`.
  Pass `--concept-image PATH` with `--model flux-canny-dev`.
- **Tiled SD3**. MultiDiffusion-style tiled denoise for MMDiT —
  1024-px tiles work on every SD3 variant within the variant's
  `pos_embed_max_size` cap.
- **Scenario ↔ generate sync**. Per-task `fast`, `concept-image`,
  `enhance`, `tiled` overrides.
- **Two new tutorials**:
  [`FLUX_TUTORIAL.md`](Documentation/Tutorials/FLUX_TUTORIAL.md)
  walks through the Flux feature set end-to-end;
  [`SD3_TUTORIAL.md`](Documentation/Tutorials/SD3_TUTORIAL.md) does
  the same for the SD3 / SD3.5 family.

## What's new in v0.14 — the SD3.5 + NF4 + Redux release

- **Stable Diffusion 3 / 3.5 (MMDiT)**. New family — `sd35-medium`,
  `sd35-large`, `sd35-large-turbo`, `sd3-medium`. Triple text encoder
  (CLIP-L + CLIP-G + T5-XXL), 16-channel VAE, rectified-flow sampler
  with SD3 time-shift. CFG via `[neg, pos]` double-batch.
- **NF4 quantized Flux**. `--model flux-dev-nf4` loads lllyasviel's
  bitsandbytes NF4 pack — ~6 GB transformer at inference (4× weight
  savings vs BF16), pure-CPU dequant codec means it runs on any
  candle device. Phase 8b adds **NF4 + LoRA composition** via the
  same selective-dequant trick GGUF uses.
- **Flux Redux**. `--redux-image PATH` adds image conditioning via
  SigLIP-so400m + BFL's Redux adapter (729 tokens → seq-concat onto
  T5). Repeatable for multi-image stacks (`--redux-image
  style.png:weight=0.8 --redux-image subject.png:weight=0.5`). Cap
  of 4 with attention-cost guardrails. Composes with GGUF, NF4,
  LoRA, ControlNet, img2img, tiled.
- **Tiled SD 1.5 / 2.1**. `--tiled` now supported on the smaller
  SD backbones too (was SDXL-only in v0.12).
- **Flux Fill + ControlNet**. `plakat img2img --model flux-fill-dev
  --mask ... --control-spec depth:from=...` composes with the
  auto-annotator and multi-CN.
- **Hyper-FLUX / FLUX-Turbo presets**. `--fast hyper-8 | hyper-16 |
  turbo-alpha` bundles the matching distillation LoRA + recommended
  step count + guidance in one flag.
- **Shared SdCore**. Scenarios with mixed t2i + img2img tasks now
  load the SD backbone **once** (was: per-task). The t2i Pipeline's
  `Arc<SdCore>` is reused by img2img via the existing `from_core`
  path.

## What's new in v0.13 — the Flux modernization release

- **Quantized Flux (GGUF)**. Run FLUX.1-dev on 16 GB GPUs.
  `--model flux-dev-gguf` loads the 4-bit transformer (~7 GB vs ~24 GB BF16).
  `--quantize-t5` drops T5-XXL to ~3 GB. `--quant-level Q5_K_M` picks a
  different precision (Q2_K..F16 supported); same for `--t5-quant-level`.
- **Flux LoRA on quantized**. Diffusers PEFT and AI-Toolkit / kohya
  formats both compose with the GGUF backbone — affected Linears are
  dequantized once at load, rest of the model stays 4-bit.
- **Flux Inpainting**. `--model flux-fill-dev` + `--mask` runs BFL's
  dedicated 384-channel inpaint checkpoint via `plakat img2img`.
- **Flux Img2Img**. Rectified-flow init: `plakat img2img init.png
  --model flux-dev --strength 0.7 --prompt "..."`.
- **Tiled Flux denoise**. MultiDiffusion-style 2K–4K outputs on any
  Flux variant: `--tiled --tile-size 1024 --tile-stride 768`. Composes
  with ControlNet (per-tile residuals) and the tiled VAE decode.
- **Flux ControlNet polish**. Auto-annotators wire through to Flux
  (`--control-spec depth:from=photo.jpg` is now a one-liner). Step gating
  via `start=…:end=…`. Multi-Flux-CN with summed residuals.
- **Outpainting**. New `plakat outpaint` subcommand expands a canvas
  and hands off to the inpaint pipeline (SDXL-Inpaint, SD15-Inpaint, or
  Flux.1-Fill-dev).
- **Scenarios**. Every v0.13 feature above is now expressible in
  scenario HJSON: `quant-level:`, `t5-quant-level:`, `tiled:`,
  per-task `init-image:` / `mask:` / `strength:` / `outpaint:`, plus
  multi-CN via `controls: [...]`.

