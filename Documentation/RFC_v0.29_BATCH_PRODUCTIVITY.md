# RFC v0.29 — Batch productivity completion

**Status:** decisions locked 2026-05-28 — ready for phase 0.

**Predecessors:**
- [`RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md`](RFC_v0.28_ANIMATEDIFF_PRODUCTIVITY.md) — the four single-script productivity wins.
- [`RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md`](RFC_v0.27_ANIMATEDIFF_COMPLETENESS.md) — feature completeness across SD 1.5 + SDXL.

## 1. TL;DR

v0.28 made AnimateDiff pleasant to use in single Bund scripts. v0.29
makes it pleasant **at production scale** (HJSON scenarios) and closes
the two smaller v0.28 deferrals that completed the productivity picture:

1. **Animate in scenarios.** The largest unaddressed plakat batch-
   driver gap — `cli/scenario.rs::TaskDef` has zero animate-related
   fields today. Users who want "render 50 animate variants" have no
   surface; they fall back to shell loops.

2. **`animate_format` config key.** The only v0.28 surface gap in
   Bund scripting — `plakat.animate` always writes PNG frames; can't
   produce GIF/MP4/WebM from inside a script.

3. **SDXL `plakat.animate`.** v0.28 deferred SDXL animate in scripting
   because there was no shared cache slot for the ~7 GB SDXL backbone.
   Adds `loaded_animatediff_sdxl` mirroring the v0.26 stylize slot
   pattern.

Six phases, ~6 sessions total. Quality themes (per-frame video CN,
FreeNoise long-form, per-layer motion splice) defer again to a
dedicated quality cycle.

## 2. Why this is the v0.29 cycle

1. **The scenario gap is the biggest plakat productivity hole.**
   Every other major workflow has scenario batch coverage: t2i,
   img2img, portrait, stylize, outpaint. Animate doesn't. After v0.28
   wired the per-script surface, the per-batch surface is the natural
   next step.

2. **Bund parity completes.** v0.28 missed only one config key
   (`animate_format`). The fix is ~30 minutes. Worth fitting into a
   cycle while the surface is hot.

3. **SDXL scripting parity closes the v0.28 surface.** Last item on
   the v0.28 "Deferred to v0.29+" list that doesn't need new
   architecture.

4. **Quality themes can wait.** Per-frame video ControlNet, FreeNoise
   long-form, per-layer motion splice — all real deferrals from
   v0.27/v0.28. None blocked by anything. A dedicated quality cycle
   (v0.30+) can land all three together when there's appetite for the
   bigger swing.

## 3. Decisions locked (2)

User answered via AskUserQuestion 2026-05-28:

### 3.1 Cycle shape — **Lean productivity completion (A + B + D)**

Animate in scenarios + animate_format key + SDXL plakat.animate.
~6 sessions. Closes every "Deferred to v0.29+" line from v0.28 that
doesn't need new architecture. Natural follow-on to v0.28's
productivity theme.

### 3.2 Must-haves — **All three items**

A (animate in scenarios), B (animate_format), D (SDXL plakat.animate)
all locked. None optional.

## 4. Phase plan

| # | Phase | Sessions |
|---|---|---|
| 0 | `animate_format` config key + Bund format dispatch | 0.5 |
| 1 | SDXL `plakat.animate` cache slot | 2 |
| 2 | Animate `TaskDef` + HJSON schema | 1.5 |
| 3 | Animate dispatch in scenario runner | 1.5 |
| 4 | Tutorials + integration tests | 0.5 |
| 5 | Cycle close-out | 0.5 |

Total: ~6 sessions.

Phase order rationale: smallest/lowest-risk first (phase 0), then
SDXL scripting (which establishes the SDXL animate cache slot
pattern that scenarios may want to reuse), then the meatiest piece
(scenarios in two sub-phases — schema before dispatch), then
docs + close-out.

## 5. Phase 0 — `animate_format` config key

Single config key added to `GenerationConfig`:
```rust
pub animate_format: video::Format,
```
Default: `Frames`. Set via `plakat.config.set` with strings
`"frames"|"gif"|"mp4"|"webm"|"all"`.

`plakat.animate` reads the key after generating per-frame
`DynamicImage`s and dispatches:
- `Frames` → existing PNG write (unchanged)
- `Gif` → call existing `cli::animate::write_gif` on the frame paths
- `Mp4` / `Webm` → ffmpeg via `imaging::video::frames_to_mp4` /
  `frames_to_webm` (same path as CLI animate uses)
- `All` → all four

ffmpeg availability check (`imaging::video::ffmpeg_version`) fires
once at script start when `animate_format.needs_ffmpeg()`.

## 6. Phase 1 — SDXL `plakat.animate` cache slot

Add `loaded_animatediff_sdxl: Option<(String, AnimateDiffSdxlPipeline)>`
on `ScriptCtx`, mirroring the v0.26 `loaded_stylize` pattern. New
helper:
```rust
fn get_or_load_animatediff_sdxl(&mut self, alias: &str)
    -> Result<&AnimateDiffSdxlPipeline>;
```

`plakat.animate` dispatch:
- If `ctx.loaded_model()` matches an SDXL alias → use SDXL slot
- If SD 1.5 → use existing path (no SDXL load)
- LoRA stack mutation drops the slot via `mark_loras_changed`
  (same as `loaded_stylize`)

The SDXL pipeline is ~5 GB on cold load — the cache slot pays off
even on single-script use because Bund scripts often chain calls.

## 7. Phases 2 + 3 — Animate in scenarios

### Phase 2: `TaskDef` + HJSON schema

Extend `cli/scenario.rs::TaskDef` (or add `AnimateTaskDef`
discriminated by `type: "animatediff"`) with:

- `type: String` (new — defaults to `"generate"`; `"animatediff"`
  is the new value)
- `from: Option<String>` — animate prompt (mirrors `--from`)
- `frames: Option<u32>`
- `window_size: Option<u32>`
- `window_overlap: Option<u32>`
- `lcm: Option<bool>`
- `motion_lora: Option<Vec<String>>` — LoraSpec strings, repeatable
- `motion_lora_scale: Option<f32>`
- `format: Option<String>` — `"frames"|"gif"|"mp4"|"webm"|"all"`
- `gif_delay_ms: Option<u16>`
- Existing CN family already supported via `control:` / `controls:`
  fields — wire those through to animate too

Scenario-level defaults for all of these compose with per-task
overrides (standard scenario pattern).

### Phase 3: Animate dispatch in scenario runner

Wire `scenario::run_one_task` to detect `type: "animatediff"` and
dispatch through `AnimateDiffPipeline::generate_long` or
`AnimateDiffSdxlPipeline::generate_long`. Per-task seed derivation,
output dir is `<scenario_out>/<task_name>/` (mirrors the existing
per-task layout), ControlNet stack resolved via
`load_control_stack`.

`--resume` / `--only` / `--limit` / `--dry-run` semantics:
- `--resume` detects existing `<task>/frame-0000.png` and skips
- `--only NAMES` filters animate tasks the same way it filters
  generate tasks
- `--limit N` honoured
- `--dry-run` prints what would run without invoking the pipeline

## 8. Sample HJSON after phase 3

```hjson
{
  model: sd15
  type: animatediff       # scenario default — all tasks animate
  lcm: true               # 4-step AnimateLCM default
  frames: 16
  size: 512x512
  format: mp4
  tasks: [
    { name: cottage, from: "watercolor cottage at dawn" }
    { name: knight,  from: "knight in forest, oil painting" }
    { name: fox,     from: "fox in snowy meadow",
      controls: ["depth:image=./fox-depth.png:strength=0.7"] }
    { name: hero,    from: "hero on a cliff at sunset",
      frames: 32,  format: webm }     # per-row overrides
  ]
}
```

Renders 4 animations (4 sub-dirs of out_dir, each with frame PNGs
+ a chosen video format). Same scenario-level → per-task override
semantics as v0.25 onwards.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| **HJSON schema growth** | TaskDef already accepts many optional fields. Adding 8–10 more doesn't change the shape; existing tests stay green. Use serde defaults so existing scenarios keep working unchanged. |
| **SDXL animate cache slot collision** | The slot holds a 5+ GB pipeline. Alias change drops it. Document the memory footprint in the tutorial; suggest `plakat run` over `--repl` for SDXL animate work. |
| **animate_format ffmpeg requirement on CI** | The unit tests don't exercise real ffmpeg (the bail path is checked). Same approach v0.26 phase 5 took. |
| **Scenario animate compose with scenarios' resume/only/limit** | These are well-tested for `generate` tasks; reuse the same filter machinery for animate. |

## 10. What's NOT in v0.29

Deferred to v0.30+ unless user appetite signals otherwise:

- **Per-frame video ControlNet (`--control-video PATH`)** — biggest
  user-requested animator feature still on the list. Reads a video,
  per-frame annotates, per-frame CN residuals. ~4 sessions.
- **FreeNoise / FreeInit long-form** — shared-noise quality upgrade.
  RFC v0.27 §11 deferral. ~3 sessions.
- **Per-layer motion splice** — RFC v0.27 §3.2 quality escalation.
  ~4 sessions.
- **HotShot-XL** — different architecture; own cycle if pursued.
- **AnimateLCM-SDXL** — upstream repo still not publicly accessible.

## 11. Acceptance criteria

v0.29 ships when:

- [ ] `plakat run` of a Bund script using
  `"mp4" "animate_format" plakat.config.set` produces an `mp4`
  video alongside frames.
- [ ] `plakat run` against a script with `"sdxl" plakat.load`
  followed by `plakat.animate` produces SDXL animate output
  without bailing.
- [ ] `plakat scenario my-anim.hjson` with `type: animatediff`
  tasks renders the expected per-task animate outputs.
- [ ] `--resume` on the same scenario skips already-rendered tasks.
- [ ] 953+ → ~970 lib tests; new integration tests for each phase.
- [ ] Tutorials + reference updated.

## 12. Out-of-scope decisions for this RFC

Resolved inside phases:

- Whether to make `animate_format` a single key or split into
  `animate_format_video` + `animate_format_frames` for finer
  control (likely single key for simplicity).
- Per-task vs scenario-level handling of motion_loras stacking
  (mirror the existing controls behavior).
- Output sub-dir naming for animate scenario tasks (likely
  `<task_name>/` matching the generate pattern).
