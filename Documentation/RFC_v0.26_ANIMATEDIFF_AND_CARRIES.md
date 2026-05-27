# RFC v0.26 — AnimateDiff + carries closeout

**Status:** decisions locked 2026-05-27 — ready for phase 0.

**Predecessors:**
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) — 7-word MVP.
- [`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md) — 28-word expansion.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) — v0.22 deferrals.
- [`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md) — persona depth + scripting completion.
- [`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md) — art-medium presets + auto-LoRA discovery.

## 1. TL;DR

v0.26 takes the **AnimateDiff swing** that's been carrying since
v0.20, ships **SD3 / SD3.5 animate**, and closes **every v0.25
carry** in one release. After v0.26 the deferral backlog starts
empty (modulo whatever emerges during the cycle).

Three themes:

- **Theme A — Animation (phases 1-6):** AnimateDiff V3 on SD 1.5
  + SD3 / SD3.5 animate via the proven v0.20 Flux animate pattern.
  Motion LoRA composition. GIF + MP4 + WebM + PNG frame outputs.
- **Theme B — v0.25 completion (phases 10-12):** Bund
  `plakat.look.*` / `plakat.genre.*` apply on Flux + SD3 paths
  (state already sets in v0.25; apply currently SD-family only).
  Scenario auto-LoRA discovery integrated with the two-stage
  scenario LoRA pipeline.
- **Theme C — Incremental wins (phases 7-9):** `plakat.stylize`
  cache slot (one-shot 5 GB load today). `plakat.save` JSON
  sidecar + PNG tEXt + `plakat.metadata.write`. Real-ESRGAN in
  the standalone `plakat.upscale` word (already exposed via
  `plakat.hires`).

Net surface delta: ~48 → ~50 Bund host words (the
`plakat.metadata.write` addition; `plakat.upscale` gains ML
methods; animate is a new CLI subcommand augmenting the existing
`plakat animate` from v0.20).

Estimated 14 phases / ~15 sessions. Largest cycle since v0.22.

## 2. Why this is the v0.26 cycle

1. **AnimateDiff is overdue.** It's been carrying since v0.20 as
   "the natural next big swing." v0.21/22/23/24 closed the
   scripting arc; v0.25 added the medium-and-genre axes. v0.26
   has the model-architecture bandwidth.

2. **Animation has a coherent design space.** AnimateDiff (SD 1.5
   temporal-attention) and SD3 animate (3-encoder lerp +
   flow-match per frame) share the "produce N frames at a fixed
   seed" surface. Bundling them means one design pass on output
   formats, frame budgets, and `plakat animate` flag shape.

3. **v0.25 carries are small + scattered.** Five of the seven
   carries are 0.5–1.5 session items each. Bundling them with
   AnimateDiff means the cycle ships with zero residual backlog.
   Splitting would mean a tiny v0.26 + a v0.27 = same total
   work, less polish per release.

4. **Sets up the next architecture push.** With AnimateDiff +
   SD3 animate done, v0.27 can either take a quiet polish cycle
   or pick the next architecture target (Flux ControlNet
   evolution, SVD video, multi-GPU split — TBD).

## 3. Architectural constraints we keep

The seven from prior RFCs, plus one new:

1. Built-our-own VM; restricted stdlib.
2. Singleton context; one script per process.
3. Async bridge via `block_in_place`.
4. v0.22 relaxed-compat carries forward.
5. SD-family two-slot cache (SdT2i + SdPortrait) — `plakat.stylize`
   gets its own slot mirroring this pattern.
6. **v0.25:** Presets are layering, not replacement. Override-only
   on sampler fields; compositional on prompt/negative; discovery
   gated on `loras.is_empty()`.
7. **NEW: Animation is a per-call output, not state.** AnimateDiff
   and SD3 animate produce N-frame batches per call; no animation
   state lives on `ScriptCtx`. The frame count + fps come from
   CLI flags or the call-site bund word — not from a config key
   the user might set once and forget. Matches the v0.20 Flux
   animate pattern.

## 4. The deliverables

### 4.1 AnimateDiff (Theme A, phases 1-5)

| Item | What ships |
|---|---|
| `MotionAdapter` loader | Loads `guoyww/animatediff-motion-adapter-v1-5-3` weights. ~1-2 GB safetensors download on first use; cached. New `src/pipelines/motion_adapter.rs`. |
| Temporal-attention integration | Splices temporal-attention blocks into the SD 1.5 UNet's `Conv3D` / cross-attention path. Modifies `src/pipelines/unet/` to thread frame index through forward. |
| N-frame sampling loop | Reuses scheduler infra; samples `latents: [N_FRAMES, 4, H/8, W/8]` instead of `[1, ...]`. Default N=16 (Q3). |
| `--motion-lora SPEC` | Wires motion LoRAs onto the motion-adapter weights at load time. Reuses `LoraSpec` grammar from v0.17. |
| Output: `--format {gif,mp4,webm,frames}` | GIF via the existing `image` + `gif` crates; MP4 + WebM via either `Command`-driven ffmpeg or a Rust crate (decision deferred to phase 5 PR review). `frames` writes individual PNGs. |
| CLI: `plakat animate --animatediff` | Extends the existing v0.20 `plakat animate` (Flux + SD-family lerp) with an `--animatediff` mode flag. Same `--frames`, `--fps`, `--seed`, `--prompt`. |

**Scope cap (Q2):** SD 1.5 only for AnimateDiff. SDXL motion
adapters exist (`guoyww/animatediff-motion-adapter-sdxl-beta`)
but are less mature; deferred to v0.27.

### 4.2 SD3 / SD3.5 animate (Theme A, phase 6)

| Item | What ships |
|---|---|
| Three-encoder lerp | Lerp T5 + CLIP-L + CLIP-G text embeddings between two prompts. Widens the v0.20 Flux animate pattern (T5 + CLIP-L) by one encoder. |
| MMDiT flow-match per frame | Each frame uses lerped embeddings + the rectified-flow scheduler. Reuses the v0.14 SD3 scheduler infra. |
| CLI: `plakat animate --model sd35-medium` | Same flags as v0.20 Flux animate (`--prompt-a`, `--prompt-b`, `--frames`, `--fps`). |

### 4.3 v0.25 completion (Theme B, phases 10-12)

| # | Item | Today's state | What ships |
|---|---|---|---|
| B1 | Bund `plakat.look.*` apply on Flux | State (`ctx.look_name`) sets in v0.25; apply at SD-family only | Wire `apply_presets_with_discovery` into `generate_one`'s Flux branch in `script_entry.rs` |
| B2 | Same for SD3 generate path | Same as B1 | Wire into the SD3 branch of `generate_one` |
| B3 | Scenario auto-LoRA discovery | `lora_query` ignored in scenarios for v0.25 (two-stage pipeline) | Fire discovery at scenario load time keyed by `(look/genre, base_model)`; cache result across all tasks. Push into `task.loras` for tasks where `ctx.loras` is empty. |

### 4.4 Incremental wins (Theme C, phases 7-9)

| # | Item | Today's state | What ships |
|---|---|---|---|
| C1 | `plakat.stylize` cache slot | One-shot load per call (~5 GB) | New `loaded_stylize: Option<stylize::Pipeline>` slot on `ScriptCtx`. `mark_loras_changed` clears it. Mirror of the v0.23 SdT2i slot pattern. |
| C2 | `plakat.save` sidecar + PNG tEXt | `plakat.save` writes PNG-only today | Extend to write `<name>.json` sidecar (existing `GenerationMetadata` format) AND embed the A1111 `parameters` tEXt chunk in the PNG. v0.17 CLI `plakat generate` already does this; lift the helper. |
| C3 | `plakat.metadata.write` | Deferred from v0.24 | New host word: `( handle path -- )` re-extracts metadata from the saved file's context and re-attaches to a new file. Useful for re-saving after edits. |
| C4 | Real-ESRGAN in `plakat.upscale` | Lanczos x2/x4 only | Reuse `crate::imaging::upscale::ml_upscale` (already wired in `plakat.hires`). Accept `real-esrgan-x2`, `real-esrgan-x4`, `real-esrgan-anime-x4` as scale arg variants. |

### 4.5 Docs + tests (phase 13)

Mirror of v0.22 phase 12 / v0.23 phase 8 / v0.24 phase 10 / v0.25
phase 11:
- New `Documentation/ANIMATEDIFF.md` (full AnimateDiff reference)
- Animate tutorial updated (covers AnimateDiff + SD3 animate)
- `SCRIPTING.md` update for new host words + config keys
- `SCRIPTING_TUTORIAL.md` §13 "What's new in v0.26"
- Composition tests + integration tests covering AnimateDiff
  smoke + SD3 animate smoke + all carries

## 5. Cycle scope

~14 phases including hygiene + docs:

| Phase | Deliverable | Theme | Est. |
|---|---|---|---|
| 0 | RFC + branch hygiene (✓ — branch cut, version bumped) | — | done |
| 1 | AnimateDiff V3 motion-adapter loader + weight resolution | A | ~1 session |
| 2 | Temporal-attention integration into SD 1.5 UNet | A | ~1.5 sessions |
| 3 | N-frame sampling loop + scheduler adaptation | A | ~1 session |
| 4 | Motion LoRA composition (`--motion-lora`) | A | ~0.75 sessions |
| 5 | Output: GIF + MP4 + WebM + PNG frames | A | ~1.5 sessions |
| 6 | SD3 / SD3.5 animate (3-encoder lerp + MMDiT flow-match) | A | ~2 sessions |
| 7 | `plakat.stylize` cache slot (mirror SdT2i pattern) | C | ~1 session |
| 8 | `plakat.save` sidecar + PNG tEXt + `plakat.metadata.write` | C | ~1.5 sessions |
| 9 | Real-ESRGAN ML upscaling in `plakat.upscale` | C | ~0.5 sessions |
| 10 | Bund look/genre apply on Flux generate path | B | ~0.75 sessions |
| 11 | Bund look/genre apply on SD3 generate path | B | ~0.75 sessions |
| 12 | Scenario auto-LoRA discovery (once per look+base) | B | ~1.5 sessions |
| 13 | Docs + composition tests + release notes | — | ~1 session |

**Total estimate:** ~14.75 sessions. About 1.5× v0.25 (9–10
sessions) but the bulk is concentrated in AnimateDiff (phases 1-5
account for ~5.75 sessions on their own).

## 6. Decisions (locked 2026-05-27)

### Q1: AnimateDiff variant

Options:
- **A. V1** — original (legacy, lower quality)
- **B. V2** — `guoyww/animatediff-motion-adapter-v1-5-2`, most-tested
- **C. V3** — `guoyww/animatediff-motion-adapter-v1-5-3`, latest
- **D. V2 + V3 both** — ship as `--animate-variant {v2,v3}`

**Locked: C.** V3 ships. Community-favorite; longest training
window; supports Domain Adapter LoRAs cleanly. ~1-2 GB weights.

### Q2: Base coverage

Options:
- **A. SD 1.5 only** — tightest first cut
- **B. SD 1.5 + SDXL** — both families
- **C. SDXL only** — modern target

**Locked: A.** SD 1.5 only. The SDXL motion adapter
(`guoyww/animatediff-motion-adapter-sdxl-beta`) is less mature
and the community-LoRA ecosystem skews heavily SD 1.5. SDXL
animation deferred to v0.27.

### Q3: Frame count + FPS defaults

Options:
- **A. 16 frames @ 8 fps** — AnimateDiff training convention
- **B. 24 frames @ 12 fps** — standard video feel
- **C. No default, require `--frames`** — force explicit

**Locked: A.** 16 frames @ 8 fps. Matches the motion adapter's
training window — extending past 16 frames degrades quality.
Users override via `--frames N --fps F`.

### Q4: Output formats

Options:
- **A. GIF only** — smallest scope
- **B. GIF + PNG frames** — quick share + compositing
- **C. GIF + MP4 + WebM + PNG frames** — full set

**Locked: C.** Full set. Adds one phase of work for the video
encoders (decision in phase 5: ffmpeg via `Command` vs Rust
crates). Comprehensive output coverage; users with editing
workflows get MP4 / WebM directly without post-processing.

### Q5: Motion LoRA composition

Options:
- **A. Support motion LoRAs** — `--motion-lora SPEC`
- **B. Base motion adapter only** — no LoRAs
- **C. Defer to v0.27**

**Locked: A.** Support motion LoRAs. Reuses the v0.17 `LoraSpec`
grammar (`hf:org/repo:0.7`, `civitai:NNNNNN:0.5`, local paths).
Loaded onto the motion-adapter weights at load time. ~1 phase of
work; closes the "panic zoom" / "pan left" community LoRA gap.

### Q6: SD3 / SD3.5 animate scheduler

Options:
- **A. Reuse v0.20 Flux animate pattern** — T5 + CLIP-L + CLIP-G
  lerp; flow-match per frame
- **B. Design fresh MMDiT-specific scheduler** — exploit dual-
  stream attention structure
- **C. Stub for v0.27**

**Locked: A.** Reuse the proven pattern. Flux animate (v0.20)
already lerps T5 + CLIP-L + flow-matches per frame; SD3 adds
CLIP-G to the lerp surface. Lower risk than designing fresh.

### Q7: Scenario auto-LoRA discovery cadence

Options:
- **A. Once per (look/genre, base_model)** — smart-cached across
  the scenario
- **B. Once per scenario** — use scenario-level look/genre/model
- **C. Once per task** — full per-task discovery

**Locked: A.** Once per (look/genre, base_model). The discovery
cache key already includes the base model (v0.25 phase 4 design);
firing it at scenario load reuses the cache across all 100+
tasks. Per-task `look:` overrides re-trigger only when the
(look, base) tuple is new. Worst case: a scenario with 100 tasks
each with a unique `look:` + unique `model:` → 100 calls. Common
case: 1 call.

### Q8: `plakat.metadata.write` scope

Options:
- **A. JSON sidecar + PNG tEXt** — full A1111-compat
- **B. JSON sidecar only** — smaller scope
- **C. PNG tEXt only** — loses structured data

**Locked: A.** Full A1111-compat. `plakat.save` extended to write
both formats (lift from `cli::generate`'s existing helper).
`plakat.metadata.write` is then a thin wrapper that re-extracts
+ re-attaches metadata to existing files — useful for re-saving
after Bund-script edits.

## 7. Phase plan (locked 2026-05-27)

See §5.

## 8. What's NOT in v0.26 (explicitly deferred to v0.27+)

- **SDXL AnimateDiff** (Q2 cap) — beta motion adapter exists but
  unmature; deferred. v0.27 candidate.
- **AnimateDiff long-form** (>16 frames) — quality degrades past
  the motion adapter's training window. Free-form length needs
  HotShot-XL or similar; not in v0.26 scope.
- **SVD (Stable Video Diffusion)** — different architecture from
  AnimateDiff; separate v0.27+ work if pursued.
- **Multi-GPU split** for AnimateDiff — single-GPU only in v0.26.
  Memory budget on AnimateDiff is roughly SD 1.5 + 1-2 GB motion
  adapter + N-frame latent buffer; fits comfortably on 12 GB.
- **AnimateDiff with ControlNet** — ControlNet conditioning per
  frame would need temporal-coherent control signals; pattern
  isn't proven in candle yet. Phase 13 docs will note this
  limitation.

## 9. Appendix: starting state survey

Source-of-truth from the 2026-05-27 codebase (post-v0.25.0):

- **48 host words** across 13 namespaces (v0.25).
- **902 lib tests + 20 integration tests** green.
- **`src/pipelines/animate.rs`** (v0.20): Flux animate
  implementation. T5 + CLIP-L lerp + flow-match per frame.
  Pattern to reuse for SD3 animate (phase 6).
- **`src/pipelines/unet/`**: SD 1.5 UNet code. Temporal-attention
  splice point (phase 2) lands here. Hot path — careful with the
  forward signature change.
- **`src/imaging/upscale.rs`**: `Method::RealEsrganX2` / `X4` /
  `AnimeX4` already wired; `plakat.hires` consumes them via
  `hires_upscaler` config key. Phase 9 just exposes them on
  `plakat.upscale`.
- **`src/imaging/metadata.rs`**: A1111 `parameters` tEXt + JSON
  sidecar reader (v0.17). The writer side lives in
  `cli::generate` — phase 8 lifts it into `plakat.save`.
- **`src/pipelines/stylize.rs`**: IP-Adapter style-transfer
  pipeline (v0.24 phase 6). One-shot load today; phase 7 adds
  the cache slot.
- **`src/preset/discovery.rs`**: v0.25 3-source discovery client.
  Phase 12 wires it into `cli::scenario.rs`'s task loop at
  scenario load time.
- **`src/scripting/script_entry.rs::generate_one`**: SD-family
  branch wires look/genre apply (v0.25 phase 8). Flux + SD3
  branches in same function: phases 10-11 mirror the SD-family
  block.
- **HF: `guoyww/animatediff-motion-adapter-v1-5-3`**: 1.4 GB
  safetensors, public. Repo includes the `motion_modules`
  config used at load time.

## 10. Backwards-compatibility considerations

v0.26 is **mostly additive**:

- **New CLI flags only** — `plakat animate --animatediff`,
  `--motion-lora`, `--format`, etc. Existing `plakat animate`
  (v0.20 lerp mode) keeps working bit-identically when no
  `--animatediff` flag is passed.
- **`plakat.save` extension** — writes JSON sidecar + PNG tEXt
  in addition to the PNG. Existing scripts that read the saved
  PNG don't see new behavior; new files are bigger by the
  sidecar.
- **`plakat.upscale` extension** — accepts new method aliases
  (`real-esrgan-x2` etc). Existing numeric `2.0` / `4.0` scale
  values still mean Lanczos.

**One soft concern**: phase 5's video-encoder dependency. If we
pick ffmpeg via `Command`, plakat requires `ffmpeg` on `$PATH`
when `--format mp4` / `webm` is used. If we pick a Rust crate
(`mp4` + `webm-iv`), the binary grows. Decision in phase 5
review.

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Temporal-attention integration into UNet doesn't converge | Medium | Phase 2 is the highest-risk single phase. Mitigate: lift the implementation pattern from diffusers-rs reference impl; ship phase 2 behind a feature flag so partial work doesn't block downstream phases. |
| Motion-adapter weights have unexpected layout vs the SD 1.5 UNet that candle uses | Medium | Phase 1 deliverable is loading the weights + reading the config — surfaces this before phase 2's integration work. If layouts diverge, phase 2 may need additional adaptation. |
| Video encoder dep adds friction (ffmpeg requirement) | Low | Pick a Rust crate over `Command` ffmpeg if friction is a concern. Decision in phase 5. |
| AnimateDiff phases 1-5 stretch past 7 sessions | Medium | Phases 6-13 are decoupled from AnimateDiff's fate. Scope-cut option: ship phase 6 (SD3 animate) + 7-12 (carries) + defer AnimateDiff to v0.27. |
| Scenario discovery integration breaks the existing two-stage LoRA pipeline | Low | Phase 12 adds a cached discovery call BEFORE the task loop; doesn't modify the existing scenario.loras / task.loras flow. New code path adds to `task.loras` via the existing parse_resolved_loras helper. |
| AnimateDiff memory budget surprises | Low | SD 1.5 + 1-2 GB motion adapter + N-frame latent buffer fits in 12 GB easily. Document the exact requirement after phase 2 measures it. |

## 12. Cycle-cut decision tree

If at the end of phase 5 the AnimateDiff stack isn't producing
recognizable motion:

- **Cut option 1 (recommended)**: ship phase 6 (SD3 animate) +
  7-12 (carries) + 13 (docs noting AnimateDiff deferred).
  v0.26 still closes 6 of 7 carries; AnimateDiff slides to
  v0.27. Net good cycle.
- **Cut option 2**: ship phase 1 (loader) as `feature =
  "animatediff-preview"` behind a Cargo feature flag, then
  continue with phases 6-13. Lets the loader work land while
  the integration matures.
- **Hold option**: extend AnimateDiff to ~8 sessions before
  cutting. Trade off ship velocity for a clean AnimateDiff
  story.

This decision lands at the phase 5 closeout commit, not
earlier.
