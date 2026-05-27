# RFC v0.24 — Script surface completion

**Status:** decisions locked 2026-05-27 — ready for phase 0.

**Predecessors:**
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md) — the 7-word MVP.
- [`RFC_v0.22_BUND_WORDS_EXPANSION.md`](RFC_v0.22_BUND_WORDS_EXPANSION.md) — the 28-word expansion.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) — six v0.22 deferrals closed.

## 1. TL;DR

User picked **Persona depth + Scripting completion** as the
v0.24 theme. Initial scoping suggested these were two separate
~5-session efforts (~12 total). The CLI audit revealed almost
everything the user could ask for in persona depth — multi-photo
`--photo`, `--face-bbox`, `--face-landmarks`, all 4 FaceID
variants — already exists in the CLI. The v0.24 work is the
*scripting wiring* of that surface, not new CLI development.
That shrinks persona to ~2–3 phases.

By the end of the cycle:

- `plakat.portrait` accepts multiple photos with weights;
  `face_bbox` / `face_landmarks` knobs exposed; identity
  kind selectable.
- `plakat.outpaint`, `plakat.embedding.*`, `plakat.stylize`,
  `plakat.metadata.*` all ship — the v0.20+ carries close.
- Flux + SD3 ControlNet `from=` (auto-annotate) works in
  scripts.
- Flux inpaint via `flux-fill-dev` works in `plakat.inpaint`.

Word count target: 33 → ~42 (depending on persona-depth shape
in Q1 below).

## 2. Why this is the v0.24 cycle

1. **Closes the scripting arc.** v0.21 / v0.22 / v0.23 built
   the scripting layer; v0.24 finishes it. After this cycle
   there's no "use the CLI for X" gap for users who want to
   stay in scripts.

2. **Persona depth without new model work.** FaceID variants,
   multi-photo blending, manual landmarks/bbox — all the model-
   side work is done in v0.21–v0.22. Scripting wires it.

3. **Sets up the model-architecture push for v0.25+.** With
   scripting fully done, the next big swing can be AnimateDiff
   (~8–12 sessions of architecture work) or SD3 animate
   (~4–6 sessions) without a competing scripting backlog.

## 3. Architectural constraints we keep

Same five from prior RFCs:

1. Built-our-own VM; restricted stdlib.
2. Singleton context; one script per process.
3. Async bridge via `block_in_place`.
4. v0.22 relaxed-compat carries forward (no backwards-compat
   hacks).
5. SD-family two-slot cache (SdT2i + SdPortrait, sharing
   `Arc<SdCore>`) — extend only when load-time concerns require
   it (Flux/SD3 CN auto-annotate, this cycle's main cache-touching
   work).

## 4. The deliverables

### 4.1 Persona depth (2–3 phases)

| # | Item | Today's state | What ships |
|---|---|---|---|
| P1 | Multi-photo portrait | `plakat.portrait ( prompt photo -- handle )` accepts one photo path/handle | New `plakat.portrait.photo.{add, clear, list}` words OR extend `plakat.portrait` to accept a list literal |
| P2 | `face_bbox` + `face_landmarks` config keys | CLI flags exist; no script surface | Two new config keys (string CSV grammar) that thread through `portrait::GenRequest` |
| P3 | Identity-variant override | Auto-picked from alias (sd15 → PlusFace, sdxl → PlusFaceSdxl) | New config key `identity_kind`: `plus-face` / `plus-face-sdxl` / `face-id` / `face-id-sdxl` |

### 4.2 Scripting completion (6 phases)

| # | Item | Today's state | What ships |
|---|---|---|---|
| S1 | `plakat.outpaint` | `cli::outpaint` exists (416 LOC) | New host word: `( prompt input expand-spec -- handle )` |
| S2 | `plakat.embedding.*` | `embedding` module exists; TI specs flow into `t2i::LoadRequest.embeddings` | New collection-style namespace: `add` / `clear` / `list` |
| S3 | `plakat.stylize` | `stylize::Pipeline` exists for IP-Adapter style transfer | New host word: `( prompt subject-photo style-photo -- handle )` |
| S4 | `plakat.metadata.*` | `imaging::metadata` module exists; reads PNG tEXt + JSON sidecar | New namespace: `read` (push fields), maybe `write` (defer to v0.25?) |
| S5 | Flux + SD3 CN `from=` | v0.23 phase 6+7 bails on `from=` specs (auto-annotate needs per-generate dims unknown at load) | Defer annotation to first generate; cache annotated PNG alongside the pipeline; reuse for repeat calls |
| S6 | Flux inpaint via `plakat.inpaint` | v0.23 phase 5 bails on Flux | Wire `flux-fill-dev` variant + the channel-concat `img_in` path; thread mask through `flux::GenRequest.mask` |

### 4.3 Docs + tests (1 phase)

Mirror of v0.22 phase 12 / v0.23 phase 8. SCRIPTING.md update,
SCRIPTING_TUTORIAL.md §11 "What's new in v0.24", composition
tests, release notes.

## 5. Cycle scope

~10 phases including hygiene + docs:

| Phase | Deliverable | Est. |
|---|---|---|
| 0 | Sweep stale v0.24 markers + RFC commit | ~0.25 session |
| 1 | `plakat.portrait` multi-photo (P1) | ~1 session |
| 2 | `face_bbox` + `face_landmarks` config keys (P2) | ~0.5 session |
| 3 | `identity_kind` config key (P3) | ~0.5 session |
| 4 | `plakat.outpaint` (S1) | ~1 session |
| 5 | `plakat.embedding.*` (S2) | ~1 session |
| 6 | `plakat.stylize` (S3) | ~1 session |
| 7 | `plakat.metadata.*` (S4) | ~0.5 session |
| 8 | Flux + SD3 CN `from=` (S5) | ~1 session |
| 9 | Flux inpaint via `flux-fill-dev` (S6) | ~1 session |
| 10 | Docs + composition tests | ~0.5 session |

**Total estimate:** 7.25–8.25 sessions. About the same size as
v0.23.

## 6. Decisions (locked 2026-05-27)

### Q1: Multi-photo portrait shape

`plakat.portrait` is `( prompt photo -- handle )`. Multi-photo
options:

- **A. Collection namespace.** `plakat.portrait.photo.add (
  path-or-handle weight -- )`, `.clear`, `.list`. Then
  `plakat.portrait ( prompt -- handle )` reads `ctx.portrait_photos`.
  Stateful; matches the LoRA/ControlNet pattern.
- **B. Extend stack effect.** `plakat.portrait ( prompt photo1
  ... photoN n -- handle )`. Push n photos then the count; pop
  count, pop n. Forth-flavoured but unusual.
- **C. List literal.** `plakat.portrait ( prompt [photos] --
  handle )`. Bund lists exist but rust_dynamic LIST type is
  awkward in plakat scripts.

**Locked: A.** New collection namespace
`plakat.portrait.photo.{add, clear, list}` carrying the
multi-photo state on `ScriptCtx.portrait_photos`.
`plakat.portrait ( prompt -- handle )` reads it. Matches the
LoRA/ControlNet pattern.

### Q2: Flux/SD3 CN `from=` annotation strategy

v0.23 bails on `from=` specs because auto-annotation needs the
per-generate width/height the loader doesn't know. Options:

- **A. Lazy first-generate annotation.** First `plakat.generate`
  with a `from=` spec triggers the annotation using that
  generate's dims, then caches the annotated PNG to a tempfile
  bound to the pipeline. Subsequent generates with the same
  pipeline reuse. Dim changes invalidate (mark_loras_changed
  pattern).
- **B. Eager annotation at load with a default dim.** Pick
  1024×1024 (Flux/SD3 native); cache. Subsequent generates use
  the cached annotation regardless of their actual dim. Faster
  but might produce wrong-resolution conditioning.
- **C. Stay with the v0.23 bail.** Document the limitation;
  users pre-render maps.

**Locked: A.** Lazy first-generate annotation. The cached
annotation lives on the loaded Flux/SD3 pipeline; pipeline
invalidation drops it. Dim mismatch on a subsequent call
forces a re-annotation.

### Q3: `plakat.metadata.*` subscope

`imaging::metadata` reads the A1111 `parameters` tEXt chunk +
JSON sidecar plakat writes. Options:

- **A. Read-only.** `plakat.metadata.read ( path -- json )` plus
  field accessors (`.prompt` / `.seed` / `.model` / `.loras`).
- **B. Read + write.** Add `plakat.metadata.write ( handle path
  -- )` that bundles a JSON sidecar with the saved image. Users
  can replay scripts via the sidecar.
- **C. Defer entirely.** Scripts can shell out to `plakat
  metadata FILE.png` via the host system; not load-bearing for
  scripts.

**Locked: A.** Read-only — `plakat.metadata.read ( path -- json )`
plus field accessors. Write deferred to v0.25, gated on
`plakat.save` attaching JSON sidecars.

### Q4: `identity_kind` override scope

CLI auto-picks identity by model variant (sd15 → PlusFace,
sdxl → PlusFaceSdxl, sd21 → None). User override options:

- **A. Free-form string config key.** `plakat.config.set
  "identity_kind" "face-id"`. Validated against the 4 variants
  at set-time.
- **B. Discrete host words.** `plakat.identity.plus-face`,
  `plakat.identity.face-id`, etc. — 4 words.
- **C. Single host word.** `plakat.identity.set ( kind -- )`.

**Locked: A.** Config key. `plakat.config.set "identity_kind"
"face-id"`. Validated at set-time against the 4 variants
(`plus-face`, `plus-face-sdxl`, `face-id`, `face-id-sdxl`).
Empty string → auto-pick by alias (today's behaviour).

### Q5: Phase ordering

Two viable orderings:

- **A. Persona first.** Phases 1–3 ship persona depth, then 4–9
  ship scripting completion. Frontloads the smaller half.
- **B. Scripting first.** Phases 1–6 ship scripting completion,
  then 7–9 ship persona depth.
- **C. Interleave by risk.** Easy phases first (config keys,
  outpaint wrapper), harder phases later (Flux inpaint,
  CN auto-annotate).

**Locked: A.** Persona first. Phases 1–3 ship persona depth,
then 4–9 ship scripting completion. Matches §5 table order.

## 7. Phase plan (locked 2026-05-27)

See §5.

## 8. What's NOT in v0.24 (explicitly deferred to v0.25+)

- **AnimateDiff** — still the v0.20+ multi-cycle carry. The
  scripting completion of v0.24 + the persona depth wiring leave
  AnimateDiff as the natural v0.25 big swing.
- **SD3 / SD3.5 animate** — 3-encoder lerp + MMDiT integrator.
- **Real-ESRGAN ML upscaling** in `plakat.upscale` (the
  standalone word; `plakat.hires` already exposes it).
- **Metadata write** (`plakat.metadata.write`) — gated on
  `plakat.save` writing JSON sidecars.

## 9. Appendix: starting state survey

Source-of-truth from the 2026-05-27 codebase:

- 33 host words across 9 namespaces (v0.23).
- 772 lib tests green.
- `cli::outpaint`: 416 LOC; uses img2img + mask.
- `pipelines::face_models`: 1033 LOC; FaceID encoders ship.
- `pipelines::stylize`: present; CLI uses it.
- `pipelines::embedding`: present; TI specs flow into
  `t2i::LoadRequest.embeddings`.
- `imaging::metadata`: present; reads A1111 tEXt + JSON.
- CLI portrait: `--photo` repeatable with weights, `--face-bbox`,
  `--face-landmarks` all wired. Persona depth in v0.24 is
  exposing this to scripts, NOT building it.
