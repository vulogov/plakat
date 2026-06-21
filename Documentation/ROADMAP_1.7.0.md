# plakat 1.7.0 — roadmap

Track M built the map as a standalone command across 1.4–1.6: spec + geometry
engine (1.4), the linework render + vector export (1.5), and the painted SD render
+ tiled multi-tile (1.6). 1.7.0 **integrates `plakat map` into the rest of plakat** —
the scripting (Bund), scenario, and compile systems — so a map is a first-class step
in a batch, not just a one-off CLI invocation. This is **MAP-4** in the RFC.

The discipline carries over: the geometry/conditioning stays a pure function of
(spec, seed) — corpus-proven, no GPU — and only the optional SD paint is the GPU
step, decoupled exactly as in 1.6. The heavy **urban fabric** (MAP-5: street graph
via `petgraph`/DCEL, the urban `Anchor` variants) stays the 1.8 capstone.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## MAP-4 — scripting + scenario + compile integration

- [x] **`plakat.map.*` Bund words — DONE.** `plakat.map.render ( spec-path style --
      handle )` (`src/scripting/words/map.rs`, registered in `words/mod.rs`) renders a
      committed spec to an in-memory image handle (the deterministic linework path,
      seed from `ctx.config.seed`), so a Bund script produces maps alongside the
      generate→upscale→save handle chain — then `plakat.save` writes it. **Gate MET:**
      `corpus/map_script.bund` + `corpus/map_script.sh` — the saved map is
      byte-identical to the direct `--map-render`. (The GPU paint stays behind the
      CLI/scenario; scripting keeps the GPU-free render to match the handle model.)
- [x] **Scenario `map` task — DONE.** `type: map` task kind in `cli/scenario.rs`
      (schema fields `map-spec`/`map-style`/`map-paint`/`map-scale`/`map-tiles`/
      `map-sd-model`/`map-sd-lora`/`map-provider` at scenario + task level, merged by
      `effective_map_config`), dispatched to the focused delegate
      `map::scenario_task::run_map_task` (source spec → linework or SD paint →
      `<out>/<name>/map.png`). `DropAll` cache-evictor frees any t2i/animate pipeline
      before a map task's own SD load. Scene/weather are now optional so a map task
      needn't carry them (and the scene-ref + enhance pre-pass skip map tasks).
      **Gate MET:** `corpus/map_scenario.hjson` (parchment + blueprint) — the
      parchment task is **byte-identical to the direct `--map-render`** (proves the
      integration shares the deterministic path), byte-checked in `corpus/map.sh`. 4
      delegate unit tests (spec source, dry-run, linework write, LoRA resolution).
- [x] **`map:` compile block (COMPILE-1 E-C4) — DONE.** A `type: map` block (with
      `map-spec`/`map-style`/`map-paint`/`map-scale`/`map-tiles`/`map-sd-model`/
      `map-sd-lora`/`map-provider`) in a `prompts.txt` compiles to a scenario `map`
      task — prose worldbuilding + scene rendering in one document. Deterministic (no
      LLM): 9 new `CommandSpec` keys (`mod.rs`), fields on `ResolvedScene` with
      global→scene inheritance (`resolver.rs`), conditional emit (`emitter.rs`); the
      parser now allows a description-less block when it (or the global) declares a map
      task. **Gate MET:** `corpus/compile/maps.txt` → byte-stable `maps.hjson` (no-op
      `--diff`), scenario-validated, and the rendered map is byte-identical to the
      direct `--map-render` — checked in `corpus/compile.sh`.
- [ ] **Gate:** a committed scenario (and a compile `prompts.txt`) that emits the
      Isle of Vethûn map; the deterministic geometry/linework artifact is byte-checked
      in the corpus (extend `corpus/map.sh` or a new `corpus/map_script.sh`), the
      painted path is a `⬜`-until-rendered showcase like `corpus/map_render.sh`.
      Scripting/scenario/compile unit tests.

## Opportunistic polish (small, fits this cycle)

- [ ] **Conditioning-only fast path in scenarios** — let a batch emit the styled
      linework map without ever touching the GPU (the common case), with the paint
      strictly opt-in.
- [ ] **L2 named-river ↔ traced-channel matching** — the render already heuristically
      assigns river names to the longest channels; wire the spec's river ids through
      so labels follow the intended watercourse (also improves GeoJSON `id`s).

## Later in Track M

1.8 **urban fabric (MAP-5)** — the urban-scale `Anchor` variants the spec already
carries (`along_street`, `block_face`, `at_gate`, `pier_tip`, …): a street graph via
`petgraph`, block/lot subdivision (DCEL), walls + gates, resolved against the
existing geometry. The first new heavy dep (`petgraph`) lands here.

## Opportunistic / debt (off the critical path)

- L2 refinements: flat-resolution (parallel-thread artifact in flats), breach-vs-fill,
  delta detection.
- L5: `rstar` nearest-coast index + concavity for natural-harbor placement.
- Labels: non-Latin shaping (`language` = `ar`/`ru`/`zh`) wants a real shaped font +
  `ab_glyph` (the 1.5 bitmap font is Latin-only by design).
- 1.1.0 carryovers: Flux regional (Flux broken on Metal → code-only), IC-Light.
- Memory-bound render debt: SD3.5 DreamBooth render, `regional.sh sdxl/sd35`.
