# plakat 1.14.0 — roadmap: every feature, every surface

1.12.0 shipped **multiperson + face-swap** and 1.13.0 shipped the **map coastlines/worlds**,
but both landed as **CLI-only**. 1.14.0 productizes them: multiperson + the new map features
become first-class in the automation surfaces (**scenario / compile / scripting**), and the
proof corpus grows to cover them. This is the same "every feature in all surfaces" follow-
through the map track did at MAP-4 (`map/scenario_task.rs`, `scripting/words/map.rs`).

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — multiperson everywhere (headline) ✅

- [x] **`multiperson` as a scenario task** — `type: multiperson` task carries a
      `multiperson:` block (scene prompt + `people` (persona name → `at`/`prompt`/`scale`)
      + identity mode `swap`/`composite`/`pose`/`harmonize`/`restore-faces`). `people[]`
      reference the top-level `personas:` list by name (resolved to photos, validated
      before model load). Dispatches `pipelines::multiperson::run` via
      `multiperson::scenario_task::build_request`; CacheEviction::DropAll on switch;
      skips scene/weather + enhancement cross-refs (like map tasks). Serde-defaults
      throughout. (`src/pipelines/multiperson/scenario_task.rs`, `cli/scenario.rs`)
- [x] **`plakat.multiperson` scripting word** — `( spec-path -- handle )`: a self-
      contained JSON spec (task fields + inline `personas` table) → the SAME builder →
      `multiperson::run` → image handle. (`src/scripting/words/multiperson.rs`)
- [x] **Tutorial + parity check** — `MULTIPERSON_TUTORIAL.md` covers all three surfaces;
      `scenario_and_script_forms_build_identical_requests` + `defaults_mirror_the_cli_flags`
      tests assert the surfaces produce the same request with CLI-matching defaults.

## B — new map features in the automation surfaces ✅

The coastal terrain (`terrain.{peninsulas,inlets,fjords}`) and the render features
(marsh/deltas/seasonal) live in the **spec / render**, so any surface that loads a spec
already gets them — but two gaps remained:

- [x] **Multi-tile render in scenario + scripting** — extracted the CLI slicing to a
      shared `map::save_world_tiles` (CLI now calls it too). Scenario: `map-render-tiles:
      true` task/scenario field → `MapTaskCfg.render_tiles` → emits `world.png` +
      `tile_r{R}_c{C}.png`. Scripting: `plakat.map.tiles ( spec-path style out-dir -- count )`.
      A byte-for-byte parity test asserts the task path == the shared helper.
- [x] **Verify the spec-level features flow through** — committed `corpus/map/coastal.spec.json`
      (peninsulas + inlets + fjords); `coastal_terrain_features_flow_through_source_spec`
      asserts they survive the load path all surfaces share; tutorial §4 now documents
      coastal shaping + §8 notes that every spec-level feature flows through automation.
- [x] **Taught the prose parser the new terrain words** — `peninsulas`/`inlets`/`fjords`
      (+ the previously-undocumented `plateaus`/`rift_valleys`) are now in the MapSpec
      schema preamble with a coastal-language mapping rule, so "a fjord-cut northern coast"
      → `terrain.fjords` (benefits CLI prose, scenario prose, and the compile `map:` block).

## C — proof corpus expansion (confidence)

The 1.12/1.13 features shipped with thin corpus coverage. Add committed drivers + showcases
(and regenerate `GALLERY.md`):

- [ ] **multiperson** — a `--swap --pose` showcase (photos) and a `--composite` showcase.
- [ ] **coastal maps** — a spec exercising peninsulas + inlets + fjords (byte-stable).
- [ ] **multi-tile world** — a tiled world + the stitched `world.png`.
- [ ] **political export** — a polity GeoJSON/SVG proof.
- [ ] run them through the **scenario** form too, proving the automation path end-to-end.

## D — carry / polish (opt-in)

- [ ] Multi-tile **per-tile furniture** — optional frame/scale/coordinate label per tile so a
      single tile is a usable standalone map (not just a slice).
- [ ] Political **polygons** (territory rings) in GeoJSON/SVG export, not just points.
- ~~M2 regional eps-blend / FaceID-in-inpaint~~ — **dropped**: both improve the inpaint
      route that `--swap` superseded; negative ROI (see the 1.13 close-out).

## Notes

- The win is **parity**: each productized feature dispatches the same pipeline the CLI does,
  so scenario / scripting / CLI stay byte-identical (the map track's discipline).
- `--features metal` / `--features cuda` for GPU; default build is CPU-only.
- Already in 1.14.0: `multiperson --restore-faces` (1.13 B carry).
