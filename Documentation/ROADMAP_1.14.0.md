# plakat 1.14.0 — roadmap: every feature, every surface

1.12.0 shipped **multiperson + face-swap** and 1.13.0 shipped the **map coastlines/worlds**,
but both landed as **CLI-only**. 1.14.0 productizes them: multiperson + the new map features
become first-class in the automation surfaces (**scenario / compile / scripting**), and the
proof corpus grows to cover them. This is the same "every feature in all surfaces" follow-
through the map track did at MAP-4 (`map/scenario_task.rs`, `scripting/words/map.rs`).

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — multiperson everywhere (headline)

- [ ] **`multiperson` as a scenario task** — a `type: multiperson` task in the scenario
      HJSON: scene prompt + `people` (label → photo + `at` placement) + identity mode
      (`swap` / `composite`, `pose`, `harmonize`, `scale`). The runner dispatches to
      `pipelines::multiperson::run`, so a single in-process run can batch several
      people-in-scene compositions (and reuse loaded weights). Model the schema on the CLI
      flags; serde-default everything so existing scenarios are untouched.
- [ ] **`plakat.multiperson` scripting word** — a Bund host word mirroring the scenario
      task, so scripts can place specific people into generated scenes programmatically
      (same dispatch path → byte-for-byte parity with the CLI, like the map words).
- [ ] **Tutorial + parity check** — extend `MULTIPERSON_TUTORIAL.md` with the scenario +
      scripting forms; a test asserts the three surfaces produce the same request.

## B — new map features in the automation surfaces

The coastal terrain (`terrain.{peninsulas,inlets,fjords}`) and the render features
(marsh/deltas/seasonal) live in the **spec / render**, so any surface that loads a spec
already gets them — but two gaps remain:

- [ ] **Multi-tile render in scenario + scripting** — `--map-render-tiles` is CLI-only. Add
      a tiled-output option to the `map` scenario task and the `plakat.map` words (a tile
      grid + output dir), so worlds can be tiled from automation.
- [ ] **Verify the spec-level features flow through** scenario / compile / scripting
      (coastal terrain, marsh, deltas, seasonal, political export) and add a parity test +
      a one-line note where each surface exposes them.
- [ ] *(if compile reaches it)* teach the `compile` LLM prompt about the new terrain words
      so prose like "a fjord-cut northern coast" can map to `terrain.fjords`.

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
