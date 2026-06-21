# plakat 1.8.0 — roadmap

The map track is geographically complete: spec + geometry engine (1.4), linework
render + vector export (1.5), painted SD render + tiled (1.6), and scenario/compile/
scripting integration (1.7). 1.8.0 opens the last planned phase — **MAP-5, the urban
fabric**: a city-scale map built from a street graph, blocks, and walls/gates,
resolving the **urban `Anchor` variants the spec has carried since MAP-1**
(`along_street`, `block_face`, `at_gate`, `pier_tip`, `near_intersection`,
`on_wall`, `in_district`, `city_center`, `at_station`, `along_waterfront`).

This is the track's first **new heavy dependency** (`petgraph` for the street graph)
and a genuinely larger phase than the geographic layers. The through-line still
holds: the urban geometry is a **pure function of (spec, seed)** → byte-stable on-box
corpus proofs, no GPU; the optional SD paint reuses the 1.6 path.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## MAP-5 — urban fabric (the city-scale map)

Layered like the geographic engine (L0–L7), but at the street scale (scale tiers
10–12, the urban U0–U2 the spec already defines):

- [x] **U0 street graph — DONE.** `src/map/urban.rs` `StreetGraph::generate` — a
      `petgraph` `UnGraph` of junctions + street segments: a centre, a wall ring with
      **gates** (spec bearings, else four cardinals), **arterials** radiating
      centre→gate, a **ring road** just inside the wall, and a **minor-street grid**
      clipped to the wall. Structured `UrbanSpec` schema (wall/gates/streets/districts/
      waterfront/piers/station) added to `MapSpec`. Pure fn of (spec, seed) — fixed
      insert order → byte-stable. `--map-dump-streets`. Only new dep: `petgraph`.
- [x] **U1 blocks — DONE.** `StreetGraph::blocks` — each interior grid cell (4 corners
      inside the wall) becomes a **block** parcel, inset from the street centrelines,
      rendered as built-up infill. Deterministic raster order. (Lot subdivision within a
      block is a later refinement; `block_face`/`in_district` resolve against these.)
- [x] **U2 walls + gates + waterfront — DONE.** The wall ring + gates ship in U0; U2
      adds the **waterfront** (a half-plane of open water on the named edge) and
      **piers** running out into it at their positions. `on_wall` / `pier_tip` /
      `along_waterfront` resolve against these. (A dedicated station node is folded
      into the resolver's fallback for now.)
- [x] **Urban anchor resolver — DONE.** `StreetGraph::resolve_landmarks` /
      `resolve_anchor` place the spec's landmarks against U0–U2: `city_center`,
      `at_gate`, `in_district`, `pier_tip`, `along_street` (bearing-matched arterial,
      position interpolation), `on_wall`, plus the cardinal/canvas fallbacks.
      Unresolvable anchors drop (no abort).
- [x] **Urban render + integration — DONE.** `StreetGraph::render_town` draws water +
      block parcels + streets + wall + gates + piers + landmark markers + labels
      (gates / districts / piers / landmark names) + a title cartouche + frame.
      `--map-render` **routes urban specs (a `urban` block) to the town renderer**,
      geographic specs to the linework map — so the 1.7 scenario/compile/scripting
      surfaces render towns unchanged.
- [x] **Gate MET:** committed `corpus/map/town.spec.json` (Saltmere Town) →
      byte-stable `--map-dump-streets` `town-streets.png` + `--map-render`
      `town-map.png`, checked in `corpus/map_urban.sh`. 6 urban unit tests
      (connected graph, gates-on-wall, blocks-inside, anchor resolution, both renders
      deterministic). **Determinism invariant held.**

## New deps

`petgraph` (street graph; the one heavy add). A DCEL/half-edge crate **only if** the
block-face walk needs it — try a hand-rolled face walk first to avoid the dep.

## Realism + configurability (done)

- [x] **Eroded natural features — DONE.** The island coastline is multi-scale
      noise-warped (bays, peninsulas, headlands) and mountain ridgelines wander with
      varying crest height — no more smooth-potato islands or oval ranges.
- [x] **Configurable town layouts — DONE.** Rebuilt the urban plan as medieval
      radio-concentric (curved rings + radials), with `LayoutStyle { Radial | Grid |
      Organic }` picked via `urban.layout` / `--map-urban-layout` or inferred
      (mountain→organic, walled→radial, plains→grid). A straight grid only when chosen.
- [x] **Controllable erosion — DONE.** `terrain.erosion` (0 smooth … 1 natural …
      >1 rugged) scales the coast + ridge irregularity, exposed via `--map-erosion`
      and the scenario `map-erosion` field (+ `map-layout`); the LLM schema documents
      it. `erosion=1.0` is the natural default; the value flows through every render
      surface (CLI / scenario / scripting).
- [x] **Scripting setters — DONE.** `plakat.map.layout ( style -- )` and
      `plakat.map.erosion ( amount -- )` Bund words stash overrides on `ScriptCtx`,
      applied by `plakat.map.render` — so a script tunes the plan + erosion. Verified
      byte-identical to `--map-urban-layout` / `--map-erosion`.

## Opportunistic polish (small, fits this cycle)

- [ ] **L2 named-river ↔ traced-channel matching** — wire the spec's river ids
      through to the longest traced channels so labels + GeoJSON `id`s follow the
      intended watercourse (carried from 1.7).
- [ ] **`plakat.map.paint` scripting word** — the SD-painted counterpart to
      `plakat.map.render` (1.7 left scripting on the GPU-free path).

## Track M after 1.8

With MAP-5 done the planned map track is **complete**. Remaining items are debt /
refinements, folded in as wanted:

- L2: flat-resolution (parallel-thread artifact in flats), breach-vs-fill, delta detection.
- L5: `rstar` nearest-coast index + concavity for natural-harbor placement.
- Labels: non-Latin shaping (`language` = `ar`/`ru`/`zh`) — a real shaped font + `ab_glyph`.
- 1.1.0 carryovers: Flux regional (Flux broken on Metal → code-only), IC-Light.
- Memory-bound render debt: SD3.5 DreamBooth render, `regional.sh sdxl/sd35`.
