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

- [ ] **U0 street graph** — a road network as a `petgraph` graph: arterials from the
      district/gate/center anchors, a grid or organic minor-street fill, intersections
      as nodes. Pure fn of (spec, seed). `--map-dump-streets`.
- [ ] **U1 blocks + lots** — the faces of the street graph become **blocks**;
      subdivide each into **lots** (a simple OBB/strip split first; DCEL half-edge
      structure if the face walk needs it). `block_face` / `in_district` resolve here.
- [ ] **U2 walls + gates + waterfront** — city walls as a closed polyline with
      **gates** on the arterials (`at_gate`, `on_wall`), a **waterfront** + **piers**
      where the city meets water (`along_waterfront`, `pier_tip`), a **station**
      (`at_station`). 
- [ ] **Urban anchor resolver** — extend the L5 fixpoint resolver to the urban
      variants against U0–U2 (street position interpolation, block-face offset,
      gate/pier/station lookup, district interior, `city_center`).
- [ ] **Urban render + integration** — the 1.5 linework compositor draws streets /
      blocks / walls / gates / labels; `--map-scale district|settlement|city` selects
      the urban tiers. Reuses `--map-render` / `--map-render-sd` / the scenario+compile+
      scripting surfaces from 1.7 unchanged.
- [ ] **Gate:** a committed urban spec (a walled port town) → byte-stable
      `--map-dump-streets` + `--map-render` PNGs (`corpus/map.sh` or `corpus/map_urban.sh`);
      urban-anchor resolution unit tests (`along_street`, `block_face`, `at_gate`).
      **Determinism invariant**: render twice, identical bytes.

## New deps

`petgraph` (street graph; the one heavy add). A DCEL/half-edge crate **only if** the
block-face walk needs it — try a hand-rolled face walk first to avoid the dep.

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
