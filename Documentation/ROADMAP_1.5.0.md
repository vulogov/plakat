# plakat 1.5.0 — roadmap

1.4.0 shipped **MAP-1 + MAP-2**: the `MapSpec v2` schema, the LLM parser, and the
eight-layer geometry engine (L0–L7) — every layer a byte-stable corpus proof, but
no labels and no styling. 1.5.0 is the **linework render (MAP-3)**: turn the L7
feature overlay into the **first complete, user-facing map** — labelled, styled,
and exportable — still with **no SD** (memory-wall-free, fully deterministic).

The through-line holds: the render is a **pure function of (spec, seed)** → on-box
corpus proofs, no GPU, no network. The non-deterministic LLM parse stays decoupled
behind a committed `--map-spec`; the memory-bound tiled-SD render is the 1.6.0
capstone.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## MAP-3 — linework render (labels + cartographic furniture)

- [ ] **Label compositor** (`src/map/labels.rs`) — place a name at each resolved
      landmark (L5) + each named feature (ranges, rivers, regions) without overlap.
      `ab_glyph` for glyph rasterization; `unicode-bidi` + `-normalization` for
      correct shaping of non-ASCII names. Greedy candidate-position placement
      (offset ring around the anchor) with a collision grid; rivers/ranges get
      curved or along-axis labels. Deterministic order (spec order, then id).
- [ ] **Cartographic furniture** — a compass rose, a scale bar (from the spec's
      `scale_tier` / tile grid), and a legend (the landmark-kind markers + biome
      swatches actually present). Composited into a margin frame around the map.
- [ ] **Parchment / ink styling** — a styling pass over the L7 overlay: parchment
      paper base, ink coastline + linework, hill-shading from the L1 heightfield,
      muted biome fills, a border. A small set of named styles (`parchment`,
      `blueprint`, `inked`) selectable via `--map-style`.
- [ ] **`--map-render PATH`** — the single user-facing output: the fully composited,
      labelled, styled map PNG (the default `plakat map "<prose>"` artifact once the
      spec is parsed). Replaces "dump a layer" as the headline command.
- [ ] **Gate:** committed `island.spec.json` → byte-stable `--map-render` PNG
      (the complete Isle of Vethûn map: coastline, biomes, rivers, roads, every
      landmark labelled, compass + scale + legend), checked in `corpus/map.sh`.
      Label-placement + furniture unit tests. **Determinism invariant**: render
      twice, identical bytes.

## MAP-3b — vector export (optional, same data)

- [ ] **GeoJSON export** (`--map-export-geojson`) — coastline polygons, river
      polylines, road polylines, landmark points, region polygons. The geometry
      engine already holds all of it; this is a serialization pass.
- [ ] **SVG export** (`--map-export-svg`) — the linework as scalable vectors
      (labels as `<text>`), for print / further editing.

## New deps (pure Rust)

`ab_glyph` (glyph raster), `unicode-bidi` + `unicode-normalization` (text shaping),
`imageproc` (line/curve drawing, already in tree). Vector export needs no new dep
(`serde_json` for GeoJSON; SVG is string assembly).

## Later in Track M (unchanged)

1.6.0 **tiled SD render** — feed the L7 overlay (+ per-tile Canny edges) to a tiled
SDXL ControlNet pass for a painted map; 1×1 on-box, multi-tile memory-bound (the
1.1.0 memory-wall lesson). 1.7–1.8 **Bund hooks** (`plakat.map.*` scripting) +
**urban fabric** (MAP-5: street graph via `petgraph`/DCEL).

## Opportunistic / debt (off the critical path)

- L2 refinements: flat-resolution (parallel-thread artifact in flats), breach-vs-fill,
  delta detection, named-river ↔ traced-channel matching.
- L5: `rstar` nearest-coast index + concavity for natural-harbor placement.
- 1.1.0 carryovers: Flux regional (Flux broken on Metal → code-only), IC-Light.
- COMPILE-1 `map:` block (E-C4) — now unblocked (`plakat map` exists).
- Memory-bound render debt: SD3.5 DreamBooth render, `regional.sh sdxl/sd35`.
