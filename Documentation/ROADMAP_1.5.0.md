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

- [x] **Label compositor — DONE.** `src/map/labels.rs`: a hand-authored **5×7
      all-caps bitmap font** (the classic small-caps cartographic look) rasterized
      at integer scale with a paper halo — **no font asset, no `ab_glyph`** → the
      render is byte-stable across machines + toolchains (the corpus invariant).
      Upper-case + Latin-1 accent folding (`Vethûn` → `VETHUN`). Placement in
      `render.rs::place_label`: greedy 4-candidate ring (right/left/below/above)
      against a reserved-`Rect` collision list, in deterministic spec order.
      (Non-Latin shaping — the `language` field's `ar`/`ru`/`zh` — still wants a
      real shaped font + `ab_glyph`; noted under debt.)
- [x] **Cartographic furniture — DONE.** A four-point **compass rose** (N marked),
      a **scale bar** (1/2/5-rounded `nice_round` over `km_across(spec)` from
      `world_extent_km` or the per-tier nominal), a **legend** (the landmark kinds
      actually present, with their symbols), and a **title cartouche** (double rule
      + drop shadow). Each reserves its footprint first so labels route around it;
      a double **frame** borders the map.
- [x] **Styling — DONE.** `Style` over the geometry: paper-tinted biome land,
      ink coastline, NW **hill-shading** from the L1 gradient, bathymetric sea
      shading, distinct **per-kind landmark symbols** (city block / fortress tower /
      lighthouse beacon / temple diamond / port / ruin ring). Three named styles —
      **`parchment`** (default), **`inked`**, **`blueprint`** — via `--map-style`.
- [x] **`--map-render PATH` — DONE.** The headline output: the fully composited,
      labelled, styled map PNG. `--map-style` selects the palette. The no-dump
      footer note now points here.
- [x] **Gate MET:** committed `island.spec.json` → byte-stable `--map-render`
      `corpus/images/map/island-render.png` (the complete Isle of Vethûn: coast,
      biomes, hill-shading, rivers, the salt road, four labelled landmarks, The
      Grey Sound in open water, compass + scale + legend + title), byte-checked in
      `corpus/map.sh`. 9 new label/render unit tests (determinism, style divergence,
      glyph coverage, `nice_round`). **Determinism invariant held**: render twice,
      identical bytes.

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
