# plakat 1.6.0 — roadmap

1.4.0 built the geometry engine; 1.5.0 made the **finished linework map**
(`--map-render` + GeoJSON/SVG) — both pure functions of (spec, seed), byte-stable
on-box. 1.6.0 is the **painted render (MAP-6)**: feed the MAP-2 feature overlay (the
L7 composite, optionally with per-tile Canny edges) through a **tiled SDXL
ControlNet** pass so the map looks hand-painted — the **only GPU step** on the map
track, and the one place the 1.1.0 **memory-wall** lesson bites.

The discipline that protected the rest of the track still holds where it can: the
*conditioning* (the L7 overlay + edges) is a deterministic artifact, already
corpus-proven. Only the SD denoise is non-deterministic / memory-bound, and it's
decoupled behind `--map-render-sd` so the geometry pipeline never depends on a GPU.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## MAP-6 — tiled SDXL render

- [ ] **Conditioning builder** — assemble the per-render ControlNet input from the
      L7 feature overlay: the composited biome/coast/river/road image as the base,
      plus an optional **Canny edge** pass (coast + linework) for a `canny`
      ControlNet. Reuse the existing `imageproc` Canny annotator. Deterministic;
      add it to `corpus/map.sh` (the conditioning is byte-stable even though the SD
      output isn't).
- [ ] **1×1 on-box render** — the single-tile path first: L7 overlay → SDXL img2img
      + ControlNet (canny) at a modest size, a cartography prompt derived from the
      spec (climate/era/biome words). `--map-render-sd PATH`, `--map-sd-model`
      (default an SDXL checkpoint), `--map-sd-strength`, `--map-sd-steps`. Verify a
      painted Isle of Vethûn renders on Metal at a size that fits 24 GB.
- [ ] **Tiled multi-tile render** — reuse the **`plakat.tiled.*`** SDXL hi-res
      machinery (base-anchored tiling) to cover larger tile grids tile-by-tile with
      overlap blending, so a 4×4 map renders without a single giant allocation.
      Memory-bound: gate tile size + document the safe envelope (the Metal
      single-buffer OOM ceiling), and self-limit like the 1.0 OOM watchdog.
- [ ] **Label/furniture re-composite** — the painted output loses the crisp labels;
      re-apply the 1.5.0 label + furniture pass (the `render.rs` compositor) **over**
      the SD result so the final map is painted *and* legible. Optional
      `--map-sd-raw` to skip the overlay.
- [ ] **Gate:** the conditioning (L7 overlay + Canny) is byte-stable in
      `corpus/map.sh`; the SD render itself is verified by a committed showcase image
      + a structural test (right size, labels composited), **not** a byte-check (it's
      the one non-deterministic map artifact — like every other SD pipeline in the
      proof corpus). Document the memory envelope.

## New deps

None expected — SDXL + ControlNet + tiled hi-res + Canny all already ship. (If a
dedicated cartography ControlNet is worth pulling, gate it behind discovery, not a
hard dep.)

## Later in Track M

1.7–1.8 **Bund hooks** (`plakat.map.*` scripting — drive the map from a scenario)
and the **urban fabric** (MAP-5: street graph via `petgraph`/DCEL, the urban
`Anchor` variants the spec already carries). The `map:` compile block (COMPILE-1
E-C4) is now unblocked and slots in here.

## Opportunistic / debt (off the critical path)

- L2 refinements: flat-resolution (parallel-thread artifact in flats), breach-vs-fill,
  delta detection, named-river ↔ traced-channel matching.
- L5: `rstar` nearest-coast index + concavity for natural-harbor placement.
- Labels: non-Latin shaping (`language` = `ar`/`ru`/`zh`) wants a real shaped font +
  `ab_glyph` (the 1.5.0 bitmap font is Latin-only by design).
- 1.1.0 carryovers: Flux regional (Flux broken on Metal → code-only), IC-Light.
- Memory-bound render debt: SD3.5 DreamBooth render, `regional.sh sdxl/sd35`.
