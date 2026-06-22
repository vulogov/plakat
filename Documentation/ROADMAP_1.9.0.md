# plakat 1.9.0 — roadmap

The planned **map track is complete** (MAP-1 spec/parser · MAP-2 geometry · MAP-3
linework + vector export · MAP-6 painted SD + tiled · MAP-4 toolchain integration ·
MAP-5 urban fabric), and the geography is realistic + tunable (eroded coasts,
configurable town plans, the `--map-erosion` knob across every surface).

1.9.0 has **no single mandated track** — it's a polish / debt / opportunity cycle.
The candidate directions below are independent; pick per appetite. The through-line
holds: deterministic work earns a byte-stable on-box corpus proof; GPU work is
verified by a committed showcase, not a byte-check.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — map polish (small, high-leverage)

- [x] **L2 named-river ↔ traced-channel matching — DONE.** `resolver::match_rivers_to_channels`
      pairs each named spec river to the traced channel whose mouth is nearest the
      river's resolved mouth (greedy + unique), so labels + GeoJSON `id`/`name` follow
      the *intended* watercourse, not the longest channel. The render labels the
      matched channel; the GeoJSON exports every channel with the matched one carrying
      its real id + name (`The Ashflow`), the rest as `channel_<i>`. Refactored the
      mouth resolution into a shared `build_context`. Deterministic; matcher unit test.
- [x] **Realize lakes — DONE.** `engine::apply_lakes` carves a smooth sub-sea-level
      basin (`smoothstep` shore, floor below `DEFAULT_SEA_LEVEL`) at each spec lake's
      anchor *after* normalize, so the **existing** coast/biome/hydrology/render
      pipeline realizes it as water for free: blue fill + a shoreline ring, marked as
      Sea biome with a beach edge, and rivers drain into it (lake cells count as sea
      in tracing). `lake_radius` maps `size` → extent fraction. Verified: the island's
      Mournmere (crater lake, endorheic, centre) now renders as a blue tarn in the
      volcanic massif. Lake unit test; geographic proofs regenerated.
- [x] **`plakat.map.paint` scripting word — DONE.** `( spec-path style -- handle )`
      SD-painted counterpart to `plakat.map.render`: `render_sd::render_sd` (img2img +
      Canny) → temp PNG → image handle, bridged to async via `block_in_place` like the
      cascade words. Honours `plakat.map.layout`/`.erosion` (shared `load_spec_with_overrides`).
      Verified on Metal (island painted, 722/722 LoRA → handle → `plakat.save`).
- [x] **Urban lot subdivision — DONE.** `subdivide_block` splits each block quad into
      a small bilinear grid of **building lots** (split count from edge length 1–3 per
      axis, noise-jittered split lines), each lot inset for a thin lane, with per-lot
      tone wobble — so towns read at the building scale, not big colored blocks. Works
      across radial/grid/organic. Deterministic; lot subdivision unit test; town proofs
      regenerated.
- [x] **Non-Latin label shaping — DONE.** A `shaped-labels` Cargo feature pulls
      `ab_glyph` (optional dep) and adds `plakat map --map-font <PATH.ttf>`: a real
      TrueType/OpenType font rasterizes every label (per-glyph, kerned, LTR) via a
      thread-local active font that the bitmap functions defer to. **Bitmap stays the
      default** — no feature → byte-stable, asset-free corpus; and even with the
      feature compiled in, no `--map-font` → byte-identical bitmap output (verified).
      No font vendored (user supplies). Cyrillic + CJK render; complex shaping
      (Arabic RTL/contextual) is glyphs-only, pending a full shaper. Verified: a
      Cyrillic town (`Соль-Мере`, `Северные Врата`, `Рынок`) renders correctly.

## B — proposed optional map features (bigger, opt-in)

- [ ] **River + dry canyons** — carve relief along high-accumulation channels (a
      negative-elevation gorge with steep walls + hill-shading) and realize the
      schema's `terrain.rift_valleys` as **dry canyons**. Opt-in via a `canyons` spec
      flag / `--map-canyons`; deterministic. *Med.*
- [ ] **Lakes + marshland as real water** — beyond the polish "realize lakes": lake
      polygons with a reflection tint, marsh hatching for `Wetland` biome regions, and
      river **deltas** at navigable mouths. *Med.*
- [ ] **Plateaus / mesas** — realize `terrain.plateaus` (a schema stub today) as
      flat-topped raised terrain with a scarp edge. *S–M.*
- [ ] **Political layer** — draw region borders + polity fills/labels from the unused
      `RegionSpec.political` (`PoliticalSpec`/`BorderSpec` already in the schema). *Med.*
- [ ] **Seasonal / biome palette variants** — `--map-season winter|arid|autumn`
      reshades the biome fills (snow line, dry browns). *S.*
- [ ] **Game-grid overlay** — optional hex/square grid + coordinate labels for TTRPG
      use (`--map-grid hex|square`). *S.*
- [ ] **Multi-tile world maps** — the geographic engine caps at 2048²; stitch it
      across a larger tile grid for continent-scale maps (memory-bound; tile + blend
      like the SD path). *Med.*

## C — model-training expansion

- [ ] **See [`PLAN_TRAINING_EXPANSION.md`](PLAN_TRAINING_EXPANSION.md)** — concrete
      per-family plans for **SD 2.1 LoRA + DreamBooth** (start here, on-box), **PixArt-Σ
      LoRA**, **SD 3.5 Textual Inversion**, **Stable Cascade Stage-C LoRA**; **Flux
      back-burnered** (unverifiable on Metal). Each lands as its own increment with a
      `*_train.sh` driver + committed showcase.

## D — carried product debt

- [ ] **Flux regional prompting** — code-only on Metal (Flux is broken on Metal); the
      regional path exists for SD1.5/SDXL/SD3.5. Land the Flux code path + CPU/CI proof.
- [ ] **IC-Light** — relighting (a 1.1.0 carryover). Net-new pipeline.
- [ ] **Memory-bound render debt** — SD3.5 DreamBooth render + `regional.sh sdxl/sd35`
      OOM on 24 GB. Document the envelope; verify on a bigger box or shrink the demo.

## E — corpus / verification

- [ ] **Fill `corpus/images/train/`** — run `resume_train.sh` (the one ungenerated
      corpus proof that's *not* memory-blocked — a few GPU-minutes).
- [ ] **Map showcase in the gallery** — the town + eroded-island renders aren't in
      `GALLERY.md`; add a curated map section.

## F — new direction

- Open. (Past cycles: a new model family, a new editing primitive, a new track.)

## Notes

- Build with `--features metal` for GPU on Apple Silicon (`cargo build --release`
  alone is CPU-only — the default has no backend).
- The map engine's only heavy dep is `petgraph` (urban) + `noise` (terrain); keep new
  deps gated behind features where practical.
