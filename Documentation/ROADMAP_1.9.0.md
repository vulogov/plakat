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
- [ ] **Realize lakes (close the "label-only" gap)** — `WaterSpec.lakes` is in the
      spec + labelled, but no water body is drawn. Place an endorheic depression at
      the lake's anchor and fill it to a level (a small radial basin in L1 + a sea-
      style mask in L3 so biome/coast treat it as water), then the render draws blue.
      Deterministic; ~the falloff machinery already exists. *Approach: a `lake_mask`
      alongside the sea mask; render lakes before rivers.*
- [ ] **`plakat.map.paint` scripting word** — `( spec-path style -- handle )` SD-
      painted counterpart to `plakat.map.render`; reuse `render_sd::render_sd` to a
      temp file → load into an image handle. Honours `plakat.map.layout`/`.erosion`.
- [ ] **Urban lot subdivision** — split each U1 block quad into 2–4 building lots
      (strip split along the longer edge, noise-jittered), drawn as finer parcels so
      town maps read at the building scale. Deterministic; pure geometry on the quads.
- [ ] **Non-Latin label shaping** — the 1.5 bitmap font is Latin-only; `language` =
      `ar`/`ru`/`zh` wants a real shaped font + `ab_glyph` (the one map feature that
      adds an asset/dep — gate it behind a `shaped-labels` feature flag; bitmap stays
      the default so the corpus stays asset-free + byte-stable).

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
