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
- [ ] **`plakat.map.paint` scripting word** — the SD-painted counterpart to
      `plakat.map.render` (scripting currently stays on the GPU-free linework path).
- [ ] **Urban lot subdivision** — split each U1 block into building lots (a strip /
      OBB split), so town maps read at the parcel scale. Deterministic.
- [ ] **Non-Latin label shaping** — the 1.5 bitmap font is Latin-only; `language` =
      `ar`/`ru`/`zh` wants a real shaped font + `ab_glyph` (the one map feature that
      adds an asset/dep — gate it behind a feature flag).

## B — carried product debt

- [ ] **Flux regional prompting** — code-only on Metal (Flux is broken on Metal); the
      regional path exists for SD1.5/SDXL/SD3.5. Land the Flux code path + CPU/CI proof.
- [ ] **IC-Light** — relighting (a 1.1.0 carryover). Net-new pipeline.
- [ ] **Memory-bound render debt** — SD3.5 DreamBooth render + `regional.sh sdxl/sd35`
      OOM on 24 GB. Document the envelope; verify on a bigger box or shrink the demo.

## C — corpus / verification

- [ ] **Fill `corpus/images/train/`** — run `resume_train.sh` (the one ungenerated
      corpus proof that's *not* memory-blocked — a few GPU-minutes).
- [ ] **Map showcase in the gallery** — the town + eroded-island renders aren't in
      `GALLERY.md`; add a curated map section.

## D — new direction

- Open. (Past cycles: a new model family, a new editing primitive, a new track.)

## Notes

- Build with `--features metal` for GPU on Apple Silicon (`cargo build --release`
  alone is CPU-only — the default has no backend).
- The map engine's only heavy dep is `petgraph` (urban) + `noise` (terrain); keep new
  deps gated behind features where practical.
