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

- [x] **Conditioning builder — DONE.** `render_sd::build_conditioning` = the styled
      **base map** (terrain/coast/rivers/roads, *no labels*) via the 1.5.0
      `paint_base_map` (factored out of `render`), the img2img init **and** the Canny
      ControlNet source. `--map-dump-conditioning PATH`. Deterministic → byte-stable
      `corpus/images/map/island-conditioning.png`, byte-checked in `corpus/map.sh`.
- [x] **1×1 on-box render — DONE.** `render_sd::render_sd`: base → **SDXL img2img +
      Canny ControlNet** (reuses the `img2img` pipeline wholesale; Canny auto-
      annotates the base) with a **cartography prompt derived from the spec**
      (climate/era/elevation/biome words, leading with the "fantasy map" trigger).
      `--map-render-sd PATH`, `--map-sd-model` (**any** plakat model — sdxl default,
      sd15/sd21/turbo/HF-repo all work), `--map-sd-lora` (**optional**; SDXL-family
      defaults to `Muapi/fantasy-map`, others none; `none` disables),
      `--map-sd-strength`/`--map-sd-steps`/`--map-sd-guidance`/`--map-sd-raw`.
      Verified end-to-end (Isle of Vethûn painted at 512²; `corpus/map_render.sh`).
- [x] **Label/furniture re-composite — DONE.** `render::apply_labels_and_furniture`
      (factored out of `render`) re-applies the 1.5.0 labels + furniture **over** the
      SD output so the painted map stays legible; `--map-sd-raw` skips it.
- [x] **compvis-layout SDXL LoRA support — DONE.** `pipelines/lora.rs`
      `compvis_unet_kohya_to_diffusers` remaps compvis/SAI block keys
      (`lora_unet_input_blocks_*`/`_middle_block_*`/`_output_blocks_*`) to the
      diffusers layout (`down_blocks`/`mid_block`/`up_blocks`) the candle UNet uses,
      hooked into `resolve_lora_base` as a fallback **after** the normal lookup and
      **validated against the real base keys** (a wrong guess just doesn't match — no
      regression). Standard `layers_per_block=2` stage arithmetic; the sub-module name
      after the block is layout-invariant and passes through. Lives in the shared
      `merge_loras_into_weights` (sd_core), so **every** LoRA path benefits (generate /
      portrait / img2img / scenario / map). Verified: `Muapi/fantasy-map` went from
      **0/722 → 722/722** UNet targets merged; the painted Isle of Vethûn carries the
      full style. 2 remap unit tests + diffusers/TE-untouched guard.
- [x] **Tiled multi-tile render — DONE.** A canvas wider/taller than `--map-sd-tile`
      paints in **overlapping image-space tiles**, each a full img2img+Canny pass that
      fits memory, **Hann-feathered** back into the canvas (`paint_tiled` / `tile_starts`
      / `hann2d` in `render_sd.rs`). The SD pipeline + LoRA load **once** and every tile
      reuses it. Chosen over latent-space tiling because plakat's tiled denoise doesn't
      compose with ControlNet — and the deterministic conditioning base already supplies
      global structure, so independent per-tile paints stay coherent. Memory-safe by
      construction (each tile is a normal SDXL render; per-tile work completes before the
      next). `--map-sd-tile`/`--map-sd-tile-stride`. Verified: 2×2 forced tiling on the
      island (`corpus/map_render.sh` → `island-painted-tiled.png`). 3 pure-helper tests
      (tile coverage + edge-snap + Hann window).
- [x] **Gate (phase 1) MET.** The conditioning is byte-stable in `corpus/map.sh`; the
      SD render is driven by `corpus/map_render.sh` (deterministic conditioning check +
      the GPU paint, `MODEL=`/`LORA=` env, `NO_GPU=1` to skip) with a committed
      showcase `corpus/images/map/island-painted.png` — **not** byte-checked (the one
      non-deterministic map artifact). Structural tests: conditioning determinism +
      label-free, prompt derivation, `round8`.

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
