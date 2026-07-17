# plakat 3.2.0 — roadmap (shipped)

**The `plakat photos` editor becomes a darkroom.** 3.1 gave the collection manager a two-tier editor
(pixel + model), search, and a natural-language command pane. 3.2 deepens the *non-destructive* pixel
editor into a full editing surface — layers & masks, a complete tonal/colour adjustment set, and
file-management ops — every one reachable from **both** the searchable edit palette (`E`) and the
free-form `:` command (deterministic fast-path + grounded LLM, closed album-scoped vocabulary).

Binary: `plakat photos [ROOT_DIR]` · Feature flag: `--features photos` · Storage: per-album
`album.hjson` (additive, sparse). Everything non-destructive; default CLI image output byte-identical.

Status: `[x]` done.

## Editing surface (Phase 8 + adjustments + management)

- [x] **Exact crop / resize** — `EditOp::CropPx` (centred pixel crop) + `EditOp::Resize` (fit-within,
  Lanczos); palette prompts (`WxH` / `W×H` / `N`). Crop-box keys documented in `Ctrl-B h`.
- [x] **Layers** (`photos/layers.rs`) — interactive overlay stack over the cursor image.
  `Layer{src,x,y,scale,opacity,blend}` (x/y/scale = fractions of base → resolution-independent) +
  `Blend`(normal/multiply/screen/overlay). Live composited preview + top-left HUD; arrows move,
  `+/-` scale, `< >` opacity, `b` blend, `{ }` z-order, `n/p` select, `x` delete, `a` add
  (album file or path), Enter **flatten** → deduped `_layered.png` variant (base untouched).
  Stack persisted on `ImageRecord.layers` (hjson `LayerEntry`).
- [x] **Per-layer masks** — `Mask::Shape{kind:ellipse|rect, feather, invert}` (feathered region) or
  `Mask::Image{src}` (grayscale luminance matte). `m` cycle · `k` matte · `[ ]` size · `,/.` feather
  · `/` invert. **Free-position** shape masks via the `M` sub-mode (arrows move the mask; resize
  pivots on its own centre). hjson `MaskEntry`.
- [x] **Tonal / colour adjustments** (`edit::adjust`) — exposure, brilliance (adaptive), highlights,
  midrange, shadows, black point, brightness, contrast, saturation, vibrance, warmth, tint,
  definition, sharpen/soften, noise reduction. Signed-amount `EditOp`s; per-pixel luma-weighted tone
  bands + spatial (blur-pass) sharpen/definition/denoise. Up/down palette entries; chainable.
- [x] **Auto-enhance** — per-channel histogram stretch (auto levels + auto colour balance), one tap.
- [x] **Straighten** — rotate by an angle (tenths of a degree, bilinear) + auto-crop the empty
  corners (largest-inner-rect). Palette prompts for degrees; NL emits `straighten:<deg>`.
- [x] **Strip metadata** (`photos/scrub.rs`) — remove EXIF/XMP/IPTC/GPS in place; **lossless** JPEG
  (APP1..APPn/COM splice) + PNG (eXIf/tEXt/… chunk drop), else decode+re-encode. Confirms first.
- [x] **Convert / resize** — write a NEW in-album file (`jpg`/`png`/`webp`); longest-side cap or
  JPEG target-KB (quality binary-search). Source untouched; deduped filename.

## Free-form command (`:`) parity

- [x] `nl::edit_op_word` maps natural phrasings → canonical tags; `EditOp::from_tag` resolves
  directional verbs (sharpen / warmer / desaturate / lift shadows / recover highlights / add clarity
  / auto enhance / `straighten:N` …) with sensible default amounts.
- [x] New actions `StripMeta` + `Convert{fmt,max_px}`; LLM system prompt enumerates the full tag
  vocabulary. Stays inside the closed, album-scoped, **no external read / no exec** model
  (`export` + `convert` are the only outward/create-only writes).

## Verification

- [x] 64 `photos::*` unit tests (blend math, mask coverage, adjustment pixel direction, from_tag
  verbs, lossless JPEG strip, convert resize + target-KB, NL verb pipelines) + full lib suite green.
- [x] No warnings; docs (`PHOTOS_TUTORIAL.md`, README, this roadmap) updated.
