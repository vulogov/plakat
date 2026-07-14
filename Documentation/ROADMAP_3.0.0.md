# plakat 3.0.0 — roadmap (planning)

**The 3.x flagship: `plakat photos` — a TUI photo & image collection manager.** Full spec in
[`RFC_PHOTOS_1.md`](RFC_PHOTOS_1.md). The 2.6–2.8 quality / curation / editing pipelines are its
engine; 3.x wraps them in a browse → curate → edit → generate loop over a real image library.

Binary: `plakat photos [ROOT_DIR]` · Feature flag: `--features photos` (does NOT chain on `ui`).

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Ground truth (verified before pivot)

All ~20 pieces the RFC reuses exist in the current tree (img2img/outpaint/stylize/relight/segment
`run`, transparent, compose_images, install_tui_sink, read_parameters_chunk, ml_upscale,
prompt_editor, palette, gemini, GenerationMetadata, scrfd, CLIP). The 2.8 additions
(`upscale --diffusion`, aesthetic scoring, `restore-faces`) are bonus photos-operations. The RFC's
storage model is **per-album `album.hjson`** (not a separate index) — so there is no "collection
index" to build; it's the HJSON model itself.

## Phased build (per RFC §29)

- [~] **Phase 1 — Core library + display.** `plakat photos ~/Photos` opens: tree + thumbnail grid.
  - [ ] `cli/photos.rs` + `cli/mod.rs` subcommand skeleton (feature-gated).
  - [ ] `photos/library/` — walk, classify (folder vs album), HJSON read/write (atomic).
  - [ ] `photos/loader/` — standard + RAW decode, thumbnail cache, EXIF reader (kamadak-exif).
  - [ ] `photos/library/watcher.rs` — notify watcher + 500 ms debounce.
  - [ ] `photos/panes/tree.rs` + `album_grid.rs` — layout, nav, thumbnail render.
  - [ ] `photos/state.rs` + `mod.rs` — event loop, focus model, 100 ms tick.
  - [ ] Tree mutations + selection model.
- [ ] **Phase 2 — Image view + curation** (rating/flag/reject/label/notes, culling, filter, smart albums).
- [ ] **Phase 3 — T1 pixel editing** (image/imageproc ops, crop, mask paint, undo).
- [ ] **Phase 4 — T2 ML editing** (dispatch to existing pipelines; prompt modal; progress).
- [ ] **Phase 5 — Browse features** (side-by-side/survey, stacking, dedup/pHash, timeline, batch, export).
- [ ] **Phase 6 — View analysis** (histogram, focus peaking, exposure, sharpness map, pixel probe).
- [ ] **Phase 7 — Vision + AI** (gemini vision, describe/lookalike/analyze-and-generate, autotag, face-scan, semantic search).

**3.0.0 releases** when Phase 1–2 (browse + curate) is usable — a real flagship, not an empty bump.
Later phases can land in 3.1+.

## New deps (all pure-Rust, optional, behind `photos`)

`image-extras`, `rawloader` 0.37, `kamadak-exif` 0.6, `notify` 7. Optional: `printpdf` (`photos-pdf`),
`ureq` (`photos-map`). Extend the existing `image` entry with `tiff`/`avif`/`qoi`.

## House-keeping

- [x] **Open 3.0.0** — branch off `main` (2.8.0 release), version bump `2.8.0 → 3.0.0`.
- [ ] Deferred features (RFC §28) tracked under GALLERY-* placeholder RFCs.
