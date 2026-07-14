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
  - [x] `cli/photos.rs` + `cli/mod.rs` subcommand (feature-gated; root = arg > env > ~/Pictures).
  - [x] `photos/library.rs` — walk, classify (folder vs album), image/RAW detection. Tested.
  - [x] `photos/hjson.rs` — folder/album HJSON store, sparse per-image records, atomic writes. Tested.
  - [x] `photos/loader.rs` — standard + RAW (2×2-quad demosaic) decode + XDG thumbnail cache. Tested.
  - [x] `photos/exif.rs` — kamadak-exif → ExifRecord.
  - [x] `photos/watcher.rs` — notify watcher + 500 ms debounced rescan.
  - [x] `photos/mod.rs` — three-pane shell, Tree pane (nav/collapse) + Album grid (lazy thumbnails,
        StatefulProtocol, [/] columns), Tree↔Album focus, event loop + tick.
  - [x] Tree mutations (n/a/R/D via command pane, pending-action model) + grid selection model
        (Space/Ctrl-A/D/I). **Phase 1 COMPLETE.**
- [~] **Phase 2 — Image view + curation** (the 3.0.0 release gate).
  - [x] Image view (Enter → full-pane render, ←/→, Esc) + EXIF overlay (`i`, cached kamadak read).
  - [x] Curation: 1–5/0 rating, `f` flag, `x` reject, `c` color-label → album.hjson; grid badges.
  - [ ] Remaining: culling mode (`Ctrl-b c`), filter bar + sort (`Ctrl-b f`), smart albums (`Ctrl-b F`),
        notes/caption editing.
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
