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
- [x] **Phase 2 — Image view + curation** (the 3.0.0 release gate). COMPLETE.
  - [x] Image view (Enter → full-pane render, ←/→, Esc) + EXIF overlay (`i`, cached kamadak read).
  - [x] Curation: 1–5/0 rating, `f` flag, `x` reject, `c` color-label → album.hjson; grid badges.
  - [x] Filter bar (`/`, view layer + grammar rating/flag/rejected/tag/ai/free-text, unit-tested) +
        culling loupe (`C`, keep/reject/rate one-at-a-time).
  - [x] Notes/caption/title/tags editing (`e`/`N`/`T`/`t` → command pane, prefilled, empty clears)
        + sort order (`s` cycles name/date/rating/score, persisted in `album.hjson`, cursor-stable).
        Shown in the image-view panel (incl. `--import` generation recipe). Unit-tested.
  - [x] Smart albums — library-wide saved searches. `F` saves the current filter (named) to root
        `folder.hjson`; ★ tree rows open a cross-album grid of matches; curation routes writes back
        to each image's own album (path-keyed record model, `smart_*` maps + `edit_record_at`); `D`
        deletes the saved search. **Phase 2 COMPLETE.**
- [x] **`--import` for the generation commands** (user request) — `generate` + `upscale` /
      `portrait` / `multiperson` / `img2img` / `outpaint` / `stylize` / `relight` all gain
      `--import <album>` (+ `--import-move`): the output (and its `.json` sidecar) is copied/moved
      into the album, its gen params (`GenerationMetadata`) are written into the `album.hjson`
      per-image record, and the album is updated. Shared helper `cli/import.rs`
      (`ImportArgs` flatten + `run_with_import` snapshot wrapper); feature-gated on `photos` with a
      fast-fail build hint. Closes the loop: generate → land curated in the manager (live via the
      watcher). RFC_PHOTOS_IMPORT.md.
- [x] **Metadata semantic search** (`?`) — library-wide TF-IDF relevance ranking over each image's
      text metadata (filename/title/caption/notes/tags + `--import` prompt/model). Extracted the
      History ranker to a shared, feature-agnostic `crate::textsearch`; a relevance-ranked smart
      view (curation routes back to source albums). Unit-tested (`doc_for` + moved ranker tests).
      Visual/CLIP-embedding search + a derived index/vector store are the Phase-7 follow-on (see the
      storage note below).
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
- [x] **`photos` is default-on** — `default = ["ui", "photos"]`. The flagship ships in
      `cargo install plakat` out of the box; `--no-default-features` still gives the lean CLI.
- [x] **Cut 3.0.0** — Phase 1–2 (browse + curate) + `--import` complete; README "What's new" +
      `PHOTOS_TUTORIAL.md` + RELEASE_HISTORY migration; 1724 lib tests green. Tagged `v3.0.0`.
- [ ] Deferred features (RFC §28) tracked under GALLERY-* placeholder RFCs.
