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
- [~] **Phase 3 — T1 pixel editing.** DONE: non-destructive, replayable edit engine (`photos/edit.rs`
      — rotate ⟳/⟲/180, flip H/V, grayscale, brightness, contrast, crop 1:1) with a modal edit menu
      (`E`), undo (`u`) + revert (`0`). Pristine original backed up once to hidden `.plakat_edits/`
      (skipped by the walk); visible file re-derived by replaying the `edits` log in `album.hjson`
      (one re-encode regardless of edit count; works in normal + smart/search views). Unit-tested
      (apply/replay/roundtrip/backup-revert). REMAINING: interactive crop/mask paint (deferred —
      needs pixel-precise pointer input; the mask half folds into Phase 4 T2 inpaint).
- [~] **Phase 4 — T2 ML editing.** DONE: `M` menu dispatches the cursor image to existing pipelines
      via the `plakat::api` builders (`photos/mledit.rs` — ML upscale ×4, img2img w/ prompt, relight
      w/ prompt). Runs with the TUI **suspended** (leave alt-screen → job on a dedicated-runtime
      thread so no `block_on` on the event-loop thread → resume), showing the pipeline's own progress;
      output lands as a new `<name>_<op>.png` in the album, linked as a `variant`, cursor jumps to it.
      Unit-tested (`dest_path` dedup). REMAINING: true background (non-blocking) progress, mask/region
      steering (folds in the deferred Phase-3 mask paint), more ops (stylize/restore-faces), model choice.
- [x] **Phase 5 — Browse features** (core COMPLETE; watermark/portfolio → 3.1+). DONE: dedup (`#` — 64-bit dHash `photos/dedup.rs`, greedy
      grouping, tags all-but-best `dup` + focuses `tag:dup`); export (`X` — `photos/export.rs`, copy
      selection/view to a dir, optional `MAXPX` downscale, deduped); survey/compare (`=` — 2–4 images
      side-by-side, focus + rate/flag/reject the focused, `AlbumMode::Compare`); batch rename (`r` —
      `photos/rename.rs` `#`-run pattern, album-local, two-phase stage so intra-set swaps can't
      clobber, migrates record + edit backup); stacking (`S` — collapse derivative `variant`s under
      their base, `⧉N` badge, via `all_variant_names` + `rebuild_view`); timeline (`@` — modal of
      `YYYY-MM` EXIF-capture buckets over a date-sorted view, jump the grid to a month). All
      unit-tested. **Phase 5 COMPLETE** for the browse core; watermark/portfolio left as 3.1+ polish.
- [~] **Phase 6 — View analysis.** DONE: image-view analysis panel (`H`) — `photos/analysis.rs`
      pure `analyze()` → luma histogram (64 bins) + mean + highlight/shadow clip % + focus score
      (variance of the Laplacian); rendered as an 8-row bar chart + stats (clip flagged red),
      recomputed on ←/→. Unit-tested (flat/clip/edges). REMAINING: focus peaking + pixel probe
      (need per-pixel overlays on the graphics-protocol image — deferred, same constraint as mask paint).
- [~] **Phase 7 — Vision + AI.** DONE: Gemini vision autotag/describe (`A` menu → `t` tags / `d`
      caption). `prompt/gemini.rs` gained `describe_image` (image→text; re-encodes ≤1024 JPEG, inline
      base64 — tiny inline encoder, no new dep) + `photos/vision.rs` (VisionOp + reply parsing,
      unit-tested). Runs off-thread via `run_vision_job` (quick net call, no alt-screen suspend);
      tags merge, caption overwrites; feeds the existing `textsearch` metadata search — the *text*
      half of semantic search. (Metadata `?` search already shipped.) **+ CLIP visual search (`V`):**
      `pipelines/clip_embed.rs` `ClipEmbedder` loads openai/clip-vit-large-patch14 (shared aesthetic
      weights; both towers + projections via candle `ClipModel`, `get_image_features`/`get_text_features`
      + `div_l2_norm`) → 768-d joint embeddings; `photos/visual_search.rs` ranks the library by cosine
      with an in-session embed cache. Runs TUI-suspended (model load + per-image embed) → relevance
      view. **Persistent vector store DONE**: per-album hidden `.plakat_clip` binary cache
      (magic+count+[name,mtime,768×f32]) via `load_cache`/`save_cache` — seeded before embed, saved
      after, mtime-invalidated; corrupt files load empty (unit-tested). Load validated by candle's own
      CLIP example layout; `#[ignore]` real-load smoke test blocked here (cache disk offline — same
      failure `plakat rank` hits, not a code issue). REMAINING: lookalike (image→image),
      analyze-and-generate, face-scan.

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
