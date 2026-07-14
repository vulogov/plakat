# RFC PHOTOS-1 — plakat photos: TUI Photo & Image Gallery (3.x flagship)

**Status:** Implementation-ready · **Binary:** `plakat photos [ROOT_DIR]` · **Feature:** `--features photos`

Compound RFC (supersedes GALLERY-1/-EDIT/-EDIT2/-FEATURES/-VISION). This file is the load-bearing
implementation reference; see [`ROADMAP_3.0.0.md`](ROADMAP_3.0.0.md) for phase tracking.

## Overview

A terminal photo & image collection manager wrapping the 2.6–2.8 pipelines in a
**browse → curate → edit → generate** loop over a real image library: three panes (tree · album grid ·
command), image view + editor, T1 pixel ops (image/imageproc) and T2 ML ops (dispatch to the existing
img2img / outpaint / stylize / relight / segment / upscale / restore-faces / diffusion-upscale
pipelines), plus Gemini vision (describe / lookalike / analyze-and-generate / autotag).

## Storage model (per-album HJSON — there is no separate index)

- **Folder** (only sub-dirs) → `folder.hjson`; **Album** (holds ≥1 image) → `album.hjson`. Derived
  from contents; HJSON written lazily. Atomic writes (`.tmp` → rename).
- `album.hjson`: `name/description/cover/sort/thumb_size` + sparse `images{ "<file>": record }`.
- Per-image record: `exif` (auto, once) · `title/rating(0–5)/tags/color_label/caption/notes/flagged/
  rejected` · `score` (LAION, carried from the gen sidecar / `rank`) · `variants` · `analysis`
  (vision) · `edits[]` (append-only). See `src/photos/hjson.rs` for the implemented structs.

## Reused infra (all verified present in-tree)

img2img/outpaint/stylize/relight/segment `cli::*::run`, `imaging::upscale` (+ `--diffusion`),
`imaging::transparent::make_transparent`, `imaging::grid::compose_images`, `imaging::io::
read_parameters_chunk` + `patch_sidecar_score`, `imaging::metadata::GenerationMetadata`,
`pipelines::aesthetic` (rank/score), `pipelines::scrfd` + `restore-faces`, `prompt::gemini` /
`prompt::complete`, `ui/tui/screens/{prompt_editor,palette}`, `ui::progress::install_tui_sink`,
`ui::tui::output::OutputPane`, `centered_modal`. New deps (behind `photos`): `rawloader`,
`kamadak-exif`, `notify`; `image` extended with tiff/bmp/tga/qoi/ff/pnm/ico.

## Module layout (`src/photos/`)

`mod.rs` (run + event loop + 3-pane draw) · `library.rs` (walk/classify) · `hjson.rs` (storage) ·
then per RFC §25: `state.rs`, `events.rs`, `chords.rs`, `panes/` (tree, album_grid, image_view,
zoom_strip, exif_panel, edit_panel, output_pane, command, help), `edit/` (t1, t2, crop, canvas_mask),
`vision/`, `commands/`, `loader/` (standard, raw, exif, thumb), `library/` (walk, hjson, watcher,
doctor), `export/`, `ui/` (modal, prompt_modal, palette). `cli/photos.rs` = subcommand.

## Phased build (release 3.0.0 at Phase 1–2)

1. **Core library + display** — walk/classify + HJSON + loader/RAW/EXIF/thumb-cache + notify watcher +
   tree/grid panes + event loop. *(scaffold landed: library + hjson + runnable tree shell.)*
2. **Image view + curation** — rating/flag/reject/label/notes, culling, filter, smart albums.
3. **T1 pixel editing** — image/imageproc ops, crop, mask paint, undo.
4. **T2 ML editing** — dispatch to existing pipelines; prompt modal; background progress.
5. **Browse** — side-by-side/survey, stacking, dedup (pHash), timeline, batch rename/tag, import/export, watermark, portfolio.
6. **View analysis** — histogram, focus peaking, exposure, sharpness map, pixel probe, map.
7. **Vision + AI** — gemini vision, describe/lookalike/analyze-and-generate, autotag, caption, face-scan, semantic search.

## CLI

```
plakat photos [ROOT_DIR] [--thumb-size PX] [--thumb-workers N] [--no-watch]
              [--protocol kitty|sixel|iterm2|halfblocks] [--allow-halfblocks]
ROOT_DIR: $PLAKAT_PHOTOS_ROOT, then ~/Pictures.
```

## Deferred (RFC §28, GALLERY-* placeholders)

EXIF/XMP/IPTC write-back, local vision model (Moondream2), ICC color management, video thumbnails,
tethered shooting, print/PDF layout, ai-batch + rate limiter, cross-library prompt history.
