# plakat 3.9.0 — roadmap (in progress)

**Theme — present & share.** The manager can now organize, edit, and place a collection; 3.9.0 is
about getting the results *out* and *shown*. A portable offline web gallery to hand someone, and a
hands-free slideshow for viewing on the machine.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — share

- [x] **Static web gallery** (`Ctrl-B w`, prompt `DIR [MAXPX] | title`) — `src/photos/webgallery.rs`
      writes a portable, **fully-offline** folder: `index.html` (self-contained — inline CSS + JS, no
      CDN / no network), a `thumbs/` grid (bounded JPEGs) and a `full/` set (source copies, optionally
      down-sized to `MAXPX`). Responsive dark grid + a keyboard lightbox (←/→/Esc, click to advance).
      Title defaults to the album name; every string HTML-escaped. Create-only, like
      `export` / `portfolio` — the album copies stay put. Sits alongside the print-oriented
      `portfolio` (watermarked copies + contact sheet, `Ctrl-B p`).

## Track B — view

- [x] **Slideshow** (`S` in the image view) — auto-advances through the current view on a timer,
      wrapping at the end; `[` / `]` slow down / speed up (1–30 s, default 4 s), `S` again or `Esc`
      stops it. A green `▶ Ns` badge in the top bar shows it running. Paced off the event-loop tick,
      so thumbnails keep decoding and input stays responsive.

## Track C — metadata write-back

- [x] **Binary-EXIF write-back** (`Ctrl-B d w`, confirms) — `src/photos/exifwrite.rs` writes an
      image's album-record metadata (title / author / copyright / capture-date / geotag) into the
      **file's own EXIF**, in place, so it travels with the file. Hand-rolled + dependency-free like
      `scrub`: builds a little-endian TIFF/EXIF block from scratch (IFD0 + Exif SubIFD for
      DateTimeOriginal + a GPS IFD) and splices it into a JPEG (`APP1 "Exif\0\0"`) or PNG (`eXIf`
      chunk), replacing any existing EXIF; the pixel stream is never touched. Tag map: title →
      ImageDescription, author → Artist, copyright → Copyright, date → DateTime + DateTimeOriginal,
      geotag → GPS IFD. Round-trips through the `kamadak-exif` reader (5 tests, JPEG + PNG). Closes
      the deferred 3.8 metadata gap (only records with writable fields are touched; JPEG/PNG only).

## Docs

- [x] KEYMAP: `Ctrl-B w` web gallery + `S`/`[`/`]` slideshow + `Ctrl-B d w` EXIF write-back.
- [~] README what's-new + PHOTOS_TUTORIAL note.

## Distribution

- [ ] Per-cut (crates.io + GH release via the tag-triggered CI + FF `main`).

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec. The web gallery
  (writes named dir) is palette/chord-only, exactly like `portfolio` — not in the NL planner.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.

## Candidate follow-ups (not committed)

- Single-file HTML gallery (data-URI embedded) — one emailable file for small sets.
- Slideshow: random order, inter-image fade, per-slide dwell from rating.
- Interactive / brush masks for local adjustments (the deferred 3.8 Track-C follow-up).
- EXIF write-back for WebP/TIFF (currently JPEG/PNG); XPKeywords for tags.
