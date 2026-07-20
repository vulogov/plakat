# plakat 3.11.0 — roadmap (in progress)

**Theme — the Ken Burns slideshow.** 3.10 gave the slideshow shuffle + rating-weighted dwell. 3.11
adds the documentary **Ken Burns** effect: each slide slowly pans and zooms across the frame, turning
a still review into gentle motion.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — Ken Burns pan/zoom

- [x] **Ken Burns effect** (`k` in the slideshow) — each slide animates a smooth crop from one framing
      to another (random pan + zoom, smoothstep-eased), rendered by cropping the cached `working_base`
      each event-loop tick (`tick_ken_burns` → `render_ken_burns`; no re-decode). The motion spans the
      slide's (rating-weighted) dwell; each new slide picks a fresh `kb_from`/`kb_to` framing (window
      kept inside the image). A `🎥` marker in the `▶` badge + status; off by default (opt-in motion),
      cleared when the slideshow stops or you leave the image view. Pure crop-rect / easing math in
      `kb_lerp` + `kb_crop_rect`; 1 test.

## Track B — richer web gallery

- [x] **Embed captions / ratings / EXIF / tags in the exported HTML** — `webgallery::export` now takes
      `Photo` items (title, rating, tags, date, EXIF summary). The grid shows a `★` overlay on rated
      images; the lightbox shows the caption + stars, a `date · camera · lens · exposure · ISO` line
      (`exif_summary_line`), and tag chips — all embedded as `data-*` attributes, still self-contained
      (inline CSS/JS, no network). 2 tests.

## Track C — WebP / TIFF EXIF write-back

- [x] **WebP EXIF** — `exifwrite::embed_webp` extends the `d w` write-back to WebP: a simple
      `RIFF/WEBP/VP8|VP8L` file is upgraded to the extended `VP8X` form (EXIF flag + canvas size) and an
      `EXIF` chunk appended; an extended file has its flag set + stale `EXIF` replaced. Round-trips
      through the `kamadak-exif` reader (date + GPS). 1 test.
- [ ] **TIFF EXIF** — append a merged IFD (little-endian) at EOF and repoint the header, keeping the
      image strips in place. (In progress / may defer if the IFD-offset surgery proves too fiddly.)

## Ground rules (unchanged)

- Non-destructive; the effect is view-only (never alters the file or the edit stack).
- Cheap: reuse the per-image `working_base` cache; bound the render resolution like normal viewing.
- Verify-safe: default CLI image output byte-identical; new logic lands with unit tests where it's
  pure (the crop-rect / easing math).
