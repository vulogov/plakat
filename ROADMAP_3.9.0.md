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

## Track D — editing: interactive brush masks

- [x] **Brush-mask local adjustments** (`Ctrl-B r x/k/s/w/u`) — the freeform companion to 3.8's
      parametric (graduated/radial) local adjustments. New `EditOp::BrushAdjust { adjust, amount,
      dabs }`: paint soft dabs in the pick-mode (Space stamps, `+`/`-` sizes, a magenta tint previews
      the mask), Enter applies the chosen adjustment (exposure ±, saturation, warmth, blur) through the
      painted mask. Reuses `local_base_op` + a new `adjust::brush_mask` (union of cosine-falloff
      circles) + `blend_masked`. The dabs are stored in the op (per-mille), so it's a single
      **replayable** edit that re-applies exactly on the pristine original — closes the deferred 3.8
      Track-C follow-up. Serde round-trips the dab list; 1 test.

## Track E — shared volumes & concurrent instances (the important one)

- [x] **Concurrency-safe `album.hjson` on shared volumes** (Dropbox / NFS) — the library can live on a
      synced/shared volume with **multiple `plakat photos` instances** open at once without losing
      curation. Two lock-free mechanisms (cross-machine file locks are unreliable, so we don't rely on
      them):
  - **Merge-on-write** (`hjson::merge_album` / `write_album_merged`): every save of the open album
    re-reads the current on-disk copy and overlays only the records/fields *this* instance changed
    (three-way merge against a `album_baseline` captured at load). A concurrent instance editing
    *other* images is never clobbered; same-record conflicts resolve last-writer-wins. `save_album`
    and the open-album path of `edit_album_meta_at` route through it.
  - **External-change reload** (`reload_album_if_changed`): a throttled `(mtime, len)` stamp on the
    open album's `album.hjson`; when another instance / a sync changes it, we adopt those changes
    (merging in any of our own not-yet-saved edits) and refresh the view — a metadata-only reload that
    keeps the file list + cursor.
  - 4 merge tests (concurrent different-image edits both survive; adds/deletes/conflicts; album-level
    fields; end-to-end `write_album_merged`).
- [x] **`folder.hjson` (smart albums / presets) merge + cross-instance sync** — `hjson::merge_folder`
      / `write_folder_merged` extend the same three-way merge to the root folder's name-keyed lists
      (a shared `merge_named` helper). All smart-album/preset writes route through a new
      `App::edit_folder` (read fresh baseline → apply delta → merge-write → refresh the live list), so
      concurrent instances adding different smart albums both survive. The throttled poll now also
      watches `folder.hjson`, so a smart album added by one instance **appears in the others'** tree
      without a restart. 1 merge test.
- [x] **"Others editing" indicator** — a yellow `⟳ others editing` badge in the top status bar,
      shown for ~12 s after each externally-driven reload (`last_shared_change`), so frequent
      concurrent edits keep it lit — a clear signal another instance / a sync is touching the library.
- [x] **Same-machine `flock` fast-path** — `hjson::FileLock` takes a best-effort `flock(LOCK_EX)` on a
      `.lock` sibling around each merge-write, so concurrent writers on one machine serialize (no
      interleaved read-modify-write at all) on local disks. Bounded spin then give-up (never stalls the
      UI); a no-op where `flock` is unsupported — the merge still guarantees correctness, so the lock
      only ever *adds* safety. `libc::flock`, no new dependency. 1 test (mutual exclusion).
- [x] **Per-record "last editor" note** — each record we change is stamped `last_editor`
      (`$PLAKAT_EDITOR` or `user@host`) + `last_edited` (ISO-8601 UTC via a dependency-free `now_iso`).
      Shown in the info panel ("Edited by …"). On a same-record **conflict** (we and another instance
      both changed the same image since our load) the merge reports it and the save status warns
      `⚠ saved — N record(s) also changed elsewhere, kept yours (also edited by <who>)`. 2 tests.

## Docs

- [x] KEYMAP: `Ctrl-B w` web gallery + `S`/`[`/`]` slideshow + `Ctrl-B d w` EXIF write-back + brush.
- [~] README what's-new + PHOTOS_TUTORIAL note + a shared-volume section.

## Distribution

- [ ] Per-cut (crates.io + GH release via the tag-triggered CI + FF `main`).

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec. The web gallery
  (writes named dir) is palette/chord-only, exactly like `portfolio` — not in the NL planner.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.

## Candidate follow-ups (not committed)

- Slideshow: random order, inter-image fade, per-slide dwell from rating.
- EXIF write-back for WebP/TIFF (currently JPEG/PNG); XPKeywords for tags.
- Shared-volume: a per-image edit-history (append last N editors) rather than just the latest; an
  interactive "review conflicts" pane; presence heartbeat (`.plakat_presence`) listing live instances.
