# `plakat photos` — the tour

A guided walk through the terminal photo & image collection manager. It's the flagship of `plakat`:
a keyboard-driven TUI that browses, curates, edits, generates, and shares an image library — entirely
offline once your models are cached. This tutorial follows the natural arc **organize → edit →
metadata → AI → collaborate → present → search-at-scale**. The exhaustive key reference is
[`KEYMAP.md`](KEYMAP.md); this is the narrative.

Everything is **non-destructive**: your curation lives in a human-editable `album.hjson` per folder
(never a hidden database), and edits are a replayable stack over the pristine original.

```bash
plakat photos ~/Pictures        # open a library at a directory
plakat photos                   # opens the current directory
```

---

## 1. Getting around

The screen is a **tree** (left) and a **grid** (right). `Tab` switches focus; `q` quits.

- **Tree** — your folders. A directory with images is an **album**; one with only sub-folders is a
  **folder**; both is a mixed album (its own images + children). The badge shows each album's own
  image count. `Enter`/`→` opens an album into the grid; `/` filters the tree by name.
- **Grid** — thumbnails of the open album (or a smart view). Arrows move the cursor; `Enter` opens the
  **image view** (single-image loupe). In the image view, `←`/`→` step through, `Z`/`z` zoom, `i`/`I`
  toggle the info panel, `H` the analysis panel (histogram / waveform / parade), `o` cycles diagnostic
  overlays (clipping zebras → focus peaking).

On a warm launch (a persisted index exists) the whole library opens as one **All Photos** grid — type
`:all` any time to get it, and `/` to filter it live.

## 2. Organize

- **Rate / flag / label** on the cursor or a `Space`-selection: `1`–`5` (rating, `0` clears), `f` flag,
  `x` reject, `c` cycle a colour label. `u`/`U` undo/redo curation.
- **Tags, caption, notes, title** — `t` / `e` / `N` / `T` open a quick editor.
- **Smart albums** — save a filter as a library-wide view. The filter grammar is composable, e.g.
  `rating>=4 -rejected camera:canon date>2023 has-gps tag:keeper`. Star entries at the top of the tree
  re-evaluate across every album on open.
- **Flatten** (`*`) shows every image beneath a folder / mixed album in one grid; curation still routes
  to each image's own source album.
- **Move / copy / trash** — `Ctrl-B m o` / `m p` move or copy the selection to another album (carrying
  the sidecar + record); `m t` soft-deletes to a restorable `.trash`.

## 3. Edit (the darkroom)

Open an image and press `E` for the **searchable edit palette**, or use the **`Ctrl-B` chord** map
(mirrors the palette): `g` geometry · `c` crop · `a` adjust · `k` colour · `x` effects · `e`
edit-stack · `s` stylize · `m` manage · `d` metadata · `r` retouch.

- **Adjustments** — exposure, brilliance, highlights/shadows, contrast, saturation, vibrance, warmth,
  definition, sharpen, denoise … each a live `←`/`→` slider. **Levels** and **curves** have their own
  interactive editors.
- **Local & brush masks** — apply any adjustment through a **graduated / radial** mask (`a g`, `a i`),
  or **paint** a freeform mask (`r x/k/s/w/u`: Space to stamp dabs, then it applies through them).
- **Retouch** — a crosshair pick-mode: spot-heal, clone stamp, red-eye, dodge/burn, 4-point
  perspective.
- **Layers** — `E → layers` composites images over the base with per-layer position/scale/opacity/blend
  and masks, then flattens to a new variant. The base is never touched.
- **Looks & filters** — one-tap presets (film, B&W mixes, tilt-shift, LUTs …).
- Every edit is replayable: `u`/`U` undo/redo, `Ctrl-B e 0` reverts to the original, and you can
  **copy/paste** an edit stack or save it as a **preset**.

## 4. Metadata — and writing it back

`Ctrl-B d` edits **title / author / copyright / capture-date / geotag** into the album record
(non-destructive, shown in the info panel). Shot metadata is **filterable** in smart albums
(`iso>3200`, `focal=50`, `lens:35`, `date:2024`, `has-gps`, `author:jane`, …).

`Ctrl-B d w` **writes that metadata (plus your tags) into the file's own binary EXIF** — JPEG, PNG,
WebP, and TIFF — so it travels with the file to other tools. It confirms first and never touches the
pixels.

## 5. AI (optional, opt-in)

The manager embeds `plakat`'s generation stack via a resident, OOM-guarded worker.

- **Create** — `Ctrl-B n`: text→image generate, img2img transform of the cursor image, a portrait from
  a prompt (+ the cursor image as an identity face), or a multi-person scene.
- **ML edits** — `M`: relight (IC-Light), ×4 upscale, face restore.
- **AI menu** — `A`: analyze-and-generate (describe a reference → img2img), face-scan (SCRFD/ArcFace),
  aesthetic auto-cull (rank + keep top N), and hybrid face-polish (AI-detected mask + non-AI smoothing).

A memory indicator on the status line steers you away from heavy AI when RAM is low.

## 6. Collaborate — shared volumes

Keep the library on **Dropbox / iCloud / NFS** and open it in **several `plakat photos` at once**.
Saves use a **lock-free three-way merge**: only the records *you* changed are written, so a colleague
rating *other* photos is never clobbered. Changes made elsewhere are picked up **live** (a smart album
added in one window appears in the others), a **`⟳ others editing`** badge lights up, and each record
shows **who edited it** (info panel). `:conflicts` reviews same-image conflicts (jump / take-theirs);
`:who` lists live instances (a **`👥 N`** status badge) and which album each is in. Set `PLAKAT_EDITOR`
to control the name in the stamp.

## 7. Present & share

- **Web gallery** — `Ctrl-B w` writes a portable, **fully-offline** HTML gallery (thumbnail grid +
  keyboard lightbox with captions, star ratings, an EXIF summary and tag chips). Open it locally or
  drop it on any static host.
- **Portfolio** — `Ctrl-B p` writes watermarked copies + a contact sheet.
- **Slideshow** — `S` in the image view auto-advances (higher-rated images linger longer); `[`/`]`
  pace, `r` 🔀 shuffle, `k` 🎥 **Ken Burns** pan/zoom.
- **Maps** — `m` plots geotagged photos on an offline world map; `:geocode` tags them with
  `place:<city>` (fetches a small gazetteer once, then offline).

## 8. Search — and search at scale

Three ways to find images, all offline once indexed:

- **Metadata search** — relevance-ranked over filenames / titles / captions / tags / prompts.
- **Perceptual lookalike** (`Ctrl-B l`) — "images that look like this" by a fast dHash; no model.
- **CLIP visual search** — text→image ("a dog on a beach at sunset") and semantic **lookalike**
  (`Ctrl-B L`) in CLIP's joint space.

For **large libraries**, a **derived index** keeps everything fast: a rebuildable snapshot of every
image's curation + EXIF (`album.hjson` stays the source of truth). `:embed` pre-computes + persists
CLIP vectors (int8-quantized, folded into the index); the model is kept **resident** so repeat searches
don't reload it; and above ~20k images text search ranks through a persisted **HNSW** index for
sub-linear results. `:stats` shows library facets (rating histogram, cameras, years); `:reindex`
rebuilds the index from scratch (it's always safe to delete).

---

## Where to go next

- [`KEYMAP.md`](KEYMAP.md) — every chord, `:` command, and image-view key.
- Prefer plain language? The `:` command pane speaks an album-scoped vocabulary — pipe steps with
  `then`: `find rating>=4 then upscale then export to ~/best 2000`.
- The design rationale lives in [`../RFC_PHOTOS_1.md`](../RFC_PHOTOS_1.md).
