# `plakat photos` — the photo & image collection manager

`plakat photos` is the 3.x flagship: a full-screen terminal application for
**browsing, curating, and generating into** a library of images — your
camera photos, your RAW files, and everything plakat generates. It reuses the
same decode, EXIF, aesthetic-scoring, and generation pipelines as the rest of
plakat, so it's not a separate app bolted on: it's the collection front-end to
the engine you already have.

```
plakat photos ~/Pictures
```

With no argument it opens `$PLAKAT_PHOTOS_ROOT`, or `~/Pictures` if that's unset.

> **Terminal support.** Thumbnails and the image view use the terminal's
> graphics protocol (Kitty, Ghostty, WezTerm, iTerm2, or any Sixel terminal).
> In a terminal without graphics support the UI still runs — you just get
> placeholders where images would be. `plakat photos` ships **on by default**
> (the `photos` feature); a lean build (`--no-default-features`) omits it.

---

## 1. The layout

The screen is three panes plus a status bar and a command line:

```
 ┌ status ─────────────────────────────────────────────┐
 │ ~/Pictures  ·  Album: Iceland  ·  42 images  · …     │
 ├ tree ───────┬ album grid ──────────────────────────┤
 │ ▸ 2023      │  ▢ ▢ ▢ ▢ ▢ ▢                          │
 │ ▾ 2024      │  ▢ ▢ ▢ ▢ ▢ ▢                          │
 │   • Iceland │  ▢ ▢ ▢ ▢ ▢ ▢                          │
 │   • Faroes  │                                        │
 │ ▸ Generated │                                        │
 ├─────────────┴────────────────────────────────────────┤
 │ :                                                     │  ← command line
 └───────────────────────────────────────────────────────┘
```

- **Tree** (left) — your folders. plakat classifies each directory as a plain
  **folder** (holds sub-folders) or an **album** (holds images). Albums are the
  unit of curation.
- **Album grid** (right) — thumbnails of the selected album, lazily rendered.
- **Status bar** (top) — root, current album, image count, active filter.
- **Command line** (bottom) — where rename / new-album / delete prompts and the
  filter appear.

`Tab` moves focus between the tree and the grid. `q` quits from anywhere.

**Quickhelp.** `Ctrl-B` is a leader key: press it, then `h` for a **key-chord** card
or `H` for a **commands** card. The card is **contextual** — it shows what's
relevant to where you are (tree vs grid vs image view vs cull vs compare) and is
seeded with live state (current sort, how many images are selected, the active
filter, whether stacking is on). Any key closes it.

### The storage model — `album.hjson`, not a database

There is **no hidden index**. Each album stores its metadata in a plain
`album.hjson` file *in the album directory*, next to the images. Ratings, tags,
flags, colour labels, captions, and (for generated images) the full generation
recipe live there as sparse, human-readable HJSON. Delete it and you lose only
the curation, never the pictures. Copy an album folder and its curation travels
with it. Your files stay yours.

---

## 2. Browsing

**In the tree** (`Tab` to focus it):

| Key | Action |
|-----|--------|
| `j` / `k` or ↓ / ↑ | move the cursor |
| `g` / `G` | jump to top / bottom |
| `l` / → / `Enter` | open an album (or expand a folder) |
| `h` / ← | collapse |
| `Tab` | switch focus to the grid |

**In the album grid** (`Tab` to focus it):

| Key | Action |
|-----|--------|
| `h` `j` `k` `l` or arrows | move the selection |
| `g` / `G` | first / last image |
| `[` / `]` | fewer / more columns (thumbnails resize) |
| `Enter` | open the full **image view** |
| `/` | open the **filter** (see §4) |
| `C` | enter the **culling loupe** (see §5) |

### Image view

`Enter` on a thumbnail opens the image full-pane. `←` / `→` step through the
album, and `Esc` returns to the grid. **Zoom** with `Z` (in) / `z` (out) — a
centre crop-zoom up to 8×; each image starts fit. The **info panel** shows EXIF
(camera, lens, focal length, aperture, shutter, ISO, GPS — read once and cached)
plus curation and, for a plakat-generated image, its recipe (prompt, seed, steps,
model): press `i` to dock it on the **right**, or `I` (Shift-i) to dock it across
the **bottom**.

RAW files (`.cr2`, `.nef`, `.arw`, `.dng`, …) are decoded and demosaiced
in-process, so they thumbnail and display alongside JPEG/PNG/WebP/TIFF without a
separate conversion step.

**Analysis (`H`).** In the image view, press `H` for a side panel with a **luma
histogram** (an 8-row bar chart across the tonal range), the image's **mean**
brightness, **highlight/shadow clipping** percentages (flagged red when a shot is
blown out or crushed), and a **focus score** (variance of the Laplacian — higher =
sharper, useful for picking the in-focus frame from a burst). It updates as you
step through images with `←`/`→`. Press `H` again to hide it.

---

## 3. Curating

Curation is **non-destructive** — it only ever writes `album.hjson`, never your
image files. From the grid (or the image view), acting on the selected image (or
every selected image — see §6):

| Key | Action |
|-----|--------|
| `1`–`5` | set a star rating |
| `0` | clear the rating |
| `f` | toggle the **flag** (a keep/pick marker) |
| `x` | toggle **reject** |
| `c` | cycle the **colour label** (red / yellow / green / blue / purple) |

Each thumbnail shows a small badge with its rating / flag / reject / colour so
the state of the whole album is visible at a glance. Flagged images also get a
**gold cell border** (rejected → dim red) and a `⚑ N` count in the album header, so
your picks stand out. Everything is written atomically (`.album.hjson.tmp` →
rename), so an interrupted write never corrupts your curation.

**Undo / redo.** `u` undoes the last curation change (a rating, tag, flag, reject,
colour — including a bulk one over a selection), `U` redoes it. It's a full history
of your curation for this session (in a regular album; smart/search views edit each
image's own album directly).

### Versioning

Editing is **non-destructive and versioned by design**. A T1 pixel edit (`E`) never
overwrites the original — the pristine file is kept and the visible image is
re-derived from an **edit log**, so `u` / `U` step backward and forward through the
versions and `0` returns to the original. A model edit (`M`) writes a **new file**
linked as a `variant` of the source, so each generation is its own version you can
keep side by side. Nothing you do throws away an earlier state.

### Text metadata

Beyond the one-key states above, each image carries free-text fields you can edit
from the grid or image view — a small prompt opens on the command line, prefilled
with the current value (empty input clears it):

| Key | Field |
|-----|-------|
| `t` | **tags** — comma-separated; feeds the `tag:` filter |
| `e` | **caption** — a short description |
| `N` | **notes** — longer free text |
| `T` | **title** — a display name |

Tags, caption, notes, and title all show in the image-view panel (`i`), alongside
the EXIF and — for plakat-generated images — the generation recipe.

### Sorting

Press `s` in the grid to cycle the album's **sort order**; it's shown in the pane
title (`↕ name-asc`) and persisted in `album.hjson`:

`name-asc → name-desc → date-desc → date-asc → rating-desc → score-desc`

`date-*` sorts by file time, `rating-desc` puts your top picks first, and
`score-desc` orders by the LAION aesthetic score (from `--score` / `rank`). The
cursor stays on the same image across a re-sort.

---

## 4. Filtering

Press `/` in the grid to type a **filter**. The grid narrows to matching images
live as you type; `Esc` clears it. The grammar is space-separated tokens (all
must match):

| Token | Matches |
|-------|---------|
| `rating>=4` `rating>3` `rating=5` | by star rating |
| `unrated` | rating 0 |
| `flag` / `-flag` | flagged / not flagged |
| `rejected` / `-rejected` | rejected / not rejected |
| `ai` | generated by plakat (has a recipe) |
| `scored` | has an aesthetic score |
| `tag:sunset` / `-tag:sunset` | has / lacks a tag |
| any other word | free-text match on the filename |

Example: `rating>=4 -rejected tag:iceland` → your four- and five-star Iceland
keepers.

### Smart albums (library-wide saved searches)

A filter is scoped to one album; a **smart album** is a saved filter evaluated
across your *whole* library. With a filter active in the grid, press `F` and give
it a name — it's saved to the root `folder.hjson` and appears as a ★ entry at the
top of the tree.

Open a smart album (→ / `Enter` on its ★ row) and the grid fills with every
matching image from every album at once — "all my five-star picks", "everything
AI-generated", "everything tagged `portfolio`". You can browse, open the image
view, and **curate**: a rating or flag you set here writes straight back to that
image's own album, so the source stays the source of truth. `D` on a ★ row
deletes the saved search (never the images).

Examples worth saving: `rating>=4 -rejected` (keepers), `ai scored` (generated,
scored), `flag` (your picks across shoots).

### Search by tag (`Ctrl-B t`)

Don't remember which tags you've used? Press `Ctrl-B t` for a **tag browser** — every
tag in the album, most-used first, with counts. Arrow to one and `Enter` filters the
grid to it (`tag:…`). It's the discoverable front-end to the `tag:` filter.

**Auto-tagging AI images.** Anything you `--import` (or that carries a generation
recipe) is auto-tagged from that recipe — an `ai` marker, the model, and a few prompt
keywords — so generated images are searchable the moment they land. To backfill an
existing album, `A` → `g` (offline, no LLM). For descriptive tags on *photos*, use the
LLM autotag (`A` → `t`).

### Metadata search (`?`)

A filter matches exact tokens; **search** ranks by *relevance*. Press `?` (from
the tree or the grid) and type a free-text query — plakat scores every image in
the library against its text metadata (filename, title, caption, notes, tags, and
the generation prompt/model for `--import`ed images) and shows the best matches
first, 🔎 in the pane title.

It's a local, model-free TF-IDF cosine ranker (the same engine `plakat ui`'s
History uses) — instant, no model download, no network. Because it's
meaning-aware rather than substring, `winter mountain` surfaces a caption like
"fresh snow on the peaks" even with no shared word. The result is a live view you
can curate (writes route back to each source album) and narrow further with `/`.

---

## 5. Culling

Pressing `C` in the grid opens the **culling loupe** — a one-image-at-a-time
review mode for going through a shoot quickly:

| Key | Action |
|-----|--------|
| `→` / `Space` | keep and advance |
| `x` | reject and advance |
| `f` | flag and advance |
| `1`–`5` | rate and advance |
| `←` | back |
| `i` | toggle EXIF |
| `Esc` | leave the loupe |

Every decision advances to the next image, so you can rate a hundred frames
without touching the mouse.

---

## 6. Multi-select

In the grid you can act on many images at once:

| Key | Action |
|-----|--------|
| `Space` | toggle the current image's selection |
| `Ctrl-a` | select all (in the current filtered view) |
| `Ctrl-d` | clear the selection |
| `Ctrl-i` | invert the selection |

With a selection active, a curation key (rating / flag / reject / colour) applies
to **every** selected image at once.

---

## 7. Managing folders and albums

From the tree:

| Key | Prompt |
|-----|--------|
| `n` | **new folder** under the cursor |
| `a` | **new album** under the cursor |
| `R` | **rename** the folder/album |
| `D` | **delete** (asks `y/N` first) |

Each opens on the command line at the bottom; type and `Enter`, or `Esc` to
cancel.

---

## 8. Pixel editing (`E`)

Press `E` (in the grid or image view) to open the **edit menu** — quick,
**non-destructive** pixel edits on the cursor image:

| Key | Edit |
|-----|------|
| `r` / `R` | rotate 90° clockwise / counter-clockwise |
| `t` | rotate 180° |
| `h` / `v` | flip horizontal / vertical |
| `g` | grayscale |
| `s` | crop to a centered 1:1 square |
| `-` / `+` | brightness down / up |
| `<` / `>` | contrast down / up |
| `u` | undo the last edit |
| `0` | revert — discard all edits, restore the original |
| `Esc` | close the menu |

Edits chain (rotate, then brighten, then crop) and the thumbnail/image update
live. Nothing is destroyed: the **pristine original** is copied once into a hidden
`.plakat_edits/` folder, and the visible file is re-derived from it by replaying
the edit list — so `u` and `0` always get you back, and ten edits cost one
re-encode from the original, not ten. The edit list is stored in the image's
`album.hjson` record (`edits`), so it persists across sessions and shows in the
image-view panel (`i`).

## 9. ML editing (`M`)

The T1 edits above are instant, geometric. For **model-powered** edits, press `M`
to open the ML menu on the cursor image:

| Key | Edit | Needs |
|-----|------|-------|
| `u` | ML upscale ×4 (Real-ESRGAN) | — |
| `i` | img2img transform | a prompt |
| `l` | relight (IC-Light) | a lighting prompt |

These run an actual pipeline (the same engine as `plakat generate` / `upscale` /
`relight`), so they take a while and may download a model on first use. The
manager **pauses** while a job runs — the alternate screen drops so you see the
familiar plakat download/denoise progress bars, then it resumes automatically.

The result is a **new** image (your source is never touched): it lands in the same
album as `<name>_upscale.png` / `_img2img.png` / `_relight.png`, is linked as a
`variant` of the source, and the cursor jumps to it. Generate → edit → keep, all
inside the library.

> First-run notes: img2img/relight load SDXL (heavy — the manager runs one job at a
> time with everything else freed). Interactive crop/mask painting to steer inpaint
> is a later step.

## 10. AI metadata — autotag & describe (`A`)

Press `A` to open the AI menu and have your **configured LLM** look at the cursor
image:

| Key | Result |
|-----|--------|
| `t` | **autotag** — 5–12 content/style tags merged into the image's `tags` |
| `d` | **describe** — a one-sentence `caption` |

This makes an otherwise unlabeled library *searchable*: run autotag on a shoot,
then `/ tag:...` or `?` metadata-search finds images by what's actually in them.
Tags are merged (never replace what you've set), captions overwrite. It's a quick
network call — the status shows `querying <provider>…` for a beat, then the record
updates.

It routes to whichever vision-capable LLM you have configured: **Gemini**
(`GEMINI_API_KEY`) or an **OpenAI-compatible** endpoint like DeepSeek
(`DEEPSEEK_API_KEY`), preferring Gemini when both are set. The local (text-only)
LLM has no vision model, so it reports that and asks you to configure one.

### Natural-language commands (`:`)

Press `:` and type a command in plain language — optionally a **pipeline** of steps
joined by `then`:

```
find rating>=4 then upscale then export to ~/best 2000
all photos then autotag
take flag then tag as keeper then rate 5
```

A command is an optional **selector** (`find`/`take` a pattern — the same filter
grammar as `/`, or `all` / `selected`) followed by actions run in order: `rate`,
`flag`, `reject`, `tag`, `autotag`, `describe`, `upscale`, `img2img …`,
`relight …`, `grayscale`/`rotate`/`crop`, `export to …`, `rename …`, `sort by …`,
`dedup`, `stack`, `smart album …`.

**Security model.** The command pane acts on the current album's images and their
metadata. The one outward operation is `export`, which *copies* images to a
destination you name — **create-only**: it never overwrites, never reads, and never
touches anything outside the album. There is no vocabulary for *reading* an external
file or *running a command*, so the LLM can't be talked into either.

Common phrasings are parsed **locally** (no network); anything else is handed to
your configured LLM, **grounded with the album's metadata**, which returns a plan
from that same fixed vocabulary — it routes your intent, it never runs arbitrary
code. You always get a `[y/N]` confirmation with a summary before anything runs.
Batch model operations (upscale-all, autotag-all) queue up and run one at a time.

### Visual search (`V`)

Where metadata search (`?`) matches the *words* on an image, **visual search**
matches the *pixels*. Press `V`, type a description, and plakat ranks your whole
library by **CLIP** similarity — the image and your text are embedded into the
same space and scored by cosine, so "golden hour on a mountain lake" finds the
shot whether or not anyone ever tagged it.

The first search loads the CLIP model and embeds every image (the UI pauses and
shows progress on the terminal, like an ML edit); embeddings are **cached to disk**
per album (a hidden `.plakat_clip` file), so refining the query — or searching
again next session — only re-embeds images that are new or changed. Results open
as a relevance-ranked view you can curate and narrow with `/`. No API key needed —
it runs locally.

Together, `?` (words) and `V` (pixels) are the two halves of semantic search.

### Lookalike (`Ctrl-B l`)

To find images that *look like the one under the cursor* — other frames from a
burst, near-duplicates, the same scene — press `Ctrl-B l`. It ranks the whole
library by **perceptual-hash similarity** to the current image, nearest first. This
one is fully **offline** (no model, no network, no API key) — it's the quick,
always-available cousin of CLIP visual search, matching visual similarity rather
than deep semantics.

## 11. Browse — compare, duplicates, rename & export

**Compare side-by-side (`=`).** Select 2–4 images (`Space`) and press `=` to see
them large and side by side — the way to pick the keeper from a burst. `←`/`→`
moves the focus (cyan border); `1`–`5`, `f`, `x` rate / flag / reject the focused
image without leaving the comparison; `Esc` returns to the grid. With nothing
selected, `=` compares the cursor image and its next few neighbours.

**Batch rename (`r`).** Rename the selection (or the whole view) with a pattern:
type it with a run of `#` where the sequence number goes.

```
trip_###        → trip_001.jpg, trip_002.png, trip_003.jpg …
```

Extensions are preserved, numbering follows the current order, and the rename is
safe (staged so intra-set swaps can't clobber) — each image's ratings/tags and
its edit backup move with it. Album-local (open a real album, not a smart view).

**Find near-duplicates (`#`).** Shot the same thing five times? Press `#` in the
grid to hash every image in the current view (a perceptual dHash — robust to
scaling, mild exposure/colour shifts, and re-compression) and group the ones that
match. The best image in each group (highest rating, then aesthetic score) is
kept; every other member is tagged `dup`, and the view narrows to `tag:dup` so you
can review them — press `C` to cull, `x` to reject, or clear the tag (`t`) on
keepers. Nothing is deleted; duplicates are only *marked*.

**Export (`X`).** Press `X` to copy the current selection (or the whole view, if
nothing's selected) out of the library. Type a destination directory; add a
trailing number to cap the longer side in pixels:

```
~/Desktop/share            # full-size copies
~/Desktop/share 1600       # downscaled so the longer side ≤ 1600 px
```

Exports are copies — your originals stay in the album — and name collisions in the
destination are suffixed `-2`, `-3`, ….

**Stacking (`S`).** When you edit an image (T1 `E` or ML `M`), the result is linked
as a *variant* of its source. Press `S` to **stack** — the derivatives collapse
out of the grid and their base shows a `⧉N` badge (N variants). Press `S` again to
expand. A tidy way to keep a shoot readable when it's full of edits and upscales.

**Timeline (`@`).** Press `@` for a timeline of capture months. The view
date-sorts and a popup lists each `YYYY-MM` bucket with its count; `↑`/`↓` picks a
month and `Enter` jumps the grid straight to it — fast travel through a big
library. (Buckets read the EXIF capture date; images without one group as
`undated`.)

## 12. Closing the loop — `--import`

The point of the manager is that **generation flows into it**. Every
image-producing command takes `--import <album>`:

```bash
plakat generate "a red fox in snow, golden hour" \
    --model sdxl --count 4 --keep-best 2 \
    --import ~/Pictures/Generated/Foxes
```

This generates four images, keeps the two best (LAION aesthetic scorer), and
**lands those two in the album** — copied in with their full recipe (prompt,
seed, steps, model, LoRAs) recorded in `album.hjson`. If `plakat photos` is open
on that album in another terminal, the filesystem watcher folds the new images
into the grid **live**, already curated with their score and parameters.

`--import` is available on `generate`, `upscale`, `portrait`, `multiperson`,
`img2img`, `outpaint`, `stylize`, and `relight`. Add `--import-move` to move the
output into the album instead of copying it (leaving only the album copy).

```bash
# Upscale a keeper and file the result in the same album
plakat upscale --in fox.png --out fox-4k.png --scale 4 \
    --import ~/Pictures/Generated/Foxes
```

A generated image imported this way shows up under the `ai` filter, sorts by its
aesthetic `score`, and carries its recipe in the EXIF overlay — the same recipe
`plakat clone` can turn back into a runnable command.

---

## 13. A quick workflow

1. `plakat photos ~/Pictures` — open your library.
2. `Tab` to the tree, arrow to a shoot, `Enter` to open the album.
3. `C` to cull: fly through with `→` / `x` / `1`–`5`.
4. `Esc`, then `/ rating>=4 -rejected` to see just the keepers.
5. `Ctrl-a`, then `c` to colour-label the whole set.
6. In another terminal, `plakat generate "…" --import <that album>` and watch
   the result land curated in the grid.

That's the loop: **browse → curate → generate → back into the library.**

---

## See also

- [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md) — the generation flags `--import` files away.
- [`RANK_TUTORIAL.md`](RANK_TUTORIAL.md) — the aesthetic scorer behind `scored` / `--keep-best`.
- [`UI_TUTORIAL.md`](UI_TUTORIAL.md) — `plakat ui`, the conversational generation TUI.
- [`METADATA_TUTORIAL.md`](METADATA_TUTORIAL.md) — the recipe sidecar `--import` records.
