# `plakat photos` — keymap

The full chord reference for the terminal photo manager. The star of this file is the
**`Ctrl-B` edit-chord map** (image view), which mirrors the Edit palette (`E`) so every edit
has a fast keyboard path.

Categories after `Ctrl-B` (image view): `g` geometry · `c` crop · `a` adjust · `k` colour ·
`x` effects · `e` edit-stack · `m` manage · **`s` stylize** (looks & filters).

## The `Ctrl-B` leader

`Ctrl-B` (tmux-style) starts a chord. The next key(s) run a command:

| After `Ctrl-B` | Does | Where |
|----------------|------|-------|
| `h` / `H` | quickhelp: key chords / commands (contextual) | anywhere |
| `t` | tag browser | anywhere |
| `v` | version browser | anywhere |
| `p` | portfolio export (watermarked copies + contact sheet) | anywhere |
| `w` | web gallery export (offline HTML + lightbox) | anywhere |
| `l` / `L` | lookalike: perceptual / CLIP | anywhere |
| `g c a k x e m` | **edit category** → then an item key (below) | **image view** |

In the image view, a category key opens a which-key card; press the item key to run the
edit. So an edit chord is **`Ctrl-B` + `<category>` + `<item>`** (e.g. `Ctrl-B a b` =
brightness). The same commands live in the searchable Edit palette (`E`), which shows each
command's chord in its right column.

## Edit chords (image view)

### `g` — geometry
| Chord | Command |
|-------|---------|
| `Ctrl-B g r` | rotate clockwise ⟳ |
| `Ctrl-B g l` | rotate counter-clockwise ⟲ |
| `Ctrl-B g 2` | rotate 180° |
| `Ctrl-B g h` | flip horizontal |
| `Ctrl-B g v` | flip vertical |
| `Ctrl-B g g` | grayscale / desaturate |
| `Ctrl-B g a` | auto-enhance (auto levels + colour) |
| `Ctrl-B g s` | straighten (rotate by degrees) |
| `Ctrl-B g d` | lens distortion (barrel / pincushion)… |
| `Ctrl-B g k` | keystone vertical (fix verticals)… |
| `Ctrl-B g K` | keystone horizontal… |

### `c` — crop
| Chord | Command |
|-------|---------|
| `Ctrl-B c f` | crop free-form (interactive) |
| `Ctrl-B c x` | crop to exact size (WxH px) |
| `Ctrl-B c z` | resize to exact size (WxH or N px) |
| `Ctrl-B c s` | crop to square 1:1 |
| `Ctrl-B c 4` | crop 4:5 (portrait) |
| `Ctrl-B c 5` | crop 5:4 |
| `Ctrl-B c 3` | crop 3:2 (photo) |
| `Ctrl-B c 2` | crop 2:3 (portrait) |
| `Ctrl-B c w` | crop 16:9 (wide) |
| `Ctrl-B c t` | crop 9:16 (tall) |
| `Ctrl-B c b` | border (white frame) · `c o` circle crop (black) |
| (palette) | border/letterbox black · 1:1 · 16:9 · 4:5 blurred · circle crop white |

### `a` — adjust (light / tone)
| Chord | Command |
|-------|---------|
| `Ctrl-B a b` | brightness… |
| `Ctrl-B a c` | contrast… |
| `Ctrl-B a e` | exposure… |
| `Ctrl-B a r` | brilliance… |
| `Ctrl-B a h` | highlights… |
| `Ctrl-B a m` | midrange… |
| `Ctrl-B a s` | shadows… |
| `Ctrl-B a k` | black point… |
| `Ctrl-B a l` | levels (black / white / gamma)… |
| `Ctrl-B a u` | curves (tone curve)… |
| `Ctrl-B a q` | CLAHE (adaptive contrast)… |
| `Ctrl-B a g` | local exposure — graduated (top)… (bottom / saturation / warmth graduated palette-only) |
| `Ctrl-B a i` | local exposure — radial (centre)… (saturation / edges / radial-blur palette-only) |

**Local (masked) adjustments** apply any base adjustment through a **linear gradient** (from an edge)
or **radial** mask — the slider sets the amount; e.g. darken a bright sky, warm the foreground, or
blur the edges for a focus effect.

### `k` — colour
| Chord | Command |
|-------|---------|
| `Ctrl-B k s` | saturation… |
| `Ctrl-B k v` | vibrance… |
| `Ctrl-B k w` | warmth (warm / cool)… |
| `Ctrl-B k k` | Kelvin white balance… |
| `Ctrl-B k a` | auto white balance (gray-world)… |
| `Ctrl-B k e` | gray-point white balance (eyedropper — sample centre) |
| `Ctrl-B k d` | HSL: darken blues (sky) · more selective-colour / HSL bands are palette-only |
| `Ctrl-B k t` | tint (magenta / green)… |
| `Ctrl-B k h` | hue rotate… |
| `Ctrl-B k p` | split-tone… |
| `Ctrl-B k r` / `k R` | selective colour: boost / mute reds |
| `Ctrl-B k g` / `k G` | selective colour: boost / mute greens |
| `Ctrl-B k b` / `k B` | selective colour: boost / mute blues |

### `x` — effects / detail
| Chord | Command |
|-------|---------|
| `Ctrl-B x d` | definition (clarity)… |
| `Ctrl-B x s` | sharpen / soften… |
| `Ctrl-B x n` | noise reduction… |
| `Ctrl-B x g` | film grain… |
| `Ctrl-B x k` | despeckle (median)… |
| `Ctrl-B x z` | dehaze… |
| `Ctrl-B x j` | bilateral denoise (edge-preserving)… |
| `Ctrl-B x A` | chromatic aberration removal (defringe)… |
| `Ctrl-B x m` | tilt-shift / miniature… |
| `Ctrl-B x C` | motion blur (horizontal; vertical/diagonal palette-only)… |
| `Ctrl-B x w` | zoom blur (radial)… · `x q` spin blur (rotational)… |
| `Ctrl-B x N` | film negative → positive · `x B` B&W red filter (more B&W mixes palette-only) |
| `Ctrl-B x y` | enhance sky (auto-mask polarizer)… |
| `Ctrl-B x c` | face polish — AI-detect faces, then 0–100 % skin smoothing… |
| `Ctrl-B x v` | vignette… |
| `Ctrl-B x r` | radial dodge / burn… |
| `Ctrl-B x t` / `x b` | graduated ND from top / bottom |
| `Ctrl-B x l` / `x R` | graduated ND from left / right |
| `Ctrl-B x f` | blur (soft focus)… · `x o` bloom / glow… |
| `Ctrl-B x i` | invert (negative) |
| `Ctrl-B x e` | sepia |
| `Ctrl-B x u` | duotone |
| `Ctrl-B x p` | posterize… |
| `Ctrl-B x a` | solarize… |
| `Ctrl-B x h` | threshold (black & white) |

### `e` — edit stack
| Chord | Command |
|-------|---------|
| `Ctrl-B e h` | edit history (step / trim)… |
| `Ctrl-B e y` | layers — overlay / compose images |
| `Ctrl-B e c` | copy edits (from this image) |
| `Ctrl-B e v` | paste edits (to selection / cursor) |
| `Ctrl-B e s` | save edits as preset… |
| `Ctrl-B e a` | apply preset… |
| `Ctrl-B e u` | undo |
| `Ctrl-B e o` | redo |
| `Ctrl-B e 0` | revert to original |

### `m` — manage
| Chord | Command |
|-------|---------|
| `Ctrl-B m m` | strip metadata (EXIF / GPS) |
| `Ctrl-B m g` | redact GPS only (keep other EXIF) |
| `Ctrl-B m c` | convert format / resize (jpg·png·webp) |
| `Ctrl-B m w` | watermark / caption (burn in text)… |
| `Ctrl-B m u` | apply LUT (.cube colour grade)… |
| `Ctrl-B m f` | find near-duplicates (perceptual hash → tag `dup`) |
| `Ctrl-B m q` | cull soft / badly-exposed (non-AI → reject) |
| `Ctrl-B m o` | move to album… · `m p` copy to album… |
| `Ctrl-B m t` | move to trash (soft-delete) · `m b` browse trash (restore / empty palette-only) |

Also, `*` (tree or grid) = **flatten browse** — show every image beneath a folder / mixed album in
one grid; `:flatten` does the same. `m` (grid) = **geo map** — plot geotagged photos on an offline
world map (arrows pan · `+`/`-` zoom · `0` world · Enter → nearby photos · Esc); `:map` too.
**Reverse-geocode** to `place:<city>` tags via the manage palette / `:geocode` (fetches a place
gazetteer once, then offline).

### `d` — metadata (per-image, stored in the album record)
| Chord | Command |
|-------|---------|
| `Ctrl-B d t` | set title… |
| `Ctrl-B d a` | set author / creator… |
| `Ctrl-B d c` | set copyright… |
| `Ctrl-B d d` | set capture date… |
| `Ctrl-B d g` | set geotag (lat, lon)… |
| `Ctrl-B d e` | set caption… |
| `Ctrl-B d w` | **write metadata → file EXIF** (title/author/©/date/geo · JPEG/PNG · confirms) |

Edited metadata (title / author / copyright / date / geotag) is stored non-destructively in
`album.hjson` and shown in the info panel. It's **filterable** in the filter bar / smart albums:
`iso>3200`, `focal=50`, `camera:canon`, `lens:35`, `date>2023` / `date:2024`, `has-gps` /
`-has-gps`, `author:jane`, `copyright:acme`, `title:sunset` (all read from the cached record).

`d w` **writes it back into the file's own binary EXIF** (in place, JPEG/PNG only) so the metadata
travels with the file to other tools — title→ImageDescription, author→Artist, copyright→Copyright,
date→DateTime/DateTimeOriginal, geotag→GPS IFD, and the image's **tags→XPKeywords**. It confirms
first and never touches the pixels.

### `s` — stylize (looks & filters)
| Chord | Command |
|-------|---------|
| `Ctrl-B s o` | oil paint (style 3) · `s k` pencil sketch · `s g` charcoal · `s t` cartoon |
| `Ctrl-B s w` | watercolour (style 5) · `s e` emboss · `s f` halftone · `s x` pixelate… |
| `Ctrl-B s i` | ink: European · `s j` Japanese sumi-e · `s h` Chinese wash · `s r` Russian icon |
| `Ctrl-B s b` | false colour: thermal (infrared / night-vision palette-only) |
| `Ctrl-B s y` | cross-hatch · `s m` gradient map: warm (cyanotype / fire / teal-orange palette-only) |
| `Ctrl-B s z` | crystallize (Voronoi / low-poly)… |
| `Ctrl-B s v` | look: vintage · `s l` lomo · `s c` cross-process |
| `Ctrl-B s n` | look: noir · `s p` pop-art · `s d` golden hour · `s a` old photo · `s q` daguerreotype |

### `n` — AI create (loads a model; prompt-driven → a new album image)
| Chord | Command |
|-------|---------|
| `Ctrl-B n g` | AI generate (txt2img) — prompt… |
| `Ctrl-B n i` | AI img2img — transform this image with a prompt… |
| `Ctrl-B n p` | AI portrait — this image as the face + a prompt… |
| `Ctrl-B n m` | AI multiperson scene — selected images as people + a scene prompt… |

(Also on the ML menu `M` and as `:` commands — `generate …`, `portrait …`, `scene …`, `img2img …`.)

### `r` — retouch (interactive crosshair pick-mode)
| Chord | Command |
|-------|---------|
| `Ctrl-B r h` | spot heal (remove blemish / dust)… |
| `Ctrl-B r c` | clone stamp (copy a region — pick source then destination)… |
| `Ctrl-B r e` | red-eye removal… |
| `Ctrl-B r d` | dodge (lighten) brush · `r b` burn (darken) brush… |
| `Ctrl-B r p` | perspective rectify (pick 4 corners TL→TR→BR→BL)… |
| `Ctrl-B r x` | **brush mask: lighten** (paint exposure) · `r k` darken |
| `Ctrl-B r s` | **brush mask: saturation** · `r w` warmth · `r u` blur/soften |

In the pick-mode a **crosshair** appears on the image: **arrows** move it (fine), **h/j/k/l** jump,
**`+`/`-`** resize the brush, **Enter** sets the next point (the op applies once all its points are
picked), **Esc** cancels. Each is a normal replayable edit (undo with `u`).

The **brush-mask** ops (`r x/k/s/w/u`) paint a freeform local adjustment: move the crosshair,
**`+`/`-`** the brush size, **Space** to stamp a dab (paint as many as you like — a magenta tint
shows the mask), then **Enter** applies the chosen adjustment through the painted mask (**Esc**
cancels). It's a single replayable `BrushAdjust` edit (the dabs are stored, so it re-applies exactly
on the original) — the Lightroom-style *brush* companion to the *graduated/radial* local adjustments
(`a g`/`a i`).

All filters open the **slider** (strength 0–100%); dial the intensity, `Enter` to apply.

**Oil paint 1–10** and **watercolour 1–10** are palette-only (type "oil" or "water" in the
Edit palette to pick a numbered style). Any Edit-palette command with no chord shown is
palette-only.

The scalar adjustments (`…`) open a live **slider**: `←`/`→` = **fine ±1**, `[`/`]` (or
`-`/`+`, PgUp/PgDn) = the **coarse jump**, `Enter` apply, `Esc` cancel. Curves, levels,
layers, and the history scrubber open their own interactive modes (see `Ctrl-B h` in each).

## Image-view keys (no leader)
| Key | Does |
|-----|------|
| `←` / `→` | previous / next image · `Esc` back to grid |
| `Z` / `z` | zoom in / out |
| `\` | before / after (pristine original vs edited) |
| `i` / `I` | info panel: right / bottom |
| `H` | analysis panel (histogram, RGB, balance, dominant, waveform, RGB parade) |
| `o` | overlay: clipping zebras → focus peaking → off |
| `S` | slideshow ▶ auto-advance · `[` slower · `]` faster · `r` 🔀 shuffle · `k` 🎥 Ken Burns · `S`/`Esc` stop (higher-rated linger longer) |
| `E` / `M` / `A` | edit palette / ML-edit / AI-vision menus |
| `1`–`5` `0` `f` `x` `c` | rate / clear · flag · reject · colour |

## Tree-pane keys
| Key | Does |
|-----|------|
| `↑↓` `PgUp/PgDn` `Home/End` | navigate |
| `→` / `Enter` | reveal children / open album |
| `←` / `h` | collapse folder · up one level |
| `n` · `a`/`+` · `R` · `D`/`-` | new folder · new album · rename dir · delete |
| `/` | filter the tree by name |
| `i` / `I` | album info panel / info editor |
| `t` / `T` | add / edit album tags |
| `e` / `E` | export album / export + convert (★ row: materialize smart album) |
| `r` | regenerate thumbnails |

`Tab` toggles focus between the tree and the grid. `q` quits.

## Shared volumes & multiple instances

The library is safe to keep on a **shared / synced volume** (Dropbox, iCloud Drive, NFS) and to open
in **several `plakat photos` at once**. Curation is stored in `album.hjson` / `folder.hjson`, and
those are written with a **three-way merge**: each save re-reads the current file and overlays only
the records/fields *you* changed, so a colleague (or another window) rating *other* photos is never
clobbered. Changes made elsewhere are **picked up automatically** — a smart album added in one
instance appears in the others, and a yellow **`⟳ others editing`** badge lights up briefly when that
happens. Each record you touch is stamped **"Edited by "** (shown in the info panel `i`); if two
instances change the *same* image at once, the later save keeps yours and warns
`⚠ … also changed elsewhere`. Set `PLAKAT_EDITOR` to control the name that appears in the stamp
(defaults to `user@host`). On a local disk an advisory `flock` additionally serializes writers; on
network volumes the merge alone keeps things consistent.

Each record also keeps a small **edit history** (a few recent "↳ who · when" lines in the info
panel). **`:conflicts`** opens a review pane of same-image conflicts this session — **Enter** jumps to
the image, **`t`** takes the other version, **`c`** clears. A **`👥 N`** badge in the status bar counts
other live instances; **`:who`** lists them (a `.plakat_presence/` heartbeat that ages out on exit or
crash).

---
*This file is the reference; the running app's Edit palette (`E`) and `Ctrl-B h`/`H` cards are
generated from the same table, so they always agree.*
