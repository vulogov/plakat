# `plakat photos` — keymap

The full chord reference for the terminal photo manager. The star of this file is the
**`Ctrl-B` edit-chord map** (image view), which mirrors the Edit palette (`E`) so every edit
has a fast keyboard path.

## The `Ctrl-B` leader

`Ctrl-B` (tmux-style) starts a chord. The next key(s) run a command:

| After `Ctrl-B` | Does | Where |
|----------------|------|-------|
| `h` / `H` | quickhelp: key chords / commands (contextual) | anywhere |
| `t` | tag browser | anywhere |
| `v` | version browser | anywhere |
| `p` | portfolio export | anywhere |
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

### `k` — colour
| Chord | Command |
|-------|---------|
| `Ctrl-B k s` | saturation… |
| `Ctrl-B k v` | vibrance… |
| `Ctrl-B k w` | warmth (warm / cool)… |
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
| `Ctrl-B x v` | vignette… |
| `Ctrl-B x r` | radial dodge / burn… |
| `Ctrl-B x t` / `x b` | graduated ND from top / bottom |
| `Ctrl-B x l` / `x R` | graduated ND from left / right |

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

---
*This file is the reference; the running app's Edit palette (`E`) and `Ctrl-B h`/`H` cards are
generated from the same table, so they always agree.*
