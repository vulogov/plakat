# Reproducing the loose‑watercolour look — controls & recommendations

Goal: turn a **photo or a photorealistic generation** into a painting with the character of the reference
watercolours — loose luminous washes, wet blooms, **spatter**, granulation, reserved whites, and a crisp,
recognizable subject. This note inventories what `plakat paint` can already do, the controls **added** for this
style, the **recommended recipe**, and the **honest gaps** still worth building.

See also [`PAINT_CONTROLS.md`](PAINT_CONTROLS.md) (every knob) and [`PAINT_EXAMPLES.md`](PAINT_EXAMPLES.md).

---

## The reference characteristics, mapped to controls

Each hallmark of the reference images and the control that produces it:

| # | Watercolour characteristic (in the references) | Control | Status |
|---|-----------------------------------------------|---------|--------|
| 1 | Loose, non‑photographic masses (not a filter over a photo) | `--style impressionist` / `legible` | existing |
| 2 | Wet‑in‑wet **blooms**, soft colour fusion where washes meet | `--bleed` | existing |
| 3 | Transparent **luminosity** — the paper glows through | `--opacity` (low, ~0.5–0.6) | existing |
| 4 | **Reserved whites** — light shafts, sky sparkle, snow gaps | `--reserve` (raise to close stray holes) | existing |
| 5 | **Granulation** — pigment settling into rough‑paper tooth | `--granulate` | existing |
| 6 | **Spatter / droplets** — flicked spray (streets, snow, texture) | **`--splatter`** | **NEW** |
| 7 | **Edge pooling** — the darker pigment ring a wash dries into | **`--edge-pool`** | **NEW** |
| 8 | **Wet‑on‑dry hard edges** on the focal subject vs soft ground | `--define` + `--preserve-face` / `--focus-detail` | existing |
| 9 | Loose‑wash body **+ a crisp, recognizable subject/face** | `--preserve-face` (detected) / `--focus-detail` | existing |
| 10 | **Variegated** warm/cool washes, pigment separation | `--palette image` + `--chroma` | existing |
| 11 | **Drying value shift** — washes dry a touch lighter | `--dry-shift` (+) | existing |
| 12 | **Line‑and‑wash** ink accents (trams, architecture) | `--contour` | existing |
| 13 | Thin background, worked‑up subject (not over‑painted flat) | `--saliency` + a moderate `--budget` | existing |
| 14 | Vivid, saturated passages (sunset, stained glass) | `--chroma` (>1) | existing |
| 15 | **Deckled paper edge / torn‑paper vignette** (fade to white paper at the border) | **`--paper-edge`** | **NEW** |

**Three controls were missing for this style — all added:**

### `--splatter <0..1>` (HJSON `splatter:`)
Flicks fine pigment **droplets** across the painting — the single most recognizable mark in every reference and
the one plakat could not make. Most droplets are tiny dark spots in the palette's darkest pigment; a few are
coarser blobs; and on a surface‑white medium a fraction **lift to the paper** for the bright speckle of
spray/snow/sparkle. Droplets are recorded as strokes, so a `paint replay` reproduces them exactly. Laid before
the wet‑bleed, so a few spots soften into little blooms. Start ~`0.4–0.6`.

### `--edge-pool <0..1>` (HJSON `edge_pool:`)
Darkens pigment where a wash meets a **hard boundary** — the pigment ring a real watercolour wash dries into (the
"cauliflower"/edge‑bloom). An output‑stage effect on the pigment field (recorded in the score header, so replay
matches). Gives washes a settled, hand‑painted rim instead of a flat fill. Start ~`0.3–0.5`.

### `--paper-edge <0..1>` (HJSON `paper_edge:`)
Fades the painting to **bare paper** at the borders with an irregular **deckled** edge — the torn‑paper vignette
every reference portrait sits in. An output‑stage effect (recorded in the header). It is the single biggest thing
that makes a repaint read as *watercolour on paper* rather than a full‑bleed digital frame. Start ~`0.5–0.7` for
a portrait.

### Finish grade — `--contrast` / `--warmth` / `--clarity`
A small, **painting‑safe** tonal grade (the safe subset of a `naturalize` pass), applied at output and **recorded
in the score** so replay stays exact. Use these instead of a separate pixel‑filter pass:
- `--contrast <0.5..2>` — tonal punch (S‑curve). ~`1.15–1.3` gives washes the depth of the references.
- `--warmth <−1..1>` — white balance; + for the amber sunlight of the boat/couple scenes, − to cool a portrait.
- `--clarity <0..1>` — gentle *local* contrast (midtone separation), **not** edge sharpening. Keep it low (~`0.3`).
  Deliberately no "sharpen/details" knob: sharpening re‑introduces the photographic micro‑detail the painting
  discarded (this repo's proven regression). Saturation is already `--chroma`. Heavy grading/LUT → `plakat naturalize`.

---

## Portrait references — what makes them read as watercolour portraits

The four portrait references (kneading woman, man on the boat, freckled face, couple at the table) share a
recipe that is now fully reachable:

- **A crisp, tonally‑modelled face** floating in **looser washes** — the face reads (eyes, brows, lips, skin
  light/shadow) while the clothing and background dissolve. → `--preserve-face` over a loose base
  (`--style impressionist`).
- **Large reserved‑white clothing** with only shadow washes → low `--opacity` + high `--reserve`.
- **Line‑and‑wash** — dark ink accents on hair strands, folds, the boat, the jar → `--contour 0.3–0.5`.
- **Spatter** — as spray, and in the freckled portrait literally as **freckles/skin texture** → `--splatter`.
- **Wet edge rings and blooms** in the washes → `--edge-pool` + `--bleed`.
- **The torn‑paper deckled border** framing every one → `--paper-edge`.
- **Restrained, harmonious palette** (skin, ochres, blue‑blacks, cool shadows) → `--palette image` + `--chroma`.

The remaining distance to these is **face refinement** — soft, correct tonal modelling of the face — which is
governed by the armature/reference quality and the detail‑tier brushwork, i.e. tuning (`--preserve-face` strength,
`--budget`, `--define`), not a missing control.

---

## The most important switch: paint from an ARMATURE, not the photo

Before any surface tuning: a photo painted at full resolution gets **traced** (beard → scribble, skin → mush) —
that is a filter, not a painting. Paint from a coarse **armature** so the brushwork invents washes, and give the
face a finer armature so it stays crisp. This is the single biggest quality lever, and `--plan auto` sets it for
you. See **[PAINT_PLAN.md](PAINT_PLAN.md)**.

```bash
# Let the art director choose the structure (armature, focal face, value-key, reserve, budget):
plakat paint from photo.png --plan auto \
    --preserve-face 0.5 --bleed 0.6 --edge-pool 0.35 --granulate 0.12 \
    --dry-shift 0.08 --splatter 0.4 --paper-edge 0.55 --contrast 1.2 --warmth 0.15

# Or set the structure by hand: coarse body armature + fine face armature.
plakat paint from photo.png --medium watercolour --palette image --style impressionist \
    --armature 64 --armature-face 200 --value-key 0.7 ...
```

## Recommended recipe (photo → loose watercolour portrait)

```bash
plakat paint from photo.png --medium watercolour --palette image \
    --style impressionist \    # loose masses (legible for a touch more structure)
    --preserve-face 0.55 \     # crisp, recognizable face over the loose wash (detected box)
    --bleed 0.6 \              # wet-in-wet fusion
    --opacity 0.6 \            # transparent, luminous — but firm enough to avoid gap-holes
    --reserve 0.9 \            # only near-white kept as paper (closes stray white holes)
    --granulate 0.15 \         # rough-paper tooth
    --dry-shift 0.08 \         # dries a touch lighter
    --splatter 0.5 \           # ← spatter droplets (spray / snow / sparkle / freckles)
    --edge-pool 0.4 \          # ← wash edge-bloom rings
    --paper-edge 0.6 \         # ← deckled torn-paper border (the watercolour vignette)
    --contour 0.3 \            # ← line-and-wash accents (hair, folds)
    --budget 6000 \            # moderate — a big budget over-works the washes
    -o out.png
```

For a **scene** (street, interior, landscape) instead of a portrait, drop `--preserve-face` and use
`--focus-detail 0.3` (a central focal region) or nothing, and raise `--chroma` to ~1.1–1.2 for the saturated
warm/cool of the references. Add `--contour 0.4` for line‑and‑wash architecture.

HJSON spec equivalent:
```hjson
reference: photo.png
medium: watercolour
palette: image
style: impressionist
preserve_face: 0.55
bleed: 0.6
opacity: 0.6
reserve: 0.9
granulate: 0.15
dry_shift: 0.08
splatter: 0.5
edge_pool: 0.4
paper_edge: 0.6
contour: 0.3
budget: { strokes: 6000 }
```

---

## Honest gaps — recommended future controls

`plakat paint` now covers the dominant marks, but a few reference qualities are only approximated. Ranked by
visual payoff:

1. **True backruns / cauliflowers (`--bloom`)** — `--edge-pool` darkens a wash's outer rim, but a real *backrun*
   is an irregular hard‑edged bloom pushing *outward from a wet drop into damp paint* mid‑wash (very visible in
   the potter's sky and the church). Recommend a `--bloom <0..1>` pass that seeds a few irregular blooms in the
   wettest passages (procedural, weight‑free, recorded).
2. **Dry‑brush / scumble (`--dry-brush`)** — the broken, scratchy strokes on rough paper (thatch, foliage, the
   wet street texture). Partly reachable via bristle `streak`, but no dedicated mode. Recommend a `--dry-brush
   <0..1>` that raises streak + skips tooth‑valleys so the mark breaks up over the paper.
3. **Blooming light glow (`--glow`)** — the soft radial halo around lamps/sun/stained‑glass. Currently only
   approximated by `--bleed`. Recommend a `--glow` that blooms bright reserved highlights outward.
4. **Gravity pooling** — pigment settling toward the lower edge of a wash (directional, not just boundary). A
   refinement of `--edge-pool`.

None of these blocks the target look — the recipe above already gets most of the way — but `--bloom` and
`--dry-brush` are the two that would close the remaining distance to the references.

---

## Notes

- All the controls here are **opt‑in and default‑off**; a plain `paint from` is unchanged.
- Everything is **weight‑free and CPU‑only** except `--preserve-face` (loads the SCRFD face detector) and the
  optional `--crisp`/`--critic` on the spec path.
- The stroke **score** (`out.strokes`) records splatter and edge‑pool, so `paint replay` reproduces the exact
  painting at any resolution.
- Budget is a **ceiling, not a quota** — with the gates on (`--saliency`, `--reserve`, `--preserve-face`) the run
  may report "*N of M strokes performed*"; that is expected, not a bug.
