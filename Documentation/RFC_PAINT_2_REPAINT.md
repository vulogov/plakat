# RFC PAINT-2 — `plakat paint repaint`: the same painting in another medium

Status: **PROPOSED** (plan only; nothing built). Depends on RFC PAINT-1 (the stroke score).

## 1. Summary

A stroke score (`.strokes`) already rebuilds its painting byte-exact with no source image, no models and no
plan: `plakat paint replay`. `repaint` takes that same file and a **medium** and replicates the painting
in that medium:

```
plakat paint repaint night.strokes --medium oil        -o night_oil.png  [--strokes night_oil.strokes]
plakat paint repaint night.strokes --medium watercolour -o night_wc.png   [--plan plan_watercolor.hjson]
```

The DRAWING is kept — every stroke's path, width, order and stage, and the colour it was aiming at — and
the MEDIUM is swapped: the canvas model (opaque paint vs the transmittance film), the brush (how it loads,
lays, dries, skips), the stage crossings (drying, the fluid stage), and the finish (impasto/ridges/weave vs
granulation/dry-shift). The output is a picture **and a new score in the target medium**, which replays
exactly like any other.

## 2. Why it is not just `replay` with a different header

A score is medium-specific in three places:

1. **The mix is a charge for THAT medium.** A watercolour stroke's `mix` is the residual glaze that takes
   the canvas from what is there to the target — rendered on bare white it is not the colour the painter
   meant. An oil stroke's mix is the opaque colour itself. A watercolour score replayed opaquely is pale;
   an oil score replayed as glazes is black.
2. **Wetness, pressure, deposit rate, film cap, hold** are the brush of that medium. A wash brush's
   `wet=0.62 cap=1 hold=1` means nothing to a bristle brush.
3. **The crossings and the finish** live in the header (`bleed dry flow granulate impasto ridges skip
   weave …`) and are the medium's physics, not the drawing's.

So `repaint` = keep (paths, widths, taper, order, stages, intent colour) · re-derive (charge, brush, crossings,
finish) from the target medium's profile · run the target canvas.

## 3. Design

### 3.1 The intent colour — one new per-stroke field (P0)

Record, for every stroke in every medium, the colour the stroke was aiming at: `tgt=r,g,b` (the seed's
target in the pass's reference, sRGB). Optional, ignored by `replay` (so replay stays byte-exact), cheap
(~12 bytes a stroke). This is what makes repaint exact in **intent**: the target medium re-solves `tgt`
into its own charge with its own mixing model.

Old scores without `tgt`: fall back to the mix rendered on white in the SOURCE medium's model (right for
oil, approximate for watercolour glazes — documented as such).

### 3.2 Palette

`palette=image` scores carry the pigments as `P name r g b` masstone lines. Both media can use the same
masstones: oil mixes them by Kubelka–Munk at an opaque charge, watercolour by transmittance densities. A
named palette (`zorn`, `earth`) is looked up as now. `--palette` may override.

### 3.3 Medium profile → stroke translation

Per stroke, by its stage and radius (both recorded; `stages=` gives the pass order):

| from the score | oil-direct | watercolour (wash recipe) |
|---|---|---|
| path, w0/w1, taper, order | kept | kept |
| colour | `tgt` → KM charge, opaque | `tgt` → transmittance densities aimed at the colour ON WHAT IS THERE (the glaze solve the recipe uses) |
| wetness/pressure | profile per stage (block-in / restate / detail) | broad stages = wash brush (`cap hold flat kd`), fine = wet-on-dry, dry over texture (`skip`) |
| lights | painted | **reserved**: a stroke whose `tgt` is above the reserve luma and off the subject is dropped (the paper is the light) |
| crossings | dry between passes | dry + the fluid stage (`flow`), tide lines, grain |
| finish | impasto, ridges, weave, sheen | granulate, dry-shift, chroma |
| plan dials | `impasto_map`, `ridges`, `stroke_width/length` apply | `fine_lines`, `skip`, `granulate` apply |

The subject/face masks are not in the score. Where a rule needs them (watercolour's reserve, the face
economy) P1 uses a proxy from the score itself: the fine stages' seed density (where the painter worked
finest IS the subject). P3 may record a coarse `subject` mask line in the header.

### 3.4 Running it

```
score.parse → palette → for stroke in order: translate(stroke, profile) → rasterize on the target Canvas
            → at stage boundaries: the target medium's crossings
            → finish → image; and the translated strokes → the new score (header = target medium)
```

Deterministic; the new score replays to the same image (tested). Same-medium repaint (`oil → oil`) must
equal `replay` byte for byte (the identity test), which pins the translation layer.

## 4. Phases

- **P0 — intent colour.** `tgt=` recorded by every painter path; parse/write; replay ignores it. Guard 0
  (the image does not change). One commit.
- **P1 — the verb, same medium = replay.** `plakat paint repaint` with `--medium`, the translation layer,
  the identity test, new-score replay test. Oil ↔ watercolour on the night picture judged by the user
  (crops to `quality/REPAINT_*`).
- **P2 — the cross-medium rules.** wc→oil: opaque charges from `tgt`, block-in coverage from the broad
  stages. oil→wc: the reserve, the film cap, flow/grain/skip from the recipe, the fine-stage economy.
  Line media (pen-ink, ink-wash) as targets.
- **P3 — the rest.** `--plan` mark dials; `--size` (structure-preserving rescale, as replay); a header
  `subject` mask; `plakat ui` hook.

## 5. Acceptance

- same-medium repaint = replay (AE 0) on all 16 media;
- the new score replays its own picture (AE 0);
- cross-medium: the drawing is the same drawing (edge-map DSSIM vs the source-medium render under a bound),
  faces recognizable (the user judges), no source image needed;
- the default `paint` path byte-identical (guard 0); one heavy job at a time on the bench.

## 6. Non-goals

- Inventing strokes the score does not have (a watercolour's reserved paper is a dropped stroke, not a new
  one); re-reading the source image; "style transfer".
