# `plakat paint --medium durer` — copperplate engraving

`durer` cuts the picture as an **engraving in the manner of Albrecht Dürer**: black line on white paper, tone
built by the *weight* of evenly spaced cuts and by *crossing* them, contours drawn as continuous lines of
varying weight. It is a drawing medium (the density mark model): no brush, no pigment mixing, no washes.

```
plakat paint from picture.png --plan auto --medium durer --seed 42
plakat paint from picture.png --plan auto --medium durer --seed 42 --oldpaper true
```

It follows the deterministic model of `plakat paint`: no weights and no diffusion, only geometry computed
from the picture and the seed. Every cut is a stroke in the score, and `plakat paint replay` reproduces the
plate byte for byte.

## How the plate is planned

The plate is cut from the **picture itself**, not from the simplified armature the painting media use — the
armature has already smoothed away the detail a burin draws.

1. **Tone.** The paper is the picture's light (its 90th-percentile value) and the deepest cut its dark. In
   between, the plate does not copy the picture's key. A surface is shaded against its surroundings, so the
   turn of a form and the shadow in a joint read stronger than the flat between them; and the subject is
   spread over the plate's whole range mostly by the *order* of its values, with its middle value about a
   fifth inked: an engraving is mostly paper. A subject that is dark all over is therefore still cut from
   open line to crossed shadow. The deepest tone is 0.76, and no line is wider than two fifths of its
   spacing, so the deepest shadow is a net of distinct lines with paper in every mesh — never a filled black.
2. **Contours.** An edge map of the value (Sobel, thinned, kept by hysteresis; `--contour` sets how much of
   it) is linked into chains; fragments that end where another begins, running the same way, are joined
   into one line, and each line is pulled taut so it does not step from pixel to pixel. A chain is drawn if its length *and* strength together earn it: a short firm
   edge is a rivet, a long faint one the turn of a cheek, a short faint one is craquelure or grain and is
   left out. The line's width follows the contrast it separates — heavy on a silhouette, a hair inside a
   form — and it is cut like a burin's groove: entering as a needle, swelling, leaving as a needle.
3. **Direction.** Each layer's lines run along the form: the dominant orientation of the picture's edges,
   smoothed over a wide window, so a tyre is cut in rings and a tube along its length. Where no orientation
   dominates, or far inside a flat face, the lines fall back to the engraver's resting diagonal.
4. **Even lines.** A layer is a set of streamlines kept one spacing apart: every new line is seeded beside an
   existing one. A line stops at a drawn contour, at the paper, or beside another line.
5. **Weight, then crossing.** A cut swells with the darkness under it, thins to a point where the tone
   lifts, and wavers a little along its length with the pressure of the hand. The first layer carries a tone
   up to about a third inked; a second layer crosses it at 60° where one layer cannot carry the tone,
   leaving lozenges of paper, and a third at 120° for the darks, leaving triangles. There is no fourth: it
   would cut through the meshes and close them. The layers together ink the tone asked for.
6. **Flicks, and a little stipple.** Below a hair's worth of tone a line is not made thinner — it is broken.
   The first layer breaks into short flicks whose length carries the tone. The first crossing enters as a
   narrow band of dots, so the edge of a shadow is softened; the second crossing begins as a line.

Line spacing is 1/300 of the sheet's long side (7 px at 2048); a contour is 1 to 6 px wide at that size.

## Old paper and the plate mark

`--oldpaper true` (plan and spec key `oldpaper: true`; default `false`) prints the finished picture on an
aged sheet: cream laid paper — the fine ribs of the mould's wires and its wider chain lines — unevenly
yellowed, with foxing, specks and edges darkened by handling. It
works with **every medium** — the picture's white becomes the paper and its black a warm ink. With `durer`
the sheet also carries the **plate mark**: the line of ink a copperplate's edge leaves a little inside the
border, and is inked like a hand-pulled impression: unevenly, the ink breaking on the paper's tooth. There
is no watermark and no monogram. The sheet is a function of the seed and the sheet's size, and is recorded in the score, so a replay
ages the same way.

## Levers

| flag | effect |
|---|---|
| `--contour F` (medium default 0.55) | how much of the edge map becomes drawn line: lower keeps only the main lines, higher draws every fold and rivet; 0 = no contours |
| `--budget N` (plan ×3 by default, 360 000 at most) | the cap on cuts; when the hatch would exceed it the spacing widens to fit, so a small budget gives an open plate |
| `--seed N` | where the lines start and where the flicks break |
| `--oldpaper true` | the aged sheet and the plate mark |
| `--palette sumi` | forced: the plate is black on paper whatever the picture's colours |

## Known limits

- **Direction comes from edges, not from depth.** A cylinder is cut along its outline rather than around its
  girth, and a flat face with no nearby edge is always the resting diagonal. A hand engraver would choose.
- **Materials are not told apart.** Metal, leather, glass and rubber get the same kind of line; only their
  tone and their edges differ.
- **Low-resolution or heavily compressed sources** show their blocks as steps in the edge of a tone.
- **A face** is cut like any other surface, from the picture at full detail; there is no separate portrait
  pass. Lit skin is mostly paper with flicks.
