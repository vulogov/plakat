# `plakat paint --medium durer` — copperplate engraving

`durer` renders the armature as an **engraving in the manner of Albrecht Dürer**: pure black line on white
paper, tone built by the *weight* and *density* of cut lines, contours traced as continuous strokes. It is a
drawing medium (the density mark model): no brush, no pigment mixing, no washes — every mark is a line the
painter records and replays byte-exact like any stroke.

```
plakat paint from illustration.png --plan auto --medium durer --seed 42
```

## What it does

The engraver's plate is planned from the armature in three layers, all derived from the image's own values —
nothing is tuned per scene:

1. **Contours.** An edge map of the perceptual value (Sobel → non-maximum suppression → hysteresis against the
   image's own gradient distribution; the medium's `contour` strength decides how much of it is kept) is
   linked into chains and cut as continuous lines. Short chains are kept: a window bar, an eye, a hand.
   Inside a detected head the contours are traced from the full-resolution source and kept by the head's own
   edge distribution, so a face's folds are not lost to the foliage behind it.
2. **Tone by line weight.** A first, form-following layer runs over every surface but the highlight; its line
   width carries the tone — a hair in the lights, a full cut in the darks (the burin cuts deeper for a darker
   passage). Line direction follows the local form's isophotes (a trunk, a sleeve, a cheek); flat surfaces
   (a wall, a road, judged by their value spread over a wide window) are ruled with straight parallel lines.
3. **Crossings.** Where even full-weight lines cannot carry the tone, further layers cross the first at the
   classic angles (90°, 45°, 135°, 22.5°, 112.5°). Each crossing starts exactly where the picture is as dark
   as that many crossings would ink the paper — the plate's tone matches the picture's value by construction,
   at any resolution — and the darkest passage still keeps paper between its lines: a burin never fills a
   black. Cuts stop at silhouettes (a jump in tone) and at the edge of the mass they shade.

Line spacing is 1/300 of the sheet's long side, line width 1/1400 (swelling to ~2× in the darks); the stroke
budget is ×3 the plan's (fine lines need count). A detected head is engraved as its own finer pass: half the
spacing, short cuts, tone honest to that geometry, direction following the head's form rather than the skin.

## What it is good for — and what it is not (read this first)

The engraving model reads the **structure** of the armature. It works when that structure is *drawn*:

- **Illustrations, flat-coloured art, line-and-wash, comics, graphic renders** — clean edges and flat fills.
  Contours trace cleanly, tone reads as hatch, faces keep their features because the features *are* edges in
  the source. This is the case the medium was built and judged on (an autumn watercolour illustration of two
  walkers: figures, faces, guitar, trunks all engraved and recognizable).
- **Photographs and photo-real renders** — soft gradients and fine texture (stone, grass, skin, foliage). Here
  the isophote field follows the *texture* rather than the form, cuts wriggle, and a soft-lit face yields no
  feature lines to the edge tracer (its strongest gradients are skin and beard texture, not the eye line). The
  result reads as a busy net with an unrecognizable face. As of this version that limit is real, not a
  setting: no combination of the levers below produced a Dürer face from a photograph.

**Recommendation:** use `durer` on illustrations and graphic sources. For a photograph, paint it first into a
graphic register and engrave that — e.g. `--medium pen-ink` or `--medium black-pencil` (both keep faces),
or run `plakat naturalize`/`plakat paint --medium gouache` and feed the *result* to `durer` — or accept that
the face will be a mass and let the composition carry the plate.

## Levers

| flag | effect |
|---|---|
| `--contour F` (medium default 0.55) | how much of the edge map becomes drawn line: lower = only the composition's main lines, higher = every fold and window bar |
| `--budget N` (plan ×3 by default) | the cap on lines; when the hatch would exceed it the spacing widens to fit, so a small budget gives an open, sparse plate |
| `--seed N` | the deterministic hand: line waver, seed jitter, hatch angles' small rotations |
| `--palette sumi` | forced: the plate is black on paper whatever the picture's colours |

The engraving replays from its stroke score (`plakat paint replay`) exactly like a painting.

## Known limits (this version)

- Photographic sources: see above — busy texture and no facial features.
- Grass, foliage and other fine texture engrave as a uniform net rather than the sparse directional strokes a
  hand would use.
- Lines follow local form with a slight waver; a real burin cuts straighter.
