# `plakat paint --medium durer` — copperplate engraving

`durer` cuts the picture as an **engraving in the manner of Albrecht Dürer**: black line on white paper, tone
built by the *weight* of evenly spaced cuts and by *crossing* them, contours drawn as continuous lines of
varying weight. It is a drawing medium (the density mark model): no brush, no pigment mixing, no washes.

```
plakat paint from picture.png --plan auto --medium durer --seed 42
plakat paint from picture.png --plan auto --medium durer --seed 42 --oldpaper true
plakat paint from picture.png --plan auto --medium durer --seed 42 --normals auto --depth auto
plakat paint from hare.png --plan auto --medium durer --seed 42 --normals auto --materials "fur: hare"
plakat paint from portrait.png --plan auto --medium durer --seed 42 --normals auto --follow 0.3
```

## What it is good for

**Machinery, architecture, hard surfaces** — this is where the plate is at its best, with the defaults: the
form-following hatch wraps a tyre, a lamp, a pipe; the cast shadow lies level on the ground; the glints are
clean paper. **Portraits and animals** need to be told more: name the fur (`--materials "fur: …"`), and
hold the hatch back from circling the nose and cheeks like a contour map (`--follow 0.3`). A face is
otherwise cut like any other surface.

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
   open line to crossed shadow. A highlight — a light spot that outshines what surrounds it — is not cut at all: clean paper,
   not a faint tone. A broad light (a lit cheek, a white dress) is no highlight and is modelled lightly.
   The deepest tone is 0.78: in the deepest shadow each line swells to two fifths of its
   spacing, so the black is made by the weight of the lines, not by more of them, and is still a net of
   distinct lines with paper in every mesh — never a filled black. Beside a shadow, a tone already begun is
   carried at single-hatch weight, so between a crossed shadow and the stippled light lies a band of plain
   one-way hatching.
2. **Contours.** An edge map of the value (Sobel, thinned, kept by hysteresis; `--contour` sets how much of
   it) is linked into chains; fragments that end where another begins, running the same way, are joined
   into one line, and each line is pulled taut so it does not step from pixel to pixel. A chain is drawn if its length *and* strength together earn it: a short firm
   edge is a rivet, a long faint one the turn of a cheek, a short faint one is craquelure or grain and is
   left out. The line's width follows the contrast it separates — heavy on a silhouette, a hair inside a
   form — and it is cut like a burin's groove: entering as a needle, swelling, leaving as a needle.
3. **Direction.** Each layer's lines run along the form: the dominant orientation of the picture's edges,
   smoothed over a wide window, so a tyre is cut in rings and a tube along its length. Where no orientation
   dominates, or far inside a flat face, the lines fall back to the engraver's resting diagonal — or, where
   a tone lies alone on open paper (a cast shadow, the ground under the subject), they rest level. With a
   normal map (`--normals`) or a depth map (`--depth`), both below, the lines also turn with the form itself.
4. **Even lines.** A layer is a set of streamlines kept one spacing apart: every new line is seeded beside an
   existing one. A line stops at a drawn contour, at the paper, or beside another line.
5. **Weight, then crossing.** A cut swells with the darkness under it, thins to a point where the tone
   lifts, and wavers a little along its length with the pressure of the hand. A line goes on swelling under its
   crossings all the way into the deepest shadow, so its weight is never flat. The first layer carries a tone
   up to about a quarter inked; a second layer crosses it at 60° where one layer cannot carry the tone,
   leaving lozenges of paper, and a third at 120° for the darks, leaving triangles. There is no fourth: it
   would cut through the meshes and close them. The layers together ink the tone asked for.
6. **Flicks, and a little stipple.** Below a hair's worth of tone a line is not made thinner — it is broken.
   The first layer breaks into short flicks whose length carries the tone, and at its lightest into dots,
   ever sparser, so a form passes into the light through stipple instead of over an edge. The first
   crossing also enters as a narrow band of dots; the second crossing begins as a line. The tone curve has
   a toe — the lightest tones fall off no faster than three tenths of the value — so the dots reach well
   into the lights before the paper is bare.
7. **No rings.** A hatch line may turn through about a right angle over its length, then the burin lifts and
   a new line begins: round a tyre the hatch is arcs, never a closed ring, and no whorl can form.

Line spacing is 1/300 of the sheet's long side (7 px at 2048); a contour is 1 to 6 px wide at that size.

## Surfaces: `--normals`

The picture's edges say where a surface ends, not which way it faces. `--normals` gives the plate a **normal
map** of the picture — for every point, the direction its surface faces — and you do not have to make one:

- `--normals auto` estimates the map from the picture with Marigold-Normals, as a pass of its own before the
  plate is planned, and saves it beside the output as `<out>.normals.png`. The model is fetched the first
  time (about 2 GB). The pass starts from a fixed noise, so a picture always gets the same map. The picture
  is mirrored outward by a margin before it is read — the model takes a picture's border for a wall turning
  away, and with the margin that wall falls outside the picture and is cropped off.
- `--normals map.png` reads a normal-map picture in the usual colours (red = right, green = up, blue = toward
  the viewer) at any resolution: the saved `auto` map, retouched or not, or a pass from a 3D scene.

Two things are read from it, both in the normal's own units, so nothing depends on the picture:

- Where the surface **turns** — its facing swings much more one way than the other: a barrel, a tyre, a limb
  — the cuts run that way, round the form. A sphere, which turns alike every way, and a form finer than a
  hatch could wrap are left to the picture's edges; so is the crease between two faces.
- Where the surface is steady and **tilted** away from the viewer — the ground above all, a wall seen aslant
  — and no edge of the picture leads the line, the cuts lie level with it: a cast shadow is laid in strokes
  that lie on the ground.

- Where the surface neither turns nor tilts — a wall behind a head, the flat face of a machine — the cuts
  are **straight**, on the resting diagonal, whatever the picture's values do there: the drift of a light or
  the grain of a canvas does not bend the hatching of a flat surface.

The normals lead the line; they do not change the tone. Given together with `--depth`, the normals set the
direction and the depth sets what is far.

## Materials: `--materials`

A burin cuts fur and polished metal differently from a plain surface, and the picture's values do not say
which is which (telling them apart by the picture's own statistics was tried and does not work). So the
materials are **named, in words**, and the masks are found for you:

```
--materials "fur: hare"
--materials "glass: headlamp, gauge; metal: engine"
```

`kind: thing, thing; kind: thing`. For every thing named, OWL-ViT finds its instances in the picture and
MobileSAM outlines each — the models `plakat remove --what` uses; nothing new is fetched if you have used
it. The map is saved beside the output as `<out>.materials.png`, in the kinds' colours; look at it, retouch
it if you like, and give it back as `--materials map.png`. Where two masks overlap, the thing written first
keeps the pixel. A thing that is not found is reported and left out.

| kind | how it is cut |
|---|---|
| `fur` (hair, feathers) | cut in **short hairs**, each 5 to 12 line spacings long and its own length, each straying up to ±14° from the coat, lying over one another and not in ranks; they run the way the coat grows — the direction its own strands agree on — and the crossings only lean off it (±17° instead of 60° and 120°), so the hairs lie beside one another even in shadow |
| `glass`, `metal` | polished: light and middle tones open to the paper, the darks stay, a light needs to outshine its surroundings only half as much to be left as a clean glint, and the line is **ruled** — its weight does not waver with the hand, and it is never broken into flicks or dots: into the light it runs on as an unbroken hair |

Whole objects ("hare", "wheel") are found firmly. Parts of a machine ("headlamp", "gauge") are found only
faintly, and some ("engine", "pipe") not at all — the detector is a small one. The report says which.

## Relief: `--depth`

The picture's edges say where a surface ends, not which way it turns. `--depth` gives the plate the
picture's relief as a depth map, and where that map says a surface **turns** — the barrel of a cylinder, the
roll of a tyre, the bulge of a cushion — the cuts run round it, as an engraver's line wraps a limb.

- `--depth auto` estimates the map with Depth-Anything-V2 (the one model in this medium; it runs once,
  before the plate is planned) and saves it beside the output as `<out>.depth.png`.
- `--depth map.png` reads a grey image, white = near, at any resolution: a render pass from a 3D scene, a
  map from another tool, or an `auto` map you have retouched.

The direction is the axis of greatest curvature of the depth — the direction in which the surface normal,
the depth's slope, swings fastest. It is used only where the surface bends firmly and consistently, judged
on an absolute scale so that a picture with almost no relief (a head against a wall) does not have the
estimator's ripples promoted to form. At the step between two objects, and on the flanks the step casts,
the picture's own direction stands. Where a broad flat surface recedes one way — the ground — the lines rest
level with it instead of on the diagonal.

The depth also gives the plate **aerial perspective**: the farther part of the subject is cut up to 30%
lighter — so its lines are finer and it loses its crossings first — and its contours are drawn finer. The
fade is gentle on purpose: a dark far wall still reads darker than a near half-tone.

For the line's direction a normal map is the better input: an estimated depth records the order of objects
far better than the roundness of each. Use `--depth` with `--normals` for the aerial perspective, or alone
when the normals are not wanted.

The plate is still deterministic: the maps are inputs like the picture, the same maps give the same plate,
and a replay needs none of them — every cut is in the score.

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
| `--follow F` (default 1) | how far the hatch follows the form: 1 wraps every surface; 0 is a ruled plate — every line on the diagonal or level on the ground, only the contours bend. A portrait wants about 0.3 |
| `--contour F` (medium default 0.55) | how much of the edge map becomes drawn line: lower keeps only the main lines, higher draws every fold and rivet; 0 = no contours |
| `--budget N` (plan ×3 by default, 360 000 at most) | the cap on cuts; when the hatch would exceed it the spacing widens to fit, so a small budget gives an open plate |
| `--seed N` | where the lines start and where the flicks break |
| `--oldpaper true` | the aged sheet and the plate mark |
| `--normals auto\|FILE` | the picture's surfaces: the cuts wrap the form and lie level on the ground (see above) |
| `--materials SPEC\|FILE` | what things are made of, in words: fur is cut along the coat, glass and metal open their lights (see above) |
| `--depth auto\|FILE` | the picture's relief: the far part is cut lighter; without `--normals` it also turns the cuts |
| `--palette sumi` | forced: the plate is black on paper whatever the picture's colours |

## Known limits

- **Without `--normals` or `--depth`, direction comes from edges alone.** A cylinder is cut along its outline
  rather than around its girth, and a flat face with no nearby edge is always the resting diagonal. With
  `--depth` alone the turn shows on thin rounded parts (a tyre, a pipe) and hardly on broad ones (a head, a
  lamp). With `--normals` it shows on both; small parts (spokes, levers) are finer than a hatch can wrap and
  keep the edge direction.
- **Materials are told apart only where they are named** (`--materials`), and only three kinds: fur, glass,
  metal. Leather, rubber and cloth are cut as plain surfaces. The detector finds whole objects well and
  small parts of a machine poorly.
- **An estimated normal map is a guess**, good on a head, a limb, a tyre, poorer on fine machinery; the
  estimate is made with a mirrored margin so the picture's border casts no wall, but the saved map is
  worth a look, and can be retouched and given back as a file.
- **Low-resolution or heavily compressed sources** show their blocks as steps in the edge of a tone.
- **A face** is cut like any other surface, from the picture at full detail; there is no separate portrait
  pass. Lit skin is mostly paper with flicks.
