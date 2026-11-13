# `plakat make-photo` — a picture through the photographic chain of a period and a style

```
plakat make-photo <image> --media daguerreotype -o plate.png
plakat make-photo render.png --media photo --report          # a render made a photograph
plakat make-photo portrait.jpg --media photo --focal 85 --keep-subject   # the 24 → 85 portrait
plakat make-photo street.jpg --media tri-x --make-plan --make-plan-iterate 2
plakat make-photo list | keys
plakat make-photo measure a.png b.png -o table.md
plakat make-photo plan <image> --media kodachrome [--verdict out.png --plan out.plan.hjson]
```

`make-photo` takes a picture — a photograph, a render, a painting — and passes it through a **photographic
chain**: the scene in linear light, a **camera** that may be moved and re-lensed, a **lens**, a **shutter**, an
**emulsion or a sensor**, a **process**, a **carrier**, the **age**, and the **digital** pipeline of the
cameras that have one. The chain is one; a medium (`--media`) is a **recipe** (`assets/photo/<media>.hjson`)
that sets its dials — thirty-two media from the daguerreotype of 1839 to the computational phone, and `photo`,
the modern camera, for the render that should come out as a photograph. Every dial is a key of the plan
(`make-photo keys`), 0 or "none" when the stage is off, and a medium is tuned in its recipe, never in the code.
Everything but the three neural options is deterministic, weight-free and runs on the CPU: the same plan and
seed give the same bytes.

The command does not disguise anything: the output's metadata names the simulation (`plakat make-photo`,
`media:<name>`, `simulated photograph`) and no camera's EXIF is forged.

## The chain

| stage | what it does | the keys |
|---|---|---|
| **the scene** | the sRGB undone to linear light; the depth (Depth‑Anything‑V2), the subject's matte (U²‑Net), the faces and their eyes (SCRFD), the genre and the stuff (CLIP) read once, each optional | — |
| **the camera** | the picture re-projected through its depth from a new point with a new lens; the opened ground filled, the band past the picture's edge cropped; the share invented reported | `focal` `source_focal` `distance` `depth_range` `dolly` `truck` `pedestal` `pan` `tilt` `roll` `shift` `keep_subject` `fill` `fill_cap` |
| **the lens** | the depth of field from the aperture, the format's focal length, the focus and the metric depth (the circle of confusion per pixel); the field's curvature, the diffraction and the resolving power in quadrature; the Petzval's swirl; the anamorphic's standing ovals; the distortion and the lateral aberration; the pictorial soft focus; the flare and the bloom; the vignette; the haze by the depth | `aperture` `focus` `focus_distance` `field_curve` `vignette` `distort` `ca` `fringe` `flare` `soft` `swirl` `anamorphic` `diffraction` `haze` `resolution` |
| **the shutter** | the hand's shake past 1/30; the water, the foliage and the sky smeared past a second (where the stuff was read); the figures ghosted past four seconds — the scene grown into their place, the figure laid half over it | `exposure_time` `shake` `motion` `ghost` |
| **the emulsion / the sensor** | the exposure; the spectral sensitivity (colour through a dye matrix; `pan`, `ortho` — the reds dark and the sky white, `blue` — the wet plate, `infrared` — the leaves white); the characteristic curve (`digital` to a hard clip, `film` with a toe and a shoulder, `plate` with its short scale, `paper`) and its contrast and latitude; the grain — its size from the micrometres on the negative and the format's width in pixels (a 35 mm frame's grain a pixel wide at 4000 px, an 8×10's under a tenth), its clumps a value noise, its weight on the mid-tones; the sensor's photon noise and read noise, then the Bayer mosaic and its demosaic; the halation; the autochrome's starch reseau | `exposure` `iso` `format` `spectral` `mono` `curve` `gamma` `toe` `shoulder` `latitude` `matrix` `saturation` `warm` `cross` `grain` `grain_size` `sensor_noise` `demosaic` `halation` `reseau` |
| **the process** | the development pushed or pulled; the toning of a monochrome print (sepia, selenium, gold, cyanotype, platinum, albumen, salt, the plate's silver, the collodion's cream); the print's black and white; the hand tint | `develop` `tone` `tone_amount` `black` `white` `tint` |
| **the carrier** | the mirror of a direct positive; the mottle of a hand-poured plate and the sheen of its silver in the shadows; the frame — the brass mat's oval, the tintype's chipped edge, the card, the Polaroid's and the Instax's white, the slide's mount | `carrier` `frame` `mirror` `mottle` `sheen` |
| **the age** | the fade (the blacks up, the dyes toward a hue — a C‑print's magenta), the yellowing, the silvering from the edges, the scratches, the dust, the foxing; `age` sets a profile for the dials left at 0 | `age` `fade` `fade_hue` `yellowing` `silvering` `scratches` `dust` `foxing` |
| **the digital** | the on-camera flash by the depth and the red eye on the faces; the HDR flattening; the face smoothing; the phone's portrait mode (only with a face found, as a phone engages it; the matte's hard edge, the hair wrong); the sharpening and its halo; a real JPEG round trip at the quality; the date stamp | `jpeg` `sharpen` `hdr` `face_smooth` `portrait_mode` `date_stamp` `flash` `redeye` `megapixels` |
| **the neural** (`--neural`) | the fill, the refine, the relight — each behind the flag, each disclosed | `refine` `prompt` `neural_px` `relight` |

## The camera: a new point, a new lens

Zoom and the point of view are different things, and a photograph shows the difference: a 24 mm from two
metres and an 85 mm from seven give one size of face and two faces. So the camera has three independent
controls:

- **the lens** (`--focal`, 35 mm-equivalent): alone, a change of focal length from the same point — a crop
  and a magnification, the perspective kept; zooming in is a real magnification (then the lens's resolving
  power), zooming out opens the frame's edge;
- **the move** (`--dolly` toward or back, `--truck` sideways, `--pedestal` up or down, metres) and **the turn**
  (`--pan`, `--tilt`, `--roll`, degrees; `--shift` the architectural rise): the picture's pixels become points
  in space through the depth and the source lens, the camera moves, the points are projected again. A forward
  pass with a z-buffer finds what each new pixel sees; a backward pass takes that pixel's depth back through the
  move to the source and samples it, checking the depth it lands on is the depth it left from — a pixel that
  lands on a nearer thing is a **hole**, the ground the move opened;
- **the tie** (`--keep-subject`): the focal length and the dolly bound by the subject's size — give one and the
  other follows. `--focal 85 --keep-subject` on a wide portrait dollies the camera back until the face is the
  size it was, with the proportions a long lens gives it.

**The scale.** The depth is relative; the move needs metres. The subject's distance is read from a face (the
eyes 63 mm apart through the source lens), else set by the director from the genre (a portrait 1.5 m, a street
12, a landscape 60), else 3 m; `--distance` overrides it. The source lens is the director's guess by the genre
(a head shot 85, a portrait 50, a scene 28) unless `--source-focal` is given. The model's depth is relative
and is stretched over the scene's **depth range** (`depth_range`, the far ground so many times the near): the
director sets it by the genre — a studio portrait 4 (its wall is a few face-distances back, not twenty), a
figure 10, a street 20, a landscape 50 — and the same relative depth makes a shallow room of a portrait and a
deep street of a townscape.

**The holes.** `fill: propagate` (the default) grows the ground into them from behind the thing that hid them:
each hole is a component whose boundary has a near side (the figure's edge) and the rest; the fill advances a
pixel a pass from every boundary pixel a quarter farther than the near side (the figure's edge does not smear
into its background; the wall, the pavement and the sky on the other sides meet in the hole's middle), a
normalised convolution for what is left with the far pixels weighing more, a softening inside the hole. `fill: neural` leaves them to a diffusion model
(`--neural fill`); `fill: none` leaves them black. Past the picture's edge the camera sees nothing the
picture has: with `propagate` the frame is **cropped** to the rows and columns the picture reached (a row or
column over a quarter past the edge is cut, the rest of the band grown into) and brought back to size — a
small truck trims the edge, it does not smear it; `none` shows the band, `neural` leaves it to the model. The
report prints **the share of the frame invented** — what opened behind things, what fell outside the picture,
what no pixel reached (the tears where a near thing's edge pulls away from its ground) — and the critic
scales a move down to `fill_cap`
(8 % by default). The honest limits: from one picture the camera makes small and middling moves (a dolly of a
third of the distance, a turn of ten degrees); it will not walk behind the subject, and the neural fill there
is the model's invention. Hair and glass tear at the layers' edge; a mirror's and water's depth is wrong and
the depth reads them as surfaces.

The camera runs before the lens, so the depth of field is computed from the re-projected depth with the new
lens — the bokeh belongs to the new point of view, not the old.

## The media

`make-photo list` prints the catalogue. Each is a recipe; read it to see what the medium does, change it with
`--set key=value` or a plan file.

| medium | years | the look |
|---|---|---|
| `daguerreotype` | 1839–1860 | the silver plate, laterally reversed, the Petzval's draw and swirl, blind to all but blue, the mirror in the shadows, the brass mat's oval |
| `calotype` | 1841–1855 | the paper negative's fibre, a salt print, a minute's exposure, foxed |
| `ambrotype` | 1854–1865 | the collodion positive on glass, cream lights, the pour's mottle |
| `tintype` | 1856–1900 | collodion on iron, dark and warm, the plate clipped |
| `albumen` | 1855–1895 | the cabinet card: purple-brown on cream, mounted, faded |
| `cyanotype` | 1842– | Prussian blue on paper |
| `platinum` | 1880–1920 | the long soft scale, warm greys, no shoulder |
| `ortho-press` | 1890–1925 | the press plate: red dark, sky white, contrasty, dusty |
| `pictorialist` | 1890–1915 | the soft-focus lens, the glow, the sepia gum print |
| `autochrome` | 1907–1935 | the starch reseau's pointillist colour, muted and warm |
| `box-brownie` | 1900–1940 | the snapshot: soft corners, a vignette, a small sepia print |
| `kodachrome` | 1936–2009 | saturated reds and blues, clean dark shadows |
| `agfacolor` | 1936–1970 | cooler, softer, the greens toward olive |
| `ektachrome` | 1946–2012 | the cool clean slide, cyan in the shadows |
| `tri-x` | 1954– | 400 pushed to 1600: the grain a fabric, blocked blacks |
| `hp5` | 1976– | the longer scale, moderate grain |
| `large-format` | 1930– | 4×5: stopped down, diffraction-sharp, the verticals standing (`--shift`) |
| `contact-8x10` | 1900– | the grainless contact print |
| `polaroid-sx70` | 1972– | pastel, warm, the dyes migrating, the white frame |
| `polaroid-600` | 1981– | the flash in every face, cyan shadows, red eyes |
| `c-print-70s` | 1970s | the faded chromogenic print leaning magenta |
| `holga` | 1982– | the toy camera: a heavy vignette, cross-processed colour |
| `point-and-shoot` | 1985–2005 | the compact's flash, red eyes, the orange date |
| `disposable` | 1986–2010 | fixed focus, an 800 film, the flash and the black beyond |
| `early-digital` | 1999–2005 | 1.3 Mpx, the purple fringe, the hard clip, JPEG blocks |
| `dslr-2008` | 2006–2012 | APS-C at 800, the kit zoom, oversharpened JPEG |
| `phone-2016` | 2014–2018 | everything sharp, the shadows smeared, the early HDR |
| `phone-2023` | 2020– | the HDR flattening, the faces smoothed, the portrait mode's cut-out bokeh |
| `cinema-anamorphic` | 1953– | oval bokeh, the blue flare, halation, a letterbox grade |
| `instax` | 1998– | pale pastel, the flash always on |
| `infrared` | 1930s– | the leaves white, the sky black, the glow |
| `photo` | 2020s | the modern camera: the sensor's noise and demosaic, f/5.6 (a portrait opens it), the camera's JPEG |

## `--media photo`: a render made a photograph

What a render lacks is physical and enumerable: a noise that grows with the signal and has the demosaic's
correlation; a plane of focus and a lens softer at the edge; highlights that clip or roll off; a vignette and a
lateral aberration; the sensor pipeline's sharpening and JPEG. `photo` runs the whole chain of a modern camera
with those dials at a camera's values — for a render that is already photographic, a subtle, measurable
change (`make-photo measure` shows it). A drawing, a watercolour, a painting need `--neural refine 0.6–0.75`
first: the physics cannot turn brushwork into a scene. The content's errors (hands, geometry, an impossible light) are not the
chain's business — `plakat naturalize --geometry/--anatomy` has them. The goal is physical plausibility, not
the defeat of a detector, and the output says what it is.

`make-photo measure <picture>` prints the **camera statistics** of any picture — the noise in the shadows and
in the lights and its lag-1 correlation, the sharpness's variation over the frame and at the edge, the clipped
and black shares, the vignette, the fringes at the corners against the centre (the signed mean radial offset
of the blue image against the red at the strong edges: an aberration pulls one way and keeps its sign, the
noise's offsets cancel), the JPEG grid — each with what it usually means. Several pictures give a table. The readings are not a judgement: a studio shot at f/16 is
sharp everywhere, a scanned plate has no JPEG grid.

## The director and the critic

The director composes the plan for THIS picture as THIS medium on every run — the picture's facts, the
medium's recipe, the rules' adjustments, and the command line's keys over them; `--make-plan` writes it beside
the output (`<out>.plan.hjson`), three sections with every line's reason, and `--plan <file>` makes the picture
by a plan instead. The rules, each universal:

| the fact | the move |
|---|---|
| no face | the distance by the genre (a portrait 1.5 m … a landscape 60 m) |
| the genre | the source lens (a head shot 85, a portrait 50, a scene 28, an animal 135); the depth range (a portrait 4, a figure 10, a street 20, a landscape 50) |
| a portrait, a figure, a still life, an animal, on a medium whose recipe ties the subject (`keep_subject: 1`, the studio media) | the lens by the genre (85, 50, 90, 135), the camera dollied to keep the subject's size |
| a face / a subject / a scene | the focus on the eyes / the subject / the far ground |
| a scene | the aperture stopped down, ×2 and at least f/8 |
| a portrait (the face over 8 % of the frame) | the aperture from the face's depth: its near and far (the model's depth inside the face's box over the depth range, at the distance the eyes give) must sit inside the depth of field, the circle of confusion 0.03 mm scaled to the format — N = f²·Δz / (2·c·z²) on 35 mm, then scaled by the format's crop factor (a plate's lens is as many times longer for the same framing and its depth of field as many times thinner — the plate's portrait was stopped to f/32–f/64), between f/2 (a phone's f/1.6) and f/16 on 35 mm, f/64 on the largest plate; a tight head shot asks the studio's f/16, a face in a street opens to f/2.8; without a depth inside the face, the lens opened ×0.6 down to f/2 |
| a portrait (the face over 8 % of the frame) on an exposure of a second or more | the sitter braced in the head clamp: `motion` and `ghost` to 0 |
| a nocturne (key < 0.3) | the exposure time ×8 on an emulsion that has one, the figures ghosted on a street; the ISO ×4 on a sensor; the lamps' halation and flare up |
| a high key (> 0.72) | the exposure −0.3 |
| water and an open shutter | the water smoothed |
| a figure or a street on a 2 s exposure | the people ghosted (a sitter is braced, a street is not) |
| the far ground on a scene | the haze deepened |
| the sky on an orthochromatic or blue emulsion | the exposure held −0.3 (the sky goes white anyway) |
| a face and a flash | the red eye |

`--make-plan-iterate N`: make, measure, amend, make again — N more times at most, stopping when the critic
moves nothing; the intermediate pictures kept as `<stem>.i<k>.<ext>`, every verdict appended to the one plan.
The critic's moves, each bounded and each from a reading of the result against the source:

| the reading | the move |
|---|---|
| clipped beyond 8 % and the source's +3 % | `exposure` −0.3 |
| black beyond 50 % and the source's +20 % (not on a print) | `exposure` +0.3 |
| the key moved by more than 0.15 (not on a print or a plate) | `exposure` ∓0.3 |
| invented beyond `fill_cap` | the move scaled to the cap, the turn eased, the lens change eased |
| the focus region softer than the rest with an aperture | the focus set on the eyes or the subject; if it already was, the aperture ×1.4 |
| the subject's size off by more than 20 % with the tie | the distance re-read by the root of the ratio |
| a face lost | the neural fill stands down, or the move halved |

`make-photo plan <image> --media M` writes the director's plan alone; `--verdict <picture> --plan <file>`
appends the critic's verdict on a picture made by that plan.

## The neural stages (`--neural fill,refine,relight`)

Each is optional, needs weights and a device, and is disclosed in the report; a failure is reported and the
chain goes on.

- **fill** — the opened ground inpainted by a diffusion model (`--neural-model`, default `sdxl`) inside the
  holes' mask with the medium's prompt, the rest of the frame kept to the pixel (`fill: neural` in the plan or
  `--fill neural`);
- **refine** — a whole-frame img2img at `refine` with the medium's `prompt`, the structure held by a
  ControlNet (`--neural-control depth:strength=0.5` by default; `canny:…`; `none`), run at the model's working
  size (`neural_px`, 1024 on the long side — SDXL draws at 1024 and a 2048 frame does not fit a GPU's buffer)
  and brought back to the frame's. At `--refine 0.3` it
  refines a photograph — the material the physics cannot give (the albumen's gloss, the collodion's dirt), the
  faces composited back from before the pass. At `--refine 0.6–0.75` it is **the conversion of a drawing or a
  painting into a photograph**: the brushwork, the paper and the illustrator's light are content, and no lens
  can change content — this pass re-renders the picture as the medium's prompt says while the depth holds its
  composition; the faces are re-rendered too. It runs before the physical chain, so the result then goes
  through the camera, the emulsion and the carrier; it is off by default and disclosed when it runs;
- **relight** — IC‑Light's restaging of the subject's light by a preset (`--relight top`; `top`, `key-left`,
  `key-right`, `rim`, `softbox`, `golden-hour`, `sunset`, `moonlight`, `candlelight`, `overcast`, `neon`); the
  subject is matted out and relit on a directional backdrop, and the result is the model's whole frame,
  background included — the report says so.

## The plan file

```hjson
{
# ---- the house recipe: tri-x ----
format: "35mm"
aperture: 2
focus: "auto"
...
# ---- the director's adjustments ----
source_focal: 28   # a scene: a wide lens (townscape (0.52))
focus: "far"   # a scene (townscape (0.52)) — the focus on the far ground
aperture: 4   # a scene (townscape (0.52)) is shot stopped down, f/2.0 → f/4.0
# ---- the command line ----
grain: 0.8   # the command line
}
```

A key read twice is refused, so a line the director or the critic overrides is kept as a comment
(`(moved below by the director)`). `--plan <file>` makes the picture by a plan; `--set key=value` and the flags
override any key; `make-photo keys` lists every key with its meaning.
