# Transparency, binarisation, print size — the B/W-native finisher

This is the counter-intuitive core of `bookart`, kept separate the way persona's
[details how-to](PERSONA_DETAILS_HOWTO.md) is. The headline: **you do not generate a grey illustration
and then remove its background.** Black-and-white ink has a *better* transparency model available than
photo-matting can reach, and `bookart` uses it. Everything below runs in the **finisher** — the
ordered post chain that turns a raw render into the deliverable: `technique binarise → transparency →
symmetry → (opt) vectorise`, then Layer-5 page sizing.

## Why not "generate grey, then remove the background"

The naive control (measured in the G0.2 baseline harness) fails three ways at once:

| metric | naive woodcut | naive line-art | bookart finisher |
|---|---|---|---|
| chroma (coloured fraction) | 0.380 | 0.162 | **0.000** |
| alpha-halo (partial-alpha ring) | 0.418 | 0.355 | **0.000** |
| page-haze (near-white lift) | 3.4 | 2.1 | **0.0** |

Diffusion "black and white" is really a **16–38% tinted** desaturated photo, and keying it out leaves a
**35–42% partial-alpha halo** — a grey ring that fringes every line and greys the page it lands on. The
finisher zeroes all three, because it never treats the image as a photo with a background to subtract.

## B/W-native transparency — ink darkness *is* opacity

The insight: on a page of ink, dark = present, white = absent. So map **luminance directly to alpha**.
There is no foreground/background segmentation, no matte, no halo — a mid-grey hatch line simply becomes
a *semi-transparent* grey mark that sits correctly on any page colour or texture.

The `luminance` curve (the default, tuned in the G0.5 probe):

```
alpha = clamp( (1 − L − white_cut) / (1 − white_cut) , 0, 1 ) ^ gamma
```

with `L` the pixel luminance in `[0,1]` and the frozen defaults **`white_cut ≈ 0.07, gamma ≈ 0.70`**.
Two design choices earn their keep:

- **`white_cut`** snaps near-white paper (and the pale haze diffusion sprays across a "white"
  background) fully to zero alpha → **page-haze 0.0**. Without it, linear luminance leaves a 9.5-unit
  haze that dulls the page under the ornament.
- **`gamma < 1`** lifts mid-grey line coverage so thin and grey strokes stay *opaque* rather than
  fading into semi-transparent ghosts (mid-grey alpha 147 vs 127 linear), while the `white_cut` still
  kills the anti-alias halo. A steeper `gamma 0.6` over-lifts and re-introduces haze (33.2); `0.70` is
  the sweet spot.

Black ink → alpha 255 (opaque); white and near-white paper → alpha 0 (transparent); grey → partial,
proportional to darkness. That is the whole model.

### The transparency modes (`ink.transparency`)

| Mode | For | Behaviour |
|---|---|---|
| **`luminance`** | line / hatch / most ornament (default) | the curve above — ink darkness = opacity |
| **`threshold`** | crisp 1-bit line | a hard cut with a soft anti-alias ramp around mid-grey (preserves edge AA without a halo) |
| **`matte`** | solid silhouette pieces | U2Net subject mask → a solid silhouette in the ink tint (weights; a convenience — quality on ornament is unverified) |
| **`fade`** | vignettes / spot art | luminance alpha with a radial edge falloff over the outer `fade` fraction, so a spot dissolves into the page |

## Ink tint — recolour without regenerating

Transparency lives in the alpha channel; the ink colour lives in RGB. So **one transparent asset drops
onto any page *and* re-tints without regeneration.** `output.tint` (or `ink.color`) recolours every
inked pixel while the alpha is untouched:

`black` → `[0,0,0]`, `white` → `[255,255,255]`, `sepia` → `[80,54,28]`, or any `#rrggbb`.

Because the recolour is a pure RGB swap on a finished PNG, it is a `post`-class edit — `bookart edit
border.png --tint sepia --out border-sepia.png` re-tints with **no re-render, no GPU**.

## Technique binarisers

Before transparency, the render is binarised per the `technique`, so "line" and "woodcut" of the same
origin read as genuinely different hands. The finisher dispatches on the binariser the lexicon assigns
to each technique:

| Technique | Binariser | What it does |
|---|---|---|
| `line` (default) | **xdog** | XDoG / adaptive threshold → a clean contour line |
| `woodcut` | **threshold-bold** | high-contrast threshold + bold-mass cleanup (ink dilate) |
| `engraving` | **engrave-invert** | white-on-black fine lines (invert + fine-line preserve) |
| `stipple` | **dither** | Floyd–Steinberg error-diffusion dots |
| `cross-hatch` | **xdog** | cross-hatched pen lines |
| `silhouette` | **matte-solid** | a solid filled shape |
| `ink-wash` | **halftone** | a Bayer halftone that keeps tone under a 1-bit target |
| `scratchboard` | **threshold-invert** | white lines on black |

All are deterministic; `ink.weight` biases the threshold (heavier ink → more coverage). The procedural
tier is **born-clean** and skips binarisation entirely — it goes straight from vector strokes to
transparency.

## Born-vector SVG vs raster trace

Vectorisation is **off by default** and off the critical path — the PNG is always the deliverable. When
you ask for SVG (`--svg`, or `output.formats` includes `svg`):

- **Procedural tier — born vector.** The parametric generators emit polylines directly; the finisher
  serialises them to print-sized SVG paths (physical `mm` size + a px `viewBox`, stroked in the ink
  tint, with the layout's corner flips applied). Mathematically exact, near-free — no tracing.
- **Diffusion / composite — raster trace.** Tracing a finished raster to SVG (with the permissive
  `vtracer`/`visioncortex` tracer, MIT/Apache-2.0 — never GPL potrace) is a documented **fast-follow**;
  today `render --svg` on a non-procedural tier keeps the PNG and prints a note. Transparency would ride
  the traced paths as fill-opacity.

So SVG is the escape hatch for infinite-DPI / editable procedural ornament; raster is the core for
everything.

## The exact-print-size model

An ornament is useless for print at an arbitrary pixel resolution. `bookart` sizes to **true page
geometry**:

- **Named sizes → px at DPI.** `a4`/`a5`/`a6`/`b5`/`letter`/`legal`/`trade`/`mass-market` (and
  `custom: {w_mm,h_mm}`) resolve mm → px at the target DPI. A4 @300 → 2480×3508 px; A5 @300 →
  1748×2480. `orientation` (portrait/landscape) swaps the axes.
- **DPI embedded.** The output PNG carries the DPI in its **`pHYs` chunk**, so downstream tools (InDesign,
  LaTeX) place it at the correct physical size, not at screen resolution.
- **Text-block / margins / bleed.** Ornaments anchor to the **text block** derived from
  `page.margins` + `gutter_mm` (book defaults if unset) — a headpiece is the text-block width, not the
  raw page width — with an optional `bleed_mm` for pieces that run to the trim edge. The layout engine
  places each ornament type at its canonical position against that block (headpiece top band, tapering
  tailpiece under the last line, four inward-flipped corners, page-fill frontispiece, and so on).

`bookart show` prints the resolved canvas (px @ DPI, mm, bleed); `bookart verify --page` proves the
placement by writing a page-sized PNG and its `resolution` probe confirms `px == size × dpi`.

## The symmetry engine — a geometric guarantee diffusion can't hold

Most ornament is bilaterally or radially symmetric *by construction*. Diffusion cannot hold that — a
"symmetric border" comes out lopsided (the baseline measured a bilateral symmetry RMS of 0.29–0.48 that
the finisher **cannot** fix; transparency and binarisation leave asymmetry untouched). So symmetry is a
*geometric* operation, applied after the finish:

- **`bilateral`** — mirror-union about the vertical axis (exact; RMS → 0).
- **`radial:N`** — N-fold rotational union (bilinear resample).
- **`frieze:GROUP`** / **`none`** — passthrough.

For the procedural tier the symmetry is inherent to the generator; for diffusion the finished ornament
is folded/replicated after the fact, which is why `verify --symmetrize` is the exact tool for the one
defect the finisher provably can't touch. (The composite tier skips it — its frame is already symmetric
and the inlaid picture is a scene, not a mirror-double.)

The takeaway that ties this document together: **chroma, halo, and haste** are killed by luminance-alpha
+ binarise, **size** is a page/DPI fact, and **symmetry** is a geometric guarantee — three independent
guarantees a prompt cannot make, which is the whole reason `bookart` exists.
