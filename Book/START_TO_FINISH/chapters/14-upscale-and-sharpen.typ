#import "../design.typ": *
#chapter(number: 14, title: "Bigger, and Sharper")

#dropcap("A")n SDXL poster is born at 1024 pixels square. That is a fine size for a
screen and far too small for a print — a B2 poster wants thousands of pixels on a
side. The last technical step is to enlarge the image to print dimensions without it
going soft. plakat offers two roads: a fast classical resize that just makes the
pixels bigger, and a diffusion upscaler that *invents* believable detail as it grows.
Which you choose depends on how far you are enlarging and how close a viewer will
stand.

#section("The fast road: classical upscale")

The default `upscale` uses a high-quality classical filter (Lanczos). It is instant,
deterministic, and adds no new detail — it interpolates what is already there. For a
modest enlargement, or when you simply need more pixels for layout, it is exactly
right.

#screen(caption: "A clean classical enlargement")[```
  $ plakat upscale --in out/market-graded.png --scale 2 \
      --out out/market-2x.png
    ✓ 1024² → 2048²  (Lanczos)
```]

#figure_img("assets/08-upscaled.png", "Image 08 — the graded frame enlarged to print resolution, its detail intact at the size a viewer will actually stand from.")

Push a classical upscale too far — 4× and beyond — and the image softens: there is
only so much a filter can do with detail that was never captured. That is where the
second road comes in.

#section("The detailed road: diffusion upscale")

`upscale --diffusion` enlarges the image *and* hallucinates plausible fine detail as
it goes — cobblestone texture, the grain of the cart's wood, threads in the vendor's
apron — by running a tile-based diffusion pass guided by the original. It is slower
and it invents, but for a large print viewed close it is the difference between soft
and crisp.

#screen(caption: "Enlarge and add detail")[```
  $ plakat upscale --in out/market-graded.png --diffusion --scale 4 \
      --out out/market-print.png
    ↳ tiling · ControlNet-tile guidance · colour-matched seams
    ✓ 1024² → 4096²
```]

#term("Diffusion upscale")[
  A tile-based super-resolution pass (`upscale --diffusion`) that enlarges an image
  while a diffusion model, guided by the original via a tile ControlNet, invents
  believable high-frequency detail. Seams between tiles are colour-matched and the
  input pre-sharpened so the result is seamless. Slower than classical, and it adds
  detail rather than merely interpolating.
]

#callout(label: "Guided, not reinvented")[
  Because the diffusion upscale is *guided by your image* tile by tile, it adds
  detail without wandering off your composition — unlike a naive whole-image
  `img2img` at high strength. It is the one place a diffusion re-pass is the right
  tool on a finished poster, precisely because it is constrained to enlarging what is
  already there.
]

#section("Faces survive the enlargement")

Enlarging can expose a face that was fine at 1024² but soft at 4096². Pair the
upscale with the face restorer from Chapter 12 — upscale first, then restore — so the
faces are refined at the size they will actually be seen.

#screen(caption: "Upscale, then sharpen the faces")[```
  $ plakat upscale --in out/market-graded.png --diffusion --scale 4 \
      --out out/market-print.png
  $ plakat restore-faces out/market-print.png
```]

#section("How big is big enough?")

A quick rule keeps you from over- or under-cooking the enlargement. Prints are
usually reckoned at 300 DPI, so multiply the physical size in inches by 300 to get
the pixels you need on each side.

#chord_table((
  chord_row("A4 / Letter", "~2500 × 3500 px. A 2–3× upscale from 1024² covers it."),
  chord_row("A3 / Tabloid", "~3500 × 5000 px. A 4× diffusion upscale, then restore faces."),
  chord_row("A2 / B2 poster", "~5000 × 7000 px. 4× diffusion upscale of a portrait-outpainted render, viewed from a step back."),
))

#warn(label: "Order matters at the finish")[
  Do the finishing steps in the right sequence: compose and edit at native size,
  *naturalize*, then *upscale*, then *restore faces* on the enlarged image. Grade
  before you enlarge (so the grain scales with the image), and restore faces after
  (so you sharpen the pixels a viewer will actually see). Getting the order wrong
  gives you grain that looks digital or faces that soften again on enlargement.
]

#section("A poster at print size")

Our NIGHT MARKET is now a large, crisp, print-ready image: composed over Part IV,
graded in the last chapter, enlarged here to poster dimensions with its detail intact
and its faces sharp. It is, as a *picture*, done. One thing separates a finished
picture from a finished *poster* you can hand to a printer or a client: proof of
where it came from, its recipe for remaking, and the title type across the top. That
is the final chapter.

#recap((
  [Renders are born near a model's native size (SDXL at 1024²) and *upscaled* to
  print dimensions — never rendered huge directly.],
  [Classical `upscale` (Lanczos) is instant and interpolates existing detail — right
  for modest enlargements; it softens past ~4×.],
  [`upscale --diffusion` enlarges *and* invents believable detail via a tile
  ControlNet, with colour-matched seams — the choice for large prints viewed close,
  and a *guided* re-pass that won't wander off your composition.],
  [Enlarge *then* `restore-faces`, so faces are sharpened at the size they'll be
  seen; size to 300 DPI × the physical inches you need.],
  [Finish in order: compose → naturalize → upscale → restore faces, so grain scales
  correctly and faces stay crisp.],
))
