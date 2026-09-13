#import "../design.typ": *
#chapter(number: 12, title: "Editing What You Have")

#dropcap("N")ot every problem is worth a re-render. When a poster is ninety per cent
right — the composition holds, the light is good — but one thing is wrong (a garbled
hand, a distracting sign, a face that came out soft), you do not throw the image away
and roll the seed again. You *edit* the pixels you have. plakat has a full set of
surgical tools for exactly this, and reaching for them instead of re-rolling is often
the difference between finishing tonight and chasing a better seed until midnight.

#section("Changing the whole image a little: `img2img`")

The broadest edit takes an existing image and re-renders it *guided by itself*, so
the composition survives while the treatment shifts. A low strength nudges; a high
strength reinvents.

#screen(caption: "Guided re-render")[```
  $ plakat img2img out/market-lane.png \
      --prompt "warmer amber light, softer fog" --strength 0.35
```]

#term("Strength (img2img)")[
  How far an `img2img` render is allowed to travel from the original, `0.0–1.0`. Low
  (`0.2–0.4`) preserves the composition and adjusts tone or detail; high (`0.6+`)
  keeps only the rough layout and repaints the rest. Start low — you can always push.
]

#warn(label: "Low strength on finished art")[
  A whole-image `img2img` at high strength will happily *undo* the careful
  composition you built over ten chapters. On a cohesive poster, keep strengths low
  and prefer the *localised* edits below. The lesson plakat learned the hard way:
  re-rendering a whole cohesive image tends to make it worse, not better.
]

#section("Fixing one spot: masks and inpaint")

Most edits should touch *one region*, not the whole frame. Give `img2img` a `--mask`
and it changes only the masked pixels — the rest is untouched. This is *inpainting*,
and it is how you fix a bad hand or repaint a single object without disturbing
anything else.

You need a mask — a black-and-white stencil of the region. You can paint one, or let
plakat make one by *clicking* the thing you mean, using Segment-Anything.

#screen(caption: "Select by clicking, then inpaint")[```
  # click the garbled hand → a mask
  $ plakat segment out/market-lane.png --point 620,540 \
      --out mask.png
  # repaint only that region
  $ plakat img2img out/market-lane.png --mask mask.png \
      --prompt "a clean hand holding a paper cup" --strength 0.8
```]

#section("Removing something entirely: `remove`")

When the fix is *deletion* — a stray sign, a smudge, an object that cluttered the
frame — `remove` erases it and fills the hole seamlessly, in one command. Point at it,
box it, pick it by depth, or name it in words.

#screen(caption: "Four ways to say what to remove")[```
  $ plakat remove out/market-lane.png --point 210,300    # click it
  $ plakat remove out/market-lane.png --box 0,0,180,400  # box it
  $ plakat remove out/market-lane.png --what "modern sign" # name it
```]

`--what` is open-vocabulary: an object detector finds the thing you named, and the
region is inpainted away while the rest of the poster is preserved. It is the fastest
way to declutter a frame after the fact.

#section("Swapping the background: `replace-bg`")

Sometimes the subject is perfect and the *background* is not. `replace-bg` mattes the
subject out, generates (or loads) a new background, and composites the subject back
over it — decontaminating the edges so there is no halo.

#screen(caption: "Keep the vendor, change the night behind him")[```
  $ plakat replace-bg out/vendor-cut.png \
      --prompt "a deep foggy market lane at night, bokeh string lights" \
      --keep vendor
```]

#section("Faces: `restore-faces` and `faceswap`")

Faces are where the eye goes first, so plakat gives them dedicated tools. When a face
renders soft or slightly wrong — common on small figures far from the camera —
`restore-faces` detects each face, refines it with a diffusion pass, and composites
it back, sharper and truer, without touching the rest of the image.

#screen(caption: "Sharpen the faces only")[```
  $ plakat restore-faces out/market-lane.png
    ↳ SCRFD found 2 faces · refined · composited
```]

And when you need a *specific* face — the persona from the last chapter, or a fixed
model across a series — `faceswap` transplants a source face onto the figure in your
image, aligned and colour-matched so it belongs.

#screen(caption: "Put a specific face on the vendor")[```
  $ plakat faceswap out/market-lane.png --source faces/vendor.jpg \
      --out out/market-final.png
```]

#callout(label: "A licensing note")[
  `faceswap` uses the inswapper model, whose weights are *non-commercial* (InsightFace's
  terms). It is perfect for personal work, series continuity, and studies; check the
  licence before you put a swapped face on something you sell. `restore-faces`, which
  only sharpens what is already there, carries no such restriction.
]

#section("Extending the frame: `outpaint`")

Finally, when the poster needs *more canvas* — headroom for a title, a wider lane —
`outpaint` pads the image and paints the new border to match, so a 1:1 render can
grow into the tall format a poster wants.

#screen(caption: "Grow the canvas upward for a title")[```
  $ plakat outpaint out/market-lane.png --top 320 \
      --prompt "string lights and dark sky above the lane"
```]

#section("Edit, don't re-roll")

Every tool in this chapter shares one idea: *keep the good render and change only what
is wrong.* A soft face, a stray object, a background, a too-square frame — none of
these is a reason to abandon an image you spent real effort composing. The instinct to
re-roll the seed is usually the expensive instinct. With the poster now edited to
clean — hands fixed, clutter gone, the face right — we can move to the two passes that
turn a correct render into a finished, printable object: making it feel real, and
making it big.

#recap((
  [Edit the pixels you have instead of re-rolling: `img2img` re-renders guided by the
  image (keep *strength* low on finished art — whole-image high-strength re-renders
  tend to make cohesive work worse).],
  [Fix one region with a `--mask` (inpaint); make the mask by *clicking* the object
  with `segment`, or paint your own.],
  [`remove` erases an object and fills the hole — select it by `--point`, `--box`,
  depth band, or open-vocabulary `--what "a modern sign"`.],
  [`replace-bg` swaps the background while keeping and decontaminating the subject;
  `restore-faces` sharpens soft faces, and `faceswap` transplants a specific face
  (non-commercial weights).],
  [`outpaint` extends the canvas — headroom for a title, a wider frame — painting the
  new border to match; the through-line is *keep the good render, change only what's
  wrong*.],
))
