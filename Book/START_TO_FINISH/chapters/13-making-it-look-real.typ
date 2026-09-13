#import "../design.typ": *
#chapter(number: 13, title: "Making It Look Real")

#dropcap("T")here is a particular sheen that says "made by a model": too clean, too
even, edges a hair too crisp, a plasticky perfection no camera and no printing press
ever produced. Your poster may be beautifully composed and still wear that sheen.
The `naturalize` pass exists to take it off — to add back the small analog
imperfections that make an image read as a *made object* rather than a generation.
It is the quiet, final craft step, and it is entirely weight-free: no model, no GPU,
just honest signal processing.

#section("The AI fingerprint, and how to dull it")

`naturalize` applies an analog post-pass — the kind of thing that happens to a real
photograph on its way through a lens, a sensor, and a print: a whisper of film grain,
a touch of chromatic aberration at the edges, a gentle vignette, a little bloom in
the highlights, and a desaturating film grade that pulls the colour off its digital
saturation.

#screen(caption: "The finishing pass")[```
  $ plakat naturalize out/market-final.png --out out/market-graded.png
    ◐ detecting medium … photographic
    ◐ grade · grain · aberration · vignette · bloom
    ✓ out/market-graded.png
```]

#term("naturalize")[
  A weight-free analog post-pass that reduces the "AI-generated" fingerprint of an
  image — film grain, chromatic aberration, vignette, bloom, and a desaturating film
  grade. It aims at *realism*, not a vintage-filter look, and touches only the
  finish, never the composition. No model runs; it is fast and deterministic.
]

#figure_img("assets/07-naturalized.png", "Image 07 — after the naturalize pass: the same frame with grain, a gentle grade, and a little bloom. Less rendered, more made.")

The result should be subtle. Put the two side by side and the graded one simply
looks *less rendered* — the amber glow has a little bloom, the fog carries faint
grain, the corners fall off gently the way a real lens darkens them. Nothing about
the scene changed; only its surface.

#section("Realism, not a filter")

It is worth being precise about what naturalize is *for*, because the name invites a
wrong guess. It is not a nostalgia filter that makes everything look like a 1970s
snapshot. It is a *de-slop* pass: it targets the specific tells of machine generation
and dials them down toward how real optics and real media behave. Ask for "vintage"
and you want a *look* (Chapter 11's job — a LoRA or a grade). Ask for naturalize and
you want the machine-ness *reduced*, whatever the style.

#warn(label: "Why not just img2img it?")[
  It is tempting to run a finished poster through a light `img2img` to "make it more
  real." Resist it on cohesive art. A whole-image re-render — even a gentle one —
  tends to *regress* a carefully composed image: it softens intentional choices and
  re-introduces its own artefacts. naturalize is weight-free precisely so it can
  improve the *finish* without re-rolling the *content*. Any part of the pipeline
  that produces smearing should be fixed upstream (model, steps, seed), not papered
  over with another render.
]

#section("Protecting the figures")

The plain grade above is weight-free and gentle — it will not maul a face. But
naturalize has a heavier, *model-backed* mode as well: `--repaint` re-paints the
image in a chosen medium (oil, watercolour, ink) to build real brush character. That
mode *can* disturb faces, so it carries a protection scope that shields the people
while it loosens everything else.

#screen(caption: "A painterly repaint that spares the figures")[```
  $ plakat naturalize out/market-final.png \
      --repaint --repaint-protect figures
```]

`--repaint-protect figures` (the default when `--repaint` is on) preserves each
figure's whole body — face, hands, clothing — feather-composited back over the
repainted surroundings; `faces` protects only the faces; `none` repaints everything
(most painterly, but small figures can melt). For a portrait-forward poster, keep the
figures protected: the lane, fog, and cart take the painterly texture while the two
people stay crisp. `--repaint` needs a model; the plain weight-free grade does not,
and is the right default for a photographic poster like ours.

#section("Naturalize from the prose")

Because finishing is part of making an image, plakat can bake a scene-tuned
naturalize into the compile itself — the analyzer's companion, `--make-composition`,
even suggests one. In prose it is a directive, so the finish travels with the scene.

#screen(caption: "A finish that travels with the scene")[```
  name: market-lane
  naturalize: grain=0.3 vignette=0.2 desaturate=0.15
  A night market lane, a vendor at his cart, a customer leaning in.
```]

Now every render of this scene emerges already graded, with the figures protected —
no separate command, no forgotten step. For a poster you will render many times, that
consistency is worth having; the finish becomes part of the source, like everything
else in the prose project.

#callout(label: "You will see it working")[
  A naturalize pass prints progress as it runs — detecting the medium, then each
  stage of the grade — rather than sitting silent. On a large print-resolution image
  the pass takes a moment; the running messages tell you it is working, not stuck.
]

#section("A correct image becomes a finished one")

With the poster graded, it has crossed the line from *correct* to *finished*: it no
longer announces that a model made it. Two practical steps remain before it is a
print — making it physically large enough, and signing it so its origin travels with
it. Those are the last chapter of the making.

#recap((
  [`naturalize` is a *weight-free* analog post-pass — grain, chromatic aberration,
  vignette, bloom, a desaturating grade — that reduces the "AI-generated" fingerprint
  without touching the composition.],
  [It aims at *realism*, not a vintage filter: it is a *de-slop* pass that dials the
  machine-tells toward how real optics behave; a "look" is still Chapter 11's job.],
  [Do *not* finish cohesive art with a whole-image `img2img` — it regresses careful
  work; naturalize improves the finish precisely because no model re-renders the
  content.],
  [The heavier *model-backed* `--repaint` mode builds real brush character;
  `--repaint-protect figures|faces|none` shields the people while it works — but the
  plain weight-free grade is the right default for a photographic poster.],
  [Bake it into the prose with a `naturalize:` directive so the finish travels with
  the scene and every render emerges already graded.],
))
