#import "../design.typ": *
#chapter(number: 11, title: "A Look, and a Face That Stays")

#dropcap("O")ur poster is composed: two figures, a cart, a relationship, warm light.
What it lacks is a *look* — a deliberate visual treatment that makes it a poster and
not a photograph — and, because we want a *series* of NIGHT MARKET posters, a vendor
whose face is the same in every one. This chapter covers both: how to reach for a
style, whether borrowed, trained, or transferred; and how to pin a person's identity
so it survives from image to image.

#section("Reaching for a style")

The quickest way to a look is to ask for one. The `--look` and `--genre` flags send
plakat to find a matching style adapter — a small trained file called a *LoRA* — and
apply it, so "watercolour" or "ink poster" or "vintage travel print" becomes a
treatment rather than a word the base model half-understands.

#screen(caption: "Ask for a look")[```
  $ plakat generate "<the market-lane prompt>" --model sdxl \
      --look "vintage screen-print poster" --smart-discovery
    ↳ discovered style LoRA: retro-screenprint-xl  (strength 0.8)
    ✓ ./out/plakat-<seed>.png
```]

#figure_img("assets/06-ink-poster.png", "Image 06 — the same scene as a bold two-colour screen-print poster: a look is a treatment applied over the composition, not a new picture.", width: 58%)

`--smart-discovery` is worth adding: a small local judge re-ranks the candidates to
pick a genuine *style* adapter and reject character or person LoRAs that would hijack
your subject. For reproducible or offline runs, `--offline` restricts discovery to
what is already cached.

#term("LoRA")[
  A small trained adapter that nudges a base model toward a style, subject, or
  concept, layered on at generation time. plakat can *find* one for you (`--look` /
  `--genre`), or you can name specific LoRAs and stack them. Many LoRAs need a
  *trigger word* in the prompt to activate.
]

#section("Naming a LoRA — and its trigger")

When you have a specific LoRA in mind — downloaded, or trained yourself — name it
directly in the scenario, and give it its *trigger word*, the token it was trained to
respond to. Forget the trigger and the LoRA loads but does nothing.

#screen(caption: "A named LoRA with its trigger")[```
  model: sdxl
  loras: ink-poster-xl:0.65
  lora-trigger: inkposter

  name: market-lane
  foreground: vendor, customer
  objects: cart
  An inkposter night market: a vendor at his cart, a customer leaning in.
```]

Two details matter here. The `:0.65` sets the LoRA's *strength* — how hard it pulls;
0.6–0.8 is a sane range, and pushing to 1.0 often overwhelms the scene. And
`lora-trigger:` injects the trigger word (`inkposter`) so the adapter actually fires.
plakat keeps the trigger verbatim and reserves room for it in the token budget, so
your careful prose isn't crowded out by it.

#warn(label: "A trigger word is not optional")[
  The single most common "my LoRA does nothing" bug is a missing trigger word. If a
  named LoRA has no visible effect, check its documentation for the trigger and add
  it with `lora-trigger:`. plakat auto-detects some (like LCM speed LoRAs); style and
  character LoRAs usually need you to name theirs.
]

#section("Two other roads to a style")

Borrowing a LoRA is one road; there are two more, for when you want a look that is
truly yours.

#subsection("Transfer a style from a reference image")

If you have an image whose *look* you want — a print you admire, an earlier poster —
`stylize` transfers its style onto your content without a trained adapter at all.

#screen(caption: "Style from a reference")[```
  $ plakat stylize --in out/market-lane.png \
      --ref refs/old-travel-poster.jpg --out out/styled.png
```]

#subsection("Train your own style LoRA")

And if you have a body of work in a consistent style — say a dozen of your own ink
drawings — you can *train* a LoRA on them, so "inkposter" means *your* ink, not a
stranger's. `plakat style train` builds an SDXL or SD 3.5 style LoRA from a folder of
examples; the result is a file you name in `loras:` like any other.

#screen(caption: "Training a style is a real option")[```
  $ plakat style train --data ./my-ink-drawings \
      --out ./loras/my-ink-xl --model sdxl
```]

(There is a parallel path for a *specific object or character* — textual-inversion
embeddings via `plakat embedding`, injected with `generate --embedding` — when a
single subject, not a whole style, is what you want to pin.)

#section("A face that stays: personas")

Now the harder identity problem. A NIGHT MARKET *series* wants the same vendor in
every poster — same face, same build — not a new stranger each render. A seed does
not give you that; change anything else and the face drifts. plakat's answer is a
*persona*: a reproducible synthetic person you define once and place into scenes.

#term("Persona")[
  A controllable synthetic person, defined once in a `PersonaSpec` and reused across
  images so the *same* face and build appear every time. `plakat persona new`
  scaffolds a spec; `multiperson` places one or more personas into a generated scene,
  each pinned to a location — the way to give a poster *series* a consistent cast.
]

#screen(caption: "Define the vendor once, reuse him")[```
  $ plakat persona new vendor --out personas/vendor.hjson
    ✓ scaffolded personas/vendor.hjson  (edit the traits, then lint)
  $ plakat persona lint personas/vendor.hjson
    ✓ valid

  # place the fixed vendor (and a customer) into the scene
  $ plakat multiperson "a night market lane, warm string lights" \
      --at "vendor:right at the cart" \
      --at "customer:left, leaning in"
```]

With the vendor a persona, every poster in the series shows the same man at his cart.
`multiperson` pins each persona to a relative spot ("right at the cart," "left,
leaning in") and lets a scene-aware planner place anyone you leave unpinned. If you
have a *reference photo* of the face you want, it can anchor the persona's identity to
that face.

#callout(label: "Identity vs. a one-off face")[
  For a single poster you don't need a persona — a good seed and a clear description
  are enough. Personas earn their keep the moment you want the *same* person *again*:
  a series, a campaign, a character across a comic (plakat makes those too). Reach for
  identity control only when continuity is the requirement.
]

#section("The poster, dressed")

NIGHT MARKET now has its look — a screen-print ink treatment, borrowed or trained —
and, for the series, a vendor whose face will not wander. Compositionally and
stylistically, the image is *done*: it says what we want, staged how we want, in the
treatment we chose. What remains is finishing — editing the last small flaws, making
the render feel like a made object rather than a machine output, enlarging it to print
size, and signing it. That is the final part.

#recap((
  [`--look` / `--genre` (with `--smart-discovery`) find and apply a *style LoRA* so a
  named treatment becomes real; `--offline` keeps discovery reproducible.],
  [Name specific LoRAs in `loras: name:strength` (0.6–0.8 is sane) and *always* give
  a style/character LoRA its `lora-trigger:` word — the top cause of "my LoRA does
  nothing."],
  [Two other roads to a look: `stylize` transfers a style from a *reference image*
  with no adapter, and `plakat style train` builds a LoRA from *your own* body of
  work.],
  [A *persona* pins a synthetic person's identity across images — `persona new` to
  define, `multiperson` to place — the way to give a poster *series* a consistent
  cast; reach for it only when continuity is required.],
  [With a look and a stable face, the poster is compositionally done; what remains is
  finishing — the last part of the book.],
))
