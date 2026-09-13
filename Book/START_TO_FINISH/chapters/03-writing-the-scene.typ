#import "../design.typ": *
#chapter(number: 3, title: "Writing the Scene in Prose")

#dropcap("A") blank lane is a backdrop, not a poster. In this chapter we write the
scene properly: we give it a subject, describe the light and the mood, name the
pieces that will recur, lean on a word or two we care about, and tell the model
what we do *not* want. Everything here is still plain prose in `prompts.txt` — we
are composing, not configuring — but by the end the file holds a real market
scene ready for the analyze/fix pass in the next two chapters.

#section("Sentences, not keyword soup")

If you have used other image tools you may reach for the old habit:
`market lane, night, lanterns, fog, cobblestones, cinematic, masterpiece, 8k,
highly detailed, trending on artstation`. Resist it. plakat's compiler enhances
your prose for whichever model family you target — it knows SDXL likes some
keyword clustering and that Flux and SD3 want flowing description — so your job is
to say clearly *what is in the scene and what it feels like*, in sentences.

#screen(caption: "The scene as prose")[```
  name: market-lane
  A narrow cobbled market lane at night, wet stones catching lantern light.
  A food vendor works a small wooden cart under a canopy strung with warm
  string lights; steam rises from a pot. Soft blue fog gathers beyond the
  cart, and two iron lanterns on stone posts light the foreground. Warm
  amber glow against the cold night, quiet and inviting.
```]

#figure_img("assets/02-vendor-cart.png", "Image 02 — the scene this prose will compile and render to: a vendor at his cart, warm light against cold fog. You are writing the source; the poster comes later, out of the scenario (Chapter 7).")

That reads like a sentence a person would say. It names a subject (the vendor and
cart), places the light (lanterns, string lights), sets the palette (warm amber
against cold blue), and gives a mood (quiet, inviting). That is enough for the
compiler to work with — and, crucially, it is enough for the *analyzer* in the next
chapter to reason about.

#section("Naming the pieces: components")

Our poster will exist in several versions — the empty lane, the single vendor, the
vendor-and-customer, a series with a recurring face. The lane and the cart recur in
all of them, and they should look *identical* every time. Retyping their
descriptions invites drift: the cart gains a canopy in one scene and loses it in the
next. The cure is to define each recurring piece once, as a *component*, and refer to
it by name.

#term("Component")[
  A named, reusable fragment of scene description, declared `component.<name>: <text>`.
  Define it once and refer to it by name from a *directive* — `composition:`,
  `foreground:`, `objects:`, `background:`, or `relate:`. The compiler substitutes the
  component's text wherever it is referenced, so a recurring cart or lane has a single
  source of truth. Components are one shared namespace whether declared in the global
  block or inside a scene.
]

#subsection("Defining components")

You declare components as `key: value` lines, most often in the global block so every
scene can reach them. The name after the dot is how you will refer to it.

#screen(caption: "Defining the recurring pieces")[```
  model: sdxl
  size: 1024x1024
  seed: 1000
  out: ./out

  component.lane:   a narrow cobbled market lane at night, wet stones catching
                    lantern light, soft blue fog beyond, two iron lanterns on
                    stone posts
  component.cart:   a small wooden food cart under a canopy strung with warm
                    string lights, a little chalkboard menu, steam rising
  component.vendor: a food vendor in a dark apron, sleeves rolled up
```]

#warn(label: "A component is not summoned by mention")[
  This is the mistake to avoid. Writing the word "cart" in a free-text sentence does
  *not* pull in `component.cart` — the compiler only substitutes a component where a
  *directive* references it by name. A component you define but never reference from a
  directive is simply unused. Referring to components is a deliberate act, and the
  rest of this section is the four ways to do it.
]

#subsection("Composing a scene from components: `composition:`")

The most direct way to build a scene *out of* components is the `composition:`
directive. List the components you want, comma-separated, and the compiler joins their
descriptions in order — *then* appends whatever free-text prose you add. This is the
"compose, then prose" model: the components lay down the reliable, reusable substance;
your prose adds the staging and mood on top.

#screen(caption: "Compose from components, then add prose")[```
  name: market-lane
  composition: lane, cart, vendor
  The vendor works the cart, (warm amber glow:1.3) against the cold night,
  quiet and inviting.
```]

That scene's prompt is built from the *full* descriptions of `lane`, `cart`, and
`vendor` — every detail you wrote once — followed by the staging sentence. Change
`component.cart` in the global block and this scene, and every other scene that lists
`cart`, updates at once. A scene can even be pure composition, with no prose at all,
when the components already say everything.

#callout(label: "Bare name or `component.` prefix")[
  Inside a directive you may write either the bare name (`composition: lane, cart`) or
  the full form (`composition: component.lane, component.cart`) — the compiler treats
  them the same, stripping the `component.` prefix. Use whichever reads more clearly to
  you; this book uses the short bare names.
]

#subsection("Components as figures and objects")

The other references are the *tiered* directives that Part IV builds on — they, too,
take component names. `foreground:` lists the scene's hero figures, `objects:` the
props that must stay separate, `background:` the supporting elements. Each is a
comma-list of components, and each does something specific with them (fixing the figure
count, reserving a prop its own region) that the composition chapters cover.

#screen(caption: "The same components, used as figures and objects")[```
  name: market-lane
  foreground: vendor      # the hero figure (Chapter 9)
  objects: cart           # a prop kept separate from the figure (Chapter 10)
  composition: lane       # the setting, composed in
  A quiet night; the vendor works the cart, warm amber glow.
```]

The point for now is that *one* `component.vendor` definition serves every role — hero
figure here, a relationship endpoint in Chapter 10, a persona's description in Chapter
11. Define the vendor once; use him many ways.

#subsection("Held props fold into their figure")

There is one clever behaviour worth knowing early, because it prevents a classic
failure. When a component is *attached* to a figure by a holding or wearing
relationship — `relate: vendor holding cup` — the compiler folds the held object's
description *into the figure's own* prompt rather than letting it float as a separate
thing. A held paper cup becomes part of "the vendor," which is exactly where a held
object belongs; it stops the model from painting a disembodied cup hovering nearby.
(Placement relationships like `behind` or `beside` stay as layout, not folding — the
difference is the subject of Chapter 10.)

#screen(caption: "A held object travels with the figure")[```
  component.cup: a steaming paper cup
  name: market-lane
  foreground: vendor
  relate: vendor holding cup    # 'cup' folds INTO the vendor's description
  The vendor offers the cup across the cart.
```]

#subsection("Local overrides")

Components live in one namespace, but a scene may declare its own — with the same name
— to override the global one just for that scene. Global definitions load first, so a
scene-local `component.vendor` shadows the shared one. This is how a single poster in a
series can dress the vendor differently (a raincoat for the "wet night" edition)
without disturbing the others.

#screen(caption: "Overriding a component for one scene")[```
  # global: component.vendor: a food vendor in a dark apron
  name: rainy-night
  component.vendor: a food vendor in a yellow oilskin raincoat, hood up
  foreground: vendor
  composition: lane
  Rain streaks the lantern light; the vendor hunches over the cart.
```]

Taken together, components are how a project stays coherent as it grows: one definition
per recurring piece, referenced by name from wherever it is needed, overridable where a
scene genuinely differs. When the vendor gains a customer in Chapter 9, the cart
becomes a separated object in Chapter 10, and the vendor becomes a persona for the
series in Chapter 11, each of those is the *same* component, reused — never retyped, and
never drifting.

#section("Leaning on a word: weighting")

Sometimes one element matters more than the rest and the model underweights it —
the amber glow washes out, or the fog takes over the frame. plakat understands
*attention weighting* right in the prose: wrap a phrase in parentheses with a
number, and the model pays it proportionally more (above `1.0`) or less (below)
attention.

#screen(caption: "Attention weights in prose")[```
  name: market-lane
  The lane opens before us. A food vendor works the cart,
  (warm amber glow:1.3) against the (cold blue fog:0.9),
  quiet and inviting.
```]

`(warm amber glow:1.3)` tells the model to push the amber; `(cold blue fog:0.9)`
eases the fog back so it frames rather than floods. Use this sparingly — a scene
full of competing weights fights itself. One or two deliberate emphases is plenty,
and the compiler preserves your exact weights through enhancement and budget-fitting
so a `(phrase:1.3)` you wrote survives to the render untouched.

#callout(label: "Weights are not volume knobs")[
  A weight nudges attention, it does not guarantee an outcome. `(amber glow:1.8)`
  will not force a sunset; it just makes the model lean harder on the words you
  already wrote. If a concept refuses to appear at `1.3`, the fix is usually
  clearer *description*, not a bigger number.
]

#subsection("Splitting attention with BREAK")

When two subjects bleed into each other — the vendor's colours smearing onto the
cart, say — you can tell the compiler to encode them as separate chunks with the
keyword `BREAK`. Everything before `BREAK` is encoded independently of everything
after, which keeps distinct elements from contaminating one another.

#screen(caption: "Separating elements with BREAK")[```
  A food vendor in a dark apron works the cart.
  BREAK
  A wooden food cart under a warm string-lit canopy, chalkboard menu.
```]

`BREAK` is a blunt instrument — reach for it only when elements are genuinely
bleeding together. For most scenes, clear prose and a component or two do the job.
The heavier machinery for keeping figures apart — regions, and control-generate —
waits in Part IV, where the poster actually needs it.

#section("Saying what you don't want: negatives")

Every render also has a *negative* prompt: things to steer away from. plakat builds
a sensible one for you automatically — a curated quality set that suppresses the
usual failure modes (extra limbs, watermarks, blur) — so you rarely start from
nothing. When your scene has a specific thing to avoid, add it with a `negative:`
directive and it joins the automatic set.

#screen(caption: "A scene-specific negative")[```
  name: market-lane
  The lane opens before us. A food vendor works the cart, warm amber glow
  against the cold night, quiet and inviting.
  negative: daylight, blue sky, modern signage, cars, crowd
```]

Our poster is a *night* market, so `daylight` and `blue sky` are worth excluding
explicitly; we want it intimate, so `crowd` and `cars` go too. Keep negatives
short and specific to *this* scene — the automatic quality terms already handle the
generic defects, and a bloated negative list dilutes the ones that matter.

#warn(label: "Don't negate in the positive")[
  Write what you *want* in the prose, and what you *don't* in the negative. Phrases
  like "a lane with no people" in the positive prompt often backfire — the model
  reads "people" and paints some. Put `people, crowd` in the `negative:` instead.
  The analyzer in the next chapter flags this exact mistake.
]

#section("Where the scene stands")

Our `prompts.txt` now has real content: shared components for the lane and cart, a
scene that places a vendor in warm light against cold fog, one or two deliberate
weights, and a tight negative list. It reads like a description a person would give
a painter. That readability is not just for you — it is what lets plakat *check*
the scene before rendering it, which is exactly where we go next.

#recap((
  [Write scenes as *sentences* describing what is present and how it feels — the
  compiler handles model-appropriate phrasing, so keyword-soup and "masterpiece,
  8k" boilerplate are unnecessary.],
  [Define recurring pieces once as `component.<name>:`, then *reference them from a
  directive* — `composition:` (compose-then-prose), `foreground:`, `objects:`,
  `relate:`. A bare mention in prose does NOT pull a component in; a scene-local
  definition overrides the global one.],
  [Lean on a phrase with attention weighting, `(phrase:1.3)` up or `:0.9` down, and
  separate bleeding elements with `BREAK` — both sparingly; the compiler preserves
  your weights to the render.],
  [Add scene-specific `negative:` terms on top of plakat's automatic quality set,
  and describe what you *want* in the positive — never "no people," which tends to
  summon them.],
  [Readable prose is not just tidy; it is what makes the scene *analyzable* — the
  subject of the next chapter.],
))
