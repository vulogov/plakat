#import "../design.typ": *
#chapter(number: 9, title: "More Than One Figure")

#dropcap("U")p to now our poster has had one focal figure — the vendor — and the
model has placed him wherever it liked. That is fine for one person. Add a second,
and diffusion models start to struggle: two figures blur into each other, swap
limbs, or drift to opposite corners of the frame. Our poster needs the vendor *and*
a customer, standing where we want them. This chapter is about taking that control
without overreaching for it — climbing the control ladder exactly one rung.

#section("The control ladder")

Before adding machinery, it helps to see the whole range of control plakat offers,
because the craft is in using *the least that works*.

#control_ladder()

Every rung up costs effort and can cost naturalness — a heavily pinned scene can look
stiff. So we climb only as far as the image needs. One figure, free-form, needed no
rungs. Two figures who must not merge need one: *shaping* them in prose, and letting
plakat's composition pass keep them apart.

#section("Naming the figures: `foreground:`")

The first tool is a directive that tells the compiler how many deliberate people are
in the scene and who they are.

#term("foreground:")[
  A list of the scene's intended focal figures, e.g. `foreground: vendor, customer`.
  It tells the compiler the exact number of people to plan for — which becomes the
  render's figure count — so a second person is a *decision*, not an accident the
  model may or may not honour.
]

#screen(caption: "Two named figures")[```
  component.vendor:   a food vendor in a dark apron, sleeves rolled up
  component.customer: a customer in a wool coat, seen from behind

  name: market-lane
  foreground: vendor, customer
  A vendor works the cart under warm string lights while a customer
  leans in to order. Warm amber glow, soft fog beyond.
```]

Naming the two figures does two things: it fixes the count (the model will not add a
third or lose one), and it hands each figure a stable description you can reuse — the
same instinct as the components from Chapter 3, now applied to people.

#section("Turning on composition: `control-generate`")

For two figures to reliably hold their positions, plakat can run a *composition
pass* — it plans where each figure goes, drafts the layout, and then finishes the
image. You switch it on with one directive.

#term("control-generate")[
  A directive (`control-generate: sdxl`) that enables plakat's multi-stage
  composition pipeline: it builds a pose/placement plan for the named figures, draws
  a regional draft that puts each where it belongs, and hands that draft to a finish
  pass. The value names the *draft* model. Pair it with `control-generate-size` for
  that model's native resolution.
]

#screen(caption: "Enabling the composition pass")[```
  name: market-lane
  foreground: vendor, customer
  control-generate: sdxl
  control-generate-size: 1024x1024
  A vendor works the cart while a customer leans in to order.
```]

Under the hood this runs in stages — a placement plan, a regional draft that paints
each figure in its own area of the canvas, and a finishing pass that harmonises the
whole. You do not manage the stages; you describe the scene and choose the draft
model. `--explain` shows the compiled control directives if you want to see them.

#compile_stages()

#section("Posing the figures")

The plan needs to know *how* each figure stands, and here prose does most of the
work: describe the posture and the compiler reads it. "A vendor *standing* at the
cart" and "a customer *leaning in*" become distinct poses in the plan. You can also
state them explicitly when you want no ambiguity.

#screen(caption: "Posture from prose, or stated outright")[```
  # read from the prose verbs …
  A vendor stands at the cart; a customer leans across the counter.

  # … or pinned explicitly, one per foreground figure
  control-generate-figure-poses: standing, leaning
```]

plakat knows a vocabulary of postures — standing, sitting, kneeling, leaning,
reaching, carrying, and more — and builds a pose skeleton for each figure that the
draft honours.

#warn(label: "An honest word about hard poses")[
  Standing figures are where pose control is strongest. Seated, crouched, and other
  unusual poses are genuinely harder for SDXL's pose adapters — the draft may fight
  you. If a pose refuses to hold, two things help: the SD 1.5 draft model
  (`control-generate: sd15`, at `512x512`) has stronger pose control for difficult
  postures, and simplifying the pose ("leaning in," not "crouched and twisting")
  costs you nothing. Our vendor and customer both stand or lean, which is the
  reliable range.
]

#section("The regional toggle")

By default the composition pass paints each figure in its own region so they don't
merge. Occasionally — a very simple two-figure scene — you may want to let the finish
pass blend more freely. The `control-generate-regional` toggle controls that.

#screen(caption: "Regional draft on or off")[```
  control-generate-regional: true     # each figure in its own area (default)
  control-generate-regional: false    # let the finish pass blend more
```]

For our poster we keep it on: the vendor belongs at the cart on the right, the
customer at the counter on the left, and we want that separation to survive the
render. If figures ever come out *too* separated — like cut-outs pasted side by
side — turning regional off and leaning on clear prose can soften the seam.

#section("Where the poster stands")

#figure_img("assets/04-two-figures.png", "Image 04 — two figures held apart: the vendor at the cart and a customer leaning in, each keeping position instead of merging.")

Run it, and the draft now reliably shows two people: the vendor at his cart, the
customer leaning in to order, each holding position instead of melting together. We
climbed one rung — named the figures, enabled composition, let posture flow from the
prose — and stopped there. The scene did not need pinned coordinates or personas
yet.

But there is still a lie in the frame: the cart. The model wants to treat the vendor
and his cart as one blob of "person-with-stuff," fusing the wooden cart into his
body. Keeping a *prop* distinct from a *person* is a different problem from keeping
two people apart — and it is the next chapter.

#recap((
  [Climb the *control ladder* only as far as the image needs — every rung costs
  effort and can cost naturalness; two non-merging figures need just one rung.],
  [`foreground: a, b` fixes the scene's figure *count* and names each person, so a
  second figure is a decision the render honours, not an accident.],
  [`control-generate: <draft-model>` (with `control-generate-size`) turns on the
  composition pass — plan, regional draft, finish — that keeps figures in their
  places; you describe the scene, plakat runs the stages.],
  [Posture flows from prose verbs (standing, leaning, reaching…) or can be pinned
  with `control-generate-figure-poses:`; *standing/leaning are the reliable range*,
  and the `sd15` draft is stronger for hard poses.],
  [`control-generate-regional` keeps each figure in its own area (default on);
  turn it off only if figures come out looking pasted-on.],
))
