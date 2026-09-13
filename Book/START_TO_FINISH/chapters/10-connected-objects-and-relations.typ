#import "../design.typ": *
#chapter(number: 10, title: "A Cart, and a Relationship")

#dropcap("A") diffusion model has a strong instinct to fuse. Give it "a vendor at a
cart" and it will happily paint one object — a person-shaped mass with cart-like bits
growing out of the torso — because to the model the vendor and his cart are one
region of "stuff that goes together." Our poster needs them *distinct*: a person, and
beside him a separate wooden cart he is working. It also needs the two figures to
relate — the customer facing the vendor, not staring past him. Both are jobs for a
small, specific vocabulary that this chapter covers.

#section("The fusion problem, concretely")

You have already met fusion's cousin. In Chapter 4 the analyzer warned about
"person + cargo fusion" — a big carried object melting into the figure holding it. A cart
beside a vendor is the same failure in a new costume: the model blends the prop into
the person. Prose alone struggles here, because "a vendor at a wooden cart" *reads*
as one thing. You have to tell plakat that the cart is its own object.

#section("Declaring a separate object: `objects:`")

The `objects:` directive names props that must stay distinct from the figures — the
compiler plans them their own place in the composition rather than letting them
collapse into a person.

#term("objects:")[
  A list of scene props that must render as *separate* objects, not fused into the
  figures — e.g. `objects: cart`. The compiler gives each its own region in the
  composition plan, with a gap from the figures, so a cart stays a cart beside the
  vendor instead of melting into him. The tool for the thing the model wants to
  absorb.
]

#screen(caption: "The cart as its own object")[```
  component.vendor: a food vendor in a dark apron, sleeves rolled up
  component.cart:   a wooden food cart with a canopy and a chalkboard menu

  name: market-lane
  foreground: vendor
  objects: cart
  control-generate: sdxl
  control-generate-size: 1024x1024
  A vendor works beside his cart under warm string lights.
```]

#figure_img("assets/05-cart-object.png", "Image 05 — the cart as its own object: a separate wooden cart beside the vendor, not a person-shaped mass with cart parts growing out of it.")

With the cart declared as an object, the composition pass reserves it a region of the
canvas next to the vendor — the wooden cart on one side, the person on the other,
each rendered as itself. The blob becomes two things.

#callout(label: "Objects vs. held props")[
  Not every prop needs `objects:`. A small thing a figure *holds* — the paper cup the
  vendor hands over — travels with the figure and is fine in prose. Reserve `objects:`
  for the large props the model tries to *absorb*: a cart, a vehicle, a market stall.
  If a prop keeps fusing into a person, that is the signal to promote it to an object.
]

#section("Relating the figures: `relate:`")

Two figures in a frame are not yet a *scene* — a scene has them interacting. Our
customer should face the vendor across the cart, leaning in to order. The `relate:`
directive states that relationship so the composition honours it.

#term("relate:")[
  A relationship between two named components, written
  `relate: <a> <relationship> <b>` — e.g. `relate: customer facing vendor`. The
  compiler uses it to orient and place the figures so they interact as described,
  rather than standing back-to-back or ignoring each other.
]

#screen(caption: "The customer faces the vendor")[```
  name: market-lane
  foreground: vendor, customer
  objects: cart
  relate: customer facing vendor
  control-generate: sdxl
  control-generate-size: 1024x1024
  The customer leans across the cart to order; the vendor reaches out
  a paper cup. Warm amber glow, soft fog beyond.
```]

Now the plan knows the geometry of the moment: vendor at his cart, customer opposite
and facing him, the cart between them as its own object. That is the full staging of
our poster's hero shot — two people, one prop, one interaction — expressed in three
short directives on top of plain prose.

#subsection("Contacts: where figures touch")

When figures don't just relate but *touch* — a hand on a shoulder, an arm around a
companion — `control-generate-figure-contacts` pins that contact so the limbs meet
where they should. Our customer and vendor don't touch (the cart is between them), so
we don't need it here, but it is the tool when an embrace or a handshake must land.

#screen(caption: "Pinning a physical contact")[```
  # e.g. two figures side by side, one arm around the other
  control-generate-figure-contacts: vendor arm-around customer
```]

#section("An honest word on the ceiling")

It would be dishonest to promise that any arrangement of figures, objects, and
relations renders perfectly. Some do not. Two *coupled wheeled vehicles* — an engine
towing a separate trailer — sit right at the edge of what today's SDXL composition
can hold; the model wants to make them one four-wheeled thing no matter how you
declare them. Deeply *seated* figures in a busy scene are similarly hard. When you
reach that edge, the honest path is not to fight the render but to change the plan:
simplify the arrangement, split an impossible scene into two images, or — for a truly
fixed layout — reach for the higher rungs of the ladder (pinned poses, a
control pre-image) that a specialised job needs.

#warn(label: "Read the analyzer before you climb")[
  `compile --analyze` (Chapter 4) knows these ceilings. It flags fused-vehicle and
  person-cargo risks *before* you render, and its advice — "use a control pre-image
  for two coupled vehicles," "split the over-stuffed scene" — is exactly the guidance
  above, delivered early. When composition gets ambitious, analyze first.
]

Our poster is comfortably within the reliable range: two upright figures, one
separated prop, a facing relationship. It stages cleanly. What it does *not* have yet
is a consistent *look* and, for the series we want to make, a vendor whose face stays
the same from poster to poster. That is style and identity — the next chapter.

#recap((
  [Diffusion models *fuse* a prop into the person beside it; a cart at a vendor is
  the "person + cargo fusion" the analyzer warned about, wearing a new costume.],
  [`objects: <prop>` declares a prop that must render *separately* — the compiler
  reserves it its own region with a gap from the figures, so a cart stays a cart.],
  [`relate: <a> <relationship> <b>` (e.g. `customer facing vendor`) states how two
  figures interact, so they orient toward each other instead of ignoring the moment.],
  [`control-generate-figure-contacts` pins limb-level touches (an arm around a
  shoulder) when figures physically meet — not needed when a cart sits between them.],
  [Some arrangements sit at the *ceiling* — coupled wheeled vehicles, deep seated
  poses in a crowd; the analyzer flags these early, and the honest fix is a simpler
  plan or a higher rung, not a fight with the render.],
))
