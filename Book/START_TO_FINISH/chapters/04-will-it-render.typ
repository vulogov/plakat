#import "../design.typ": *
#chapter(number: 4, title: "Will It Even Render?")

#dropcap("H")ere is the scene that plays out for every image-maker sooner or later.
You write an ambitious prompt. You hit render. You wait — one minute, two, three —
watching the progress bar crawl. And what comes back is a mess: a figure fused into
a table, a cart with a person's arm growing out of it, six things fighting for the
centre of a frame that only had room for two. You burned the compute, and the
scene was doomed from the first word. You just didn't know it yet.

plakat has a command whose entire job is to tell you *before* you spend that
render. It reads your prose, reasons about what will probably fail, grades the
scene, and hands you specific fixes — and it renders nothing.

#section("The feasibility report: `compile --analyze`")

Let us deliberately over-reach, the way one does at the end of a good idea, and ask
the poster to hold everything at once.

#screen(caption: "An over-stuffed scene")[```
  name: market-lane
  A narrow cobbled market lane at night. A food vendor in an apron works a
  wooden cart while a customer leans in to order and two children run past
  and a cat sits on a barrel and a musician plays in the background, string
  lights and lanterns and a chalkboard menu and steam and reflections in the
  wet stones, a bustling crowd beyond, no empty space, everything happening
  at once.
```]

#figure_img("assets/03-overstuffed.png", "Image 03 — the over-stuffed scene the analyzer warns about: seven subjects fighting for one frame, figures fusing, detail dissolving into noise.")

Now, instead of compiling and rendering it, ask whether it *can* work.

#screen(caption: "plakat compile --analyze")[```
  $ plakat compile prompts.txt --analyze
    scene 'market-lane' — feasibility 3/10 · HIGH RISK

    ▸ OVER-STUFFED (high): 7+ distinct subjects (vendor, customer, two
      children, cat, musician, crowd) compete for one frame. SDXL reliably
      renders 2–3 focal subjects; the rest will blur or fuse.
      fix: cut to the vendor + one customer; imply the rest as "a few
      distant figures in the fog."

    ▸ PERSON+CROWD FUSION (med): "bustling crowd beyond" behind a busy
      foreground tends to grow extra limbs on the focal figures.
      fix: push the crowd fully into the background and out of focus.

    ▸ NEGATION-IN-POSITIVE (low): "no empty space" describes an absence;
      the model can't paint "no."  fix: describe density positively, e.g.
      "a full, lively lane," or move it to negative:.

    recommendation: reduce to two figures + the cart; re-run --analyze
    until the grade is LOW RISK before you compile.
```]

That report cost seconds and zero renders. It caught the three things that would
have wrecked the image: too many subjects, a crowd that fuses with the foreground,
and a "no" the model cannot paint. Each risk comes with a *concrete* fix, not a
scold.

#term("Feasibility grade")[
  `compile --analyze` scores a scene from 1–10 and labels it LOW / MODERATE / HIGH
  RISK, then lists the specific failure modes it predicts — over-stuffing, figure
  fusion, rare or invented object names, negation-in-positive, hard poses, count
  ambiguity, token bloat — each with a targeted fix. It writes nothing and renders
  nothing. It is a critic, run before the cost.
]

#section("Why this is worth a whole command")

The value of `--analyze` is not that it is clever; it is that it is *cheap and
early*. Consider the arithmetic. A HIGH-RISK scene on SDXL might take two minutes
to render and come back unusable, and you might try it three or four times —
re-rolling the seed, nudging a weight — before you accept that the *scene*, not the
seed, was the problem. That is ten minutes of waiting to learn what `--analyze`
tells you in five seconds.

#warn(label: "The wall you don't want to hit")[
  The worst outcome in image work is not a bad image — it is a *slow* bad image,
  followed by the realisation that the prompt was impossible for this model from the
  start. `--analyze` exists so you meet that wall as a sentence on your screen, not
  as three wasted renders and a sinking feeling.
]

#section("The polish loop")

`--analyze` is not a one-shot verdict; it is the companion to a loop. You read the
risks, edit the prose to answer them, and re-analyze — as many times as it takes to
bring the grade down. Only when the scene reads LOW RISK do you compile and spend a
real render.

Let us walk the loop on our poster. We take the analyzer's advice and cut the scene
to its heart: the vendor, one customer, the cart, the light.

#screen(caption: "The scene, trimmed — and re-analyzed")[```
  name: market-lane
  A narrow cobbled market lane at night. A food vendor works a wooden cart
  under warm string lights; a single customer leans in to order. Two iron
  lanterns light the foreground; soft blue fog and a few distant figures
  beyond. Warm amber glow, quiet and inviting.

  $ plakat compile prompts.txt --analyze
    scene 'market-lane' — feasibility 8/10 · LOW RISK

    ▸ note (low): two figures + a cart is well within SDXL's reliable range.
      For the tightest control of who stands where, consider control-generate
      (Ch. 9) — optional; the scene will render without it.

    recommendation: good to compile.
```]

From 3/10 to 8/10 by *removing*, not adding. This is the most common lesson the
analyzer teaches: over-stuffing is the default failure, and the fix is almost
always subtraction. The children, the cat, the musician, the crowd were all lovely
ideas; none of them survived contact with a frame that could hold two people and a
cart. The poster is stronger for their absence.

#section("Where analyze fits the arc")

Think of `--analyze` as the read-through you do before committing to a draft. It
does not write your scene and it does not render it; it tells you, honestly and
early, whether the scene you wrote is one this model can make. Run it whenever a
scene grows — after you add a figure, after you pile on detail, before you reach for
a bigger, slower model. It is the cheapest insurance in the whole toolchain.

But reading a report and hand-editing prose to answer it is still work, and some of
the fixes are mechanical — rename a rare word, delete a negation, trim a clause. The
next chapter shows how plakat can apply those safe fixes *for* you, back up your
originals, and — this is the part that matters for a long project — record *why*
each change was made, so the reasoning of your polish loop is never lost.

#recap((
  [`compile --analyze` reads your prose and predicts what will fail — over-stuffing,
  figure fusion, rare names, negation-in-positive, hard poses — grading the scene
  1–10 and labelling it LOW / MODERATE / HIGH RISK, all without rendering.],
  [Its value is being *cheap and early*: it turns "three wasted renders and a
  sinking feeling" into a five-second sentence you can act on.],
  [Use it as a *loop* — read the risks, edit the prose, re-analyze — and compile
  only once the scene reads LOW RISK.],
  [The most common cure is *subtraction*: over-stuffing is the default failure, and
  our poster went from 3/10 to 8/10 by cutting a crowd down to two figures and a
  cart.],
  [Analyze is the read-through before the draft; the next chapter automates its
  safe fixes and records the reasoning behind them.],
))
