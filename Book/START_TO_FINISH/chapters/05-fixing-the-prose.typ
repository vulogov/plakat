#import "../design.typ": *
#chapter(number: 5, title: "Fixing the Prose — and Remembering Why")

#dropcap("T")he analyzer tells you what is wrong. Some of what it finds is
judgment — *which* subject to cut, how to reframe a crowd — and stays your call.
But some of it is mechanical: a rare word the model won't know, a negation in the
positive prompt, a clause of pure token bloat. plakat can apply those safe fixes
for you, without touching a thing it isn't sure about, and — for a poster you will
polish over many sessions — it can *remember why each change was made*. That memory
is where a small, quiet tool called smysl earns its place.

#section("Applying the safe fixes: `--analyze --fix`")

Add `--fix` to an analyze run and plakat proposes safe, high-confidence text edits,
finds each offending phrase in the exact source file it lives in, backs that file up
first, applies the edit in place, and reports what changed where. Anything it can't
fix safely — splitting an over-stuffed scene, choosing which figure is the hero — it
leaves for you, clearly labelled.

Suppose our lane picked up a couple of these mechanical problems: a fanciful name
for the cart and a bloated boilerplate tail.

#screen(caption: "The prose with fixable problems")[```
  name: market-lane
  A narrow cobbled lane at night. A food vendor works a gastro-nocturne
  provisions barrow under warm string lights; a customer leans in to order.
  Soft fog beyond. Warm amber glow, masterpiece, best quality, ultra
  detailed, 8k, trending on artstation.
  negative: no daylight
```]

#screen(caption: "compile --analyze --fix")[```
  $ plakat compile prompts.txt --analyze --fix
    scene 'market-lane' — feasibility 5/10 · MODERATE RISK
    ▸ RARE NAME (med): "gastro-nocturne provisions barrow" is invented; the
      model will hallucinate. fix → "wooden food cart"
    ▸ TOKEN BLOAT (low): "masterpiece, best quality, ultra detailed, 8k,
      trending on artstation" adds tokens, dilutes the scene. fix → cut
    ▸ NEGATION-IN-POSITIVE (low): "no daylight" belongs in negative.

    --fix  applying safe auto-fixes to the source prose…
    ✓ prompts.txt  (backup → prompts.txt.1)
        rare/invented name → "gastro-nocturne provisions barrow"
          → "wooden food cart"
        token bloat → "masterpiece, best quality, ultra detailed, 8k,
          trending on artstation" → (removed)
    → re-run --analyze to confirm the grade improved.
```]

Three things happened, and each matters:

#subsection("It backed up your original")

Before editing, plakat copied `prompts.txt` to `prompts.txt.1` (and `.2`, `.3` on
later runs). Your words are never destroyed; you can always diff against the backup
or restore it.

#subsection("It edited the right file")

If your prose is split across files with `@include` (Chapter 6), `--fix` traces each
phrase to the *actual* file it lives in and edits that one — not a copy, not the
wrong include. The offending clause is found verbatim and replaced verbatim.

#subsection("It respected your language")

plakat's prose is often multilingual — a Russian scene description with English
negative terms is common. `--fix` repairs a phrase *in its own language*: a Russian
phrase is fixed in Russian, an English one in English. It never quietly translates
your words while "fixing" them.

#warn(label: "What --fix will NOT touch")[
  `--fix` only applies edits it is confident are safe: renaming a rare word, cutting
  boilerplate, moving a negation. It never restructures your scene, never invents
  content, never edits a component name or a directive. Splitting an over-stuffed
  poster, or deciding the vendor is the hero and the customer the bystander, remains
  yours — it is reported under "needs your hand," not silently changed.
]

#section("Remembering why: the smysl corpus")

Here is a problem that only shows up on a project you return to. You polish the
prose over three evenings. On the fourth you look at the file and cannot remember
why the cart is a "wooden food cart" and not the vivid name you first wrote — or
whether you already dealt with that crowd, or why a clause you liked is gone. The
edits survive; the *reasoning* evaporated into a terminal you closed days ago.

plakat can keep that reasoning. When `--fix` runs, it records each decision as a
small provenance note in a sidecar file — a *smysl corpus* — beside your prose.

#term("smysl corpus")[
  A sidecar file (`<name>.smysl`) that records the *why* behind your polish loop: a
  `smysl` document is a light AI→AI→Human knowledge format. In plakat it captures
  each applied fix as a claim *grounded in* the risk it addressed, plus the
  budget decisions made at compile time. It is a memory of the polish loop — not a
  file you author. You keep writing prose; the corpus records the reasoning
  underneath it.
]

#callout(label: "smysl is a sidecar, not a new syntax")[
  You never write smysl by hand, and it never replaces your prose or your scenario.
  It is a record that accumulates *underneath* the authoring loop, so the polish
  history is queryable later. If you never look at it, nothing changes about how you
  work. This is the only role smysl plays in plakat — the memory of prompt
  polishing.
]

The corpus is *born* at the fix step, because that is where decisions accumulate,
and it *grows* across runs rather than being overwritten — polish the prose again
next week and the new decisions are merged in beside the old, with identical
decisions de-duplicated automatically.

#screen(caption: "The corpus written beside the prose")[```
  $ plakat compile prompts.txt --analyze --fix
    …
    ✓ prompts.txt  (backup → prompts.txt.1)
    smysl corpus → prompts.smysl  (2 fix→finding record(s), 0 open)

  $ cat prompts.smysl
  @finding f/risk-3eh4u { status: speculative }
  ~ rare/invented object name — model will hallucinate

  @claim c/fix-9a2kd { status: derived, grounds: [f/risk-3eh4u] }
  ~ replace "gastro-nocturne provisions barrow" with "wooden food cart"
```]

Read that middle line the way plakat intends it: the *fix* ("use wooden food cart")
exists *because of* the *finding* ("rare name — the model will hallucinate"). The
edit is grounded in the risk it answered. That grounding is the whole point — three
weeks later you can ask *why* a phrase is the way it is and get an answer, not a
shrug.

#section("Asking the corpus: `compile --trace`")

Because the reasoning is recorded, you can query it. `compile --trace "<phrase>"`
answers a question every returning author asks: *why is this in — or gone from — my
prompt?* It runs no model and renders nothing; it reads the corpus and the scene and
explains.

#screen(caption: "Why did this change?")[```
  $ plakat compile prompts.txt --trace "wooden food cart"
    2 unit(s) mention "wooden food cart":

    • c/cart [claim, Speculative]
        a small wooden food cart under a warm string-lit canopy
    • c/fix-9a2kd [claim, Derived]
        replace "gastro-nocturne provisions barrow" with "wooden food cart"
        ← grounded in f/risk-3eh4u
             (rare/invented object name — model will hallucinate)
```]

The trace shows both where the phrase lives in your scene *and* the decision that
put it there. When we reach budget-fitting in Chapter 8, the same `--trace` will
also explain why a phrase was *dropped* to fit a model's token limit — the corpus
records the budget decisions too. That is its whole ambition: to make the polish
loop's history answerable, so a poster you return to after a month still makes
sense.

#section("Where the loop has taken us")

We have written a scene in prose, analyzed it before spending a render, and fixed
its safe problems while keeping a record of why. The prose now reads clean and
LOW RISK. That is the moment to leave authoring and *compile* — to turn this proven
prose into a scenario and finally make the image. That is Part III.

#recap((
  [`compile --analyze --fix` applies only the *safe*, mechanical fixes — renaming a
  rare word, cutting boilerplate, moving a negation — backing up your original to
  `prompts.txt.N` first and reporting exactly what changed where.],
  [It edits the right source file (tracing phrases through `@include`s) and repairs
  each phrase *in its own language*; structural changes stay yours, reported under
  "needs your hand."],
  [The *smysl corpus* (`<name>.smysl`) records the *why*: each fix as a claim
  grounded in the risk it answered. It is a sidecar memory of the polish loop, not a
  syntax you author — the only role smysl plays in plakat.],
  [The corpus is born at `--fix` and *accumulates* across runs (identical decisions
  de-duplicated), so a long polish history is preserved rather than overwritten.],
  [`compile --trace "<phrase>"` answers "why is this in — or gone from — my prompt?"
  with no render, reading the corpus to explain both scene content and decisions.],
))
