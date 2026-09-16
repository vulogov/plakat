#import "../design.typ": *
#appendix(letter: "C", title: "The Polish Loop, on One Page")

When you are making your own poster and want the loop without the story, this is it:
the sequence that keeps you in prose and spends a render only on a scene that will
survive it.

#pipeline_arc()

#section("The loop, step by step")

#chord_table((
  chord_row("1 · Draft", "Write the scene in prose — sentences, components, a weight or two, a tight negative:."),
  chord_row("2 · Analyze", "compile --analyze. Read the grade and the risks. Aim for LOW RISK."),
  chord_row("3 · Fix", "compile --analyze --fix applies the safe edits (backup + smysl record). Handle the structural notes yourself."),
  chord_row("4 · Re-analyze", "Repeat 2–3 until LOW RISK. This is where the time is saved."),
  chord_row("5 · Compile", "compile --out scenario.hjson. Inspect with --explain / --dry-run if unsure."),
  chord_row("6 · Improve (optional)", "compile --improve raises the aesthetic score automatically and writes the winner into the scenario. The corpus is its memory — no re-marching. --improve-skip-good leaves already-good scenes alone."),
  chord_row("7 · Run", "scenario scenario.hjson. Tune seed / steps / guidance / scheduler."),
))

#section("The commands, in order")

#screen(caption: "A whole polish loop")[```
  $ plakat compile prompts.txt --analyze          # will it render?
  $ plakat compile prompts.txt --analyze --fix    # trim & repair safely
  $ plakat compile prompts.txt --analyze          # confirm LOW RISK
  $ plakat compile prompts.txt --out scenario.hjson --smysl
  $ plakat compile prompts.txt --improve-all      # (optional) auto-raise the score,
                                                  #   winner written back to the HJSON
  $ plakat scenario scenario.hjson                # spend the render
```]

#section("Why analyze first")

#budget_flow()

The whole point is arithmetic: a HIGH-RISK scene wastes minutes per render and often
several renders before you accept the scene, not the seed, was wrong. `--analyze`
turns that into a five-second sentence. Subtract before you add — over-stuffing is the
default failure, and the cure is almost always *fewer* subjects.

#section("smysl: the memory of the loop")

You never author smysl; it records *why*. Born at `--fix`, it accumulates across runs
and answers questions later.

#chord_table((
  chord_row("<name>.smysl", "The corpus beside your prose: each fix a claim grounded in the risk it answered, plus budget-pack decisions."),
  chord_row("compile --trace \"X\"", "Why is X in — or gone from — the prompt? Reads the corpus; renders nothing."),
  chord_row("Accumulates", "Repeated --fix merges new decisions with old; identical ones de-duplicate."),
  chord_row("Active memory", "Under --improve the corpus is a tabu list: every rewrite tried (kept or reverted) is remembered, so the loop never re-marches and can skip already-good scenes."),
))

#section("Reading a feasibility report")

#chord_table((
  chord_row("8–10 / LOW RISK", "Good to compile."),
  chord_row("4–7 / MODERATE", "Fixable — run --fix, then reduce the flagged subjects by hand."),
  chord_row("1–3 / HIGH RISK", "Over-stuffed or impossible for this model. Subtract, split, or climb the control ladder."),
))

#section("The control ladder, when analyze says climb")

#control_ladder()

#callout(label: "The one rule")[
  Stay in prose. Everything — analyze, fix, trace, compile, improve, control-generate,
  naturalize — serves the prose or the pixels. Climb the ladder only as far as the
  image needs, spend a render only on a scene that has earned it, and let the corpus
  remember why you made each choice.
]
