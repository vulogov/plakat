# `plakat persona` tutorial

A hands-on pass through the whole pipeline. It uses the committed corpus persona
[`corpus/mira.hjson`](../corpus/mira.hjson); [`corpus/PERSONA_CORPUS.md`](../corpus/PERSONA_CORPUS.md)
runs everything below over two personas on sd15 + sd35 in one script.

Build the release binary first — debug diffusion is ~50× slower:

```sh
cargo build --release
alias plakat=./target/release/plakat
export PLAKAT_OOM_GUARD_GB=0        # the macOS free-page guard mis-fires under render loops
```

## 1. Author a spec (no weights)

Scaffold, then edit — or drive the Q/A interview:

```sh
plakat persona new alice.hjson --name alice --age 30 --depth standard   # a valid partial spec
plakat persona interview alice.hjson --tui                              # or the interactive TUI (§17)
plakat persona lint alice.hjson                                         # schema · ranges · contradictions
```

The TUI shows a **live wireframe** that moves as you drag a slider — the geometry engine is CPU-cheap,
so structural authoring needs no GPU and works over SSH (braille fallback). For scripting or CI, replay
a flat answer map instead:

```sh
plakat persona interview alice.hjson --answers corpus/mira-answers.hjson   # deterministic
```

## 2. See what it resolves to (no weights)

```sh
plakat persona show corpus/mira.hjson --model sd15     # compiled prompt + salience + grade badges
plakat persona geometry corpus/mira.hjson --out geo --calibrate sd15   # the conditioning maps
```

`geometry` writes mesh / wireframe / depth / pose-skeleton / region-mask / dentition / figure maps.
`--calibrate <model>` pre-distorts the deformation through that family's response curves so a requested
scalar *lands* at its value (§13.2).

## 3. Cast a reference set (weights)

Render candidates, composite the persona's details onto them, score against the spec, keep the best,
and validate that they are one person:

```sh
plakat persona cast corpus/mira.hjson --model sd15 --count 12 --keep-best 4 --out persona-mira
```

Reads `persona-mira/reference_set.json` — the kept references (with ArcFace embeddings + scorecards)
and the identity-coherence matrix. If the worst pairwise cosine is below threshold, the cast produced
several different people; raise `--count` or rely on the swap bridge (next step) to unify identity.

## 4. Render into a scene (weights)

```sh
plakat persona render persona-mira --scene "in a sunlit garden, photograph" --model sd15 --out shot.png
```

The universal Tier-B path: generate the scene → swap the canonical reference face in → restore →
composite the persona's details **after** the swap. Works on any family (sd15, sd35, sdxl, …) because
identity comes from the swap, not the sampler. Two personas in one frame:

```sh
plakat persona render persona-mira --with persona-idris --scene "two friends at a cafe" --model sd15 --out pair.png
```

Each persona is assigned its own figure and swapped only into its own face; a low-confidence face is
left absent, not mis-attributed (§14.2).

## 5. Measure and repair (weights)

```sh
plakat persona verify corpus/mira.hjson --image shot.png --model sd15     # the scorecard
plakat persona repair corpus/mira.hjson --image shot.png --attr eyes.color --model sd15 --out fixed.png
```

`verify` reports per-attribute pass/fail (geometry scalars vs the prior, colour ΔE, mark presence +
position, beard/glasses detection) and a weighted aggregate. `repair` fixes just the named attribute —
a **detail** recomposites, a **surface** attribute is inpainted over its region and kept only if the
score improves, a **structural** one is reported as re-cast-only.

## 6. Edit an existing persona

```sh
plakat persona diff alice.hjson alice-v2.hjson   # what does this change cost?
```

Changing hair colour is a surface edit (repair in place); changing eye spacing is structural (a
re-cast). `diff` tells you before you spend the compute. See [`PERSONA_LEXICON.md`](PERSONA_LEXICON.md)
for the class of every attribute.

## Where to go next

- [`PERSONA_DETAILS_HOWTO.md`](PERSONA_DETAILS_HOWTO.md) — authoring marks, jewelry, dentition.
- [`PERSONA_ANCHORS.md`](PERSONA_ANCHORS.md) — the anchor vocabulary for placing details.
- [`PERSONA_CASTING.md`](PERSONA_CASTING.md) — tiers, coherence, and baking (`persona bake`).
- [`PERSONA.md`](PERSONA.md) — the command + layer reference.
