# PERSONA-1 casting & rendering (RFC §11)

Geometry, details and prompts constrain *what a face looks like*. They do not by themselves produce a
face stable enough to recognise across renders. The **cast** does: it renders candidates, composites
the persona's details, scores them against the spec, keeps the best, and validates that they are *one
person* — then stores the reference set every later render anchors to.

## Casting

```
plakat persona cast alice.hjson --model sdxl --count 32 --keep-best 4 [--aesthetic]
```

1. Compile the spec for the casting family → prompt + negative.
2. Render `count` candidates at distinct seeds.
3. Composite the persona's details (§8.4) onto each — a reference set *without* the persona's mole
   pulls later renders away from it (§11.1). `--no-details` skips this.
4. Score each candidate against the spec (the calibrated geometric scalars + `eyes.color`).
5. Rank by spec conformance; `--aesthetic` adds LAION as a distant secondary key (a beautiful face
   that is the wrong face is a failure).
6. Keep the top `--keep-best`, embed each with ArcFace, and validate **identity coherence**: the worst
   pairwise cosine must exceed `0.50`, or the cast produced several different people and must be re-run
   with tighter conditioning. The coherence matrix is reported either way.

The result is a persona directory:

```
persona-alice/
  reference_set.json      # manifest: refs (image, seed, embedding, score, centroid cosine) + coherence
  references/ref_0.png …  # the kept reference images (most-representative first)
  candidates/…            # every rendered candidate, for inspection
```

**Scope today.** Candidates are rendered from the prompt and *selected* by the scorecard, then anchored
by the swap bridge at render time. **Geometry-ControlNet casting is wired** (`cast --geometry-control
depth|pose|off`, default `depth`, `--geometry-strength 0.55`): the §10 conditioning map drives an
SD-UNet ControlNet (sd15/sdxl only — the DiT families hold the attribute list in T5/Gemma instead) so
the authored proportions are realised via conditioning rather than competing for CLIP tokens. The
depth map is **framed as a head-and-shoulders bust** (`geometry::add_bust_base`) — a bare face-oval
floating on black is ambiguous enough that a weakly-bound SD1.5 renders it as an *object* (a spoon, a
wooden blade), so a neck + shoulder mound grounds the silhouette as a person. Each candidate currently
reloads the model — the resident scoring/render worker (§22) is deferred. Multi-view/expression sheets
(§11.2) are P5c.

## Coherence

`reference_set.json` records, per set: the unit-normalised centroid embedding, the mean and **minimum**
pairwise ArcFace cosine, and pass/fail vs the threshold. A set is only as coherent as its worst pair —
that minimum is the "did the cast produce one person?" check. Each reference's cosine to the centroid
is its default weight (the most representative faces dominate) and picks the **canonical** face the
render path swaps from.

## Rendering (Tier B, §11.5)

```
plakat persona render alice-persona/ --scene "a woman in a sunlit garden, photograph" --model <any>
```

The universal path: generate the scene on any family (the persona's appearance prompt merged in) →
detect the scene face (SCRFD) → swap the canonical reference face in → restore the swapped region at
gentle strength (identity-preserving) → run the detail compositing pass **after** the swap (a hard
ordering constraint: the swap replaces the face region wholesale and would destroy any mark composited
before it). `--no-restore` / `--no-details` opt out; `--spec` overrides the stashed spec. When the
scene render produces no detectable face after a few seeds, the render is left un-swapped and that is
reported.

## Tiers (§11.4)

| Tier | Mechanism | Requires |
|---|---|---|
| **A** | IP-Adapter-Plus-Face from the reference set + detail compositing | an adapter for the family (sd15 / sd21 / sdxl family) |
| **B** | native generation → face swap from the reference set → restore → detail compositing | nothing family-specific (**universal**) |
| **C** | a baked per-base adapter (TI / LoRA) from the reference set + detail compositing | a trainer for the base (`persona bake`, forthcoming) |

`persona render --tier auto` (the default) picks **A** where a face adapter exists, else **B**; `--tier B`
forces the universal swap path, `--tier A` uses the adapter (falling back to B if none exists). Detail
compositing is tier-independent — the small distinguishing features that make a persona feel specific
work identically on every family, because they never go through a sampler. **Tier C (baking) is
deferred to P6.**

## Render robustness

A text-prompted portrait fails in a handful of characteristic ways; the render path guards each:

- **Framing.** A bare "portrait photograph" lets an SD-UNet zoom to an extreme face-macro that overflows
  the frame (a single eye + cheek). `compile::framing_guard` emits a framing-aware crop phrase
  (headshot → "head-and-shoulders, the whole head in frame with headroom, centred") plus anti-macro
  negatives, on both Tier A and Tier B.
- **No-face retry (both tiers).** A bad seed can drive the render to a non-photo with no detectable
  face — an empty scene (Tier B) or a stylised tiled mosaic (Tier A). Both tiers now retry up to 3 seeds
  until the render contains a face, rather than shipping the garbage.
- **Anti-text.** An SD-UNet scrawls gibberish signage/captions on plain backgrounds (the same tendency
  that seeds "object + handwritten notes" hallucinations); the identity-render negatives suppress
  `text, watermark, signature, letters, writing, caption`.
- **Occlusion-aware jewelry.** Before compositing an earring/stud, the pass probes the render at the
  resolved ear anchor; if those pixels read as hair or shadow rather than bare cheek skin
  (`reads_as_skin`), the piece is culled and reported (`ear occluded`) instead of pasted over a curtain
  of hair. Facial sites (nostril/lip) are always on skin, so only ear sites are gated.

## Rejection sampling (§12.3)

`persona cast --min-score <s>` keeps rendering (up to `--max-attempts`) until `--keep-best` candidates
score at least `s`, instead of rendering a fixed `--count`. Candidates below the bar are reported. This
trades compute for a higher-quality reference set when the family's hit rate is low.

## Body identity is out of scope (§11.7)

Every identity mechanism here is **face-only**: there is no body ArcFace, no body-reference adapter, no
body swap. `figure` attributes are conditioned through the pose skeleton, silhouette and prompt — weak
signals — and recoverable only by baking. Body-*sited* details (forearm tattoo, wrist jewelry) are the
exception: they composite against the body skeleton and work as well as body-landmark detection does.
