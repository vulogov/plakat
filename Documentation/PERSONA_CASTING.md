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

**Honest scope today (P5).** Casting is **Tier B** (universal): candidates are rendered from the prompt
and *selected* by the scorecard, then anchored by the swap bridge at render time. Geometry-ControlNet
casting (feeding the §10 conditioning map, a Tier-A bonus on SD1.5/2.1) needs the lower-level
`t2i::Request.controls` path and is a follow-on. Each candidate currently reloads the model — the
resident scoring/render worker (§22) is deferred. Multi-view/expression sheets (§11.2) are P5c.

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
scene render produces no detectable face, the render is left un-swapped and that is reported.

## Tiers (§11.4)

| Tier | Mechanism | Requires |
|---|---|---|
| **A** | IP-Adapter-Plus-Face / FaceID + landmark conditioning + detail compositing | an adapter port for the family |
| **B** | native generation → face swap from the reference set → restore → detail compositing | nothing family-specific (**universal**) |
| **C** | a baked per-base adapter (TI / LoRA) from the reference set + detail compositing | a trainer for the base |

Detail compositing is tier-independent — the small distinguishing features that make a persona feel
specific work identically on every family, because they never go through a sampler.

## Body identity is out of scope (§11.7)

Every identity mechanism here is **face-only**: there is no body ArcFace, no body-reference adapter, no
body swap. `figure` attributes are conditioned through the pose skeleton, silhouette and prompt — weak
signals — and recoverable only by baking. Body-*sited* details (forearm tattoo, wrist jewelry) are the
exception: they composite against the body skeleton and work as well as body-landmark detection does.
