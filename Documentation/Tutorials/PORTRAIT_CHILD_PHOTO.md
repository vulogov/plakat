# Creating a child's portrait from two parents' photos

This tutorial walks you through blending two parent photos into a
plausible-child portrait using plakat's weighted multi-reference
portrait feature. You'll learn how the merge works, how to set
weights, how the prompt does most of the age work, and — importantly
— what this technique honestly is and isn't.

## What you'll learn

- How multi-reference portrait can blend two people into a single
  "average" identity
- How to combine parent merging with child-appropriate prompt cues
- How weight skew produces "looks more like mom" or "looks more like
  dad" results
- The honest limits — this is creative blending, not genetic
  prediction

## Before you start

- Work through `GENERATE_TUTORIAL.md` and `PORTRAIT_TUTORIAL.md`.
- Read `PORTRAIT_TUTORIAL.md` section 11 ("Merging multiple reference
  photos") for the multi-reference basics.
- Optionally read `PORTRAIT_HOW_TO_AGE.md` — that tutorial covers the
  same merging mechanism but for a single person at different ages;
  this one applies the same machinery to two different people.
- Have **2 photos**: one of each parent, head-and-shoulders, similar
  framing, good lighting, forward-facing.
- For the strongest results, configure FaceID (see
  `Documentation/PERSONA.md` "FaceID setup"). ArcFace's
  identity-discriminative embedding interpolates between two distinct
  people much more cleanly than Plus-Face's CLIP-H.

---

## 1. How this works

Plakat's identity adapters turn each face photo into an *identity
embedding* — a fixed-size feature vector or token set capturing
"who this person is" in a form the UNet can condition on.

When you merge two photos of **different** people with equal weight,
plakat does a weighted sum of those embeddings (renormalized to unit
length for FaceID since ArcFace lives on the unit sphere). The result
is a *point in identity space* that's halfway between the two parents'
identities. The UNet sees this merged identity as a single coherent
input and tries to render a face that satisfies it.

Because identity space encodes facial geometry, proportions, eye
shape, mouth, jawline, hair, and skin tone, the midpoint between two
parents looks like an "average" of those features. Add a "young child"
prompt and the model renders that average identity as a young face.

Combine the two and you get something that looks like a plausible
offspring of those two parents — at the age you described.

### What this isn't

This is **not** genetic prediction. Real children don't inherit
visual averages; they inherit specific gene combinations that often
produce one parent's nose with the other's eyes, etc. Plakat doesn't
model heredity at all. What it does is feature blending in image
space, which happens to roughly mimic what an "average child" might
look like — visually plausible but not biologically meaningful.

If you want a creative tool for what-if visualizations, family-tree
illustrations, fictional characters, or "average face" composites:
this works well. If you want a forensic prediction of how a real
unborn child will look: stop here, this isn't the tool.

---

## 2. Picking parent photos

Both photos should be at comparable quality. Mismatched inputs bleed
into the merge — if one parent photo is crisp and forward-facing and
the other is grainy and angled, the merge picks up the framing
differences as identity signals.

Checklist for each parent photo:

- **Head-and-shoulders crop.** Full-body photos waste pixels on
  features the encoder ignores. Crop to roughly head + upper torso
  before feeding to plakat.
- **Forward-facing.** Profile shots produce weak embeddings. Both
  parents looking roughly at the camera works best.
- **Good lighting on the face.** Heavy shadows confuse the encoder.
  Soft, even lighting from the front or side is ideal.
- **Similar resolution.** Both 1080p or both 4K is fine. One being 4K
  and the other being 320p produces a lopsided merge dominated by
  the higher-resolution photo's features.
- **Similar age.** If one parent photo is at 25 and the other at 55,
  the merge captures their age difference as much as their identity
  difference. Pick photos taken within ~5 years of each other for
  cleanest results.

---

## 3. The baseline — single-photo of each parent

Before merging, see what plakat produces from each parent alone. This
calibrates your expectations and reveals input quality issues before
the merge step.

```bash
# Parent A baseline
plakat portrait "a portrait, plain backdrop" \
    --photo ./refs/mom.jpg \
    --face-strength 0.85 \
    --steps 30 \
    --out ./out/baseline_mom

# Parent B baseline
plakat portrait "a portrait, plain backdrop" \
    --photo ./refs/dad.jpg \
    --face-strength 0.85 \
    --steps 30 \
    --out ./out/baseline_dad
```

Each output should clearly resemble its respective parent. If not,
the identity strength is too low, or the photo isn't giving the
encoder enough to work with. Fix it before proceeding.

---

## 4. The 50/50 merge — an adult midpoint

Now merge both parent photos at equal weight, with an age-neutral
adult prompt:

```bash
plakat portrait "a portrait of an adult, plain backdrop, three-quarter view" \
    --photo ./refs/mom.jpg \
    --photo ./refs/dad.jpg \
    --face-strength 0.85 \
    --steps 30 \
    --out ./out/merged_adult
```

The output is an adult who visually averages both parents. Eye shape,
nose proportions, jaw, hairline — all are blended. This is the
"adult version of the average child" — it's the same merged identity
you'll use for child portraits, just rendered at adult age.

Run this first to check the merge is doing something reasonable
before you switch to a child prompt. If the adult merge already
doesn't resemble either parent, the inputs need work.

---

## 5. The child portrait — same merge, different prompt

Now keep the same `--photo` setup but change the prompt to describe a
child:

```bash
plakat portrait "a portrait of a young child around 8 years old, soft features, gentle smile" \
    --photo ./refs/mom.jpg \
    --photo ./refs/dad.jpg \
    --face-strength 0.85 \
    --steps 30 \
    --out ./out/child_balanced
```

What changed: the merged identity is identical to step 4; only the
**rendering** age shifted to "young child." The model interprets the
merged identity as a child by combining the embedding (which provides
*whose features*) with the prompt (which provides *how old / what
context*).

You're using plakat's identity adapter for *who* and the diffusion
model's prompt-following for *age*. The interplay between the two
produces the child portrait.

### Tweaking the prompt for age + style

The prompt does heavy lifting. Vary it to explore different
interpretations:

```bash
# Toddler
plakat portrait "a portrait of a toddler around 3 years old, soft baby features" --photo ... --photo ...

# Pre-teen
plakat portrait "a portrait of a child around 10 years old, school photograph" --photo ... --photo ...

# Pre-teen with mood
plakat portrait "a candid portrait of a serious-looking 9-year-old, outdoor setting" --photo ... --photo ...
```

Each of these uses the same merged identity but renders it at a
different age and in a different context. The "average child"
identity is consistent; only the age cues and styling change.

---

## 6. Skewed weights — "looks more like mom" / "looks more like dad"

The 50/50 merge is the visual midpoint. Skewing the weights pulls the
identity toward one parent:

```bash
# Looks more like Parent A (mom): 0.7 / 0.3
plakat portrait "a portrait of a young child, school photo style" \
    --photo ./refs/mom.jpg:0.7 \
    --photo ./refs/dad.jpg:0.3 \
    --face-strength 0.85 \
    --out ./out/child_like_mom

# Looks more like Parent B (dad): 0.3 / 0.7
plakat portrait "a portrait of a young child, school photo style" \
    --photo ./refs/mom.jpg:0.3 \
    --photo ./refs/dad.jpg:0.7 \
    --face-strength 0.85 \
    --out ./out/child_like_dad
```

Weights are proportions, normalized to sum to 1.0. `0.7 / 0.3`,
`70 / 30`, and `7 / 3` are all equivalent.

A typical use of this is generating "a few different children" from
the same parent pair to explore a range of inherited-feature
distributions:

```bash
# Run three weight balances with the same seed so faces compare cleanly
SEED=42
for weights in "0.7 0.3" "0.5 0.5" "0.3 0.7"; do
    set -- $weights
    plakat portrait "..." \
        --photo ./refs/mom.jpg:$1 --photo ./refs/dad.jpg:$2 \
        --seed $SEED --steps 30 \
        --out ./out/sibling_$1
done
```

(Same seed = same composition / pose / lighting / scene; only the
identity merge differs across runs.)

---

## 7. Multiple "children" — varying the seed

If you keep the merge weights fixed and vary the seed instead, you
get different *individual children* who share the same parental
identity:

```bash
# Same weights, different seeds — visualize a "set of children"
for SEED in 100 200 300 400; do
    plakat portrait "a portrait of a young child, school photo style" \
        --photo ./refs/mom.jpg:0.5 --photo ./refs/dad.jpg:0.5 \
        --face-strength 0.85 \
        --seed $SEED \
        --out ./out/child_seed_$SEED
done
```

Each output is a different child rendered from the same merged
identity. They look like siblings — visibly sharing the merged
features, but with the individual variation diffusion sampling
introduces.

This is closer to "what could this family look like" than "what does
the specific child look like." A useful framing for storytelling,
character creation, or family-tree illustrations.

---

## 8. Configuration — what to set

Recommended setup for child-portrait generation:

| Setting | Recommendation | Rationale |
|---|---|---|
| `--identity` | `faceid` (with ArcFace weights configured) | Stronger identity preservation; cleaner blending between different people |
| Alignment | Pre-cropped head-shots, or SCRFD via `PLAKAT_SCRFD_HF` | Each parent face needs its own alignment when SCRFD is on |
| `--face-strength` | `0.75 – 0.9` | Lower lets the child-age prompt push harder; higher locks in identity at the cost of age-rendering flexibility |
| `--steps` | `30 – 40` | Faces benefit from more steps; child faces especially (skin/eye detail) |
| `--scheduler` | `euler-a` (default for portraits) | Smoother skin/eye rendering |
| `--aspect` | `3:4` (default) | Portrait framing |
| Seed | Fixed (e.g., `--seed 42`) when comparing weight skews | Lets you isolate the merge's effect from random variation |

For Plus-Face (the default, no ArcFace weights needed), everything
above still works — output is just softer. The merged identity is
less crisp because CLIP-H's penultimate hidden state is a more
diffuse representation than ArcFace's discriminative embedding. If
you don't have FaceID set up, expect children that share the
*style* of both parents (palette, framing, mood) more than their
specific facial geometry.

---

## 9. Scenarios — a "family album" batch

To produce multiple children, multiple weight skews, and multiple
seeds in one run:

```hjson
{
    model: sd15
    base: 768
    aspect: 3:4
    out: ./out/family_album
    enhancer: deepseek

    personas:
    [
        {
            name: child_balanced
            photos:
            [
                { path: ./refs/mom.jpg, weight: 0.5 }
                { path: ./refs/dad.jpg, weight: 0.5 }
            ]
            identity: faceid
            face-strength: 0.85
        }
        {
            name: child_like_mom
            photos:
            [
                { path: ./refs/mom.jpg, weight: 0.7 }
                { path: ./refs/dad.jpg, weight: 0.3 }
            ]
            identity: faceid
            face-strength: 0.85
        }
        {
            name: child_like_dad
            photos:
            [
                { path: ./refs/mom.jpg, weight: 0.3 }
                { path: ./refs/dad.jpg, weight: 0.7 }
            ]
            identity: faceid
            face-strength: 0.85
        }
    ]

    scene:
    [
        {
            name: studio
            prompt: "a studio portrait, plain backdrop, three-quarter view"
        }
    ]
    weather:
    [
        {
            name: soft
            prompt: "soft even lighting, warm tones"
        }
    ]

    tasks:
    [
        {
            name: balanced_child
            scene: studio
            weather: soft
            prompt: "a portrait of a young child around 8 years old, gentle expression"
            personas:
            [
                child_balanced
            ]
        }
        {
            name: child_resembling_mom
            scene: studio
            weather: soft
            prompt: "a portrait of a young child around 8 years old, soft features"
            personas:
            [
                child_like_mom
            ]
        }
        {
            name: child_resembling_dad
            scene: studio
            weather: soft
            prompt: "a portrait of a young child around 8 years old, sharper features"
            personas:
            [
                child_like_dad
            ]
        }
    ]
}
```

Run with `plakat scenario family_album.hjson` and the output tree has
three portraits — a balanced child plus two skewed versions —
generated in one pass.

You can extend this pattern further: define five personas at weights
0.9/0.1, 0.7/0.3, 0.5/0.5, 0.3/0.7, 0.1/0.9 and produce a "spectrum"
of children spanning "mostly mom" through to "mostly dad."

---

## 10. Honest limits

**This is feature blending, not genetic prediction.** Repeating from
the top because it matters: real children inherit specific
combinations, not visual averages. The output is a plausible-looking
face that statistically combines the parents' features — useful for
creative purposes, meaningless for biological inference.

**Output diversity is limited by the diffusion model.** SD 1.5 has
known biases in face generation (training-set distribution affects
features the model defaults to). A merged identity that's far from
the model's "common face" distribution may produce something
unexpected. SDXL is more flexible but heavier.

**Ethnicity blends can fail unevenly.** If the two parents have
visually distinct ethnicities, the model sometimes "rounds" toward
whichever ethnicity dominates the training set rather than producing
a true visual midpoint. There's no good plakat-level fix for this;
it's a diffusion model limitation. Document the output for what it
is, not what it claims to predict.

**Age extrapolation is bounded by the prompt.** Putting "newborn
baby" in the prompt with adult parent photos works less well than
"young child" — the further the requested age is from the model's
"plausible face" distribution, the noisier the output. Middle-of-the-
distribution ages (4-12 years) produce the most stable results.

**Ethical considerations.** This is creative tooling. Generating
"realistic photos" of fictional children of real people requires
thought:

- Don't use this to produce deepfake-style content of identifiable
  minors.
- Be transparent with anyone you share the output with: it's a blend,
  not a prediction.
- Subjects in the parent photos have a stake in how their merged
  identity is used. Get consent if the photos are of real, recognizable
  people.

---

## 11. Recipe summary

For most use cases:

1. **Inputs:** one tight head-shot per parent, comparable quality.
2. **Strategy:** `--identity faceid` with ArcFace weights (strongly
   recommended). Default `plus-face` works but is softer.
3. **Alignment:** SCRFD via `PLAKAT_SCRFD_HF`, or pre-crop both
   photos to consistent head-and-shoulders framing.
4. **Weights:** start with `0.5 / 0.5` for a balanced child. Skew to
   `0.7 / 0.3` for "looks more like X" variants.
5. **Prompt:** explicit child-age vocabulary ("8-year-old", "young
   child", "toddler"). Plus context cues ("school photo style",
   "soft lighting", "candid").
6. **Face strength:** 0.75–0.9. Too high overpowers the age prompt;
   too low loses the parental identity.
7. **Seed:** fix it (`--seed 42`) when comparing weight skews so
   you're seeing the merge's effect, not random variation.

Iterate. Few first-try runs produce exactly what you want. Plan to
generate 10-30 candidates and pick the ones that read most like a
plausible child of those parents.

---

## Where to next

- **Same-person aging (interpolating between photos of one person at
  different ages)** → `PORTRAIT_HOW_TO_AGE.md`
- **General multi-reference portrait reference** →
  `PORTRAIT_TUTORIAL.md` section 11
- **Persona schema for scenarios** →
  `Documentation/PERSONA.md`, "Persona fields" → "photos"
- **The embedding-space math behind weighted merging** →
  `Documentation/PERSONA.md`, "How merging works"
