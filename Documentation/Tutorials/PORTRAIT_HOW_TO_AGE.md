# How to age a portrait — interpolating between photos at different ages

This tutorial shows you how to render a person at any age between (or
near) ages where you have reference photos. The trick is plakat's
weighted multi-reference portrait feature: you give plakat photos of
the same person at different ages, set the weights to bias the merge
toward the age you want, and let the prompt nudge the rendering toward
age-appropriate features.

## What you'll learn

- Why multi-reference portrait can approximate aging at all
- How to gather and prepare reference photos for age interpolation
- How to walk the weights from "young" through "middle" to "old"
- How to combine the merge with prompt-level age cues
- The honest limits — what this does and doesn't do

## Before you start

- Work through `GENERATE_TUTORIAL.md` and `PORTRAIT_TUTORIAL.md` first.
  This tutorial assumes you've made at least one single-photo portrait.
- Read `PORTRAIT_TUTORIAL.md` section 11 ("Merging multiple reference
  photos") for the multi-reference basics.
- Have **2-4 photos of the same person at different ages**. Head-and-
  shoulders crops with good lighting and a roughly forward-facing pose
  work best.
- For the strongest results, configure FaceID (see
  `Documentation/PERSONA.md` "FaceID setup"). ArcFace embeddings are
  more identity-discriminative than Plus-Face's CLIP-H, so age
  interpolation reads more cleanly. Plus-Face works but is softer.

---

## 1. Why this works at all

Plakat's identity adapters reduce a face photo to a fixed-size
embedding — a single vector (for FaceID) or a small set of tokens (for
Plus-Face) — that encodes "who this person is" in a form the UNet can
condition on. That embedding mixes many signals: bone structure,
skin texture, hair, eye shape, *and* age cues like wrinkles, hairline,
softness around the jaw.

When you merge two photos of the same person at different ages,
plakat does a weighted sum of those embeddings in their natural space
(then renormalizes for FaceID, since ArcFace lives on the unit
sphere). The result is a *blended identity* that contains some of each
photo's age signal. Heavy-weighting the older photo pulls the
embedding toward older-face features; heavy-weighting the younger one
pulls it the other way.

It's **not** an age-transformation model. There's no learned mapping
from "face at 25" to "face at 50." What you're doing is interpolating
between two points in the identity embedding space — and because both
points belong to the same person, the line between them runs through
plausible representations of *that person* at intermediate ages.

The closer your reference photos are to the same identity (same
person, just different ages), the more this works. Two photos of
different people would interpolate between *people*, not ages — that's
the next tutorial, `PORTRAIT_CHILD_PHOTO.md`.

---

## 2. Picking reference photos

Quality of inputs is the largest factor.

**Same person, different ages.** Two photos at age 25 and 55 give you
the widest interpolation range. Three or four at intermediate ages
(25, 35, 45, 55) give you finer control and a more stable merge.

**Consistent framing.** All photos should be similar crops (head-and-
shoulders, mostly forward-facing). Wildly different angles or zoom
levels confuse the identity encoder. If the inputs are mismatched,
crop them yourself before feeding to plakat, or configure SCRFD so
each photo gets aligned independently.

**Comparable lighting and quality.** Major lighting shifts between
photos (one studio shot + one dim phone selfie) bleed into the
embedding too. The merge picks up "good lighting" features from the
studio shot and "muddy" features from the phone shot. Photos taken
under similar conditions interpolate more cleanly.

**Reasonable resolution.** 512px+ on the short side. Anything smaller
loses detail before the 224² (CLIP-H) or 112² (ArcFace) preprocess.

**For practice:** plakat ships with two Rembrandt self-portraits at
different ages that work as a stand-in if you don't have your own
photos handy:

- `tests/fixtures/style_catalog/oil_painting/01_rembrandt_self_portrait.jpg`
  (Rembrandt in his prime, ~mid-life)
- `tests/fixtures/style_catalog/holdout/oil_painting_rembrandt63.jpg`
  (Rembrandt at age 63, near end of life)

These work because they're the same person, painted by himself,
decades apart — a real-world age interpolation pair, just rendered as
paintings rather than photographs.

---

## 3. Baseline runs — single-photo at each age

Before merging, see what plakat produces from each photo alone. This
gives you a baseline for what each end of the age spectrum looks like.

```bash
# Young/mid-life reference
plakat portrait "a portrait of a man with thoughtful expression" \
    --photo ./refs/person_age_25.jpg \
    --face-strength 0.8 \
    --steps 30 \
    --out ./out/baseline_young

# Older reference
plakat portrait "a portrait of a man with thoughtful expression" \
    --photo ./refs/person_age_55.jpg \
    --face-strength 0.8 \
    --steps 30 \
    --out ./out/baseline_old
```

Compare the two outputs side by side. The face shape, eyes, mouth, and
overall identity should clearly come from each respective photo. If
not, your identity strength is too low or the reference photos aren't
giving the encoder enough to work with — fix the inputs before
proceeding to the merge.

---

## 4. The even merge — visual midpoint

Now run the same prompt with both photos at equal weight:

```bash
plakat portrait "a portrait of a man with thoughtful expression" \
    --photo ./refs/person_age_25.jpg \
    --photo ./refs/person_age_55.jpg \
    --face-strength 0.8 \
    --steps 30 \
    --out ./out/merge_50_50
```

This produces a face that visually averages the two inputs. Because
both inputs are the same person, the average looks like *that person*
— but at a perceived age somewhere between the two reference ages.
It's not exactly the midpoint (that depends on which features the
encoder captured most strongly), but it's recognizably "the same
person, at some age in between."

The plakat log will confirm the merge:

```
✓ identity encoded
```

(no per-photo breakdown is printed for brevity; only the total photo
count appears in verbose mode).

---

## 5. Walking the weights — younger, older, in between

The fun part: by varying the weights, you walk along the embedding-
space interpolation line.

```bash
# Mostly young — 80/20
plakat portrait "..." \
    --photo ./refs/person_age_25.jpg:0.8 \
    --photo ./refs/person_age_55.jpg:0.2 \
    --face-strength 0.8 --out ./out/age_lean_young

# Even — 50/50
plakat portrait "..." \
    --photo ./refs/person_age_25.jpg:0.5 \
    --photo ./refs/person_age_55.jpg:0.5 \
    --face-strength 0.8 --out ./out/age_midpoint

# Mostly old — 20/80
plakat portrait "..." \
    --photo ./refs/person_age_25.jpg:0.2 \
    --photo ./refs/person_age_55.jpg:0.8 \
    --face-strength 0.8 --out ./out/age_lean_old
```

What you should see: a progression in the perceived age of the
generated portrait. The "lean young" output looks closer to the young
reference; "lean old" closer to the old reference; "midpoint" is the
in-between identity from step 4.

The weights are proportions, normalized to sum to 1.0. `0.8 / 0.2` is
exactly equivalent to `4.0 / 1.0` or `80 / 20` — plakat renormalizes
internally.

---

## 6. Adding prompt-level age cues

The merge handles *identity*; the prompt should handle *age cues* in
the rendering. The two work together.

Without an age cue in the prompt, the model defaults to whatever it
considers typical (often a middle-aged interpretation, biased by
training-set distribution). Adding explicit age descriptors steers the
final rendering:

```bash
# "Young" prompt + heavily young-weighted merge
plakat portrait "a portrait of a man in his late twenties, smooth skin, dark hair" \
    --photo ./refs/age_25.jpg:0.8 \
    --photo ./refs/age_55.jpg:0.2 \
    --face-strength 0.8

# "Old" prompt + heavily old-weighted merge
plakat portrait "a portrait of a man in his late fifties, fine wrinkles, greying hair" \
    --photo ./refs/age_25.jpg:0.2 \
    --photo ./refs/age_55.jpg:0.8 \
    --face-strength 0.8

# Middle-aged: midpoint merge + middle-aged prompt
plakat portrait "a portrait of a man in his early forties" \
    --photo ./refs/age_25.jpg:0.5 \
    --photo ./refs/age_55.jpg:0.5 \
    --face-strength 0.8
```

The combination is what does the work:
- The **merge** establishes whose identity is being drawn.
- The **prompt** establishes how old that person should look.

Mismatches between merge weight and prompt age (e.g., heavily-young
merge + "in his sixties" prompt) sometimes produce interesting hybrids
— the model tries to render an aged version of the young embedding —
but usually one signal wins and the other is muted. Keep them aligned
for predictable results.

---

## 7. With three or more photos — richer interpolation

Two photos give you a 1D line in embedding space. Three or more give
you a richer mix and reduce single-photo bias:

```bash
plakat portrait "a portrait, neutral expression" \
    --photo ./refs/age_25.jpg:0.4 \
    --photo ./refs/age_40.jpg:0.4 \
    --photo ./refs/age_55.jpg:0.2 \
    --face-strength 0.8
```

This emphasizes the 25 + 40 photos (each 0.4) with a 0.2 contribution
from 55. The output reads as "around 35-ish" — biased toward
younger but with some older-face features mixed in.

Auto-weight fill works the same as in any multi-photo command. Without
a weight on the 40 photo, it auto-fills the remainder:

```bash
# age_25 explicit 0.5, age_55 explicit 0.3, age_40 auto-fills to 0.2
plakat portrait "..." \
    --photo ./refs/age_25.jpg:0.5 \
    --photo ./refs/age_40.jpg \
    --photo ./refs/age_55.jpg:0.3
```

---

## 8. FaceID vs Plus-Face for aging

FaceID is the recommended strategy for age interpolation:

| Strategy | Identity discrimination | Age interpolation quality |
|---|---|---|
| `plus-face` (default) | Moderate; CLIP-H captures general visual features | Soft — age signals diffuse, midpoint blurs |
| `faceid` | High; ArcFace was trained for face recognition | Cleaner — age signals are localized in the 512-d embedding and interpolate distinctly |

Switch with `--identity faceid` (requires ArcFace weights — see
`Documentation/PERSONA.md` "FaceID setup"):

```bash
plakat portrait "a portrait of a man in his forties" \
    --identity faceid \
    --photo ./refs/age_25.jpg:0.5 \
    --photo ./refs/age_55.jpg:0.5 \
    --face-strength 0.85
```

FaceID also benefits more from per-photo SCRFD alignment — set
`PLAKAT_SCRFD_HF` if you haven't already.

---

## 9. Scenarios — sweep a range of ages in one batch

For producing a series of "this person at 25, 30, 35, …, 55" in one
run, define each weight combination as its own task in a scenario:

```hjson
{
    model: sd15
    base: 768
    aspect: 3:4
    out: ./out/aging_sweep
    enhancer: deepseek

    # The persona uses ALL the age references with auto-equal weight.
    # Each task overrides which photos appear in its persona (one way
    # is to define multiple personas with different weights; another
    # is to use the simpler approach below).
    personas:
    [
        {
            name: at_30
            photos:
            [
                { path: ./refs/age_25.jpg, weight: 0.75 }
                { path: ./refs/age_55.jpg, weight: 0.25 }
            ]
            face-strength: 0.8
        }
        {
            name: at_40
            photos:
            [
                { path: ./refs/age_25.jpg, weight: 0.5 }
                { path: ./refs/age_55.jpg, weight: 0.5 }
            ]
            face-strength: 0.8
        }
        {
            name: at_50
            photos:
            [
                { path: ./refs/age_25.jpg, weight: 0.25 }
                { path: ./refs/age_55.jpg, weight: 0.75 }
            ]
            face-strength: 0.8
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
            prompt: "soft even lighting"
        }
    ]

    tasks:
    [
        {
            name: portrait_at_30
            scene: studio
            weather: soft
            prompt: "a portrait of a person in their early thirties"
            personas:
            [
                at_30
            ]
        }
        {
            name: portrait_at_40
            scene: studio
            weather: soft
            prompt: "a portrait of a person in their early forties"
            personas:
            [
                at_40
            ]
        }
        {
            name: portrait_at_50
            scene: studio
            weather: soft
            prompt: "a portrait of a person in their early fifties"
            personas:
            [
                at_50
            ]
        }
    ]
}
```

Run it with `plakat scenario aging_sweep.hjson` and you get three
portraits of the same person at three apparent ages, in one batch.

---

## 10. Limits — what this doesn't do

Be honest with yourself about what's happening here.

**This is not age transformation.** Plakat doesn't have an aging model
that learned what 5 years of age does to a face. It interpolates
between identity embeddings you supplied. The "ages" you can produce
are bounded by the ages in your input photos. You can extrapolate a
bit (lean further toward one weight) but the further you go from your
inputs, the less plausible the output becomes.

**Real human aging isn't linear in embedding space.** Faces change at
different rates across decades. Skin softens unevenly, hair recedes
non-uniformly, bone structure shifts subtly. A 50/50 merge between age
20 and age 60 doesn't produce "age 40" — it produces a face that
visually averages the two inputs, which only approximates "age 40" if
those input ages happen to land symmetrically around 40 in feature
space (rarely true).

**The prompt does heavy lifting.** Without "in their forties" or
similar in the prompt, the model often defaults to a younger or middle
rendering regardless of the merge. The merge sets *whose face*; the
prompt sets *how old that face looks*. Always pair the two.

**Garbage in, garbage out.** Bad inputs (wildly different framing,
wildly different lighting, low resolution) produce muddy merges. Take
the time to crop / re-light your inputs first if needed.

**Not a substitute for dedicated age tools.** If you need
"photorealistic aging" rather than "plausible portrait at a different
age," you want a model like SAM or a diffusion-based aging pipeline.
Plakat's tool is creative blending, not face forensics.

---

## 11. Recipe summary

For most use cases:

1. **Inputs:** 2-4 head-shots of the same person, similar framing,
   different ages.
2. **Strategy:** `--identity faceid` if you have ArcFace weights;
   otherwise the default `plus-face` is fine but softer.
3. **Alignment:** SCRFD if configured (recommended); otherwise tight
   head-and-shoulders crops manually.
4. **Weights:** start with equal (`/N` each), then bias toward the
   target age (e.g., 0.7 / 0.3 toward the closer reference).
5. **Prompt:** add explicit age vocabulary ("in their thirties",
   "soft wrinkles around the eyes", "greying at the temples").
6. **Face strength:** 0.7-0.9. Higher locks in identity at the cost
   of prompt adherence; lower lets the age prompt push harder.

Iterate with a fixed seed (`--seed 42`) so you can compare weight
sweeps without random variation muddying the picture.

---

## Where to next

- **Blending two different people into a "child" portrait** →
  `PORTRAIT_CHILD_PHOTO.md`
- **Full multi-reference portrait reference** →
  `PORTRAIT_TUTORIAL.md` section 11
- **Persona schema for scenarios** →
  `Documentation/PERSONA.md`, "Persona fields" → "photos"
- **The math behind embedding-space merging** →
  `Documentation/PERSONA.md`, "How merging works"
