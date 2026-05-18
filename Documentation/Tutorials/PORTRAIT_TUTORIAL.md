# Portraits — making images of specific people

This tutorial walks you from "I want a portrait" to "I have a photo of
my friend and want them rendered in a watercolor scene." No prior
text-to-image experience assumed.

## What you'll learn

- Why portraits get a dedicated `plakat portrait` command instead of
  just using `plakat generate`
- How to make a portrait without a reference photo (purely text-driven)
- How to make a portrait that **resembles a specific person** in a
  photo (identity preservation)
- The two identity strategies (`plus-face` and `faceid`) and when to
  use which
- How to put portraits into broader scenes via scenarios
- How to put **multiple** people into one scene

## Before you start

- Work through `GENERATE_TUTORIAL.md` first if you haven't generated
  any images yet — this tutorial assumes you know the basics of
  prompts, seeds, and steps.
- For text-only portraits: nothing extra needed.
- For identity-from-photo portraits: a head-and-shoulders photo of
  the subject. ~512 pixels on the short side is plenty; JPEG or PNG.
- The first identity-photo run downloads ~2.5 GB of CLIP-H image
  encoder weights and ~50 MB of IP-Adapter weights. One-time cost.

---

## 1. Why a dedicated portrait command?

You *could* generate a portrait with `plakat generate "a portrait of
a woman with red hair"`. Plakat would do it. But:

- The aspect ratio defaults are wrong (square instead of 3:4).
- The negative prompt defaults don't include the face/anatomy fixers
  ("deformed face, asymmetric eyes, extra fingers, …") that
  drastically improve portrait output.
- There's no path to use a reference photo for identity.

`plakat portrait` is the same underlying pipeline but with
portrait-tuned defaults and an optional identity branch. Use it when
the *who* of the image matters as much as the *what*.

---

## 2. Your first portrait — text only

The simplest invocation:

```bash
plakat portrait "a woman with long red hair, smiling, soft window light"
```

Plakat picks:

- 3:4 aspect ratio (768x1024 by default)
- 30 steps (default)
- The Euler-A scheduler (smoother skin tones than the alternatives)
- A baseline negative prompt covering common portrait failure modes

Output lands in `./out/plakat-portrait-<seed>.png`.

### Tweaking the basics

Same flags as `plakat generate`:

```bash
plakat portrait "a man with grey hair, contemplative" \
    --count 4 \
    --steps 40 \
    --seed 42
```

This produces 4 portraits with seeds 42-45. Good for picking a
favorite to iterate on.

---

## 3. The 3:4 aspect default

Portraits look more natural in 3:4 (or sometimes 2:3) than in
squares. Override with `--aspect`:

```bash
# Square portrait
plakat portrait "a woman, smiling" --aspect 1:1

# Tall portrait
plakat portrait "a woman, smiling" --aspect 2:3
```

`--base 768` sets the *shorter* side; the other side is computed from
aspect. SD 1.5 was trained at 512, but 768x1024 portraits work well
and look more detailed.

---

## 4. Identity preservation — using a reference photo

This is the part that's specific to portraits. Pass a reference photo
with `--photo`:

```bash
plakat portrait "wearing a knit sweater in front of a fireplace" \
    --photo ./my_friend.jpg
```

Plakat:

1. Encodes your photo through an image model that captures facial
   features.
2. Injects those features into the generation pipeline.
3. Produces a portrait that *resembles* the person in your photo,
   but with the scene/clothing/expression you described in the prompt.

This is **identity preservation**: the output keeps the subject's
identity but everything else (pose, clothing, lighting, background)
follows the prompt.

### What makes a good reference photo

- Head-and-shoulders crop. Full-body works but wastes pixels on
  features the face encoder ignores.
- Face mostly forward-facing. Profile shots work less well.
- Decent lighting on the face. Heavy shadows confuse the encoder.
- Single person. The encoder picks the most prominent face if there
  are multiple, but it's a coin flip which one.
- Reasonable resolution — 256+ on the short side. Tiny photos lose
  detail.

---

## 5. Choosing the identity strategy

There are two main approaches, controlled by `--identity`:

| Strategy | What it does | When to use |
|---|---|---|
| `plus-face` (default) | Uses CLIP-H image features. Robust, fast, no extra setup. | Most cases. Good identity preservation, no setup hassle. |
| `faceid` | Uses InsightFace's ArcFace embedding — purpose-built for face recognition. Much better identity fidelity. | When you need the *strongest* possible likeness and you're willing to download ~150 MB of ArcFace weights. |

`plus-face` works out of the box. `faceid` requires a one-time setup:

```bash
# Tell plakat where the ArcFace weights live (HuggingFace path).
export PLAKAT_ARCFACE_HF="MonsterMMORPG/SECourses_FaceID#arcface.safetensors"

# Or, if you've downloaded them locally:
export PLAKAT_ARCFACE_WEIGHTS=/path/to/arcface_r50.safetensors
```

See `Documentation/PERSONA.md` "FaceID setup" for the full setup
options.

Once configured:

```bash
plakat portrait "wearing a knit sweater" \
    --photo ./my_friend.jpg \
    --identity faceid
```

If you're not sure, start with `plus-face`. Switch to `faceid` if
the resemblance isn't strong enough.

There are also SDXL variants: `--identity plus-face-sdxl` and
`--identity faceid-sdxl`. These require `--model sdxl` and produce
larger, more detailed portraits at higher cost.

---

## 6. Tuning identity strength

How "strongly" does plakat enforce the photo's identity? The default
is moderate. You can adjust with `--face-strength`:

```bash
# Default (0.8) — solid resemblance
plakat portrait "..." --photo ... --face-strength 0.8

# Stronger (1.0+) — strongly anchored to the photo
plakat portrait "..." --photo ... --face-strength 1.2

# Subtler — useful when the photo is dominating too much
plakat portrait "..." --photo ... --face-strength 0.5
```

Higher = more like the photo, less responsive to prompt. Lower =
more responsive to prompt, less like the photo. There's a tradeoff;
0.6-1.0 is usually the sweet spot.

---

## 7. Helping the encoder find the face

For `faceid`, alignment matters a lot. ArcFace was trained on tight
face crops aligned to a canonical template. If your reference photo
isn't a tight head-shot, the encoder fights bad alignment.

Three options, in increasing order of accuracy:

### a) Centre-crop (default, no setup)

Plakat resizes the photo to 224x224 and centre-crops. Works for
already-tight portraits. Mediocre for landscape orientations.

### b) Manual bbox

You tell plakat where the face is:

```bash
plakat portrait "..." \
    --photo ./my_friend.jpg \
    --identity faceid \
    --face-bbox "0.2,0.05,0.8,0.65"
```

The bbox is `x0,y0,x1,y1` in normalized coordinates (0..1, origin
top-left). The example crops a face that's in the top-middle of the
image.

### c) Auto-detection (SCRFD)

If you have SCRFD detector weights configured (see
`Documentation/PERSONA.md` "Optional SCRFD auto-detection"), plakat
detects the face for you. No bbox needed.

For `plus-face`, alignment is less critical — CLIP-H is more
forgiving than ArcFace.

---

## 8. Putting a portrait into a scene

A portrait CLI command produces one image: the subject in a setting
described by the prompt. If you want to render the same person across
many scenes (forest, beach, city, with different lighting), use a
scenario with **personas**.

### What's a persona?

A persona is a named identity bundle in a scenario file: a reference
photo + identity settings. Tasks can pull personas in by name, and
the task's scene/weather/prompt becomes the setting.

Minimal persona scenario:

```hjson
{
    # Standard global settings (see GENERATE_TUTORIAL.md).
    model: sd15
    base: 768
    aspect: 3:4
    steps: 30
    out: ./out
    enhancer: deepseek

    # Define the people up here.
    personas:
    [
        {
            name: alice
            photo: ./refs/alice.jpg
            identity: plus-face
            face-strength: 0.85
        }
    ]

    scene:
    [
        { name: forest, prompt: "a mossy forest clearing with shafts of light" }
        { name: harbor, prompt: "an old wooden fishing harbor at sunset" }
    ]
    weather:
    [
        { name: warm, prompt: "warm golden afternoon light" }
    ]

    # Tasks pull a persona in by name.
    tasks:
    [
        {
            name: alice_in_forest
            scene: forest
            weather: warm
            prompt: "walking among the trees, wearing a wool coat"
            personas: [ alice ]
        }
        {
            name: alice_at_harbor
            scene: harbor
            weather: warm
            prompt: "looking out at the boats, wearing a knit sweater"
            personas: [ alice ]
        }
    ]
}
```

Run it:

```bash
export DEEPSEEK_API_KEY="..."
plakat scenario my_persona_scenario.hjson
```

Two portraits land in `./out/alice_in_forest/` and
`./out/alice_at_harbor/`. Both feature the same person (Alice's face
from the reference photo) but in different scenes.

### Why this beats two separate `plakat portrait` invocations

- One command runs the batch.
- The base model loads once (not twice — saves ~30 seconds on slow
  disks).
- The CLIP-H encoder loads once.
- You get a tidy output tree organized by task name.
- You can add per-task overrides (sizes, seeds, etc.) without rewriting
  long CLI invocations.

For 2-3 portraits, CLI invocations are fine. For 10+ portraits or
recurring batches, scenarios pay off.

---

## 9. Multiple people in one image

Sometimes you want two friends in the same scene. Use the
**multi-persona** task form:

```hjson
{
    # ... globals ...

    personas:
    [
        { name: alice, photo: ./refs/alice.jpg, face-strength: 0.85 }
        { name: bob,   photo: ./refs/bob.jpg,   face-strength: 0.85 }
    ]

    tasks:
    [
        {
            name: alice_and_bob_harbor
            scene: harbor
            weather: warm
            prompt: "two friends looking out at the boats"
            size: 1024x576           # wider aspect — gives each face room

            # The multi-persona form: each persona gets a bbox in the
            # output image telling plakat where to place them.
            personas:
            [
                {
                    name: alice
                    bbox: [ 0.05, 0.10, 0.48, 0.95 ]   # left half
                }
                {
                    name: bob
                    bbox: [ 0.52, 0.10, 0.95, 0.95 ]   # right half
                }
            ]
        }
    ]
}
```

`bbox` is `[x0, y0, x1, y1]` in normalized coordinates. Plakat
generates the image and applies each persona's identity to their
designated bbox region.

### Tips for multi-persona compositions

- **Use a wider aspect.** 16:9 (`1024x576`) gives each face enough
  pixels to be recognizable. Square aspects squeeze faces too small.
- **Leave room between bboxes.** Overlap causes the regions to fight.
- **Place faces in the upper half of each bbox.** Real photos rarely
  have heads at the bottom; the model has learned this and produces
  more natural results with upper-half placement.
- **All personas in a scenario must use the same identity strategy.**
  You can't mix `plus-face` and `faceid` in one scenario.

---

## 10. Persona-specific negative prompts

A persona can carry its own negative prompt — useful for traits the
photo *has* but you *don't* want:

```hjson
personas:
[
    {
        name: alice
        photo: ./refs/alice.jpg
        negative: "smiling, mustache, glasses"   # prepended to task negative
    }
]
```

When that persona is used, its negative is combined with the task's
effective negative. Stays attached to the persona because it
describes the *who*, not the *scene*.

---

## 11. Merging multiple reference photos

Pass `--photo` more than once to merge facial features across several
reference photos — useful when you have multiple photos of the same
person, or want to blend two people for a fictional likeness.

The merge happens **at the encoder's embedding level**, not by
pixel-averaging the inputs. Each photo is encoded independently into
the identity adapter's natural feature space (CLIP-H hidden state for
`plus-face`, ArcFace 512-d vector for `faceid`), the encodings are
weighted-summed, and the merged identity drives generation.

```bash
# Two photos, equal weight — averages identity across them.
# Useful for smoothing single-photo noise (lighting, pose).
plakat portrait "..." \
    --photo alice_smile.jpg \
    --photo alice_neutral.jpg

# Two photos, weighted — 70% of one, 30% of the other.
plakat portrait "..." \
    --photo alice.jpg:0.7 \
    --photo bob.jpg:0.3

# Three photos with auto-fill — alice gets 0.6 explicitly,
# the other two each auto-fill to 0.2.
plakat portrait "..." \
    --photo alice_a.jpg:0.6 \
    --photo alice_b.jpg \
    --photo alice_c.jpg
```

Weight rules:

- Weights are **proportions**, internally normalized to sum to 1.0.
- Missing weights split the remainder equally among the unweighted
  photos.
- `--face-strength` is independent — it controls the *total* identity
  influence on the output. Weights only control the *mix among photos*.

Tip: for the best results with multiple photos, configure SCRFD
auto-detection (see `Documentation/PERSONA.md` "Optional SCRFD
auto-detection"). It runs per-photo so each face is aligned correctly
before the embedding step.

In scenarios, the persona definition takes a `photos:` list instead of
a single `photo:`:

```hjson
personas:
[
    {
        name: alice_averaged
        photos:
        [
            { path: ./refs/alice_smile.jpg,   weight: 0.5 }
            { path: ./refs/alice_neutral.jpg, weight: 0.3 }
            { path: ./refs/alice_serious.jpg, weight: 0.2 }
        ]
        face-strength: 0.85
    }
]
```

`photo:` (singular) and `photos:` (list) are mutually exclusive — set
one or the other.

---

## 12. Common issues

**Output doesn't look like the photo.**
Try `--face-strength 1.0` or higher. If still not matching, switch
from `plus-face` to `faceid` (better identity but needs ArcFace
setup). Also check the photo: tight head-shot, forward-facing, good
lighting.

**Face looks "off" — wrong proportions, weird eyes.**
This is the model's underlying face-generation quality, not the
identity preservation. Increase `--steps` (40+). Make sure
`--negative` includes "deformed face, asymmetric eyes" (the default
already does). Try a different seed.

**Identity is too strong — the person looks "pasted in."**
Drop `--face-strength` to 0.5 or 0.6. The output becomes more
responsive to the prompt at the cost of slightly less likeness.

**Two-persona compositions blend the faces together.**
The bboxes are overlapping or too close. Move them apart, or use a
wider canvas (`size: 1280x576`).

**SDXL FaceID needs ArcFace too?**
Yes. Both SD 1.5 and SDXL FaceID use the same ArcFace backbone. Same
environment variable setup applies.

**`--photo` ignored?**
Check the path. Plakat errors loudly if the file doesn't exist; if it
doesn't error, the photo *is* being read but the identity adapter
might not be applying strongly enough — bump `--face-strength`.

---

## Where to next

- **General text-to-image and scenarios** → `GENERATE_TUTORIAL.md`
- **Apply art styles to your portraits** → `STYLES_TUTORIAL.md`
  (combining identity + style is one of the most powerful workflows)
- **Build your own style catalog** → `HOW_TO_CREATE_MY_OWN_STYLE.md`
- **Full persona/portrait reference** → `Documentation/PERSONA.md`
