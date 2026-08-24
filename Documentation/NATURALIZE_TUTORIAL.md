# Tutorial — de-slopping AI images with `plakat naturalize`

A practical, honest walkthrough of making AI/computer output **less sloppy — a genuinely better
picture** (RFC QUALITY-1/2/3). Read the mental model first; it's the difference between using this well
and being disappointed.

## 1. The mental model (read this)

`naturalize` has two halves, and they are **not** equal:

| Half | What it does | Reliability |
| --- | --- | --- |
| **Weight-free** (the headline) | Improves **colour, contrast, detail, skin** — no GPU, deterministic. **Preserves the original** exactly (composition, style, faces). | ✅ Reliable, non-destructive. Use by default. |
| **Model-backed** (best-effort) | *Regenerates* regions to attempt structure/anatomy/clutter. | ⚠️ Can introduce new artifacts, shift colour/style. Opt-in only. |

**The one thing to internalise:** a diffusion model **re-paints, it does not reason.** It cannot "remove
an extra arm" or "fix a bad hand" — it can only regenerate that area and hope, which invents new problems
about as often as it fixes old ones. So:

- **Structural tells** — extra limbs, fused fingers, floating wires, melting architecture — are *baked into
  the generation*. No post-pass reliably fixes them without regenerating into a **different picture**.
- **Surface tells** — oversaturation, muddy contrast, a colour cast, plastic skin, too-clean texture — are
  what the weight-free pass genuinely fixes, safely.

Naturalize is **de-slop, not disguise**: the goal is a cleaner, better version of *your* picture — not to
pass it off as hand-made, and not to hallucinate a new one.

## 2. Quick start

```bash
plakat naturalize in.png --out out.png --preset photo
```

That runs the weight-free pipeline: **polish → micro-texture → light film grade**. Every output prints an
**AI-tell score** (`0` = reads human, `1` = reads AI) so you can see the before/after direction.

## 3. What the weight-free pass actually does

Run in this order, all deterministic and GPU-free:

1. **`polish`** — the quality core:
   - gray-world **white balance** (kills the AI colour cast, clamped so a real sunset survives),
   - robust, ratio-preserving **auto-levels** (muddy/washed → true black & white without shifting hue),
   - **vibrance** (tames blown-out oversaturation, lifts dull colour toward natural),
   - **unsharp** (crisps the soft AI mush).
2. **`micro`** — fine pore / micro-wrinkle texture added *only* where the image is unnaturally smooth
   (variance-gated, mid-tones) — the fix for **plastic AI skin**.
3. **grade** — a *light* film finish (a little grain / desaturation / vignette). Chromatic aberration is
   kept near zero — it's a degradation, not an improvement.

## 4. Recipes by image type

```bash
# Photoreal image — the general preset
plakat naturalize photo.png --out out.png --preset photo

# Portrait with waxy/plastic skin — lean on micro-texture
plakat naturalize portrait.png --out out.png --people 1.5 --micro 1

# Landscape — banding sky / cloud-foliage mush
plakat naturalize land.png --out out.png --sky 1 --vegetation 1 --landscape 1

# AI ART / illustration — clean colour WITHOUT touching the style; keep the palette faithful
plakat naturalize art.png --out out.png --polish 0.9 --desaturate 0 --aberration 0

# Just the pure correction, nothing else
plakat naturalize any.png --out out.png --polish 0.85 --grain 0 --vignette 0 --micro 0
```

Content focuses combine freely: `--people --sky --vegetation --cityscape --landscape --sea --river
--mechanics --household --animal --food --interior --textile --foliage-macro <N>` (each blends a
subject-tuned profile). Every analog knob is overridable: `--polish --micro --grain --desaturate --warm
--vignette --bloom --defocus --aberration`.

### Watercolor paper / pigment authenticity — `--paper` (art only)
For genuine **watercolour / ink-wash** art, `--paper <N>` models the physical medium so it stops reading as
"simulated media": **paper tooth** (pigment settles into the cold-press valleys), **granulation** (pigment
speckle scaled by wash density), and **edge pooling** (darker wash rims — the wet-on-wet signature). It's
**pigment-gated** — bare paper and photos are left alone. Recommended **~0.6** (subtle 0.5 · overtly
hand-painted 1.0+). Do not use on photos.

```bash
plakat naturalize watercolor.png --out out.png --paper 0.6 --polish 0.7
```

**Auto-paper (6.14):** when the medium is **wet** — named via `--medium watercolor`/`--style`, or CLIP
**auto-detected** — `--paper` is applied at 0.6 automatically (no need to name it); `--paper 0` disables. On
a pure weight-free run, add `--auto-medium` to let the detector decide. It's also reachable from a spec:
`generate --naturalize "photo paper=0.6"`, a scenario `naturalize:` field, and `api::Naturalize`.

### Batch — a whole folder (6.14)
Point `naturalize` at a **directory** and it de-slops every image into the `--out` directory (same
filenames). Model-backed passes reload per image (a convenience, not a resident server).

```bash
plakat naturalize ./shots/ --out ./deslop/ --preset photo
```

## 5. Picking the least-AI frame of a batch

The AI-tell score is weight-free, so you can rank/select on it:

```bash
plakat rank shots/ --ai-tells                          # least-AI-looking first
plakat generate "..." --count 8 --keep-best 2 --ai-tells   # keep the 2 most human-looking (aesthetic − λ·ai_tell)
```

## 5b. Diagnose first — the scorecard (6.15)

Not sure what an image needs? Ask for a **scorecard** — plakat's own AI-tell verdict plus the exact recipe
to run:

```bash
plakat naturalize suspect.png --report            # bar-graph scorecard + recommended command
plakat naturalize suspect.png --report --json     # structured
```
It decomposes the AI-tell into **oversaturation** and **over-smoothness**, reports the **CLIP-detected
medium**, and prints the `naturalize` flags to run (e.g. `--polish 0.7 --micro 0.5 --paper 0.6`).

## 5c. Region focuses — different subjects, different de-slop (6.15)

A frame with several subjects gets each its own profile, composited with feathered seams:

```bash
plakat naturalize street.png --out out.png --auto-regions          # faces→people, sky band→sky, rest→base
plakat naturalize street.png --out out.png --region "0,0,1,0.4:sky=1.5" --region "0,0.4,1,1:vegetation=1"
```
`--region "x0,y0,x1,y1:<spec>"` (normalized 0..1, repeatable) applies a spec to a feathered rectangle;
`--auto-regions` detects the sky band (weight-free) and people (SCRFD).

## 5d. Video / animation de-slop (6.15)

Point `naturalize` at a **video/animation** (mp4/mov/webm/mkv/avi/gif) and it de-slops every frame and
re-encodes (container follows the `--out` extension). The pass is **weight-free** and its grain is
**frame-invariant** — the texture sits still while the image moves, so there's no flicker. Needs `ffmpeg`.

```bash
plakat naturalize clip.mp4 --out clean.mp4 --preset photo
plakat naturalize anim.gif --out clean.gif --polish 0.8 --micro 0.3
```

## 6. As a step in generation

```bash
plakat generate "a forest path" --quality high --naturalize photo   # naturalize each output in place
```
Or a scenario field: `naturalize: "photo vegetation=1"`. Library: `plakat::api::Naturalize::new("photo
polish=0.9 micro=0.2").run("in.png","out.png")`.

## 7. The best-effort model tools (opt-in — may introduce artifacts)

These **regenerate** pixels. They can help on the right image and hurt on the wrong one — always compare
against the input. The medium is **auto-detected** (CLIP zero-shot) when you don't pass `--style`/`--medium`,
so a re-paint stays *in style* instead of drifting to photoreal; override with `--style "<desc>"` /
`--medium watercolor|oil|ink|gouache|pencil|acrylic|pastel|comic`.

```bash
# Figure-scoped repair (default) — protects faces AND background; repaints ONLY the figures, in the
# auto-detected medium. This is the character-preserving structural tool.
plakat naturalize kids.png --out out.png --repair 1            # medium auto-detected
plakat naturalize kids.png --out out.png --repair 1 --repair-scope non-face   # also regen the background

# Whole-image structure fix (photoreal / non-figure only — regresses cohesive art)
plakat naturalize photo.png --out out.png --geometry 1 --style "photograph"

# Remove named clutter (best-effort; wires use a weight-free sky-gated detector)
plakat naturalize street.png --out out.png --declutter "overhead wires"
```

**Honest expectations** for these: they preserve faces (repair) and style (with `--style`), and make a
*bounded* attempt at local defects — but they will **not** reliably remove an extra limb or a floating
wire, and they may change the surrounding composition. If the result is worse, drop back to the weight-free
recipe. On cohesive art, prefer weight-free.

## 8. When naturalize is the wrong tool

If the image has **broken structure you must fix** (extra limbs, wrong hands, nonsensical mechanics), that's
a **generation** problem, not a post-pass one. Regenerate the source with structural guidance
(`generate` with a ControlNet pose/depth), accepting it will be a *new* picture. No amount of naturalizing
repairs a hallucinated composition without replacing it.

## 9. Honest limits (the short version)

- Weight-free = reliable surface improvement, preserves your picture.
- Model tools = best-effort, may add artifacts, opt-in.
- The AI-tell score is a coarse ranking heuristic, not a verdict.
- Structural correctness is a generation-time property; a post-pass can't invent it.

See also: [`QUALITY.md`](QUALITY.md), [`RFC_QUALITY_1`](RFC_QUALITY_1.md) /
[`_2`](RFC_QUALITY_2.md) / [`_3`](RFC_QUALITY_3.md). Drivers: `corpus/naturalize_run.sh`,
`corpus/naturalize_art_run.sh`.
