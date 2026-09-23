# The painting plan & the armature (guidance layer)

This is the part of `plakat paint` that makes it a **painter**, not a filter. It implements the RFC PAINT‑1
governing principle:

> **Structure is low‑resolution and model‑derived. Surface is full‑resolution and stroke‑derived. The two never
> meet at the same scale.** (§1.1)

If the engine paints from a full‑resolution photo, error minimisation makes it **trace** the detail — a beard
becomes a scribble, skin becomes mush. That is a painterly filter, not a painting. The fix is to paint from a
**coarse structural armature** (a plan), so the brushwork has to *invent* the surface.

---

## `--armature <px>` — paint from structure, not pixels (structure‑preserving)

Paint from an armature at this structure resolution instead of the raw photo. **The single most important switch
for a good result from a photo** — but *how* it reduces the image matters, and this is where an earlier version
went wrong.

The armature is built by an **edge‑preserving smooth (bilateral) + value‑mass quantise (posterise)**, NOT a
blur. A blur destroys structure (edges, the boundaries of value masses) *along with* texture, so the engine
paints a structureless **smear (mush)**. The structure‑preserving armature is *coarse in texture but sharp in
structure* — the beard is a dark mass with a defined edge, the face a light mass, the eyes dark accents — so the
strokes paint recognizable form.

```bash
plakat paint from photo.png --medium oil-direct --palette image --armature 160 --budget 30000 ...
```

- **LARGER = more structure retained** (finer armature, lighter smoothing). A moderately fine armature (~120–190)
  keeps modelling; density then comes from the **budget**, not from over‑coarsening.
- **`--armature-levels <N>`** = number of value masses (default 14). Fewer = bolder, flatter block‑in masses (but
  too few flattens the reference so much that the restate passes have nothing to paint → sparse); more = subtler.
- **Budget is the density lever.** A rich, non‑washed result needs a *dense* budget (the analyzer now defaults to
  ~area/9). A sparse budget starves the painting into a wash — that was a second cause of the earlier smears.
- **Unset** = paint from the full‑resolution photo (the legacy "filter" behaviour).

**Honest note:** low‑contrast subjects (a white beard/shirt on a light ground) have little value structure to
paint, so they read sparser than high‑contrast subjects however you tune — that's the subject, not the engine.

A single global armature has one problem: coarse enough to keep the beard from scribbling is **too** coarse for
the face. That is what the focal armature solves.

## `--armature-face <px>` — a fine face, a wash body

With `--armature`, paint the **detected face** from a *finer* armature than the rest, blended by the face mask
(SCRFD). Crisp, recognizable face **and** a wash beard/background in one pass (RFC §5.2, per‑figure armatures).

```bash
plakat paint from photo.png --medium watercolour --palette image \
    --armature 64 --armature-face 200 ...
```

- Must be **larger** than `--armature`. **~160–220px** keeps the eyes/features while still being an armature
  (not tracing skin pores).
- Detects a face automatically — no need for `--preserve-face` (though they compose: `--preserve-face` also fires
  the crisp *detail tier* inside the face).
- No face detected → falls back to the uniform coarse armature.

## `--armature-body <px>` + `--recede` — the three‑tier (multi‑region) armature

The face‑vs‑rest split is two tiers. A portrait wants **three**: a *coarsest* background, a *mid* subject body,
and a *fine* face. `--armature-body` mattes the subject (U2Net) and paints the body from a mid armature, so the
background settles to the calmest washes / paper while the body keeps a little more structure than the sky.
`--recede` then veils the (matted) **background** so the subject advances — aerial perspective from the matte,
no depth model needed.

```bash
plakat paint from photo.png --medium watercolour --palette image \
    --armature 44 --armature-body 104 --armature-face 200 --recede 0.28 ...
```

`--plan auto` sets all three tiers + recession automatically when it detects a subject. Ordering:
`--armature` (background) < `--armature-body` (body) < `--armature-face` (face).

## `--semantic` — semantic region tiers (OWL‑ViT)

Detect named parts and give each its own armature tier. v1 detects **hair/beard** (open‑vocab, OWL‑ViT) and
assigns it a **coarse wash tier** — softer than the subject body — so a beard stays a flowing wash rather than
picking up body‑level structure. The tiers blend coarse→fine with body/face, so the finer **face** tier is laid
last and wins where a beard box overlaps the jaw.

```bash
plakat paint from photo.png --plan auto --semantic ...     # (--plan auto enables it automatically for a subject)
```

Tiers are `(mask, resolution)` regions in the same blend — more parts (skin, clothing, hands) add as further
tiers with no paint‑side change.

## `--sam` — precise masks (MobileSAM)

The U2Net matte and feathered detection boxes give *soft* boundaries, so a silhouette edge drawn on them is
weak. `--sam` prompts **MobileSAM** from the detected face (a fact) to segment the **precise subject silhouette**
and a **face‑shaped focal region** — a sharp silhouette edge and detail concentrated on the actual face, not a
rectangle. `--plan auto` enables it when a subject is present (and raises the face armature, since the focal
region is now precise). It runs last and overrides the soft masks.

```bash
plakat paint from photo.png --plan auto --sam --silhouette 0.6 --silhouette-mode line ...
```

`--semantic` detects several parts, each given an armature tier **derived from the plan's own coarse/body/face**
resolutions (fact‑driven, not fixed constants):
- **hair/beard** → a **coarse** wash tier (softer than the body);
- **skin** (neck/forehead/bald head) → a **mid** tier (smooth form);
- **hands** → a **fine** tier (a secondary focal — hands need structure);
- **clothing/shoulders** → unioned into the **subject fact**, so a light shirt is *painted* (a light mass)
  instead of being reserved to blank paper (the fix for vanished shoulders).

Each is a `(mask, resolution)` region in the same blend; adding more parts is one entry in the detector, no
paint‑side change.

## Painting decisively over the facts — not a wash

Detecting regions is not enough; the deterministic painter must *paint decisively over them*, or a light subject
just washes out. Three fact‑driven controls, all enabled by `--plan auto` and tunable via CLI/plan:

- **`--commit-shadows <0..1>`** (RFC §3.3) — in the dark value masses (derived from *this image's* own value
  percentiles), the reserve is lifted, the darks deepen pass over pass, and more pigment is loaded — a **solid
  value backbone** instead of a pale wash.
- **Subject‑gated reserve** — `reserve` is a fact about the *background*: bright cells keep paper only *outside*
  the detected subject. Inside the subject, a light shirt/skin is painted as a light mass, never reserved away.
- **`--silhouette <0..1>` + `--silhouette-mode`** (RFC §5/§7) — mark the detected subject boundary so a light
  subject reads by its **edge**. The mode is how a painter marks an edge:
  - `line` — a soft drawn contour (default),
  - `colour` — a temperature/hue shift, *no line*,
  - `knife` — a scraped/lifted crisp lighter edge,
  - `lost` — dissolved (no marking).

---

## The painting plan — analyze once, paint deterministically

The **art director** stage: a model/analysis inspects the image and writes a *plan* — the structural decisions a
painter would make — and the deterministic, weight‑free stroke engine executes it. **No pixel of the deliverable
comes from a model** (G1); the plan decides only *structure* (armature resolutions, value key, reserve), never
surface.

### `plakat paint plan <image>`

Analyze an image and write an inspectable, editable `plan.hjson`:

```bash
plakat paint plan photo.png            # writes photo.plan.hjson
plakat paint plan photo.png --out -    # print to stdout
plakat paint plan photo.png --medium oil-direct --palette earth
```

Example output (the reasoning is preserved as comments):
```hjson
# coarse armature 72px — paint from structure, not pixels (no tracing)
# 1 face(s) → focal armature 200px (crisp face over a wash body)
# luma σ 0.179 → value-key 0.59 (expand a flat reference's tonal range)
# surface-white medium → reserve 0.90 (keep the paper for the lights)
# budget 8729 strokes (area-scaled)
medium: watercolour
palette: image
style: impressionist
armature: 72
armature_face: 200
value_key: 0.59
reserve: 0.90
budget: 8729
```

The analyzer (v1, deterministic) decides:
- **coarse armature** from canvas size,
- **focal face armature** when a face is detected (SCRFD),
- **value‑key** from the reference's measured tonal contrast (a flat, foggy photo is keyed harder),
- **reserve** for surface‑white media,
- **budget** scaled to area.

### `plakat paint from <image> --plan auto` — one command

Analyze *and* paint in a single command. The plan fills every structural flag you leave unset; **explicit flags
always win**, and surface knobs compose on top:

```bash
plakat paint from photo.png --plan auto \
    --preserve-face 0.5 --bleed 0.6 --edge-pool 0.35 --granulate 0.12 \
    --dry-shift 0.08 --splatter 0.4 --paper-edge 0.55 --contrast 1.2 --warmth 0.15
```

Paint from a saved/edited plan instead of re‑analysing:
```bash
plakat paint from photo.png --plan photo.plan.hjson ...
```

### What the plan does *not* do
The plan is **structure**. Surface — bleed, granulation, splatter, edge‑pool, paper‑edge, the finish grade — is
the stroke engine and the tuning knobs ([PAINT_CONTROLS.md](PAINT_CONTROLS.md),
[PAINT_WATERCOLOR.md](PAINT_WATERCOLOR.md)). This separation is deliberate: the art director plans, the painter
paints.

### Extending the art director
The v1 analyzer is deterministic (face + contrast + medium). A **model‑driven** art director — a vision model or
a low‑resolution diffusion block‑in that proposes semantic regions, value masses, focal point and per‑region
armature resolutions (RFC §5, §12.2) — plugs in behind the same `PaintPlan` interface, with **no change to the
paint side**. The model only ever writes the plan.

---

## Honest limits

- **Faces are the ceiling.** A weight‑free stroke engine painting from a coarse armature produces a *recognizable,
  soft* face, not a skilled artist's confident facial drawing. The focal armature makes it as good as the approach
  allows; it does not make it a master's portrait.
- The plan/armature layer is what turns "a filter that scribbles" into "a painting that invents washes." That is
  the RFC's core thesis, and it holds — but it is a *painter's procedure over a structural target* (N2), not a
  claim to paint like a human.
