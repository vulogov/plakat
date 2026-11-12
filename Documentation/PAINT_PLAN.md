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

### `--make-plan` — the art director with eyes (7.19.0, the critic 7.20.0)

`--plan auto` writes the **structure** only and leaves every dial of the medium to its stock profile. The review
cycles settled, medium by medium, a *house recipe* — the plan the reviews accepted, 50–130 keys — and the rule that
a medium is tuned in its plan, never in the source. `--make-plan` puts the recipe together with a reading of the
picture and writes the **whole plan**, every dial with its reason:

```bash
# write <out>.plan.hjson from the picture and the medium's house recipe, then paint from it
plakat paint from photo.jpg --medium oil-direct --make-plan -o out.png
# name the plan file
plakat paint from photo.jpg --medium gouache --make-plan plans/photo.gouache.hjson -o out.png
# the plan alone, to read and edit before painting
plakat paint plan photo.jpg --medium watercolour --director --out photo.plan.hjson
plakat paint from photo.jpg --plan photo.plan.hjson -o out.png
```

The director composes the plan in three layers, written as three sections of the file:

1. **The structure** — `plan_from`'s own reading (the armature tiers, the face tier, families, the measured value
   key, the budget), as `--plan auto` writes it.
2. **The house recipe** — `assets/recipes/<medium>.hjson`, shipped with the binary as a *plan*, not as code: the
   medium's accepted dials with their own comments. A key the recipe names is a dial the medium **uses**; the
   director moves no dial the medium does not use, and turns on none the recipe holds at 0 (a watercolour's
   `hotspot: 0` is the medium's decision, not a gap). (A medium the cycles have not tuned yet — `line-and-wash`,
   `early-book-illustration`, `durer`, `black-pencil` — has a three-key recipe and gets the structure only.)
3. **The picture's adjustments** — the rules, each universal, each written with its fact and measurement, e.g.
   `shadow_floor: 0.1   # nocturne: mean luma 0.18, darks 71% — the shadow family may sit deeper … (was 0.16)`.
   The recipe's line a rule moved stays in the file as a comment, so the reader sees what the picture changed.

The facts the rules read, and where they come from:

| fact | source | what it moves |
|---|---|---|
| mean luma, luma σ, dark/light shares | pixels | the key (`value_key`, measured — the recipe's yields); a **nocturne** (mean < 0.3) deepens `shadow_floor`, sets `point`/`point_field`/`glint`, cuts `cavity`; a **high key** (mean > 0.72) lifts `shadow_floor`, `reserve 0.95` on paper; a **flat** picture (σ < 0.12) raises `lost`/`aerial`/`sfumato`/`aqua_soft`/`aqua_bloom`, cuts `micro`/`detail_restate` |
| structure (Laplacian σ), calm and busy shares (the local luma σ over a 9×9 window at 1024 px: under 0.02 calm, over 0.08 busy) | pixels | **busy** → `micro`, `adapt`, `detail_restate`, `aqua_detail`, `aqua_crisp`; **calm fields** → `smooth`, `settle`, `vary`, less `collide`/`fray` |
| mean chroma, warmth | pixels | `pigments` 8 for a near-monochrome picture, 20 for a wide palette (`palette: image`, opaque media) |
| points of light on a dark ground per megapixel (100 or more) | the point pass's own detector, the ground under 0.25 | `point`, `point_field`, `hotspot` |
| faces and their share | SCRFD | `cavity_face 0.3`, `periphery`, `advect` cut (the drag blurs the modelling), `loose` on a close portrait |
| the subject's share and centre | U²-Net matte | a compact subject → `periphery`, `loose`; none → `periphery` down |
| the far ground's share (relative depth: the nearest thing at 1) | Depth-Anything-V2 | `aerial`, `lost`, `nib_depth`, `aqua_vignette` — only on a scenic picture: no face and no large subject, or CLIP's landscape/townscape (a portrait's wall is as far as a distance in a relative depth, and is not one; 7.20.0) |
| the light's azimuth and coherence (a plane fitted to the luma over the sheet: it rises toward the light, and the share of the variance it explains is the coherence — a storm sky 0.72, a lamplit hull 0.67, a street 0.10; the finish's convention — 0 from the right, 90 from below, 180 from the left, 270 from above) | pixels | the relight's lamp (`light`, or the knife's `blade_light` where the recipe names it) set where the picture's light comes from, when the direction is coherent (over 0.25) and the medium has a relief (7.20.0) |
| genre (portrait / figure / landscape / townscape / still life / interior / animal) | CLIP zero-shot over the sheet | still life → `hotspot`, `glint`; interior → `shade`, `lost`; animal → `semantic`; landscape → `aerial` |
| sky, water, foliage, stone shares | CLIP zero-shot over a 4×4-ish grid of crops | water → `glint`, `hotspot`, `smooth`; foliage → `fray`, `vary`, `haste`, `nib_tuft`; stone/townscape → `adapt`, `chisel`, `micro` |

Every model fact is optional: a model that cannot run (no weights, no device) leaves its fact unknown, the header
says so, and the rules that need it stand down. `PLAKAT_PAINT_DIRECTOR_PIXELS=1` reads the pixel facts only.
Explicit flags still win over the plan, as always; `--make-plan` and `--plan` exclude each other.

**The model's pass** (`--make-plan-model <provider>`, or `paint plan --director --model <provider>`): a vision
model (`gemini`, `deepseek`, `auto`) is shown the picture, the facts, the composed plan and the glossary of the
dials it may move, and answers in the insights' own form — `## Insights` and a `## Recommendations` block of
`key: value   # reason` lines. Its lines pass the same gate as the rules: only dials the plan names, never the
structure (armature, tiers, budget, medium, palette, the switches), never a face's dial, values clamped to the
dial's range; the accepted lines are written as their own section with the provider named, the refused ones as
comments with the reason. A hosted provider means the picture leaves the machine, and the terminal says so. The
local LLM has no vision model and the pass is skipped with a message; the plan stands without it.

### The critic — `--make-plan-iterate`, `paint plan --verdict`, `paint compare` (7.20.0)

The director reads the picture before the painting; the critic reads the SHEET after it, with the same eyes. Both
pictures are reduced to the director's working size (the source's aspect at 1024) and measured alike
(`src/paint/measure.rs`): the mean luma and its spread, the dark and light shares, the calm and busy fields by the
local σ, the chroma and its 90th percentile, the points of light on a dark ground, the bare white, the edge energy;
and the two compared — what the sheet KEPT of the picture: the tone histograms' overlap, the key shift, the value
range's ratio, the local contrast kept in the focus and on the periphery, the calm fields kept calm and the busy
passages kept busy, the strong edges carried (within a pixel, at half the strength), the points of light laid, the
chroma ratio. Every number is a share or a ratio, so a 2k oil and a 1k watercolour of one picture sit on one table.

```sh
# the review's table for several sheets of one picture, and a contact sheet beside
plakat paint compare pic.jpg pic_oil_v8.png pic_oil_v9.png --sheet pic_cmp.png -o pic_cmp.md
# with a focus mask (the face, the subject) for the focus/periphery contrast
plakat paint compare pic.jpg pic_oil.png --focus pic_face.png

# paint from the director's plan, let the critic amend it from the sheet, paint again — twice at most
plakat paint from pic.jpg --medium oil-direct --make-plan --make-plan-iterate 2 -o pic_oil.png
# the verdict alone, appended to the plan the sheet was painted with
plakat paint plan pic.jpg --medium oil-direct --director --verdict pic_oil.png --plan pic_oil.plan.hjson
```

**The verdict** (`director::verdict`) is written into the plan as its own section — every reading as a comment,
every move as `key: value   # the measurement — the aim (was …)`, the moved recipe line kept as a comment. The
moves are bounded steps (a fifth of a 0..1 dial, a tenth of the key, a twentieth of the floor) under the model's
gate: only dials the plan names, never the structure but the key and the floor, never a face's dial, never a
dial the medium holds at 0. The rules, each from one reading:

| reading | move |
|---|---|
| the sheet's mean luma beyond ±0.08 of the source's | `shadow_floor` a step down (the sheet lighter) or up |
| the value spread under 0.8 of the source's / over 1.25 | `value_key` +0.1 / −0.1 |
| the focus's local contrast under 0.7, or the busy passages kept under 50% | `micro`, `adapt`, `aqua_detail`, `aqua_crisp`, `dot_fine` +0.2; `detail_restate` +0.03 (cap 0.2) |
| the focus's local contrast over 1.4 (overworked) | `micro`, `detail_restate` × 0.7 |
| the calm fields kept under 60%, or the periphery's contrast over 1.3 | `smooth`, `settle`, `aqua_soft` +0.2; `fray`, `collide` × 0.7 |
| the strong edges carried under 50% | `adapt`, `chisel` +0.2; `lost`, `sfumato` × 0.7 |
| the points of light (100+ per Mpx on the source) laid under half | `point`, `point_field`, `hotspot` +0.2 |
| the chroma under 0.7 of the source's / over 1.4 (a palette picture) | `pigments` +4 / −4 |
| bare white beyond the source's lights by 15% of the sheet (a paper medium) | `aqua_reserve`, `paper` × 0.7; `reserve` −0.1 |

`--make-plan-iterate N` runs the loop: paint, measure, amend, paint again, N more times at most, and stops when
the critic moves nothing; the intermediate sheets stay beside the output as `<stem>.i<k>.<ext>`, each verdict is
appended to the one plan file, so the final plan tells the whole story. The closeness printed with the readings
is the mean of the kept shares — a summary for the table, not a judgement of the painting: a critic that measures
can say what the sheet lost, not whether the loss was the painter's intent. That is why its steps are bounded and
its count is yours.

What `--make-plan` does **not** do: it changes nothing in the painter. A run without the flag is byte-identical
to before; a run with it is a run with a `--plan <file>`, replayable from the file. And nothing scene-specific
is baked in: a rule moves a dial only on a fact of the picture, writes the fact next to the value, and leaves
every dial the recipe does not name alone. The recipes are the place to tune a medium; the rules are the place
to tune the reading of a picture.

### Extending the art director
The v1 analyzer is deterministic (face + contrast + medium); `--make-plan` (above) is the second: the same
`PaintPlan` interface, the picture read by the crate's vision models, the medium's recipe as the base. A further
**model‑driven** art director — a vision model or a low‑resolution diffusion block‑in that proposes semantic
regions, value masses, focal point and per‑region armature resolutions (RFC §5, §12.2) — plugs in behind the same
interface, with **no change to the paint side**. The model only ever writes the plan.

---

## Honest limits

- **Faces are the ceiling.** A weight‑free stroke engine painting from a coarse armature produces a *recognizable,
  soft* face, not a skilled artist's confident facial drawing. The focal armature makes it as good as the approach
  allows; it does not make it a master's portrait.
- The plan/armature layer is what turns "a filter that scribbles" into "a painting that invents washes." That is
  the RFC's core thesis, and it holds — but it is a *painter's procedure over a structural target* (N2), not a
  claim to paint like a human.
