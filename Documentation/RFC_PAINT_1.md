# RFC PAINT-1 — `plakat paint`: a stroke-space painting engine

| | |
|---|---|
| **RFC** | PAINT-1 |
| **Status** | Draft |
| **Target** | 7.0.0 (flagship) |
| **Author** | Vladimir Ulogov |
| **Depends on** | ControlNet annotators, SAM/MobileSAM, OWL-ViT, LAION aesthetic scorer, `compile` plan machinery, `persona` landmark engine, `texture` normal/height pipeline |
| **Compatibility** | Fully additive. No existing subcommand, spec, or output format changes. |

---

## 1. Summary

`plakat paint` inverts the relationship between the diffusion model and the image.

Today the model renders pixels, and everything else in plakat conditions, decorates, or post-processes those pixels. Under PAINT-1 the model never renders the deliverable. It produces a **low-resolution structural armature** — value masses, plane assignment, coarse colour intent, semantic regions, depth, edges, pose — and a deterministic, weight-free **stroke engine** paints the final image at full resolution from that armature, using the working method of a real painter in a declared medium.

The canonical artifact is a **stroke score**: an ordered, replayable, resolution-independent record of every stroke laid, attributed to a plane, a stage, a family, and a semantic region. The image is one rendering of that score, and it is still what `plakat paint` writes by default.

### 1.1 The governing principle

> **Structure is low-resolution and model-derived. Surface is full-resolution and stroke-derived. The two never meet at the same scale.**

If the reference the engine paints from carries full-resolution detail, error minimisation will cause the engine to trace it, and the output becomes a painterly filter. An armature has no detail to trace. The brushwork must therefore invent the surface, which is what makes the strokes generative rather than decorative.

### 1.2 Why this is not a filter

Stroke-based rendering (Haeberli 1990, Litwinowicz 1997, Hertzmann 1998, Hays & Essa 2004) is a solved family of algorithms that reliably produces output recognisable as a Photoshop filter. The reason is precise: error minimisation against a full-resolution target makes each stroke meaningful with respect to *that target*, not with respect to an intent, and convergence toward the target is exactly what destroys the painting.

PAINT-1 differs on five structural points, each a hard constraint rather than a tuning parameter:

1. **The reference is an armature, not an image.** Tracing is physically impossible.
2. **The stroke budget is finite and inviolable.** When a pass's budget is spent, that pass is finished regardless of residual error.
3. **Colour is constrained to a limited palette under physical pigment mixing**, at most three pigments per mixture.
4. **The canvas carries pigment concentrations and height, not RGB.** Mixing, glazing, pickup, and mud are emergent.
5. **Stage order is derived from the medium's irreversibility structure**, not from a fixed schedule.

---

## 2. Goals and non-goals

### 2.1 Goals

- **G1.** Demote the diffusion model from renderer to art director and consultant. No pixel of the deliverable originates from a model.
- **G2.** Execute the documented working method of oil (direct and indirect), watercolour, gouache, egg tempera, pen-and-ink, and ink wash, with the differences between them derived from declared medium invariants rather than hardcoded per-medium paths.
- **G3.** Produce a first-class, inspectable, replayable stroke score with per-stroke semantic attribution.
- **G4.** Resolution independence. The same score renders at 512² or 8192² with no upscaler and no hallucinated detail.
- **G5.** Keep the entire weight-free half runnable with no GPU, consistent with `map`, `bookart`, `texture`, and `fractals`.

### 2.2 Non-goals

- **N1.** Photorealism. If the objective function is "match the reference", the system is a filter with extra steps.
- **N2.** Claiming the system "paints like a human". It executes a painter's *procedure* over a structural target. That is the claim that survives scrutiny, and it is the only one this document makes.
- **N3.** Replacing `generate`. PAINT-1 is a peer studio, not a successor.
- **N4.** Per-stroke model evaluation. Prohibitively expensive, rejected outright (§10.1).
- **N5.** Fluid simulation in v1 (§8.5).

---

## 3. Terminology

Three axes are orthogonal and have historically been conflated. This RFC fixes distinct terms for each, and the codebase must not reuse the word "layer" for any of them.

### 3.1 Plane — spatial depth ("who is closer")

A **plane** is a depth band of the composition. Planes are numbered:

```
front  →  1  →  2  →  …  →  N  →  background
```

`front` is nearest the viewer, `background` is most distant. `front` and `background` are ordinary planes with reserved names, not special cases; every rule in this document applies to them identically.

Paint order is always the reverse of plane order. To make the inversion unrepresentable in code, `PlaneOrder` exposes exactly one iterator, `far_to_near()`, and hand-written loops over plane indices are forbidden by lint.

The term is already present in plakat's artefact zone vocabulary (`middle_plan`) and is the standard painterly term — picture plane, middle distance plane.

### 3.2 Stage — a painting layer separated by a state change ("how we paint")

A **stage** is a layer of paint separated from its neighbours by a **state change**, normally a drying interval. A stage boundary is a phase transition, not a sequence number. Each stage has a distinct job; stage 1 is not a worse stage 3.

Example stage sequences, generated rather than hardcoded (§6.4):

| Medium | Stages |
|---|---|
| Oil, indirect | imprimatura → underdrawing → grisaille → dead-colour → glaze → scumble → accents → highlights |
| Oil, direct | ground → placement → shadow-mass → light-mass → halftone → accents → edges |
| Watercolour | reserve-plan → first-wash → mid-tone → darks → accents |
| Gouache | ground → flat-block → modelling → opaque-highlight → drybrush |
| Egg tempera | imprimatura → verdaccio → hatch ×N → highlight-hatch |
| Ink wash | reserve-plan → wash |

### 3.3 Family — light family and shadow family

Real painters partition the image first into a **light family** and a **shadow family** and treat them as two paintings with different rules. This is the primary partition for opaque media; plane is nested inside it.

| | Shadow family | Light family |
|---|---|---|
| Paint body | thin, transparent | opaque, thick |
| Value range | narrow | full |
| Chroma variation | low | full |
| Temperature | unified | varied |
| Detail | minimal | concentrated at the terminator |
| Height (impasto) | none | present |

**The family separation invariant:**

> No value in the light family may be darker than the lightest value in the shadow family.

A single comparison over the merged armature, trivially enforceable. Violating it is what makes a painting read as spotty rather than solid. Checked by `paint lint` and enforced during value re-key (§5.5).

### 3.4 Pass — the unit of work

A **pass** is one `(stage × family × plane)` cell. The budget allocator spends against passes; the stroke engine executes one pass at a time; the stroke score records the pass each stroke belongs to. This is the only unit that carries a stroke budget.

### 3.5 Other terms

| Term | Meaning |
|---|---|
| **Armature** | The low-resolution structural stack the model produces (§5) |
| **Figure** | A semantic subject discovered by prompt analysis; may span several planes |
| **Region** | A connected area with a single `(figure_id, plane_id)` pair |
| **Seam** | A boundary polyline between two regions on different planes |
| **Seam table** | The set of all seams with their classification (§7.5) |
| **Stroke score** | The ordered record of every stroke (§11.2) |
| **Medium profile** | The invariant set that generates a stage schedule and physics parameters (§6) |
| **Load** | A brush's current pigment content, a vector in concentration space |
| **Tooth** | The static height field of the painting surface |

---

## 4. Architecture

```
PaintSpec (HJSON)
      │
      ├─► Figure discovery ──────────── prompt analysis (reuse: layered generation)
      │
      ├─► Armature construction ─────── SDXL + procedural perspective  [MODEL, GPU]
      │       per-figure value/colour fields, planes, depth, edges,
      │       regions, saliency, pose skeletons
      │
      ├─► Armature merge ────────────── global value re-key, single key light,
      │       global palette quantisation, plane assignment,
      │       family split, seam table
      │
      ├─► Plan compilation ──────────── medium profile → stage schedule,
      │       budget allocation                              [NO GPU]
      │
      ├─► Stroke synthesis ──────────── orientation field, placement, geometry,
      │       pigment mixing                                  [NO GPU]
      │
      ├─► Rasterisation ─────────────── bristle physics on pigment canvas,
      │       height accumulation                             [NO GPU]
      │
      ├─► Critic loop (optional) ────── pass-level accept/reject  [MODEL, GPU]
      │
      └─► Output ───────────────────── stroke score + image + sidecar (§11.5)
```

Everything below the merge is a pure function of `(plan, seed)` and runs with no GPU. Everything above it is the model acting as a consultant. This is the same `structured data → deterministic resolve → render` shape as `map`, `bookart`, and `texture`.

---

## 5. The armature

### 5.1 Channels

The armature is a stack, not an image. Resolutions are deliberate: structure survives downsampling, detail does not.

| Channel | Resolution | Source | Purpose |
|---|---|---|---|
| Value masses (notan) | 64² per figure | `control-generate mode: blockin` | stage 1–2 targets |
| Colour field | 64² per figure | low-step SDXL, KM-quantised | pigment loading intent |
| Region map + labels | 512² | SAM / MobileSAM + OWL-ViT | attribution, seams, budget |
| Depth | 512² | depth annotator | plane assignment, recession |
| Surface normals | 128² | depth-derived | terminator estimate, form flow |
| Edge map + quality | 512² | lineart / softedge | seam classification |
| Pose skeleton | 512² | OpenPose ControlNet | figure geometry, gesture |
| Perspective frame | 512² | **procedural** (§5.3) | ground plane, recession |
| Saliency | 128² | focal inference | budget allocation |

Note the split: **value and colour downsample; geometry does not.** Lines dissolve into smudge at 64². Wireframe, skeleton, and edge channels stay at 512².

### 5.2 Per-figure armatures

Figures are **not** rendered into a shared global 64² grid. A small background object at global 64² is sub-pixel, which contradicts the constraint carried over from layered generation that fine detail must not be stripped.

Instead, each discovered figure is rendered at 512² within its own bounding box, downsampled to its own armature resolution, and placed into the composition at its actual scale. Armature fidelity is therefore independent of canvas coverage.

Per-figure resolution scales with importance:

| Role | Resolution |
|---|---|
| Focal figure | 96² |
| Supporting figure | 64² |
| Background mass | 32² |

These values are the primary unknown in the design and are the first thing P0 measures (§12.1).

### 5.3 Perspective is procedural, not generated

SDXL cannot produce consistent perspective. It produces images that look perspectivally plausible without coherent vanishing points, and that incoherence survives downsampling and is amplified by a paint pass that adds no new structure.

Horizon, vanishing points, ground grid, and recession scale are therefore computed as a pure function of a camera specification — the same geometry-engine approach as `map` — and supplied to SDXL as a ControlNet conditioner. The model fills mass into a frame it did not invent.

### 5.4 What SDXL is good for here

- **Pose skeletons** via OpenPose. A skeleton is semantically typed geometry; every line means something specific. A legitimate armature primitive.
- **Value mass distribution** via `blockin`.
- **Coarse colour intent.**

What it is not good for: generic line art as a primary armature (undifferentiated strokes carry no side-of-boundary information), and perspective (§5.3).

### 5.5 Merge

Figure discovery yields a cast, not a composition. The merge resolves six things, all cheap at armature resolution:

1. **Plane assignment.** Cluster the depth histogram into 3–5 bands, intersect with the region map, order back to front. Reuse the `multiperson` spatial resolver for relative placement.
2. **Global value re-key.** Each figure generated alone uses the full value range for itself. Compress every figure's value field into its assigned slot in one global value scale. This is the step that makes the picture read as a single image.
3. **Single key light.** Re-shade each figure to the declared light direction using the normals channel. Trivial at 128²; intractable at 512².
4. **Global palette quantisation.** Every figure quantises to the same limited palette (§8.3). This is the unification that edge blurring failed to provide, and it costs nothing.
5. **Family split.** Partition into light and shadow families using the normals channel and the declared key light, then enforce the family separation invariant (§3.3).
6. **Recession.** Per-plane value compression and contrast reduction, applied after merge rather than per figure.

Upsampling from armature to canvas resolution uses **edge-aware joint upsampling** (guided filter, lineart map as guide). Bilinear upsampling makes region boundaries wobble and causes strokes to straddle seams incorrectly.

**Explicitly not done at merge:** blending at figure boundaries. Each boundary is classified and recorded (§7). A boundary is a decision, not an artifact to hide. This is the single most important difference from the cut-paste-blur compositing previously rejected.

---

## 6. Medium profiles

A stage schedule is not a property of the painter. It is a consequence of the medium's **irreversibility structure**. Declare the invariants and the schedule derives itself. There are no per-medium code paths.

### 6.1 Invariant set

```hjson
medium: {
  name: oil-direct
  value_direction: mid_out        # dark_to_light | light_to_dark | mid_out
  opacity: opaque                 # opaque | transparent | mixed
  white_source: pigment           # pigment | surface
  reversibility: hours            # none | minimal | hours | days
  stage_budget: 1                 # max stages before degradation
  height: impasto_lights          # none | impasto | impasto_lights
  dry_shift: none                 # none | lighter | toward_mid
  pickup: 0.65                    # 0.0 – 1.0
  rework: nondestructive          # nondestructive | destructive
  families: split                 # split | unified
  subtractive: allowed            # allowed | forbidden
  mark_model: continuous          # continuous | density
  finish_policy: uniform          # uniform | back_to_front
}
```

### 6.2 Profile table

| Invariant | Oil indirect | Oil direct | Watercolour | Gouache | Tempera | Pen-ink | Ink wash |
|---|---|---|---|---|---|---|---|
| Value direction | dark→light | mid-out | **light→dark** | mid-out | dark→light | light→dark | light→dark |
| Opacity | mixed | opaque | transparent | opaque | transparent | transparent | transparent |
| White source | pigment | pigment | **surface** | pigment | pigment | **surface** | **surface** |
| Reversibility | days | hours | minimal | minimal | none | none | none |
| Stage budget | 8 | 1 | **3** | 3 | 40 | 1 | 1 |
| Height | impasto lights | impasto | none | none | none | none | none |
| Dry shift | none | none | lighter | toward mid | lighter | none | none |
| Pickup | 0.4 | 0.65 | 0.15 | **0.8 destructive** | 0.05 | 0.0 | 0.9 |
| Families | split | split | unified | split | split | unified | unified |
| Subtractive | allowed | **allowed** | lifting | forbidden | forbidden | forbidden | forbidden |
| Mark model | continuous | continuous | continuous | continuous | **density** | **density** | continuous |
| Finish policy | uniform | uniform | back_to_front | back_to_front | uniform | uniform | back_to_front |

### 6.3 Consequences that change architecture, not parameters

**Watercolour needs a reserve plan before stroke one.** White comes from the paper and cannot be recovered. The highest value band of the merged armature becomes a **reserve mask** that no stroke may enter. This is a first-class planning artifact with no oil analogue, and it is the clearest case of the armature earning its existence. Getting it wrong produces a gouache painting.

**Stage budget constrains plane allocation.** If depth decomposition yields five planes and the medium permits three stages, planes must be merged before painting. The medium profile feeds back into plane allocation, not only into stroke physics.

**Density media need a different rasteriser path.** Pen-ink and egg tempera build value by mark density — hatching, crosshatching, stippling, *tratteggio* — not by pigment concentration. Deposit is binary and the brush is a mark generator whose direction and spacing derive from the orientation field. A separate code path (§8.6), not a parameter change.

**Pen-ink and ink wash are two media.** One is a density model with zero pickup; the other is a continuous loaded-gradient stroke — a single brush carrying dark-to-light along its own length — with extreme absorbency and no correction. Conflating them gets both wrong.

**Mud is emergent.** Too many subtractive glazes converging toward neutral is exactly what Kubelka-Munk predicts. The engine can *detect* that a watercolour plan will go muddy by simulating it, rather than encoding a rule that says "maximum four stages". Gouache's destructive rework falls out of high pickup plus fast drying by the same mechanism.

### 6.4 Stage schedule generation

```
generate_schedule(medium, plan):
    stages = []
    if medium.white_source == surface:
        stages += [reserve_plan]
    if medium.opacity != transparent:
        stages += [ground]
    if medium.families == split:
        stages += [value_stage(medium.value_direction)]     # grisaille / notan
        stages += [shadow_mass, light_mass]
    else:
        stages += [value_stage(medium.value_direction)]
    stages += colour_stages(medium.stage_budget - len(stages))
    stages += [halftone] if medium.reversibility >= hours
    stages += [accents]
    if medium.white_source == pigment:
        stages += [highlights]                # irreversible operation last
    return stages
```

The last rule generalises: **the irreversible operation goes last.** Opaque highlights in oil, darkest darks in watercolour.

---

## 7. Seams: how planes intersect during painting

This is where the previous compositing approach failed, so the mechanism differs deliberately.

### 7.1 A seam is a termination policy, not a mask

A painter does not paint objects into masked regions. They paint the background *through* where the object will be, then paint the object *over* it, and the boundary comes into existence as a consequence of where the last stroke stopped.

> **Nothing is ever clipped to a mask. The seam is a stroke termination policy.**

Mask-first produces cutouts. Stroke-termination-first produces edges.

### 7.2 Opaque media: forward overspray

Painting far to near, when plane *i* is painted, every plane behind it is finished and every plane in front is bare. Plane *i*'s paint region is its visible region dilated by the maximum brush radius, **but only into not-yet-painted territory**:

```
P_i = V_i ∪ ( dilate(V_i, r_max) ∩ ⋃_{j < i} V_j )
```

Dilating backwards would destroy completed work. Dilating forwards is free, because it gets covered. This asymmetry eliminates halos, otherwise the most recognisable stroke-rendering artifact.

### 7.3 Transparent media: reservation

Light never goes over dark, so covering is impossible. The far plane paints *around* the near object, reserving its silhouette eroded by a small negative margin so the later wash meets it cleanly rather than leaving a white gap:

```
P_i = V_i \ dilate( ⋃_{j < i} V_j , −m )
```

Occlusion in watercolour is a reservation problem, not a covering problem. Same seam table, opposite sign, derived from `medium.opacity` rather than coded twice.

### 7.4 Seam classification

| Class | Termination | Typical use |
|---|---|---|
| **Hard / cut-in** | stroke ends exactly on the seam, brush tangent to it | focal point only |
| **Soft** | crosses by ≈0.5 brush radius, pickup blends | most of the picture |
| **Lost** | crosses freely, 1–2 radii, shared value | hair, foliage, shadow-side |

**Hard-edge cap: 10–20% of total seam length**, concentrated where saliency is highest. A painting where every boundary is cut in reads as collage — the same failure as cut-paste-blur, achieved by hand. Enforced by the planner, verified by `paint lint`.

### 7.5 Seam extraction and the T-junction validator

Boundaries are extracted from the plane index map via marching squares and split at junctions. Per segment the seam table records: plane pair, figure pair, polyline, length, depth gap, value contrast across it, peak saliency, material class, assigned edge class.

A **T-junction** — where region A's boundary terminates against region B's boundary — implies A is behind B. This is a classical occlusion cue independent of the depth estimate, which makes it a validator. If depth ordering and T-junction ordering disagree, `paint lint` surfaces the conflict rather than painting it wrong silently.

### 7.6 Three cases that break the simple model

**Interleaving.** An arm in front of a torso puts one figure on multiple planes. `figure_id` and `plane_id` are therefore independent attributes of a region, not a hierarchy. Two constraints follow:

- **Palette lock per figure across planes.** Otherwise the arm loads from a different mixture than the torso and comes out a visibly different colour despite being the same flesh.
- **Wet-state tracking per figure.** The near part paints several passes later, by which time the far part may have dried.

**Contact seams.** Feet on ground, object on table. Neither plane owns the seam, and letting z-order decide makes things float. Contact regions are a shared unit painted after both planes, with their own budget line, and they carry the darkest values in the picture. `product`'s grounded contact shadow work is the physical precedent.

**Partial-coverage boundaries.** Hair, foliage, smoke, water edges are not boundaries; they are coverage fields in [0,1]. Strokes are laid individually rather than as mass, and stroke count scales with coverage rather than area. Per unit of seam length these are by far the most expensive thing in the painting — a portrait with loose hair can exceed a naive budget by an order of magnitude — and the allocator must reserve for it explicitly.

---

## 8. The stroke engine

### 8.1 The canvas is not RGB

The canvas carries **per-pigment concentrations plus height, wetness, and static tooth**, converted to sRGB only on output. This is what makes mixing, glazing, pickup, and mud physical rather than faked.

```
canvas.concentration : [N_pigments] f16   # N ≤ 8
canvas.height        : f16
canvas.wetness       : u8
canvas.tooth         : u8                 # static, from surface profile
```

Memory at 4096², 8 pigments, f16: ≈270 MB plus auxiliaries. Acceptable. Tile above 4096².

### 8.2 Orientation field

Luminance gradient rotated 90° gives the isophote direction but is unreliable in flat regions. The field is built by:

1. Structure tensor on the armature value channel.
2. Edge tangent flow smoothing (Kang et al. 2007), or RBF interpolation of orientations outward from high-confidence edge pixels (Hays & Essa 2004).
3. Bias by depth-derived surface normals so strokes follow **form** across smooth passages rather than noise.
4. Per-figure override from the pose skeleton where available — limb axis dominates.

### 8.3 Palette and pigment mixing

Mixing is **Kubelka-Munk two-constant**, not RGB interpolation. RGB lerp of blue and yellow gives grey; KM gives green, and people see the difference without knowing why.

- 4–8 named pigments per palette. Built-in: `zorn`, `split-primary`, `verdaccio`, `earth`, `limited-landscape`, `sumi`.
- Each mixture is solved as the pigment combination minimising ΔE in CIELAB against the target colour, **subject to at most three pigments per mixture**. This is how a painter actually mixes, and the constraint generates colour harmony for free.
- Palette is locked per figure across planes (§7.6).

### 8.4 Stroke synthesis

**Placement.** Error map weighted by saliency and semantic importance, sampled at adaptive density (gradient magnitude plus tone), restricted to the pass's plane × family region, hard-capped by the pass budget.

**Geometry.** Catmull-Rom spline grown along the orientation field from the placement point, with maximum curvature and maximum length limits, terminating when the local armature colour deviates past tolerance from the stroke's loaded mixture (Hertzmann's rule) or when a seam termination policy fires (§7.4).

**Width** from the pass's brush radius, with end taper and seeded jitter.

**Colour.** One mixed load per stroke, sampled at the stroke origin; optionally two for a deliberately dirty brush. Not per-pixel.

**Minimum brush size per stage is inviolable.** If a pass's budget is exhausted, that pass is finished, residual error notwithstanding.

### 8.5 Stroke physics — bristle bundle with pickup

Three tiers were considered. Tier A (geometric ribbon, alpha composite, no pickup) is cheap and looks like a filter; rejected as default. Tier C (shallow-water or lattice-Boltzmann pigment advection) is justified only where the medium's behaviour *is* the look; deferred behind `medium: watercolour` to a later phase. **Tier B is the v1 model.**

Per bristle: spine offset, `load` (concentration vector), `wetness`.
Per integration step along the spine:

```
contact = pressure · tooth_contact(texel)
deposit = min(load, k_d · contact · (1 − saturation))
pickup  = k_p · wetness(texel) · (1 − |load| / load_max)

load    ← load − deposit + pickup · concentration(texel)
canvas  += deposit
height  += deposit · viscosity
```

No solver, no iteration, cost is O(spine length × bristles), fully deterministic.

The payoff worth stating explicitly: because pickup mixes canvas pigment back into the brush, consecutive strokes in a region **automatically harmonise**. The dirty brush is a real phenomenon and it delivers colour unity with no explicit harmony rule. Combined with the ≤3-pigment constraint, that is most of what makes paint look like paint.

**Impasto** accumulates into the height buffer and is relit at output by a Blinn-Phong pass using the declared key light. Reuses the `texture` normal-map machinery.

### 8.6 Density mark model

For `mark_model: density` (pen-ink, egg tempera), deposit is binary and strokes are generated as hatch sets:

- Direction from the orientation field, spacing inversely proportional to target darkness.
- Crosshatch angle offsets accumulate across stages.
- Stipple mode for tempera fine passages.
- Value is a function of mark density, not concentration. No pickup, no wetness.

### 8.7 Subtraction and negative painting

Two operations real painters use structurally, not as error correction. Both are cheap on a pigment canvas.

**`wipe` stroke.** Removes wet pigment and partially exposes the stage beneath — scraping, tonking, wiping back. Sargent scraped down entire heads and restated them routinely; Turner wiped extensively. Watercolour lifting is the same operation with different parameters. Gated on `medium.subtractive`.

**Negative painting.** Defining a shape by painting the space around it. The watercolour reserve logic (§6.3) is already half of this; generalising it gives a technique central to watercolour and common in oil backgrounds.

### 8.8 Ordering constraint

Pickup makes stroke order globally significant: stroke 400 depends on what strokes 1–399 left wet. Therefore **overlapping strokes within a pass cannot be rasterised in parallel.** The score replays in order, which is also what makes it a faithful record — and what makes `--resume` correct (§11.7).

Parallelism lives on the independent axis: disjoint regions within the same pass, and tiles for strokes that do not span tile boundaries. This falls out of the plane/region structure, constrains the rasteriser interface, and must be designed in from the start rather than retrofitted.

---

## 9. Budget allocation

With total budget *B*:

```
budget(stage, family, plane) = B · w_stage · w_family · w_plane · r_recession

w_plane      ∝ saliency mass of the plane
r_recession  ≈ 0.35 at background → 1.0 at focal plane
```

The recession multiplier produces atmospheric perspective — fewer, larger, lower-contrast strokes in the distance — from the same term. It is paired with a **per-plane minimum brush size that grows with depth**, so the far background cannot receive fine detail even when the error map demands it.

Two hard rules:

- The final two stages (accents, highlights) are restricted to the focal plane. The strongest "a person made choices here" signal available.
- Partial-coverage regions (§7.6) receive a reserved allocation before general distribution.

### 9.1 Style presets are budget and schedule, not LoRAs

| Preset | Strokes | Character |
|---|---|---|
| `sargent` | ≈300 | brutal minimum size, detail at the terminator only |
| `alla-prima` | ≈600 | uniform finish, high pickup |
| `tonal` | ≈1200 | soft edges throughout, narrow value range |
| `impressionist` | ≈4000 | small strokes, high jitter, broken colour |
| `impasto` | ≈900 | heavy height accumulation in lights |
| `tratteggio` | ≈40000 | density model, tempera |

Same plan plus same palette plus same brush yields **the same hand across a whole series**, which is a standing weakness of LoRA-based style consistency.

### 9.2 Finish policy

- **`uniform`** — stage *n* runs across every plane before stage *n+1* begins. The whole canvas stays at one state of finish, which is why alla prima paintings read as unified. Oil (both methods), impressionist, tempera.
- **`back_to_front`** — each plane completes through all stages before the next begins. Watercolour, gouache illustration, ink wash, where reservation forces it.

---

## 10. The model's role

Four tiers, descending involvement.

| Tier | Role | Determinism | v1 |
|---|---|---|---|
| **Reference** | armature construction (§5) | pure fn of (spec, seed, armature) | ✅ |
| **Analyst** | depth / segmentation / saliency / pose shape the plan | deterministic given aide outputs | ✅ |
| **Critic** | aesthetic or CLIP score accepts/rejects a **pass** | seeded, reproducible | ✅ gated |
| **Gradient source** | SDS over a differentiable brush rasteriser | optimisation, slow, incoherent | ❌ research |

### 10.1 Critic loop

Pass-level only. Score the canvas before and after a pass with the LAION aesthetic predictor plus CLIP-vs-prompt; accept or reject the whole pass; cache rejections in a tabu corpus so the loop never retries a rejected configuration. This is the topology already proven in `compile --improve` with the smysl corpus.

Per-stroke evaluation would require thousands of forward passes per painting and is rejected.

### 10.2 Tier 4 is explicitly out of scope

A differentiable wet-pigment rasteriser plus thousands of SDS iterations (the CLIPasso / VectorFusion lineage) is technically reachable — candle has autograd — but historically yields evocative-but-incoherent results and represents a multi-month spike. It does not go on a release milestone.

---

## 11. Interfaces

### 11.1 PaintSpec

```hjson
{
  paint: {
    version: 1
    subject: "a fisherman mending nets on a harbour wall, late afternoon"

    medium: oil-direct
    style:  sargent

    surface: {
      size: 1024x1365
      ground: toned
      ground_pigment: raw-umber
      tooth: linen-medium
    }

    palette: zorn

    light: {
      key: { azimuth: -35, elevation: 40 }
      ratio: 4.0
      temperature: warm
    }

    composition: {
      focal: "the sitter's hands"
      value_key: low
      finish: uniform
      camera: { fov: 40, horizon: 0.58, vanishing: auto }
    }

    budget: { strokes: 300 }

    planes: auto        # or explicit: [ { name: front, figures: [...] }, ... ]

    armature: {
      source: sdxl
      focal_res: 96
      support_res: 64
      background_res: 32
      steps: 8
      fast: lcm-sdxl
    }

    critic: { enabled: true, scorer: laion-aesthetic, tabu: true }
    seed: 42
  }
}
```

### 11.2 Stroke score

Line-oriented, append-only, replayable. One record per stroke.

```
# plakat stroke score v1
# spec-hash: 3f9a…  palette: zorn  medium: oil-direct  seed: 42
S 00001 stage=shadow-mass family=shadow plane=2 figure=fisherman region=r14 \
        spline=[(412,880),(438,902),(471,911)] w0=38 w1=31 taper=0.4 \
        mix=[ochre:0.52,black:0.31,red:0.17] wet=1.0 press=0.8 \
        term=soft seam=s07 just=error:0.34,saliency:0.61
W 00002 stage=shadow-mass plane=2 region=r14 spline=[…] strength=0.6
C 01000 checkpoint=canvas-01000.z                 # §11.8
```

Guarantees:

- **Replay.** Re-render at any resolution.
- **Truncation.** Any intermediate state is reconstructible.
- **Query.** `paint explain --region "left cheek"` returns every stroke describing that plane of the form.
- **Resume.** Append-only plus deterministic order means an interrupted paint continues (§11.7).
- **Provenance.** The score *is* the construction record, beyond what `--etch` provides.

### 11.3 Definition of a meaningful stroke

Operational and testable. A stroke is meaningful iff it:

1. belongs to a named stage, family, and plane;
2. is attributed to a semantic region;
3. measurably reduces a saliency-weighted objective, recorded in `just=`;
4. carries that justification in the score.

### 11.4 One rasteriser, three entry points

Score → pixels is a single function. It is reachable three ways, and the distinction matters because only the first needs a GPU:

| Entry | Input | Does |
|---|---|---|
| `paint <SPEC>` | spec | armature → plan → strokes → raster. Writes score **and** image |
| `paint replay <SCORE>` | score | raster only, at any size. No GPU |
| `paint export <SCORE>` | score | derived products that are not the canonical image |

The score is authoritative under re-render; the image is still what lands on disk by default. §1 makes a claim about authority, not about output.

### 11.5 Default outputs

`plakat paint <SPEC>` writes, at the spec's declared size:

| File | Content |
|---|---|
| `plakat-<seed>.png` | the painting, impasto-relit, sRGB |
| `plakat-<seed>.strokes` | the stroke score (§11.2) |
| `plakat-<seed>.json` | recipe sidecar — spec hash, medium, palette, plan digest, stroke count, budget actuals |

The PNG carries the A1111 `parameters` tEXt chunk as every other plakat output does, so `metadata`, `clone`, `photos --import`, and `--etch` work unchanged. `--format webp`, `--no-metadata`, `--import <album>`, and `--grid` are inherited from the existing output path. `--size` overrides the spec. `--no-image` writes only the score, for pipelines that render later or elsewhere.

### 11.6 Export targets

`paint export` covers what a flat image cannot carry. All of these are derived, never authoritative, and all are regenerable from the score.

| Target | Output | Why |
|---|---|---|
| `height` | 16-bit PNG | impasto relief; feeds `texture`, external relighting |
| `normal` | PNG | derived from height, engine-ready |
| `separations` | OpenRaster `.ora` | one raster layer per **stage**, or per **plane**, or the product |
| `stages` | contact sheet | the painting at the end of each stage, one grid |
| `armature` | PNG stack | value, colour, plane index, depth, normals, edges, saliency, seams |
| `seams` | HJSON + overlay | the seam table with its classification, rendered over the image |
| `heatmap` | PNG | stroke density, budget spend by pass |

`separations --by stage` is the one worth calling out: it is the artist-legible form of the whole thesis. An oil painting exports as imprimatura / grisaille / dead-colour / glaze / accents as editable layers, which no diffusion pipeline can produce. ORA is a zip of PNGs plus XML — pure Rust, no new dependency family, opens in Krita and GIMP.

### 11.7 CLI surface

```
plakat paint <SPEC>              paint from a PaintSpec → image + score + sidecar
plakat paint new                 scaffold a spec
plakat paint lint <SPEC>         validate; family invariant, hard-edge cap,
                                 stage budget vs plane count, T-junction conflicts
plakat paint show <SPEC>         resolved plan: stages, planes, budget table
plakat paint armature <SPEC>     build and dump the armature stack only
plakat paint plan <SPEC>         compile the plan, no painting  [NO GPU]
plakat paint replay <SCORE>      re-render a score at any size  [NO GPU]
plakat paint export <SCORE>      derived products (§11.6)       [NO GPU]
plakat paint explain <SCORE>     query strokes by region / stage / plane
plakat paint timelapse <SCORE>   frames → mp4 / gif             [NO GPU]
plakat paint palette <NAME>      inspect pigments, KM constants, mixing gamut
plakat paint diff <A> <B>        compare two scores
plakat paint verify <SCORE>      halo / cutout / mud detection (§12.2)
plakat paint from <IMAGE>        repaint an existing image in stroke space
```

Flags on `paint` and `replay`:

```
--size WxH                      override the spec's surface size
--until N                       stop after stroke N (or replay to that point)
--stage NAME[,NAME…]            render only these stages
--plane front|1..N|background   render only these planes
--preview-every N               write a progress frame every N strokes
--no-image                      score only
--resume                        continue an interrupted paint from its score
```

`--until`, `--stage`, and `--plane` are the same filter applied at three points, which is what makes intermediate states free: truncation is replay with a stroke cap.

`--resume` matters more here than in `generate` — a 40k-stroke tempera render is long. It is correct only because stroke order is already globally significant from pickup (§8.8); the constraint that blocks parallelism is the one that makes resume sound.

Plus integration on the existing rails: `scenario` (`type: paint`), `compile` (`paint:` block), `run` (`plakat.paint.*` Bund words), and `plakat::api::Paint`.

### 11.8 Checkpoints

Naive replay is O(n) per frame, so a 40k-stroke timelapse is quadratic. The score therefore carries periodic **canvas checkpoints** — compressed snapshots of concentration, height, and wetness at stroke *k* — written every `checkpoint_every` strokes (default 1000).

Replay to stroke *n* seeks the nearest checkpoint ≤ *n* and replays forward only from there. This makes `--until`, `timelapse`, and `--resume` one mechanism rather than three, and it is the difference between a timelapse costing minutes and costing hours.

Checkpoints are a cache, not part of the record: a score with its checkpoints stripped still replays correctly, only slower. `paint export`, `diff`, and `verify` ignore them.

---

## 12. Verification

### 12.1 The armature threshold experiment

The central unknown: the armature must be coarse enough to prevent tracing and coherent enough to be paintable. Resolutions in §5.2 are a guess.

**Protocol.** Sweep per-figure armature resolution over {32, 48, 64, 96, 128, 192}. At each point measure:

- **Traceability** — correlation between the painted output and a full-resolution render of the same prompt. Should *fall* with resolution.
- **Structural fidelity** — does the intended subject survive? Human-scored plus CLIP-vs-prompt.
- **Prompt drift** — CLIP distance between output and prompt.

The answer is the resolution at which structure survives and detail dies, and it is almost certainly a function of figure role rather than a single number.

### 12.2 Automatic failure detection

Two failure modes are diagnosable from the seam table without human judgement, and both belong in the P0 harness rather than being eyeballed.

**Halo** — a bright or dark rim tracing every object. Signature: a value spike in a narrow band along seams. Cause: masking winning over overspray; check dilation direction (§7.2).

**Cutout** — objects reading as pasted despite correct values. Signature: hard-edge fraction above the cap, or boundary pickup too low. This is the failure already encountered once by a different route, and it is the one that matters.

A third, medium-specific: **mud** — chroma collapse toward neutral in a glazed passage, predicted by simulating the KM stack before painting (§6.3).

### 12.3 Acceptance criteria

| | Criterion |
|---|---|
| **A1** | A blind panel distinguishes PAINT-1 output from Hertzmann-style SBR output at above chance, preferring PAINT-1 |
| **A2** | Replay at 512² and 4096² from one score yields no structural divergence |
| **A3** | `paint plan`, `replay`, `export`, and `timelapse` complete with no GPU present |
| **A4** | Score replay is byte-identical on the same machine and backend, given identical `(plan, seed)` |
| **A5** | Replay from a checkpoint is byte-identical to replay from stroke 1 |
| **A6** | Halo and cutout detectors report clean on the corpus set |
| **A7** | A watercolour spec with a reserve plan produces unpainted paper in the reserved region — zero stroke coverage |
| **A8** | Family separation invariant holds on every rendered corpus image |
| **A9** | Hard-edge fraction stays within the configured cap |

---

## 13. Phasing

| Phase | Content | Gate |
|---|---|---|
| **P0** | Pigment canvas, bristle brush, KM mixer, coarse-to-fine over an input image, halo/cutout detectors, armature threshold experiment (§12.1) | If output reads as a filter with budget and palette constraints active, the core thesis is wrong — stop here, cheaply |
| **P1** | `PaintSpec`, plan compiler, stage generator, stroke score, checkpoints, replay, export, timelapse, `paint from <IMAGE>`. Oil direct and gouache only | Shippable standalone |
| **P2** | Armature construction, figure discovery, merge, seam table, planes, family split. Adds watercolour (reservation) and pen-ink (density) | The flagship |
| **P3** | Critic loop with tabu memory; indirect oil, tempera, ink wash; subtraction and negative painting | |
| **P4** | Differentiable rasteriser and SDS | Research branch, no milestone |

---

## 14. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Output reads as a filter | **Fatal** | Hard budget, KM palette, impasto relight, edge policy, minimum brush size. P0 gates on it |
| Armature too coarse to be paintable | High | §12.1 sweep; per-role resolution |
| Faces destroyed at small scale | High | If SCRFD bbox < N px, reserve minimum brush and landmark-anchored budget via the persona engine, or composite the face rather than painting it |
| Figure cast without composition | High | LLM art-director stage producing the composition plan — prose in, structured plan out, same shape as `compile`. Without it: correctly rendered figures with no design, which is worse than one flawed render |
| Terminator estimate fails | Medium | Normals plus declared key light works on a sphere and fails on foliage. Fall back to `families: unified` when normal confidence is low |
| Armature cost — one render per figure | Medium | Few-step presets, resident pipeline. Bounded and front-loaded |
| Ordering blocks parallelism | Medium | Parallelise on disjoint regions and tiles; design the rasteriser interface for it from the start (§8.8) |
| Timelapse cost quadratic in stroke count | Medium | Checkpoints (§11.8) |
| Partial-coverage budget blowout | Medium | Reserved allocation before general distribution |
| Canvas memory above 4096² | Low | f16 concentrations, tiling |

---

## 15. Open questions

1. **Armature resolution per figure role.** The P0 sweep answers this; everything else is downstream of it.
2. **Composition plan schema.** How much does the art-director stage declare — focal point, value key, mass balance — versus infer from the armature? This is arguably a second RFC.
3. **Palette selection.** Declared, inferred from the prompt, or scored across candidates?
4. **Stroke score format.** Text is inspectable and diffable; binary is compact. A 40k-stroke tempera score in text is large. Possibly both, with text canonical for interchange and binary for storage.
5. **Checkpoint interval and compression.** Default 1000 is a guess. The tradeoff is disk against replay latency, and it likely differs between `timelapse` (wants dense) and `--resume` (wants sparse).
6. **Contact-seam budget.** §7.6 asserts contact regions need their own line. The fraction is unknown.
7. **Does `paint from <IMAGE>` want the full plan machinery**, or is the Hertzmann path sufficient for the repaint case?
8. **Interaction with `--etch`.** The score supersedes etching for plakat-native provenance, but etching still matters for derivatives. Does the score hash get etched into the PNG?
