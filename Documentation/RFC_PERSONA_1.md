# RFC PERSONA-1 — Controllable persona composition

**Status:** draft
**Track:** P (persona)
**Scope:** a new `plakat persona` subcommand, a `PersonaSpec` HJSON schema, a
deterministic spec resolver, a face/figure geometry engine, a localized-detail
subsystem covering marks, jewelry and dentition, an identity-anchor pipeline, a
measurement scorecard, and a Q/A composition TUI.
**Compatibility:** fully additive. No existing flag, scenario field, Bund word,
sidecar key, or default image output changes.

---

## 1. Summary

`plakat persona` turns a declarative HJSON description of a human — facial
geometry, colouring, marks, scars, birthmarks, hair, facial hair, dentition,
jewelry, and build — into a *reproducible person* that renders recognisably
across every model family plakat supports.

The spec is authored by hand, extracted from a photograph, generated from prose,
or composed interactively in a guided Q/A TUI (`plakat persona --tui`). It
resolves through a deterministic pipeline into per-family conditioning, a
geometric conditioning map, a set of landmark-anchored localized details, and a
curated identity reference set. Every render can then be *measured* against the
spec that produced it, and the measurement drives both automatic retry and
attribute-targeted repair.

The through-line: **a persona is a spec, an identity artifact, and a
measurement — not a prompt fragment.**

---

## 2. Motivation

### 2.1 The problem

Describing a person inline in a generation prompt produces uneven results. The
unevenness has four distinct causes, and they require four distinct remedies.
Conflating them is why prompt-engineering approaches plateau.

**2.1.1 Underdetermination.** A prompt specifies a *distribution* over faces, not
a face. Identical prompt, different seed, different person. Adding adjectives
narrows the distribution asymptotically at best, and on CLIP-family encoders the
77-token budget is exhausted long before the face is pinned. No prompt is ever a
sufficient identity anchor.

**2.1.2 Attribute entanglement.** CLIP text encoders behave close to
bag-of-words for modifiers. `green eyes, auburn hair, olive jacket` leaks colour
across all three noun phrases. Longer, denser descriptions make entanglement
*worse* — each additional modifier is another opportunity for cross-binding. The
failure is not that the model ignores the description; it is that the model
applies it to the wrong slots.

**2.1.3 Encoder heterogeneity.** plakat spans CLIP-L, dual CLIP-L+CLIP-G, triple
encoder stacks, T5-XXL, and Gemma-2. These have radically different
prompt-following characteristics and radically different optimal input shapes. A
paragraph that Gemma-2's complex-human-instruction path rewards is a paragraph
that overruns CLIP-L's context entirely. One authored description cannot be
optimal — or even equivalent — across all of them.

**2.1.4 Localized detail is not steerable by text at all.** A mole below the left
eye, a two-centimetre scar through the right brow, a septum ring, a chipped upper
incisor: these are *positional, small-area* features. Text conditioning has no
mechanism to place them, and diffusion sampling will scatter them at random or
omit them entirely. This is a categorically different failure from the first
three and needs a categorically different remedy — anchoring and compositing,
not better prompting (§8).

### 2.2 Why the existing surfaces do not solve this

plakat already has substantial identity machinery, but every piece of it is
**photo-driven**:

| Existing surface | Requires | Gap |
|---|---|---|
| `portrait` (IP-Adapter-Plus-Face / FaceID) | reference photograph | cannot synthesise a person who does not exist |
| face-swap stack (SCRFD / ArcFace / inswapper) | source face image | same |
| `multiperson` | per-person photographs | same |
| UI People library | imported photos or scenario personas | stores identities, cannot *author* them |
| scenario `personas:` | free-text description | inherits every failure in §2.1 |
| `compile` prompt rewriting | prose | family-aware, but unstructured and lossy |
| artefact compositing | authored PNG cutouts + named zones | the right *mechanism* for localized detail, but not wired to anatomy |

There is no path from *"a person I am inventing"* to *"a stable identity"*. This
RFC supplies it, and — importantly — reuses every mechanism above once the
identity exists, rather than replacing any of them. The artefact compositing path
in particular is repurposed directly by §8.

### 2.3 Baseline measurement is a prerequisite

Before implementation begins, the current unevenness must be quantified,
otherwise the feature's success is unfalsifiable.

**Baseline protocol.** For each model family: one fixed, densely-descriptive
person prompt; N = 32 seeds; render at the family's native size; compute the
pairwise ArcFace cosine matrix over detected faces; report mean, median, and 5th
percentile. Additionally report the detection-failure rate (renders with zero or
multiple detected faces), and — for §2.1.4 — the **localized-detail hit rate**:
how often a prompted mole, scar, or piercing appears at all, and how often it
appears on the correct side of the face.

These numbers are the control. Every subsequent phase reports the same
statistics. The artefact is committed as a corpus entry.

---

## 3. Goals and non-goals

### 3.1 Goals

- **G1.** A declarative, human-editable, version-tolerant HJSON schema for a
  person, covering facial geometry, colouring, hair, facial hair, dentition,
  marks (moles, freckles, scars, birthmarks, tattoos), jewelry and piercings, and
  figure.
- **G2.** A deterministic resolver: `(spec, lexicon)` → per-family conditioning,
  geometric conditioning maps, and resolved detail placements, as a pure
  function, byte-stable across machines.
- **G3.** Recognisable identity consistency across renders, seeds, scenes, and
  model families, with an explicitly tiered and honestly documented strength
  guarantee per family.
- **G4.** Measurement: a per-attribute scorecard comparing any rendered image
  against the spec that produced it, including presence and *position* checks for
  localized details.
- **G5.** A guided Q/A TUI that composes a spec without requiring the user to
  know the schema, the lexicon, or the numeric conventions — including spatial
  placement of marks and jewelry.
- **G6.** Full reach into every automation surface — scenario, compile,
  scripting, library API, both TUIs — so a persona renders identically wherever
  it is invoked.
- **G7.** Round-trip: photograph → spec → render → spec, with a measurable
  closure error.
- **G8.** Localized details land in the right place, reliably, on every family —
  by compositing where conditioning cannot reach.

### 3.2 Non-goals

- **N1.** Photorealistic likeness of a *specific real individual*. The system
  composes synthetic people. Photo extraction (§16) exists to bootstrap and to
  validate the lexicon, not to clone.
- **N2.** New base-model ports. This RFC ships no new generative model family.
- **N3.** A 3D morphable model with a licensed statistical basis. See §10.5.
- **N4.** Video or animation of a persona.
- **N5.** Guaranteed sub-pixel geometric accuracy. The system targets
  *recognisability and directional correctness*, and reports measured
  controllability honestly where it falls short.
- **N6.** Full-body identity at the fidelity of face identity. See §14.1 and the
  honest-scope note in §11.7.
- **N7.** Medical or forensic accuracy. Scars, birthmarks and dentition are
  described as *visual morphology* for image generation. The vocabulary is
  deliberately descriptive rather than clinical (§8.7).
- **N8.** Reliable jewelry on hands. See §8.5 and the honest-scope note there.

---

## 4. Terminology and naming

plakat has three overlapping names for this concept already: **People** is the
canonical keyword and the UI screen; scenarios carry a `personas:` block; this
RFC proposes `plakat persona`. Three names for one concept is a maintenance
liability.

**Resolution adopted by this RFC:**

- The **subcommand** is `plakat persona`, singular, because it operates on one
  spec at a time and reads as a verb-object pair.
- The **library** — the collection, the UI screen, the storage directory —
  remains **People**. `plakat persona` writes into the People library.
- The scenario **`personas:`** block is extended, not replaced: an entry may
  remain a free-text description (existing behaviour, unchanged) or become a
  path to a `PersonaSpec`. Both dispatch to the same downstream pipeline; the
  free-text form is treated as a spec with only the prose field populated.
- Documentation uses "persona" for *the spec* and "person" for *the resolved
  identity*. A persona is authored; a person is cast.

Additional terms used throughout:

| Term | Meaning |
|---|---|
| **Spec** | the `PersonaSpec` HJSON document |
| **Lexicon** | the data file mapping spec vocabulary to phrasings, geometry, and probes |
| **Resolution** | the deterministic expansion of a partial spec to a complete one |
| **Casting** | generating and curating the identity reference set |
| **Reference set** | the curated images that anchor identity for adapter-based paths |
| **Anchor** | a tagged, landmark-relative position for a localized detail (§8.2) |
| **Detail** | a mark, jewelry item, or dentition feature — small, localized, positional (§8) |
| **Scorecard** | the per-attribute measurement of a render against its spec |
| **Bake** | training a per-base adapter (TI or LoRA) from the reference set |
| **Manifestation** | whether an attribute is *visible* in a given render (§8.6) |

---

## 5. Architecture

### 5.1 Layers

```
                        persona.hjson  (Layer 0)
                              │
                    ┌─────────┴─────────┐
                    │   RESOLVER (pure) │  lexicon + calibration tables
                    └─────────┬─────────┘
                              │
     ┌──────────────┬─────────┼─────────┬──────────────┐
     ▼              ▼         ▼         ▼              ▼
 per-family    landmark /   detail   identity      manifestation
 prompt +      depth cond.  plan     anchor        gate
 negative      map          (§8)     (refs,        (§8.6)
 (Layer 1)     (Layer 2)             adapters)
     │              │         │         │              │
     └──────────────┴─────────┼─────────┴──────────────┘
                              ▼
                      generation (any family)
                              │
                              ▼
                   detail compositing pass (§8.4)
                              │
                              ▼
                     SCORECARD (Layer 4)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              accept / retry     targeted repair
                                 (detect → mask → inpaint)
```

Layer 5 (calibration) is orthogonal: it produces the tables the resolver reads,
and is regenerated per model family on a slow cadence (§13).

### 5.2 Determinism contract

Following the precedent set by the map track, the contract is explicit and
testable:

- **Pure.** `resolve(spec, lexicon, calibration) → ResolvedSpec` has no I/O, no
  clock, no RNG, no network. Same inputs, same bytes, on every platform.
- **Byte-stable geometry.** Landmark resolution, detail anchor resolution, and
  conditioning-map rasterisation are pure functions of `(ResolvedSpec, seed)` and
  produce byte-identical PNGs across machines. No font assets, no
  floating-point-order hazards in the rasteriser, deterministic iteration order
  over all maps and over the detail collection.
- **Stochastic boundary.** Only diffusion sampling and LLM-assisted prefill are
  non-deterministic. Both are opt-in and both have a deterministic fallback
  (`--no-enhance` for prompt assembly; the offline keyword mapper for prose
  prefill, mirroring the fractal prose→spec provider pattern).
- **Deterministic compositing.** The detail compositing pass (§8.4) is itself a
  pure function of `(image, detail plan, landmarks)` up to the optional
  harmonisation pass, which is diffusion-based and therefore stochastic. The
  compositing-only path is byte-stable and corpus-tested.
- **Corpus.** Each deterministic stage carries at least one committed corpus
  artefact regenerated by a shell script, in the established style.

---

## 6. Layer 0 — `PersonaSpec`

### 6.1 Design principles

**P1 — Scalars over adjectives where measurement is possible.** `spacing: 0.62`
can be rendered into geometry *and measured back out of an image*. "Wide-set
eyes" can do neither. Enums are retained only where interpolation is meaningless
(eye shape, nose profile, hair texture, scar morphology).

**P2 — Identity is separable from presentation.** Wardrobe, expression, pose,
lighting, and scene are not identity. They belong to the render call, not the
persona. Mixing them guarantees that every render fights over the same token
budget and that a persona cannot be reused across scenes. Jewelry sits on this
seam and is handled explicitly in §8.5.

**P3 — Deviation carries the information.** An attribute at the population mean
contributes nothing and must cost nothing. This principle drives the token-budget
solver (§9.2) and the scorecard's weighting (§12.2). Localized details are the
exception: a mole is *always* a deviation, which is why they consume budget
disproportionately and why compositing exists to take them off the budget
entirely.

**P4 — Partial specs are valid.** Every field is optional. An unset field is
*unknown*, not *average* (§6.4). The file on disk is always loadable at every
stage of authoring.

**P5 — The file is the source of truth.** Plain HJSON, comment-friendly,
hand-editable, no hidden database. Derived artefacts (references, adapters,
scorecards, caches) are additive sidecars that can be deleted and rebuilt —
consistent with the `album.hjson` doctrine.

**P6 — Position is anatomical, never pixel.** Every localized detail is anchored
to the landmark topology, so it moves correctly when the face geometry changes,
survives a change of framing, and can be measured back out of a render (§8.2).

### 6.2 Schema v1

```hjson
{
  schema: persona/1

  identity: {
    name: alice                    # slug; unique within the People library
    display_name: "Alice Vance"
    apparent_age: 34               # integer years; drives lexicon variant selection
    sex: female                    # female | male | androgynous
    notes: "field researcher"      # free text, never compiled into a prompt
  }

  face: {
    shape: oval                    # oval|round|square|heart|diamond|oblong|triangular
    width: 0.45                    # bizygomatic width relative to length
    jaw: { angle: 0.35, width: 0.4, definition: soft }   # soft|moderate|defined|angular
    chin: { projection: 0.4, width: 0.45, cleft: none }  # none|slight|pronounced
    cheekbones: { height: 0.7, prominence: 0.75 }
    forehead: { height: 0.5, slope: 0.45 }
    temples: 0.5
    asymmetry: 0.15                # 0 = perfectly symmetric; small values read as human
  }

  eyes: {
    color: hazel                   # lexicon enum; or lab: [L,a,b] for exact iris tone
    heterochromia: none            # none | { left: <color>, right: <color> }
    shape: almond                  # almond|round|hooded|monolid|downturned|upturned|deepset
    size: 0.5
    spacing: 0.62                  # inter-pupillary distance / face width
    canthal_tilt: 0.55             # 0.5 = neutral; >0.5 = upward outer corner
    hood: 0.2
    sclera_show: 0.4
    lashes: { length: 0.5, density: 0.5 }
    brow: {
      thickness: 0.7
      arch: soft                   # straight|soft|rounded|angled|high
      length: 0.55
      spacing: 0.5
      color: dark-auburn
    }
    under_eye: { hollow: 0.3, lines: 0.2 }
  }

  nose: {
    profile: straight              # straight|aquiline|roman|button|snub|hooked|concave
    length: 0.5
    bridge: { width: 0.35, height: 0.45 }
    tip: { projection: 0.4, rotation: 0.5, width: 0.4 }
    nostrils: { width: 0.4, flare: 0.35, visibility: 0.4 }
    columella: 0.5
  }

  mouth: {
    width: 0.55
    upper_lip: 0.4
    lower_lip: 0.6
    cupids_bow: defined            # flat|soft|defined|pronounced
    corners: 0.5                   # <0.5 downturned, >0.5 upturned
    philtrum: { length: 0.45, depth: 0.5 }
    lip_texture: 0.4               # smooth ← → visibly lined
    lip_color: auto                # auto = derived from skin tone; or enum / lab
  }

  # ---- dentition: manifests only when the mouth is open (§8.6) ----
  teeth: {
    visibility: auto               # auto | none | slight | full
                                   # auto = derived from the render's expression
    alignment: even                # even|slightly-crowded|crowded|gapped|
                                   # diastema|overbite|underbite|snaggle
    diastema: 0.0                  # 0..1 midline gap width, when alignment: diastema
    shade: 0.65                    # 0 = deeply stained, 1 = bright white
    shade_uniformity: 0.8          # 0 = mottled, 1 = even
    size: 0.5                      # relative crown size
    proportion: 0.5                # central-incisor width : height
    gum_show: 0.3                  # gingival display when smiling broadly
    wear: 0.2                      # edge wear / age
    features: [                    # discrete, positional; see §8
      { kind: chip,    tooth: upper-left-central,  size: 0.3 }
      { kind: missing, tooth: lower-right-lateral }
      { kind: gold-crown, tooth: upper-right-canine }
    ]
    appliance: none                # none|metal-braces|clear-aligner|retainer
  }

  ears: {
    size: 0.5
    protrusion: 0.4
    lobe: attached                 # attached | free
    shape: 0.5
  }

  skin: {
    tone: fitzpatrick-3            # fitzpatrick-1..6; or lab: [L,a,b]
    undertone: neutral             # cool|neutral|warm|olive
    texture: 0.3                   # 0 = porcelain, 1 = heavily textured
    complexion: 0.5                # matte ← → luminous
    lines: { forehead: 0.2, nasolabial: 0.3, crows_feet: 0.25 }
    pores: 0.4
    flush: { region: cheeks, intensity: 0.2 }
  }

  # ---- unified localized-detail collection; see §8 ----
  marks: [
    # freckles: distributional, not positional
    { kind: freckles
      region: cheeks-nose
      density: 0.4
      size: 0.15
      color: auto }                # auto = derived from skin tone

    # mole: single, positional
    { kind: mole
      anchor: { landmark: left-nasolabial-upper, offset: [0.02, -0.03] }
      size: 0.18
      raised: 0.3
      color: lab: [32, 8, 12]
      hairs: false }

    # scar: linear or areal, positional, with maturity
    { kind: scar
      form: linear                 # linear|surgical|burn|abrasion|pockmark|keloid
      anchor: { landmark: right-brow-outer, offset: [0.0, 0.01] }
      length: 0.16                 # fraction of face width
      width: 0.02
      orientation: 68              # degrees, face-relative; 0 = horizontal
      maturity: 0.8                # 0 = fresh/red, 1 = mature/pale
      relief: 0.35                 # 0 = flat, 1 = strongly raised
      hair_interruption: true      # breaks the brow hairline
      note: "old climbing injury" }

    # birthmark: areal, positional, soft-edged
    { kind: birthmark
      form: pigmented              # pigmented|vascular|pale|mottled
      anchor: { landmark: left-jaw-mid, offset: [-0.01, 0.04] }
      size: 0.09                   # longest axis, fraction of face width
      aspect: 1.6                  # elongation
      orientation: 20
      edge: soft                   # soft|defined|irregular
      color: lab: [44, 14, 16]
      intensity: 0.6 }             # opacity against surrounding skin

    # tattoo: areal, positional, on face or body
    { kind: tattoo
      anchor: { landmark: left-forearm-outer, offset: [0.0, 0.0] }
      motif: "fine-line botanical"
      size: 0.22
      color: monochrome            # monochrome|blackwork|colour
      age: 0.5 }                   # 0 = fresh/crisp, 1 = faded

    # generic blemish family
    { kind: birthmark-cluster | vitiligo-patch | burn-scar | ... }
  ]

  # ---- piercings are durable body features; jewelry is what is worn in them ----
  piercings: [
    { site: left-lobe,    count: 2, gauge: 0.2 }
    { site: right-lobe,   count: 1, gauge: 0.2 }
    { site: left-helix,   count: 1, gauge: 0.2 }
    { site: septum,       gauge: 0.5 }
    { site: right-nostril, gauge: 0.15 }
    { site: lower-lip-left, gauge: 0.25 }
    { site: left-brow,    gauge: 0.3 }
    # empty / healed-over piercings are still specified; jewelry may be absent
  ]

  jewelry: {
    identity_locked: false         # true = always worn, part of how they read
    items: [
      { kind: earring
        site: left-lobe
        style: hoop                # stud|hoop|drop|huggie|cuff|threader|jacket
        size: 0.3
        metal: gold                # gold|silver|rose-gold|steel|black|mixed
        stone: none }              # none|diamond|pearl|opal|coloured:<lab>

      { kind: earring, site: right-lobe, style: stud, size: 0.15, metal: gold,
        stone: pearl }

      { kind: nose-ring
        site: septum
        style: clicker             # clicker|captive-bead|horseshoe|stud|nostril-hoop
        size: 0.25
        metal: silver }

      { kind: ring
        site: right-index          # finger sites; see honest scope §8.5
        style: signet
        metal: gold
        stone: none }

      { kind: necklace
        style: pendant             # chain|pendant|choker|locket|beads|layered
        length: 0.4                # 0 = choker, 1 = long
        metal: gold
        stone: none }

      { kind: bracelet, site: left-wrist, style: bangle, metal: silver }
      { kind: watch,    site: left-wrist, style: field, metal: steel }
      { kind: glasses
        style: round               # round|rectangular|cat-eye|aviator|rimless|browline
        frame: tortoiseshell
        thickness: 0.4
        tint: none }
      { kind: nose-chain | ear-cuff | brow-bar | anklet | ... }
    ]
  }

  hair: {
    color: auburn                  # enum; or lab: [L,a,b]
    color_variation: 0.3           # roots/highlights spread
    greying: 0.1                   # 0..1 fraction
    length: shoulder               # buzz|crop|short|chin|shoulder|long|very-long
    texture: wavy                  # straight|wavy|curly|coily|kinky
    density: 0.6
    style: "loose, centre part"    # free text; presentation-adjacent, see §6.5
    hairline: { height: 0.5, shape: rounded, recession: 0.0 }
  }

  facial_hair: {
    style: none                    # none|stubble|moustache|goatee|van-dyke|
                                   # short-beard|full-beard|sideburns|...
    density: 0.0
    color: auto                    # auto = derive from hair.color
    length: 0.0
    greying: 0.0
  }

  figure: {
    height_cm: 170
    build: mesomorph               # ectomorph|mesomorph|endomorph| combinations
    weight_impression: 0.5
    shoulders: 0.45                # width relative to hips
    waist: 0.45
    limb_length: 0.5
    neck: { length: 0.5, thickness: 0.4 }
    hands: { size: 0.5 }
    posture: upright               # upright|relaxed|stooped|military
    musculature: 0.4
  }

  # ---- presentation defaults; overridable per render, never identity ----
  defaults: {
    expression: neutral            # drives teeth manifestation, §8.6
    gaze: to-camera
    framing: headshot
    lighting: soft-key
    wardrobe: "plain dark shirt"
    jewelry_worn: all              # all | none | [item indices] — per-render override
  }

  # ---- authoring provenance ----
  provenance: {
    method: tui                    # tui|manual|extracted|prose|derived
    lexicon_version: "1.0"
    derived_from: null             # persona slug for lineage ops
  }
}
```

### 6.3 Scalar semantics

A scalar in `[0,1]` denotes a position in the attribute's range where **0.5 is
the model-relative prior**, not an absolute anthropometric mean.

This is a deliberate and consequential choice. The prior is measured per model
family (§13.1) by generating an unconditioned face population and measuring the
resulting landmark distribution. `0.5` therefore means *"whatever this model
already tends to produce"*, and any other value is a measured deviation from
that. The consequences:

- The same spec expresses the same *intent* on SDXL and on a Gemma-2-conditioned
  family, even though their unconditioned priors differ.
- Minimum-effort conditioning: attributes at 0.5 require no steering at all,
  which is exactly what P3 demands.
- The mapping is empirical, so it must be recalibrated when a family, its default
  sampler, or its default size changes (§13.4).

Scalars are clamped to `[0,1]`. Values outside `[0.05, 0.95]` emit a lint warning
that the request is beyond measured controllability for most attributes.

**Detail scalars are absolute, not model-relative.** `size`, `length`, `width`,
and `offset` on a mark or jewelry item are expressed as fractions of face width
(or of the relevant body part's bounding box), because they are realised by
compositing rather than by conditioning and therefore have no model prior to be
relative to. This distinction is recorded per lexicon entry and is checked by
lint.

Absolute units are used where they are meaningful and stable: `apparent_age` in
years, `height_cm` in centimetres, `orientation` in degrees, `lab:` triples for
exact colour.

### 6.4 Unknown versus mean

These are distinct states and must never be conflated:

| State | Encoding | Compiler | Scorecard |
|---|---|---|---|
| **Unknown** | field absent, or `null`, or `unknown` | contributes nothing; no tokens, no geometry deviation | not probed; excluded from the score |
| **Explicit mean** | `0.5` | contributes nothing to the prompt, but *is* asserted in geometry | probed; deviation counts against the score |

Rationale: a partially-authored persona should not be penalised for attributes
the author never considered, and the token-budget solver must be able to
distinguish "deliberately unremarkable" from "unspecified". Conflating the two
would silently corrupt both the salience ranking and the scorecard denominator.

For collections (`marks`, `piercings`, `jewelry`) the distinction is
between an **absent key** (unknown — the author never said whether this person
has marks) and an **empty list** (asserted — this person has no marks, and a
render that produces one is a failure). Lint warns when a persona has been
authored to `standard` depth with `marks` still absent, because in practice that
is almost always an oversight rather than an assertion.

The TUI surfaces this as a first-class `[u] unknown` answer on every question,
and as an explicit "no marks" versus "skip this section" choice on collections
(§17.4).

### 6.5 Attribute classes

Every leaf carries a class in the lexicon. The class determines what an edit
invalidates:

| Class | Examples | Edit invalidates | Repair strategy |
|---|---|---|---|
| **Structural** | face shape, eye spacing, nose bridge, jaw, ear shape, figure proportions, teeth alignment and proportion | the reference set and every baked adapter | full re-cast |
| **Surface** | eye colour, hair colour, skin tone, lines, teeth shade and wear, marks, scars, birthmarks, piercing sites | nothing structural | targeted inpaint or recomposite over the existing reference set (§12.4) |
| **Detail** | the individual entries of `marks`, `teeth.features`, and `jewelry.items` | nothing | recomposite only (§8.4) — the cheapest class to change |
| **Presentation** | expression, gaze, wardrobe, lighting, framing, hair style text, `jewelry_worn` | nothing | per-render override only |

Three notes on the placement of the new attributes:

- **Teeth alignment is structural, teeth shade is surface.** Alignment and
  proportion are dentition — they change how the face reads when open and they
  should force a re-cast of any reference that shows teeth. Shade, uniformity and
  wear are colourings and can be repaired in place.
- **Piercings are surface; jewelry is presentation.** A septum piercing is a
  durable feature of the person; the ring worn in it on any given day is not.
  This split is what makes `jewelry_worn: none` a coherent per-render override,
  and it is why the two live in separate schema blocks. A persona whose jewelry
  is inseparable from how they read sets `jewelry.identity_locked: true`, which
  promotes the items to surface class and includes them in casting.
- **Marks are their own class.** They are neither structural (they do not move
  the geometry) nor merely surface (they are positional and are realised by
  compositing). Classing them separately is what lets a mark be added, moved, or
  removed without touching the reference set at all — a mark edit is a
  recomposite, measured in milliseconds.

Edits are diffed by class. The TUI and CLI both report, before saving, *"this
change is structural and will invalidate 4 references and 2 adapters"*, and the
re-cast is opt-in.

### 6.6 Validation and lint

`plakat persona lint` runs without weights, without network, in milliseconds:

- **Schema** — unknown keys, wrong types, out-of-range scalars, unknown enum
  values (with nearest-match suggestion).
- **Contradiction** — `facial_hair.style: none` with `facial_hair.density > 0`;
  `hair.length: buzz` with `hairline.recession` detail that cannot render;
  `heterochromia` set alongside a scalar `eyes.color`; `teeth.alignment:
  diastema` with `diastema: 0.0`.
- **Referential integrity for details** — a `jewelry` item whose `site` has no
  corresponding entry in `piercings`; two marks whose resolved anchors overlap;
  a `teeth.features` entry naming a tooth already marked `missing`; a mark
  anchored to a landmark that the current topology does not define.
- **Occlusion** — a mark anchored under a region that `hair.style` or
  `facial_hair.style` will cover, or a `left-brow` scar with `hair_interruption`
  on a persona with no brow. A warning, not an error: the author may want it
  visible only in some framings.
- **Geometric feasibility** — combinations the landmark engine cannot resolve
  without self-intersection (extreme `spacing` with extreme `face.width`); detail
  sizes that exceed their anchor region.
- **Controllability** — attributes set away from 0.5 whose measured grade is
  `weak` or `experimental` on the target family; a warning, never an error.
- **Budget** — projected token count per family after salience ranking, with a
  warning when the solver will be forced to drop attributes the author marked
  high-priority, and a note of how many details were routed to compositing rather
  than to the prompt.
- **Manifestation** — teeth detail authored on a persona whose `defaults.expression`
  never shows teeth; an informational note that the detail exists but will not
  appear in default renders.
- **Safety** — the age gate (§23.1), which is an error, not a warning.

Lint returns a non-zero exit code on errors so it can gate CI.

### 6.7 Versioning and migration

`schema: persona/N` is mandatory. The loader accepts the current version and all
prior versions, migrating forward in memory; `plakat persona migrate` rewrites in
place. Lexicon versions are tracked separately in `provenance.lexicon_version`,
because a lexicon change can alter the *meaning* of an unchanged spec — when a
scalar's calibration curve moves, or when a landmark is renumbered and every
anchored detail shifts, previously cast references may no longer match. The
scorecard records both versions, and a mismatch between the spec's lexicon
version and the current one is surfaced as a staleness warning.

**Landmark topology changes are breaking for anchors.** Any change to the
landmark index assignments requires a migration that remaps every `anchor`
in every stored spec. The topology is therefore versioned independently and
frozen more aggressively than the rest of the lexicon.

---

## 7. Layer 0b — The lexicon

The lexicon is the single hand-authored asset that makes cross-model behaviour
tractable. It is data, shipped in `assets/`, not code.

### 7.1 Entry structure

```hjson
eyes.spacing: {
  class: structural
  type: scalar
  range: [0, 1]
  relative_to: model_prior         # model_prior | absolute  (§6.3)

  # --- how it is asked, for the TUI (§17.3) ---
  ask: "How far apart are the eyes?"
  section: eyes
  order: 30
  depth: standard                  # quick | standard | full
  widget: scalar
  variants: [0.25, 0.4, 0.5, 0.6, 0.75]
  help: "Measured pupil-to-pupil against face width. Most faces sit near 0.5."

  # --- how it renders into geometry (§10.2) ---
  geometry: {
    basis: eye_spacing             # named deformation direction
    gain: 0.18                     # landmark displacement per unit scalar
  }

  # --- how it renders into text, per encoder class (§9.3) ---
  phrasing: {
    clip:   { low: "close-set eyes", high: "wide-set eyes", weight: 1.1 }
    t5:     { low: "with eyes set noticeably close together",
              high: "with eyes set noticeably far apart" }
    gemma:  { low: "Her eyes are set close together, ...",
              high: "Her eyes are set unusually wide apart, ..." }
  }
  anti: { low: "wide-set eyes", high: "close-set eyes" }

  # --- how it is measured back (§12.1) ---
  probe: {
    kind: landmark
    metric: interpupillary_over_facewidth
    tolerance: 0.06
  }

  # --- measured, not authored (§13.3) ---
  control: strong
  curve: "eyes.spacing"            # key into the calibration table
}
```

Detail entries carry three additional blocks:

```hjson
marks.scar: {
  class: detail
  type: record
  widget: list                     # collection widget, §17.4
  member_widget: { form: select, anchor: place, length: scalar, ... }

  # --- how it is placed (§8.2) ---
  anchoring: {
    allowed_landmarks: [brow-*, cheek-*, jaw-*, chin-*, forehead-*, lip-*, neck-*]
    default_landmark: right-brow-outer
    coordinate_space: face_normalised
  }

  # --- how it is realised (§8.3) ---
  realisation: {
    strategy: composite_then_harmonise   # prompt | condition | composite | composite_then_harmonise
    generator: procedural_scar           # the deterministic renderer
    prompt_fallback: true                # also emit text, if budget allows
    harmonise: { strength: 0.18, feather: 6 }
  }

  # --- how it is measured back (§12.1) ---
  probe: {
    kind: local_anomaly
    at: anchor
    radius: 1.5                    # multiples of the detail's own size
    tolerance: { presence: 0.6, position: 0.03 }
  }

  # --- conditional visibility (§8.6) ---
  manifest_when: "always"
  occluded_by: [hair.style, facial_hair.style, jewelry.items]

  control: strong                  # composited details are strongly controllable
}
```

```hjson
teeth.alignment: {
  class: structural
  type: enum
  values: [even, slightly-crowded, crowded, gapped, diastema, overbite,
           underbite, snaggle]
  manifest_when: "render.mouth_open || render.teeth_visible"
  ask: "How are the teeth aligned?"
  section: teeth
  gated_by: "teeth.visibility != none"
  ...
}
```

Every attribute in the schema has exactly one lexicon entry. The entry is the
join point between six subsystems — TUI, geometry, detail realisation, compiler,
scorecard, and calibration — which is what keeps them from drifting apart.

### 7.2 Deriving the interview from the lexicon

Because `ask`, `section`, `order`, `depth`, `widget`, `member_widget`,
`variants`, `gated_by`, and the dependency conditions all live in the lexicon,
the TUI's question graph is *generated*, not written. Adding an attribute — or a
whole new mark kind, or a new piercing site — requires a lexicon entry and no TUI
code. See §17.3.

### 7.3 Controllability grades

`control` is one of `strong`, `moderate`, `weak`, `experimental`, and is
**measured, never asserted** (§13.3). It is per-attribute *and per-family*; the
scalar entry above shows the default, with per-family overrides in the
calibration table. Grades are surfaced in `persona show`, in lint, and as a badge
in the TUI. An attribute graded `experimental` on a family is still emitted, but
the scorecard down-weights it and the documentation says so.

Composited details (§8.3) are graded independently and generally score `strong`
regardless of family, because their realisation does not depend on the sampler.
This asymmetry is worth stating plainly in the docs: *the small things are the
reliable things, precisely because they are not prompted.*

### 7.4 Neutrality requirements

The lexicon encodes human morphology and is therefore a place where careless
vocabulary would cause real harm. Binding constraints:

- Skin tone is expressed on the Fitzpatrick scale or as CIELAB values. No
  descriptive colour adjectives, no ethnonyms, no geographic labels anywhere in
  the schema or the lexicon.
- Morphological enum values are neutral anatomical descriptors (`monolid`,
  `aquiline`, `attached lobe`, `diastema`) chosen from standard anthropometric
  and descriptive vocabulary. Values carrying historical racial-classification
  baggage are excluded by review, with the review recorded in the lexicon's own
  comments.
- **Marks, scars and birthmarks are described as visual morphology, not as
  medical conditions.** Vocabulary is `pigmented`, `vascular`, `pale`,
  `mottled`, `raised`, `linear`, `keloid-form` — descriptions of appearance. The
  lexicon does not name diagnoses, does not imply causes, and does not attach
  any valence to a mark's presence. See also §23.4.
- No attribute may be labelled with a valence — no `attractive`, `harsh`,
  `unfortunate`, `disfiguring`. The LAION aesthetic ranker is available for those
  who want an aesthetic ordering, and it is kept strictly out of the identity
  vocabulary.
- Defaults are unset, never a default face, never a default set of marks. There
  is no fallback human.
- Enum ordering in menus is alphabetical or anatomical, never by prevalence.

---

## 8. Marks, jewelry and dentition — the localized-detail subsystem

Scars, birthmarks, moles, freckles, tattoos, piercings, jewelry, and individual
teeth share one property that separates them from every other attribute in the
spec: they are **small, positional, and high-contrast**. That combination defeats
text conditioning entirely (§2.1.4) and needs its own machinery.

This section is the part of the RFC with the least prior art in plakat, and the
part where the design is most opinionated.

### 8.1 Why details need a separate subsystem

Consider `a small mole below the left eye`. Four things go wrong at once:

1. **Placement is unspecifiable in text.** "Below the left eye" spans a region
   perhaps forty pixels across at portrait resolution. The sampler will place it
   anywhere in that region, or on the right side, or on both, or nowhere.
2. **Presence is unreliable.** Small features are the first thing the sampler
   drops under CFG pressure or at low step counts. Hit rate for a prompted mole
   is materially below 1.0 on every family.
3. **They are expensive per bit of information.** A mole costs four to six tokens
   on a CLIP encoder to express something that occupies twenty pixels — and those
   tokens compete with the structural attributes that actually determine whether
   the person is recognisable.
4. **They must be *identical* across renders.** A persona whose mole moves
   between shots is not a persona. This is a stricter requirement than for any
   scalar attribute, where "approximately right" is acceptable.

The remedy is to stop asking the model. Details are **anchored anatomically**,
**realised by compositing**, and **verified locally**. The prompt path is
retained as a fallback and as a hint, not as the mechanism.

This is the same architectural move the artefact system already makes for scene
elements — named cutouts composited into named zones with canvas-relative scale,
contact-shadow grounding, and colour harmony, optionally followed by a masked
img2img pass to smooth the seams. The detail subsystem is that mechanism,
re-pointed from scene zones to anatomical landmarks.

### 8.2 Anchoring

Every positional detail carries an **anchor**: a tagged, landmark-relative
position, in the spirit of the map track's tagged anchor type where a landmark is
placed relative to a *feature*, never to a pixel.

```hjson
anchor: { landmark: left-nasolabial-upper, offset: [0.02, -0.03] }
```

- `landmark` names a point or a named region in the 106-point topology (§10.1),
  or a body site from the figure skeleton (`left-forearm-outer`,
  `right-wrist`, `throat`).
- `offset` is a 2-vector in **face-normalised coordinates** — fractions of face
  width and height, in the canonical front-facing frame, x positive to the
  subject's left.
- Resolution maps the anchor through the *deformed* landmark set, so a mark
  anchored to the nasolabial fold moves correctly when `mouth.width` or
  `face.width` changes. This is why P6 matters: pixel coordinates would silently
  desynchronise from the geometry on every edit.
- For non-frontal views, the anchor is projected through the pose, and details
  whose projected position falls behind the visible surface are culled. Culling
  is reported, so the scorecard does not penalise a mole that is legitimately on
  the far cheek.

**Named region shorthand.** Authors may write `region: left-cheek` instead of a
landmark and offset; resolution places the detail at the region's centroid plus a
seeded jitter. This is what the prose and photo prefill paths emit when they know
the region but not the exact position, and it is what the TUI's placement widget
refines.

**Distributional details** — freckles, pockmark fields, mottling — do not carry a
point anchor. They carry a `region` and a `density`, and are realised as a
procedural field over the region mask rather than as an individual composite.

### 8.3 Realisation strategies

Each detail kind declares a strategy in the lexicon:

| Strategy | Mechanism | Used for |
|---|---|---|
| `prompt` | text only | details too large or too diffuse to composite (heavy freckling, an overall complexion) |
| `condition` | rendered into a conditioning map, no compositing | details with strong geometric consequence (brow scar that interrupts the hairline) |
| `composite` | deterministic overlay at the resolved anchor | small, well-defined marks; most jewelry |
| `composite_then_harmonise` | overlay, then a low-strength masked img2img over the overlay region | the default for anything that must sit convincingly in the skin |
| `inpaint` | mask the anchor region and regenerate it with a focused prompt | large or complex details; the repair path (§12.4) |

The default for marks is `composite_then_harmonise`. The overlay is generated
procedurally — a mole is a soft radial gradient with a shading term derived from
the scene's estimated light direction; a linear scar is a stroke with a
maturity-dependent colour ramp, a relief term rendered as a subtle normal
perturbation, and optional hairline interruption; a birthmark is a soft-edged
irregular blob with an edge-noise parameter. All are pure functions of the
detail record plus a seed, and therefore byte-stable.

The harmonisation pass is the one stochastic step, and it is what makes the
difference between "a mark drawn on a photo" and "a mark on skin". It runs at low
strength over a feathered mask that extends slightly beyond the composite, using
the same masked-img2img machinery as artefact blending.

**Jewelry is composited from assets, not procedurally generated.** A bundled
jewelry asset library — small PNGs with alpha, per style and metal, at several
orientations — lives alongside the artefact and style catalogs. Metal tone is
recoloured at composite time from the item's `metal` field; stones are tinted
from `stone`. Compositing applies the same canvas-relative scaling, contact
shadow, and colour harmony the artefact path already implements.

**Teeth are never composited.** Dentition is realised by conditioning and by
inpainting the mouth region, because teeth must sit inside a mouth whose shape,
lighting, and perspective the model determined. Compositing teeth produces the
uncanny result you would expect. See §8.6.

### 8.4 The compositing pass

Runs after generation, before scoring:

```
rendered image
  → detect and align the face; recover the realised landmark set
  → for each detail, in a deterministic order (z-order by kind, then by index):
      resolve its anchor through the REALISED landmarks, not the requested ones
      cull if occluded, out of frame, or behind the visible surface
      generate or load its overlay
      scale by the realised face width; rotate by the realised head pose
      estimate local light direction from the face's own shading
      composite with the kind's blend mode and opacity
  → union the affected regions into one mask
  → optional harmonisation img2img over that mask
  → re-detect and score (§12)
```

Resolving anchors through the *realised* landmarks rather than the *requested*
ones is the key detail. The model will not have produced exactly the geometry
that was asked for; anchoring to what it actually produced is what puts the mole
below the eye that exists rather than below the eye that was specified.

The pass is skipped entirely for details whose strategy is `prompt` or
`condition`, and it is a no-op — costing one face detection — for a persona with
no details.

### 8.5 Jewelry, piercings, and the identity seam

The split established in §6.5 is worth restating because it drives real
behaviour:

- **`piercings`** is a list of *sites and gauges*. It describes holes in the body.
  It is **surface** class: durable, part of how the person reads, included in
  casting so the reference set shows them.
- **`jewelry.items`** is a list of *worn objects*, each bound to a site (for
  piercing jewelry) or a body part (rings, bracelets, necklaces, watches,
  glasses). It is **presentation** class by default: swappable per render via
  `defaults.jewelry_worn` or a per-render flag.
- `jewelry.identity_locked: true` promotes the items to surface class for
  personas whose eyewear or signature earring is inseparable from how they read.

An empty piercing — a site with no jewelry worn — still renders, faintly, as a
healed or open hole. This matters for continuity: a persona photographed with and
without earrings should have the same ears.

**Honest scope: hands.** Rings, bracelets and watches sit on hands and wrists,
which are the least reliably rendered part of a human figure across every family
in the stack. Three consequences, all of which must be documented rather than
discovered:

1. Hand jewelry is graded `experimental` by default and the TUI says so.
2. The compositing path for hand jewelry requires a successful hand-landmark
   detection, which frequently fails; on failure the detail is culled and
   reported, not silently dropped.
3. The recommended workflow for a render where a ring matters is to compose the
   shot so the hand is prominent, then use targeted repair (§12.4) on the hand
   region — the same escalation as for a small face at wide framing (§14.1).

Glasses are the outlier in this block: they are large, salient, and reliably
rendered from a prompt, so their default strategy is `prompt` with
`condition` available, not compositing. They are also the one jewelry item that
materially changes face detection and identity embedding, which is why
`identity_locked` matters most for them — a persona cast with glasses and
rendered without will show measurable identity drift, and the scorecard should
attribute it correctly rather than blaming the sampler.

### 8.6 Manifestation: attributes that are conditionally visible

Teeth introduce a concept the rest of the schema does not need: an attribute that
exists on the person but is **absent from most renders**.

A `manifest_when` predicate in the lexicon declares the render conditions under
which an attribute is visible:

```
teeth.*          manifest_when: "render.mouth_open || render.teeth_visible"
teeth.gum_show   manifest_when: "render.broad_smile"
piercings.tongue manifest_when: "render.mouth_open"
jewelry[site=nape] manifest_when: "render.hair_up || hair.length <= short"
marks[region=neck] manifest_when: "render.framing != tight-headshot"
```

The predicate is evaluated against the resolved render context — expression,
framing, pose, hair state — which the renderer derives from the merged scene and
presentation defaults.

Consequences, threaded through the whole system:

- **Compiler (§9.2).** Non-manifesting attributes are excluded from emission
  before the budget solver runs. Spending six CLIP tokens describing dentition on
  a neutral closed-mouth portrait is pure waste, and this is the mechanism that
  prevents it.
- **Scorecard (§12.2).** Non-manifesting attributes form a **fourth exclusion
  category** alongside unknown, unmeasurable, and dropped. A persona whose teeth
  cannot be seen must not be scored on its teeth — and equally, the report must
  say so rather than silently omitting them.
- **Casting (§11.1).** The default cast is neutral-expression, so teeth do not
  appear in the reference set at all. A persona with authored dentition should
  extend its cast with at least one open-mouth view, or its teeth will be
  unanchored and will vary between renders. `persona cast --views sheet
  --expressions neutral,smile` is the recommended form for such personas, and
  lint emits an informational note when dentition is authored but the reference
  set contains no teeth-visible view.
- **TUI (§17.6).** The teeth section is gated behind an explicit question, and
  the preview switches to an open-mouth wireframe while that section is active,
  so the author can see what they are editing.

### 8.7 Dentition specifics

Teeth are realised through three cooperating mechanisms, because none alone is
sufficient:

1. **Prompt.** Alignment, shade and appliance are describable and land reasonably
   on strong prompt-followers. `crowded lower teeth`, `a small gap between the
   front teeth`, `metal braces` are all within the vocabulary the models know.
2. **Mouth-region conditioning.** The landmark topology's inner-lip contour
   defines the mouth aperture; the geometry engine renders a dentition
   conditioning hint into that region — an arch with per-tooth positions derived
   from `alignment`, `size`, `proportion`, and the `features` list. Where a
   family has usable region conditioning this materially improves placement of a
   diastema or a snaggle tooth.
3. **Mouth-region inpaint.** The reliable path, and the default for discrete
   `teeth.features`: after generation, mask the inner-lip region and regenerate
   it at native resolution with a dentition-focused prompt plus the conditioning
   hint. This is the same escalation the framing branch uses for small faces
   (§14.1), applied to a smaller region.

Individual `teeth.features` entries — a chip, a missing lateral, a gold crown —
name a tooth in standard positional notation (`upper-left-central`,
`lower-right-lateral`, `upper-right-canine`) rather than by number, because the
names are self-documenting and the notation is stable. Resolution maps the name
to a position along the resolved dental arch.

**Honest scope.** Teeth are among the hardest small structures for current
models. Expect `moderate` control on alignment and shade, `weak` on individual
feature placement without the inpaint escalation, and `experimental` on
appliances. The calibration pass will produce the real numbers; the schema should
not imply more precision than the grades support.

### 8.8 Scars and birthmarks: what makes them read as real

Both are areal, soft-edged, and highly sensitive to three parameters that a naive
implementation would omit:

- **Maturity.** A fresh scar is pink-to-red, slightly raised, with visible
  tension lines. A mature scar is paler than the surrounding skin, flatter, and
  often depressed. `maturity` interpolates the colour ramp and the relief term
  between these poles. Getting this wrong is the single most common reason a
  composited scar reads as a sticker.
- **Relief and light.** A raised or depressed mark catches light differently from
  flat skin. The compositor estimates the scene's light direction from the face's
  own shading and applies a matching highlight/shadow pair along the mark's
  orientation. Without this the mark looks painted on regardless of how good the
  colour is.
- **Edge character.** `edge: soft | defined | irregular` on a birthmark, and the
  hairline-interruption flag on a scar, are what make the boundary believable. A
  perfectly elliptical, hard-edged patch never looks like skin.

Both kinds also carry `hair_interruption` semantics: a scar crossing the brow or
the scalp interrupts the hair, and the conditioning map must reflect that or the
hair will render straight through it. This is the one case where a mark has a
`condition` component in addition to its composite.

Freckle and pockmark **fields** are distributional (§8.2) and are realised as a
seeded procedural field over the region mask, with density, size distribution,
and colour jitter — not as N individual composites, which would be both slow and
unconvincing.

### 8.9 Detail budget and ordering

Details are cheap to composite and expensive to prompt, which inverts the usual
budget logic. The resolver therefore:

- routes every detail whose strategy is `composite*` **out of the prompt entirely**
  by default, freeing its tokens for structural attributes;
- retains a `prompt_fallback` flag per kind, so a detail can also be described
  when budget permits — helpful because a model that has been *told* there is a
  scar will often light and shade the region more plausibly, making the
  harmonisation pass easier;
- composites in a fixed z-order — skin fields, then areal marks, then linear
  marks, then piercing jewelry, then worn jewelry — so overlapping details layer
  deterministically;
- caps the number of composited details with a warning, because a face carrying
  thirty marks is more likely an authoring error than an intention.

---

## 9. Layer 1 — The compiler

### 9.1 Pipeline

```
PartialSpec
  → normalise        (aliases, unit conversion, lab↔enum, auto-derivations)
  → validate         (§6.6)
  → resolve          (unknown stays unknown; derived fields computed)
  → manifest gate    (§8.6 — drop attributes invisible in this render)
  → detail routing   (§8.9 — composite vs prompt vs condition)
  → salience rank    (§9.2)
  → family emit      (§9.3)
  → negative assemble(§9.4)
  → merge with scene (§9.5)
  → ConditioningSet + DetailPlan
```

Auto-derivations are explicit and few: `facial_hair.color: auto` derives from
`hair.color`; brow colour defaults to hair colour darkened by a lexicon constant;
`marks[].color: auto` derives from `skin.tone` by a per-kind offset; `mouth.lip_color:
auto` likewise; `figure` scalars fill from `build` when unset; `teeth.visibility:
auto` resolves from the render's expression. Every derivation is recorded in the
resolved spec so `persona show` can display it as derived rather than authored.

### 9.2 Salience ranking and the token-budget solver

CLIP-family emission is budget-constrained; T5 and Gemma emission is not, but
still benefits from ordering. Both use the same ranking.

For each *known, manifesting, prompt-routed* attribute `a`:

```
salience(a) = |value(a) − prior(a, family)|          # deviation, §6.3
            × priority(a)                            # author override, default 1.0
            × control(a, family)                     # measured grade weight
            × class_weight(a)                        # structural > surface > detail
```

The solver then:

1. Sorts attributes by descending salience.
2. Emits greedily until the family's token budget is reached, reserving a
   configurable headroom for the scene prompt (default 40% of budget on CLIP
   families, since the persona must coexist with a scene).
3. Drops attributes whose salience falls below a floor even when budget remains —
   near-prior attributes are noise, not signal.
4. Records the drop list in the resolved spec so the scorecard knows which
   attributes were never actually requested on this family, and so `persona show`
   can explain the omission.

The floor and the headroom are exposed as `--budget-headroom` and
`--salience-floor` for experimentation, with the defaults committed.

**Detail routing runs before ranking**, which is the main reason the budget works
at all. On a persona with a mole, two piercings, a brow scar and a birthmark, the
naive prompt spends perhaps twenty-five CLIP tokens on details that the sampler
will place wrongly anyway; routing them to compositing returns all of it to the
structural attributes that determine recognisability.

**Grouping mitigation for entanglement.** Attributes are emitted grouped by
anatomical region with the region as the head noun (`eyes: almond, wide-set,
hazel, under heavy dark brows`) rather than interleaved. Region-headed grouping
measurably reduces cross-binding on CLIP encoders relative to flat comma lists;
the effect size is an experiment for Phase 1 (§28.1).

### 9.3 Per-family emitters

Emitters are keyed by **encoder class**, not by model, so a newly added family
inherits an emitter by declaring its encoder stack.

| Encoder class | Families | Shape | Notes |
|---|---|---|---|
| `clip` | SD 1.5, SD 2.1 | ≤77 tokens, region-grouped, weighted | aggressive salience pruning; weighting via the existing prompt-weight path |
| `clip_dual` | SDXL | ≤77 per encoder | emit identical text to both encoders by default; a `--split-encoders` experiment emits structure to CLIP-L and colouring to CLIP-G (§28.2) |
| `clip_triple` | SD 3.5 | CLIP pair + T5 | short form to the CLIP pair, long form to T5 |
| `t5` | PixArt-Σ, Flux | long natural language | full paragraph, ordered by salience, no weighting syntax |
| `gemma` | Sana | long natural language | longest form; the complex-human-instruction path rewards dense descriptive prose, so this emitter emits the *most* detail, not the least — including `prompt_fallback` detail descriptions that other emitters drop |

Each emitter is a pure function `ResolvedSpec → (positive, negative)`. Emitters
are template-driven from the lexicon `phrasing` block, with no LLM in the
deterministic path. An optional `--enhance` LLM polish pass exists for parity with
`compile`, is off by default, and never runs in corpus generation.

### 9.4 Negative assembly and deduplication

Negatives arrive from four sources and will collide:

1. **Persona anti-phrases** — from lexicon `anti` for excluded attributes
   (`facial_hair.style: none` → `beard, stubble, moustache, goatee`).
2. **Detail exclusions** — an *asserted empty* detail collection (§6.4) emits
   negatives: a persona with `marks: []` gets `moles, freckles, scars, blemishes`;
   `piercings: []` gets `piercings, earrings`; `jewelry_worn: none` gets
   `jewelry, necklace, earrings, rings`. This is a genuinely useful lever,
   because unrequested jewelry is one of the most common unwanted additions on
   portrait prompts.
3. **Family auto-negatives** — the existing `--enhance` negative stack.
4. **User negatives** — from the render call.

Assembly deduplicates on a normalised token basis, preserves user negatives at
highest priority, and applies the same budget solver in reverse — negatives are
also token-capped on CLIP families, and an unbounded negative list is a
well-known way to degrade output. The merged list is recorded in the sidecar.

A conflict check runs here: a negative that contradicts a composited detail
(`scars` in the negative while a scar is in the detail plan) is a lint error,
because the sampler will actively suppress the region the compositor is about to
draw into, and the harmonisation pass will fight it.

### 9.5 Precedence between persona and scene

A documented, testable precedence order, highest wins:

```
1. explicit per-render CLI flags
2. scene / scenario prompt
3. persona presentation defaults   (spec.defaults, incl. jewelry_worn)
4. persona surface attributes      (incl. piercings)
5. persona detail plan             (composited; asserted independently)
6. persona structural attributes   (locked; see below)
7. family auto-enhancement
```

Structural attributes sit *below* the scene in this ordering but are **locked**:
a scene cannot override them textually, because the geometric conditioning map
(Layer 2) asserts them independently of the prompt. The detail plan is similarly
independent — a scene that does not mention a scar still gets the scar, because
the compositor runs regardless of what the prompt said.

If a scene demands something that contradicts a structural attribute or a detail
(`"a clean-shaven man with no scars"` against a persona with a brow scar), the
conflict is detected at merge time and surfaced as a warning rather than being
silently resolved by whichever signal happens to win. Scene-level suppression is
available and explicit: `--suppress-details scar-0` or `--details none`.

Presentation defaults are always overridable and never warn — that is their
purpose, and `jewelry_worn` is the most-used example.

---

## 10. Layer 2 — The face and figure geometry engine

The geometry engine is what makes this an instrument rather than a prompt macro.
It converts spec scalars into an actual conditioning image, and it is what
resolves every detail anchor into a position.

Architecturally it mirrors the map track's geometry engine: pure Rust, no GPU, no
network, no model weights, a pure function of `(ResolvedSpec, seed)`, byte-stable
on-box, and independently useful before any generative step exists.

### 10.1 Landmark model

A 106-point facial landmark topology (the denser InsightFace-family convention, a
superset of the classic 68-point set) covering jaw contour, brows, eye contours,
nose bridge and base, outer **and inner** lip, and pupil centres. The denser set
is required because several spec attributes — nostril flare, canthal tilt,
philtrum, lip corner direction — have no representation in the 68-point topology,
and because the **inner-lip contour is what defines the mouth aperture** for
dentition (§8.7).

The topology is extended by this RFC with a set of **named anchor regions** that
are not landmarks in the detection sense but are derived from them: `left-cheek`,
`right-nasolabial-upper`, `left-jaw-mid`, `forehead-centre`, `chin-crease`,
`left-lobe`, `septum`, `left-helix`, and so on for every piercing site and every
plausible mark region. These are the vocabulary the `anchor.landmark` field draws
from, and they are versioned with the topology (§6.7).

The engine holds a **mean template**: a canonical, front-facing, neutral-expression
landmark configuration in a normalised face-box coordinate system, hand-authored
and committed. It is not derived from a dataset, for the licensing and
reproducibility reasons in §10.5.

An **open-mouth variant** of the template is also committed, because the
inner-lip contour of a closed mouth carries no useful dentition geometry. The
variant is selected by the manifestation gate.

### 10.2 Deformation basis

Each geometric spec attribute names a **deformation direction** — a hand-authored
displacement vector over the landmark set, applied proportionally to the
attribute's deviation from 0.5:

```
landmarks = mean_template + Σ_a  gain(a) · (value(a) − 0.5) · basis(a)
```

Design constraints on the basis:

- **Locality with anatomical coupling.** `eyes.spacing` moves the eye contours
  and pupils, but must also nudge the nose bridge width and inner brow, or the
  face reads as pasted-together. Coupling coefficients are part of the basis
  definition.
- **Approximate orthogonality.** Two attributes should not fight over the same
  landmarks. Where they must (jaw width and face width), the lexicon declares the
  interaction and the engine applies them in a fixed order.
- **Bounded composition.** After all deformations, a validity pass checks for
  self-intersection, inverted contours, and landmarks escaping the face box, and
  clamps back along the offending direction. Failures are reported by lint
  (§6.6), never rendered.
- **Asymmetry.** `face.asymmetry` applies a deterministic, seed-derived
  perturbation. Perfectly symmetric faces read as synthetic; a small asymmetry is
  one of the cheapest realism wins available and costs no tokens.
- **Anchors follow.** Detail anchors are resolved *after* deformation, against
  the deformed set, so every mark stays anatomically correct through any edit.

The basis is hand-authored rather than learned. This is the same reasoning that
produced the map track's hand-authored bitmap font: a committed, inspectable,
license-free asset that yields byte-identical output on every machine, at the
cost of some fidelity. Fidelity here is cheap to trade because the output is a
*conditioning signal*, never final pixels.

### 10.3 Conditioning map rendering

From the resolved landmarks the engine rasterises, on demand:

| Map | Use | Notes |
|---|---|---|
| **Landmark / mesh map** | face-landmark ControlNet | drawn in the convention the target ControlNet expects; per-family renderer |
| **Wireframe** | TUI preview (§17.7) | vector, also rasterisable to braille/half-blocks for terminals without a graphics protocol; annotated with detail markers |
| **Depth proxy** | depth ControlNet | landmark set lifted by a hand-authored per-region depth profile, smoothed; crude but directionally correct; scar `relief` contributes |
| **Pose skeleton (face keypoints)** | OpenPose-family conditioning | reuses the existing skeleton renderer from the multiperson path |
| **Dentition hint** | mouth-region conditioning (§8.7) | per-tooth arch positions within the inner-lip contour, from `alignment` / `size` / `proportion` / `features` |
| **Region masks** | targeted repair (§12.4), regional prompting, detail compositing | per-feature masks derived from the landmark contours; also the fallback mask source when open-vocabulary detection fails |
| **Detail overlay map** | preview and the composite pass | every resolved detail rendered at its anchor, for TUI display and as the compositor's plan |

The region-mask output deserves emphasis: it gives attribute-targeted repair a
*geometric* mask source that does not depend on a detector succeeding, which
matters because open-vocabulary detection on small facial features is unreliable —
and small facial features are precisely what this RFC has just added a great many
of.

### 10.4 Figure geometry

Distinct engine, same discipline. `figure` scalars resolve to a parametric
silhouette and a body-pose skeleton:

- **Silhouette** — a deterministic outline from height, build, shoulder/waist
  ratio, limb length, and musculature. Rendered for the TUI preview and usable as
  a soft mask.
- **Skeleton** — joint positions scaled by `height_cm` and `limb_length`, feeding
  the existing pose-conditioning path. Body scaling already exists for
  multiperson; this generalises it from a single scale factor to the full figure
  block.
- **Body anchor sites** — the skeleton also supplies the anchor vocabulary for
  body-sited details: forearm tattoos, wrist jewelry, throat pendants, nape
  piercings. Hand landmark sites are defined but flagged `experimental` per
  §8.5.

Honest scope: body conditioning is materially weaker than face conditioning
(§11.7), and the calibration pass will grade most `figure` attributes below
`strong`. The TUI must not present figure sliders as though they carry the same
authority as facial ones.

### 10.5 Licensing constraints

**A statistical 3D morphable model is explicitly out of scope for v1.** The
established bases carry research-only or otherwise restrictive licenses that are
incompatible with a public-domain project. Two consequences:

- The mean template and deformation basis are hand-authored from published
  anthropometric ranges, not fitted to a licensed dataset.
- A future 3D path (§30.3) must use a procedurally-generated parametric head
  mesh, not a licensed basis.

The jewelry asset library carries the same constraint and a second one: assets
must be original or public-domain, and must not reproduce identifiable
trade-dress of real jewelry brands. Generic styles only.

Any contribution proposing a learned basis or a third-party asset must state its
provenance and license in the PR, and the review must confirm compatibility.

---

## 11. Layer 3 — The identity anchor

Geometry, details and prompts constrain *what the face looks like*. They do not
by themselves produce a face stable enough to recognise across renders. The
anchor does.

### 11.1 Casting

```
plakat persona cast alice.hjson --count 32 --keep-best 4
```

1. Compile the spec for the casting family (default: the strongest Tier-A family
   available on this host).
2. Render `count` candidates at distinct seeds, with the geometric conditioning
   map applied, at portrait framing and native size.
3. Run the detail compositing pass (§8.4) on each candidate, so the reference set
   carries the persona's marks and piercings.
4. Score every candidate with the scorecard (§12) against the spec.
5. Optionally blend in the aesthetic ranker as a secondary sort key, weighted well
   below spec conformance — a beautiful face that is the wrong face is a failure.
6. Present the ranked contact sheet for curation (TUI) or auto-keep the top `k`
   (batch).
7. Validate the kept set for identity coherence: pairwise ArcFace cosine must
   exceed a threshold, or the cast has produced several different people and must
   be re-run with tighter conditioning. Report the matrix either way.

**Detail-carrying references matter.** A reference set without the persona's mole
will produce renders without the mole even after the compositor adds one, because
the adapter is pulling toward a face that does not have it and the harmonisation
pass has to fight. Compositing during casting closes that loop.

**Dentition requires an extended cast.** Per §8.6, a persona with authored teeth
should cast at least one teeth-visible expression, or dentition is unanchored.
`--expressions neutral,smile` is the recommended form and lint says so.

Casting is expensive and must therefore be: cached on the spec hash restricted to
structural attributes, resumable after interruption, incremental on surface- and
detail-only edits (recomposite the existing references rather than re-render,
§6.5), and memory-guarded (§22).

### 11.2 Multi-view sheet

A single frontal reference is a weak anchor. The default cast produces a view
sheet — frontal, three-quarter left, three-quarter right, and optionally profile —
by varying the pose conditioning while holding the seed and all other
conditioning fixed.

Three payoffs: reference-based adapters generalise far better across camera
angles when the reference set spans views; the cross-view ArcFace cosine is a
built-in validity check that the cast produced *one person seen from several
angles* rather than several people; and profile views are the only way to anchor
ear jewelry and helix piercings, which are invisible frontally.

Optionally extend the sheet across expression (for dentition) and lighting for a
more robust set, at linear cost.

### 11.3 Curation and storage

References are stored as images plus precomputed ArcFace embeddings and detected
landmarks, so downstream operations never re-detect. Each reference carries its
seed, model, conditioning hash, detail-plan hash, scorecard, and the ArcFace
cosine to the set centroid. Reference weighting for the multi-photo path defaults
to that centroid cosine, so the most representative faces dominate — and remains
hand-overridable.

### 11.4 Tiers

The cross-model promise is honest and tiered.

| Tier | Mechanism | Requires | Strength |
|---|---|---|---|
| **A** | face-reference adapter (IP-Adapter-Plus-Face / FaceID) + landmark conditioning + detail compositing + optional face-swap finisher | an adapter port for the family | strongest |
| **B** | native generation, then face swap from the reference set, then face restoration, then detail compositing | nothing family-specific | good; the universal path |
| **C** | baked per-base adapter (textual inversion or LoRA) from the reference set, plus detail compositing | a trainer for the base | strongest for prompt-native use, highest cost |

Tier A applies where face-reference adapters exist in plakat. Tier C applies where
a trainer exists. **Tier B applies everywhere**, which is what makes the "all
supported models" claim defensible.

**Detail compositing is tier-independent** and runs on all three. This is the
quiet win of §8: the small distinguishing features that make a persona *feel*
specific are the ones that work identically on every family, because they never
go through a sampler.

### 11.5 The swap-and-restore bridge

The universal path composes components plakat already has, all numerically
verified against their upstream references:

```
generate on any family with the compiled prompt + available conditioning
  → detect faces (SCRFD)
  → select the target figure (single face, or pose/region assignment for multiperson)
  → swap in the persona's canonical face from the reference set
  → face-restore the swapped region at gentle strength, identity-preserving
  → feather-composite
  → run the detail compositing pass (§8.4) against the REALISED landmarks
  → score (§12); retry or repair on failure
```

Detail compositing runs *after* the swap, deliberately: the swap replaces the
face region wholesale and would destroy any mark composited before it. This
ordering is a hard constraint in the pipeline and a test case.

This requires **no new model ports** and inherits the verification already
committed for each component. It is not elegant — identity comes from a
post-process rather than from the sampler — but it converts "identity works on two
families" into "identity works everywhere", which is the actual user requirement.

Known limitations, to be documented rather than hidden: swap quality degrades at
extreme pose and at small face pixel-area (§14.1); lighting transfer is imperfect
and the restoration pass is what reconciles it; the swap operates on the face
region only, so hair, build, body-sited marks and body jewelry must come from the
prompt, geometry and composite paths.

### 11.6 Baking

```
plakat persona bake alice.hjson --base <base> --method ti|lora
```

Trains from the reference set using the existing trainers, producing a per-base
artifact stored in the persona directory. Textual inversion is cheaper, smaller,
and composes with any prompt; LoRA is stronger and heavier. Subject-style training
with class prior preservation is the appropriate configuration, since the goal is
a specific subject rather than a style.

Because the reference set carries composited details, a baked adapter learns them
too — which is desirable for permanent marks and undesirable for swappable
jewelry. `bake` therefore defaults to **excluding presentation-class jewelry**
from the reference set it trains on, recompositing the references without worn
jewelry first unless `identity_locked` is set. This is a subtle behaviour and it
must be stated in the output.

Baking is optional, per-base, and must be honestly gated on host memory — several
of the transformer trainers are memory-bound well beyond typical single-GPU
budgets, and `bake` must refuse cleanly with a capability message rather than
attempting and OOM-ing.

Baked artifacts are invalidated by structural edits, by mark edits when the mark
was baked in, and by lexicon or topology version changes; the invalidation is
recorded and reported, never silently ignored.

### 11.7 Honest scope: body identity

Every identity mechanism listed above is **face-only**. There is no body
equivalent of ArcFace in the stack, no body-reference adapter, and no body swap.
Figure attributes are conditioned through the pose skeleton, the silhouette mask,
and the prompt — all comparatively weak signals — and are recoverable only by
baking, where the trainer sees whole reference images.

Body-sited details (forearm tattoo, wrist jewelry) are an exception in the
persona's favour: they are composited against the body skeleton and therefore
work as well as their landmark detection does, which for large body parts is
reasonably well and for hands is poor (§8.5).

This must be stated in the documentation, surfaced by controllability grades in
the TUI, and reflected in the scorecard's weighting. Presenting `figure` as
equally controllable would be the single most misleading thing this feature could
do.

---

## 12. Layer 4 — The scorecard

`plakat persona verify <spec> --image <png>` produces a per-attribute measurement
of how well a render matches the spec that produced it. This layer is built
**first** among the generative layers, because unevenness cannot be fixed before
it can be measured.

### 12.1 Probe types

| Probe | Mechanism | Measures | Output |
|---|---|---|---|
| `landmark` | face detection + 106-point alignment; compute the named metric | geometric attributes: spacing, widths, ratios, tilts, projections | signed delta in spec units |
| `detect` | open-vocabulary detection with the attribute's query phrase | salient objects: glasses, earrings, necklace, beard, braces, visible teeth | present/absent + confidence + box |
| `clip_probe` | contrastive cosine against `"a person with {value}"` versus `"a person with {anti}"`, normalised | colour, texture, material attributes | score in `[0,1]` |
| `region_color` | landmark-derived region mask, robust colour statistic in CIELAB | eye colour, hair colour, skin tone, teeth shade, metal tone | ΔE against target |
| `local_anomaly` | crop a neighbourhood around the resolved anchor; test for a statistically significant colour/luminance deviation from the surrounding skin, and locate its centroid | moles, scars, birthmarks, small marks | presence confidence + position error |
| `region_structure` | edge/contour statistics within a region mask | teeth alignment, diastema presence, gum show | metric-specific |
| `identity` | ArcFace cosine against the reference-set centroid | identity drift | cosine |
| `aesthetic` | LAION predictor | render quality, reported separately | score |

Every lexicon entry names its probe and tolerance. Attributes with no defined
probe are marked *unmeasurable* and excluded from the score — explicitly, with a
count, so the coverage of the scorecard is always visible.

Two probes deserve elaboration:

**`local_anomaly`** is the probe that makes marks verifiable, and it exists
because open-vocabulary detection does not reliably find a four-pixel mole. It
works in the opposite direction: rather than searching the image for a mole, it
goes to the position where the mole *should* be and asks whether the skin there
differs from its neighbourhood in the expected way. That is a far easier
question, it is robust at small scale, and it yields a *position error* as well as
a presence flag — which is exactly what §2.1.4 says is failing. It also generates
false-positive checks for asserted-empty collections: a persona with `marks: []`
is scanned for anomalies anywhere on the face, and unrequested blemishes count
against the score.

**`region_structure`** handles dentition, where colour probes are uninformative
and detection is coarse. Within the inner-lip mask it measures the count and
spacing of vertical edges (tooth boundaries), the presence of a central gap for
diastema, and the ratio of gum-coloured to tooth-coloured area for gum show.
Crude, but sufficient to distinguish `even` from `gapped` and to catch the common
failure of a mouth full of undifferentiated white.

### 12.2 Scoring

Per attribute:

```
error(a)  = normalised deviation of measurement from spec value
pass(a)   = error(a) ≤ tolerance(a)
weight(a) = priority(a) × control(a, family) × class_weight(a)
```

Aggregate score is the weighted pass fraction over *known, manifesting,
measurable, requested* attributes. **Four exclusions** matter and must be
reported separately, not folded into a single number:

- **Unknown** attributes (§6.4) — never authored.
- **Unmeasurable** attributes — no probe defined.
- **Dropped** attributes — pruned by the budget solver on this family (§9.2), so
  the model was never asked. Note that composited details are *not* dropped when
  they leave the prompt: they were realised by another mechanism and are still
  scored.
- **Non-manifesting** attributes (§8.6) — not visible in this render, such as
  dentition in a closed-mouth portrait or a nape piercing under loose hair.

Reporting a single aggregate without these four denominators would let a persona
score 100% while expressing almost nothing.

**Detail sub-score.** Because details are realised by a different mechanism, they
get their own reported sub-score with three components: *presence* (was the mark
produced), *position* (how far from its anchor), and *fidelity* (does its colour
and size match). A persona can score well structurally and badly on details, or
the reverse, and collapsing them hides the diagnosis.

Output is a table — attribute, target, measured, delta, tolerance, verdict, grade —
plus JSON for tooling, in the established convention.

### 12.3 Rejection sampling

```
plakat persona render alice.hjson --attempts 8 --min-score 0.85
```

Generate, score, keep the best; stop early on threshold. Escalation between
attempts is stepped and recorded, so a run is reproducible and explainable:

1. new seed only;
2. raise reference-adapter weight;
3. enable or strengthen geometric conditioning;
4. re-rank the salience solver to promote the failing attributes;
5. escalate a failing region to a native-resolution refinement pass (face, mouth,
   or hand, per §14.1);
6. fall back to the swap bridge if identity is the failing dimension.

Detail failures do not trigger a re-roll. A missing or misplaced mark is a
compositing failure, not a sampling failure, and is fixed by recompositing —
which is why the detail sub-score is separated out.

### 12.4 Attribute-targeted repair

When the scorecard localises the failure to a single attribute, re-rolling the
whole image is wasteful and loses everything that was right.

```
failing attribute
  → mask source, in preference order:
      1. the landmark region mask for that attribute (§10.3) — always available
      2. open-vocabulary detection on the attribute phrase — for salient objects
      3. the detail's own anchor neighbourhood — for marks
  → grow + feather
  → repair action by class:
      detail      → recomposite (milliseconds, deterministic)
      surface     → regional inpaint with an attribute-focused prompt
      structural  → escalate to a region re-render, or accept and report
  → re-score; accept only on improvement, else revert
```

This composes the existing selection, detection and inpaint verbs into a closed
loop, and is the single most compelling demonstration of the whole feature:
*"eye colour is wrong" → fix the eyes, keep the render.*

Repair is also the mechanism for surface- and detail-attribute edits on an
existing reference set (§6.5), which is what makes "same person, different hair
colour" and "same person, remove the earrings" cheap. The jewelry case is the
cheapest of all: worn jewelry is presentation-class, so changing it is a
recomposite over an unchanged reference.

### 12.5 Regression gate

The scorecard is a testable artefact and belongs in the verification harness:

- **Structural tier** — spec → resolved spec → prompt/geometry/detail-plan
  byte-comparison against committed goldens. No weights, no network, runs on
  every push.
- **Per-module tier** — geometry engine landmark output against a committed
  reference; anchor resolution under deformation; probe implementations against
  committed reference images with known ground truth, including synthetic images
  with marks composited at known positions so `local_anomaly` has exact ground
  truth.
- **End-to-end tier** — a committed persona rendered at a fixed seed on each
  family, scorecard compared against a committed baseline within tolerance,
  detail sub-score included. Weight-backed, runs on the slower cadence.

The synthetic-ground-truth trick for `local_anomaly` is worth calling out: since
the compositor is deterministic and its output positions are known exactly, the
probe can be validated against its own compositor's output without any manual
annotation. That gives a precise, free, regenerable test set for the hardest
probe in the system.

---

## 13. Layer 5 — Calibration

Calibration is what converts the schema from a suggestion box into an instrument.
It is a slow, offline, per-family process whose outputs are committed tables.

### 13.1 Prior measurement

For each family: generate an unconditioned (or minimally conditioned) face
population at fixed settings, detect and align, and compute the distribution of
every landmark metric the lexicon probes. The median becomes that family's
`prior(a, family)`, i.e. the meaning of `0.5` (§6.3). The spread becomes the
family's usable range and informs the tolerance defaults.

The same population yields two additional baselines that the detail subsystem
needs: the **spontaneous-detail rate** (how often the family produces unrequested
moles, blemishes, jewelry, or facial hair) and the **prompted-detail hit rate**
(how often a requested one appears, and on the correct side). The first
calibrates the asserted-empty negatives (§9.4); the second quantifies exactly how
much §8's compositing approach is buying, per family.

Population size, prompt, sampler, steps, and size are all part of the calibration
identity and are recorded with the table.

### 13.2 Response curves

For each geometric attribute and family: sweep the scalar across its range
holding everything else fixed, render `n` seeds per step, measure the realised
metric, and fit a monotone curve. The result is the empirical transfer function
from *requested* to *realised*.

The compiler pre-distorts through the inverse of this curve, so that requesting
0.7 lands near 0.7 rather than near 0.56. Where the curve is non-monotone or flat,
the attribute is not controllable on that family and is graded accordingly.

Composited details do not need response curves — their realisation is exact by
construction. They do need a **harmonisation calibration**: the strength at which
the blending pass integrates the composite without erasing it. Too low and the
mark reads as a sticker; too high and the sampler removes it. This is a single
scalar per family per detail kind, measured by sweeping strength and scoring with
`local_anomaly`, and it is the one calibration the detail path genuinely
requires.

Sweeps are expensive. They run offline, produce committed tables, and are
regenerated only under §13.4.

### 13.3 Grade assignment

Grades derive from the fitted curve, never from opinion:

| Grade | Criterion |
|---|---|
| `strong` | monotone, slope above threshold, low seed-variance |
| `moderate` | monotone, reduced slope or elevated variance |
| `weak` | detectable but small effect, or high variance |
| `experimental` | no reliable effect measured |

Grades feed the salience weighting (§9.2), the scorecard weighting (§12.2), lint
warnings (§6.6), and the TUI badges (§17.7).

Composited details are graded by their measured presence and position error
post-harmonisation rather than by a response curve, and will generally land
`strong` — which is the point.

### 13.4 Recalibration policy

Tables are invalidated by: a new model family; a change to a family's default
sampler, step count, or native size; a lexicon change that alters a deformation
basis or gain; a landmark topology change; a change to the landmark aligner; and,
for the harmonisation constants, a change to the compositor. Each table records
the inputs it was measured under, and a mismatch produces a staleness warning
rather than a silent wrong answer.

---

## 14. Rendering policy

### 14.1 Framing and the region-escalation ladder

Feature pixel-area determines which mechanisms function at all. A full-body render
at typical resolution puts the face at a small fraction of the frame, where
face-reference adapters degrade, swapping produces artefacts, and a four-pixel
mole is not representable.

The renderer therefore branches on measured area, and the same ladder applies at
three scales:

```
render at requested framing
  → detect; measure the area of each region of interest
  → face  area < threshold  → crop with margin, refine at native resolution
                              with full identity conditioning, composite back
  → mouth area < threshold  → after the face pass, refine the mouth region for
    (when dentition          dentition (§8.7)
     manifests)
  → hand  area < threshold  → refine the hand region for jewelry, if any
    (when hand jewelry       (best-effort; §8.5)
     is present)
  → detail compositing runs against the final realised landmarks
```

Each escalation is a crop, a native-resolution refinement with focused
conditioning, and a feathered composite back — the same mechanism at three
scales. Thresholds are committed constants, exposed as flags, and measured during
calibration.

This wires the existing face-restoration and detail-refinement passes in
automatically instead of leaving the user to discover that identity, or teeth, or
a ring silently stopped working when they asked for a wider shot.

### 14.2 Multiperson

Multiple personas in one frame is the hardest case and inherits four known
problems:

- **Assignment** — which face belongs to which figure. Solved as it is today, by
  pinning a pose skeleton per region and assigning by region overlap. Detail
  compositing inherits the same assignment: each persona's marks composite only
  against its own assigned figure's landmarks.
- **Cross-contamination** — face-reference adapters are not natively regional;
  conditioning bleeds between figures. Mitigations, in order of preference:
  per-region conditioning via the existing soft-mask path; sequential per-figure
  refinement passes at native resolution (§14.1) after a base render; or generate
  the scene without identity and apply the swap bridge per figure.
- **Budget** — N personas multiply the token cost. Detail routing helps
  disproportionately here, since every composited mark is budget the scene keeps.
  The salience solver runs per-persona with a divided budget and drops harder as N
  grows.
- **Detail misattribution** — figure A's scar composited onto figure B is a
  catastrophic and very visible failure. The compositor therefore refuses to place
  a detail when the figure assignment confidence is below threshold, reports the
  refusal, and leaves the mark absent rather than wrong.

The sequential-refinement path is the recommended default for N > 2: render the
scene for composition, then refine each figure's face region individually with
that persona's full conditioning, then composite that persona's details.

### 14.3 Scene composition

Persona conditioning and scene conditioning are merged by §9.5's precedence. The
geometric conditioning map must be positioned to match the figure's location in
the frame — for single-subject portraits this is the frame itself; for scenes it is
derived from the pose skeleton's head region, which means pose conditioning is
effectively a prerequisite for geometric face conditioning in multi-figure
scenes.

---

## 15. Lineage operations

The spec representation makes several derivations natural, and they are cheap to
add once the resolver exists.

**Aging.** `plakat persona derive alice.hjson --at-age 55` applies a committed
age-transform to the spec: lexicon-defined deltas on skin lines, greying, volume
loss, hairline, jaw definition — and on details, which is where the transform
earns its keep. Scars mature (`maturity` advances toward 1.0), birthmarks may
darken or spread slightly, teeth `shade` drops and `wear` rises, and age-related
marks may be added at a lexicon-defined rate. The result is a new spec with
`provenance.derived_from` set, and it re-casts to a recognisably-related but
distinct person. The relationship is quantified by ArcFace cosine between the two
reference sets — related-but-aged should land in a measurable band, and that band
is itself a test.

**Blending.** `plakat persona blend a.hjson b.hjson --weight 0.5 [--age 8]`
produces a plausible offspring or intermediate. Scalars interpolate; enums select
by weighted draw from a fixed seed, or by a lexicon-defined dominance table where
one exists. Details do **not** interpolate — a scar is not half-inherited — so
the blend draws each detail independently with probability proportional to weight
and a lexicon `heritable` flag: birthmarks and dentition traits are heritable,
scars and piercings are not. Combined with the age transform, this is the
spec-native form of the parent-blend portrait workflow.

**Variation.** `plakat persona vary alice.hjson --sigma 0.1 --count 6` perturbs
scalars by a seeded Gaussian, respecting class (structural only, surface only, or
detail only). This is the engine behind the TUI's evolve mode (§17.9) and a fast
way to populate a cast of related-looking extras. Detail variation perturbs
positions and sizes rather than adding or removing details, unless
`--vary-details add` is given.

**Sibling / family sets.** Repeated blending and variation from shared parents,
with a shared random seed, produces a coherent family. Useful, cheap, and it falls
out of the above three operations without new machinery.

---

## 16. The inverse direction: photograph → spec

`plakat persona extract <image> [--out alice.hjson]`

Extraction is not a convenience feature; it is the **validation mechanism for the
entire lexicon**, and it should be built early.

**Mechanism.** Detect and align the face; compute every `landmark` probe metric
(§12.1) and invert through the calibration curves to recover scalars; run
`detect` probes for salient objects (glasses, earrings, visible jewelry, braces);
run `region_color` probes for iris, hair, skin and teeth; run a **mark sweep** —
`local_anomaly` applied across the whole face rather than at a known anchor —
to find moles, scars and birthmarks, and record each as a detail with its
resolved anchor, size, orientation and colour; estimate `apparent_age` and build
from available estimators or leave unknown.

The mark sweep is the interesting half. It is the same probe running in discovery
mode: segment the face into a fine grid, test each cell for a local colour or
luminance anomaly against its neighbourhood, cluster the hits, classify each
cluster by shape (compact → mole; elongated → scar; large and soft-edged →
birthmark; many small and distributed → freckles), and emit the corresponding
detail record anchored to the nearest named landmark region. Classification will
be imperfect and every extracted detail therefore carries a confidence and a
`method: extracted` tag; low-confidence detections are emitted commented-out in
the HJSON for the author to accept or delete, which is a friendlier failure mode
than either dropping or asserting them.

Every extracted field carries a confidence, and low-confidence fields are written
as `unknown` rather than guessed.

**Why it matters:**

1. **Round-trip test.** `photo → spec → cast → extract → spec'`. The distance
   between `spec` and `spec'` is the closure error, and it is the strongest
   available evidence that the scalars mean anything at all. For details the
   round-trip is far tighter than for scalars — a composited mole should extract
   at nearly its exact anchor — which makes the detail round-trip a sensitive
   regression test for the whole anchor-resolution chain.
2. **Bootstrapping.** The existing People library and any folder of reference
   photos become editable specs.
3. **Parametric editing of real references.** "This person, ten years older,
   without the beard, without the earrings" becomes a spec edit rather than a
   prompt gamble.
4. **TUI prefill.** Extraction turns the interview from authoring into
   confirmation (§17.6), which is dramatically faster and more accurate — and for
   marks it is transformative, since placing a mole by eye is far harder than
   confirming one the system already found.

**Constraint.** Extraction produces a *description*, not a likeness claim. The
documentation must be explicit that a spec derived from a photograph of a real
person is a set of measurements, and that the system is not a cloning tool
(§23.2).

---

## 17. The composition TUI

`plakat persona --tui`

### 17.1 Design premise

This is a **forensic composite** interface, and that literature has one finding
that should drive the entire design: feature-by-feature *verbal recall* produces
poor likenesses, while *comparative recognition* produces good ones. People are
bad at "describe the nose" and excellent at "which of these six is closest."

Therefore the Q/A flow almost never asks the user to name a value. The question
text ("Define the eyes — spacing") is framing; the answer mechanism is
**selection among rendered candidates**, with a slider for refinement afterwards.

Marks and jewelry are the exception that proves the rule: for those, the natural
answer mechanism is neither naming nor selecting but **pointing**, which is why
§17.4 introduces a spatial placement widget.

### 17.2 Headless interview core

The most important implementation decision in this section: **the interview engine
is headless, and the TUI is a view over it.**

```rust
pub fn next_question(lex: &Lexicon, answers: &AnswerLog, depth: Depth)
    -> Option<Question>;
pub fn apply(answers: &mut AnswerLog, id: QuestionId, a: Answer)
    -> PartialSpec;
pub fn progress(lex: &Lexicon, answers: &AnswerLog) -> Progress;
```

Pure functions over data structures, no terminal, no I/O. This buys:

- unit-testable flow logic with no PTY and no snapshot fragility;
- a non-interactive `--answers answers.hjson` mode for scripting, CI, and
  reproducing a user's session from a bug report;
- a byte-stable corpus entry proving a fixed answer sequence always produces the
  same spec;
- the ability to drive the same interview from a future non-terminal surface
  without rewriting it.

Collection questions (marks, piercings, jewelry) complicate this slightly: an
answer may be *"add an item"*, which pushes a nested sub-interview onto the
question stack. The core models this explicitly as a stack rather than a flat
cursor, and the answer log records entry and exit, so replay is exact.

### 17.3 The question graph is data

The graph is generated from the lexicon (§7.2), not written in Rust. Each entry
supplies `ask`, `section`, `order`, `depth`, `widget`, `member_widget`,
`variants`, `help`, a `when` condition, `gated_by`, and optional `gates`:

```hjson
facial_hair.style: {
  ask: "What kind of facial hair?"
  section: facial_hair
  widget: select
  depth: standard
  when: "identity.sex != female || flags.allow_all"
  gates: { none: "skip_section(facial_hair)" }
}

teeth: {
  ask: "Will this person be shown with their mouth open or smiling?"
  section: teeth
  widget: select
  options: [rarely, sometimes, often]
  gates: { rarely: "skip_section(teeth)" }
  note: "Teeth only appear in open-mouth renders."
}
```

Conditions are evaluated against the answer log with a small, total expression
language — comparison, boolean composition, presence tests, nothing
Turing-complete. Adding an attribute, a new mark kind, or a new piercing site
requires a lexicon entry and no TUI code, and the lexicon and the interview can
never drift apart.

Ordering is **coarse to fine, structural before surface before detail**, which
matches both the forensic-composite convention and the invalidation semantics of
§6.5: settle the things that force a re-cast before spending time on the things
that do not. Marks and jewelry come last precisely because they are free to
change.

### 17.4 Widgets

A small, closed taxonomy:

| Widget | Interaction | Used for |
|---|---|---|
| `select` | horizontal variant strip, `←/→`, `Enter` | enums with visual variants |
| `scalar` | variant strip for coarse choice, then `+/-` slider for refinement | all `[0,1]` scalars |
| `multi` | checklist, `Space` toggles | piercing sites, appliance options |
| `compare` | forced A/B, several rounds, binary search | attributes users cannot verbalise |
| `text` | line editor | name, notes, hair style prose, tattoo motif |
| `number` | numeric entry with range validation | age, height, orientation degrees |
| `color` | swatch grid, with a Lab entry escape | iris, hair, skin, mark, metal, stone |
| **`place`** | live crosshair over the wireframe face or body; `hjkl`/arrows move, `Enter` drops, `HJKL` for fine steps | mark anchors, piercing sites, tattoo positions |
| **`list`** | a managed collection: `a` add, `e` edit, `d` delete, `Enter` open; each entry runs a nested sub-interview | `marks`, `piercings`, `jewelry.items`, `teeth.features` |
| **`tooth`** | a rendered dental arch; select a tooth position by name or by pointing | `teeth.features` entries |

Three of these are new and carry the weight of this revision:

**`place`** is the crosshair pick-mode pattern already established in the photo
manager's retouch tools, applied to the wireframe. Because the wireframe is
CPU-cheap and redraws instantly, the user sees the mark move in real time against
the actual face geometry they have been composing. On drop, the crosshair
position is converted to the *nearest named landmark plus an offset* rather than
stored as a raw coordinate — so the authored anchor is anatomical from the moment
it is created, per P6. The status line shows the resolved anchor
(`left-nasolabial-upper +0.02,−0.03`) so the author can see and hand-edit it
later.

**`list`** turns a collection into a small nested interview. Adding a mark asks:
kind → placement (`place`) → size → colour → kind-specific follow-ups (maturity
and orientation for a scar; edge and intensity for a birthmark; raised and hairs
for a mole). Adding jewelry asks: kind → site (constrained to sites that exist in
`piercings`, with an inline "add a piercing there first?" escape) → style → metal
→ stone. The nested interview is generated from `member_widget` in the lexicon,
so a new mark kind needs no code.

**`tooth`** renders the dental arch as a labelled diagram and lets a feature be
attached to a named position, avoiding both numeric notation and free text.

**`[u] unknown` is available on every question**, is semantically distinct from
choosing the middle value (§6.4), and is the correct answer for most attributes in
a quick pass. For collections there is an additional explicit distinction, offered
as two separate options: **"none — this person has no marks"** (asserted empty,
which emits negatives and is scored) versus **"skip — I haven't decided"**
(unknown). The UI must make both feel like legitimate answers.

### 17.5 Preview tiers

**Tier 1 — geometry wireframe.** The landmark engine is CPU-only and
microsecond-cheap, so it redraws on every keypress. This is what makes a scalar
slider feel like an instrument rather than a form field: you *see* the eyes move.
Always available, no weights, no GPU.

The wireframe also renders **detail markers** — a small glyph at each resolved
mark, piercing and jewelry anchor, highlighted when its entry is selected in the
list. This is what makes the `place` widget usable, and it means the entire
detail-authoring workflow needs no GPU at all.

On terminals without a graphics protocol, the wireframe rasterises to braille or
half-block glyphs. This is a strictly better degradation story than a placeholder
box, and it means the entire structural and detail half of the interview is usable
over a plain SSH session.

**Tier 2 — diffusion preview.** Debounced after the last edit, executed on a
resident model worker, cancellable mid-denoise via the existing step-hook path.
Few-step distilled presets are what make this viable at all — a low-resolution
few-step render is inside the tolerance for a live editor, where a full-step
render is not. The detail compositing pass runs on the preview too, which is
cheap and means marks appear in it.

Tier 2 is optional, memory-guarded, off by default on constrained hosts, and
toggleable at runtime. **The TUI must be fully usable with Tier 2 disabled** — see
§28.4.

### 17.6 Entry modes

The interview should rarely start blank:

| Mode | Behaviour |
|---|---|
| **Blank** | full interview at the chosen depth: `quick` (~12 structural questions), `standard` (~30), `full` (every leaf, including all detail collections) |
| **From photo** | extraction (§16) prefills; the interview degrades to confirm-and-adjust, each prefilled answer showing its measured confidence. Discovered marks are presented for accept/reject/adjust one at a time — by far the fastest way to author a detailed persona |
| **From prose** | the user types a paragraph; an LLM maps it to a partial spec, with an offline keyword mapper as the deterministic fallback, mirroring the established prose→spec provider pattern; the interview confirms. Prose commonly mentions marks ("a scar through one eyebrow") without position, so these arrive as region-shorthand anchors for the `place` widget to refine |
| **From persona** | fork an existing spec for a sibling, an aged variant, or a lineage blend (§15) |
| **From scenario** | import an existing free-text scenario persona as a prose prefill |

### 17.7 Layout

```
┌ persona: alice ──────────────────── standard · 24/38 ── mem 6.2/24 GB ─┐
│ SECTIONS      │ Marks — scar #1 — placement       │  PREVIEW          │
│ ▸ basics    ✓ │                                   │      ___          │
│ ▸ face      ✓ │  Point to where the scar sits.    │    /  o  \        │
│ ▸ eyes      ✓ │                                   │   | ✚   o |  ←    │
│ ▸ nose      ✓ │   hjkl move · HJKL fine · Enter   │   |  ─── |        │
│ ▸ mouth     ✓ │                                   │    \____/         │
│ ▸ teeth     — │   anchor: right-brow-outer        │                   │
│ ▸ skin      ✓ │           +0.00, +0.01            │  ◆ strong control │
│ ▾ marks   2/3 │                                   │  (composited)     │
│    mole     ✓ │   length 0.16  ──────●──────      │                   │
│    scar     ◂ │   angle   68°                     │ ─── SPEC ───      │
│    [+ add]    │                                   │ marks: [          │
│ ▸ piercings ✓ │   [u] unknown   [d] delete        │  { kind: mole ... │
│ ▸ jewelry   · │                                   │  { kind: scar     │
│ ▸ hair      · │   ? Scars mature from red to      │    anchor: {...}  │
│ ▸ figure    · │     pale; set maturity next.      │    length: 0.16   │
└───────────────┴───────────────────────────────────┴───────────────────┘
 Tab next · hjkl place · +/- adjust · u unknown · Ctrl-K palette · F1 help
```

The left pane is simultaneously a progress indicator and a jump table. This is
what rescues a linear interview from being a prison: answer in order the first
time, then navigate freely forever after. Section state shows answered /
partial / untouched / not-applicable (the `—` beside `teeth`, gated off), and the
controllability badge sits next to the active question so expectations are set at
the moment of authoring. Composited details carry a `(composited)` note beside
the badge, because "this one is reliable" is genuinely useful information at
authoring time.

The figure section replaces the face preview with a parametric silhouette and
proportion guides, and `place` operates against it for body-sited marks and
jewelry. The teeth section switches the preview to the open-mouth wireframe with
the dental arch drawn.

### 17.8 Workbench mode

The interview must not terminate at "wrote alice.hjson." On completion it drops
into a second mode within the same TUI:

```
 [c] cast 16   [s] contact sheet   [v] scorecard   [e] edit   [b] bake   [x] export
```

The scorecard view is the payoff: per-attribute deltas in a colour-coded table,
with the detail sub-score broken out into presence / position / fidelity, and
**`Enter` on a failing row jumping directly back to that attribute's question**.
For a misplaced mark, that jump lands in the `place` widget with the crosshair at
the *measured* position and the target position ghosted, so the correction is
immediate and obvious. Measure, adjust, re-cast, re-measure, without leaving the
tool. That loop is the entire justification for calling this an instrument.

The contact sheet supports promotion and rejection of individual candidates into
the reference set, with the ArcFace coherence matrix displayed alongside so an
incoherent cast is visible rather than inferred.

### 17.9 Evolve mode

An optional refinement mode for the last increment that nobody can articulate:
render `n` perturbations of the current spec (§15, variation), the user picks the
closest, the spec moves toward it, repeat. Perturbation sigma decays across
rounds. Restricted to one attribute class at a time (structural, surface, or
detail) so the search space stays tractable and the invalidation semantics stay
clear.

### 17.10 Keybindings

Consistent with the existing TUIs; no new idioms except the placement chords,
which follow the retouch crosshair convention.

| Key | Action |
|---|---|
| `Tab` / `Shift-Tab` | next / previous question |
| `←` `→` | cycle variants |
| `+` `-` | adjust scalar |
| `hjkl` / arrows | move the placement crosshair |
| `HJKL` | fine placement steps |
| `Space` | toggle (multi) |
| `a` / `e` / `d` | add / edit / delete (list widget) |
| `u` | unknown / skip |
| `Enter` | confirm and advance; drop the crosshair |
| `?` | expand help for the current attribute |
| `p` | cycle preview tier |
| `m` | toggle detail markers on the wireframe |
| `h` | toggle the live spec pane |
| `Ctrl-S` | save |
| `Ctrl-Z` / `Ctrl-Shift-Z` | undo / redo |
| `Ctrl-K` | command palette |
| `:` | command pane |
| `F1` | cheatsheet |

### 17.11 Persistence and safety

- **Save on every answer.** The partial spec is always a valid HJSON file;
  `Ctrl-S` is a formality. Crash-safe by construction.
- **Persist the answer log, not just the spec.** It makes a session auditable,
  replayable, reproducible from a bug report, and migratable when the lexicon
  version bumps. The nested sub-interview entries are recorded with their stack
  depth so collection authoring replays exactly.
- **Class-aware edit warning.** Editing a structural attribute on a persona that
  already has references or adapters prompts before invalidating them; editing a
  surface attribute offers repair-in-place; editing a detail says nothing at all,
  because it costs nothing (§6.5).
- **Workspace wizard first.** Per the established convention, launching with no
  workspace runs the workspace wizard before anything else.
- **Variant thumbnail cache.** Geometry attributes render live from the wireframe;
  colour, texture and jewelry attributes cannot be shown by a wireframe and need
  real images. These are generated once per lexicon version into the workspace
  cache rather than shipped in the release archive, so the archive stays small and
  the cache stays regenerable. The jewelry asset library is the exception: it
  ships, because it is needed for compositing and not merely for preview.

### 17.12 Non-interactive parity

Everything the TUI does must be reachable without it, because the headless core
makes that nearly free: `--answers file.hjson` replays a session; `persona new
--depth quick --defaults` scaffolds; `persona set alice.hjson eyes.spacing=0.62`
edits a single field with the same validation path; `persona mark add alice.hjson
--kind scar --at right-brow-outer --length 0.16` manipulates collections. The TUI
is the preferred surface, never the only one.

---

## 18. CLI surface

```
plakat persona new     <out.hjson> [--depth quick|standard|full] [--from-photo IMG]
                                   [--from-prose TEXT|FILE] [--from PERSONA]
plakat persona --tui   [SPEC]                     # composition TUI (§17)
plakat persona lint    <spec>                     # schema, contradiction, budget, safety
plakat persona show    <spec> [--model M] [--json]# resolved spec, prompts, drops, grades,
                                                  # detail plan, manifestation report
plakat persona geometry<spec> [--out MAP.png]
                       [--kind landmark|depth|wireframe|mask|dentition|details]
plakat persona extract <image> [--out spec] [--marks on|off]
plakat persona cast    <spec> [--count N] [--keep-best K]
                       [--views sheet|frontal] [--expressions neutral,smile]
plakat persona render  <spec> [--model M] [--scene TEXT] [--framing F]
                              [--attempts N] [--min-score S]
                              [--jewelry all|none|LIST] [--details all|none|LIST]
                              [--suppress-details ID ...]
plakat persona verify  <spec> --image IMG [--json] [--details-only]
plakat persona repair  <spec> --image IMG [--attribute A ...] [--recomposite]
plakat persona composite <spec> --image IMG       # run the detail pass alone
plakat persona bake    <spec> --base B [--method ti|lora] [--with-jewelry]
plakat persona derive  <spec> [--at-age N] [--vary SIGMA] [--count N]
plakat persona blend   <a> <b> [--weight W] [--age N]
plakat persona sheet   <spec> [--views ...] [--expressions ...] [--lighting ...]
plakat persona mark    add|edit|rm <spec> [--kind K --at ANCHOR --size S ...]
plakat persona jewelry add|edit|rm <spec> [--kind K --site S --style ST ...]
plakat persona ls | show-library | rm | export | import
plakat persona migrate <spec>
```

Flags follow the existing grouped-help convention, with sections for Spec,
Casting, Conditioning, Details, Scoring, and Output plus the shared global group.

`persona composite` as a standalone verb is worth having: it lets a user apply a
persona's marks and jewelry to an image that was generated some other way, which
is both a useful escape hatch and the simplest possible integration test for the
detail subsystem.

---

## 19. Artifact layout

Consistent with the plain-text, no-hidden-database doctrine. Everything except
`persona.hjson` is derived and safely deletable.

```
<people>/alice/
  persona.hjson              # source of truth; hand-editable
  answers.hjson              # interview transcript (§17.11)
  resolved.hjson             # cached resolution; derived
  geometry/
    landmarks.json
    landmark.png  depth.png  wireframe.svg  dentition.png
    masks/{eyes,nose,mouth,inner-lip,hair,cheek-left,...}.png
  details/
    plan.json                # resolved anchors, z-order, strategies
    overlays/                # generated procedural overlays, cached
  refs/
    frontal-01.png  three-quarter-left-01.png  smile-01.png  ...
    embeddings.bin           # ArcFace vectors + landmarks, precomputed
    coherence.json           # pairwise cosine matrix
  adapters/
    <base>.safetensors       # baked; per base
  scorecards/
    <timestamp>-<model>.json
  cache/
    variants/                # TUI thumbnails, keyed by lexicon version
  lock.hjson                 # resolved seeds, model ids, lexicon +
                             # topology + calibration versions
```

`lock.hjson` is what makes a persona reproducible months later: it pins the
lexicon version, the **landmark topology version** (without which every anchor is
suspect), the calibration table version, the model identifiers, and the seeds used
for each reference. Without it, a lexicon change silently redefines the persona.

**Portable bundle.** `persona export` produces a single archive of the directory;
`persona import` restores it, validates the schema and topology versions, and
reports any staleness. A persona should be shareable as one file. Export
optionally excludes `refs/` and `adapters/` for a spec-only share, which is the
form most people will want to publish.

---

## 20. Integration surfaces

Following the established rule that a feature stranded on one surface is not
finished:

| Surface | Integration |
|---|---|
| **scenario** | `type: persona` task; `personas:` entries accept a spec path alongside the existing free-text form; per-task overrides for scene, framing, attempts, min-score, `jewelry`, and `details` |
| **compile** | a `persona:` directive in a prompts file resolves to a persona task; the spec's compiled prompt participates in the existing family-aware rewriting |
| **scripting** | `plakat.persona.load` · `.render` · `.cast` · `.verify` · `.geometry` · `.composite` · `.derive` — pushing image handles into the existing image-handle pipeline so a persona flows through save / upscale / relight / metadata like any generated image |
| **library API** | a `Persona` builder mirroring the existing builder shape, returning in-memory images plus the scorecard and the detail plan |
| **generation UI** | the People screen becomes the persona library: create, edit (launching the composition TUI), cast, bake, and reference in chat via the existing mention syntax |
| **photo manager** | persona section in the image info panel including the detail plan; filter by persona; face-scan grouping links to personas instead of anonymous groups; "extract persona from this photo" opens the TUI prefilled with discovered marks |
| **multiperson** | accepts persona specs where it currently requires photographs; §14.2 governs, including per-figure detail attribution |
| **import** | `--import <album>` on every persona command, consistent with the other image-producing commands |
| **sidecar** | persona slug, spec hash, detail-plan hash, lexicon and topology versions, and scorecard summary written into the PNG metadata, enabling `--persona-clone` to recover the spec from any image |

---

## 21. Documentation deliverables

- `Tutorials/PERSONA_TUTORIAL.md` — the guided path: TUI, cast, render, score,
  repair.
- `Tutorials/PERSONA_DETAILS_HOWTO.md` — marks, scars, birthmarks, piercings,
  jewelry and dentition end to end, including the anchor model and why details
  are composited rather than prompted. This is the tutorial that explains the most
  counter-intuitive design decision in the feature and it should not be folded
  into the main one.
- `PERSONA.md` — reference manual: full schema, every attribute, every probe,
  every flag.
- `PERSONA_LEXICON.md` — the vocabulary, the neutrality rationale, and how to
  extend it, including how to add a new mark kind or piercing site.
- `PERSONA_ANCHORS.md` — the landmark topology, every named anchor region, and the
  topology versioning policy.
- Cross-model capability matrix — per family: tier, available conditioning,
  controllability grades per attribute, detail hit rates before and after
  compositing, honest scope notes.
- Doctor integration — a persona section in the capability report showing which
  tiers, which conditioning paths, and which detail strategies are available on
  this host.

---

## 22. Memory and performance

**Model residency is the binding constraint.** The scorecard alone wants a face
detector, a landmark aligner, an open-vocabulary detector, a CLIP model, and an
identity encoder resident — on top of a generation pipeline. Naively loaded, this
OOMs on typical unified-memory hosts at exactly the moment scoring matters most.

Requirements:

- **Resident scoring worker.** A single background worker owns the scoring models,
  loads them lazily and individually, and frees the generation pipeline before
  loading and vice versa, following the established staged-free discipline.
- **Probe tiering.** Landmark, `region_color` and `local_anomaly` probes are cheap
  and always run — notably, `local_anomaly` needs no model at all beyond the
  aligner, which is why detail verification is nearly free. `detect` and
  `clip_probe` are opt-in per attribute and batched across candidates so the
  models load once per cast, not once per image.
- **Cast batching.** Score the whole candidate batch in one residency window.
- **Compositing is cheap.** The detail pass costs one face detection plus
  pure-CPU rasterisation; only the optional harmonisation touches the GPU, and it
  is a single low-strength masked img2img over a small region. A persona with
  twenty marks does not meaningfully cost more than one with two.
- **Memory guard.** Every phase is guarded; under pressure the operation refuses
  cleanly with a capability message rather than being killed.
- **Preview budget.** The TUI's Tier-2 preview holds the smallest viable pipeline
  and releases it on idle, with the memory indicator visible in the top bar. Tier
  1 plus detail markers needs no model at all.

**Caching.** Resolution, geometry maps, detail overlays, variant thumbnails, and
reference embeddings are all cached on content hashes — the resolution cache on
`(spec_hash, lexicon_version, topology_version, calibration_version, family)`, the
reference cache on the *structural* spec hash only, and the overlay cache on the
individual detail record, so surface and detail edits never trigger a re-cast.

**Cost transparency.** `persona cast` prints an estimate before starting:
renders, approximate time, peak memory. `--dry-run` prints it and exits.

---

## 23. Safety

### 23.1 Age gating

Specs with a low `apparent_age`, or whose extracted or derived age falls below the
threshold, are subject to a hard constraint: certain wardrobe, framing, jewelry
and scene combinations are refused outright.

Enforcement lives in the **resolver**, not in the CLI layer, so that no surface —
scenario, compile, Bund script, library API, TUI, or a future one — can route
around it. The check is a lint error, an emit-time error, and a render-time error;
all three, deliberately redundant. The TUI surfaces it at the moment of authoring
rather than at the moment of rendering. The lineage operations (§15) re-run the
gate on their output, since an aging or blending operation can move a persona
across the threshold.

### 23.2 Likeness and provenance

The system composes synthetic people, which is a meaningful advantage over
photo-driven identity tooling: a persona authored from a spec has no real subject
and therefore no consent question.

Extraction (§16) changes that, and the documentation must be direct about it: a
spec extracted from a photograph of a real person is a set of measurements
describing that person — including, now, the positions of their scars, birthmarks
and moles, which are among the most individuating features a person has. The
appropriate treatment is the same as for any reference photograph the user
supplies. The feature exists to bootstrap and validate; it is not a cloning tool,
and the docs should not market it as one.

Provenance metadata — persona slug, spec hash, detail-plan hash, derivation chain
— is written into every output sidecar, so any rendered image can be traced back
to the spec and the lineage that produced it.

### 23.3 Marks and dignity

Scars, birthmarks and dental features describe how real people look. Two binding
constraints follow, beyond the vocabulary rules in §7.4:

- The lexicon attaches **no valence** to any mark. There is no `disfiguring`, no
  `flaw`, no `blemish severity`. A mark has a form, a size, a colour and a
  position, and nothing else.
- Documentation and TUI help text describe marks neutrally and do not frame their
  removal as improvement. The `remove` verbs exist because specs are editable, not
  because marks are problems.

### 23.4 Not a medical tool

Nothing in the marks, birthmark or dentition vocabulary is diagnostic, and the
documentation states so plainly. The system describes appearances for the purpose
of generating images. It must not be presented as, or extended toward, clinical
depiction, and extraction output must not be described as an assessment of
anything.

### 23.5 Lexicon review

The neutrality constraints in §7.4 and §23.3 are binding on contributions.
Lexicon changes require review specifically on vocabulary, and the rationale is
recorded in the lexicon file itself so the reasoning survives contributor
turnover. The same review covers additions to the jewelry asset library, for the
trade-dress constraint in §10.5.

---

## 24. Testing and verification

| Tier | Content | Weights | Cadence |
|---|---|---|---|
| **Structural** | schema round-trip, migration, lint rules, expression language, resolver purity, manifestation gate, detail routing, salience solver, negative dedup, precedence, geometry landmark output, anchor resolution under deformation, overlay rasterisation byte-stability, compositing determinism, headless interview flow including nested collection sub-interviews, answer-log replay | none | every push |
| **Per-module** | probe implementations against committed reference images with known ground truth; `local_anomaly` against synthetically composited marks at known positions; calibration curve fitting; extraction accuracy on a committed reference set; geometry validity clamping; occlusion culling | some | every push where cheap |
| **End-to-end** | committed persona cast at fixed seed per family; scorecard within committed tolerance; detail sub-score within tolerance; identity coherence within band; round-trip closure error within budget | yes | slow cadence / release gate |

**Corpus artefacts**, regenerated by committed scripts:

1. A reference persona's resolved spec and per-family prompts — byte-stable.
2. Its geometry maps, dentition hint, and detail plan at several seeds —
   byte-stable.
3. Its detail overlays and a composite-only render over a fixed base image —
   byte-stable, since compositing without harmonisation is deterministic.
4. A fixed answer sequence replayed through the headless interview, including a
   nested mark-authoring sub-interview — byte-stable.
5. Its cast sheet and scorecard per family — tolerance-compared, not
   byte-compared.
6. The §2.3 baselines recomputed — identity variance *and* localized-detail hit
   rate — so the improvement over prompt-only personas is a committed, tracked
   number rather than a claim.

**Property tests** worth having: any valid spec resolves without panic; any
resolution is idempotent; geometry never self-intersects after clamping; anchors
always resolve inside the face box or are culled; the budget solver never exceeds
the budget; unknown attributes never appear in any emitted prompt; a composited
detail is always found by its own `local_anomaly` probe on the composited image
(the compositor and the probe are each other's test).

That last property is worth stating as a design goal rather than merely a test:
**the compositor and the detail probe must agree**. If the probe cannot find what
the compositor drew, one of them is wrong, and the pair form a closed loop that
needs no external ground truth.

---

## 25. Phasing

Ordered so that each phase is independently useful and so that measurement
precedes the work it is meant to evaluate.

| Phase | Content | Weights | Independently useful as |
|---|---|---|---|
| **0** | `PersonaSpec` v1, lexicon skeleton, resolver, manifestation gate, salience solver, per-family emitters, `lint` / `show` / `new` / `migrate`, structural corpus | none | a deterministic, testable spec→prompt compiler |
| **1** | Scorecard: probes including `local_anomaly` and `region_structure`, scoring, `verify`; the §2.3 baselines | inference | the measurement instrument; fixes nothing yet but makes everything falsifiable |
| **2** | Geometry engine: landmarks, named anchor regions, deformation basis, conditioning maps, dentition hint, region masks, `geometry` | none | conditioning maps usable by hand today |
| **3** | Detail subsystem: anchor resolution, procedural overlay generators, jewelry asset library, compositing pass, harmonisation, `composite` | partial | marks and jewelry on *any* image, from any source |
| **4** | Calibration: priors, response curves, harmonisation constants, controllability grades, committed tables | inference | turns scalars from suggestions into measurements |
| **5** | Casting, reference sets, multi-view and multi-expression sheets, coherence validation, rejection sampling, `cast` / `render` | yes | working personas on Tier-A families |
| **6** | Cross-model: swap-and-restore bridge, region-escalation ladder, mouth and hand refinement, `bake` | yes | the "all supported models" promise |
| **7** | Repair loop, `repair`, surface- and detail-incremental re-cast | yes | cheap iteration |
| **8** | TUI: headless interview core, question graph, widgets including `place` / `list` / `tooth`, wireframe preview with detail markers, workbench, evolve | none for Tier 1 | the authoring surface |
| **9** | Extraction including the mark sweep, round-trip validation, lineage operations | inference | bootstrapping and derivation |
| **10** | Integration surfaces: scenario, compile, scripting, API, both TUIs, multiperson, sidecar | — | parity |

Notes on the ordering:

- **Phase 1 before Phase 2, 3 and 5** is the central sequencing argument. Without
  the scorecard, every subsequent phase is evaluated by eye, and the original
  complaint — "results are uneven" — remains unfalsifiable.
- **Phase 3 is unusually valuable early.** The detail subsystem is nearly
  independent of the rest: given any image with a detectable face, it can place
  marks and jewelry deterministically. It ships value before casting exists, and
  it is the phase most likely to produce a visible "oh, that just works" moment.
- **Phases 0–3 are almost entirely deterministic**, which means they carry corpus
  entries and CI gates from the beginning and are testable on hosts without
  capable GPUs.
- **Phase 8 could move earlier** if authoring ergonomics matter more than
  correctness in early feedback; the headless core makes it loosely coupled, and
  the `place` widget only needs Phase 2's wireframe. The argument against is that
  the TUI's badges and variant strips are only honest once Phase 4 has measured
  them.

**5.0.0 cut line (adopted): P0–P8.** Extraction (P9) + full integration parity
(P10) land as 5.1.

---

## 26. Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Face-landmark ControlNets are unavailable or weak for most families | geometric control is Tier-A-only | survey early (§28.3); the swap bridge inherits geometry indirectly; depth and pose conditioning as fallbacks; detail compositing is unaffected |
| R2 | Calibration shows most scalars are weakly controllable | the schema over-promises | controllability grades are shipped as first-class output; the TUI and lint tell the truth; scalars still drive geometry and the scorecard even where prompt control is weak |
| R3 | Attribute entanglement survives region-grouped emission | colour bleed persists | detail routing already removes the worst offenders from the prompt; fall back to face-region regional prompting or a separate face-only inpaint pass (§28.1) |
| R4 | Casting cost makes iteration painful | the feature feels heavy | structural-hash caching, incremental surface and detail re-cast, few-step presets for preview, resumable casts, cost estimates up front |
| R5 | Scoring-model residency OOMs on typical hosts | scoring unusable when needed | resident worker, probe tiering, batched scoring, hard memory guards (§22); the cheapest probes cover details |
| R6 | Swap bridge artefacts at extreme pose or small face area | identity fails in exactly the interesting shots | region-escalation ladder (§14.1); documented limits; refinement pass |
| R7 | Lexicon vocabulary causes harm | serious | §7.4 and §23.3 constraints, mandatory review, recorded rationale |
| R8 | Schema or topology churn strands existing personas | user trust; every anchor invalidated | mandatory `schema:`, independently versioned topology frozen aggressively, forward migration, `lock.hjson`, staleness warnings |
| R9 | Three-name confusion (persona / People / personas) persists | maintenance drag | §4 resolution adopted before implementation begins |
| R10 | Multiperson identity contamination and detail misattribution | the most-wanted case fails, visibly | sequential per-figure refinement above N=2; compositor refuses on low assignment confidence (§14.2) |
| R11 | Composited marks read as stickers | the detail subsystem's core promise fails | maturity and relief modelling, light-direction estimation, harmonisation calibrated per family (§13.2); `local_anomaly` measures it, so failure is visible |
| R12 | Harmonisation erases the composite | marks silently vanish | strength is calibrated against `local_anomaly` presence, not chosen by feel; the pass is skippable |
| R13 | Jewelry asset library is a maintenance and licensing burden | scope creep | small, generic, original-or-public-domain, reviewed (§10.5); glasses excluded from compositing since prompting works |
| R14 | Hand jewelry is unreliable | user disappointment | graded `experimental`, culled-and-reported on detection failure, documented escalation path (§8.5) |
| R15 | Dentition rarely manifests, so it is authored and never seen | wasted effort, confusing | manifestation gate with informational lint; cast recommends a teeth-visible expression; TUI gates the section behind an explicit question |

---

## 27. Open questions

**27.1 — Entanglement mitigation.** Does region-headed grouping meaningfully
reduce colour bleed on CLIP encoders relative to flat comma-separated lists, and
does a separate face-only inpaint pass outperform both? An experiment, not an
opinion; it should run in Phase 1 once the scorecard can measure the answer.

**27.2 — Split-encoder emission on dual-encoder families.** Does emitting
structural attributes to one encoder and colouring to the other reduce
entanglement, or does it simply halve the effective conditioning?

**27.3 — Landmark conditioning availability.** Which families have usable
face-landmark or face-mesh ControlNets, at what resolution, under what license?
This survey gates the reach of Layer 2 and should be the first research task,
because a negative result reshapes the architecture toward the swap bridge as the
primary path.

**27.4 — How much does the TUI actually need a GPU?** If the wireframe is
sufficient for geometric attributes and detail placement, and cached variant
thumbnails cover colour and texture, Tier-2 preview may only be needed at section
boundaries and at cast time — which would make the entire composition TUI runnable
with no GPU at all.

**27.5 — Prompt-only ceiling on the strongest prompt-follower.** The
Gemma-2-conditioned family is the best prompt-follower in the stack. If it holds a
persona from a long structured paragraph alone, the architecture partially
inverts: that family becomes the canonical casting renderer, and the reference set
exists mainly to bridge *back* to the CLIP families.

**27.6 — Reference-set size and composition.** How many references, across how
many views, expressions and lighting conditions, before adapter-based identity
saturates? Determines the default cast size and the sheet composition — and, with
dentition in scope, how many teeth-visible views are needed.

**27.7 — Scorecard tolerances.** Authored per attribute, or derived from the
measured seed-variance during calibration? Derived is more principled and more
work; authored is a reasonable Phase 1 placeholder.

**27.8 — Does `figure` deserve v1 scope at all?** Given §11.7, an argument exists
for shipping face-only in v1. **Adopted: `figure` is IN v1 scope** — the schema
shape and the pose conditioning are cheap, body-sited details need the skeleton
anyway, and omitting them would force a breaking schema change later.

**27.9 — Composite versus inpaint as the default detail strategy.** Compositing is
deterministic, exact and free; inpainting integrates better but is stochastic and
imprecise in position. The proposal makes compositing the default with
harmonisation, but the crossover point — the detail size above which inpainting
wins — is unmeasured. Phase 3 should measure it and set the threshold per kind.

**27.10 — How aggressive should the mark sweep be during extraction?** A sensitive
sweep finds every pore and shadow; a conservative one misses real marks. The
proposal emits low-confidence findings commented-out, but the threshold and the
classification taxonomy both need tuning against real photographs.

**27.11 — Should details participate in casting scoring at all?** Since they are
composited onto every candidate identically, they add no discriminating signal
between candidates and merely cost scoring time. The counter-argument is that
harmonisation quality *does* vary with the underlying skin, so a candidate whose
cheek takes the mole convincingly is genuinely better.

---

## 28. Alternatives considered

**A1 — Prompt templates only.** A well-authored template library, no geometry, no
casting, no compositing, no scorecard. Cheap, and genuinely helps. Rejected as the
whole answer because it addresses only §2.1.3 and leaves underdetermination,
entanglement, and localized detail untouched. Retained as a *component*: the
emitters are exactly this, done systematically.

**A2 — Photo-first only.** Require a reference photograph for every persona and
skip the spec entirely. Much simpler, and the existing machinery already does it.
Rejected because it cannot author a person who does not exist, which is the stated
requirement.

**A3 — LLM as the resolver.** Hand the spec to a language model and let it write
the per-family prompts. Attractive and much less code. Rejected as the *default*
because it destroys determinism, byte-stability, corpus testing, and offline
operation — all established properties of the project. Retained as an opt-in
enhancement pass and as the prose-prefill path, both with deterministic fallbacks.

**A4 — Bake-only identity.** Skip casting and adapters; train a LoRA per persona
per base directly from a small synthetic set. Rejected as the primary path because
training cost and memory requirements put it out of reach on typical hosts and it
must be per-base. Retained as Tier C.

**A5 — Learned deformation basis.** Fit the landmark basis to a face dataset via
PCA. Higher fidelity. Rejected for v1 on licensing and reproducibility grounds
(§10.5); revisitable if a permissively-licensed basis is identified.

**A6 — Full 3D morphable head.** Strongest geometric control and free multi-view
synthesis. Rejected for v1 on licensing and scope. A procedural parametric mesh is
the future path (§30.3).

**A7 — Prompt-only marks with heavy negative prompting.** Describe every mark in
the prompt and use negatives to suppress unwanted ones. Rejected because it
addresses presence but not position, consumes the budget that structural
attributes need, and cannot deliver the cross-render stability that makes a mark
part of an identity rather than an accident.

**A8 — Inpaint every detail rather than compositing.** Higher integration quality,
no procedural overlay generators to write, no jewelry asset library to maintain.
Rejected as the default because inpainting cannot place a four-pixel feature at a
specified coordinate — the sampler decides — which is exactly the failure being
fixed. Retained as the strategy for large details and as the repair path, and
§27.9 asks where the crossover actually is.

**A9 — Store detail positions as pixel coordinates on a canonical face image.**
Simpler than anchors, and easy to author with a mouse. Rejected because it
desynchronises from the geometry on every structural edit, does not survive a
change of framing or pose, and cannot be projected onto a rendered face whose
proportions differ from the canonical one. The anchor model costs more code and is
correct.

---

## 29. Future work

**29.1 — Body identity.** A body-embedding equivalent of the face identity encoder
would upgrade `figure` from weak conditioning to a real anchor, and would make
full-body persona consistency achievable rather than approximate.

**29.2 — Learned inverse for extraction.** A small regressor from landmarks to
spec scalars would improve extraction accuracy over the calibration-curve
inversion, at the cost of a training dependency. The mark sweep would benefit even
more from a learned classifier.

**29.3 — Procedural 3D head.** A parametric mesh driven by the same spec scalars,
rendered to depth and normal maps, would give stronger structural control, free
arbitrary-view reference sheets, and — importantly for this revision — correct
occlusion and foreshortening for detail anchors at any pose. License-clean because
it is generated, not fitted.

**29.4 — Expression and pose as spec-native axes.** Currently presentation-only; a
spec-native expression system with its own deformation basis would let a persona
carry a characteristic smile as part of identity, and would give dentition a
proper home.

**29.5 — Learned overlay generators.** The procedural scar and birthmark
generators are hand-authored. A small generative model conditioned on the detail
record would produce more varied and convincing overlays, at the cost of
determinism — so it would be an opt-in tier above the procedural default.

**29.6 — Jewelry from 3D.** Compositing 2D assets limits jewelry to the
orientations shipped. A small set of procedurally-generated 3D jewelry primitives
rendered at the realised head pose would remove that limit and fix the
three-quarter-view earring problem.

**29.7 — Persona sharing.** A community format and index for exported bundles,
with schema, lexicon and topology versions as compatibility keys.

**29.8 — Temporal consistency.** A persona held stable across an animated
sequence, once a video path exists. Details are the easy half of this — the
compositor already runs per frame against realised landmarks.

**29.9 — Voice and description profile.** A persona is currently visual only; a
prose profile field usable by downstream text tooling is a natural, cheap
extension.

---

## 30. Summary of what must be true for this to work

1. The scorecard exists before the generation work, so "uneven" becomes a number —
   and it reports identity variance and localized-detail hit rate separately,
   because they fail for different reasons.
2. Scalars are calibrated per family, so `0.5` means something and `0.7` lands
   near `0.7`.
3. Unknown is distinct from average, and for collections, *asserted empty* is
   distinct from *unspecified*.
4. Structural, surface, detail and presentation classes are distinguished, so
   edits invalidate proportionally — and a mark edit costs nothing.
5. Every localized detail is anchored anatomically, never to a pixel, and resolves
   against the *realised* landmarks of each render.
6. Details are composited, not prompted, because text conditioning cannot place a
   four-pixel feature — and this is what frees the token budget for the attributes
   that determine recognisability.
7. The compositor and the detail probe agree: whatever is drawn is findable, and
   whatever is findable was drawn.
8. Manifestation is modelled explicitly, so dentition is neither prompted into
   closed mouths nor scored against invisible teeth.
9. The resolver is pure, so the deterministic half of the system — which after
   this revision includes the entire detail pipeline — is testable without weights.
10. The interview is data-driven from the lexicon, so six subsystems cannot drift.
11. The universal identity path requires no new model ports.
12. Controllability is measured and reported, so the tool never claims a slider
    works when it does not — and says plainly that the small composited things are
    the *most* reliable, not the least.
