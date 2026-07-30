# PERSONA-1 lexicon & edit classes

The lexicon (`assets/persona/lexicon.hjson`) is **data, not code**: it maps each prompt-bearing
attribute to its class, section, phrasing, and interview widget, so the resolver, the emitters, the
question graph, and the scorecard can never drift from the schema (RFC §7). It ships as a **skeleton**
(the highest-signal attributes) and grows one HJSON entry at a time — the `PersonaSpec` schema carries
many more fields (see `src/persona/spec.rs`) than the skeleton yet phrases.

## Scalars, enums, colours

- **Scalars** are `[0, 1]` where `0.5` is the family's own prior (its *measured* meaning, per
  [`PERSONA_CALIBRATION.md`](PERSONA_CALIBRATION.md)) — **not** a default face. Leaving a field out
  means *unknown*; an explicit `0.5` asserts "deliberately average" and is distinct (§6.4).
- **Enums** are neutral anatomical descriptors (`oval`, `aquiline`, `almond`, `monolid`).
- **Colours** are a lexicon name (`hazel`, `auburn`) or an exact CIELAB triple `{ lab: [L, a, b] }`.
  Skin tone is Fitzpatrick (`fitzpatrick-1..6`) or CIELAB only — no colour adjectives, ethnonyms, or
  geographic labels (§7.4).

## Skeleton attributes

| Path | Class | Kind | Poles / values |
|---|---|---|---|
| `face.shape` | surface | enum | oval · round · square · heart · … |
| `face.width` | structural | scalar | narrow ↔ wide |
| `face.jaw.width` | structural | scalar | narrow ↔ wide |
| `face.jaw.definition` | surface | enum | soft · defined · sharp |
| `face.chin.projection` | structural | scalar | receding ↔ prominent |
| `face.cheekbones.prominence` | structural | scalar | low ↔ high |
| `eyes.shape` | surface | enum | almond · round · monolid · hooded · … |
| `eyes.spacing` | structural | scalar | close-set ↔ wide-set |
| `eyes.canthal_tilt` | structural | scalar | downturned ↔ upturned |
| `eyes.color` | surface | colour | (named / lab) |
| `eyes.brow.thickness` | surface | scalar | thin ↔ thick |
| `eyes.brow.arch` | surface | enum | straight · soft · rounded · angled · high |
| `nose.profile` | surface | enum | straight · aquiline · button · … |
| `nose.length` | structural | scalar | short ↔ long |
| `mouth.width` | structural | scalar | narrow ↔ wide |
| `mouth.lower_lip` | surface | scalar | thin ↔ full |
| `mouth.cupids_bow` | surface | enum | flat · soft · defined · pronounced |
| `skin.tone` | surface | enum | fitzpatrick-1..6 → neutral tone words |
| `skin.undertone` | surface | enum | cool · neutral · warm |
| `skin.texture` | surface | scalar | smooth ↔ textured |
| `hair.color` · `hair.length` · `hair.texture` | surface | colour/enum | — |
| `facial_hair.style` | surface | enum | none · stubble · moustache · goatee · full-beard · … |
| `figure.build` | structural | enum | ectomorph · mesomorph · endomorph |

`plakat persona show <spec> --model <m>` prints the resolved attributes with their per-family
controllability grade badge (`[strong|moderate|weak|experimental]`, measured — §13.3).

## Edit classes (§6.5) — what a change invalidates

Every leaf has a class that determines the cost of changing it. `plakat persona diff <old> <new>`
reports the classes of an edit and whether a re-cast is required.

| Class | Examples | An edit invalidates | Repair |
|---|---|---|---|
| **Structural** | face/jaw/nose/eye geometry, teeth alignment, figure proportions | the reference set + every baked adapter | full re-cast (`persona cast`) |
| **Surface** | eye/hair colour, skin tone, teeth shade, piercing sites | nothing structural | targeted inpaint / recomposite (`persona repair`) |
| **Detail** | individual `marks`, `teeth.features`, `jewelry.items` entries | nothing | recomposite only — milliseconds (`persona composite`) |
| **Presentation** | expression, gaze, framing, `jewelry_worn` | nothing | per-render override |

This is why "same person, different hair colour" is cheap and a nose change is honestly expensive.

## Neutrality (binding, §7.4/§23)

No valence adjectives (`attractive`, `harsh`) anywhere in the schema or lexicon. Marks/scars/birthmarks
are described as visual morphology (`pigmented`, `vascular`, `raised`, `linear`), never as diagnoses.
Enum ordering is anatomical or alphabetical, never by prevalence. There is no fallback human — an
unset field stays unknown.
