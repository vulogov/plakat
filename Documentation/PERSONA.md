# `plakat persona` — controllable synthetic-person composition

`persona` composes a *specific, reusable* synthetic person from a small HJSON document (a
`PersonaSpec`) and renders that same person, recognisably, across scenes and model families. It is the
plakat 5.0 flagship (RFC [`RFC_PERSONA_1.md`](RFC_PERSONA_1.md)), and it is **fully additive** — no
existing command or output changes.

The idea: text prompts are a poor instrument for identity. "A woman with wide-set hazel eyes, a small
mole below the left eye, and a scar through one eyebrow" places the mole anywhere, drops it under CFG
pressure, and moves it between renders. `persona` fixes this by treating a person as **structured
data** that is resolved deterministically, conditioned geometrically, realised for small details by
compositing (not prompting), anchored to one identity, and **measured**.

## The layer model

| Layer | What | Command(s) | Weights? |
|---|---|---|---|
| 0 — spec + resolver | the HJSON schema → per-family prompt | `new` · `lint` · `show` · `interview` | no |
| 1 — compiler | salience-ranked, budget-solved emission | (inside `show`) | no |
| 2 — geometry | landmark deformation → conditioning maps | `geometry` | no |
| 2.5 — details | marks / jewelry / dentition, composited at anchors | `composite` | partial |
| 4 — scorecard | measure a render against the spec | `verify` | detect only |
| 5 — calibration | per-family priors + response curves + grades | `calibrate` | offline |
| 3 — identity | cast a reference set, render into scenes | `cast` · `render` · `bake` | yes |
| — edit/repair | class-aware diff + targeted in-place fix | `diff` · `repair` | partial |

The determinism contract (RFC §5.2): everything in layers 0–2 + detail compositing is a **pure,
byte-stable function** — testable in CI without weights or a GPU.

## Commands

```
plakat persona new <out> [--depth quick|standard|full] [--name --age]   scaffold a spec
plakat persona lint <spec>                                              validate (schema / range / contradiction)
plakat persona show <spec> --model <m>                                  the compiled prompt + salience + grades
plakat persona interview <out> [--depth] [--answers <f>] [--tui]        author via the Q/A interview (§17)
plakat persona geometry <spec> --out <dir> [--map …] [--calibrate <m>]  rasterise the conditioning maps
plakat persona composite <spec> --image <png> [--harmonise]             composite the persona's details onto a render
plakat persona verify <spec> --image <png> --model <m>                  the scorecard
plakat persona calibrate <fam> --bootstrap | --from <dir> --out <t>     build a family calibration table
plakat persona cast <spec> --model <m> [--count --keep-best]            render + score → a reference set
plakat persona render <persona-dir> --scene "…" [--with <dir>] [--tier] cast persona → into a scene
plakat persona bake <persona-dir> --base <m> --method ti|lora           train a per-base adapter (Tier C)
plakat persona diff <old> <new>                                         classify an edit (structural vs surface/detail)
plakat persona repair <spec> --image <png> --attr <path>                fix one attribute in place
```

A worked end-to-end demo of all of these lives in [`../corpus/PERSONA_CORPUS.md`](../corpus/PERSONA_CORPUS.md).

## Identity tiers (§11.4)

Identity is honest and tiered. `persona render --tier auto` (default) picks the best available:

- **Tier A** — an IP-Adapter-Plus-Face reference (where a family adapter exists: sd15/sd21/sdxl).
- **Tier B** — native render → **face swap** from the reference set → restore → detail composite.
  Requires nothing family-specific, so it is the **universal** path (every supported model).
- **Tier C** — a **baked** per-base TI/LoRA (`persona bake`); strongest for prompt-native use.

Detail compositing (marks, most jewelry) is **tier-independent** — the small distinguishing features
work identically on every family, because they never go through a sampler.

## Honest scope

- **Body identity is face-only** (§11.7). There is no body ArcFace, no body-reference adapter, no body
  swap. `figure` attributes are conditioned through the pose skeleton, silhouette and prompt — weak
  signals — and are graded accordingly. Body-*sited* details (a forearm tattoo) composite against the
  body skeleton and work as well as its landmark detection does (poorly for hands, §8.5).
- **Landmark topology is WFLW-98** (Phase-G decision, not the RFC's original 106-pt InsightFace) — a
  license-clean PIPNet-98 port. Named anchors are defined on it ([`PERSONA_ANCHORS.md`](PERSONA_ANCHORS.md)).
- **Calibration tables ship as provisional bootstraps.** The per-family response curves + priors are
  measured by an offline render sweep (`persona calibrate --from`); until then the grades are lexicon
  defaults. See [`PERSONA_CALIBRATION.md`](PERSONA_CALIBRATION.md).
- **Neutrality is binding** (§7.4, §23.3): skin tone is Fitzpatrick/CIELAB only, no ethnonyms, no
  valence, no default face. `apparent_age` is a plain grounding attribute — it sets the rendered age
  like any prompt term, with **no minimum enforced** (parity with the rest of `plakat`).

## Companion documents

- [`RFC_PERSONA_1.md`](RFC_PERSONA_1.md) — the full design.
- [`PERSONA_TUTORIAL.md`](PERSONA_TUTORIAL.md) — a hands-on walkthrough.
- [`PERSONA_LEXICON.md`](PERSONA_LEXICON.md) — the attribute vocabulary + edit classes.
- [`PERSONA_ANCHORS.md`](PERSONA_ANCHORS.md) — the WFLW-98 named-anchor vocabulary for details.
- [`PERSONA_DETAILS_HOWTO.md`](PERSONA_DETAILS_HOWTO.md) — authoring marks, jewelry, and dentition.
- [`PERSONA_CASTING.md`](PERSONA_CASTING.md) · [`PERSONA_CALIBRATION.md`](PERSONA_CALIBRATION.md) ·
  [`PERSONA_GATING.md`](PERSONA_GATING.md) — casting/render, calibration, and the gating research.
