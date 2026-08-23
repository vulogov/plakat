# RFC QUALITY-5 — naturalize follow-ups round 2 (6.14.0)

**Status:** SHIPPED (6.14.0) · **Depth cycle** on [`RFC_QUALITY_4.md`](RFC_QUALITY_4.md). No new studio;
finishes the ergonomic + coverage gaps in the 6.13 tools.

## What ships

### P1 — auto-paper for watercolor media
When the medium (named via `--medium`/`--style`, or CLIP auto-detected) is a **wet-media** family
(watercolor / gouache / ink-wash) and `--paper` was not set explicitly, auto-apply `--paper` at the
recommended **0.6** so watercolor art gets pigment/paper authenticity by default. No surprise model load:
auto-paper fires from an explicit `--medium` (free), or from the auto-detect that a model correction
already ran; a pure weight-free run opts in with `--auto-medium`.

### P2 — person-detection for repair (catch faceless figures)
Figure-scoped repair (6.13) projects body boxes from **faces** — so a figure whose face isn't detected
(back turned, distant, occluded) is missed. Add an OWL-ViT **"person"** detection pass that unions with the
face-projected boxes, so *all* figures are covered; faces still subtracted (protected). Falls back to the
face-only path when the detector is unavailable.

### P3 — `--paper` spec/api parity + batch de-slop
- **Spec parity:** make `paper=N` reachable from the naturalize **spec** string, so `generate --naturalize
  "photo paper=0.6"`, a scenario `naturalize:` field, and `api::Naturalize` all support it (today `--paper`
  is CLI-only).
- **Batch:** `naturalize` accepts a **directory / multiple inputs** (like `rank`) and writes each result to
  an `--out` directory, so a folder of images can be de-slopped in one call.

### P4 — parity + docs + cut 6.14.0
Tutorial + QUALITY + doctor + README; corpus. Cut 6.14.0 (bump Cargo+lock, gate, turbofish on new
`.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits (unchanged)
Person-detection widens repair *coverage*; it does not change the honest ceiling — figure repair still makes
a bounded, in-style attempt at anatomy and cannot guarantee an extra limb is removed.

## Sequencing
**P1** auto-paper → **P2** person-detection → **P3** spec parity + batch → **P4** cut. Independent phases.
