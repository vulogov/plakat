# RFC QUALITY-4 — naturalize follow-ups (6.13.0)

**Status:** draft (6.13.0) · **Depth cycle** on [`RFC_QUALITY_3.md`](RFC_QUALITY_3.md). No new studio;
tightens the 6.12 de-slop tools against the concrete failures the owner found.

## Motivation

6.12 shipped `--repair` (face-protected) as a best-effort structural tool. Owner review on the kids
watercolor exposed two real regressions and one ergonomic gap:

1. **The background changed** — `--repair` protects *faces* but regenerates *everything else*, so the
   surrounding composition (street, buildings) shifts and an umbrella-like artifact appeared where only the
   figures should have been touched.
2. **Colours drifted** — same cause (whole-non-face regen) plus the grade.
3. **`--style` is manual** — you must name the medium or the re-paint drifts to photoreal.

## What ships

### P1 — figure-scoped repair (protect the background too)
Repair should touch **only the figures**, not the whole non-face canvas. Reuse the SCRFD faces already
detected: **project a body box** from each face (a running child ≈ 5–6 head-heights tall, ≈ 2.5 head-widths
wide), union the body boxes, then **subtract the face boxes** → the repair mask is *figure-bodies-minus-
faces*. Everything else — sky, buildings, cobbles — is **preserved pixel-for-pixel** (black in the mask).
Result: faces stay soft (already), background stays put (new), only the broken limbs/torsos get the gentle
in-style re-paint. Kills the "background changed / umbrella appeared / colours drifted" regression.
`--repair-scope figures` (default) vs `non-face` (the 6.12 behaviour) vs `full`.

### P2 — auto medium-detection
Drop the manual `--style` requirement: a **CLIP zero-shot** medium classifier (reuse the aesthetic
scorer's CLIP ViT-L) scores the image against a bank of medium prompts ("watercolor painting", "oil
painting", "ink drawing", "graphite pencil sketch", "gouache", "3d render", "photograph", …) and picks the
best. `--repair`/`--geometry` without `--style`/`--medium` then auto-anchor to the detected medium (printed,
overridable). Weight-free-ish (one CLIP forward).

### P3 — more content focuses
Extend the weight-free focus set beyond the current nine with the subjects that recur in AI slop:
**animal** (fur/feather over-smoothness), **food** (plastic sheen), **interior** (flat CGI light),
**textile/fabric**, **foliage-macro**. Each is a `Params` profile blended like the others; all combine.

### P4 — parity + docs + cut 6.13.0
Update the tutorial + QUALITY.md + doctor + README; corpus demo. Cut 6.13.0 (bump Cargo+lock, gate,
turbofish on new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits (unchanged, restated)
- Figure-scoped repair preserves the background and faces and makes a **bounded** attempt at the figures'
  anatomy — it still cannot reliably remove an extra limb (a diffusion model re-paints, it doesn't reason).
- Auto-medium is a best-effort classifier; `--style` still overrides.

## Sequencing
**P1** figure-scoped repair → **P2** auto-medium → **P3** more focuses → **P4** cut. Independent phases.
