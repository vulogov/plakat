# RFC QUALITY-7 — naturalize round 4: folder scorecard · LUT export · preset library (6.16.0)

**Status:** draft (6.16.0) · **Depth cycle** on [`RFC_QUALITY_6.md`](RFC_QUALITY_6.md). Reporting at scale,
interop, and reusable recipes.

## What ships

### P1 — folder scorecard
`naturalize <dir> --report` scans a whole folder and prints a **ranked scorecard table** (worst-AI first)
with each image's AI-tell + its drivers, plus an **aggregate summary** (mean AI-tell, count over a
threshold, the folder's dominant tell). `--json` emits an array. Turns "which of these 200 renders are the
sloppiest, and what do they need?" into one command — the batch companion to the 6.15 single-image report.

### P2 — LUT export (`.cube`)
`naturalize --export-lut grade.cube [grade flags]` bakes the fixed **film grade** (desaturate-toward-luma +
warm lift + vibrance) into a standard **`.cube` 3-D LUT** (default 33³) so plakat's colour grade can be
applied in **DaVinci Resolve / Premiere / OBS / Lightroom**. Honest scope: only the *fixed* colour transform
is LUT-able — the polish **white-balance and auto-levels are per-image adaptive** and are documented as *not*
captured by a static LUT.

### P3 — preset library
Beyond `subtle`/`photo`/`painting`, add a curated set of named **spec presets** for common cases
(`portrait`, `landscape`, `product`, `anime`, `film`, `restore`) that expand to a full naturalize spec, plus
`naturalize --list-presets` to print them. A named preset is just a saved spec string, so `--preset
portrait` == the recipe the scorecard would recommend for a portrait.

### P4 — parity + docs + cut 6.16.0
Tutorial + QUALITY + doctor + README; corpus. Cut 6.16.0 (bump Cargo+lock, gate, turbofish on new
`.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
The folder scorecard is a coarse **ranking** aid, not a verdict. The LUT captures only the fixed grade (not
the adaptive white-balance / auto-levels). Presets are convenience recipes, not magic.

## Sequencing
**P1** folder scorecard → **P2** LUT export → **P3** preset library → **P4** cut. Independent phases.
