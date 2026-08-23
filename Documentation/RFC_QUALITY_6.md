# RFC QUALITY-6 — naturalize round 3: scorecard · video · region masks (6.15.0)

**Status:** draft (6.15.0) · **Depth cycle** on [`RFC_QUALITY_5.md`](RFC_QUALITY_5.md). Extends the arc to
diagnosis, motion, and spatial control.

## What ships

### P1 — naturalize scorecard / report
`plakat naturalize <img> --report` **analyzes** an image and prints a de-slop scorecard instead of
processing it — plakat's own version of the "AI-detection verdict": the **AI-tell score** decomposed into
its drivers (oversaturation, texture over-smoothness), the **CLIP-detected medium**, and a **recommended
recipe** (the flags to run). Weight-free except the optional CLIP medium probe. `--json` for a structured
report. Turns "is this sloppy, and what do I run?" into one command.

### P2 — video / animation de-slop
`naturalize in.mp4|gif --out out.mp4` de-slops **every frame** and re-encodes (reuse the ffmpeg plumbing
`animate`/fractals use). The weight-free pass must be **temporally stable**: grain / paper / micro noise is
today seeded per-pixel — fine for a still, but it would **flicker** frame-to-frame. Add a frame-invariant
noise mode (seed by pixel coords only, not frame index) so the texture sits still while the image moves.
Model passes are per-still only (documented; video de-slop is the weight-free surface pass).

### P3 — per-region focus masks
A single frame often has several subjects (sky + people + foliage). Today one focus profile blends over the
whole image. Add **auto-region** focuses: detect faces (SCRFD) → apply the `people`/`micro` profile in those
regions; a sky band (top, smooth, blue) → `sky`; the rest → the base. Compose per-region so each subject
gets its own de-slop, then feather the seams. Manual `--region "x0,y0,x1,y1:people=1"` override.

### P4 — parity + docs + cut 6.15.0
Tutorial + QUALITY + doctor + README; corpus. Cut 6.15.0 (bump Cargo+lock, gate, turbofish on new
`.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
The scorecard is a **coarse heuristic** report, not a forensic verdict. Video de-slop is the **weight-free**
surface pass only (no per-frame regeneration — that would be neither temporally stable nor affordable).
Region masks improve *targeting*; they don't change what each focus can do.

## Sequencing
**P1** scorecard → **P2** video → **P3** region masks → **P4** cut. Independent phases.
